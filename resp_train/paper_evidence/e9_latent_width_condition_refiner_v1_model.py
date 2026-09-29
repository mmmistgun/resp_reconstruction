"""E9 全局潜在宽度 × 条件末端通道重组的六个原生模型。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.blocks import (
    BidirectionalMamba2Block,
    DecoderResidual,
    ResidualDWBlock,
)
from resp_train.crd.frontends import LegacyPatchTokenEncoder
from resp_train.crd.initialization import module_seed
from resp_train.crd.model import _validate_input
from resp_train.crd.spectral_ops import fourier_interpolate
from resp_train.crd.tf_v1_model import (
    CRDTfV1Model,
    _checkpointed_mapping_branch,
    _require_feature,
    trainable_parameter_count,
)


PROTOCOL = "e9-latent-width-condition-refiner-v1"
LATENT_LENGTH = 1_800
WAVEFORM_LENGTH = 18_000
SCALE_COUNT = 97
POINTWISE_DECODER_PARAMETERS = 1_057


def _kaiming(module: nn.Conv1d | nn.Conv2d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _initialize_norm(module: nn.GroupNorm) -> None:
    nn.init.ones_(module.weight)
    nn.init.zeros_(module.bias)


class ChannelResidualRefiner(nn.Module):
    """保持时间索引不变的 D→H→D 逐点残差重组。"""

    def __init__(self, latent_channels: int, hidden_channels: int) -> None:
        super().__init__()
        latent_channels = int(latent_channels)
        hidden_channels = int(hidden_channels)
        if latent_channels not in {64, 96} or hidden_channels not in {48, 64, 65}:
            raise ValueError("E9 condition refiner 只允许 D∈{64,96}, H∈{48,64,65}")
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.expand = nn.Conv1d(latent_channels, hidden_channels, kernel_size=1, bias=False)
        self.project = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1, bias=True)
        _kaiming(self.expand)
        _kaiming(self.project)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        expected = (self.latent_channels, LATENT_LENGTH)
        if context.ndim != 3 or tuple(context.shape[1:]) != expected:
            raise ValueError(
                f"E9 condition refiner 期望 (B,{expected[0]},{expected[1]})，"
                f"实际 {tuple(context.shape)}"
            )
        return context + self.project(F.silu(self.expand(context)))


class PatchTokenFrontend64(nn.Module):
    """冻结 Patch encoder 后接 16→64、GN(8,64)、SiLU。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = LegacyPatchTokenEncoder()
        self.adapter = nn.Conv1d(16, 64, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, 64, eps=1e-5, affine=True)
        self.activation = nn.SiLU()
        _kaiming(self.adapter)
        _initialize_norm(self.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)
        if x.shape[-1] == WAVEFORM_LENGTH and tokens.shape[-1] != 140:
            raise RuntimeError(f"E9 D64 patch bridge 期望 140 tokens，实际 {tokens.shape[-1]}")
        tokens = F.interpolate(tokens, size=LATENT_LENGTH, mode="linear", align_corners=False)
        return self.activation(self.norm(self.adapter(tokens)))


class CoarseWaveformHead64(nn.Module):
    """D64 输入、既有 64→64(DW)→32→1 的 10-Hz waveform head。"""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, 64, eps=1e-5, affine=True)
        self.conv = nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False)
        self.depthwise = nn.Conv1d(64, 64, kernel_size=5, padding=2, groups=64, bias=False)
        self.reduce = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.output = nn.Conv1d(32, 1, kernel_size=1, bias=True)
        self.activation = nn.SiLU()
        _initialize_norm(self.norm)
        for layer in (self.conv, self.depthwise, self.reduce):
            _kaiming(layer)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (64, LATENT_LENGTH):
            raise ValueError(f"E9 D64 head 期望 (B,64,1800)，实际 {tuple(x.shape)}")
        x = self.activation(self.conv(self.norm(x)))
        x = self.activation(self.depthwise(x))
        return self.activation(self.reduce(x))


class D64CoarseModel(nn.Module):
    """E9-B 专属 D=64 Patch+六层 BiMamba2+refinement+pointwise decoder。"""

    def __init__(self, initialization_seed: int) -> None:
        super().__init__()
        seed = int(initialization_seed)
        with module_seed(seed, "patch_frontend"):
            self.frontend = PatchTokenFrontend64()
        with module_seed(seed, "local_trunk"):
            self.local_blocks = nn.ModuleList(
                [BidirectionalMamba2Block(64) for _ in range(6)]
            )
        with module_seed(seed, "refinement"):
            self.refinement = nn.Sequential(
                ResidualDWBlock(64, dilation=1),
                ResidualDWBlock(64, dilation=2),
            )
        with module_seed(seed, "coarse_head"):
            self.head = CoarseWaveformHead64()
        with module_seed(seed, "decoder_residual"):
            self.decoder_residual = DecoderResidual()

    def encode_local(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x)
        latent = self.frontend(x)
        if tuple(latent.shape[1:]) != (64, LATENT_LENGTH):
            raise RuntimeError(f"E9 D64 frontend 输出合同错误: {tuple(latent.shape)}")
        tokens = latent.transpose(1, 2)
        for block in self.local_blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2)

    def decode_local(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (64, LATENT_LENGTH):
            raise ValueError(f"E9 D64 decode 期望 (B,64,1800)，实际 {tuple(latent.shape)}")
        if not bool(torch.isfinite(latent).all()):
            raise FloatingPointError("E9 D64 decode latent 包含 NaN/Inf")
        refined = self.refinement(latent)
        features = self.head.features(refined)
        waveform_10hz = self.head.output(features) + self.decoder_residual(features)
        waveform = fourier_interpolate(waveform_10hz, target_length=WAVEFORM_LENGTH)
        return {"waveform": waveform, "waveform_10hz": waveform_10hz}


class D64CwtConditionBranch(nn.Module):
    """原 97-scale Morlet CWT 编码合同在 D=64 下的逐模块实现。"""

    def __init__(self, refiner: str, initialization_seed: int) -> None:
        super().__init__()
        if refiner not in {"direct", "h64", "h48"}:
            raise ValueError(f"未知 E9-B condition refiner={refiner!r}")
        # 保持 W0 branch 的构造顺序：零初始化 FiLM projection 先注册。
        self.final_projection = nn.Conv1d(64, 128, kernel_size=1, bias=True)
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
        self.conv_in = nn.Conv2d(1, 48, kernel_size=(5, 3), padding=(2, 1), bias=False)
        self.norm = nn.GroupNorm(8, 48)
        self.depthwise = nn.Conv2d(
            48, 48, kernel_size=(3, 3), padding=1, groups=48, bias=False
        )
        self.conv_out = nn.Conv2d(48, 64, kernel_size=1, bias=True)
        for module in (self.conv_in, self.depthwise, self.conv_out):
            _kaiming(module)
        _initialize_norm(self.norm)
        blocks = [ResidualDWBlock(64, dilation=2**index) for index in range(3)]
        for block in blocks:
            block.dropout = nn.Dropout(0.0)
        self.temporal = nn.Sequential(*blocks)
        if refiner == "direct":
            self.condition_refiner: nn.Module = nn.Identity()
        else:
            hidden = 64 if refiner == "h64" else 48
            with module_seed(initialization_seed, f"e9_condition_refiner_d64_{refiner}"):
                self.condition_refiner = ChannelResidualRefiner(64, hidden)
        self.refiner = refiner

    def forward(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        value = _require_feature(tf, "w", (SCALE_COUNT, 360)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        value = F.silu(self.conv_out(self.depthwise(value))).mean(dim=2)
        value = F.interpolate(value, size=LATENT_LENGTH, mode="linear", align_corners=False)
        context = self.condition_refiner(self.temporal(value))
        gamma, beta = self.final_projection(context).chunk(2, dim=1)
        return gamma, beta


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    family: str
    latent_channels: int
    condition_refiner: str
    hidden_channels: int | None
    trainable_parameters: int
    condition_refiner_macs: int
    declared_covered_macs: int


def _condition_macs(latent_channels: int, hidden_channels: int | None) -> int:
    if hidden_channels is None:
        return 0
    return 2 * int(latent_channels) * int(hidden_channels) * LATENT_LENGTH


def _declared_covered_macs(
    latent_channels: int,
    hidden_channels: int | None,
) -> int:
    """统计 E9 明确受管的 Conv/Linear；排除 Patch encoder、Mamba 内核与 Fourier。"""

    d = int(latent_channels)
    t = LATENT_LENGTH
    cwt_spatial = SCALE_COUNT * 360
    patch_adapter = 16 * d * t
    mamba_merges = 6 * (2 * d) * d * t
    cwt = (
        48 * cwt_spatial * 15
        + 48 * cwt_spatial * 9
        + d * cwt_spatial * 48
    )
    residual_block = d * t * 5 + d * (2 * d) * t + (2 * d) * d * t
    condition_temporal = 3 * residual_block
    film_projection = d * (2 * d) * t
    post_film_refinement = 2 * residual_block
    waveform_head = (
        64 * t * d * 5
        + 64 * t * 5
        + 32 * t * 64
        + t * 32
        + 32 * 32 * t
        + t * 32
    )
    return (
        patch_adapter
        + mamba_merges
        + cwt
        + condition_temporal
        + _condition_macs(d, hidden_channels)
        + film_projection
        + post_film_refinement
        + waveform_head
    )


# 参数数由原生模型 state 验证；禁止按参数预算填充 D64。
_RAW_SPECS = (
    ("e9a_d96_h65", "e9a", 96, "h65", 65, 1_219_850),
    ("e9a_d96_h64", "e9a", 96, "h64", 64, 1_219_658),
    ("e9a_d96_h48", "e9a", 96, "h48", 48, 1_216_586),
    ("e9b_d64_direct", "e9b", 64, "direct", None, 592_706),
    ("e9b_d64_h64", "e9b", 64, "h64", 64, 600_962),
    ("e9b_d64_h48", "e9b", 64, "h48", 48, 598_914),
)


ARM_SPECS: dict[str, ArmSpec] = {
    arm: ArmSpec(
        arm=arm,
        family=family,
        latent_channels=latent,
        condition_refiner=refiner,
        hidden_channels=hidden,
        trainable_parameters=parameters,
        condition_refiner_macs=_condition_macs(latent, hidden),
        declared_covered_macs=_declared_covered_macs(latent, hidden),
    )
    for arm, family, latent, refiner, hidden, parameters in _RAW_SPECS
}
ARMS = tuple(ARM_SPECS)
E9A_ARMS = tuple(arm for arm in ARMS if ARM_SPECS[arm].family == "e9a")
E9B_ARMS = tuple(arm for arm in ARMS if ARM_SPECS[arm].family == "e9b")


class E9LatentWidthConditionRefinerModel(nn.Module):
    """E9 六臂统一接口；D96 复用冻结 W0，D64 使用专属逐模块实现。"""

    def __init__(self, arm: str, initialization_seed: int) -> None:
        super().__init__()
        if arm not in ARM_SPECS:
            raise ValueError(f"未知 E9 arm={arm!r}")
        self.arm_spec = ARM_SPECS[arm]
        self.experiment_arm = arm
        self.initialization_seed = int(initialization_seed)
        spec = self.arm_spec
        if spec.family == "e9a":
            self._build_d96(spec)
        else:
            self._build_d64(spec)
        self._validate_contract()

    def _build_d96(self, spec: ArmSpec) -> None:
        self.base_model = CRDTfV1Model("crd_tf102_w", self.initialization_seed)
        branch = self.base_model.branches["w"]
        if spec.condition_refiner != "h65":
            assert spec.hidden_channels is not None
            with module_seed(
                self.initialization_seed,
                f"e9_condition_refiner_d96_{spec.condition_refiner}",
            ):
                branch.parameter_fill = ChannelResidualRefiner(
                    96, spec.hidden_channels
                )

    def _build_d64(self, spec: ArmSpec) -> None:
        self.base_model = None
        self.base64 = D64CoarseModel(self.initialization_seed)
        with module_seed(self.initialization_seed, "tf_branch_w"):
            branch = D64CwtConditionBranch(
                spec.condition_refiner,
                self.initialization_seed,
            )
        self.branches64 = nn.ModuleDict({"w": branch})

    @property
    def base(self) -> nn.Module:
        return self.base_model.base if self.base_model is not None else self.base64

    @property
    def branches(self) -> nn.ModuleDict:
        return self.base_model.branches if self.base_model is not None else self.branches64

    def _validate_contract(self) -> None:
        spec = self.arm_spec
        branch = self.branches["w"]
        projection = branch.final_projection
        if (
            projection.in_channels != spec.latent_channels
            or projection.out_channels != 2 * spec.latent_channels
            or not bool(projection.weight.detach().eq(0).all())
            or not bool(projection.bias.detach().eq(0).all())
            or not isinstance(self.base.decoder_residual, DecoderResidual)
            or not bool(self.base.decoder_residual.output.weight.detach().eq(0).all())
            or not bool(self.base.decoder_residual.output.bias.detach().eq(0).all())
        ):
            raise RuntimeError(f"E9 模型合同错误: {spec.arm}")
        if spec.family == "e9a":
            refiner = branch.parameter_fill
            if spec.condition_refiner == "h65":
                valid_refiner = refiner.__class__.__name__ == "_ActiveParameterFill"
            else:
                valid_refiner = (
                    isinstance(refiner, ChannelResidualRefiner)
                    and refiner.latent_channels == 96
                    and refiner.hidden_channels == spec.hidden_channels
                )
        else:
            refiner = branch.condition_refiner
            valid_refiner = (
                isinstance(refiner, nn.Identity)
                if spec.condition_refiner == "direct"
                else isinstance(refiner, ChannelResidualRefiner)
                and refiner.latent_channels == 64
                and refiner.hidden_channels == spec.hidden_channels
            )
        if not valid_refiner:
            raise RuntimeError(f"E9 condition refiner 合同错误: {spec.arm}")
        expected = spec.trainable_parameters
        if expected and trainable_parameter_count(self) != expected:
            raise RuntimeError(
                f"E9 参数合同错误: {spec.arm}: "
                f"{trainable_parameter_count(self)} != {expected}"
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        _validate_input(x)
        if tf is None or set(tf) != {"w"}:
            raise ValueError(f"E9 要求 TF keys=['w']，实际={sorted(tf or {})}")
        if self.arm_spec.family == "e9a":
            return self.base_model(x, tf=tf)
        latent = self.base.encode_local(x)
        gamma, beta = _checkpointed_mapping_branch(self.branches["w"], tf)
        conditioned = latent * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)
        return self.base.decode_local(conditioned)

    @property
    def shared_state_prefixes(self) -> tuple[str, ...]:
        if self.arm_spec.family == "e9a":
            return (
                "base_model.base.frontend.",
                "base_model.base.local_blocks.",
                "base_model.base.refinement.",
                "base_model.base.head.",
                "base_model.base.decoder_residual.",
                "base_model.branches.w.conv_in.",
                "base_model.branches.w.norm.",
                "base_model.branches.w.depthwise.",
                "base_model.branches.w.conv_out.",
                "base_model.branches.w.temporal.",
                "base_model.branches.w.final_projection.",
            )
        return (
            "base64.frontend.",
            "base64.local_blocks.",
            "base64.refinement.",
            "base64.head.",
            "base64.decoder_residual.",
            "branches64.w.conv_in.",
            "branches64.w.norm.",
            "branches64.w.depthwise.",
            "branches64.w.conv_out.",
            "branches64.w.temporal.",
            "branches64.w.final_projection.",
        )


def model_contract(arm: str) -> dict[str, Any]:
    if arm not in ARM_SPECS:
        raise ValueError(f"未知 E9 arm={arm!r}")
    spec = ARM_SPECS[arm]
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "family": spec.family,
        "latent_channels": spec.latent_channels,
        "condition_refiner": spec.condition_refiner,
        "condition_hidden_channels": spec.hidden_channels,
        "condition_formula": "c + W2(SiLU(W1(c)))" if spec.hidden_channels else "identity",
        "cwt": {"wavelet": "morlet", "scale_count": 97, "aggregation": "mean"},
        "film_projection": (
            f"conv1d_{spec.latent_channels}_to_{2 * spec.latent_channels}_zero_init"
        ),
        "film_bounds": {"gamma": 0.5, "beta": 0.5, "activation": "tanh"},
        "mamba": {
            "layers": 6,
            "bidirectional": True,
            "d_model": spec.latent_channels,
            "d_state": 64,
            "d_conv": 4,
            "expand": 2,
            "headdim": 32,
            "ngroups": 1,
            "chunk_size": 256,
        },
        "decoder": "pointwise_waveform_residual",
        "waveform_generation_hz": 10,
        "upsampling": "fourier_10_to_100_hz",
        "trainable_parameters": spec.trainable_parameters,
        "condition_refiner_macs": spec.condition_refiner_macs,
        "declared_covered_macs": spec.declared_covered_macs,
        "declared_covered_macs_scope": (
            "E9-managed Conv/Linear projections; excludes Patch encoder, "
            "Mamba internal kernels, normalization/activation, and Fourier interpolation"
        ),
    }


def build_e9_latent_width_condition_refiner_model(
    cfg: Any,
) -> E9LatentWidthConditionRefinerModel:
    from omegaconf import OmegaConf

    contract = cfg.model.get("e9_latent_width_condition_refiner_v1")
    if not OmegaConf.is_config(contract):
        raise ValueError("缺少 E9 模型合同")
    arm = str(contract.arm)
    if (
        str(cfg.protocol.name) != PROTOCOL
        or str(cfg.model.name) != "crd_v1"
        or str(cfg.model.variant) != "crd_tf102_w"
        or OmegaConf.to_container(contract, resolve=True) != model_contract(arm)
    ):
        raise ValueError("E9 模型只允许独立科学配置入口")
    return E9LatentWidthConditionRefinerModel(
        arm,
        int(cfg.model.initialization_seed),
    )
