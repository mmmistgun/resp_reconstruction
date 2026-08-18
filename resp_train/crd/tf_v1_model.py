from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from resp_train.crd.blocks import ResidualDWBlock
from resp_train.crd.initialization import module_seed
from resp_train.crd.model import CRDCoarseModel, _validate_input
from resp_train.crd.tf_v1_features import LearnableCarrierModulation
from resp_train.crd.tf_w_v2 import (
    P1_VARIANT_REPRESENTATIONS,
    P1_VARIANTS,
    P3_VARIANT_REPRESENTATIONS,
    P3_VARIANTS,
    apply_p1_w_view,
    p1_w_scale_count,
)


P_BRANCH_TARGET = 150_000
P_BRANCH_TOLERANCE = 0.02
TF_BRANCH_CHECKPOINT_BATCH_CHUNK = 8

TF_VARIANT_REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    "crd_tf101_m": ("m",),
    "crd_tf102_w": ("w",),
    "crd_tf103_l": ("l",),
    "crd_tf104_s": ("s",),
    "crd_tf201_mw": ("m", "w"),
    "crd_tf202_ml": ("m", "l"),
    "crd_tf203_ms": ("m", "s"),
    "crd_tf204_wl": ("w", "l"),
    "crd_tf205_ws": ("w", "s"),
    "crd_tf206_ls": ("l", "s"),
    "crd_tf301_mls": ("m", "l", "s"),
    "crd_tf302_wls": ("w", "l", "s"),
    "crd_tf_ctrl1": (),
    "crd_tf_ctrl2": (),
    "crd_tf_ctrl3": (),
}
TF_VARIANTS = tuple(TF_VARIANT_REPRESENTATIONS)
TF_CONTROL_COUNT = {
    "crd_tf_ctrl1": 1,
    "crd_tf_ctrl2": 2,
    "crd_tf_ctrl3": 3,
}

# P6a 独立于已冻结的 15-arm P4/P5 集合，避免改变历史矩阵完整性语义。
TF_P6_VARIANT_REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    "crd_tf401_mws_add": ("m", "w", "s"),
    "crd_tf402_mws_gate": ("m", "w", "s"),
    "crd_tf403_ctrl_gate": (),
}
TF_P6_VARIANTS = tuple(TF_P6_VARIANT_REPRESENTATIONS)
TF_ALL_VARIANT_REPRESENTATIONS = {
    **TF_VARIANT_REPRESENTATIONS,
    **TF_P6_VARIANT_REPRESENTATIONS,
    **P1_VARIANT_REPRESENTATIONS,
    **P3_VARIANT_REPRESENTATIONS,
}
TF_P6_CONTROL_COUNT = {"crd_tf403_ctrl_gate": 3}
TF_ALL_CONTROL_COUNT = {**TF_CONTROL_COUNT, **TF_P6_CONTROL_COUNT}
TF_GATED_VARIANTS = {"crd_tf402_mws_gate", "crd_tf403_ctrl_gate"}
P6_GATE_PARAMETER_COUNT = 13_347


def _initialize_conv(module: nn.Conv1d | nn.Conv2d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _initialize_norm(module: nn.GroupNorm) -> None:
    nn.init.ones_(module.weight)
    nn.init.zeros_(module.bias)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class _ActiveParameterFill(nn.Module):
    """参与 forward 的 pointwise residual，用于把各 branch 匹配到共同参数预算。"""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        if int(hidden_channels) <= 0:
            raise ValueError("parameter-fill hidden_channels 必须为正")
        self.expand = nn.Conv1d(96, int(hidden_channels), kernel_size=1, bias=False)
        self.project = nn.Conv1d(int(hidden_channels), 96, kernel_size=1, bias=True)
        _initialize_conv(self.expand)
        _initialize_conv(self.project)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.project(F.silu(self.expand(value)))


class _ConditionBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.final_projection = nn.Conv1d(96, 192, kernel_size=1, bias=True)
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
        self.parameter_fill: _ActiveParameterFill | None = None

    def finalize_parameter_match(self) -> None:
        if self.parameter_fill is not None:
            raise RuntimeError("parameter match 不得重复初始化")
        current = trainable_parameter_count(self)
        remaining = P_BRANCH_TARGET - current
        # fill 参数数为 192*h + 96；取最近整数后误差最多 96 params。
        hidden = int(round((remaining - 96) / 192))
        if hidden <= 0:
            raise RuntimeError(f"branch 基础参数 {current} 已超过预算 {P_BRANCH_TARGET}")
        self.parameter_fill = _ActiveParameterFill(hidden)
        final = trainable_parameter_count(self)
        if abs(final - P_BRANCH_TARGET) / P_BRANCH_TARGET > P_BRANCH_TOLERANCE:
            raise RuntimeError(f"branch 参数匹配失败: {final} vs {P_BRANCH_TARGET}")

    def project_condition(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or context.shape[1:] != (96, 1800):
            raise RuntimeError(f"TF context 期望 (B,96,1800)，实际 {tuple(context.shape)}")
        if self.parameter_fill is None:
            raise RuntimeError("TF branch 尚未 finalize parameter match")
        context = self.parameter_fill(context)
        gamma, beta = self.final_projection(context).chunk(2, dim=1)
        return gamma, beta


class _TemporalMixer(nn.Module):
    def __init__(self, block_count: int = 3) -> None:
        super().__init__()
        blocks = [ResidualDWBlock(96, dilation=2**index) for index in range(block_count)]
        # CRD 主干保留既有 dropout；CRD-TF 协议仅冻结所有新增 encoder 的 dropout=0。
        for block in blocks:
            block.dropout = nn.Dropout(0.0)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks(value)


class MultiResolutionStftBranch(_ConditionBranch):
    def __init__(self) -> None:
        super().__init__()
        self.slow = self._encoder()
        self.fast = self._encoder()
        self.merge = nn.Conv1d(96, 96, kernel_size=1, bias=True)
        _initialize_conv(self.merge)
        self.temporal = _TemporalMixer()
        self.finalize_parameter_match()

    @staticmethod
    def _encoder() -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, groups=32, bias=False),
            nn.Conv2d(32, 48, kernel_size=1, bias=True),
            nn.SiLU(),
        ]
        for module in layers:
            if isinstance(module, nn.Conv2d):
                _initialize_conv(module)
            elif isinstance(module, nn.GroupNorm):
                _initialize_norm(module)
        return nn.Sequential(*layers)

    def forward(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        slow = _require_feature(tf, "m_slow", (36, 101)).float()
        fast = _require_feature(tf, "m_fast", (44, 349)).float()
        slow_context = self.slow(slow[:, None]).mean(dim=2)
        fast_context = self.fast(fast[:, None]).mean(dim=2)
        slow_context = F.interpolate(slow_context, size=1800, mode="linear", align_corners=False)
        fast_context = F.interpolate(fast_context, size=1800, mode="linear", align_corners=False)
        context = self.temporal(F.silu(self.merge(torch.cat((slow_context, fast_context), dim=1))))
        return self.project_condition(context)


class CwtBranch(_ConditionBranch):
    def __init__(self, scale_count: int = 97) -> None:
        super().__init__()
        self.scale_count = int(scale_count)
        if self.scale_count not in {49, 97}:
            raise ValueError("CWT branch scale_count 只允许 49 或 97")
        self.conv_in = nn.Conv2d(1, 48, kernel_size=(5, 3), padding=(2, 1), bias=False)
        self.norm = nn.GroupNorm(8, 48)
        self.depthwise = nn.Conv2d(48, 48, kernel_size=(3, 3), padding=1, groups=48, bias=False)
        self.conv_out = nn.Conv2d(48, 96, kernel_size=1, bias=True)
        for module in (self.conv_in, self.depthwise, self.conv_out):
            _initialize_conv(module)
        _initialize_norm(self.norm)
        self.temporal = _TemporalMixer()
        self.finalize_parameter_match()

    def forward(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        value = _require_feature(tf, "w", (self.scale_count, 360)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        value = F.silu(self.conv_out(self.depthwise(value))).mean(dim=2)
        value = F.interpolate(value, size=1800, mode="linear", align_corners=False)
        return self.project_condition(self.temporal(value))


class LearnableModulationBranch(_ConditionBranch):
    def __init__(self) -> None:
        super().__init__()
        self.frontend = LearnableCarrierModulation(chunk_filters=6)
        self.conv_in = nn.Conv1d(24, 96, kernel_size=5, padding=2, bias=False)
        self.norm = nn.GroupNorm(12, 96)
        _initialize_conv(self.conv_in)
        _initialize_norm(self.norm)
        self.temporal = _TemporalMixer()
        self.finalize_parameter_match()

    def forward(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = _require_feature(tf, "l_spectrum", (9001,), complex_required=True)
        value = self.frontend(one_sided_spectrum=spectrum)
        value = F.interpolate(F.silu(self.norm(self.conv_in(value))), size=1800, mode="linear", align_corners=False)
        return self.project_condition(self.temporal(value))


class SstRidgeBranch(_ConditionBranch):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(12, 96, kernel_size=5, padding=2, bias=False)
        self.norm = nn.GroupNorm(12, 96)
        _initialize_conv(self.conv_in)
        _initialize_norm(self.norm)
        self.temporal = _TemporalMixer()
        self.finalize_parameter_match()

    def forward(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        value = _require_feature(tf, "s", (12, 360)).float()
        value = F.interpolate(F.silu(self.norm(self.conv_in(value))), size=1800, mode="linear", align_corners=False)
        return self.project_condition(self.temporal(value))


class TemporalCapacityBranch(_ConditionBranch):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(96, 96, kernel_size=1, bias=True)
        _initialize_conv(self.input_projection)
        self.temporal = _TemporalMixer()
        self.finalize_parameter_match()

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.temporal(F.silu(self.input_projection(latent)))
        return self.project_condition(context)


class TemporalFusionGate(nn.Module):
    """由共享 temporal latent 生成逐时间、逐分支的有界残差门控。"""

    def __init__(self, branch_count: int) -> None:
        super().__init__()
        if int(branch_count) != 3:
            raise ValueError("P6a fusion gate 固定要求 3 个 condition branches")
        self.branch_count = int(branch_count)
        self.norm = nn.GroupNorm(12, 96)
        self.depthwise = nn.Conv1d(96, 96, kernel_size=5, padding=2, groups=96, bias=False)
        self.expand = nn.Conv1d(96, 128, kernel_size=1, bias=False)
        self.final_projection = nn.Conv1d(128, self.branch_count, kernel_size=1, bias=True)
        _initialize_norm(self.norm)
        _initialize_conv(self.depthwise)
        _initialize_conv(self.expand)
        # 初始 factor 严格为 1，使 gated arm 与对应 additive/control arm 同函数。
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
        count = trainable_parameter_count(self)
        if count != P6_GATE_PARAMETER_COUNT:
            raise RuntimeError(f"P6a fusion gate 参数数错误: {count} != {P6_GATE_PARAMETER_COUNT}")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1:] != (96, 1800):
            raise ValueError(f"P6a gate 期望 (B,96,1800)，实际 {tuple(latent.shape)}")
        logits = self.final_projection(F.silu(self.expand(self.depthwise(self.norm(latent)))))
        return 1.0 + 0.5 * torch.tanh(logits)


BRANCH_TYPES = {
    "m": MultiResolutionStftBranch,
    "w": CwtBranch,
    "l": LearnableModulationBranch,
    "s": SstRidgeBranch,
}


class CRDTfV1Model(nn.Module):
    """C201 from-scratch anchor + branch-independent zero-init FiLM。"""

    def __init__(self, variant: str, initialization_seed: int) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in TF_ALL_VARIANT_REPRESENTATIONS:
            raise ValueError(f"未知 CRD-TF variant={variant!r}")
        self.tf_variant = variant
        self.representations = TF_ALL_VARIANT_REPRESENTATIONS[variant]
        # D4 仅缩短 local trunk；独立 module seed 保证其余 C201/W 模块逐 tensor 同初始化。
        local_block_count = 4 if variant in P3_VARIANTS else 6
        self.base = CRDCoarseModel(
            "crd_c201_decoder_10hz_cap",
            int(initialization_seed),
            local_block_count=local_block_count,
        )
        branches = {}
        for name in self.representations:
            with module_seed(int(initialization_seed), f"tf_branch_{name}"):
                if name == "w" and variant in P1_VARIANTS:
                    branches[name] = CwtBranch(scale_count=p1_w_scale_count(variant))
                else:
                    branches[name] = BRANCH_TYPES[name]()
        self.branches = nn.ModuleDict(branches)
        controls = []
        for index in range(TF_ALL_CONTROL_COUNT.get(variant, 0)):
            with module_seed(int(initialization_seed), f"tf_control_{index}"):
                controls.append(TemporalCapacityBranch())
        self.controls = nn.ModuleList(controls)
        if variant in TF_GATED_VARIANTS:
            with module_seed(int(initialization_seed), "tf_fusion_gate"):
                self.fusion_gate: TemporalFusionGate | None = TemporalFusionGate(
                    len(self.representations) + len(self.controls)
                )
        else:
            self.fusion_gate = None
        self._validate_parameter_contract()

    def _validate_parameter_contract(self) -> None:
        for name, branch in self.branches.items():
            count = trainable_parameter_count(branch)
            if abs(count - P_BRANCH_TARGET) / P_BRANCH_TARGET > P_BRANCH_TOLERANCE:
                raise RuntimeError(f"TF branch {name} 参数数 {count} 不满足预算")
        for index, branch in enumerate(self.controls):
            count = trainable_parameter_count(branch)
            if abs(count - P_BRANCH_TARGET) / P_BRANCH_TARGET > P_BRANCH_TOLERANCE:
                raise RuntimeError(f"TF control {index} 参数数 {count} 不满足预算")
        if self.fusion_gate is not None:
            count = trainable_parameter_count(self.fusion_gate)
            if count != P6_GATE_PARAMETER_COUNT:
                raise RuntimeError(f"P6a fusion gate 参数数 {count} 不满足冻结预算")

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        _validate_input(x)
        latent = self.base.encode_local(x)
        gamma_sum = torch.zeros_like(latent)
        beta_sum = torch.zeros_like(latent)
        gate_factors = self.fusion_gate(latent) if self.fusion_gate is not None else None
        condition_index = 0
        expected_keys = _expected_feature_keys(self.representations)
        if self.representations:
            if tf is None or set(tf) != expected_keys:
                raise ValueError(f"{self.tf_variant} 要求 TF keys={sorted(expected_keys)}，实际={sorted(tf or {})}")
            for name, branch in self.branches.items():
                branch_features = {key: tf[key] for key in _expected_feature_keys((name,))}
                if name == "w" and self.tf_variant in P1_VARIANTS:
                    branch_features = {"w": apply_p1_w_view(branch_features["w"], self.tf_variant)}
                gamma, beta = _checkpointed_mapping_branch(branch, branch_features)
                if gate_factors is None:
                    gamma_sum = gamma_sum + 0.5 * torch.tanh(gamma)
                    beta_sum = beta_sum + 0.5 * torch.tanh(beta)
                else:
                    factor = gate_factors[:, condition_index : condition_index + 1]
                    gamma_sum = gamma_sum + factor * (0.5 * torch.tanh(gamma))
                    beta_sum = beta_sum + factor * (0.5 * torch.tanh(beta))
                condition_index += 1
        elif tf not in (None, {}):
            raise ValueError(f"{self.tf_variant} capacity control 不得读取 TF cache")
        for branch in self.controls:
            gamma, beta = _checkpointed_tensor_branch(branch, latent)
            if gate_factors is None:
                gamma_sum = gamma_sum + 0.5 * torch.tanh(gamma)
                beta_sum = beta_sum + 0.5 * torch.tanh(beta)
            else:
                factor = gate_factors[:, condition_index : condition_index + 1]
                gamma_sum = gamma_sum + factor * (0.5 * torch.tanh(gamma))
                beta_sum = beta_sum + factor * (0.5 * torch.tanh(beta))
            condition_index += 1
        conditioned = latent * (1.0 + gamma_sum) + beta_sum
        return self.base.decode_local(conditioned)

    def branch_parameter_counts(self) -> dict[str, int]:
        counts = {name: trainable_parameter_count(branch) for name, branch in self.branches.items()}
        counts.update({f"control_{index}": trainable_parameter_count(branch) for index, branch in enumerate(self.controls)})
        if self.fusion_gate is not None:
            counts["fusion_gate"] = trainable_parameter_count(self.fusion_gate)
        return counts


def _require_feature(
    tf: Mapping[str, torch.Tensor],
    name: str,
    trailing_shape: tuple[int, ...],
    *,
    complex_required: bool = False,
) -> torch.Tensor:
    if name not in tf:
        raise KeyError(f"TF cache 缺少 {name}")
    value = tf[name]
    if value.ndim != len(trailing_shape) + 1 or tuple(value.shape[1:]) != trailing_shape:
        raise ValueError(f"TF {name} 期望 (B,{','.join(map(str, trailing_shape))})，实际 {tuple(value.shape)}")
    if complex_required != bool(torch.is_complex(value)):
        expected = "complex" if complex_required else "real"
        raise ValueError(f"TF {name} 必须是 {expected} tensor")
    finite = torch.isfinite(value.real).all() and torch.isfinite(value.imag).all() if torch.is_complex(value) else torch.isfinite(value).all()
    if not bool(finite):
        raise FloatingPointError(f"TF {name} 包含 NaN/Inf")
    return value


def _expected_feature_keys(representations: tuple[str, ...]) -> set[str]:
    keys: set[str] = set()
    for name in representations:
        if name == "m":
            keys.update(("m_slow", "m_fast"))
        elif name == "l":
            keys.add("l_spectrum")
        else:
            keys.add(name)
    return keys


def _checkpointed_mapping_branch(
    branch: nn.Module,
    features: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """训练态按 physical-batch 分块重算完整 TF branch，降低中间 activation 峰值。"""

    keys = tuple(sorted(features))
    batch_size = int(features[keys[0]].shape[0])
    if not branch.training or batch_size <= TF_BRANCH_CHECKPOINT_BATCH_CHUNK:
        return branch(features)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, batch_size, TF_BRANCH_CHECKPOINT_BATCH_CHUNK):
        stop = min(start + TF_BRANCH_CHECKPOINT_BATCH_CHUNK, batch_size)
        chunks = tuple(features[key][start:stop] for key in keys)

        def run(*values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return branch(dict(zip(keys, values)))

        outputs.append(checkpoint(run, *chunks, use_reentrant=False, preserve_rng_state=False))
    gamma = torch.cat([output[0] for output in outputs], dim=0)
    beta = torch.cat([output[1] for output in outputs], dim=0)
    return gamma, beta


def _checkpointed_tensor_branch(
    branch: nn.Module,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not branch.training or int(latent.shape[0]) <= TF_BRANCH_CHECKPOINT_BATCH_CHUNK:
        return branch(latent)
    outputs = [
        checkpoint(
            branch,
            latent[start : start + TF_BRANCH_CHECKPOINT_BATCH_CHUNK],
            use_reentrant=False,
            preserve_rng_state=False,
        )
        for start in range(0, int(latent.shape[0]), TF_BRANCH_CHECKPOINT_BATCH_CHUNK)
    ]
    gamma = torch.cat([output[0] for output in outputs], dim=0)
    beta = torch.cat([output[1] for output in outputs], dim=0)
    return gamma, beta
