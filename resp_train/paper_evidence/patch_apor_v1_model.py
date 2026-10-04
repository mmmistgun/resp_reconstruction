"""PATCH/TM0/REF0 模块消融与 APOR 的独立模型入口。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.initialization import module_seed
from resp_train.crd.spectral_ops import fourier_interpolate
from resp_train.crd.tf_v1_model import (
    CRDTfV1Model,
    _checkpointed_mapping_branch,
    _initialize_conv,
    _require_feature,
)

PROTOCOL = "patch-apor-module-v1-20260930"


@dataclass(frozen=True)
class ArmSpec:
    family: str = "patch"
    condition: bool = True
    mamba: bool = True
    scale_neighborhood: bool = True
    condition_refiner: bool = True
    decoder_residual: bool = True
    film: str = "both"
    condition_points: int = 5
    patch_decoder: str = "mlp"
    dense_trunk: bool = False
    overlap_window: str = "hann"


ARM_SPECS = {
    "M0": ArmSpec(),
    "M1": ArmSpec(condition=False),
    "M2": ArmSpec(mamba=False),
    "M3": ArmSpec(scale_neighborhood=False),
    "M4": ArmSpec(condition_refiner=False),
    "M5": ArmSpec(decoder_residual=False),
    "M6": ArmSpec(film="add"),
    "M7": ArmSpec(film="scale"),
    "A0": ArmSpec(family="apor", decoder_residual=False),
    "A1": ArmSpec(family="apor", decoder_residual=False, condition_points=1),
    "A2": ArmSpec(family="apor", decoder_residual=False, patch_decoder="linear"),
    "A3": ArmSpec(family="apor", decoder_residual=False, dense_trunk=True),
    "A4": ArmSpec(family="apor", decoder_residual=False, overlap_window="uniform"),
}
ARMS = tuple(ARM_SPECS)
SEEDS = (20260811, 20260812, 20260813)
PARAMETERS = {
    "M0": 1031690, "M1": 994538, "M2": 80306, "M3": 1030826,
    "M4": 1019114, "M5": 1030633, "M6": 1022378, "M7": 1022378,
    "A0": 1077640, "A1": 1040776, "A2": 1068328, "A3": 1077640, "A4": 1077640,
}


def model_contract(arm: str) -> dict[str, Any]:
    payload = asdict(ARM_SPECS[arm])
    if payload["family"] == "patch":
        for key in ("condition_points", "patch_decoder", "dense_trunk", "overlap_window"):
            del payload[key]
    return {"protocol": PROTOCOL, "arm": arm, "parameters": PARAMETERS[arm], **payload}


class CoordinateSample(nn.Module):
    """在100-Hz原始采样坐标中线性取样；边界延拓到最近有效点。"""

    def __init__(self, length: int, origin: float, step: float, positions: torch.Tensor):
        super().__init__()
        if length < 2 or step <= 0 or not torch.isfinite(positions).all():
            raise ValueError("非法时间网格")
        self.length = length
        self.output_shape = tuple(positions.shape)
        indices = ((positions.double().flatten() - origin) / step).clamp(0, length - 1)
        left = indices.floor().long()
        self.register_buffer("left", left)
        self.register_buffer("right", (left + 1).clamp_max(length - 1))
        self.register_buffer("fraction", (indices - left).float())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != self.length:
            raise ValueError("时间采样输入长度不符")
        # 坐标精度保持float32，避免BF16把相邻采样权重过早舍入。
        left = value.float().index_select(-1, self.left)
        right = value.float().index_select(-1, self.right)
        result = left + (right - left) * self.fraction
        return result.reshape(*value.shape[:2], *self.output_shape).to(value.dtype)


class OverlapAdd(nn.Module):
    """256点片段、128点步长；分子/分母均以float32折叠。"""

    def __init__(self, window: str):
        super().__init__()
        if window not in {"hann", "uniform"}:
            raise ValueError("未知重叠权重")
        weight = torch.hann_window(256, periodic=False).clamp_min(1e-3)
        if window == "uniform":
            weight = torch.ones(256)
        self.register_buffer("weight", weight)
        self.register_buffer("denominator", self._fold(weight[None, :, None].expand(1, 256, 140)))
        if not bool((self.denominator > 0).all()):
            raise RuntimeError("重叠合成存在未覆盖点")

    @staticmethod
    def _fold(value: torch.Tensor) -> torch.Tensor:
        return F.fold(value, (1, 18048), (1, 256), stride=(1, 128)).flatten(2)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.shape[1:] != (140, 256):
            raise ValueError("重叠合成要求(B,140,256)")
        values = patches.float().transpose(1, 2) * self.weight[None, :, None]
        return (self._fold(values) / self.denominator)[..., :18000]


class PatchCondition(nn.Module):
    def __init__(self, source: nn.Module, spec: ArmSpec, seed: int):
        super().__init__()
        for name in ("conv_in", "norm", "depthwise", "conv_out", "parameter_fill", "final_projection"):
            setattr(self, name, getattr(source, name))
        self.spec = spec
        if not spec.scale_neighborhood:
            with module_seed(seed, "patch_apor_scale_pointwise"):
                self.conv_in = nn.Conv2d(1, 48, (1, 3), padding=(0, 1), bias=False)
                self.depthwise = nn.Conv2d(48, 48, (1, 3), padding=(0, 1), groups=48, bias=False)
                _initialize_conv(self.conv_in)
                _initialize_conv(self.depthwise)
        if not spec.condition_refiner:
            self.parameter_fill = nn.Identity()
        if spec.film != "both":
            original = self.final_projection
            with module_seed(seed, "patch_apor_film_single"):
                self.final_projection = nn.Conv1d(96, 96, 1)
            start = 96 if spec.film == "add" else 0
            with torch.no_grad():
                self.final_projection.weight.copy_(original.weight[start:start + 96])
                self.final_projection.bias.copy_(original.bias[start:start + 96])
        if spec.family == "apor":
            offsets = torch.linspace(0, 255, 5) if spec.condition_points == 5 else torch.tensor([127.5])
            positions = torch.arange(140)[:, None] * 128 + offsets[None]
            self.sample = CoordinateSample(360, 24.5, 50, positions)
            with module_seed(seed, f"apor_condition_{spec.condition_points}"):
                self.local_projection = nn.Conv1d(96 * spec.condition_points, 96, 1)
                _initialize_conv(self.local_projection)

    def forward(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        w = _require_feature(tf, "w", (97, 360)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(w)))
        value = F.silu(self.conv_out(self.depthwise(value))).mean(dim=2)
        if self.spec.family == "apor":
            sampled = self.sample(value)  # B,C,patch,位置
            value = sampled.permute(0, 1, 3, 2).reshape(value.shape[0], -1, 140)
            value = self.local_projection(value)
        else:
            value = F.interpolate(value, size=1800, mode="linear", align_corners=False)
        projected = self.final_projection(self.parameter_fill(value))
        if self.spec.film == "add":
            return torch.zeros_like(projected), projected
        if self.spec.film == "scale":
            return projected, torch.zeros_like(projected)
        return projected.chunk(2, dim=1)


class PatchAporModel(CRDTfV1Model):
    """先完整构造同seed W0，再按模块定义修改，保持公共初始化一致。"""

    def __init__(self, arm: str, seed: int):
        if arm not in ARM_SPECS:
            raise ValueError(f"未知实验臂: {arm}")
        super().__init__("crd_tf102_w", seed)
        self.arm = arm
        self.spec = ARM_SPECS[arm]
        self.base.refinement = nn.Identity()
        if self.spec.condition:
            self.branches["w"] = PatchCondition(self.branches["w"], self.spec, seed)
        else:
            self.branches = nn.ModuleDict()
            self.representations = ()
        if not self.spec.mamba:
            self.base.local_blocks = nn.ModuleList()
            self.base.local_block_count = 0
        if not self.spec.decoder_residual:
            self.base.decoder_residual = None
        if self.spec.family == "apor":
            # APOR只注册实际参与计算的片段读出，释放原coarse head与残差读出。
            self.base.head = nn.Identity()
            self.base.decoder_residual = None
            with module_seed(seed, f"apor_decoder_{self.spec.patch_decoder}"):
                self.patch_head = (
                    nn.Sequential(nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 256))
                    if self.spec.patch_decoder == "mlp" else nn.Linear(96, 256)
                )
            self.overlap = OverlapAdd(self.spec.overlap_window)
            if self.spec.dense_trunk:
                self.to_dense = CoordinateSample(140, 127.5, 128, torch.arange(1800) * 10)
                self.to_patch = CoordinateSample(1800, 0, 10, torch.arange(140) * 128 + 127.5)
        if sum(p.numel() for p in self.parameters()) != PARAMETERS[arm]:
            raise RuntimeError(f"{arm}结构参数数量漂移")

    def forward(self, x: torch.Tensor, *, tf: Mapping[str, torch.Tensor] | None = None, **_: Any):
        if x.ndim != 3 or x.shape[1:] != (1, 18000) or x.shape[0] < 1:
            raise ValueError("输入要求(B,1,18000)")
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("输入包含NaN/Inf")
        if self.spec.condition:
            if tf is None or set(tf) != {"w"} or tf["w"].shape[0] != x.shape[0]:
                raise ValueError("条件输入要求同batch的w")
        elif tf is not None and len(tf):
            raise ValueError("M1不接收条件输入")
        if self.spec.family == "apor":
            frontend = self.base.frontend
            value = frontend.encoder(x)
            if self.spec.dense_trunk:
                value = self.to_dense(value)
            latent = frontend.activation(frontend.norm(frontend.adapter(value)))
            tokens = latent.transpose(1, 2)
            for block in self.base.local_blocks:
                tokens = block(tokens)
            latent = tokens.transpose(1, 2)
            if self.spec.dense_trunk:
                latent = self.to_patch(latent)
        else:
            latent = self.base.encode_local(x)
        if self.spec.condition:
            gamma, beta = _checkpointed_mapping_branch(self.branches["w"], tf)
            latent = latent * (1 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)
        if self.spec.family == "apor":
            output = {"waveform": self.overlap(self.patch_head(latent.transpose(1, 2)))}
        elif self.spec.decoder_residual:
            output = self.base.decode_local(latent)
        else:
            coarse = self.base.head(latent)
            output = {"waveform": fourier_interpolate(coarse, target_length=18000), "waveform_10hz": coarse}
        if any(not bool(torch.isfinite(value).all()) for value in output.values()):
            raise FloatingPointError("模型输出包含NaN/Inf")
        return output


def build_model(cfg: Any) -> PatchAporModel:
    from omegaconf import OmegaConf

    contract = OmegaConf.to_container(cfg.model.patch_apor_v1, resolve=True)
    arm = contract["arm"]
    if cfg.protocol.name != PROTOCOL or contract != model_contract(arm):
        raise ValueError("模型合同不符")
    return PatchAporModel(arm, int(cfg.model.initialization_seed))
