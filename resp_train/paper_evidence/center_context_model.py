from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block, CoarseWaveformHead, DecoderResidual, ResidualDWBlock
from resp_train.crd.frontends import PatchTokenFrontend
from resp_train.crd.initialization import module_seed
from resp_train.crd.spectral_ops import fourier_interpolate
from resp_train.crd.tf_v1_model import CwtBranch
from resp_train.paper_evidence.center_context_config import CENTER_CONTEXT_INPUT_SAMPLES, CENTER_CONTEXT_MODEL_VARIANTS
from resp_train.paper_evidence.center_context_data import CENTER_OUTPUT_SAMPLES, latent_center_bounds


class VariablePatchTokenFrontend(PatchTokenFrontend):
    """保持 C201 frontend 参数/初始化，只把插值长度改为 N/10。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)
        target_length = int(x.shape[-1]) // 10
        tokens = F.interpolate(tokens, size=target_length, mode="linear", align_corners=False)
        return self.activation(self.norm(self.adapter(tokens)))


class DynamicCwtBranch(CwtBranch):
    """复用 49-scale CWT branch 参数化，只动态处理 cache 时间轴和 latent 长度。"""

    def forward_dynamic(
        self,
        tf: Mapping[str, torch.Tensor],
        *,
        input_samples: int,
        latent_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if set(tf) != {"w"}:
            raise ValueError(f"W-reduced 要求且只允许 TF key=['w']，实际={sorted(tf)}")
        value = tf["w"]
        expected_context = int(input_samples) // 50
        if value.ndim != 3 or tuple(value.shape[1:]) != (49, expected_context):
            raise ValueError(f"中心 W-reduced feature 期望 (B,49,{expected_context})，实际 {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("中心 W feature 包含 NaN/Inf")
        value = value.float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        value = F.silu(self.conv_out(self.depthwise(value))).mean(dim=2)
        value = F.interpolate(value, size=int(latent_length), mode="linear", align_corners=False)
        context = self.temporal(value)
        if self.parameter_fill is None:
            raise RuntimeError("中心 W branch 尚未完成参数匹配")
        context = self.parameter_fill(context)
        return self.final_projection(context).chunk(2, dim=1)


class CenterContextModel(nn.Module):
    """C201/W-reduced 的独立变长输入、固定中心 60 s 输出实现。"""

    def __init__(
        self,
        variant: str,
        initialization_seed: int,
        *,
        mamba_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        if self.variant not in CENTER_CONTEXT_MODEL_VARIANTS:
            raise ValueError(f"中心上下文 variant 只允许 {list(CENTER_CONTEXT_MODEL_VARIANTS)}")
        seed = int(initialization_seed)
        with module_seed(seed, "patch_frontend"):
            self.frontend = VariablePatchTokenFrontend()
        with module_seed(seed, "local_trunk"):
            self.local_blocks = nn.ModuleList(
                [BidirectionalMamba2Block(96, mamba_factory=mamba_factory) for _ in range(6)]
            )
        with module_seed(seed, "refinement"):
            self.refinement = nn.Sequential(
                ResidualDWBlock(96, dilation=1),
                ResidualDWBlock(96, dilation=2),
            )
        with module_seed(seed, "coarse_head"):
            self.head = CoarseWaveformHead()
        with module_seed(seed, "decoder_residual"):
            self.decoder_residual = DecoderResidual()
        if self.variant == "w_reduced_center60":
            with module_seed(seed, "tf_branch_w"):
                self.w_branch: DynamicCwtBranch | None = DynamicCwtBranch(scale_count=49)
        else:
            self.w_branch = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        _validate_center_input(x)
        input_samples = int(x.shape[-1])
        latent_length = input_samples // 10
        latent = self.frontend(x)
        if latent.shape[1:] != (96, latent_length):
            raise RuntimeError(f"中心 frontend 输出契约错误: {tuple(latent.shape)}")
        tokens = latent.transpose(1, 2)
        for block in self.local_blocks:
            tokens = block(tokens)
        latent = tokens.transpose(1, 2)

        if self.w_branch is None:
            if tf not in (None, {}):
                raise ValueError("C201-center60 不得读取 TF cache")
        else:
            if tf is None:
                raise ValueError("W-reduced-center60 缺少 length-specific 49-scale W cache")
            gamma, beta = self.w_branch.forward_dynamic(
                tf,
                input_samples=input_samples,
                latent_length=latent_length,
            )
            latent = latent * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)

        # refinement 和带局部卷积的 head features 必须先看完整上下文，再裁中心。
        latent = self.refinement(latent)
        head_features = self.head.features(latent)
        start, stop = latent_center_bounds(input_samples)
        center_features = head_features[..., start:stop]
        if center_features.shape[1:] != (32, 600):
            raise RuntimeError(f"中心 head feature 契约错误: {tuple(center_features.shape)}")
        waveform_10hz_base = self.head.output(center_features)
        waveform_10hz = waveform_10hz_base + self.decoder_residual(center_features)
        waveform = fourier_interpolate(waveform_10hz, target_length=CENTER_OUTPUT_SAMPLES)
        if waveform.shape[1:] != (1, CENTER_OUTPUT_SAMPLES) or not bool(torch.isfinite(waveform).all()):
            raise FloatingPointError("中心模型输出 shape/finite 错误")
        return {"waveform": waveform, "waveform_10hz": waveform_10hz}

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_center_context_model(cfg: Any, *, mamba_factory: Callable[..., nn.Module] | None = None) -> CenterContextModel:
    if str(cfg.model.name) != "paper_center_context_v1":
        raise ValueError("中心上下文 builder 要求 model.name=paper_center_context_v1")
    return CenterContextModel(
        str(cfg.model.variant),
        int(cfg.model.initialization_seed),
        mamba_factory=mamba_factory,
    )


def shared_state_identity(
    c201: CenterContextModel,
    w_reduced: CenterContextModel,
) -> dict[str, Any]:
    if c201.variant != "c201_center60" or w_reduced.variant != "w_reduced_center60":
        raise ValueError("shared state audit 要求 C201-center60 与 W-reduced-center60")
    left = c201.state_dict()
    right = w_reduced.state_dict()
    shared = sorted(set(left) & set(right))
    mismatched = [name for name in shared if not torch.equal(left[name], right[name])]
    w_only = sorted(set(right) - set(left))
    if mismatched:
        raise RuntimeError(f"C201/W-reduced 共享 state 初始化不一致: {mismatched[:10]}")
    if not w_only or any(not name.startswith("w_branch.") for name in w_only):
        raise RuntimeError("W-reduced 独有 state 必须且只能来自 w_branch")
    return {
        "shared_tensor_count": len(shared),
        "shared_tensors_identical": True,
        "w_only_tensor_count": len(w_only),
        "w_only_prefix": "w_branch.",
    }


def _validate_center_input(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[1] != 1 or int(x.shape[-1]) not in CENTER_CONTEXT_INPUT_SAMPLES:
        raise ValueError(f"中心模型期望 (B,1,N), N∈{list(CENTER_CONTEXT_INPUT_SAMPLES)}，实际 {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("中心模型输入包含 NaN/Inf")


__all__ = [
    "CenterContextModel",
    "DynamicCwtBranch",
    "VariablePatchTokenFrontend",
    "build_center_context_model",
    "shared_state_identity",
]
