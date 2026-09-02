from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block, CoarseWaveformHead, DecoderResidual, ResidualDWBlock
from resp_train.crd.initialization import module_seed
from resp_train.crd.spectral_ops import fourier_interpolate
from resp_train.paper_evidence.center30_config import (
    CENTER30_INPUT_SAMPLES,
    CENTER30_MODEL_VARIANTS,
    CENTER30_OUTPUT_SAMPLES,
)
from resp_train.paper_evidence.center30_data import center30_latent_bounds
from resp_train.paper_evidence.center_context_model import DynamicCwtBranch, VariablePatchTokenFrontend


class Center30ContextModel(nn.Module):
    def __init__(
        self,
        variant: str,
        initialization_seed: int,
        *,
        mamba_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        if self.variant not in CENTER30_MODEL_VARIANTS:
            raise ValueError(f"center30 variant 只允许 {list(CENTER30_MODEL_VARIANTS)}")
        seed = int(initialization_seed)
        with module_seed(seed, "patch_frontend"):
            self.frontend = VariablePatchTokenFrontend()
        with module_seed(seed, "local_trunk"):
            self.local_blocks = nn.ModuleList(
                [BidirectionalMamba2Block(96, mamba_factory=mamba_factory) for _ in range(6)]
            )
        with module_seed(seed, "refinement"):
            self.refinement = nn.Sequential(ResidualDWBlock(96, dilation=1), ResidualDWBlock(96, dilation=2))
        with module_seed(seed, "coarse_head"):
            self.head = CoarseWaveformHead()
        with module_seed(seed, "decoder_residual"):
            self.decoder_residual = DecoderResidual()
        if self.variant == "w_reduced_center30":
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
        _validate_input(x)
        input_samples = int(x.shape[-1])
        latent_length = input_samples // 10
        latent = self.frontend(x)
        if latent.shape[1:] != (96, latent_length):
            raise RuntimeError(f"center30 frontend 输出契约错误: {tuple(latent.shape)}")
        tokens = latent.transpose(1, 2)
        for block in self.local_blocks:
            tokens = block(tokens)
        latent = tokens.transpose(1, 2)
        if self.w_branch is None:
            if tf not in (None, {}):
                raise ValueError("C201-center30 不得读取 TF cache")
        else:
            if tf is None:
                raise ValueError("W-reduced-center30 缺少 length-specific 49-scale W cache")
            gamma, beta = self.w_branch.forward_dynamic(
                tf,
                input_samples=input_samples,
                latent_length=latent_length,
            )
            latent = latent * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)
        latent = self.refinement(latent)
        head_features = self.head.features(latent)
        start, stop = center30_latent_bounds(input_samples)
        center_features = head_features[..., start:stop]
        if center_features.shape[1:] != (32, 300):
            raise RuntimeError(f"center30 head feature 契约错误: {tuple(center_features.shape)}")
        waveform_10hz = self.head.output(center_features) + self.decoder_residual(center_features)
        waveform = fourier_interpolate(waveform_10hz, target_length=CENTER30_OUTPUT_SAMPLES)
        if waveform.shape[1:] != (1, CENTER30_OUTPUT_SAMPLES) or not bool(torch.isfinite(waveform).all()):
            raise FloatingPointError("center30 输出 shape/finite 错误")
        return {"waveform": waveform, "waveform_10hz": waveform_10hz}

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_center30_model(cfg: Any, *, mamba_factory: Callable[..., nn.Module] | None = None) -> Center30ContextModel:
    if str(cfg.model.name) != "paper_center30_context_v1":
        raise ValueError("center30 builder 要求 model.name=paper_center30_context_v1")
    return Center30ContextModel(
        str(cfg.model.variant),
        int(cfg.model.initialization_seed),
        mamba_factory=mamba_factory,
    )


def center30_shared_state_identity(
    c201: Center30ContextModel,
    w_reduced: Center30ContextModel,
) -> dict[str, Any]:
    if c201.variant != "c201_center30" or w_reduced.variant != "w_reduced_center30":
        raise ValueError("center30 shared-state audit variant 错误")
    left, right = c201.state_dict(), w_reduced.state_dict()
    shared = sorted(set(left) & set(right))
    mismatched = [name for name in shared if not torch.equal(left[name], right[name])]
    w_only = sorted(set(right) - set(left))
    if mismatched or not w_only or any(not name.startswith("w_branch.") for name in w_only):
        raise RuntimeError("center30 C201/W-reduced 初始化身份不一致")
    return {"shared_tensor_count": len(shared), "shared_tensors_identical": True, "w_only_tensor_count": len(w_only)}


def _validate_input(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[1] != 1 or int(x.shape[-1]) not in CENTER30_INPUT_SAMPLES:
        raise ValueError(f"center30 模型期望 (B,1,N), N∈{list(CENTER30_INPUT_SAMPLES)}，实际 {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("center30 输入包含 NaN/Inf")


__all__ = [
    "Center30ContextModel",
    "build_center30_model",
    "center30_shared_state_identity",
]
