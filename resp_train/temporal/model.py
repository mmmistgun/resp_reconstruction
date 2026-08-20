from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from resp_train.temporal.blocks import (
    LATENT_CHANNELS,
    LATENT_LENGTH,
    OUTPUT_LENGTH,
    BidirectionalLSTMTrunk,
    BidirectionalMambaTrunk,
    DilatedTCNBlock,
    MultiscaleTemporalPyramid,
    TemporalDecoder,
    TemporalStem,
    tcn_receptive_field_tokens,
)
from resp_train.temporal.initialization import module_seed


@dataclass(frozen=True)
class RTMVariantSpec:
    variant: str
    family: str
    role: str
    expected_trainable_parameters: int
    context_description: str


RTM_VARIANT_SPECS: dict[str, RTMVariantSpec] = {
    "rtm_v1_t0_locked_stem_head": RTMVariantSpec(
        "rtm_v1_t0_locked_stem_head",
        "no_temporal_trunk",
        "control",
        59_042,
        "locked common stem/head; no temporal trunk",
    ),
    "rtm_v1_tcn_d9_h384": RTMVariantSpec(
        "rtm_v1_tcn_d9_h384",
        "dilated_tcn",
        "architecture_capability",
        729_506,
        "9 blocks; H=384; RF=2045 tokens (204.5 s)",
    ),
    "rtm_v1_bimamba2_d96_l6": RTMVariantSpec(
        "rtm_v1_bimamba2_d96_l6",
        "bimamba2",
        "architecture_capability",
        1_010_426,
        "6 independent-forward/backward official Mamba2 blocks",
    ),
    "rtm_v1_bilstm_h96_l2": RTMVariantSpec(
        "rtm_v1_bilstm_h96_l2",
        "bilstm",
        "architecture_capability",
        449_474,
        "2 bidirectional layers; H=96 per direction",
    ),
    "rtm_v1_multiscale_10_2_1_h384": RTMVariantSpec(
        "rtm_v1_multiscale_10_2_1_h384",
        "feature_level_multiscale",
        "architecture_capability",
        1_129_730,
        "10/2/1-Hz features; H=384; 1-Hz branch RF=253 s",
    ),
}

RTM_VARIANTS = tuple(RTM_VARIANT_SPECS)
TCN_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MULTISCALE_COARSE_DILATIONS = (1, 2, 4, 8, 16, 32)


class RespTemporalModel(nn.Module):
    """RTM-v1 锁定共同 substrate/decoder 与 family-specific temporal trunk。"""

    def __init__(
        self,
        variant: str,
        initialization_seed: int,
        *,
        mamba_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        variant = str(variant).strip().lower()
        if variant not in RTM_VARIANT_SPECS:
            raise ValueError(f"未知 RTM-v1 variant={variant!r}；可选 {list(RTM_VARIANT_SPECS)}")
        self.variant = variant
        self.spec = RTM_VARIANT_SPECS[variant]
        with module_seed(initialization_seed, "common_stem"):
            self.stem = TemporalStem()
        with module_seed(initialization_seed, f"trunk_{self.spec.family}"):
            self.trunk = self._build_trunk(mamba_factory=mamba_factory)
        with module_seed(initialization_seed, "common_decoder"):
            self.decoder = TemporalDecoder()

    def _build_trunk(self, *, mamba_factory: Callable[..., nn.Module] | None) -> nn.Module:
        family = self.spec.family
        if family == "no_temporal_trunk":
            return nn.Identity()
        if family == "dilated_tcn":
            return nn.Sequential(
                *[
                    DilatedTCNBlock(hidden_channels=384, dilation=dilation)
                    for dilation in TCN_DILATIONS
                ]
            )
        if family == "bimamba2":
            return BidirectionalMambaTrunk(layers=6, mamba_factory=mamba_factory)
        if family == "bilstm":
            return BidirectionalLSTMTrunk(layers=2)
        if family == "feature_level_multiscale":
            return MultiscaleTemporalPyramid(hidden_channels=384)
        raise RuntimeError(f"未实现 RTM-v1 family={family!r}")

    def forward(self, x: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (1, OUTPUT_LENGTH):
            raise ValueError(f"RTM-v1 期望输入 (B,1,{OUTPUT_LENGTH})，实际 {tuple(x.shape)}")
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("RTM-v1 输入包含 NaN/Inf")
        latent = self.trunk(self.stem(x))
        if latent.shape[1:] != (LATENT_CHANNELS, LATENT_LENGTH):
            raise RuntimeError(f"RTM-v1 trunk 输出契约错误: {tuple(latent.shape)}")
        return self.decoder(latent)


def build_resp_temporal_model(
    cfg: Any,
    *,
    mamba_factory: Callable[..., nn.Module] | None = None,
) -> RespTemporalModel:
    if str(cfg.model.name) != "resp_temporal_v1":
        raise ValueError(f"RTM-v1 builder 要求 model.name=resp_temporal_v1，当前为 {cfg.model.name!r}")
    model = RespTemporalModel(
        str(cfg.model.variant),
        int(cfg.model.initialization_seed),
        mamba_factory=mamba_factory,
    )
    validate_variant_structure(model)
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def validate_variant_structure(model: RespTemporalModel) -> None:
    """不执行 forward 的静态锁检查，供 config/test/未来 resource preflight 复用。"""

    observed = trainable_parameter_count(model)
    expected = model.spec.expected_trainable_parameters
    if observed != expected:
        raise RuntimeError(f"{model.variant} 参数数漂移: {observed} != {expected}")
    if model.spec.family == "dilated_tcn":
        receptive_field = tcn_receptive_field_tokens(TCN_DILATIONS)
        if receptive_field != 2045 or receptive_field < LATENT_LENGTH:
            raise RuntimeError(f"RTM-v1 TCN 感受野错误: {receptive_field}")
    if model.spec.family == "feature_level_multiscale":
        coarse_receptive_field = tcn_receptive_field_tokens(MULTISCALE_COARSE_DILATIONS)
        if coarse_receptive_field != 253 or coarse_receptive_field < 180:
            raise RuntimeError(f"RTM-v1 1-Hz coarse 感受野错误: {coarse_receptive_field}")
