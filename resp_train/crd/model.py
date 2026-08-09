from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block, CoarseWaveformHead, ResidualDWBlock
from resp_train.crd.frontends import DirectAnalyticFrontend, PatchTokenFrontend
from resp_train.crd.initialization import module_seed
from resp_train.crd.representations import (
    AnalyticAMRepresentation,
    LegacyEnergyRepresentation,
    MorphologyRepresentation,
)
from resp_train.crd.spectral_ops import fft_hard_lowpass, fourier_interpolate
from resp_train.models.stft_branch import TimeStftDual1D
from resp_train.models.timeseries import PatchMixer1D


CRD_VARIANTS = (
    "crd_001_b0_retrain",
    "crd_002_t4_retrain",
    "crd_101_b0_coarse",
    "crd_102_b0_local_mamba",
    "crd_103_direct_local_mamba",
    "crd_104_direct_hier_mamba",
    "crd_105_direct_coarse",
    "crd_106_b0_hier_mamba",
    "crd_202_base_legacy_energy",
    "crd_203_base_analytic_am",
    "crd_204_base_morphology",
)

CRD_S2A_VARIANTS = {
    "crd_202_base_legacy_energy",
    "crd_203_base_analytic_am",
    "crd_204_base_morphology",
}


def _validate_input(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[1:] != (1, 18000):
        raise ValueError(f"CRD-v1 期望输入 (B,1,18000)，实际 {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("CRD-v1 输入包含 NaN/Inf")


class LegacyB0Retrain(nn.Module):
    """S0 的原始 B0 结构，只更换 CRD 训练协议。"""

    def __init__(self, initialization_seed: int) -> None:
        super().__init__()
        with module_seed(initialization_seed, "patch_frontend"):
            self.patch_backbone = PatchMixer1D(
                in_channels=1,
                out_channels=1,
                base_channels=16,
                patch_len=256,
                patch_stride=128,
                mixer_layers=2,
                overlap_window="hann",
                output_smoothing_kernel=1,
            )

    def forward(self, x: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        _validate_input(x)
        return {"waveform": self.patch_backbone(x)}


class LegacyT4Retrain(nn.Module):
    """S0 的原始 T4 bandenergy/native-pre-mixer 结构。"""

    def __init__(self, initialization_seed: int) -> None:
        super().__init__()
        # TimeStftDual1D 先创建 time_backbone；同名子 seed 保证其 PatchMixer 与 B0 一致。
        with module_seed(initialization_seed, "patch_frontend"):
            self.t4 = TimeStftDual1D(
                time_backbone_name="patch_mixer1d",
                time_backbone_kwargs={
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_channels": 16,
                    "patch_len": 256,
                    "patch_stride": 128,
                    "mixer_layers": 2,
                    "overlap_window": "hann",
                    "output_smoothing_kernel": 1,
                },
                time_feat_channels=16,
                branch_mode="dual",
                out_length=18000,
                fuse_len=600,
                stft_kwargs={
                    "sample_rate": 100.0,
                    "stft_win": 2000,
                    "stft_hop": 250,
                    "low_hz": 0.05,
                    "high_hz": 8.0,
                    "out_channels": 16,
                    "norm": "n0",
                    "encoder_type": "bandenergy",
                },
                fusion_mode="native_inject",
                stft_inject_position="pre_mixer",
            )

    def forward(self, x: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        _validate_input(x)
        output = self.t4(x)
        if isinstance(output, Mapping):
            return dict(output)
        return {"waveform": output}


class GlobalContextStage(nn.Module):
    """1-Hz global BiMamba2 与 zero-init FiLM feedback。"""

    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Conv1d(96, 128, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.project.weight, mode="fan_in", nonlinearity="relu")
        self.blocks = nn.ModuleList([BidirectionalMamba2Block(128) for _ in range(4)])
        self.film = nn.Conv1d(128, 192, kernel_size=1, bias=True)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, local: torch.Tensor) -> torch.Tensor:
        global_input = fft_hard_lowpass(local, sample_rate=10.0, cutoff_hz=0.45)[..., ::10]
        if global_input.shape[-1] != 180:
            raise RuntimeError(f"global stage 期望 180 tokens，实际 {global_input.shape[-1]}")
        global_tokens = self.project(global_input).transpose(1, 2)
        for block in self.blocks:
            global_tokens = block(global_tokens)
        global_features = fourier_interpolate(global_tokens.transpose(1, 2), target_length=1800)
        gamma, beta = self.film(global_features).chunk(2, dim=1)
        return local * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)


class CRDCoarseModel(nn.Module):
    """S1/S2 共享 trunk；variant 只控制协议预注册的静态因素。"""

    def __init__(self, variant: str, initialization_seed: int) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in CRD_VARIANTS[2:]:
            raise ValueError(f"CRDCoarseModel 不支持 variant={variant!r}")
        self.variant = variant
        uses_direct = variant in {
            "crd_103_direct_local_mamba",
            "crd_104_direct_hier_mamba",
            "crd_105_direct_coarse",
        }
        uses_local = variant in {
            "crd_102_b0_local_mamba",
            "crd_103_direct_local_mamba",
            "crd_104_direct_hier_mamba",
            "crd_106_b0_hier_mamba",
            *CRD_S2A_VARIANTS,
        }
        uses_global = variant in {
            "crd_104_direct_hier_mamba",
            "crd_106_b0_hier_mamba",
        }

        with module_seed(initialization_seed, "direct_frontend" if uses_direct else "patch_frontend"):
            self.frontend: nn.Module = DirectAnalyticFrontend() if uses_direct else PatchTokenFrontend()
        if variant == "crd_202_base_legacy_energy":
            self.representation: nn.Module | None = LegacyEnergyRepresentation(initialization_seed)
        elif variant == "crd_203_base_analytic_am":
            self.representation = AnalyticAMRepresentation(initialization_seed)
        elif variant == "crd_204_base_morphology":
            self.representation = MorphologyRepresentation(initialization_seed)
        else:
            self.representation = None
        if uses_local:
            with module_seed(initialization_seed, "local_trunk"):
                self.local_blocks = nn.ModuleList([BidirectionalMamba2Block(96) for _ in range(6)])
        else:
            self.local_blocks = nn.ModuleList()
        if uses_global:
            with module_seed(initialization_seed, "global_trunk"):
                self.global_stage: GlobalContextStage | None = GlobalContextStage()
        else:
            self.global_stage = None
        with module_seed(initialization_seed, "refinement"):
            self.refinement = nn.Sequential(
                ResidualDWBlock(96, dilation=1),
                ResidualDWBlock(96, dilation=2),
            )
        with module_seed(initialization_seed, "coarse_head"):
            self.head = CoarseWaveformHead()

    def forward(self, x: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        _validate_input(x)
        latent = self.frontend(x)
        if latent.shape[1:] != (96, 1800):
            raise RuntimeError(f"CRD frontend 输出契约错误: {tuple(latent.shape)}")
        if self.representation is not None:
            representation = self.representation(x)
            if representation.shape != latent.shape:
                raise RuntimeError(
                    f"CRD representation 输出契约错误: {tuple(representation.shape)} != {tuple(latent.shape)}"
                )
            latent = latent + representation
        if self.local_blocks:
            tokens = latent.transpose(1, 2)
            for block in self.local_blocks:
                tokens = block(tokens)
            latent = tokens.transpose(1, 2)
        if self.global_stage is not None:
            latent = self.global_stage(latent)
        latent = self.refinement(latent)
        waveform_10hz = self.head(latent)
        waveform = fourier_interpolate(waveform_10hz, target_length=18000)
        return {"waveform": waveform, "waveform_10hz": waveform_10hz}

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        if isinstance(self.representation, MorphologyRepresentation):
            return {"loss_proto": self.representation.prototype_loss()}
        return {}


def build_crd_model(cfg: Any) -> nn.Module:
    if str(cfg.model.name) != "crd_v1":
        raise ValueError(f"CRD builder 要求 model.name=crd_v1，当前为 {cfg.model.name!r}")
    variant = str(cfg.model.variant).lower()
    initialization_seed = int(cfg.model.initialization_seed)
    if variant == "crd_001_b0_retrain":
        return LegacyB0Retrain(initialization_seed)
    if variant == "crd_002_t4_retrain":
        return LegacyT4Retrain(initialization_seed)
    if variant in CRD_VARIANTS[2:]:
        return CRDCoarseModel(variant, initialization_seed)
    raise ValueError(f"未知 CRD variant={variant!r}；可选 {list(CRD_VARIANTS)}")
