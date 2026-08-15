from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from resp_train.crd.blocks import (
    BidirectionalMamba2Block,
    CoarseWaveformHead,
    DecoderResidual,
    LocalTCNBlock,
    ResidualDWBlock,
)
from resp_train.crd.capacity import CapacityResidualStack, select_capacity_match
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


CRD_TF_VARIANTS = (
    "crd_tf101_m",
    "crd_tf102_w",
    "crd_tf103_l",
    "crd_tf104_s",
    "crd_tf201_mw",
    "crd_tf202_ml",
    "crd_tf203_ms",
    "crd_tf204_wl",
    "crd_tf205_ws",
    "crd_tf206_ls",
    "crd_tf301_mls",
    "crd_tf302_wls",
    "crd_tf_ctrl1",
    "crd_tf_ctrl2",
    "crd_tf_ctrl3",
)

CRD_TF_P6_VARIANTS = (
    "crd_tf401_mws_add",
    "crd_tf402_mws_gate",
    "crd_tf403_ctrl_gate",
)

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
    "crd_205_base_em_static",
    "crd_206_base_am_static",
    "crd_207_base_cap_em",
    "crd_208_base_cap_am",
    "crd_c101_b0_local_tcn",
    "crd_c201_decoder_10hz_cap",
    "crd_c202_decoder_100hz",
    *CRD_TF_VARIANTS,
    *CRD_TF_P6_VARIANTS,
)

CRD_S2A_VARIANTS = {
    "crd_202_base_legacy_energy",
    "crd_203_base_analytic_am",
    "crd_204_base_morphology",
}

CRD_S2BR_VARIANTS = {
    "crd_205_base_em_static",
    "crd_206_base_am_static",
    "crd_207_base_cap_em",
    "crd_208_base_cap_am",
}

CRD_C1_VARIANTS = {"crd_c101_b0_local_tcn"}
CRD_C2_VARIANTS = {"crd_c201_decoder_10hz_cap", "crd_c202_decoder_100hz"}
CRD_CONTROL_VARIANTS = CRD_C1_VARIANTS | CRD_C2_VARIANTS
LOCAL_TCN_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


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
            *CRD_S2BR_VARIANTS,
            *CRD_C2_VARIANTS,
        }
        uses_global = variant in {
            "crd_104_direct_hier_mamba",
            "crd_106_b0_hier_mamba",
        }
        uses_local_tcn = variant in CRD_C1_VARIANTS
        uses_decoder_residual = variant in CRD_C2_VARIANTS

        with module_seed(initialization_seed, "direct_frontend" if uses_direct else "patch_frontend"):
            self.frontend: nn.Module = DirectAnalyticFrontend() if uses_direct else PatchTokenFrontend()
        if variant == "crd_202_base_legacy_energy":
            self.representation: nn.Module | None = LegacyEnergyRepresentation(initialization_seed)
        elif variant == "crd_203_base_analytic_am":
            self.representation = AnalyticAMRepresentation(initialization_seed)
        elif variant == "crd_204_base_morphology":
            self.representation = MorphologyRepresentation(initialization_seed)
        elif variant == "crd_205_base_em_static":
            self.representation = LegacyEnergyRepresentation(initialization_seed)
        elif variant == "crd_206_base_am_static":
            self.representation = AnalyticAMRepresentation(initialization_seed)
        else:
            self.representation = None
        if variant in {"crd_205_base_em_static", "crd_206_base_am_static"}:
            self.additional_representation: nn.Module | None = MorphologyRepresentation(initialization_seed)
        else:
            self.additional_representation = None
        self.checkpoint_primary_representation = variant == "crd_206_base_am_static"
        if variant in {"crd_207_base_cap_em", "crd_208_base_cap_am"}:
            target_variant = (
                "crd_205_base_em_static" if variant == "crd_207_base_cap_em" else "crd_206_base_am_static"
            )
            self.capacity_match = select_capacity_match(target_variant)
            with module_seed(initialization_seed, f"capacity_control_{target_variant}"):
                self.capacity_control: CapacityResidualStack | None = CapacityResidualStack(
                    block_count=self.capacity_match.block_count,
                    hidden_channels=self.capacity_match.hidden_channels,
                )
        else:
            self.capacity_match = None
            self.capacity_control = None
        if uses_local:
            with module_seed(initialization_seed, "local_trunk"):
                self.local_blocks = nn.ModuleList([BidirectionalMamba2Block(96) for _ in range(6)])
        else:
            self.local_blocks = nn.ModuleList()
        if uses_local_tcn:
            with module_seed(initialization_seed, "tcn_trunk"):
                self.local_tcn_blocks = nn.ModuleList(
                    [LocalTCNBlock(dilation=dilation) for dilation in LOCAL_TCN_DILATIONS]
                )
        else:
            self.local_tcn_blocks = nn.ModuleList()
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
        if uses_decoder_residual:
            with module_seed(initialization_seed, "decoder_residual"):
                self.decoder_residual: DecoderResidual | None = DecoderResidual()
        else:
            self.decoder_residual = None

    def forward(self, x: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        _validate_input(x)
        latent = self.encode_local(x)
        return self.decode_local(latent)

    def encode_local(self, x: torch.Tensor) -> torch.Tensor:
        """执行 frontend 与 local/global trunk，返回 refinement 前的 10-Hz latent。"""

        _validate_input(x)
        latent = self.frontend(x)
        if latent.shape[1:] != (96, 1800):
            raise RuntimeError(f"CRD frontend 输出契约错误: {tuple(latent.shape)}")
        if self.representation is not None:
            if self.checkpoint_primary_representation and self.training and torch.is_grad_enabled():
                # Analytic-AM 含 dropout，重算时必须恢复 RNG state 才保持同一 stochastic forward。
                representation = checkpoint(
                    self.representation,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                representation = self.representation(x)
            if representation.shape != latent.shape:
                raise RuntimeError(
                    f"CRD representation 输出契约错误: {tuple(representation.shape)} != {tuple(latent.shape)}"
                )
            latent = latent + representation
        if self.additional_representation is not None:
            additional = self.additional_representation(x)
            if additional.shape != latent.shape:
                raise RuntimeError(
                    f"CRD additional representation 输出契约错误: {tuple(additional.shape)} != {tuple(latent.shape)}"
                )
            latent = latent + additional
        if self.capacity_control is not None:
            latent = self.capacity_control(latent)
        if self.local_blocks:
            tokens = latent.transpose(1, 2)
            for block in self.local_blocks:
                tokens = block(tokens)
            latent = tokens.transpose(1, 2)
        for block in self.local_tcn_blocks:
            latent = block(latent)
        if self.global_stage is not None:
            latent = self.global_stage(latent)
        return latent

    def decode_local(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """执行冻结 refinement 与 decoder；供 CRD-TF 在二者之间插入 FiLM。"""

        if latent.ndim != 3 or latent.shape[1:] != (96, 1800):
            raise ValueError(f"CRD decode_local 期望 (B,96,1800)，实际 {tuple(latent.shape)}")
        if not torch.isfinite(latent).all():
            raise FloatingPointError("CRD decode_local latent 包含 NaN/Inf")
        latent = self.refinement(latent)
        if self.variant not in CRD_C2_VARIANTS:
            waveform_10hz = self.head(latent)
            waveform = fourier_interpolate(waveform_10hz, target_length=18000)
            return {"waveform": waveform, "waveform_10hz": waveform_10hz}
        head_features = self.head.features(latent)
        waveform_10hz_base = self.head.output(head_features)
        if self.variant == "crd_c201_decoder_10hz_cap":
            if self.decoder_residual is None:
                raise RuntimeError("C201 缺少 decoder residual")
            waveform_10hz = waveform_10hz_base + self.decoder_residual(head_features)
            waveform = fourier_interpolate(waveform_10hz, target_length=18000)
            return {"waveform": waveform, "waveform_10hz": waveform_10hz}
        if self.variant == "crd_c202_decoder_100hz":
            if self.decoder_residual is None:
                raise RuntimeError("C202 缺少 decoder residual")
            features_100hz = fourier_interpolate(head_features, target_length=18000)
            residual_100hz = self.decoder_residual(features_100hz)
            waveform = fourier_interpolate(waveform_10hz_base, target_length=18000) + residual_100hz
            return {
                "waveform": waveform,
                "waveform_10hz": waveform_10hz_base,
                "decoder_residual_100hz": residual_100hz,
            }
        raise RuntimeError(f"未实现的 C2 decoder variant: {self.variant}")

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        if isinstance(self.representation, MorphologyRepresentation):
            return {"loss_proto": self.representation.prototype_loss()}
        if isinstance(self.additional_representation, MorphologyRepresentation):
            return {"loss_proto": self.additional_representation.prototype_loss()}
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
    if variant in (*CRD_TF_VARIANTS, *CRD_TF_P6_VARIANTS):
        from resp_train.crd.tf_v1_model import CRDTfV1Model

        return CRDTfV1Model(variant, initialization_seed)
    if variant in CRD_VARIANTS[2:]:
        return CRDCoarseModel(variant, initialization_seed)
    raise ValueError(f"未知 CRD variant={variant!r}；可选 {list(CRD_VARIANTS)}")
