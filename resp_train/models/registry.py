from __future__ import annotations

from typing import Any, Callable

from torch import nn

from resp_train.models.lowfreq import (
    BasisDecoder1D,
    DownsampledSSM1D,
    FrequencyBottleneck1D,
    MultiScaleDecompMixer1D,
    TimesNetLite1D,
)
from resp_train.models.timeseries import (
    DLinearWaveform,
    FIRFrontendPatchMixer1D,
    MultiScalePatchHannBandlimited1D,
    PatchHannBandlimitedOutput1D,
    PatchHannBasisResidualDecoder1D,
    PatchHannControlPointDecoder1D,
    PatchMixer1D,
    PeriodAwarePatchHannBandlimited1D,
    PolyphasePatchHannBandlimited1D,
    PeriodicUNet1DTiny,
)
from resp_train.models.unet1d import UNet1DTiny, UNet1DTinyNoSkip1, UNet1DTinyNoSkipAll


ModelFactory = Callable[[Any], nn.Module]


_REGISTRY: dict[str, ModelFactory] = {
    "unet1d_tiny": lambda cfg: UNet1DTiny(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
    ),
    "unet1d_tiny_noskip1": lambda cfg: UNet1DTinyNoSkip1(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
    ),
    "unet1d_tiny_noskip_all": lambda cfg: UNet1DTinyNoSkipAll(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
    ),
    "periodic_unet1d_tiny": lambda cfg: PeriodicUNet1DTiny(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        lowpass_kernel=int(cfg.model.get("lowpass_kernel", cfg.model.get("moving_avg", 101))),
        output_smoothing_kernel=int(cfg.model.get("output_smoothing_kernel", 1)),
    ),
    "patch_mixer1d": lambda cfg: PatchMixer1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        patch_len=int(cfg.model.get("patch_len", 256)),
        patch_stride=int(cfg.model.get("patch_stride", 128)),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        overlap_window=str(cfg.model.get("overlap_window", "uniform")),
        output_smoothing_kernel=int(cfg.model.get("output_smoothing_kernel", 1)),
    ),
    "patch_mixer1d_fir_frontend": lambda cfg: FIRFrontendPatchMixer1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        patch_len=int(cfg.model.get("patch_len", 256)),
        patch_stride=int(cfg.model.get("patch_stride", 128)),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        overlap_window=str(cfg.model.get("overlap_window", "hann")),
        output_smoothing_kernel=int(cfg.model.get("output_smoothing_kernel", 1)),
        fir_kernel_size=int(cfg.model.get("fir_kernel_size", 401)),
        fir_low_hz=float(cfg.model.get("fir_low_hz", 0.05)),
        fir_high_hz=float(cfg.model.get("fir_high_hz", 0.7)),
        fir_sample_rate=float(cfg.model.get("fir_sample_rate", 100.0)),
        fir_trainable=bool(cfg.model.get("fir_trainable", True)),
    ),
    "patch_hann_control_point_decoder1d": lambda cfg: PatchHannControlPointDecoder1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        patch_len=int(cfg.model.get("patch_len", 256)),
        patch_stride=int(cfg.model.get("patch_stride", 128)),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        control_points=int(cfg.model.get("control_points", 360)),
    ),
    "patch_hann_basis_residual_decoder1d": lambda cfg: PatchHannBasisResidualDecoder1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        patch_len=int(cfg.model.get("patch_len", 256)),
        patch_stride=int(cfg.model.get("patch_stride", 128)),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        basis_count=int(cfg.model.get("basis_count", 96)),
        residual_points=int(cfg.model.get("residual_points", 180)),
        residual_scale=float(cfg.model.get("residual_scale", 0.05)),
        duration_samples=int(cfg.window.get("duration_samples", 18000)),
    ),
    "patch_hann_bandlimited_output1d": lambda cfg: PatchHannBandlimitedOutput1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        patch_len=int(cfg.model.get("patch_len", 256)),
        patch_stride=int(cfg.model.get("patch_stride", 128)),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        max_freq_hz=float(cfg.model.get("max_freq_hz", 0.7)),
        sample_rate=float(cfg.window.get("target_fs", 100)),
    ),
    "multiscale_patch_hann_bandlimited1d": lambda cfg: MultiScalePatchHannBandlimited1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        patch_lengths=list(cfg.model.get("patch_lengths", [256, 512, 1024, 2048])),
        patch_stride_ratio=float(cfg.model.get("patch_stride_ratio", 0.5)),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        max_freq_hz=float(cfg.model.get("max_freq_hz", 0.7)),
        sample_rate=float(cfg.window.get("target_fs", 100)),
    ),
    "period_aware_patch_hann_bandlimited1d": lambda cfg: PeriodAwarePatchHannBandlimited1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        period_secs=list(cfg.model.get("period_secs", [2.5, 4.0, 6.0, 10.0])),
        patch_stride_ratio=float(cfg.model.get("patch_stride_ratio", 0.5)),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        max_freq_hz=float(cfg.model.get("max_freq_hz", 0.7)),
        sample_rate=float(cfg.window.get("target_fs", 100)),
    ),
    "polyphase_patch_hann_bandlimited1d": lambda cfg: PolyphasePatchHannBandlimited1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        offsets=list(cfg.model.get("offsets", [1, 2, 4])),
        patch_len=int(cfg.model.get("patch_len", 256)),
        patch_stride=int(cfg.model.get("patch_stride", 128)),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        max_freq_hz=float(cfg.model.get("max_freq_hz", 0.7)),
        sample_rate=float(cfg.window.get("target_fs", 100)),
    ),
    "dlinear_waveform": lambda cfg: DLinearWaveform(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        moving_avg=int(cfg.model.get("moving_avg", 101)),
    ),
    "basis_decoder1d": lambda cfg: BasisDecoder1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        basis_count=int(cfg.model.get("basis_count", 96)),
        encoder_stride=int(cfg.model.get("encoder_stride", 20)),
        duration_samples=int(cfg.window.get("duration_samples", 18000)),
    ),
    "multiscale_decomp_mixer1d": lambda cfg: MultiScaleDecompMixer1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        downsample_factors=list(cfg.model.get("downsample_factors", [1, 4, 16])),
        mixer_layers=int(cfg.model.get("mixer_layers", 2)),
    ),
    "timesnet_lite1d": lambda cfg: TimesNetLite1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        period_top_k=int(cfg.model.get("period_top_k", 3)),
        period_min_sec=float(cfg.model.get("period_min_sec", 2.0)),
        period_max_sec=float(cfg.model.get("period_max_sec", 20.0)),
        sample_rate=float(cfg.window.get("target_fs", 100)),
        lowpass_kernel=int(cfg.model.get("lowpass_kernel", 401)),
    ),
    "frequency_bottleneck1d": lambda cfg: FrequencyBottleneck1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        freq_bins=int(cfg.model.get("freq_bins", 128)),
        max_freq_hz=float(cfg.model.get("max_freq_hz", 0.7)),
        sample_rate=float(cfg.window.get("target_fs", 100)),
        duration_samples=int(cfg.window.get("duration_samples", 18000)),
    ),
    "downsampled_ssm1d": lambda cfg: DownsampledSSM1D(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        latent_stride=int(cfg.model.get("latent_stride", 20)),
        state_layers=int(cfg.model.get("state_layers", 2)),
    ),
}


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(cfg: Any) -> nn.Module:
    name = cfg.model.name
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_models())
        raise KeyError(f"未知模型: {name}。可用模型: {available}") from exc
    return factory(cfg)
