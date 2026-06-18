from __future__ import annotations

from typing import Any, Callable

from torch import nn

from resp_train.models.timeseries import DLinearWaveform, FIRFrontendPatchMixer1D, PatchMixer1D, PeriodicUNet1DTiny
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
    "dlinear_waveform": lambda cfg: DLinearWaveform(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
        moving_avg=int(cfg.model.get("moving_avg", 101)),
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
