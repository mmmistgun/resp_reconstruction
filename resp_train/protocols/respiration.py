from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def _torch_work_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def fft_band_project_torch(
    signal: torch.Tensor,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> torch.Tensor:
    """执行整窗、零相位、硬矩形 FFT 呼吸频带投影。"""

    if signal.ndim < 1:
        raise ValueError("signal 至少需要一个时间维")
    if not torch.isfinite(signal).all():
        raise ValueError("呼吸频带投影输入包含 NaN/Inf")
    n = int(signal.shape[-1])
    if n <= 0:
        raise ValueError("signal 时间维不能为空")
    if not (0.0 <= float(low_hz) <= float(high_hz) <= float(fs) / 2.0):
        raise ValueError("频带必须满足 0 <= low_hz <= high_hz <= Nyquist")

    work = signal.to(dtype=_torch_work_dtype(signal.dtype))
    work = work - work.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(work, n=n, dim=-1, norm="backward")
    frequencies = torch.fft.rfftfreq(n, d=1.0 / float(fs), device=work.device)
    mask = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    return torch.fft.irfft(spectrum * mask.to(spectrum.dtype), n=n, dim=-1, norm="backward")


def canonicalize_torch(
    signal: torch.Tensor,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    scale_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回规范化前频带信号 ``b`` 与正式输出 ``x=S(B(signal))``。"""

    band = fft_band_project_torch(signal, fs=fs, low_hz=low_hz, high_hz=high_hz)
    centered = band - band.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + float(scale_eps))
    return band, centered / scale


def fft_band_project_numpy(
    signal: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """NumPy 版本的整窗硬 FFT 投影，评价统一使用 float64。"""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("signal 至少需要一个时间维")
    if not np.isfinite(values).all():
        raise ValueError("呼吸频带投影输入包含 NaN/Inf")
    n = int(values.shape[-1])
    if n <= 0:
        raise ValueError("signal 时间维不能为空")
    if not (0.0 <= float(low_hz) <= float(high_hz) <= float(fs) / 2.0):
        raise ValueError("频带必须满足 0 <= low_hz <= high_hz <= Nyquist")

    centered = values - np.mean(values, axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=n, axis=-1, norm="backward")
    frequencies = np.fft.rfftfreq(n, d=1.0 / float(fs))
    mask = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    return np.fft.irfft(spectrum * mask, n=n, axis=-1, norm="backward")


def canonicalize_numpy(
    signal: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    scale_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 NumPy 版 ``b`` 与 ``x=S(B(signal))``。"""

    band = fft_band_project_numpy(signal, fs=fs, low_hz=low_hz, high_hz=high_hz)
    centered = band - np.mean(band, axis=-1, keepdims=True)
    scale = np.sqrt(np.mean(np.square(centered), axis=-1, keepdims=True) + float(scale_eps))
    return band, centered / scale


def lag_priority(max_lag_samples: int) -> list[int]:
    """返回符合“最小绝对 lag，再取较小 lag”并列规则的搜索顺序。"""

    maximum = int(max_lag_samples)
    if maximum < 0:
        raise ValueError("max_lag_samples 不能为负")
    result = [0]
    for magnitude in range(1, maximum + 1):
        result.extend((-magnitude, magnitude))
    return result


def centered_energy_numpy(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64)
    centered = values - float(np.mean(values))
    return float(np.mean(np.square(centered)))


def centered_energy_torch(signal: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    centered = signal - signal.mean(dim=dim, keepdim=True)
    return centered.square().mean(dim=dim)


def symmetric_hann_numpy(length: int) -> np.ndarray:
    if int(length) <= 1:
        raise ValueError("Hann window 长度必须大于 1")
    return np.hanning(int(length)).astype(np.float64, copy=False)


def symmetric_hann_torch(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if int(length) <= 1:
        raise ValueError("Hann window 长度必须大于 1")
    return torch.hann_window(int(length), periodic=False, device=device, dtype=dtype)


def as_batch_waveform_numpy(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal)
    if values.ndim == 1:
        return values[None, :]
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[1] == 1:
        return values[:, 0, :]
    raise ValueError(f"波形必须为 [N]、[B,N] 或 [B,1,N]，当前 shape={values.shape}")


def as_batch_waveform_torch(signal: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(signal, dict):
        if "waveform" not in signal:
            raise KeyError("模型输出 dict 必须包含 waveform")
        signal = signal["waveform"]
    if not torch.is_tensor(signal):
        raise TypeError("波形输出必须是 Tensor")
    if signal.ndim == 2:
        return signal
    if signal.ndim == 3 and signal.shape[1] == 1:
        return signal[:, 0, :]
    raise ValueError(f"波形必须为 [B,N] 或 [B,1,N]，当前 shape={tuple(signal.shape)}")


def validate_window_starts(length: int, window: int, step: int) -> Sequence[int]:
    if int(window) <= 0 or int(step) <= 0 or int(length) < int(window):
        raise ValueError("window/step/length 组合无效")
    return range(0, int(length) - int(window) + 1, int(step))
