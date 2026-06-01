from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal


def _as_1d_float(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"信号必须是一维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("信号不能为空")
    if not np.isfinite(arr).all():
        raise ValueError("信号包含非有限值")
    return arr


def rms_envelope(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """计算固定窗口 RMS 包络，并保持输出长度不变。"""
    x = _as_1d_float(signal)
    window = int(window_samples)
    if window <= 0:
        raise ValueError(f"window_samples 必须为正数，当前={window_samples}")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    # reflect padding 能减少窗口边缘能量骤降，同时保持长度稳定。
    padded = np.pad(x * x, (window // 2, window - 1 - window // 2), mode="reflect")
    return np.sqrt(np.convolve(padded, kernel, mode="valid"))


def bandpass_filter(
    signal: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """使用 Butterworth 零相位带通滤波，输出 shape 与输入一致。"""
    x = _as_1d_float(signal)
    fs = float(fs)
    low_hz = float(low_hz)
    high_hz = float(high_hz)
    if fs <= 0:
        raise ValueError(f"fs 必须为正数，当前={fs}")
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError(f"带通频率非法: low={low_hz} high={high_hz} fs={fs}")
    sos = scipy_signal.butter(int(order), [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return scipy_signal.sosfiltfilt(sos, x)


def estimate_spectral_rate_bpm(
    signal: np.ndarray,
    *,
    fs: float,
    low_hz: float = 0.05,
    high_hz: float = 0.7,
) -> float:
    """用功率谱最大峰估计呼吸率，单位 bpm。"""
    distribution = _band_distribution(signal, fs=fs, low_hz=low_hz, high_hz=high_hz)
    if not np.isfinite(distribution["power"]).all():
        return float("nan")
    return float(distribution["freqs"][int(np.argmax(distribution["power"]))] * 60.0)


def estimate_peak_rate_bpm(
    signal: np.ndarray,
    *,
    fs: float,
    distance_sec: float | None = None,
    low_hz: float = 0.05,
    high_hz: float = 0.7,
) -> float:
    """用时域峰间距估计呼吸率，单位 bpm。"""
    x = _as_1d_float(signal)
    fs = float(fs)
    if distance_sec is None:
        distance_sec = 1.0 / float(high_hz)
    min_distance = max(1, int(round(float(distance_sec) * fs)))
    prominence = max(float(np.std(x)) * 0.1, np.finfo(np.float64).eps)
    peaks, _ = scipy_signal.find_peaks(x, distance=min_distance, prominence=prominence)
    if peaks.size < 2:
        return float("nan")
    intervals = np.diff(peaks) / fs
    if intervals.size == 0 or not np.isfinite(intervals).all():
        return float("nan")
    rate = 60.0 / float(np.median(intervals))
    low_bpm = float(low_hz) * 60.0
    high_bpm = float(high_hz) * 60.0
    if peaks.size == 0 or not low_bpm <= rate <= high_bpm:
        return float("nan")
    return rate


def spectrum_similarity(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float,
    low_hz: float = 0.05,
    high_hz: float = 0.7,
) -> float:
    """计算指定频带内归一化功率分布的 Bhattacharyya 相似度。"""
    pred_dist = _band_distribution(pred, fs=fs, low_hz=low_hz, high_hz=high_hz)["power"]
    target_dist = _band_distribution(target, fs=fs, low_hz=low_hz, high_hz=high_hz)["power"]
    if not np.isfinite(pred_dist).all() or not np.isfinite(target_dist).all():
        return float("nan")
    return float(np.sum(np.sqrt(pred_dist * target_dist)))


def _band_distribution(
    signal: np.ndarray,
    *,
    fs: float,
    low_hz: float = 0.05,
    high_hz: float = 0.7,
) -> dict[str, np.ndarray]:
    """返回目标频带内归一化功率分布。"""
    x = _as_1d_float(signal)
    fs = float(fs)
    if x.size < 2:
        raise ValueError("信号长度至少为 2")
    x = scipy_signal.detrend(x, type="constant")
    freqs, power = scipy_signal.welch(x, fs=fs, nperseg=min(x.size, 4096))
    mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    if not np.any(mask):
        raise ValueError(f"频带内没有谱点: low={low_hz} high={high_hz} fs={fs} n={x.size}")
    band_freqs = freqs[mask]
    band_power = np.maximum(power[mask], 0.0).astype(np.float64, copy=False)
    total = float(band_power.sum())
    if total <= 0:
        band_power = np.full_like(band_power, np.nan, dtype=np.float64)
    else:
        band_power = band_power / total
    return {"freqs": band_freqs, "power": band_power}
