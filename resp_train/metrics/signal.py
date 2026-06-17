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
    return np.sqrt(_moving_average_reflect(x * x, window))


def relative_envelope_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float,
    envelope_window_sec: float = 2.0,
    trend_window_sec: float = 20.0,
) -> dict[str, float]:
    """比较包络相对自身趋势的变化，用于诊断增强/下降是否被模型捕捉。"""
    fs = float(fs)
    envelope_window_sec = float(envelope_window_sec)
    trend_window_sec = float(trend_window_sec)
    if fs <= 0:
        raise ValueError(f"fs 必须为正数，当前={fs}")
    if envelope_window_sec <= 0:
        raise ValueError(f"envelope_window_sec 必须为正数，当前={envelope_window_sec}")
    if trend_window_sec <= 0:
        raise ValueError(f"trend_window_sec 必须为正数，当前={trend_window_sec}")

    pred_x = np.asarray(pred, dtype=np.float64)
    target_x = np.asarray(target, dtype=np.float64)
    if pred_x.size < 2 or target_x.size < 2:
        raise ValueError("相对包络指标的信号长度至少为 2")
    pred_x = _as_1d_float(pred)
    target_x = _as_1d_float(target)
    if pred_x.shape != target_x.shape:
        raise ValueError(f"pred 和 target 长度必须一致，当前 {pred_x.shape} != {target_x.shape}")

    env_window = max(1, int(round(fs * envelope_window_sec)))
    trend_window = max(env_window, int(round(fs * trend_window_sec)))
    pred_rel = _relative_envelope_trace(pred_x, env_window, trend_window)
    target_rel = _relative_envelope_trace(target_x, env_window, trend_window)
    if not np.isfinite(pred_rel).all() or not np.isfinite(target_rel).all():
        return {"relative_envelope_corr": float("nan"), "relative_envelope_mae": float("nan")}
    corr = _corrcoef_or_nan(pred_rel, target_rel)
    mae = float(np.mean(np.abs(pred_rel - target_rel)))
    return {"relative_envelope_corr": corr, "relative_envelope_mae": mae}


def _relative_envelope_trace(signal: np.ndarray, env_window: int, trend_window: int) -> np.ndarray:
    """返回 RMS 包络除以局部趋势后的相对包络轨迹。"""
    env = rms_envelope(signal, env_window)
    # 趋势窗口加宽后不易把几十秒级增强直接吸收到基线里。
    trend = _moving_average_reflect(env, trend_window * 2)
    floor = np.finfo(np.float64).eps
    # 使用对数比例，使增强与下降在相对尺度上对称，同时消除绝对幅度缩放影响。
    return np.log(np.maximum(env, floor) / np.maximum(trend, floor))


def _moving_average_reflect(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """使用反射填充计算等长移动平均。"""
    x = _as_1d_float(signal)
    window = int(window_samples)
    if window <= 0:
        raise ValueError(f"window_samples 必须为正数，当前={window_samples}")
    # 保持原有 np.pad(..., mode="reflect") + np.convolve(..., mode="valid")
    # 的边界语义，但用累积和避免大窗口卷积在全量评价时成为瓶颈。
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="reflect")
    cumsum = np.cumsum(np.insert(padded, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def _corrcoef_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    """计算相关系数；仅一侧无波动时返回 0，表示无法跟随另一侧变化。"""
    a = _as_1d_float(x)
    b = _as_1d_float(y)
    if a.shape != b.shape:
        raise ValueError(f"相关系数输入长度必须一致，当前 {a.shape} != {b.shape}")
    if a.size < 2:
        return float("nan")
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    eps = np.finfo(np.float64).eps
    if a_std <= eps and b_std <= eps:
        return float("nan")
    if a_std <= eps or b_std <= eps:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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


def band_limited_corr(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float,
    low_hz: float = 0.05,
    high_hz: float = 0.7,
    order: int = 4,
) -> float:
    """先限制到呼吸低频带，再计算预测与目标的相关系数。"""
    pred_filtered = bandpass_filter(pred, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    target_filtered = bandpass_filter(target, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    return _corrcoef_or_nan(pred_filtered, target_filtered)


def best_lag_correlation(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float,
    max_lag_sec: float = 1.0,
    low_hz: float = 0.05,
    high_hz: float = 0.7,
    order: int = 4,
) -> dict[str, float]:
    """在低频带内搜索最佳时延；正 lag 表示 pred 相对 target 滞后。"""
    fs = float(fs)
    max_lag_sec = float(max_lag_sec)
    if fs <= 0:
        raise ValueError(f"fs 必须为正数，当前={fs}")
    if max_lag_sec < 0:
        raise ValueError(f"max_lag_sec 必须非负，当前={max_lag_sec}")

    pred_filtered = bandpass_filter(pred, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    target_filtered = bandpass_filter(target, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    if pred_filtered.shape != target_filtered.shape:
        raise ValueError(f"pred 和 target 长度必须一致，当前 {pred_filtered.shape} != {target_filtered.shape}")

    n_samples = int(pred_filtered.size)
    period_samples = max(2, int(round(fs / float(low_hz))))
    min_overlap_samples = min(n_samples, period_samples)
    max_lag_samples = min(int(round(max_lag_sec * fs)), max(0, n_samples - min_overlap_samples))
    pred_prefix = _prefix_sums(pred_filtered)
    target_prefix = _prefix_sums(target_filtered)
    pred_sq_prefix = _prefix_sums(pred_filtered * pred_filtered)
    target_sq_prefix = _prefix_sums(target_filtered * target_filtered)
    cross_corr = scipy_signal.correlate(pred_filtered, target_filtered, mode="full", method="auto")
    best_corr = float("-inf")
    best_lag_samples: int | None = None
    for lag_samples in range(-max_lag_samples, max_lag_samples + 1):
        corr = _lagged_corr_from_prefix(
            lag_samples,
            n_samples=n_samples,
            pred_prefix=pred_prefix,
            target_prefix=target_prefix,
            pred_sq_prefix=pred_sq_prefix,
            target_sq_prefix=target_sq_prefix,
            cross_corr=cross_corr,
        )
        if not np.isfinite(corr):
            continue
        if best_lag_samples is None or _is_better_lag_candidate(corr, lag_samples, best_corr, best_lag_samples):
            best_corr = corr
            best_lag_samples = lag_samples

    if best_lag_samples is None:
        return {"best_lag_corr": float("nan"), "best_lag_sec": float("nan")}
    return {"best_lag_corr": float(best_corr), "best_lag_sec": float(best_lag_samples / fs)}


def _prefix_sums(x: np.ndarray) -> np.ndarray:
    """返回前缀和，首元素为 0，便于 O(1) 计算任意半开区间求和。"""
    return np.concatenate(([0.0], np.cumsum(np.asarray(x, dtype=np.float64))))


def _range_sum(prefix: np.ndarray, start: int, end: int) -> float:
    return float(prefix[int(end)] - prefix[int(start)])


def _lagged_corr_from_prefix(
    lag_samples: int,
    *,
    n_samples: int,
    pred_prefix: np.ndarray,
    target_prefix: np.ndarray,
    pred_sq_prefix: np.ndarray,
    target_sq_prefix: np.ndarray,
    cross_corr: np.ndarray,
) -> float:
    """按原始重叠切片定义计算指定 lag 的相关系数。"""
    if lag_samples > 0:
        pred_start, pred_end = lag_samples, n_samples
        target_start, target_end = 0, n_samples - lag_samples
    elif lag_samples < 0:
        lead_samples = -lag_samples
        pred_start, pred_end = 0, n_samples - lead_samples
        target_start, target_end = lead_samples, n_samples
    else:
        pred_start, pred_end = 0, n_samples
        target_start, target_end = 0, n_samples

    n_overlap = pred_end - pred_start
    if n_overlap < 2:
        return float("nan")
    sum_pred = _range_sum(pred_prefix, pred_start, pred_end)
    sum_target = _range_sum(target_prefix, target_start, target_end)
    sum_pred_sq = _range_sum(pred_sq_prefix, pred_start, pred_end)
    sum_target_sq = _range_sum(target_sq_prefix, target_start, target_end)
    sum_cross = float(cross_corr[n_samples - 1 + lag_samples])
    numerator = sum_cross - (sum_pred * sum_target / n_overlap)
    pred_var = sum_pred_sq - (sum_pred * sum_pred / n_overlap)
    target_var = sum_target_sq - (sum_target * sum_target / n_overlap)
    eps = np.finfo(np.float64).eps
    if pred_var <= eps and target_var <= eps:
        return float("nan")
    if pred_var <= eps or target_var <= eps:
        return 0.0
    return float(numerator / np.sqrt(pred_var * target_var))


def _is_better_lag_candidate(corr: float, lag_samples: int, best_corr: float, best_lag_samples: int) -> bool:
    """比较 lag 候选；相关近似相同则优先选择更靠近 0 的时延。"""
    if corr > best_corr and not np.isclose(corr, best_corr, rtol=1e-10, atol=1e-12):
        return True
    if not np.isclose(corr, best_corr, rtol=1e-10, atol=1e-12):
        return False
    abs_lag = abs(lag_samples)
    best_abs_lag = abs(best_lag_samples)
    if abs_lag != best_abs_lag:
        return abs_lag < best_abs_lag
    return lag_samples < best_lag_samples


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
