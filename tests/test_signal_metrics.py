import numpy as np
import pytest

from resp_train.metrics.signal import (
    _moving_average_reflect,
    band_limited_corr,
    bandpass_filter,
    best_lag_correlation,
    best_lag_correlation_from_filtered,
    estimate_bandpassed_peak_rate_bpm,
    estimate_peak_rate_bpm,
    estimate_robust_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    lag_aligned_overlap,
    lag_correlation_trace_from_filtered,
    local_rr_metrics,
    local_rr_rate_trace,
    relative_envelope_metrics,
    rms_envelope,
    spectrum_similarity,
    zero_crossing_counts,
)


def _sine(freq_hz: float, fs: float, duration_sec: float) -> np.ndarray:
    t = np.arange(int(fs * duration_sec), dtype=np.float64) / fs
    return np.sin(2 * np.pi * freq_hz * t)


def _modulated_breath_signal(fs: float, duration_sec: float) -> np.ndarray:
    t = np.arange(int(fs * duration_sec), dtype=np.float64) / fs
    envelope = 1.0 + 0.25 * np.sin(2 * np.pi * 0.03 * t)
    return envelope * np.sin(2 * np.pi * 0.23 * t + 0.15)


def _double_peak_breath_signal(fs: float, duration_sec: float, period_sec: float = 6.0) -> np.ndarray:
    """构造基频明确、但每个周期里有局部次峰的信号。"""
    t = np.arange(int(fs * duration_sec), dtype=np.float64) / fs
    x = 0.2 * np.sin(2 * np.pi * (1.0 / period_sec) * t)
    for peak_time in np.arange(1.0, duration_sec - 1.0, period_sec):
        x += np.exp(-0.5 * ((t - peak_time) / 0.30) ** 2)
        x += 0.3 * np.exp(-0.5 * ((t - (peak_time + 3.05)) / 0.22) ** 2)
    return x - np.mean(x)


def _delay_with_zero_fill(signal: np.ndarray, samples: int) -> np.ndarray:
    delayed = np.zeros_like(signal)
    delayed[samples:] = signal[:-samples]
    return delayed


def _advance_with_zero_fill(signal: np.ndarray, samples: int) -> np.ndarray:
    advanced = np.zeros_like(signal)
    advanced[:-samples] = signal[samples:]
    return advanced


def test_moving_average_reflect_matches_convolve_reference():
    x = np.linspace(-1.0, 1.0, 257, dtype=np.float64)
    window = 41
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="reflect")
    expected = np.convolve(padded, kernel, mode="valid")

    actual = _moving_average_reflect(x, window)

    assert np.allclose(actual, expected)


def test_rms_envelope_保持长度():
    x = np.linspace(-1.0, 1.0, 101)

    env = rms_envelope(x, window_samples=11)

    assert env.shape == x.shape
    assert np.isfinite(env).all()


def test_relative_envelope_metrics_忽略绝对幅度缩放():
    fs = 100.0
    t = np.arange(0, 120, 1 / fs)
    carrier = np.sin(2 * np.pi * 0.25 * t)
    mod = 1.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t)
    target = mod * carrier
    pred = 5.0 * target

    metrics = relative_envelope_metrics(pred, target, fs=fs, envelope_window_sec=2.0, trend_window_sec=20.0)

    assert metrics["relative_envelope_corr"] > 0.99
    assert metrics["relative_envelope_mae"] < 0.01


def test_relative_envelope_metrics_识别相对增强缺失():
    fs = 100.0
    t = np.arange(0, 120, 1 / fs)
    carrier = np.sin(2 * np.pi * 0.25 * t)
    target_mod = np.ones_like(t)
    target_mod[(t >= 45) & (t <= 75)] = 1.8
    target = target_mod * carrier
    pred = carrier.copy()

    metrics = relative_envelope_metrics(pred, target, fs=fs, envelope_window_sec=2.0, trend_window_sec=20.0)

    assert metrics["relative_envelope_corr"] < 0.8
    assert metrics["relative_envelope_mae"] > 0.05


def test_relative_envelope_metrics_拒绝长度小于2的信号():
    with pytest.raises(ValueError, match="长度至少为 2"):
        relative_envelope_metrics(np.array([1.0]), np.array([1.0]), fs=100.0)


def test_relative_envelope_metrics_拒绝非正采样率():
    with pytest.raises(ValueError, match="fs 必须为正数"):
        relative_envelope_metrics(np.ones(10), np.ones(10), fs=0.0)


def test_relative_envelope_metrics_拒绝非正包络窗口():
    with pytest.raises(ValueError, match="envelope_window_sec 必须为正数"):
        relative_envelope_metrics(np.ones(10), np.ones(10), fs=100.0, envelope_window_sec=0.0)


def test_relative_envelope_metrics_拒绝非正趋势窗口():
    with pytest.raises(ValueError, match="trend_window_sec 必须为正数"):
        relative_envelope_metrics(np.ones(10), np.ones(10), fs=100.0, trend_window_sec=0.0)


def test_lag_aligned_overlap_uses_same_positive_lag_convention_as_best_lag():
    pred = np.arange(10, dtype=np.float64) + 100.0
    target = np.arange(10, dtype=np.float64)

    pred_overlap, target_overlap = lag_aligned_overlap(pred, target, lag_samples=3)

    np.testing.assert_array_equal(pred_overlap, pred[3:])
    np.testing.assert_array_equal(target_overlap, target[:-3])


def test_lag_aligned_overlap_uses_same_negative_lag_convention_as_best_lag():
    pred = np.arange(10, dtype=np.float64) + 100.0
    target = np.arange(10, dtype=np.float64)

    pred_overlap, target_overlap = lag_aligned_overlap(pred, target, lag_samples=-2)

    np.testing.assert_array_equal(pred_overlap, pred[:-2])
    np.testing.assert_array_equal(target_overlap, target[2:])


def test_zero_crossing_counts_reports_up_down_and_cycle_counts():
    signal = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])

    counts = zero_crossing_counts(signal)

    assert counts == {"up": 2, "down": 2, "cycle": 2}


def test_local_rr_metrics_tracks_same_local_rate_curve():
    fs = 100.0
    first = _sine(0.20, fs, 60.0)
    second = _sine(0.32, fs, 60.0)
    target = np.concatenate([first, second])
    pred = target.copy()

    metrics = local_rr_metrics(
        pred,
        target,
        fs=fs,
        window_sec=20.0,
        step_sec=5.0,
        low_hz=0.05,
        high_hz=0.7,
    )

    assert metrics["local_rr_mae"] < 0.1
    assert metrics["local_rr_valid_frac"] == 1.0
    assert metrics["local_rr_corr"] > 0.99


def test_local_rr_uses_v2_peak_spacing_to_reject_double_peak_alias():
    fs = 100.0
    target = _double_peak_breath_signal(fs, 180.0, period_sec=6.0)

    local_rates = local_rr_rate_trace(target, fs=fs, low_hz=0.05, high_hz=0.7)

    assert local_rates.size == 15
    assert np.nanmedian(local_rates) == pytest.approx(10.0, abs=0.2)


def test_local_rr_metrics_tracks_identical_double_peak_curves():
    fs = 100.0
    target = _double_peak_breath_signal(fs, 180.0, period_sec=6.0)
    pred = target.copy()

    metrics = local_rr_metrics(pred, target, fs=fs, low_hz=0.05, high_hz=0.7)

    assert metrics["local_rr_mae"] < 0.01
    assert metrics["local_rr_valid_frac"] == 1.0
    assert np.isnan(metrics["local_rr_corr"])


def test_spectral_rate_识别正弦主频():
    fs = 100.0
    x = _sine(0.25, fs, 60.0)

    rate = estimate_spectral_rate_bpm(x, fs=fs, low_hz=0.05, high_hz=0.7)

    assert rate == np.float64(rate)
    assert abs(rate - 15.0) < 0.5


def test_peak_rate_识别正弦峰值():
    fs = 100.0
    x = _sine(0.25, fs, 60.0)

    rate = estimate_peak_rate_bpm(x, fs=fs, distance_sec=2.0)

    assert abs(rate - 15.0) < 0.5


def test_peak_rate_使用峰间距而不是峰数量():
    fs = 10.0
    x = np.zeros(1000, dtype=np.float64)
    x[[10, 210, 410]] = 1.0

    rate = estimate_peak_rate_bpm(x, fs=fs, distance_sec=10.0)

    assert abs(rate - 3.0) < 0.01


def test_robust_peak_rate_忽略弱局部伪峰():
    fs = 100.0
    duration_sec = 90.0
    t = np.arange(int(fs * duration_sec), dtype=np.float64) / fs
    true_period_sec = 5.5
    x = np.zeros_like(t)
    true_peak_times = np.arange(1.0, duration_sec - 1.0, true_period_sec)
    for peak_time in true_peak_times:
        x += np.exp(-0.5 * ((t - peak_time) / 0.38) ** 2)
    x -= 0.35 * np.sin(2 * np.pi * (1.0 / true_period_sec) * t + np.pi / 2)
    for peak_time in true_peak_times[:-1]:
        x += 0.20 * np.exp(-0.5 * ((t - (peak_time + 2.7)) / 0.22) ** 2)

    naive_rate = estimate_peak_rate_bpm(x, fs=fs, distance_sec=2.0, low_hz=0.05, high_hz=0.7)
    robust_rate = estimate_robust_peak_rate_bpm(x, fs=fs, low_hz=0.05, high_hz=0.7)

    assert abs(naive_rate - 60.0 / true_period_sec) > 0.2
    assert robust_rate == pytest.approx(60.0 / true_period_sec, abs=0.2)


def test_bandpassed_peak_rate_忽略呼吸频带外尖峰():
    fs = 100.0
    t = np.arange(0, 80, 1 / fs)
    target = np.sin(2 * np.pi * 0.2 * t)
    noisy = target + 2.0 * np.sin(2 * np.pi * 2.0 * t)

    rate = estimate_bandpassed_peak_rate_bpm(noisy, fs=fs, low_hz=0.05, high_hz=0.7, order=4)

    assert abs(rate - 12.0) < 0.5


def test_flat_signal_returns_nan_for_rr_and_similarity():
    fs = 100.0
    x = np.zeros(6000, dtype=np.float64)

    assert np.isnan(estimate_spectral_rate_bpm(x, fs=fs, low_hz=0.05, high_hz=0.7))
    assert np.isnan(estimate_peak_rate_bpm(x, fs=fs, distance_sec=2.0))
    assert np.isnan(spectrum_similarity(x, x, fs=fs, low_hz=0.05, high_hz=0.7))


def test_identical_signal_spectrum_similarity_gt_099():
    fs = 100.0
    x = _sine(0.33, fs, 60.0)

    similarity = spectrum_similarity(x, x.copy(), fs=fs, low_hz=0.05, high_hz=0.7)

    assert similarity > 0.99


def test_bandpass_filter_shape_不变():
    fs = 100.0
    x = _sine(0.25, fs, 20.0)

    filtered = bandpass_filter(x, fs=fs, low_hz=0.05, high_hz=0.7, order=4)

    assert filtered.shape == x.shape
    assert np.isfinite(filtered).all()


def test_band_limited_corr_ignores_high_frequency_noise():
    fs = 100.0
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t)
    pred = target + 0.5 * np.sin(2 * np.pi * 8.0 * t)

    corr = band_limited_corr(pred, target, fs=fs, low_hz=0.05, high_hz=0.7, order=4)

    assert corr > 0.99


def test_best_lag_correlation_recovers_positive_delay_seconds():
    fs = 100.0
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t)
    pred = np.roll(target, int(round(0.5 * fs)))

    metrics = best_lag_correlation(pred, target, fs=fs, max_lag_sec=1.0, low_hz=0.05, high_hz=0.7, order=4)

    assert metrics["best_lag_corr"] > 0.99
    assert abs(metrics["best_lag_sec"] - 0.5) < 1 / fs


def test_best_lag_correlation_recovers_non_circular_positive_delay():
    fs = 100.0
    target = _modulated_breath_signal(fs, 80.0)
    pred = _delay_with_zero_fill(target, int(round(0.5 * fs)))

    metrics = best_lag_correlation(pred, target, fs=fs, max_lag_sec=1.0, low_hz=0.05, high_hz=0.7, order=4)

    assert metrics["best_lag_corr"] > 0.99
    assert abs(metrics["best_lag_sec"] - 0.5) < 1 / fs


def test_best_lag_correlation_recovers_non_circular_negative_delay():
    fs = 100.0
    target = _modulated_breath_signal(fs, 80.0)
    pred = _advance_with_zero_fill(target, int(round(0.4 * fs)))

    metrics = best_lag_correlation(pred, target, fs=fs, max_lag_sec=1.0, low_hz=0.05, high_hz=0.7, order=4)

    assert metrics["best_lag_corr"] > 0.99
    assert abs(metrics["best_lag_sec"] + 0.4) < 1 / fs


def test_best_lag_correlation_prefers_zero_for_identical_signal_with_large_max_lag():
    fs = 100.0
    target = _modulated_breath_signal(fs, 60.0)

    metrics = best_lag_correlation(target.copy(), target, fs=fs, max_lag_sec=120.0, low_hz=0.05, high_hz=0.7, order=4)

    assert metrics["best_lag_corr"] > 0.99
    assert abs(metrics["best_lag_sec"]) < 1e-6


def test_best_lag_correlation_limits_overlap_for_periodic_signal_with_large_max_lag():
    fs = 100.0
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t)

    metrics = best_lag_correlation(target.copy(), target, fs=fs, max_lag_sec=120.0, low_hz=0.05, high_hz=0.7, order=4)

    assert metrics["best_lag_corr"] > 0.99
    assert abs(metrics["best_lag_sec"]) < 1e-6


def test_lag_correlation_trace_preserves_existing_best_result_and_all_integer_lags():
    rng = np.random.default_rng(20260712)
    target = rng.normal(size=3000)
    pred = np.zeros_like(target)
    pred[37:] = target[:-37]
    pred += rng.normal(scale=0.01, size=target.size)

    trace = lag_correlation_trace_from_filtered(
        pred, target, fs=100.0, max_lag_sec=4.0, low_hz=0.05
    )
    best = best_lag_correlation_from_filtered(
        pred, target, fs=100.0, max_lag_sec=4.0, low_hz=0.05
    )

    np.testing.assert_array_equal(trace["lag_samples"], np.arange(-400, 401))
    np.testing.assert_allclose(trace["lag_sec"], np.arange(-400, 401) / 100.0, rtol=0.0, atol=0.0)
    assert trace["correlation"].shape == (801,)
    assert best["best_lag_corr"] == 0.9999497987100654
    assert best["best_lag_sec"] == 0.37
    assert trace["correlation"][437] == best["best_lag_corr"]


def test_lag_correlation_trace_argmax_uses_same_near_tie_rule_as_best_lag():
    target = np.tile(np.asarray([1.0, -1.0]), 1500)
    trace = lag_correlation_trace_from_filtered(
        target.copy(), target, fs=100.0, max_lag_sec=4.0, low_hz=0.05
    )
    best = best_lag_correlation_from_filtered(
        target.copy(), target, fs=100.0, max_lag_sec=4.0, low_hz=0.05
    )
    finite = np.isfinite(trace["correlation"])
    maximum = np.max(trace["correlation"][finite])
    tied = trace["lag_samples"][finite & np.isclose(trace["correlation"], maximum, rtol=1e-10, atol=1e-12)]
    expected = min(tied.tolist(), key=lambda lag: (abs(lag), lag))

    assert expected == 0
    assert best["best_lag_sec"] == expected / 100.0
    assert best["best_lag_corr"] == trace["correlation"][400]
