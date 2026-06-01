import numpy as np

from resp_train.metrics.signal import (
    bandpass_filter,
    estimate_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    rms_envelope,
    spectrum_similarity,
)


def _sine(freq_hz: float, fs: float, duration_sec: float) -> np.ndarray:
    t = np.arange(int(fs * duration_sec), dtype=np.float64) / fs
    return np.sin(2 * np.pi * freq_hz * t)


def test_rms_envelope_保持长度():
    x = np.linspace(-1.0, 1.0, 101)

    env = rms_envelope(x, window_samples=11)

    assert env.shape == x.shape
    assert np.isfinite(env).all()


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
