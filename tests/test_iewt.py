from __future__ import annotations

import numpy as np
import pytest

from resp_train.baselines import iewt
from resp_train.baselines.iewt import (
    IEWTConfig,
    IEWTWindowResult,
    detect_boundaries,
    extract_iewt_window,
    extract_respiration_iewt,
    meyer_filter_bank,
)


def test_detect_boundaries_uses_local_leftmost_tie() -> None:
    spectrum = np.asarray([9.0, 8.0, 5.0, 1.0, 1.0, 4.0, 6.0, 3.0, 0.5, 0.5])
    labels = np.asarray([[1, 2], [5, 6]])

    boundaries = detect_boundaries(spectrum, labels)

    np.testing.assert_array_equal(boundaries, [3, 8])


def test_detect_boundaries_rejects_missing_boundary() -> None:
    spectrum = np.asarray([8.0, 1.0, 3.0, 2.0, 4.0])
    labels = np.asarray([[1, 4]])

    with pytest.raises(ValueError, match="未检测到"):
        detect_boundaries(spectrum, labels)


def test_meyer_filter_bank_is_finite_and_has_bounded_overlap_energy() -> None:
    filters = meyer_filter_bank(np.asarray([0.35, 0.9, 1.7]), 6999)
    stacked = np.stack(filters)

    assert stacked.shape == (4, 6999)
    assert np.isfinite(stacked).all()
    overlap_energy = np.sum(stacked**2, axis=0)
    # 原 EEWT 最末 wavelet 在 pi 附近保留单侧 Meyer taper，因此 Nyquist 附近下界为 0.5。
    assert float(np.min(overlap_energy)) >= 0.5 - 1e-12
    assert float(np.max(overlap_energy)) <= 1.0 + 1e-12


def test_lowpass_is_zero_phase_around_centered_impulse() -> None:
    signal = np.zeros(3500, dtype=np.float64)
    center = signal.size // 2
    signal[center] = 1.0

    filtered = iewt._zero_phase_lowpass(signal, fs=100.0, cutoff_hz=1.0, order=3)

    np.testing.assert_allclose(
        filtered[center - 300 : center],
        filtered[center + 1 : center + 301][::-1],
        atol=1e-12,
        rtol=1e-12,
    )
    assert int(np.argmax(filtered)) == center


def test_single_window_iewt_is_deterministic_and_reconstructs_selected_modes() -> None:
    time = np.arange(3500, dtype=np.float64) / 100.0
    signal = (
        0.9 * np.sin(2.0 * np.pi * 0.23 * time)
        + 0.35 * np.sin(2.0 * np.pi * 0.41 * time + 0.3)
        + 0.02 * time
    )

    first = extract_iewt_window(signal, fs=100.0, keep_components=True)
    second = extract_iewt_window(signal, fs=100.0, keep_components=True)

    assert first.waveform.shape == (3500,)
    assert first.spectrum.shape == (36,)
    assert first.components is not None
    assert np.isfinite(first.waveform).all()
    assert np.all(np.diff(first.boundary_bins) > 0)
    assert np.all((first.selected_band_indices >= 0) & (first.selected_band_indices < len(first.components)))
    expected = np.sum(np.stack([first.components[index] for index in first.selected_band_indices]), axis=0)
    np.testing.assert_allclose(first.waveform, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first.waveform, second.waveform, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(first.boundary_bins, second.boundary_bins)
    np.testing.assert_array_equal(first.selected_band_indices, second.selected_band_indices)


def test_full_window_uses_35_second_context_and_keeps_six_exact_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[np.ndarray] = []

    def fake_window(
        signal: np.ndarray,
        *,
        fs: float,
        config: IEWTConfig,
        keep_components: bool,
    ) -> IEWTWindowResult:
        del fs, config, keep_components
        values = np.asarray(signal, dtype=np.float64).copy()
        seen.append(values)
        return IEWTWindowResult(
            waveform=values,
            spectrum=np.ones(36),
            upper_envelope=np.ones(36),
            boundary_bins=np.asarray([10]),
            boundary_hz=np.asarray([10.0 / 35.0]),
            selected_band_indices=np.asarray([0]),
        )

    monkeypatch.setattr(iewt, "extract_iewt_window", fake_window)
    signal = np.arange(18000, dtype=np.float64)
    result = extract_respiration_iewt(
        signal,
        fs=100.0,
        config=IEWTConfig(post_lowpass=False),
    )

    assert len(seen) == 6
    np.testing.assert_array_equal(seen[0], signal[:3500])
    np.testing.assert_array_equal(seen[1], signal[2500:6000])
    np.testing.assert_array_equal(seen[-1], signal[14500:18000])
    np.testing.assert_array_equal(result.waveform, signal)


@pytest.mark.parametrize(
    "signal, message",
    [
        (np.zeros(17999), "整数倍"),
        (np.full(18000, np.nan), "NaN/Inf"),
    ],
)
def test_full_window_rejects_protocol_violations(signal: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        extract_respiration_iewt(signal, fs=100.0)
