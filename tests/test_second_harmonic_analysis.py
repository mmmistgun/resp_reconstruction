from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from resp_train.analysis.second_harmonic import (
    HarmonicFeatureConfig,
    HarmonicThresholds,
    classify_harmonic_window,
    classify_model_correction,
    extract_harmonic_features,
    resolve_eligibility_status,
)


def _sine(freq_hz: float, *, fs: float = 100.0, seconds: float = 180.0) -> np.ndarray:
    time = np.arange(int(fs * seconds), dtype=np.float64) / fs
    return np.sin(2.0 * np.pi * freq_hz * time)


def _config() -> HarmonicFeatureConfig:
    return HarmonicFeatureConfig(
        fs=100.0,
        low_hz=0.05,
        high_hz=0.7,
        filter_order=4,
        welch_nperseg=4096,
        neighborhood_hz=0.025,
        energy_floor=1e-12,
        tho_rr_agreement_bpm=1.0,
    )


def _thresholds() -> HarmonicThresholds:
    return HarmonicThresholds(
        version="test-v1",
        tho_rr_agreement_bpm=1.0,
        peak_relative_tolerance=0.10,
        harmonic_to_fundamental_min=1.5,
        harmonic_band_fraction_min=0.40,
        correction_ratio_drop_min=0.20,
    )


def test_extract_harmonic_features_detects_dominant_second_harmonic() -> None:
    f0 = 0.20
    tho = _sine(f0)
    bcg = 0.35 * _sine(f0) + _sine(2.0 * f0)

    result = extract_harmonic_features(bcg, tho, cfg=_config())

    assert result.status == "eligible"
    assert result.bcg_peak_hz / result.tho_reference_hz == pytest.approx(2.0, rel=0.08)
    assert result.harmonic_to_fundamental_ratio > 1.5
    assert result.harmonic_band_fraction > 0.4
    assert classify_harmonic_window(result, _thresholds()) == "strong_harmonic"


def test_extract_harmonic_features_keeps_pure_fundamental_negative() -> None:
    f0 = 0.20
    result = extract_harmonic_features(_sine(f0), _sine(f0), cfg=_config())

    assert result.status == "eligible"
    assert result.peak_to_tho_ratio == pytest.approx(1.0, rel=0.08)
    assert result.harmonic_to_fundamental_ratio < 0.1
    assert classify_harmonic_window(result, _thresholds()) == "harmonic_negative"


def test_resolve_eligibility_status_excludes_unstable_and_out_of_band_reference() -> None:
    cfg = _config()

    assert resolve_eligibility_status(12.0, 13.5, cfg=cfg) == "tho_reference_unstable"
    assert resolve_eligibility_status(22.0, 22.5, cfg=cfg) == "second_harmonic_out_of_band"
    assert resolve_eligibility_status(12.0, 12.5, cfg=cfg) == "eligible"


def test_zero_energy_input_uses_finite_ratio_protection() -> None:
    result = extract_harmonic_features(np.zeros(18000), _sine(0.20), cfg=_config())

    assert result.status == "eligible"
    assert result.fundamental_energy == pytest.approx(0.0, abs=1e-20)
    assert result.second_harmonic_energy == pytest.approx(0.0, abs=1e-20)
    assert np.isfinite(result.harmonic_to_fundamental_ratio)
    assert np.isfinite(result.harmonic_band_fraction)


def test_extract_harmonic_features_rejects_nonfinite_input() -> None:
    bcg = _sine(0.20)
    bcg[10] = np.nan

    with pytest.raises(ValueError, match="非有限"):
        extract_harmonic_features(bcg, _sine(0.20), cfg=_config())


def test_classify_model_correction_requires_harmonic_drop_and_fundamental_peak() -> None:
    input_features = extract_harmonic_features(
        0.35 * _sine(0.20) + _sine(0.40),
        _sine(0.20),
        cfg=_config(),
    )
    corrected = replace(
        input_features,
        bcg_peak_hz=input_features.tho_reference_hz,
        peak_to_tho_ratio=1.0,
        harmonic_to_fundamental_ratio=input_features.harmonic_to_fundamental_ratio * 0.5,
    )
    partial = replace(
        corrected,
        bcg_peak_hz=2.0 * input_features.tho_reference_hz,
        peak_to_tho_ratio=2.0,
    )
    unchanged = replace(
        corrected,
        harmonic_to_fundamental_ratio=input_features.harmonic_to_fundamental_ratio * 0.95,
    )

    thresholds = _thresholds()
    assert classify_model_correction(input_features, corrected, thresholds) == "corrected"
    assert classify_model_correction(input_features, partial, thresholds) == "partially_corrected"
    assert classify_model_correction(input_features, unchanged, thresholds) == "not_corrected"


def test_config_and_thresholds_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="fs"):
        HarmonicFeatureConfig(fs=0.0)
    with pytest.raises(ValueError, match="correction_ratio_drop_min"):
        replace(_thresholds(), correction_ratio_drop_min=1.1)
