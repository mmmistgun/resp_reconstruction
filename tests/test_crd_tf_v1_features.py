from __future__ import annotations

import numpy as np
import pytest
import torch

from resp_train.crd.tf_v1_features import (
    LearnableCarrierModulation,
    RidgeParameters,
    fixed_transform_spec,
    morlet_target_frequencies,
    multires_stft_features,
    one_sided_input_spectrum,
    wsst_ridge_features,
)


def test_multires_stft_has_frozen_shapes_and_localizes_frequency() -> None:
    time = np.arange(18000, dtype=np.float64) / 100.0
    waveform = (np.sin(2.0 * np.pi * 0.2 * time) + 0.5 * np.sin(2.0 * np.pi * 4.0 * time)).astype(
        np.float32
    )

    features = multires_stft_features(waveform)

    assert features["m_slow"].shape == (36, 101)
    assert features["m_fast"].shape == (44, 349)
    assert features["m_slow"].dtype == np.float32
    assert features["m_fast"].dtype == np.float32
    slow_frequencies = np.arange(1, 37) * 100.0 / 3000
    fast_frequencies = np.arange(5, 49) * 100.0 / 600
    assert slow_frequencies[np.argmax(features["m_slow"].mean(axis=1))] == pytest.approx(0.2)
    assert fast_frequencies[np.argmax(features["m_fast"].mean(axis=1))] == pytest.approx(4.0)


def test_morlet_target_grids_are_frozen_and_increasing() -> None:
    w = morlet_target_frequencies(voices_per_octave=12, count=97)
    s = morlet_target_frequencies(voices_per_octave=48, count=387)

    assert w.shape == (97,)
    assert s.shape == (387,)
    assert np.all(np.diff(w) > 0)
    assert np.all(np.diff(s) > 0)
    assert w[-1] == pytest.approx(8.0)
    assert s[-1] == pytest.approx(8.0)
    assert w[0] >= 0.03
    assert s[0] >= 0.03


def test_learnable_carrier_modulation_supports_cached_spectrum_and_gradients() -> None:
    time = torch.arange(18000, dtype=torch.float32) / 100.0
    waveform = ((1.0 + 0.5 * torch.sin(2.0 * torch.pi * 0.2 * time)) * torch.sin(2.0 * torch.pi * 2.0 * time))
    model = LearnableCarrierModulation(chunk_filters=5)

    direct = model(waveform[None, None, :])
    spectrum = one_sided_input_spectrum(waveform)
    cached = model(one_sided_spectrum=torch.from_numpy(spectrum)[None, :])
    cached.square().mean().backward()

    assert direct.shape == (1, 24, 360)
    assert torch.allclose(direct, cached, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(cached).all()
    assert model.center_logits.grad is not None
    assert model.bandwidth_logits.grad is not None
    assert torch.isfinite(model.center_logits.grad).all()
    assert torch.isfinite(model.bandwidth_logits.grad).all()
    centers = model.centers().detach()
    assert torch.all(centers[1:] > centers[:-1])
    assert float(centers.min()) >= 0.70 - 1e-6
    assert float(centers.max()) <= 8.00 + 1e-6


def test_wsst_ridge_features_tracks_two_components_per_band() -> None:
    frequencies = np.geomspace(0.03, 8.0, 300)
    energy = np.zeros((300, 360), dtype=np.float32)
    targets = (0.2, 0.4, 2.0, 4.0)
    amplitudes = (10.0, 8.0, 12.0, 9.0)
    for target, amplitude in zip(targets, amplitudes):
        index = int(np.argmin(np.abs(frequencies - target)))
        energy[index] = amplitude

    features = wsst_ridge_features(
        energy,
        frequencies,
        RidgeParameters(smoothness_penalty=2.0, suppression_radius_bins=3),
    )

    assert features.shape == (12, 360)
    respiratory = sorted((float(np.median(features[0])), float(np.median(features[3]))))
    carrier = sorted((float(np.median(features[6])), float(np.median(features[9]))))
    assert respiratory[0] == pytest.approx(0.2, rel=0.03)
    assert respiratory[1] == pytest.approx(0.4, rel=0.03)
    assert carrier[0] == pytest.approx(2.0, rel=0.03)
    assert carrier[1] == pytest.approx(4.0, rel=0.03)
    assert np.all((features[[2, 5, 8, 11]] >= 0.0) & (features[[2, 5, 8, 11]] <= 1.0))


def test_tf_v1_features_reject_nonfinite_input() -> None:
    waveform = np.zeros(18000, dtype=np.float32)
    waveform[10] = np.nan
    with pytest.raises(FloatingPointError):
        multires_stft_features(waveform)
    with pytest.raises(FloatingPointError):
        one_sided_input_spectrum(waveform)


def test_fixed_transform_spec_has_no_target_or_test_cache() -> None:
    spec = fixed_transform_spec()
    serialized = str(spec).lower()
    assert spec["protocol"] == "crd-tf-v1-research-informed-20260812"
    assert "target" not in serialized
    assert "test" not in serialized
    assert spec["l_spectrum"]["shape"] == [9001]
