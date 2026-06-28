from types import SimpleNamespace

import numpy as np

from scripts import precompute_f_d_highfreq_cache as fd_cache


def test_high_stft_window_shape_and_finite():
    sig = np.sin(2 * np.pi * 2.0 * np.arange(18000) / 100.0)

    feat, freqs = fd_cache.compute_high_stft_window(
        sig,
        fs=100.0,
        low_hz=1.0,
        high_hz=8.0,
        win_length=800,
        hop_length=100,
        n_fft=800,
        n_frames=180,
    )

    assert feat.ndim == 2
    assert feat.shape[1] == 180
    assert feat.shape[0] == len(freqs)
    assert np.isfinite(feat).all()
    assert (freqs >= 1.0 - 1e-6).all() and (freqs <= 8.0 + 1e-6).all()


def test_modulation_features_are_fixed_channel_sequence():
    logmag = np.abs(np.random.default_rng(0).normal(size=(36, 180))).astype(np.float32)
    freqs = np.linspace(1.0, 8.0, 36, dtype=np.float32)

    features, names = fd_cache.compute_modulation_features(logmag, freqs)

    assert features.shape == (8, 180)
    assert names == fd_cache.MODULATION_FEATURE_NAMES
    assert np.isfinite(features).all()


def test_precompute_collection_supports_feature_mode(monkeypatch):
    cfg = SimpleNamespace()
    rows_by_split = {
        "train": [(10, np.array([1.0], dtype=np.float32))],
        "val": [(11, np.array([2.0], dtype=np.float32))],
    }

    def fake_iter_split_rows(_cfg, split):
        yield from rows_by_split[split]

    def fake_compute(sig, **_kwargs):
        value = float(sig[0])
        feat = np.full((36, 4), value, dtype=np.float32)
        freqs = np.linspace(1.0, 8.0, 36, dtype=np.float32)
        return feat, freqs

    monkeypatch.setattr(fd_cache, "_iter_split_rows", fake_iter_split_rows)

    row_ids, arr, freqs, feature_names = fd_cache.precompute_highfreq_features(
        cfg,
        splits=["train", "val"],
        mode="modulation",
        fs=100.0,
        low_hz=1.0,
        high_hz=8.0,
        n_frames=4,
        workers=1,
        log_every=100,
        compute_fn=fake_compute,
    )

    assert row_ids == [10, 11]
    assert arr.shape == (2, 8, 4)
    assert np.allclose(arr[:, 0, 0], [1.0, 2.0])
    assert freqs.shape == (36,)
    assert feature_names == fd_cache.MODULATION_FEATURE_NAMES
