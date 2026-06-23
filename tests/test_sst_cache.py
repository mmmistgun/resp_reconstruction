from types import SimpleNamespace

import numpy as np

from scripts import precompute_sst_cache as sst_cache
from scripts.precompute_sst_cache import compute_sst_window, BAND_LOW_HZ, BAND_HIGH_HZ


def test_sst_window_shape_and_finite():
    sig = np.sin(2 * np.pi * 0.25 * np.arange(18000) / 100.0)
    feat, freqs = compute_sst_window(sig, fs=100.0, low_hz=BAND_LOW_HZ, high_hz=BAND_HIGH_HZ, n_frames=37)
    assert feat.ndim == 2
    assert feat.shape[1] == 37
    assert feat.shape[0] == len(freqs)
    assert np.isfinite(feat).all()
    assert (freqs >= BAND_LOW_HZ - 1e-6).all() and (freqs <= BAND_HIGH_HZ + 1e-6).all()


def test_sst_downsample_uses_mean_not_subsample():
    # 时间轴前半高能、后半零，降采样到 2 帧 → 两帧应明显不同（均值池化保住能量，非抽点）。
    sig = np.concatenate(
        [np.sin(2 * np.pi * 0.25 * np.arange(9000) / 100.0), np.zeros(9000)]
    )
    feat, _ = compute_sst_window(sig, fs=100.0, low_hz=BAND_LOW_HZ, high_hz=BAND_HIGH_HZ, n_frames=2)
    assert feat[:, 0].mean() > feat[:, 1].mean()


def test_sst_window_is_log1p_n0():
    # N0 口径：存的是 log1p(|SST|)，非负。
    sig = np.sin(2 * np.pi * 0.25 * np.arange(18000) / 100.0)
    feat, _ = compute_sst_window(sig, fs=100.0, low_hz=BAND_LOW_HZ, high_hz=BAND_HIGH_HZ, n_frames=37)
    assert (feat >= 0).all()


def test_precompute_collection_skips_duplicates_and_returns_stacked_array(monkeypatch):
    cfg = SimpleNamespace()
    rows_by_split = {
        "train": [
            (10, np.array([1.0], dtype=np.float32)),
            (11, np.array([2.0], dtype=np.float32)),
        ],
        "val": [
            (10, np.array([9.0], dtype=np.float32)),
            (12, np.array([3.0], dtype=np.float32)),
        ],
    }
    computed: list[float] = []

    def fake_iter_split_rows(_cfg, split):
        yield from rows_by_split[split]

    def fake_compute(sig, *, fs, low_hz, high_hz, n_frames):
        computed.append(float(sig[0]))
        feat = np.full((2, n_frames), float(sig[0]), dtype=np.float32)
        return feat, np.array([0.1, 0.2], dtype=np.float32)

    monkeypatch.setattr(sst_cache, "_iter_split_rows", fake_iter_split_rows)

    row_ids, arr, freqs = sst_cache.precompute_sst_features(
        cfg,
        splits=["train", "val"],
        fs=100.0,
        low_hz=0.05,
        high_hz=8.0,
        n_frames=3,
        workers=1,
        log_every=100,
        compute_fn=fake_compute,
    )

    assert row_ids == [10, 11, 12]
    assert computed == [1.0, 2.0, 3.0]
    assert arr.shape == (3, 2, 3)
    assert arr.dtype == np.float32
    assert arr[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(freqs, np.array([0.1, 0.2], dtype=np.float32))
