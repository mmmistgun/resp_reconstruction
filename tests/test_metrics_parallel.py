import numpy as np
from omegaconf import OmegaConf

from resp_train.metrics.evaluate import build_target_feature_cache
from resp_train.metrics.parallel import build_target_feature_cache_chunked


def _cfg():
    return OmegaConf.create(
        {
            "window": {"target_fs": 100},
            "loss": {
                "envelope_window_sec": 2.0,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 0.7,
            },
            "evaluation": {
                "lag_bandpass_order": 4,
                "raw_peak_min_good_segment_sec": 20.0,
                "local_rr_window_sec": 40.0,
                "local_rr_step_sec": 10.0,
            },
        }
    )


def test_build_target_feature_cache_chunked_matches_serial():
    fs = 100
    t = np.arange(0, 80, 1 / fs)
    target_a = np.sin(2 * np.pi * 0.22 * t).astype(np.float32)
    target_b = np.sin(2 * np.pi * 0.28 * t + 0.2).astype(np.float32)
    target_c = (0.8 * np.sin(2 * np.pi * 0.18 * t)).astype(np.float32)
    predictions = {
        "r_tho_hat": np.stack([target_a, target_b, target_c]).reshape(3, 1, -1),
        "tho_ref": np.stack([target_a, target_b, target_c]).reshape(3, 1, -1),
        "dataset_row_id": np.asarray([1, 2, 3]),
    }

    serial = build_target_feature_cache(predictions, _cfg(), show_progress=False)
    chunked = build_target_feature_cache_chunked(
        predictions,
        _cfg(),
        target_workers=2,
        target_chunk_size=1,
        show_progress=False,
    )

    assert serial.keys() == chunked.keys()
    for key in serial:
        np.testing.assert_allclose(chunked[key], serial[key], rtol=1e-10, atol=1e-10)
