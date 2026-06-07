import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.metrics.baseline import evaluate_baseline_dataset


class TinyRespDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        t = np.linspace(0, 2 * np.pi, 512, dtype=np.float32)
        x = np.sin(t + idx).astype(np.float32)
        y = np.sin(t + idx * 0.1).astype(np.float32)
        return {
            "x": __import__("torch").from_numpy(x).view(1, -1),
            "target": __import__("torch").from_numpy(y).view(1, -1),
            "meta": {
                "dataset_row_id": idx,
                "split": "val",
                "input_set": "mixed_zscore",
                "samp_id": 1,
                "segment_id": 1,
                "window_id_in_segment": idx + 1,
                "residual_quality_class": "near_zero_residual",
            },
        }


def test_evaluate_baseline_dataset_returns_metrics_frame():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100},
            "loss": {"spectrum_low_hz": 0.05, "spectrum_high_hz": 5.0, "envelope_window_sec": 0.2},
            "baseline": {"bandpass_low_hz": 0.05, "bandpass_high_hz": 5.0, "filter_order": 2},
        }
    )

    frame = evaluate_baseline_dataset(TinyRespDataset(), cfg)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert set(frame.columns) >= {"method", "dataset_row_id", "rr_spec_abs_error", "spectrum_similarity"}
