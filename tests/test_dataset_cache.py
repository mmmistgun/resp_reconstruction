from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from resp_train.data.dataset import RespWindowDataset


def test_dataset_slices_npz_window_and_returns_metadata(tmp_path: Path):
    root = tmp_path / "dataset"
    npz_dir = root / "whole_night" / "mixed_bcg_to_tho" / "88"
    npz_dir.mkdir(parents=True)
    np.savez(
        npz_dir / "sample.npz",
        bcg=np.arange(200, dtype=np.float32),
        tho=np.arange(200, dtype=np.float32) * 2,
        valid=np.ones(2, dtype=np.uint8),
    )
    df = pd.DataFrame(
        [
            {
                "dataset_row_id": 10,
                "split": "train",
                "input_set": "mixed_zscore",
                "source_npz": "../whole_night/mixed_bcg_to_tho/88/sample.npz",
                "bcg_signal_key": "bcg",
                "target_signal_key": "tho",
                "valid_sec_key": "valid",
                "window_start_sample": 10,
                "window_end_sample": 30,
                "window_duration_samples": 20,
                "target_fs": 100,
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 2,
                "residual_quality_class": "near_zero_residual",
                "usable": True,
            }
        ]
    )
    cfg = OmegaConf.create({"window": {"duration_samples": 20}, "data": {"filter_unusable": True}})
    dataset = RespWindowDataset(root / "training" / "dataset_index.csv", df, cfg, preload_windows=True)

    sample = dataset[0]
    assert torch.equal(sample["x"], torch.arange(10, 30, dtype=torch.float32).view(1, -1))
    assert torch.equal(sample["target"], (torch.arange(10, 30, dtype=torch.float32) * 2).view(1, -1))
    assert sample["meta"] == {
        "dataset_row_id": 10,
        "split": "train",
        "input_set": "mixed_zscore",
        "samp_id": 88,
        "segment_id": 1,
        "window_id_in_segment": 2,
        "residual_quality_class": "near_zero_residual",
    }


def test_dataset_rejects_short_npz_slice(tmp_path: Path):
    root = tmp_path / "dataset"
    npz_dir = root / "whole_night" / "mixed_bcg_to_tho" / "88"
    npz_dir.mkdir(parents=True)
    np.savez(
        npz_dir / "sample.npz",
        bcg=np.arange(15, dtype=np.float32),
        tho=np.arange(15, dtype=np.float32),
    )
    df = pd.DataFrame(
        [
            {
                "dataset_row_id": 11,
                "split": "train",
                "input_set": "mixed_zscore",
                "source_npz": "../whole_night/mixed_bcg_to_tho/88/sample.npz",
                "bcg_signal_key": "bcg",
                "target_signal_key": "tho",
                "valid_sec_key": "valid",
                "window_start_sample": 0,
                "window_end_sample": 20,
                "window_duration_samples": 20,
                "target_fs": 100,
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 3,
                "residual_quality_class": "near_zero_residual",
                "usable": True,
            }
        ]
    )
    cfg = OmegaConf.create({"window": {"duration_samples": 20}, "data": {"filter_unusable": True}})
    dataset = RespWindowDataset(root / "training" / "dataset_index.csv", df, cfg)

    with pytest.raises(ValueError, match="实际切片长度异常"):
        dataset[0]


def test_dataset_filters_unusable_when_configured(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "dataset_row_id": 1,
                "split": "train",
                "input_set": "mixed_zscore",
                "source_npz": "missing.npz",
                "bcg_signal_key": "bcg",
                "target_signal_key": "tho",
                "valid_sec_key": "valid",
                "window_start_sample": 0,
                "window_end_sample": 20,
                "window_duration_samples": 20,
                "target_fs": 100,
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 1,
                "residual_quality_class": "bad",
                "usable": False,
            }
        ]
    )
    cfg = OmegaConf.create({"window": {"duration_samples": 20}, "data": {"filter_unusable": True}})
    dataset = RespWindowDataset(tmp_path / "training" / "dataset_index.csv", df, cfg)
    assert len(dataset) == 0
