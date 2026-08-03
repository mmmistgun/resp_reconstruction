from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from resp_train.baselines.fixed_band import (
    FIXED_BAND_EXPECTED_SIGNAL_KEY,
    FIXED_BAND_METHOD,
    add_baseline_summary_metadata,
    attach_alignment_metadata,
    evaluate_fixed_band_loader,
    prepare_fixed_band_config,
)


def _config() -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {"format": "research_v2", "bcg_input_key": "rawish", "preload_windows": True},
            "window": {"target_fs": 100, "duration_samples": 18000},
            "loss": {
                "band_low_hz": 0.05,
                "band_high_hz": 0.70,
                "scale_eps": 1e-8,
                "dynamic_eps": 1e-8,
                "corr_eps": 1e-8,
                "envelope_eps": 1e-8,
                "max_lag_sec": 0.30,
                "envelope_window_sec": 10,
                "envelope_step_sec": 5,
            },
            "evaluation": {
                "local_rr_window_sec": 60,
                "local_rr_step_sec": 15,
                "ibi_peak_distance_samples": 142,
                "ibi_match_tolerance_sec": 0.5,
                "ibi_coverage_threshold": 0.8,
                "ndtw_fs": 10,
                "ndtw_radius_sec": 0.3,
            },
            "training": {"device": "auto"},
        }
    )


def test_prepare_fixed_band_config_does_not_mutate_source() -> None:
    cfg = _config()
    prepared = prepare_fixed_band_config(cfg)

    assert cfg.data.bcg_input_key == "rawish"
    assert prepared.data.bcg_input_key == "bcg_input_segment_soft_z_key"
    assert prepared.data.preload_windows is False
    assert prepared.training.device == "cpu"


def test_fixed_band_loader_reuses_frozen_metrics_without_refiltering() -> None:
    cfg = _config()
    time = np.arange(18000, dtype=np.float32) / 100.0
    signal = np.sin(2.0 * np.pi * 0.25 * time).astype(np.float32)
    samples = [
        {
            "x": torch.from_numpy(signal).view(1, -1),
            "target": torch.from_numpy(signal.copy()).view(1, -1),
            "meta": {
                "dataset_row_id": index + 1,
                "split": "val",
                "input_set": "research_v2_waveform",
                "samp_id": 100 + index,
                "coupling_state_id": 1,
            },
        }
        for index in range(2)
    ]

    metrics, summary = evaluate_fixed_band_loader(DataLoader(samples, batch_size=2), cfg)

    assert metrics["method"].tolist() == [FIXED_BAND_METHOD, FIXED_BAND_METHOD]
    np.testing.assert_allclose(metrics["whole_rr_abs_error_bpm"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["local_rr_mae_bpm"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["lag_aware_signed_pcc"], 1.0, atol=1e-10)
    assert summary.loc[0, "n_samples"] == 2


def test_alignment_metadata_and_expected_signal_key_are_traced() -> None:
    metrics = pd.DataFrame({"dataset_row_id": [1], "whole_rr_abs_error_bpm": [0.5]})
    rows = pd.DataFrame(
        {
            "dataset_row_id": [1],
            "bcg_signal_key": [FIXED_BAND_EXPECTED_SIGNAL_KEY],
            "state_alignment_method": ["constant_shift"],
            "state_alignment_is_reference_assisted": [1],
            "state_alignment_lag_s": [0.2],
        }
    )
    attached = attach_alignment_metadata(metrics, rows)
    summary = add_baseline_summary_metadata(
        pd.DataFrame({"method": [FIXED_BAND_METHOD], "n_samples": [1]}),
        attached,
        split="val",
    )

    assert attached.loc[0, "state_alignment_method"] == "constant_shift"
    assert summary.loc[0, "source_signal_key"] == FIXED_BAND_EXPECTED_SIGNAL_KEY
    assert summary.loc[0, "split"] == "val"
