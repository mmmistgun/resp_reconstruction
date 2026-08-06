from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from resp_train.baselines import iewt
from resp_train.baselines.fixed_band import attach_alignment_metadata
from resp_train.baselines.iewt import (
    IEWT_EXPECTED_SIGNAL_KEY,
    IEWT_METHOD,
    IEWTResult,
    IEWTWindowResult,
    add_iewt_summary_metadata,
    evaluate_iewt_loader,
    prepare_iewt_config,
)


def _config() -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {"format": "research_v2", "bcg_input_key": "other", "preload_windows": True},
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
                "envelope_quantile_method": "linear",
                "envelope_strata_low": 0.1,
                "envelope_strata_high": 0.5,
            },
            "training": {"device": "auto"},
        }
    )


def test_prepare_iewt_config_does_not_mutate_source() -> None:
    cfg = _config()
    prepared = prepare_iewt_config(cfg)

    assert cfg.data.bcg_input_key == "other"
    assert cfg.data.preload_windows is True
    assert prepared.data.bcg_input_key == "bcg_rawish_segment_soft_z_key"
    assert prepared.data.preload_windows is False
    assert prepared.training.device == "cpu"


def test_prepare_iewt_config_rejects_non_protocol_window() -> None:
    cfg = _config()
    cfg.window.duration_samples = 9000

    with np.testing.assert_raises_regex(ValueError, "100 Hz、180 秒"):
        prepare_iewt_config(cfg)


def test_iewt_loader_uses_bcg_only_and_reuses_frozen_metrics(monkeypatch) -> None:
    cfg = _config()
    time = np.arange(18000, dtype=np.float32) / 100.0
    signal = np.sin(2.0 * np.pi * 0.25 * time).astype(np.float32)
    calls: list[np.ndarray] = []

    def identity_iewt(sample: np.ndarray, *, fs: float, config) -> IEWTResult:
        del fs, config
        values = np.asarray(sample, dtype=np.float64).copy()
        calls.append(values)
        diagnostic = IEWTWindowResult(
            waveform=np.zeros(3500),
            spectrum=np.ones(36),
            upper_envelope=np.ones(36),
            boundary_bins=np.asarray([10]),
            boundary_hz=np.asarray([10.0 / 35.0]),
            selected_band_indices=np.asarray([0]),
        )
        return IEWTResult(waveform=values, windows=(diagnostic,) * 6)

    monkeypatch.setattr(iewt, "extract_respiration_iewt", identity_iewt)
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

    metrics, summary = evaluate_iewt_loader(DataLoader(samples, batch_size=2), cfg)

    assert len(calls) == 2
    assert metrics["method"].tolist() == [IEWT_METHOD, IEWT_METHOD]
    np.testing.assert_allclose(metrics["whole_rr_abs_error_bpm"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["local_rr_mae_bpm"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["lag_aware_signed_pcc"], 1.0, atol=1e-10)
    assert summary.loc[0, "n_samples"] == 2
    assert metrics["iewt_boundary_count_min"].tolist() == [1, 1]
    assert metrics["iewt_selected_mode_count_max"].tolist() == [1, 1]


def test_iewt_summary_records_source_and_protocol() -> None:
    metrics = pd.DataFrame({"dataset_row_id": [1], "whole_rr_abs_error_bpm": [0.5]})
    rows = pd.DataFrame(
        {
            "dataset_row_id": [1],
            "bcg_signal_key": [IEWT_EXPECTED_SIGNAL_KEY],
            "state_alignment_method": ["constant_shift"],
            "state_alignment_is_reference_assisted": [1],
            "state_alignment_lag_s": [0.2],
        }
    )
    attached = attach_alignment_metadata(metrics, rows)
    summary = add_iewt_summary_metadata(
        pd.DataFrame({"method": [IEWT_METHOD], "n_samples": [1]}),
        attached,
        split="val",
    )

    assert summary.loc[0, "source_signal_key"] == IEWT_EXPECTED_SIGNAL_KEY
    assert summary.loc[0, "split"] == "val"
    assert summary.loc[0, "iewt_context_sec"] == 35.0
    assert summary.loc[0, "iewt_output_step_sec"] == 30.0
    assert summary.loc[0, "iewt_post_lowpass_hz"] == 1.0
    assert summary.loc[0, "iewt_filter_phase"] == "zero_phase"
