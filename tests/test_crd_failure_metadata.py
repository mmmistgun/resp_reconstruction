from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resp_train.crd.failure_metadata import build_metadata_diagnostics


def _consensus() -> pd.DataFrame:
    index = np.arange(8)
    return pd.DataFrame(
        {
            "dataset_row_id": index,
            "samp_id": np.where(index < 5, 1, 2),
            "coupling_state_id": np.where(index < 4, 1, 2),
            "envelope_target_stratum": np.where(index < 4, "low", "high"),
            "target_envelope_modulation": index / 8.0,
            "local_rr_mae_bpm_seed_mean": index.astype(float),
            "local_rr_mae_bpm_seed_sd": index / 100.0,
            "envelope_trajectory_mae_seed_mean": index / 10.0,
            "global_envelope_modulation_error_seed_mean": index / 20.0,
            "lag_aware_signed_pcc_seed_mean": 1.0 - index / 10.0,
            "ibi_coverage_seed_mean": 1.0 - index / 20.0,
            "local_rr_mae_bpm_persistent_failure": [False, False, True, True, False, True, True, True],
            "persistent_core_failure_count": [0, 0, 2, 3, 0, 2, 2, 2],
        }
    )


def _index() -> pd.DataFrame:
    index = np.arange(8)
    frame = pd.DataFrame(
        {
            "dataset_row_id": index,
            "split": "val",
            "samp_id": np.where(index < 5, 1, 2),
            "coupling_state_id": np.where(index < 4, 1, 2),
            "window_start_s": index * 30.0,
            "window_end_s": index * 30.0 + 180.0,
            "window_duration_s": 180.0,
        }
    )
    for column in (
        "hard_valid_ratio",
        "state_alignment_valid_ratio",
        "transient_motion_ratio",
        "posture_transition_ratio",
        "amplitude_reliable_ratio",
        "normalization_reliable_ratio",
        "rate_confidence_score",
        "phase_confidence_score",
        "event_confidence_score",
        "waveform_confidence_score",
        "alignment_confidence_score",
        "supervision_confidence_score",
        "state_alignment_lag_s",
        "state_alignment_drift_s_per_hour",
        "training_finite_ratio",
    ):
        frame[column] = index / 10.0
    for column in (
        "rate_confidence_level",
        "phase_confidence_level",
        "event_confidence_level",
        "waveform_confidence_level",
        "alignment_confidence_level",
        "supervision_confidence_level",
    ):
        frame[column] = np.where(index < 4, "low", "high")
    frame["state_alignment_method"] = "none"
    frame["state_alignment_is_reference_assisted"] = 0
    frame["allowed_losses"] = "rate;waveform"
    return frame


def test_build_metadata_diagnostics_links_and_groups_episodes() -> None:
    outputs = build_metadata_diagnostics(_consensus(), _index())

    assert len(outputs["windows"]) == 8
    local = outputs["episodes"].loc[outputs["episodes"]["failure_type"].eq("local_rr")]
    assert local["failure_window_n"].tolist() == [2, 3]
    assert local["span_sec"].tolist() == [210.0, 240.0]
    assert set(outputs["categorical"]["metadata"]) == {
        "rate_confidence_level",
        "phase_confidence_level",
        "event_confidence_level",
        "waveform_confidence_level",
        "alignment_confidence_level",
        "supervision_confidence_level",
        "state_alignment_method",
        "state_alignment_is_reference_assisted",
        "allowed_losses",
    }
    constant_index = _index()
    constant_index["hard_valid_ratio"] = 1.0
    constant = build_metadata_diagnostics(_consensus(), constant_index)["associations"]
    assert constant.loc[constant["metadata"].eq("hard_valid_ratio"), "spearman_rho"].isna().all()


def test_build_metadata_diagnostics_rejects_identity_mismatch() -> None:
    index = _index()
    index.loc[0, "samp_id"] = 999
    with pytest.raises(RuntimeError, match="identity"):
        build_metadata_diagnostics(_consensus(), index)
