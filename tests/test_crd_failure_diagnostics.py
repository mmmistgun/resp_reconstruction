from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resp_train.crd.config import FORMAL_SEEDS
from resp_train.crd.failure_diagnostics import build_failure_diagnostics


def _frame(seed_offset: float = 0.0) -> pd.DataFrame:
    n = 2675
    index = np.arange(n)
    local = index.astype(float) + seed_offset
    frame = pd.DataFrame(
        {
            "evaluation_split": "validation",
            "split": "val",
            "input_set": "research_v2_waveform",
            "dataset_row_id": index,
            "samp_id": index % 7,
            "coupling_state_id": index % 17,
            "whole_rr_abs_error_bpm": index.astype(float) / 10.0 + seed_offset,
            "whole_rr_target_eligible": True,
            "local_rr_mae_bpm": local,
            "local_rr_target_eligible": True,
            "local_rr_target_eligible_windows": 9,
            "envelope_trajectory_mae": index.astype(float) / 100.0 + seed_offset,
            "global_envelope_modulation_error": index.astype(float) / 200.0 + seed_offset,
            "target_envelope_modulation": index.astype(float) / n,
            "envelope_target_stratum": np.where(index % 3 == 0, "low", np.where(index % 3 == 1, "medium", "high")),
            "target_stratified_envelope_spearman": 1.0 - index.astype(float) / (n + 1) - seed_offset,
            "envelope_spearman_target_eligible": True,
            "envelope_spearman_prediction_degenerate": False,
            "lag_aware_signed_pcc": 1.0 - index.astype(float) / (n + 1) - seed_offset,
            "best_lag_sec": np.where(index % 20 == 0, 0.30, 0.10),
            "joint_target_eligible": True,
            "joint_prediction_degenerate": False,
            "ibi_medae_sec": index.astype(float) / 1000.0 + seed_offset,
            "ibi_coverage": 1.0 - index.astype(float) / (n + 1) - seed_offset,
            "ibi_interpretable": index % 5 != 0,
            "ibi_target_eligible": True,
        }
    )
    return frame


def test_build_failure_diagnostics_marks_stable_worst_tail() -> None:
    frames = {seed: _frame(position * 0.001) for position, seed in enumerate(FORMAL_SEEDS)}
    outputs = build_failure_diagnostics(frames)
    consensus = outputs["window_consensus"]

    assert len(consensus) == 2675
    assert consensus.iloc[-1]["local_rr_mae_bpm_persistent_failure"]
    assert consensus.iloc[-1]["persistent_core_failure_count"] == 5
    assert consensus.iloc[0]["persistent_core_failure_count"] == 0
    assert "rate" in consensus.iloc[-1]["failure_signature"]
    assert len(outputs["thresholds"]) == 25
    assert set(outputs["strata_summary"]["stratum_axis"]) == {
        "samp_id",
        "coupling_state_id",
        "envelope_target_stratum",
        "ibi_interpretability_group",
        "lag_boundary_group",
    }
    assert len(outputs["seed_agreement"]) == 24


def test_build_failure_diagnostics_rejects_cross_seed_target_drift() -> None:
    frames = {seed: _frame(position * 0.001) for position, seed in enumerate(FORMAL_SEEDS)}
    frames[FORMAL_SEEDS[-1]].loc[0, "target_envelope_modulation"] = 999.0

    with pytest.raises(RuntimeError, match="target/static"):
        build_failure_diagnostics(frames)
