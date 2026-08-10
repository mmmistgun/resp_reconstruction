from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resp_train.crd.config import FORMAL_SEEDS
from resp_train.crd.prototype_diagnostics import summarize_prototype_frame
from resp_train.crd.s2_selection import (
    BASE_VARIANT,
    REPORT_METRICS,
    apply_s2a_decision,
)


def _seed_rows(values_by_variant: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for variant, values in values_by_variant.items():
        for seed in FORMAL_SEEDS:
            row = {"variant": variant, "seed": seed}
            row.update({metric: 0.5 for metric in REPORT_METRICS})
            row.update(values)
            rows.append(row)
    return pd.DataFrame(rows)


def test_s2a_decision_rejects_energy_and_morphology_when_required_checks_fail() -> None:
    seed_summary = _seed_rows(
        {
            BASE_VARIANT: {
                "local_rr_mae_bpm_mean": 0.55,
                "envelope_trajectory_mae_mean": 0.15,
                "global_envelope_modulation_error_mean": 0.1875,
                "lag_aware_signed_pcc_mean": 0.865,
                "ibi_coverage_mean": 0.84,
            },
            "crd_202_base_legacy_energy": {
                "local_rr_mae_bpm_mean": 0.54,
                "envelope_trajectory_mae_mean": 0.153,
                "global_envelope_modulation_error_mean": 0.185,
                "lag_aware_signed_pcc_mean": 0.866,
                "ibi_coverage_mean": 0.84,
            },
            "crd_203_base_analytic_am": {
                "local_rr_mae_bpm_mean": 0.552,
                "envelope_trajectory_mae_mean": 0.16,
                "global_envelope_modulation_error_mean": 0.18,
                "lag_aware_signed_pcc_mean": 0.857,
                "ibi_coverage_mean": 0.83,
            },
            "crd_204_base_morphology": {
                "local_rr_mae_bpm_mean": 0.57,
                "envelope_trajectory_mae_mean": 0.1505,
                "global_envelope_modulation_error_mean": 0.19,
                "lag_aware_signed_pcc_mean": 0.858,
                "ibi_coverage_mean": 0.828,
            },
        }
    )

    _, decision = apply_s2a_decision(seed_summary)

    assert decision["energy_selection"]["selected_energy"] == "none"
    assert decision["morphology_eligibility"]["eligible"] is False
    assert decision["s2b_activated"] is False
    assert decision["selected_model_after_s2a"] == BASE_VARIANT
    assert decision["outcome"] == "retain_base"


def test_s2a_decision_prefers_simpler_e_when_both_energy_candidates_are_eligible() -> None:
    seed_summary = _seed_rows(
        {
            BASE_VARIANT: {
                "local_rr_mae_bpm_mean": 0.55,
                "envelope_trajectory_mae_mean": 0.15,
                "global_envelope_modulation_error_mean": 0.1875,
                "lag_aware_signed_pcc_mean": 0.865,
                "ibi_coverage_mean": 0.84,
            },
            "crd_202_base_legacy_energy": {
                "local_rr_mae_bpm_mean": 0.548,
                "envelope_trajectory_mae_mean": 0.145,
                "global_envelope_modulation_error_mean": 0.188,
                "lag_aware_signed_pcc_mean": 0.864,
                "ibi_coverage_mean": 0.84,
            },
            "crd_203_base_analytic_am": {
                "local_rr_mae_bpm_mean": 0.549,
                "envelope_trajectory_mae_mean": 0.146,
                "global_envelope_modulation_error_mean": 0.187,
                "lag_aware_signed_pcc_mean": 0.864,
                "ibi_coverage_mean": 0.84,
            },
            "crd_204_base_morphology": {
                "local_rr_mae_bpm_mean": 0.554,
                "envelope_trajectory_mae_mean": 0.151,
                "global_envelope_modulation_error_mean": 0.19,
                "lag_aware_signed_pcc_mean": 0.868,
                "ibi_coverage_mean": 0.835,
            },
        }
    )

    _, decision = apply_s2a_decision(seed_summary)

    assert decision["energy_eligibility"]["crd_202_base_legacy_energy"]["eligible"] is True
    assert decision["energy_eligibility"]["crd_203_base_analytic_am"]["eligible"] is True
    assert decision["energy_selection"]["selected_energy"] == "E"
    assert decision["morphology_eligibility"]["eligible"] is True
    assert decision["s2b_activated"] is True
    assert decision["selected_model_after_s2a"] is None


def test_prototype_summary_reports_global_and_samp_usage_entropy() -> None:
    frame = pd.DataFrame(
        {
            "dataset_row_id": [1, 2],
            "samp_id": [10, 11],
            "prototype_token_entropy_normalized": [0.4, 0.6],
            "prototype_hard_usage_00": [1.0, 0.0],
            "prototype_hard_usage_01": [0.0, 1.0],
            "prototype_soft_usage_00": [0.75, 0.25],
            "prototype_soft_usage_01": [0.25, 0.75],
        }
    )

    summary, samp = summarize_prototype_frame(frame, prototype_count=2)

    assert summary["hard_usage"] == pytest.approx([0.5, 0.5])
    assert summary["soft_usage"] == pytest.approx([0.5, 0.5])
    assert summary["hard_usage_entropy_normalized"] == pytest.approx(1.0)
    assert summary["soft_usage_entropy_normalized"] == pytest.approx(1.0)
    assert summary["mean_token_entropy_normalized"] == pytest.approx(0.5)
    assert len(samp) == 2
    assert np.isfinite(samp.select_dtypes(include=[np.number]).to_numpy()).all()
