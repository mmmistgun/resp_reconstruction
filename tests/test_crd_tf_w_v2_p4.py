from __future__ import annotations

import json

import pandas as pd
import pytest

from resp_train.crd.tf_w_v2 import CANDIDATE_LOCK
from resp_train.crd.tf_w_v2_p4 import (
    CANDIDATE_ORDER,
    D4,
    MODEL_INPUT_ELEMENTS,
    MODEL_VARIANTS,
    PCC_METRIC,
    PRIMARY_METRICS,
    REQUIRED_OUTPUT_FILES,
    SCIENTIFIC_ROLES,
    TRAINABLE_PARAMETERS,
    W0,
    W1,
    W2,
    W3,
    build_candidate_eligibility,
    build_pareto_summary,
)


def _summary_schema() -> dict:
    return json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))["summary_schema"]


def _synthetic_seed_metrics() -> pd.DataFrame:
    values = {
        W0: (1.0, 1.0, 1.0, 1.0, 0.8000, 100.0, 100.0),
        W1: (1.010, 1.012, 1.004, 0.976, 0.7961, 101.0, 100.0),
        W2: (0.930, 0.975, 1.006, 1.017, 0.7969, 101.0, 100.0),
        W3: (0.960, 0.992, 0.975, 0.998, 0.7980, 104.0, 100.0),
        D4: (1.0025, 1.0183, 1.0128, 0.9968, 0.7971, 125.0, 77.0),
    }
    rows = []
    for order, candidate_id in enumerate(CANDIDATE_ORDER):
        whole, local, trajectory, global_envelope, pcc, throughput, allocated = values[candidate_id]
        for seed in (20260811, 20260812, 20260813):
            rows.append(
                {
                    "candidate_order": order,
                    "candidate_id": candidate_id,
                    "model_variant": MODEL_VARIANTS[candidate_id],
                    "scientific_role": SCIENTIFIC_ROLES[candidate_id],
                    "seed": seed,
                    "whole_rr_abs_error_bpm_mean": whole,
                    "local_rr_mae_bpm_mean": local,
                    "envelope_trajectory_mae_mean": trajectory,
                    "global_envelope_modulation_error_mean": global_envelope,
                    PCC_METRIC: pcc,
                    "median_train_samples_per_second": throughput,
                    "peak_allocated_mib": allocated,
                    "peak_reserved_mib": allocated + 10.0,
                    "peak_reserved_fraction": (allocated + 10.0) / 160.0,
                    "trainable_parameters": TRAINABLE_PARAMETERS[candidate_id],
                    "model_input_elements_per_sample": MODEL_INPUT_ELEMENTS[candidate_id],
                }
            )
    return pd.DataFrame(rows)


def test_p4_schema_keeps_strict_eligibility_separate_from_descriptive_tradeoff() -> None:
    eligibility = build_candidate_eligibility(_synthetic_seed_metrics(), _summary_schema()).set_index(
        "candidate_id"
    )

    assert bool(eligibility.loc[W3, "quality_eligible"])
    assert not bool(eligibility.loc[W3, "efficiency_eligible"])
    assert not bool(eligibility.loc[D4, "quality_eligible"])
    assert not bool(eligibility.loc[D4, "efficiency_eligible"])
    assert bool(eligibility.loc[D4, "descriptive_noncatastrophic_efficiency_tradeoff"])
    assert not bool(eligibility.loc[D4, "catastrophic_failure"])
    assert float(eligibility.loc[D4, "throughput_relative_gain"]) == pytest.approx(0.25)
    assert float(eligibility.loc[D4, "peak_allocated_relative_reduction"]) == pytest.approx(0.23)
    assert float(eligibility.loc[D4, "local_rr_mae_bpm_mean_relative_delta"]) > 0.01
    assert float(eligibility.loc[D4, "envelope_trajectory_mae_mean_relative_delta"]) > 0.01
    assert not bool(eligibility.loc[W1, "strict_pool_qualified"])
    assert not bool(eligibility.loc[W2, "strict_pool_qualified"])


def test_p4_pareto_retains_d4_tradeoff_without_putting_it_in_p5_allowlist() -> None:
    eligibility = build_candidate_eligibility(_synthetic_seed_metrics(), _summary_schema())
    pareto = build_pareto_summary(eligibility)

    assert pareto["strict_quality_pool"] == [W3]
    assert pareto["strict_efficiency_pool"] == []
    assert pareto["strict_quality_pareto"] == [W3]
    assert pareto["strict_efficiency_pareto"] == []
    assert pareto["descriptive_noncatastrophic_efficiency_tradeoffs"] == [D4]
    assert pareto["descriptive_quality_efficiency_pareto"] == [W3, D4]
    assert pareto["p5_quality_primary_if_authorized"] == W3
    assert pareto["p5_efficiency_candidate_if_authorized"] is None
    assert pareto["strict_eligibility_controls_p5_allowlist"] is True
    assert pareto["descriptive_pareto_does_not_override_hard_gates"] is True
    assert pareto["no_total_score_used"] is True


def test_p4_catastrophic_tradeoff_is_not_retained() -> None:
    seed_metrics = _synthetic_seed_metrics()
    seed_metrics.loc[seed_metrics["candidate_id"] == D4, "local_rr_mae_bpm_mean"] = 1.031
    eligibility = build_candidate_eligibility(seed_metrics, _summary_schema()).set_index("candidate_id")

    assert bool(eligibility.loc[D4, "catastrophic_failure"])
    assert not bool(eligibility.loc[D4, "descriptive_noncatastrophic_efficiency_tradeoff"])


def test_p4_candidate_lock_requires_exact_output_schema_and_no_samp_id_analysis() -> None:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))

    assert tuple(lock["output_contract"]["p4_required_files"]) == REQUIRED_OUTPUT_FILES
    assert lock["summary_schema"]["no_total_score"] is True
    assert lock["summary_schema"]["samp_id_analysis_allowed"] is False
    assert lock["forbidden"]["samp_id_analysis"] is True
    assert set(PRIMARY_METRICS) == {
        item["name"] for item in lock["summary_schema"]["primary_metrics"]
    }
