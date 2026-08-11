from __future__ import annotations

import pandas as pd

from resp_train.crd.c1_selection import C1_VARIANT, SEED_SUMMARY_METRICS, apply_c1_decision
from resp_train.crd.config import FORMAL_SEEDS
from resp_train.crd.s2_selection import BASE_VARIANT


def _seed_summary(
    *,
    base_local: tuple[float, float, float] = (0.55, 0.56, 0.57),
    candidate_local: tuple[float, float, float] = (0.54, 0.55, 0.56),
    base_trajectory: float = 0.15,
    candidate_trajectory: float = 0.151,
    base_pcc: float = 0.865,
    candidate_pcc: float = 0.863,
) -> pd.DataFrame:
    rows = []
    for position, seed in enumerate(FORMAL_SEEDS):
        for variant, local, trajectory, pcc in (
            (BASE_VARIANT, base_local[position], base_trajectory, base_pcc),
            (C1_VARIANT, candidate_local[position], candidate_trajectory, candidate_pcc),
        ):
            values = {
                "whole_rr_abs_error_bpm": 0.5,
                "local_rr_mae_bpm": local,
                "envelope_trajectory_mae": trajectory,
                "global_envelope_modulation_error": 0.2,
                "lag_aware_signed_pcc": pcc,
                "ibi_medae_sec": 0.08,
                "ibi_coverage": 0.83,
                "target_stratified_envelope_spearman": 0.5,
            }
            rows.append(
                {"variant": variant, "seed": seed, **{f"{key}_mean": values[key] for key in SEED_SUMMARY_METRICS}}
            )
    return pd.DataFrame(rows)


def test_c1_quality_superior_requires_all_four_checks() -> None:
    decision = apply_c1_decision(_seed_summary())
    assert decision["outcome"] == "quality_superior"
    assert decision["quality_superior"] is True
    assert decision["paired_local_rr_improved_seed_count"] == 3
    assert decision["retain_crd_102"] is False

    failed = apply_c1_decision(_seed_summary(candidate_trajectory=0.154))
    assert failed["quality_checks"]["trajectory_worsening_at_most_1_5pct"] is False
    assert failed["outcome"] == "mamba_retained_control_failure"
    assert failed["retain_crd_102"] is True


def test_c1_quality_near_is_not_promoted_without_efficiency_benchmark() -> None:
    decision = apply_c1_decision(
        _seed_summary(candidate_local=(0.551, 0.561, 0.571), candidate_trajectory=0.151, candidate_pcc=0.863)
    )
    assert decision["quality_superior"] is False
    assert decision["quality_near_guardrails"] is True
    assert decision["outcome"] == "quality_near_requires_efficiency_benchmark"
    assert decision["efficiency_benchmark_required"] is True
