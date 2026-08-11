from __future__ import annotations

import pandas as pd

from resp_train.crd.config import FORMAL_SEEDS
from resp_train.crd.s2_selection import BASE_VARIANT, REPORT_METRICS
from resp_train.crd.s2br_selection import S2A_SINGLE_VARIANTS, S2BR_VARIANTS, apply_s2br_decision


def _seed_rows(values_by_variant: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for variant in (BASE_VARIANT, *S2A_SINGLE_VARIANTS, *S2BR_VARIANTS):
        for seed in FORMAL_SEEDS:
            row = {"variant": variant, "seed": seed}
            row.update({metric: 0.5 for metric in REPORT_METRICS})
            row.update(values_by_variant[variant])
            rows.append(row)
    return pd.DataFrame(rows)


def _common(*, local: float, trajectory: float, pcc: float, coverage: float) -> dict[str, float]:
    return {
        "local_rr_mae_bpm_mean": local,
        "envelope_trajectory_mae_mean": trajectory,
        "lag_aware_signed_pcc_mean": pcc,
        "ibi_coverage_mean": coverage,
    }


def test_s2br_decision_retains_base_when_both_combinations_fail() -> None:
    values = {
        BASE_VARIANT: _common(local=0.55, trajectory=0.15, pcc=0.865, coverage=0.84),
        "crd_202_base_legacy_energy": _common(local=0.54, trajectory=0.153, pcc=0.865, coverage=0.84),
        "crd_203_base_analytic_am": _common(local=0.56, trajectory=0.16, pcc=0.857, coverage=0.83),
        "crd_204_base_morphology": _common(local=0.57, trajectory=0.151, pcc=0.858, coverage=0.829),
        "crd_205_base_em_static": _common(local=0.56, trajectory=0.149, pcc=0.859, coverage=0.829),
        "crd_206_base_am_static": _common(local=0.561, trajectory=0.157, pcc=0.853, coverage=0.832),
        "crd_207_base_cap_em": _common(local=0.551, trajectory=0.15, pcc=0.866, coverage=0.84),
        "crd_208_base_cap_am": _common(local=0.548, trajectory=0.15, pcc=0.866, coverage=0.84),
    }

    _, decision = apply_s2br_decision(_seed_rows(values))

    assert decision["passing_combinations"] == []
    assert decision["selected_model"] == BASE_VARIANT
    assert decision["s3_activated"] is False
    assert decision["outcome"] == "retain_base"


def test_s2br_decision_opens_s3_only_for_combination_passing_all_three_comparators() -> None:
    values = {
        BASE_VARIANT: _common(local=0.55, trajectory=0.15, pcc=0.865, coverage=0.84),
        "crd_202_base_legacy_energy": _common(local=0.54, trajectory=0.153, pcc=0.865, coverage=0.84),
        "crd_203_base_analytic_am": _common(local=0.56, trajectory=0.16, pcc=0.857, coverage=0.83),
        "crd_204_base_morphology": _common(local=0.57, trajectory=0.151, pcc=0.858, coverage=0.829),
        "crd_205_base_em_static": _common(local=0.53, trajectory=0.151, pcc=0.864, coverage=0.835),
        "crd_206_base_am_static": _common(local=0.561, trajectory=0.157, pcc=0.853, coverage=0.832),
        "crd_207_base_cap_em": _common(local=0.54, trajectory=0.15, pcc=0.866, coverage=0.84),
        "crd_208_base_cap_am": _common(local=0.548, trajectory=0.15, pcc=0.866, coverage=0.84),
    }

    _, decision = apply_s2br_decision(_seed_rows(values))

    assert decision["passing_combinations"] == ["crd_205_base_em_static"]
    assert decision["selected_model"] == "crd_205_base_em_static"
    assert decision["s3_activated"] is True
    assert decision["outcome"] == "open_s3"
