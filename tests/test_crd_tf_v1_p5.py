from __future__ import annotations

import pandas as pd

from resp_train.crd.tf_v1_model import TF_VARIANTS
from resp_train.crd.tf_v1_p5 import (
    _aggregate_metrics,
    _candidate_sets,
    _eligibility_tables,
    _interaction_table,
    _pair_variant,
)
from resp_train.crd.tf_v1_selection import PRIMARY_METRICS


def _synthetic_seed_metrics() -> pd.DataFrame:
    rows = []
    variants = ("tf000_c201_anchor", *TF_VARIANTS)
    for variant in variants:
        for seed in (20260811, 20260812, 20260813):
            error = 1.0
            pcc = 0.5
            if variant == "crd_tf101_m":
                error = 0.99
                pcc = 0.503
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "whole_rr_mae": error,
                    "local_rr_mae": error,
                    "trajectory_mae": error,
                    "global_envelope_error": error,
                    "signed_pcc": pcc,
                }
            )
    return pd.DataFrame(rows)


def test_pair_mapping_is_order_independent() -> None:
    assert _pair_variant("m", "l") == "crd_tf202_ml"
    assert _pair_variant("l", "m") == "crd_tf202_ml"
    assert _pair_variant("w", "s") == "crd_tf205_ws"


def test_p5_tables_keep_metrics_separate_and_never_construct_total_score() -> None:
    seed_metrics = _synthetic_seed_metrics()
    aggregate = _aggregate_metrics(seed_metrics)
    interactions = _interaction_table(seed_metrics)
    eligibility, capacity = _eligibility_tables(seed_metrics, aggregate, interactions)
    candidates = _candidate_sets(aggregate, eligibility)

    assert len(aggregate) == 16 * len(PRIMARY_METRICS)
    assert len(interactions) == 8 * len(PRIMARY_METRICS)
    assert len(capacity) == 12 * len(PRIMARY_METRICS)
    assert len(eligibility) == 12
    assert "total_score" not in aggregate.columns
    assert "total_score" not in interactions.columns
    assert "crd_tf101_m" in eligibility.loc[eligibility["qualified"], "variant"].tolist()
    assert "crd_tf101_m" in candidates["single_pareto"]
