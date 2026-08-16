from __future__ import annotations

import pandas as pd

from resp_train.crd.tf_v1_research_test_summary import (
    BASE_VARIANT,
    CANDIDATES,
    _aggregate,
    _paired_vs_base,
    _selection,
)
from resp_train.crd.tf_v1_selection import PRIMARY_METRICS


def _seed_metrics() -> pd.DataFrame:
    means = {
        BASE_VARIANT: [1.00, 1.00, 1.00, 1.00, 0.800],
        "crd_tf101_m": [0.99, 0.99, 0.99, 0.99, 0.797],
        "crd_tf102_w": [0.80, 0.80, 0.80, 1.03, 0.801],
        "crd_tf203_ms": [0.84, 0.84, 0.84, 0.98, 0.799],
    }
    rows = []
    for variant, values in means.items():
        for seed in (20260811, 20260812, 20260813):
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    **dict(zip(PRIMARY_METRICS, values)),
                }
            )
    return pd.DataFrame(rows)


def test_research_test_selection_reports_pareto_and_local_rr_lead_without_total_score() -> None:
    seed_metrics = _seed_metrics()
    aggregate = _aggregate(seed_metrics, PRIMARY_METRICS)
    comparisons = _paired_vs_base(seed_metrics)

    eligibility, decision = _selection(aggregate, comparisons)

    assert set(eligibility.loc[eligibility["qualified"], "variant"]) == set(CANDIDATES)
    assert decision["tolerance_pareto"] == ["crd_tf102_w", "crd_tf203_ms"]
    assert decision["local_rr_lead"] == "crd_tf102_w"
    assert decision["unique_winner_selected"] is False
    assert decision["decision"] == "retain_research_test_pareto_without_total_score"


def test_research_test_aggregate_uses_sample_sd_and_frozen_seed_order() -> None:
    frame = _seed_metrics()
    frame.loc[
        (frame["variant"] == "crd_tf101_m") & (frame["seed"] == 20260813), "local_rr_mae"
    ] = 1.02

    aggregate = _aggregate(frame, ("local_rr_mae",))
    row = aggregate.loc[aggregate["variant"] == "crd_tf101_m"].iloc[0]

    assert row["mean"] == (0.99 + 0.99 + 1.02) / 3
    assert row["sample_sd"] > 0.0
    assert row["seed_20260811"] == 0.99
    assert row["seed_20260813"] == 1.02

