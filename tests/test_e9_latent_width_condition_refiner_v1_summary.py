from __future__ import annotations

import pandas as pd
import pytest

from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1 as e9
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_summary as summary
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import ARMS


def _seed_frame() -> pd.DataFrame:
    utilities = {
        "e9a_d96_h65": 0.0,
        "e9a_d96_h64": 1.0,
        "e9a_d96_h48": 2.0,
        "e9b_d64_direct": 3.0,
        "e9b_d64_h64": 4.0,
        "e9b_d64_h48": 5.0,
    }
    return pd.DataFrame(
        [
            {
                "arm": arm,
                "seed": seed,
                **{metric: 100.0 - utilities[arm] for metric in summary.ERRORS},
                summary.PCC: 0.5 + utilities[arm] / 100.0,
            }
            for seed in e9.SEEDS
            for arm in ARMS
        ]
    )


def test_planned_contrasts_are_exact_and_complete():
    frame = _seed_frame()
    by_seed = summary.planned_contrasts_by_seed(frame)
    across = summary.planned_contrasts_across_seed(by_seed)
    assert len(by_seed) == len(summary.CONTRASTS) * 3 * 5
    assert len(across) == len(summary.CONTRASTS) * 5
    selected = across.loc[
        across.contrast.eq("h64_minus_h65")
    ].set_index("metric")
    assert selected.loc[summary.ERRORS[0], "oriented_benefit_mean"] == pytest.approx(1.0)
    assert selected.loc[summary.PCC, "oriented_benefit_mean"] == pytest.approx(0.01)


def test_summary_decision_and_pareto_are_complete():
    frame = _seed_frame()
    contrasts = summary.planned_contrasts_across_seed(
        summary.planned_contrasts_by_seed(frame)
    )
    decision = summary.build_decision(frame, contrasts)
    assert decision["e9a"]["preferred_anchor"] == "h64"
    assert decision["e9b"]["preferred_structure"] == "h64"
    assert set(decision["tolerance_aware_pareto"]["pareto_arms"]).issubset(set(ARMS))


def test_historical_e8_reference_has_four_pointwise_arms():
    frame = summary.historical_e8_pointwise_reference()
    assert len(frame) == 4
    assert set(frame.historical_arm) == {
        "e8_direct_pointwise",
        "e8_fill65_pointwise",
        "e8_res96_pointwise",
        "e8_res192_pointwise",
    }


def test_summary_lifecycle_rejects_duplicate_completed_attempt(tmp_path):
    lock_hash = "c" * 64
    amendment_hash = "d" * 64
    parent = tmp_path / "summary"
    with summary._summary_attempt(
        parent,
        lock_hash=lock_hash,
        amendment_hash=amendment_hash,
    ) as output:
        (output / "result.txt").write_text("ok", encoding="utf-8")
    with pytest.raises(FileExistsError, match="已完成"):
        with summary._summary_attempt(
            parent,
            lock_hash=lock_hash,
            amendment_hash=amendment_hash,
        ):
            pass

