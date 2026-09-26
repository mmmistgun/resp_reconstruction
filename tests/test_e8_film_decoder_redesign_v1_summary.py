from __future__ import annotations

import pandas as pd
import pytest

from resp_train.paper_evidence import e8_film_decoder_redesign_v1 as e8
from resp_train.paper_evidence import e8_film_decoder_redesign_v1_summary as summary
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_model import ARMS, ARM_SPECS


def _e8_seed_frame() -> pd.DataFrame:
    condition_value = {"fill65": 0.0, "direct": 1.0, "res96": 3.0, "res192": 6.0}
    decoder_value = {"pointwise": 0.0, "single": 2.0, "temporal": 5.0}
    rows = []
    for seed in e8.SEEDS:
        for arm, spec in ARM_SPECS.items():
            value = condition_value[spec.condition_refiner] + decoder_value[spec.decoder]
            interaction = (
                7.0
                if spec.condition_refiner == "direct" and spec.decoder == "temporal"
                else 0.0
            )
            utility = value + interaction
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    **{metric: 100.0 - utility for metric in summary.ERRORS},
                    summary.PCC: 0.5 + utility / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_e8_planned_contrasts_and_interaction_are_exact():
    frame = _e8_seed_frame()
    by_seed = summary.planned_contrasts_by_seed(frame)
    across = summary.planned_contrasts_across_seed(by_seed)
    assert len(by_seed) == 3 * 5 * 33
    assert len(across) == 5 * 33
    selected = across.loc[
        across.contrast_type.eq("interaction")
        & across.contrast.eq("direct_x_temporal")
    ].set_index("metric")
    assert selected.loc[summary.ERRORS[0], "oriented_benefit_mean"] == pytest.approx(7.0)
    assert selected.loc[summary.PCC, "oriented_benefit_mean"] == pytest.approx(0.07)
    direct = across.loc[
        across.contrast_type.eq("condition_marginal")
        & across.contrast.eq("direct_vs_fill65")
        & across.metric.eq(summary.ERRORS[0])
    ].iloc[0]
    assert direct.oriented_benefit_mean == pytest.approx(1.0 + 7.0 / 3.0)


def test_e8_arm_summary_reference_and_pareto_are_complete():
    frame = _e8_seed_frame()
    arm_summary = summary.arm_primary_summary(frame)
    comparison = summary.arm_reference_comparison(frame)
    pareto = summary.tolerance_pareto(frame)
    assert len(arm_summary) == 12 * 5
    assert len(comparison) == 11 * 5
    assert set(pareto["pareto_arms"]).issubset(set(ARMS))
    recommended = comparison.loc[comparison.arm.eq("e8_direct_temporal")]
    assert len(recommended) == 5
    assert (recommended.reference_normalized_benefit_seed_mean > 0).all()


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
