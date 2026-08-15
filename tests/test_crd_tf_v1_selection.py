from __future__ import annotations

import numpy as np
import pytest

from resp_train.crd.tf_v1_model import TF_VARIANTS
from resp_train.crd.tf_v1_selection import (
    EXPECTED_SEEDS,
    interaction_summary,
    pair_interaction,
    paired_material_improvement,
    passes_base_guardrails,
    tolerance_dominates,
    tolerance_pareto_set,
    triple_interaction,
    validate_formal_matrix,
)


def _metrics(**overrides: float) -> dict[str, float]:
    values = {
        "whole_rr_mae": 1.0,
        "local_rr_mae": 1.0,
        "trajectory_mae": 1.0,
        "global_envelope_error": 1.0,
        "signed_pcc": 0.50,
    }
    values.update(overrides)
    return values


def test_pair_and_triple_interactions_are_computed_per_paired_seed() -> None:
    base = [10.0, 10.0, 10.0]
    a = [9.0, 9.0, 9.0]
    b = [8.0, 8.0, 8.0]
    c = [7.0, 7.0, 7.0]
    ab = [6.5, 6.5, 6.5]
    ac = [5.5, 5.5, 5.5]
    bc = [4.5, 4.5, 4.5]
    abc = [2.0, 2.0, 2.0]

    pair = pair_interaction("local_rr_mae", base=base, arm_a=a, arm_b=b, pair=ab)
    triple = triple_interaction(
        "local_rr_mae",
        base=base,
        arm_a=a,
        arm_b=b,
        arm_c=c,
        pair_ab=ab,
        pair_ac=ac,
        pair_bc=bc,
        triple=abc,
    )

    np.testing.assert_allclose(pair, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(triple, [0.5, 0.5, 0.5])
    assert interaction_summary([0.1, -0.01, 0.2])["descriptive_positive"] is True
    assert interaction_summary([0.1, -0.01, -0.02])["descriptive_positive"] is False


def test_guardrail_and_tolerance_aware_pareto_logic() -> None:
    base = _metrics()
    eligible = _metrics(local_rr_mae=1.014, trajectory_mae=1.014, signed_pcc=0.495)
    failed = _metrics(local_rr_mae=1.016)
    assert passes_base_guardrails(eligible, base)
    assert not passes_base_guardrails(failed, base)

    better = _metrics(whole_rr_mae=0.99)
    close = _metrics()
    assert tolerance_dominates(better, close)
    assert not tolerance_dominates(close, better)
    assert tolerance_pareto_set({"better": better, "close": close}) == ("better",)


def test_formal_matrix_completeness_gate_is_strict() -> None:
    complete = {variant: list(EXPECTED_SEEDS) for variant in TF_VARIANTS}
    validate_formal_matrix(complete)
    incomplete = dict(complete)
    incomplete[TF_VARIANTS[0]] = list(EXPECTED_SEEDS[:-1])
    with pytest.raises(RuntimeError, match="seeds 不完整"):
        validate_formal_matrix(incomplete)


def test_paired_capacity_improvement_requires_threshold_and_two_seed_direction() -> None:
    passed = paired_material_improvement(
        "local_rr_mae",
        candidate=[0.99, 0.98, 1.01],
        reference=[1.0, 1.0, 1.0],
    )
    threshold_only = paired_material_improvement(
        "signed_pcc",
        candidate=[0.504, 0.499, 0.499],
        reference=[0.5, 0.5, 0.5],
    )

    assert passed["threshold_passed"] is True
    assert passed["positive_seed_count"] == 2
    assert passed["passed"] is True
    assert threshold_only["positive_seed_count"] == 1
    assert threshold_only["passed"] is False
