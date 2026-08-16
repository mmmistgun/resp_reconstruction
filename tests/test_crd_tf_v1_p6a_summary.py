from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from resp_train.crd.tf_v1_p6a_summary import (
    REQUIRED_COMPARISONS,
    _audit_early_stopping_history,
    _eligibility_and_decision,
)
from resp_train.crd.tf_v1_selection import PRIMARY_METRICS


def _aggregate_rows() -> pd.DataFrame:
    values = {
        "tf000_c201_anchor": {
            "whole_rr_mae": 1.0,
            "local_rr_mae": 1.0,
            "trajectory_mae": 1.0,
            "global_envelope_error": 1.0,
            "signed_pcc": 0.8,
        },
        "crd_tf401_mws_add": {
            "whole_rr_mae": 0.9,
            "local_rr_mae": 1.0,
            "trajectory_mae": 1.0,
            "global_envelope_error": 1.0,
            "signed_pcc": 0.79,
        },
        "crd_tf402_mws_gate": {
            "whole_rr_mae": 0.9,
            "local_rr_mae": 0.99,
            "trajectory_mae": 0.99,
            "global_envelope_error": 0.99,
            "signed_pcc": 0.8,
        },
    }
    return pd.DataFrame(
        [
            {"variant": variant, "metric": metric, "mean": value}
            for variant, metrics in values.items()
            for metric, value in metrics.items()
        ]
    )


def _comparison_rows(passed_names: set[str]) -> pd.DataFrame:
    rows = []
    for required in REQUIRED_COMPARISONS.values():
        for name in required:
            for metric in PRIMARY_METRICS:
                rows.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "passed": name in passed_names and metric == "local_rr_mae",
                    }
                )
    return pd.DataFrame(rows)


def test_p6a_decision_requires_guardrails_and_every_arm_specific_comparison() -> None:
    all_required = {name for names in REQUIRED_COMPARISONS.values() for name in names}
    eligibility, decision = _eligibility_and_decision(_aggregate_rows(), _comparison_rows(all_required))

    add = eligibility.loc[eligibility["variant"] == "crd_tf401_mws_add"].iloc[0]
    gate = eligibility.loc[eligibility["variant"] == "crd_tf402_mws_gate"].iloc[0]
    assert not bool(add["base_guardrails_passed"])
    assert not bool(add["qualified"])
    assert bool(gate["qualified"])
    assert decision["decision"] == "advance_p6a_candidate"
    assert decision["selected_for_future_lock"] == ["crd_tf402_mws_gate"]


def test_p6a_decision_retains_p5_pool_when_gate_mechanism_comparison_fails() -> None:
    passed = {"mws_add_vs_ms", "mws_add_vs_ctrl3", "mws_gate_vs_ctrl_gate"}
    _, decision = _eligibility_and_decision(_aggregate_rows(), _comparison_rows(passed))

    assert decision["decision"] == "no_p6a_candidate_retain_p5_pool"
    assert decision["selected_for_future_lock"] == []
    assert decision["fallback_if_none"] == ["crd_tf101_m", "crd_tf102_w", "crd_tf203_ms"]


def test_p6a_history_audit_accepts_exact_patience_thirty_stop() -> None:
    history = pd.DataFrame(
        {
            "epoch": list(range(1, 32)),
            "optimizer_update": [epoch * 80 for epoch in range(1, 32)],
            "val_local_rr_mae": [0.5] * 31,
            "early_stopping_improved": [1] + [0] * 30,
            "early_stopping_wait": list(range(31)),
            "early_stopping_triggered": [0] * 30 + [1],
        }
    )

    assert _audit_early_stopping_history(history, Path("synthetic")) == (31, 1, True)


def test_p6a_history_audit_rejects_early_trigger() -> None:
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "optimizer_update": [80, 160],
            "val_local_rr_mae": [0.5, 0.5],
            "early_stopping_improved": [1, 0],
            "early_stopping_wait": [0, 1],
            "early_stopping_triggered": [0, 1],
        }
    )

    with pytest.raises(RuntimeError, match="early-stop history"):
        _audit_early_stopping_history(history, Path("synthetic"))

