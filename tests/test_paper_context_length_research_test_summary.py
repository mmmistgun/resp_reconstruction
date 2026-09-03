from __future__ import annotations

from pathlib import Path

import pandas as pd

from resp_train.paper_evidence.context_length_research_test import OUTPUT_ROOT
from resp_train.paper_evidence.context_length_research_test_summary import (
    OUTPUT_FILENAMES,
    audit_research_test_evaluations,
    build_paired_primary_directions,
    build_seed_relative_changes,
    build_task_decision,
)


def _synthetic_seed_rows(task: str) -> pd.DataFrame:
    prefix = "center30" if task == "center30" else "center"
    lengths = (30, 45, 60, 90) if task == "center30" else (60, 90, 180)
    records = []
    for model_id in ("C201", "W-reduced"):
        for input_sec in lengths:
            for seed in (20260811, 20260812, 20260813):
                # RR 随长度稳定下降；其余字段仅提供完整构造输入。
                metric_base = 1.0 - input_sec / 1000.0
                records.append(
                    {
                        "task": task,
                        "experiment_id": f"{task}_{model_id}_{input_sec}",
                        "model_id": model_id,
                        "variant": model_id,
                        "input_sec": input_sec,
                        "output_sec": 30 if task == "center30" else 60,
                        "seed": seed,
                        "selected_epoch": 1,
                        "parameter_count": 1,
                        f"{prefix}_rr_mae_bpm_mean": metric_base,
                        f"{prefix}_ibi_medae_sec_mean": metric_base,
                        f"{prefix}_envelope_trajectory_mae_mean": metric_base,
                        f"{prefix}_global_envelope_modulation_error_mean": metric_base,
                        f"{prefix}_lag_aware_signed_pcc_mean": 0.5 + input_sec / 1000.0,
                        f"{prefix}_ibi_coverage_mean": 0.8,
                        f"{prefix}_ibi_interpretable_fraction": 0.7,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_generated_evaluations_close_exact_42_matrix_and_provenance() -> None:
    frames, inputs = audit_research_test_evaluations(OUTPUT_ROOT)
    assert len(inputs) == 42
    assert len(frames["center30"]) == 24
    assert len(frames["center60"]) == 18
    assert {record["evaluation_commit"] for record in inputs} == {
        "52d723891e34ad5051f18bdf93f2b0fbe4eb9d7f"
    }
    assert {record["physical_gpu"] for record in inputs} == {"0", "1"}


def test_center30_relative_changes_are_same_seed_and_materiality_aware() -> None:
    changes = build_seed_relative_changes("center30", _synthetic_seed_rows("center30"))
    paired = build_paired_primary_directions(changes)
    rr = paired.loc[
        paired["metric"].eq("center30_rr_mae_bpm")
        & paired["from_sec"].eq(30)
        & paired["to_sec"].eq(45)
    ]
    assert len(changes) == 150
    assert len(rr) == 2
    assert (rr["materially_improved_seed_count"] == 3).all()


def test_task_decisions_keep_center30_and_center60_claims_distinct() -> None:
    center30_changes = build_seed_relative_changes("center30", _synthetic_seed_rows("center30"))
    center60_changes = build_seed_relative_changes("center60", _synthetic_seed_rows("center60"))
    center30 = build_task_decision("center30", center30_changes)
    center60 = build_task_decision("center60", center60_changes)
    assert center30["validation_claim"] == "45s_is_rr_priority_reasonable_input_lower_bound"
    assert center60["validation_claim"] == "longer_input_has_no_stable_rr_benefit"
    assert center30["cross_task_absolute_comparison_performed"] is False
    assert center60["cross_task_absolute_comparison_performed"] is False


def test_summary_output_contract_has_two_separate_panels() -> None:
    assert len(OUTPUT_FILENAMES) == 16
    assert sum(name.startswith("center30_") for name in OUTPUT_FILENAMES) == 7
    assert sum(name.startswith("center60_") for name in OUTPUT_FILENAMES) == 7
    assert "joint_decision.json" in OUTPUT_FILENAMES
    assert "summary_receipt.json" in OUTPUT_FILENAMES
