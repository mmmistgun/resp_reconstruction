from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import resp_train.paper_evidence.center_context_summary as summary
from resp_train.paper_evidence.center_context_summary import (
    ARM_SPECS,
    P3_RUN_COMMIT,
    build_length_changes,
    evaluate_upgrade_signal,
    run_p3_validation_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def _directional_fixture() -> pd.DataFrame:
    values = {
        ("C201", 60): (1.00, 1.00, 1.00, 1.00, 0.80),
        ("C201", 90): (1.02, 0.99, 0.99, 0.98, 0.801),
        ("C201", 180): (1.04, 0.98, 0.98, 0.97, 0.8015),
        ("W-reduced", 60): (1.00, 1.00, 1.00, 1.00, 0.80),
        ("W-reduced", 90): (0.99, 1.01, 1.01, 1.02, 0.803),
        ("W-reduced", 180): (0.98, 1.02, 1.02, 1.03, 0.799),
    }
    rows = []
    for (model_id, input_sec), metrics in values.items():
        row = {"model_id": model_id, "input_sec": input_sec}
        for spec, value in zip(summary.METRIC_SPECS, metrics, strict=True):
            row[f"{spec.name}_mean"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_p3_summary_matrix_and_signal_rules_are_frozen() -> None:
    assert len(ARM_SPECS) == 6
    assert {(item.model_id, item.input_sec) for item in ARM_SPECS} == {
        (model, input_sec)
        for model in ("C201", "W-reduced")
        for input_sec in (60, 90, 180)
    }
    changes = build_length_changes(_directional_fixture())
    assert len(changes) == 30
    decision = evaluate_upgrade_signal(changes)
    assert decision["upgrade_signal_detected"] is True
    assert decision["stable_context_effect_claim_allowed"] is False
    assert decision["p4_training_authorized"] is False
    interaction = decision["triggered_conditions"]["model_by_context_interaction"]
    assert {item["metric"] for item in interaction} >= {
        "center_rr_mae_bpm",
        "center_ibi_medae_sec",
    }


def test_p3_materiality_boundaries_are_exact() -> None:
    frame = _directional_fixture()
    metric = "center_envelope_trajectory_mae_mean"
    frame.loc[(frame.model_id == "C201") & (frame.input_sec == 90), metric] = 0.995
    frame.loc[(frame.model_id == "C201") & (frame.input_sec == 180), metric] = 0.995
    decision = evaluate_upgrade_signal(build_length_changes(frame))
    saturation = decision["triggered_conditions"]["candidate_90s_saturation"]
    assert any(item["model_id"] == "C201" and item["metric"] == metric.removesuffix("_mean") for item in saturation)


def test_repository_p3_runs_close_identity_hashes_and_upgrade_signal(tmp_path: Path) -> None:
    output = tmp_path / "summary"
    receipt_path = run_p3_validation_summary(
        repo_root=ROOT,
        output_dir=output,
        command="pytest read-only fixture",
        require_clean_git=False,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["execution"]["p3_run_commit"] == P3_RUN_COMMIT
    assert receipt["counts"] == {
        "expected_runs": 6,
        "actual_runs": 6,
        "epochs_per_run": 80,
        "validation_rows_per_run": 2675,
        "validation_metric_rows_read": 16050,
        "arm_summary_rows": 6,
        "length_change_rows": 30,
        "model_difference_rows": 15,
    }
    assert receipt["access"]["research_test_accessed"] is False
    assert receipt["access"]["checkpoint_content_read"] is False
    assert receipt["decision"]["upgrade_signal_detected"] is True
    assert {item["metric"] for item in receipt["decision"]["triggered_conditions"]["model_by_context_interaction"]} == {
        "center_rr_mae_bpm",
        "center_ibi_medae_sec",
        "center_envelope_trajectory_mae",
        "center_global_envelope_modulation_error",
    }
    assert set(path.name for path in output.iterdir()) == {*summary.OUTPUT_FILENAMES, "artifact_manifest.json"}


def test_existing_output_fails_before_git_or_input_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(summary, "_git_identity", lambda *_: (_ for _ in ()).throw(AssertionError("not called")))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_p3_validation_summary(repo_root=ROOT, output_dir=output, command="unused")


def test_summary_cli_rejects_all_overrides() -> None:
    script = ROOT / "scripts/summarize_paper_center_context_v1.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for forbidden in ("--output", "--run-root", "--seed", "--checkpoint", "--split", "--test"):
        assert forbidden not in result.stdout
