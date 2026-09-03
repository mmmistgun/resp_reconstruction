from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import resp_train.paper_evidence.center30_summary as summary
from resp_train.paper_evidence.center30_summary import (
    P4S_RUN_COMMIT,
    build_relative_changes,
    run_p4s_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> pd.DataFrame:
    rows = []
    for model_id, offset in (("C201", 0.0), ("W-reduced", 0.1)):
        for input_sec, value in ((30, 1.0), (45, 0.9), (60, 0.8), (90, 0.85)):
            row = {"model_id": model_id, "input_sec": input_sec}
            for spec in summary.METRIC_SPECS:
                row[f"{spec.name}_mean"] = (0.8 + offset if spec.direction == "maximize" else value + offset)
            rows.append(row)
    return pd.DataFrame(rows)


def test_relative_changes_use_oriented_baseline_and_frozen_contrasts() -> None:
    changes = build_relative_changes(_fixture())
    assert len(changes) == 50
    rr = changes.loc[
        changes["model_id"].eq("C201")
        & changes["from_sec"].eq(30)
        & changes["to_sec"].eq(60)
        & changes["metric"].eq("center30_rr_mae_bpm")
    ].iloc[0]
    assert rr["oriented_relative_change"] == pytest.approx(0.2)
    assert bool(rr["materially_improved"])


def test_repository_p4s_summary_closes_eight_runs_and_freezes_relative_signal(tmp_path: Path) -> None:
    output = tmp_path / "summary"
    receipt_path = run_p4s_summary(
        repo_root=ROOT,
        output_dir=output,
        command="pytest read-only P4-S2 summary",
        require_clean_git=False,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["execution"]["run_commit"] == P4S_RUN_COMMIT
    assert receipt["counts"] == {
        "expected_runs": 8,
        "actual_runs": 8,
        "epochs_per_run": 80,
        "validation_rows_per_run": 2675,
        "validation_metric_rows_read": 21400,
        "relative_change_rows": 50,
        "companion_change_rows": 20,
        "model_difference_rows": 20,
    }
    decision = receipt["decision"]
    assert decision["all_models_all_longer_inputs_materially_improve_rr_vs_30s"] is True
    assert decision["best_rr_input_sec_by_model"] == {"C201": 60, "W-reduced": 45}
    assert decision["stable_minimum_context_claim_allowed"] is False
    assert decision["p4s_three_seed_extension_authorized"] is False
    assert receipt["access"]["checkpoint_content_read"] is False
    assert receipt["access"]["research_test_accessed"] is False
    assert set(path.name for path in output.iterdir()) == {*summary.OUTPUT_FILENAMES, "artifact_manifest.json"}


def test_existing_output_fails_before_git_or_input_access(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(summary, "_git_identity", lambda *_: (_ for _ in ()).throw(AssertionError("not called")))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_p4s_summary(repo_root=ROOT, output_dir=output, command="unused")


def test_summary_cli_has_no_result_or_access_overrides() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_paper_center30_context_v1.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for forbidden in ("--output", "--run-root", "--seed", "--checkpoint", "--split", "--test"):
        assert forbidden not in result.stdout
