from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resp_train.paper_evidence.center90_summary as summary
from resp_train.paper_evidence.center90_summary import (
    FORMAL_RUN_COMMIT,
    build_paired_directions,
    run_center90_validation_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paired_directions_report_relative_absolute_and_materiality() -> None:
    frame = pd.DataFrame(
        {
            "seed": [20260811, 20260812, 20260813],
            "model_id": ["C201"] * 3,
            "from_sec": [90] * 3,
            "to_sec": [180] * 3,
            "metric": ["center90_rr_mae_bpm"] * 3,
            "direction": ["minimize"] * 3,
            "material_kind": ["relative"] * 3,
            "material_threshold": [0.005] * 3,
            "oriented_relative_change": [0.01, -0.006, 0.001],
            "oriented_absolute_change": [0.02, -0.012, 0.002],
            "protocol_material_value": [0.01, -0.006, 0.001],
        }
    )
    row = build_paired_directions(
        frame,
        group_columns=("model_id", "from_sec", "to_sec", "metric"),
        relative_column="oriented_relative_change",
        absolute_column="oriented_absolute_change",
    ).iloc[0]
    assert row["oriented_relative_change_seed_mean"] == pytest.approx(np.mean([0.01, -0.006, 0.001]))
    assert row["oriented_absolute_change_seed_sample_sd"] == pytest.approx(np.std([0.02, -0.012, 0.002], ddof=1))
    assert (row["improved_seed_count"], row["worsened_seed_count"]) == (2, 1)
    assert (row["materially_improved_seed_count"], row["materially_worsened_seed_count"]) == (1, 1)


def test_repository_summary_closes_all_18_runs_read_only(tmp_path: Path) -> None:
    output = tmp_path / "center90_summary"
    receipt_path = run_center90_validation_summary(
        repo_root=ROOT,
        output_dir=output,
        command="pytest read-only center90 summary",
        require_clean_git=False,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["execution"]["formal_run_commit"] == FORMAL_RUN_COMMIT
    assert receipt["counts"] == {
        "expected_runs": 18,
        "actual_runs": 18,
        "epochs_per_run": 80,
        "validation_rows_per_run": 2675,
        "validation_metric_rows_read": 48150,
        "seed_arm_rows": 18,
        "three_seed_arm_rows": 6,
        "seed_length_change_rows": 90,
        "paired_length_direction_rows": 30,
        "seed_model_difference_rows": 45,
        "paired_model_direction_rows": 15,
    }
    assert receipt["aggregation"]["seed_sd"] == "sample_sd_ddof1"
    assert receipt["decision"]["model_or_length_selection_performed"] is False
    assert receipt["access"]["checkpoint_content_read"] is False
    assert receipt["access"]["research_test_accessed"] is False
    assert set(path.name for path in output.iterdir()) == {*summary.OUTPUT_FILENAMES, "artifact_manifest.json"}


def test_existing_output_fails_before_git_or_input_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(summary, "_git_identity", lambda *_: (_ for _ in ()).throw(AssertionError("not called")))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_center90_validation_summary(repo_root=ROOT, output_dir=output, command="unused")


def test_summary_cli_has_no_result_or_access_overrides() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_paper_center90_context_v1.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for forbidden in ("--output", "--run-root", "--seed", "--checkpoint", "--split", "--test"):
        assert forbidden not in result.stdout
