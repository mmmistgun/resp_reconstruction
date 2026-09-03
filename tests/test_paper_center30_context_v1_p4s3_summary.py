from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resp_train.paper_evidence.center30_p4s3_summary as summary
from resp_train.paper_evidence.center30_p4s3_summary import (
    P4S3_RUN_COMMIT,
    build_paired_primary_directions,
    run_p4s3_validation_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paired_primary_directions_use_arithmetic_mean_sample_sd_and_materiality() -> None:
    frame = pd.DataFrame(
        {
            "seed": [20260811, 20260812, 20260813],
            "model_id": ["C201"] * 3,
            "from_sec": [30] * 3,
            "to_sec": [60] * 3,
            "metric": ["center30_rr_mae_bpm"] * 3,
            "direction": ["minimize"] * 3,
            "material_kind": ["relative"] * 3,
            "material_threshold": [0.005] * 3,
            "oriented_relative_change": [0.01, -0.006, 0.001],
            "oriented_absolute_change": [0.02, -0.012, 0.002],
            "protocol_material_value": [0.01, -0.006, 0.001],
        }
    )
    result = build_paired_primary_directions(frame).iloc[0]
    assert result["oriented_relative_change_seed_mean"] == pytest.approx(np.mean([0.01, -0.006, 0.001]))
    assert result["oriented_relative_change_seed_sample_sd"] == pytest.approx(
        np.std([0.01, -0.006, 0.001], ddof=1)
    )
    assert (result["improved_seed_count"], result["worsened_seed_count"]) == (2, 1)
    assert (result["materially_improved_seed_count"], result["materially_worsened_seed_count"]) == (1, 1)
    assert result["within_materiality_seed_count"] == 1


def test_repository_p4s3_summary_closes_24_runs_without_test_or_checkpoint_access(tmp_path: Path) -> None:
    output = tmp_path / "p4s3_summary"
    receipt_path = run_p4s3_validation_summary(
        repo_root=ROOT,
        output_dir=output,
        command="pytest read-only P4-S3 summary",
        require_clean_git=False,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["execution"]["p4s3_run_commit"] == P4S3_RUN_COMMIT
    assert receipt["counts"] == {
        "expected_runs": 24,
        "actual_runs": 24,
        "validation_rows_per_run": 2675,
        "validation_metric_rows_read": 64200,
        "seed_arm_rows": 24,
        "three_seed_arm_rows": 8,
        "seed_relative_change_rows": 150,
        "paired_relative_direction_rows": 50,
        "seed_companion_change_rows": 60,
        "paired_companion_direction_rows": 20,
        "seed_model_difference_rows": 60,
        "paired_model_direction_rows": 20,
    }
    assert receipt["aggregation"]["seed_sd"] == "sample_sd_ddof1"
    assert receipt["decision"]["automatic_stable_minimum_context_claim_allowed"] is False
    assert set(receipt["decision"]["rr_best_input_sec_by_model_and_seed"]) == {"C201", "W-reduced"}
    assert receipt["access"]["checkpoint_content_read"] is False
    assert receipt["access"]["research_test_accessed"] is False
    assert set(path.name for path in output.iterdir()) == {*summary.OUTPUT_FILENAMES, "artifact_manifest.json"}


def test_existing_output_fails_before_git_or_input_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(summary, "_git_identity", lambda *_: (_ for _ in ()).throw(AssertionError("not called")))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_p4s3_validation_summary(repo_root=ROOT, output_dir=output, command="unused")


def test_p4s3_summary_cli_has_no_result_or_access_overrides() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_paper_center30_context_p4s3_v1.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for forbidden in ("--output", "--run-root", "--seed", "--checkpoint", "--split", "--test"):
        assert forbidden not in result.stdout
