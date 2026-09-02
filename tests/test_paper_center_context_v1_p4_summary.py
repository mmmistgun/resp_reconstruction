from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resp_train.paper_evidence.center_context_p4_summary as summary
from resp_train.paper_evidence.center_context_p4_summary import (
    P4_RUN_COMMIT,
    build_paired_seed_directions,
    run_p4_validation_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paired_seed_directions_use_arithmetic_mean_sample_sd_and_materiality() -> None:
    frame = pd.DataFrame(
        {
            "seed": [20260811, 20260812, 20260813],
            "model_id": ["C201"] * 3,
            "from_sec": [60] * 3,
            "to_sec": [180] * 3,
            "metric": ["center_rr_mae_bpm"] * 3,
            "oriented_improvement": [0.01, -0.006, 0.001],
        }
    )
    result = build_paired_seed_directions(
        frame,
        group_columns=("model_id", "from_sec", "to_sec", "metric"),
        improvement_column="oriented_improvement",
    ).iloc[0]
    assert result["improvement_seed_mean"] == pytest.approx(np.mean([0.01, -0.006, 0.001]))
    assert result["improvement_seed_sample_sd"] == pytest.approx(np.std([0.01, -0.006, 0.001], ddof=1))
    assert (result["improved_seed_count"], result["worsened_seed_count"]) == (2, 1)
    assert (result["materially_improved_seed_count"], result["materially_worsened_seed_count"]) == (1, 1)
    assert result["within_materiality_seed_count"] == 1


def test_repository_p4_summary_closes_18_runs_without_automatic_stability_claim(tmp_path: Path) -> None:
    output = tmp_path / "p4_summary"
    receipt_path = run_p4_validation_summary(
        repo_root=ROOT,
        output_dir=output,
        command="pytest read-only P4 summary",
        require_clean_git=False,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["execution"]["p4_run_commit"] == P4_RUN_COMMIT
    assert receipt["counts"] == {
        "expected_runs": 18,
        "actual_runs": 18,
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
    assert receipt["decision"]["automatic_stability_gate_preregistered"] is False
    assert receipt["decision"]["automatic_stable_effect_claim_allowed"] is False
    assert receipt["access"]["checkpoint_content_read"] is False
    assert receipt["access"]["research_test_accessed"] is False
    assert set(path.name for path in output.iterdir()) == {*summary.OUTPUT_FILENAMES, "artifact_manifest.json"}


def test_existing_output_fails_before_git_or_input_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(summary, "_git_identity", lambda *_: (_ for _ in ()).throw(AssertionError("not called")))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_p4_validation_summary(repo_root=ROOT, output_dir=output, command="unused")


def test_p4_summary_cli_has_no_result_or_access_overrides() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_paper_center_context_p4_v1.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for forbidden in ("--output", "--run-root", "--seed", "--checkpoint", "--split", "--test"):
        assert forbidden not in result.stdout
