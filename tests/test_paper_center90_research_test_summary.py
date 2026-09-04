from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import resp_train.paper_evidence.center90_research_test_summary as summary
from resp_train.paper_evidence.center90_research_test_summary import (
    EVALUATION_COMMIT,
    run_research_test_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_summary_closes_all_18_evaluations_read_only(tmp_path: Path) -> None:
    output = tmp_path / "center90_test_summary"
    receipt_path = run_research_test_summary(
        repo_root=ROOT,
        output_dir=output,
        command="pytest read-only center90 research-test summary",
        require_clean_git=False,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["execution"]["evaluation_commit"] == EVALUATION_COMMIT
    assert receipt["counts"] == {
        "expected_evaluations": 18,
        "actual_evaluations": 18,
        "test_rows_per_evaluation": 2310,
        "test_metric_rows_read": 41580,
        "seed_arm_rows": 18,
        "three_seed_arm_rows": 6,
        "seed_length_change_rows": 90,
        "paired_length_direction_rows": 30,
        "seed_model_difference_rows": 45,
        "paired_model_direction_rows": 15,
    }
    assert receipt["aggregation"]["seed_sd"] == "sample_sd_ddof1"
    assert receipt["decision"]["checkpoint_model_or_length_reselection_performed"] is False
    assert receipt["access"]["checkpoint_content_read"] is False
    assert receipt["access"]["model_inference_used"] is False
    assert set(path.name for path in output.iterdir()) == {*summary.OUTPUT_FILENAMES, "artifact_manifest.json"}


def test_existing_output_fails_before_git_or_input_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(summary, "_git_identity", lambda *_: (_ for _ in ()).throw(AssertionError("not called")))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_research_test_summary(repo_root=ROOT, output_dir=output, command="unused")


def test_summary_cli_has_no_result_or_access_overrides() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_paper_center90_research_test_v1.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for forbidden in ("--output", "--run-root", "--seed", "--checkpoint", "--split", "--test"):
        assert forbidden not in result.stdout
