from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.paper_evidence.center90_metrics import summarize_center90_metrics
from resp_train.paper_evidence.center90_research_test import (
    EVALUATION_SCHEMA_VERSION,
    EXPECTED_RUNTIME,
    FORMAL_SEEDS,
    OUTPUT_ROOT,
    SUPPLEMENT_CACHE_MANIFEST_SHA256,
    SUPPLEMENT_CACHE_ROOT,
    _cache_hash,
    _cache_root,
    _validate_metrics,
    expected_research_test_evaluations,
)
from resp_train.paper_evidence.center90_research_test_cache import (
    EXPECTED_DATASET_INDEX_SHA256,
    EXPECTED_TEST_ROW_IDS_SHA256,
    EXPECTED_TEST_SAMP_IDS,
    EXPECTED_TEST_WINDOWS,
    INPUT_KEY,
    PROTOCOL_ID,
    TARGET_KEY,
    TEST_SAMPLE_SEED,
    TEST_SAMPLE_STRATEGY,
    sha256_file,
)
from resp_train.paper_evidence.center90_summary import (
    ARM_SPECS,
    METRIC_SPECS,
    SUMMARY_COLUMNS,
    build_paired_directions,
    build_seed_length_changes,
    build_seed_model_differences,
    build_three_seed_arm_summary,
)


SUMMARY_SCHEMA_VERSION = "paper-center90-context-research-test-summary-v1"
EVALUATION_COMMIT = "9a48dc39872dcf20b6514e69fd5d7360a62f07a7"
DEFAULT_OUTPUT_DIR = Path("runs/paper_evidence_v1/center90_context_research_test_summary")
REQUIRED_ARTIFACTS = {
    "checkpoint_identity.json",
    "evaluation_receipt.json",
    "research_test_metrics.csv",
    "research_test_metrics_summary.csv",
    "resolved_evaluation_config.yaml",
    "runtime_identity.json",
    "test_data_identity.json",
}
OUTPUT_FILENAMES = (
    "test_seed_arm_summary.csv",
    "test_arm_three_seed_summary.csv",
    "test_seed_length_changes.csv",
    "test_paired_seed_length_directions.csv",
    "test_seed_model_differences.csv",
    "test_paired_seed_model_directions.csv",
    "test_decision.json",
    "summary_receipt.json",
)


def audit_evaluations(run_root: str | Path = OUTPUT_ROOT) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = Path(run_root).resolve()
    allowlist = expected_research_test_evaluations()
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None
    for key in sorted(allowlist):
        spec = allowlist[key]
        evaluation_dir = root / spec.experiment_id / f"seed_{spec.seed}"
        row, input_record, identity = _audit_evaluation(evaluation_dir, spec)
        if reference_identity is None:
            reference_identity = identity
        elif not identity.equals(reference_identity):
            raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} test identity 不一致")
        rows.append(row)
        inputs.append(input_record)
    frame = pd.DataFrame.from_records(rows).sort_values(["model_id", "input_sec", "seed"])
    if len(frame) != 18 or frame[["experiment_id", "seed"]].duplicated().any():
        raise RuntimeError("center90 research-test 必须恰好闭合 6 arms × 3 seeds")
    return frame, inputs


def build_test_decision(seed_rows: pd.DataFrame, paired_length: pd.DataFrame) -> dict[str, Any]:
    rr = paired_length.loc[paired_length["metric"].eq("center90_rr_mae_bpm")]
    best_inputs: dict[str, dict[str, int]] = {}
    for model_id in ("C201", "W-reduced"):
        per_seed: dict[str, int] = {}
        for seed in FORMAL_SEEDS:
            group = seed_rows.loc[seed_rows["model_id"].eq(model_id) & seed_rows["seed"].eq(seed)]
            per_seed[str(seed)] = int(group.loc[group["center90_rr_mae_bpm_mean"].idxmin(), "input_sec"])
        best_inputs[model_id] = per_seed
    return {
        "evidence_scope": "three-seed_independent_test_center90_context_sensitivity",
        "rr_paired_length_directions": rr.to_dict(orient="records"),
        "rr_best_input_sec_by_model_and_seed": best_inputs,
        "validation_claim_under_test": "center90_context_effect_is_weak_and_representation_dependent",
        "reporting_disposition": "report_mean_sample_sd_and_paired_seed_directions",
        "weighted_score_used": False,
        "checkpoint_model_or_length_reselection_performed": False,
        "cross_output_absolute_metric_comparison_allowed": False,
    }


def run_research_test_summary(
    *,
    repo_root: str | Path,
    command: str,
    run_root: str | Path = OUTPUT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    require_clean_git: bool = True,
) -> Path:
    root = Path(repo_root).resolve()
    source = _resolve(root, run_root)
    output = _resolve(root, output_dir)
    if output.exists():
        raise FileExistsError(f"center90 research-test 汇总输出禁止覆盖: {output}")
    git = _git_identity(root)
    if require_clean_git and (git["error"] is not None or git["dirty"] is not False):
        raise RuntimeError("center90 research-test 冻结汇总要求干净 Git 工作树")
    seed_rows, inputs = audit_evaluations(source)
    arm_summary = build_three_seed_arm_summary(seed_rows)
    seed_length = build_seed_length_changes(seed_rows)
    paired_length = build_paired_directions(
        seed_length,
        group_columns=("model_id", "from_sec", "to_sec", "metric"),
        relative_column="oriented_relative_change",
        absolute_column="oriented_absolute_change",
    )
    seed_models = build_seed_model_differences(seed_rows)
    paired_models = build_paired_directions(
        seed_models,
        group_columns=("input_sec", "metric"),
        relative_column="w_reduced_oriented_relative_change",
        absolute_column="w_reduced_oriented_absolute_change",
    )
    decision = build_test_decision(seed_rows, paired_length)
    _assert_finite(seed_rows, arm_summary, seed_length, paired_length, seed_models, paired_models)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        outputs = {
            "test_seed_arm_summary.csv": seed_rows,
            "test_arm_three_seed_summary.csv": arm_summary,
            "test_seed_length_changes.csv": seed_length,
            "test_paired_seed_length_directions.csv": paired_length,
            "test_seed_model_differences.csv": seed_models,
            "test_paired_seed_model_directions.csv": paired_models,
        }
        for filename, frame in outputs.items():
            frame.to_csv(temporary / filename, index=False)
        _write_json(temporary / "test_decision.json", decision)
        receipt = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": "three-seed independent test center90 context sensitivity",
            "execution": {"command": command, "cwd": str(root), "git": git, "evaluation_commit": EVALUATION_COMMIT},
            "inputs": inputs,
            "counts": {
                "expected_evaluations": 18,
                "actual_evaluations": len(seed_rows),
                "test_rows_per_evaluation": EXPECTED_TEST_WINDOWS,
                "test_metric_rows_read": EXPECTED_TEST_WINDOWS * len(seed_rows),
                "seed_arm_rows": len(seed_rows),
                "three_seed_arm_rows": len(arm_summary),
                "seed_length_change_rows": len(seed_length),
                "paired_length_direction_rows": len(paired_length),
                "seed_model_difference_rows": len(seed_models),
                "paired_model_direction_rows": len(paired_models),
            },
            "aggregation": {
                "run_metric": "sample_direct_mean",
                "seed_mean": "arithmetic_mean",
                "seed_sd": "sample_sd_ddof1",
                "pairing": "same_seed",
                "weighted_score": False,
                "p_value": False,
            },
            "access": {
                "evaluation_metrics_accessed": True,
                "evaluation_stored_summary_accessed": True,
                "checkpoint_content_read": False,
                "dataset_or_index_accessed": False,
                "signal_or_target_array_accessed": False,
                "model_training_used": False,
                "model_inference_used": False,
                "gpu_used": False,
            },
            "decision": decision,
            "artifacts": _artifact_records(temporary),
        }
        _write_json(temporary / "summary_receipt.json", receipt)
        _write_json(
            temporary / "artifact_manifest.json",
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "complete",
                "read_only_research_test_summary": True,
                "source_evaluation_commit": EVALUATION_COMMIT,
                "files": _artifact_records(temporary),
            },
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "summary_receipt.json"


def _audit_evaluation(evaluation_dir: Path, spec: Any) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    lifecycle_path = evaluation_dir / "lifecycle.json"
    receipt_path = evaluation_dir / "evaluation_receipt.json"
    manifest_path = evaluation_dir / "artifact_manifest.json"
    if _read_json(lifecycle_path) != {"status": "complete"}:
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} lifecycle 未 complete")
    receipt = _read_json(receipt_path)
    manifest = _read_json(manifest_path)
    if (
        receipt.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "complete"
        or receipt.get("identity") != asdict(spec)
        or manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "complete"
        or manifest.get("identity") != {"experiment_id": spec.experiment_id, "seed": spec.seed}
    ):
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} receipt/manifest identity 漂移")
    execution = receipt.get("execution", {})
    runtime = execution.get("runtime", {})
    expected_command = (
        f"{Path(sys.executable)} scripts/eval_paper_center90_research_test_v1.py "
        f"--model {spec.model_key} --input-sec {spec.input_sec} --seed {spec.seed} "
        "--device cuda:0 --confirm-research-test"
    )
    if (
        execution.get("command") != expected_command
        or execution.get("cwd") != str(Path(__file__).resolve().parents[2])
        or execution.get("git_commit") != EVALUATION_COMMIT
        or execution.get("git_dirty") is not False
        or any(runtime.get(key) != value for key, value in EXPECTED_RUNTIME.items())
        or runtime.get("logical_device") != "cuda:0"
        or runtime.get("cuda_visible_devices") not in {"0", "1"}
        or runtime.get("cuda_visible_device_count") != 1
        or runtime.get("cuda_device_name") != "NVIDIA GeForce RTX 4070 Ti SUPER"
    ):
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} execution/runtime identity 漂移")
    expected_data = {
        "dataset_index_sha256": EXPECTED_DATASET_INDEX_SHA256,
        "split": "test",
        "test_rows": EXPECTED_TEST_WINDOWS,
        "test_samp_ids": EXPECTED_TEST_SAMP_IDS,
        "test_row_ids_sha256": EXPECTED_TEST_ROW_IDS_SHA256,
        "non_test_row_overlap_count": 0,
        "sample_strategy": TEST_SAMPLE_STRATEGY,
        "sample_seed": TEST_SAMPLE_SEED,
        "input_key": INPUT_KEY,
        "target_key": TARGET_KEY,
        "input_samples": spec.input_sec * 100,
        "output_samples": 9000,
        "w_cache_path": str(_cache_root(spec.input_sec)) if spec.model_key == "wr" else None,
        "w_cache_manifest_sha256": _cache_hash(spec.input_sec) if spec.model_key == "wr" else None,
    }
    data = receipt.get("data", {})
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} test data identity 漂移")
    expected_access = {
        "train_signal_or_target_read": False,
        "validation_signal_target_or_prediction_read": False,
        "test_input_read": True,
        "test_target_array_read": True,
        "checkpoint_content_read": True,
        "checkpoint_final_read": False,
        "w_cache_read": spec.model_key == "wr",
        "model_training_used": False,
        "model_inference_used": True,
        "checkpoint_reselection_used": False,
    }
    if receipt.get("counts") != {"test_rows": EXPECTED_TEST_WINDOWS, "expected_test_rows": EXPECTED_TEST_WINDOWS} or receipt.get("access") != expected_access:
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} count/access identity 漂移")
    files = manifest.get("files")
    if not isinstance(files, list) or {item.get("filename") for item in files} != REQUIRED_ARTIFACTS:
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} artifact 集合漂移")
    for record in files:
        filename = str(record["filename"])
        path = evaluation_dir / filename
        if Path(filename).name != filename or not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} artifact hash/size 漂移: {filename}")
    receipt_artifacts = receipt.get("artifacts")
    if not isinstance(receipt_artifacts, list) or {item.get("filename") for item in receipt_artifacts} != REQUIRED_ARTIFACTS - {"evaluation_receipt.json"}:
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} receipt artifact 集合漂移")
    config = OmegaConf.load(evaluation_dir / "resolved_evaluation_config.yaml")
    if (
        str(config.model.variant) != spec.variant
        or int(config.window.input_sec) != spec.input_sec
        or int(config.window.output_sec) != 90
        or int(config.training.seed) != spec.seed
        or str(config.protocol.stage) != "research_test"
        or list(config.data.access_splits) != ["test"]
    ):
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} resolved evaluation config 漂移")
    metrics = pd.read_csv(evaluation_dir / "research_test_metrics.csv")
    _validate_metrics(metrics, spec)
    stored = pd.read_csv(evaluation_dir / "research_test_metrics_summary.csv")
    recomputed = summarize_center90_metrics(metrics)
    _validate_stored_summary(stored, recomputed, spec)
    identity = metrics.loc[:, ["dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id"]].reset_index(drop=True)
    arm = next(item for item in ARM_SPECS if item.experiment_id == spec.experiment_id)
    summary_row = stored.iloc[0].to_dict()
    row = {
        "experiment_id": spec.experiment_id,
        "model_id": arm.model_id,
        "model_label": arm.model_label,
        "variant": spec.variant,
        "input_sec": spec.input_sec,
        "seed": spec.seed,
        "parameter_count": spec.parameter_count,
        "w_scale_count": arm.w_scale_count,
        "best_epoch": spec.selected_epoch,
        "best_center90_rr_mae_bpm": float(summary_row["center90_rr_mae_bpm_mean"]),
        **summary_row,
    }
    input_record = {
        "experiment_id": spec.experiment_id,
        "seed": spec.seed,
        "evaluation_dir": str(evaluation_dir),
        "lifecycle_status": "complete",
        "lifecycle_sha256": sha256_file(lifecycle_path),
        "evaluation_receipt_sha256": sha256_file(receipt_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "metrics_sha256": next(item["sha256"] for item in files if item["filename"] == "research_test_metrics.csv"),
        "summary_sha256": next(item["sha256"] for item in files if item["filename"] == "research_test_metrics_summary.csv"),
        "checkpoint_content_read": False,
    }
    return row, input_record, identity


def _validate_stored_summary(stored: pd.DataFrame, recomputed: pd.DataFrame, spec: Any) -> None:
    if stored.shape != recomputed.shape or list(stored.columns) != list(recomputed.columns):
        raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} stored summary schema 漂移")
    for column in stored.columns:
        left, right = stored[column].iloc[0], recomputed[column].iloc[0]
        matches = int(left) == int(right) if isinstance(right, (int, np.integer)) else bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12, equal_nan=True))
        if not matches:
            raise RuntimeError(f"{spec.experiment_id}/seed_{spec.seed} stored summary 重算不一致: {column}")


def _assert_finite(*frames: pd.DataFrame) -> None:
    for frame in frames:
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all():
            raise FloatingPointError("center90 research-test summary output 含 NaN/Inf")


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [{"filename": path.name, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)} for path in sorted(directory.iterdir()) if path.is_file()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须为 object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _git_identity(root: Path) -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    return {"commit": commit.stdout.strip() if commit.returncode == 0 else None, "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None, "error": None if commit.returncode == 0 and status.returncode == 0 else "git identity failed", "python_version": sys.version.split()[0]}


__all__ = ["EVALUATION_COMMIT", "OUTPUT_FILENAMES", "SUMMARY_SCHEMA_VERSION", "audit_evaluations", "build_test_decision", "run_research_test_summary"]
