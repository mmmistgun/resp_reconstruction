from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from resp_train.paper_evidence.center_context_config import (
    CENTER_CONTEXT_P3_OUTPUT_ROOT,
    CENTER_CONTEXT_P3_W_CACHE_MANIFEST_SHA256,
    CENTER_CONTEXT_PROTOCOL_ID,
    load_center_context_config,
)
from resp_train.paper_evidence.center_context_experiment import center_experiment_id
from resp_train.paper_evidence.center_context_metrics import (
    CENTER_PRIMARY_METRICS,
    summarize_center_metrics,
)


P3_SUMMARY_SCHEMA_VERSION = "paper-center-context-p3-validation-summary-v1"
P3_RUN_COMMIT = "f57583b91f04e5ab5ea9475042489e8aeabb4e9f"
P3_SEED = 20260811
EXPECTED_TRAIN_ROWS = 10141
EXPECTED_VALIDATION_ROWS = 2675
EXPECTED_EPOCHS = 80
EXPECTED_UPDATES = 6400
EXPECTED_TRAIN_ROW_IDS_SHA256 = "f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e"
EXPECTED_VALIDATION_ROW_IDS_SHA256 = "b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a"
EXPECTED_DATASET_INDEX_SHA256 = "f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f"
DEFAULT_RUN_ROOT = Path(CENTER_CONTEXT_P3_OUTPUT_ROOT)
DEFAULT_OUTPUT_DIR = Path("runs/paper_evidence_v1/center_context/p3_validation_summary")


@dataclass(frozen=True)
class ArmSpec:
    experiment_id: str
    config_filename: str
    model_id: str
    model_label: str
    variant: str
    input_sec: int
    parameter_count: int
    w_scale_count: int


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    material_threshold: float
    material_kind: str


ARM_SPECS = (
    ArmSpec("CCV1_C201_60", "p3_ccv1_c201_60.yaml", "C201", "C201-center60", "c201_center60", 60, 1069802, 0),
    ArmSpec("CCV1_C201_90", "p3_ccv1_c201_90.yaml", "C201", "C201-center60", "c201_center60", 90, 1069802, 0),
    ArmSpec("CCV1_C201_180", "p3_ccv1_c201_180.yaml", "C201", "C201-center60", "c201_center60", 180, 1069802, 0),
    ArmSpec("CCV1_WR_60", "p3_ccv1_wr_60.yaml", "W-reduced", "W-reduced-center60", "w_reduced_center60", 60, 1219850, 49),
    ArmSpec("CCV1_WR_90", "p3_ccv1_wr_90.yaml", "W-reduced", "W-reduced-center60", "w_reduced_center60", 90, 1219850, 49),
    ArmSpec("CCV1_WR_180", "p3_ccv1_wr_180.yaml", "W-reduced", "W-reduced-center60", "w_reduced_center60", 180, 1219850, 49),
)
METRIC_SPECS = (
    MetricSpec("center_rr_mae_bpm", "minimize", 0.005, "relative"),
    MetricSpec("center_ibi_medae_sec", "minimize", 0.005, "relative"),
    MetricSpec("center_envelope_trajectory_mae", "minimize", 0.005, "relative"),
    MetricSpec("center_global_envelope_modulation_error", "minimize", 0.005, "relative"),
    MetricSpec("center_lag_aware_signed_pcc", "maximize", 0.002, "absolute"),
)
ERROR_METRICS = tuple(spec.name for spec in METRIC_SPECS if spec.direction == "minimize")
COMPANION_COLUMNS = (
    "center_ibi_coverage_mean",
    "center_ibi_coverage_n",
    "center_ibi_interpretable_fraction",
    "center_ibi_target_eligible_n",
)
EXPECTED_RUNTIME = {
    "device_type": "cuda",
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_benchmark": False,
    "amp_enabled": True,
    "amp_dtype": "bfloat16",
}
REQUIRED_ARTIFACTS = {
    "checkpoint_best_center_rr.pt",
    "checkpoint_final.pt",
    "data_identity.json",
    "optimizer_parameter_groups.json",
    "resolved_config.yaml",
    "runtime_identity.json",
    "train.log",
    "train_history.csv",
    "validation_center_metrics.csv",
    "validation_center_metrics_summary.csv",
}
METRICS_IDENTITY_COLUMNS = ("dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id")
OUTPUT_FILENAMES = (
    "p3_arm_validation_summary.csv",
    "p3_length_changes.csv",
    "p3_model_differences.csv",
    "p3_decision.json",
    "summary_receipt.json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_row_ids_sha256(values: pd.Series | np.ndarray) -> str:
    row_ids = np.asarray(values, dtype=np.int64)
    if row_ids.ndim != 1 or row_ids.size == 0 or np.unique(row_ids).size != row_ids.size:
        raise ValueError("validation dataset_row_id 必须一维、非空且唯一")
    return hashlib.sha256(np.sort(row_ids).tobytes(order="C")).hexdigest()


def audit_p3_runs(run_root: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = Path(run_root).resolve()
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None
    for spec in ARM_SPECS:
        run_dir = root / spec.experiment_id / f"seed_{P3_SEED}"
        row, input_record, identity = audit_formal_arm(
            run_dir,
            spec,
            seed=P3_SEED,
            expected_commit=P3_RUN_COMMIT,
            config_filename=spec.config_filename,
        )
        if reference_identity is None:
            reference_identity = identity
        elif not identity.equals(reference_identity):
            raise RuntimeError(f"{spec.experiment_id} validation row/metadata identity 与首个 arm 不一致")
        rows.append(row)
        inputs.append(input_record)
    return pd.DataFrame.from_records(rows), inputs


def build_length_changes(arm_summary: pd.DataFrame) -> pd.DataFrame:
    indexed = arm_summary.set_index(["model_id", "input_sec"])
    records: list[dict[str, Any]] = []
    for model_id in ("C201", "W-reduced"):
        for from_sec, to_sec in ((60, 90), (90, 180), (60, 180)):
            for metric in METRIC_SPECS:
                column = f"{metric.name}_mean"
                from_value = float(indexed.loc[(model_id, from_sec), column])
                to_value = float(indexed.loc[(model_id, to_sec), column])
                improvement = _oriented_improvement(from_value, to_value, metric)
                records.append(
                    {
                        "model_id": model_id,
                        "from_sec": from_sec,
                        "to_sec": to_sec,
                        "metric": metric.name,
                        "direction": metric.direction,
                        "from_value": from_value,
                        "to_value": to_value,
                        "raw_delta_to_minus_from": to_value - from_value,
                        "oriented_improvement": improvement,
                        "improvement_unit": metric.material_kind,
                        "material_threshold": metric.material_threshold,
                        "material_absolute_change": abs(improvement) >= metric.material_threshold,
                    }
                )
    return pd.DataFrame.from_records(records)


def build_model_differences(arm_summary: pd.DataFrame) -> pd.DataFrame:
    indexed = arm_summary.set_index(["model_id", "input_sec"])
    records: list[dict[str, Any]] = []
    for input_sec in (60, 90, 180):
        for metric in METRIC_SPECS:
            column = f"{metric.name}_mean"
            c201 = float(indexed.loc[("C201", input_sec), column])
            w_reduced = float(indexed.loc[("W-reduced", input_sec), column])
            records.append(
                {
                    "input_sec": input_sec,
                    "metric": metric.name,
                    "direction": metric.direction,
                    "c201_value": c201,
                    "w_reduced_value": w_reduced,
                    "w_reduced_minus_c201": w_reduced - c201,
                    "w_reduced_oriented_improvement": _oriented_improvement(c201, w_reduced, metric),
                    "improvement_unit": metric.material_kind,
                }
            )
    return pd.DataFrame.from_records(records)


def evaluate_upgrade_signal(length_changes: pd.DataFrame) -> dict[str, Any]:
    indexed = length_changes.set_index(["model_id", "from_sec", "to_sec", "metric"])

    def improvement(model_id: str, from_sec: int, to_sec: int, metric: str) -> float:
        return float(indexed.loc[(model_id, from_sec, to_sec, metric), "oriented_improvement"])

    monotonic: list[dict[str, Any]] = []
    saturation: list[dict[str, Any]] = []
    interaction: list[dict[str, Any]] = []
    for model_id in ("C201", "W-reduced"):
        for metric in ERROR_METRICS:
            first = improvement(model_id, 60, 90, metric)
            second = improvement(model_id, 90, 180, metric)
            total = improvement(model_id, 60, 180, metric)
            if first >= 0.0 and second >= 0.0 and total >= 0.005:
                monotonic.append({"model_id": model_id, "metric": metric, "improvement_60_to_180": total})
            if first >= 0.005 and abs(second) < 0.005:
                saturation.append(
                    {
                        "model_id": model_id,
                        "metric": metric,
                        "improvement_60_to_90": first,
                        "improvement_90_to_180": second,
                    }
                )
    for metric_spec in METRIC_SPECS:
        left = improvement("C201", 60, 180, metric_spec.name)
        right = improvement("W-reduced", 60, 180, metric_spec.name)
        opposite = left * right < 0.0
        material = max(abs(left), abs(right)) >= metric_spec.material_threshold
        if opposite and material:
            interaction.append(
                {
                    "metric": metric_spec.name,
                    "c201_improvement_60_to_180": left,
                    "w_reduced_improvement_60_to_180": right,
                    "improvement_unit": metric_spec.material_kind,
                    "material_threshold": metric_spec.material_threshold,
                }
            )
    upgrade = bool(monotonic or saturation or interaction)
    return {
        "protocol_gate": "p3_single_seed_directional_diagnostic",
        "upgrade_signal_detected": upgrade,
        "triggered_conditions": {
            "monotonic_improvement": monotonic,
            "candidate_90s_saturation": saturation,
            "model_by_context_interaction": interaction,
        },
        "decision": (
            "p3_upgrade_signal_detected_user_authorization_required"
            if upgrade
            else "stop_window_queue_no_upgrade_signal"
        ),
        "directional_result_only": True,
        "stable_context_effect_claim_allowed": False,
        "p4_training_authorized": False,
        "p4_additional_runs_if_authorized": 12,
        "research_test_authorized": False,
        "weighted_score_used": False,
    }


def run_p3_validation_summary(
    *,
    repo_root: str | Path,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    command: str,
    require_clean_git: bool = True,
) -> Path:
    root = Path(repo_root).resolve()
    source_root = _resolve_under_root(root, run_root)
    output = _resolve_under_root(root, output_dir)
    if output.exists():
        raise FileExistsError(f"P3 汇总输出禁止覆盖: {output}")
    git = _git_identity(root)
    if require_clean_git and (git["error"] is not None or git["dirty"] is not False):
        raise RuntimeError("P3 冻结汇总要求干净 Git 工作树")

    arm_summary, inputs = audit_p3_runs(source_root)
    length_changes = build_length_changes(arm_summary)
    model_differences = build_model_differences(arm_summary)
    decision = evaluate_upgrade_signal(length_changes)
    _assert_finite_summary_outputs(arm_summary, length_changes, model_differences)

    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        arm_summary.to_csv(temporary / "p3_arm_validation_summary.csv", index=False)
        length_changes.to_csv(temporary / "p3_length_changes.csv", index=False)
        model_differences.to_csv(temporary / "p3_model_differences.csv", index=False)
        _write_json(temporary / "p3_decision.json", decision)
        artifacts = _artifact_records(temporary)
        receipt = {
            "schema_version": P3_SUMMARY_SCHEMA_VERSION,
            "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": "single-seed validation directional diagnostic",
            "execution": {
                "command": command,
                "cwd": str(root),
                "git": git,
                "p3_run_commit": P3_RUN_COMMIT,
            },
            "inputs": inputs,
            "counts": {
                "expected_runs": 6,
                "actual_runs": len(arm_summary),
                "epochs_per_run": EXPECTED_EPOCHS,
                "validation_rows_per_run": EXPECTED_VALIDATION_ROWS,
                "validation_metric_rows_read": EXPECTED_VALIDATION_ROWS * len(arm_summary),
                "arm_summary_rows": len(arm_summary),
                "length_change_rows": len(length_changes),
                "model_difference_rows": len(model_differences),
            },
            "access": {
                "train_history_accessed": True,
                "validation_metrics_accessed": True,
                "validation_stored_summary_accessed": True,
                "checkpoint_content_read": False,
                "dataset_or_index_accessed": False,
                "signal_or_target_array_accessed": False,
                "model_training_used": False,
                "model_inference_used": False,
                "gpu_used": False,
                "research_test_accessed": False,
            },
            "decision": decision,
            "artifacts": artifacts,
        }
        _write_json(temporary / "summary_receipt.json", receipt)
        manifest = {
            "schema_version": P3_SUMMARY_SCHEMA_VERSION,
            "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
            "status": "complete",
            "read_only_validation_summary": True,
            "source_run_commit": P3_RUN_COMMIT,
            "files": _artifact_records(temporary),
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "summary_receipt.json"


def audit_formal_arm(
    run_dir: Path,
    spec: ArmSpec,
    *,
    seed: int,
    expected_commit: str,
    config_filename: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"中心 formal run 缺失: {run_dir}")
    lifecycle_path = run_dir / "lifecycle.json"
    manifest_path = run_dir / "artifact_manifest.json"
    lifecycle = _load_json(lifecycle_path)
    manifest = _load_json(manifest_path)
    if lifecycle != {"status": "complete"}:
        raise RuntimeError(f"{spec.experiment_id} lifecycle 未 complete")
    expected_manifest = {
        "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
        "task": "paper_center_context_v1",
        "experiment_id": spec.experiment_id,
        "seed": seed,
        "parameter_count": spec.parameter_count,
        "selector": "full_validation_center_rr_mae_bpm_strict_lower_tie_earlier",
        "train_access": True,
        "validation_access": True,
        "test_access": False,
        "test_cache_created": False,
        "outer_target_supervision": False,
        "checkpoint_initialization_used": False,
        "scientific_role": "auxiliary_context_sensitivity_outside_main_model_experiment",
        "strict_w0_equivalent": False,
        "w_scale_count": spec.w_scale_count,
        "early_stopping": False,
        "resume": False,
        "runtime": EXPECTED_RUNTIME,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"{spec.experiment_id} manifest {key} 漂移")
    if manifest.get("git") != {"commit": expected_commit, "dirty": False, "error": None}:
        raise RuntimeError(f"{spec.experiment_id} Git identity 漂移")
    expected_command = [
        "scripts/train_paper_center_context_v1.py",
        "--config",
        f"configs/paper_evidence_v1/{config_filename}",
        "--confirm-formal-training",
    ]
    if manifest.get("command") != expected_command:
        raise RuntimeError(f"{spec.experiment_id} formal command identity 漂移")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise RuntimeError(f"{spec.experiment_id} artifacts 不完整")
    for filename, record in artifacts.items():
        if Path(filename).name != filename or not isinstance(record, dict):
            raise RuntimeError(f"{spec.experiment_id} artifact 记录非法")
        path = run_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"{spec.experiment_id} artifact 缺失: {filename}")
        if int(record.get("size_bytes", -1)) != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"{spec.experiment_id} artifact hash/size 漂移: {filename}")

    cfg = load_center_context_config(run_dir / "resolved_config.yaml")
    if center_experiment_id(str(cfg.model.variant), int(cfg.window.input_samples)) != spec.experiment_id:
        raise RuntimeError(f"{spec.experiment_id} resolved config identity 漂移")
    if str(cfg.model.variant) != spec.variant or int(cfg.window.input_sec) != spec.input_sec:
        raise RuntimeError(f"{spec.experiment_id} variant/length 漂移")

    runtime = _load_json(run_dir / "runtime_identity.json")
    if runtime != EXPECTED_RUNTIME:
        raise RuntimeError(f"{spec.experiment_id} runtime identity 漂移")
    data = _load_json(run_dir / "data_identity.json")
    expected_data = {
        "train_rows": EXPECTED_TRAIN_ROWS,
        "val_rows": EXPECTED_VALIDATION_ROWS,
        "train_row_ids_sha256": EXPECTED_TRAIN_ROW_IDS_SHA256,
        "val_row_ids_sha256": EXPECTED_VALIDATION_ROW_IDS_SHA256,
        "row_overlap_count": 0,
        "samp_id_overlap_count": 0,
        "subject_session_overlap_count": 0,
        "views_share_parent_rows": True,
        "dataset_index_sha256": EXPECTED_DATASET_INDEX_SHA256,
        "input_samples": spec.input_sec * 100,
        "input_slice": {
            60: [6000, 12000],
            90: [4500, 13500],
            180: [0, 18000],
        }[spec.input_sec],
        "target_slice": [6000, 12000],
        "test_access": False,
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise RuntimeError(f"{spec.experiment_id} data identity {key} 漂移")
    if spec.w_scale_count:
        expected_cache = CENTER_CONTEXT_P3_W_CACHE_MANIFEST_SHA256[spec.input_sec * 100]
        if data.get("center_w_cache_manifest_sha256") != expected_cache:
            raise RuntimeError(f"{spec.experiment_id} W cache identity 漂移")
        if Path(str(data.get("center_w_cache_path"))).resolve() != Path(
            str(cfg.data.center_w_cache_path)
        ).resolve():
            raise RuntimeError(f"{spec.experiment_id} W cache path identity 漂移")
    elif "center_w_cache_manifest_sha256" in data:
        raise RuntimeError(f"{spec.experiment_id} 不得登记 W cache")

    history = pd.read_csv(run_dir / "train_history.csv")
    _validate_history(history, manifest, spec)
    metrics = pd.read_csv(run_dir / "validation_center_metrics.csv")
    identity = _validate_metrics(metrics, spec)
    stored_summary = pd.read_csv(run_dir / "validation_center_metrics_summary.csv")
    recomputed = summarize_center_metrics(metrics)
    _validate_stored_summary(stored_summary, recomputed, spec)
    summary_row = stored_summary.iloc[0].to_dict()
    if not np.isclose(
        float(manifest["best_center_rr_mae_bpm"]),
        float(summary_row["center_rr_mae_bpm_mean"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{spec.experiment_id} selector 与最终 validation summary 不一致")
    row = {
        "experiment_id": spec.experiment_id,
        "model_id": spec.model_id,
        "model_label": spec.model_label,
        "variant": spec.variant,
        "input_sec": spec.input_sec,
        "seed": seed,
        "parameter_count": spec.parameter_count,
        "w_scale_count": spec.w_scale_count,
        "best_epoch": int(manifest["best_epoch"]),
        "best_center_rr_mae_bpm": float(manifest["best_center_rr_mae_bpm"]),
        **summary_row,
    }
    input_record = {
        "experiment_id": spec.experiment_id,
        "seed": seed,
        "run_dir": str(run_dir),
        "lifecycle_status": "complete",
        "lifecycle_sha256": sha256_file(lifecycle_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "validation_metrics_sha256": artifacts["validation_center_metrics.csv"]["sha256"],
        "validation_summary_sha256": artifacts["validation_center_metrics_summary.csv"]["sha256"],
        "checkpoint_content_read": False,
    }
    return row, input_record, identity


def _validate_history(history: pd.DataFrame, manifest: Mapping[str, Any], spec: ArmSpec) -> None:
    required = {"epoch", "optimizer_update", "val_center_rr_mae_bpm"}
    if len(history) != EXPECTED_EPOCHS or not required.issubset(history.columns):
        raise RuntimeError(f"{spec.experiment_id} train history 不完整")
    if history["epoch"].astype(int).tolist() != list(range(1, EXPECTED_EPOCHS + 1)):
        raise RuntimeError(f"{spec.experiment_id} epoch identity 漂移")
    numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or int(history.iloc[-1]["optimizer_update"]) != EXPECTED_UPDATES:
        raise FloatingPointError(f"{spec.experiment_id} history 非有限或 updates 未闭合")
    values = history["val_center_rr_mae_bpm"].to_numpy(dtype=np.float64)
    best_index = int(np.argmin(values))
    if int(manifest.get("best_epoch", -1)) != best_index + 1:
        raise RuntimeError(f"{spec.experiment_id} selector epoch 与 history 不一致")
    if not np.isclose(float(manifest.get("best_center_rr_mae_bpm", np.nan)), values[best_index], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{spec.experiment_id} selector value 与 history 不一致")


def _validate_metrics(metrics: pd.DataFrame, spec: ArmSpec) -> pd.DataFrame:
    required = {"method", *METRICS_IDENTITY_COLUMNS, *CENTER_PRIMARY_METRICS, "center_ibi_coverage"}
    if len(metrics) != EXPECTED_VALIDATION_ROWS or not required.issubset(metrics.columns):
        raise RuntimeError(f"{spec.experiment_id} validation metrics shape/schema 漂移")
    if metrics["method"].astype(str).nunique() != 1 or metrics["method"].astype(str).iloc[0] != spec.variant:
        raise RuntimeError(f"{spec.experiment_id} method identity 漂移")
    if set(metrics["split"].astype(str)) != {"val"} or set(metrics["input_set"].astype(str)) != {
        "research_v2_waveform"
    }:
        raise RuntimeError(f"{spec.experiment_id} validation split/input_set 漂移")
    if dataset_row_ids_sha256(metrics["dataset_row_id"]) != EXPECTED_VALIDATION_ROW_IDS_SHA256:
        raise RuntimeError(f"{spec.experiment_id} validation row set 漂移")
    always_finite = [metric for metric in CENTER_PRIMARY_METRICS if metric != "center_ibi_medae_sec"]
    if not np.isfinite(metrics[always_finite].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError(f"{spec.experiment_id} validation primary metrics 非有限")
    ibi = pd.to_numeric(metrics["center_ibi_medae_sec"], errors="coerce").to_numpy(dtype=np.float64)
    coverage = pd.to_numeric(metrics["center_ibi_coverage"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isinf(ibi).any() or np.isinf(coverage).any():
        raise FloatingPointError(f"{spec.experiment_id} IBI/coverage 含 Inf")
    return metrics.loc[:, METRICS_IDENTITY_COLUMNS].reset_index(drop=True)


def _validate_stored_summary(stored: pd.DataFrame, recomputed: pd.DataFrame, spec: ArmSpec) -> None:
    if stored.shape != recomputed.shape or list(stored.columns) != list(recomputed.columns):
        raise RuntimeError(f"{spec.experiment_id} stored summary schema 漂移")
    for column in stored.columns:
        left = stored[column].iloc[0]
        right = recomputed[column].iloc[0]
        if isinstance(right, (int, np.integer)):
            matches = int(left) == int(right)
        else:
            matches = bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12, equal_nan=True))
        if not matches:
            raise RuntimeError(f"{spec.experiment_id} stored summary 与逐样本重算不一致: {column}")


def _oriented_improvement(from_value: float, to_value: float, metric: MetricSpec) -> float:
    if not np.isfinite(from_value) or not np.isfinite(to_value):
        raise FloatingPointError(f"{metric.name} change 输入必须有限")
    if metric.material_kind == "relative":
        if from_value <= 0.0:
            raise ValueError(f"{metric.name} relative change 基准必须为正")
        return (from_value - to_value) / from_value
    return to_value - from_value


def _assert_finite_summary_outputs(*frames: pd.DataFrame) -> None:
    for frame in frames:
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError("P3 summary output 含 NaN/Inf")


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {"filename": path.name, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为 object: {path}")
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_under_root(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _git_identity(root: Path) -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "error": None if commit.returncode == 0 and status.returncode == 0 else "git identity failed",
        "python_version": sys.version.split()[0],
    }


__all__ = [
    "ARM_SPECS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RUN_ROOT",
    "METRIC_SPECS",
    "P3_RUN_COMMIT",
    "P3_SUMMARY_SCHEMA_VERSION",
    "audit_formal_arm",
    "audit_p3_runs",
    "build_length_changes",
    "build_model_differences",
    "evaluate_upgrade_signal",
    "run_p3_validation_summary",
    "sha256_file",
]
