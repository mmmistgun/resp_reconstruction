from __future__ import annotations

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

from resp_train.paper_evidence.center90_config import (
    CENTER90_FORMAL_OUTPUT_ROOT,
    CENTER90_FORMAL_SEEDS,
    CENTER90_PROTOCOL_ID,
    CENTER90_W_CACHE_MANIFEST_SHA256,
    load_center90_config,
)
from resp_train.paper_evidence.center90_experiment import center90_experiment_id
from resp_train.paper_evidence.center90_metrics import (
    CENTER90_PRIMARY_METRICS,
    summarize_center90_metrics,
)
from resp_train.paper_evidence.center_context_summary import dataset_row_ids_sha256, sha256_file


SUMMARY_SCHEMA_VERSION = "paper-center90-context-validation-summary-v1"
FORMAL_RUN_COMMIT = "033e3ff2a9fdb793ad0ffc93fbb9f7821c25b24a"
DEFAULT_RUN_ROOT = Path(CENTER90_FORMAL_OUTPUT_ROOT)
DEFAULT_OUTPUT_DIR = Path("runs/paper_evidence_v1/center90_context/validation_summary")
EXPECTED_TRAIN_ROWS = 10141
EXPECTED_VALIDATION_ROWS = 2675
EXPECTED_EPOCHS = 80
EXPECTED_UPDATES = 6400
EXPECTED_TRAIN_ROW_IDS_SHA256 = "f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e"
EXPECTED_VALIDATION_ROW_IDS_SHA256 = "b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a"
EXPECTED_DATASET_INDEX_SHA256 = "f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f"
EXPECTED_RUNTIME = {
    "device_type": "cuda",
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_benchmark": False,
    "amp_enabled": True,
    "amp_dtype": "bfloat16",
}
REQUIRED_ARTIFACTS = {
    "checkpoint_best_center90_rr.pt",
    "checkpoint_final.pt",
    "data_identity.json",
    "optimizer_parameter_groups.json",
    "resolved_config.yaml",
    "runtime_identity.json",
    "train.log",
    "train_history.csv",
    "validation_center90_metrics.csv",
    "validation_center90_metrics_summary.csv",
}
METRICS_IDENTITY_COLUMNS = ("dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id")
OUTPUT_FILENAMES = (
    "seed_arm_validation_summary.csv",
    "arm_three_seed_summary.csv",
    "seed_length_changes.csv",
    "paired_seed_length_directions.csv",
    "seed_model_differences.csv",
    "paired_seed_model_directions.csv",
    "decision.json",
    "summary_receipt.json",
)


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


ARM_SPECS = tuple(
    ArmSpec(
        experiment_id=f"C90V1_{prefix}_{input_sec}",
        config_filename=f"c90v1_{short}_{input_sec}_seed{{seed}}.yaml",
        model_id=model_id,
        model_label=model_label,
        variant=variant,
        input_sec=input_sec,
        parameter_count=parameter_count,
        w_scale_count=w_scale_count,
    )
    for prefix, short, model_id, model_label, variant, parameter_count, w_scale_count in (
        ("C201", "c201", "C201", "C201-center90", "c201_center90", 1069802, 0),
        ("WR", "wr", "W-reduced", "W-reduced-center90", "w_reduced_center90", 1219850, 49),
    )
    for input_sec in (90, 135, 180)
)
METRIC_SPECS = (
    MetricSpec("center90_rr_mae_bpm", "minimize", 0.005, "relative"),
    MetricSpec("center90_ibi_medae_sec", "minimize", 0.005, "relative"),
    MetricSpec("center90_envelope_trajectory_mae", "minimize", 0.005, "relative"),
    MetricSpec("center90_global_envelope_modulation_error", "minimize", 0.005, "relative"),
    MetricSpec("center90_lag_aware_signed_pcc", "maximize", 0.002, "absolute"),
)
SUMMARY_COLUMNS = (
    *(f"{spec.name}_mean" for spec in METRIC_SPECS),
    "center90_ibi_coverage_mean",
    "center90_ibi_interpretable_fraction",
)


def audit_formal_runs(run_root: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = Path(run_root).resolve()
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None
    for seed in CENTER90_FORMAL_SEEDS:
        for spec in ARM_SPECS:
            run_dir = root / spec.experiment_id / f"seed_{seed}"
            row, input_record, identity = _audit_formal_arm(run_dir, spec, seed)
            if reference_identity is None:
                reference_identity = identity
            elif not identity.equals(reference_identity):
                raise RuntimeError(f"{spec.experiment_id}/seed_{seed} validation identity 不一致")
            rows.append(row)
            inputs.append(input_record)
    result = pd.DataFrame.from_records(rows).sort_values(["model_id", "input_sec", "seed"])
    if len(result) != 18 or result[["experiment_id", "seed"]].duplicated().any():
        raise RuntimeError("center90 formal 必须恰好闭合 6 arms × 3 seeds")
    return result, inputs


def build_three_seed_arm_summary(seed_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in ARM_SPECS:
        group = seed_rows.loc[seed_rows["experiment_id"].eq(spec.experiment_id)].sort_values("seed")
        _require_three_seeds(group, spec.experiment_id)
        record: dict[str, Any] = {
            "experiment_id": spec.experiment_id,
            "model_id": spec.model_id,
            "model_label": spec.model_label,
            "variant": spec.variant,
            "input_sec": spec.input_sec,
            "parameter_count": spec.parameter_count,
            "w_scale_count": spec.w_scale_count,
            "seeds": "|".join(str(int(value)) for value in group["seed"]),
            "best_epochs": "|".join(str(int(value)) for value in group["best_epoch"]),
        }
        for column in SUMMARY_COLUMNS:
            values = group[column].to_numpy(dtype=np.float64)
            record[f"{column}_seed_mean"] = float(np.mean(values))
            record[f"{column}_seed_sample_sd"] = float(np.std(values, ddof=1))
            record[f"{column}_seed_n"] = len(values)
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_seed_length_changes(seed_rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for seed in CENTER90_FORMAL_SEEDS:
        indexed = seed_rows.loc[seed_rows["seed"].eq(seed)].set_index(["model_id", "input_sec"])
        records = []
        for model_id in ("C201", "W-reduced"):
            for from_sec, to_sec in ((90, 135), (135, 180), (90, 180)):
                for metric in METRIC_SPECS:
                    from_value = float(indexed.loc[(model_id, from_sec), f"{metric.name}_mean"])
                    to_value = float(indexed.loc[(model_id, to_sec), f"{metric.name}_mean"])
                    relative, absolute, protocol_value = _oriented_changes(from_value, to_value, metric)
                    records.append(_change_record(seed, model_id, from_sec, to_sec, metric, from_value, to_value, relative, absolute, protocol_value))
        frames.append(pd.DataFrame.from_records(records))
    return pd.concat(frames, ignore_index=True)


def build_seed_model_differences(seed_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for seed in CENTER90_FORMAL_SEEDS:
        indexed = seed_rows.loc[seed_rows["seed"].eq(seed)].set_index(["model_id", "input_sec"])
        for input_sec in (90, 135, 180):
            for metric in METRIC_SPECS:
                c201 = float(indexed.loc[("C201", input_sec), f"{metric.name}_mean"])
                wr = float(indexed.loc[("W-reduced", input_sec), f"{metric.name}_mean"])
                relative, absolute, protocol_value = _oriented_changes(c201, wr, metric)
                records.append(
                    {
                        "seed": seed,
                        "input_sec": input_sec,
                        "metric": metric.name,
                        "direction": metric.direction,
                        "c201_value": c201,
                        "w_reduced_value": wr,
                        "w_reduced_minus_c201": wr - c201,
                        "w_reduced_oriented_relative_change": relative,
                        "w_reduced_oriented_absolute_change": absolute,
                        "material_kind": metric.material_kind,
                        "material_threshold": metric.material_threshold,
                        "protocol_material_value": protocol_value,
                    }
                )
    return pd.DataFrame.from_records(records)


def build_paired_directions(frame: pd.DataFrame, *, group_columns: tuple[str, ...], relative_column: str, absolute_column: str) -> pd.DataFrame:
    records = []
    for identity, group in frame.groupby(list(group_columns), sort=False):
        group = group.sort_values("seed")
        _require_three_seeds(group, str(identity))
        if not isinstance(identity, tuple):
            identity = (identity,)
        protocol_values = group["protocol_material_value"].to_numpy(dtype=np.float64)
        relative_values = group[relative_column].to_numpy(dtype=np.float64)
        absolute_values = group[absolute_column].to_numpy(dtype=np.float64)
        threshold = float(group["material_threshold"].iloc[0])
        records.append(
            {
                **dict(zip(group_columns, identity, strict=True)),
                "direction": str(group["direction"].iloc[0]),
                "material_kind": str(group["material_kind"].iloc[0]),
                "material_threshold": threshold,
                "oriented_relative_change_seed_mean": float(np.mean(relative_values)),
                "oriented_relative_change_seed_sample_sd": float(np.std(relative_values, ddof=1)),
                "oriented_absolute_change_seed_mean": float(np.mean(absolute_values)),
                "oriented_absolute_change_seed_sample_sd": float(np.std(absolute_values, ddof=1)),
                "improved_seed_count": int(np.sum(protocol_values > 0.0)),
                "worsened_seed_count": int(np.sum(protocol_values < 0.0)),
                "equal_seed_count": int(np.sum(protocol_values == 0.0)),
                "materially_improved_seed_count": int(np.sum(protocol_values >= threshold)),
                "materially_worsened_seed_count": int(np.sum(protocol_values <= -threshold)),
                "within_materiality_seed_count": int(np.sum(np.abs(protocol_values) < threshold)),
            }
        )
    return pd.DataFrame.from_records(records)


def build_descriptive_decision(seed_rows: pd.DataFrame, paired_length: pd.DataFrame) -> dict[str, Any]:
    rr = paired_length.loc[paired_length["metric"].eq("center90_rr_mae_bpm")]
    best_inputs: dict[str, dict[str, int]] = {}
    for model_id in ("C201", "W-reduced"):
        per_seed: dict[str, int] = {}
        for seed in CENTER90_FORMAL_SEEDS:
            group = seed_rows.loc[seed_rows["model_id"].eq(model_id) & seed_rows["seed"].eq(seed)]
            per_seed[str(seed)] = int(group.loc[group["center90_rr_mae_bpm_mean"].idxmin(), "input_sec"])
        best_inputs[model_id] = per_seed
    return {
        "evidence_scope": "three-seed_validation_center90_context_sensitivity",
        "rr_paired_length_directions": rr.to_dict(orient="records"),
        "rr_best_input_sec_by_model_and_seed": best_inputs,
        "reporting_disposition": "report_mean_sample_sd_and_paired_seed_directions",
        "weighted_score_used": False,
        "model_or_length_selection_performed": False,
        "cross_output_absolute_metric_comparison_allowed": False,
        "research_test_authorized": False,
    }


def run_center90_validation_summary(*, repo_root: str | Path, command: str, run_root: str | Path = DEFAULT_RUN_ROOT, output_dir: str | Path = DEFAULT_OUTPUT_DIR, require_clean_git: bool = True) -> Path:
    root = Path(repo_root).resolve()
    source = _resolve(root, run_root)
    output = _resolve(root, output_dir)
    if output.exists():
        raise FileExistsError(f"center90 冻结汇总输出禁止覆盖: {output}")
    git = _git_identity(root)
    if require_clean_git and (git["error"] is not None or git["dirty"] is not False):
        raise RuntimeError("center90 冻结汇总要求干净 Git 工作树")
    seed_rows, inputs = audit_formal_runs(source)
    arm_summary = build_three_seed_arm_summary(seed_rows)
    seed_length = build_seed_length_changes(seed_rows)
    paired_length = build_paired_directions(seed_length, group_columns=("model_id", "from_sec", "to_sec", "metric"), relative_column="oriented_relative_change", absolute_column="oriented_absolute_change")
    seed_models = build_seed_model_differences(seed_rows)
    paired_models = build_paired_directions(seed_models, group_columns=("input_sec", "metric"), relative_column="w_reduced_oriented_relative_change", absolute_column="w_reduced_oriented_absolute_change")
    decision = build_descriptive_decision(seed_rows, paired_length)
    _assert_finite(seed_rows, arm_summary, seed_length, paired_length, seed_models, paired_models)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        outputs = {
            "seed_arm_validation_summary.csv": seed_rows,
            "arm_three_seed_summary.csv": arm_summary,
            "seed_length_changes.csv": seed_length,
            "paired_seed_length_directions.csv": paired_length,
            "seed_model_differences.csv": seed_models,
            "paired_seed_model_directions.csv": paired_models,
        }
        for filename, frame in outputs.items():
            frame.to_csv(temporary / filename, index=False)
        _write_json(temporary / "decision.json", decision)
        receipt = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "protocol_id": CENTER90_PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": "three-seed validation center90 context sensitivity",
            "execution": {"command": command, "cwd": str(root), "git": git, "formal_run_commit": FORMAL_RUN_COMMIT},
            "inputs": inputs,
            "counts": {
                "expected_runs": 18,
                "actual_runs": len(seed_rows),
                "epochs_per_run": EXPECTED_EPOCHS,
                "validation_rows_per_run": EXPECTED_VALIDATION_ROWS,
                "validation_metric_rows_read": EXPECTED_VALIDATION_ROWS * len(seed_rows),
                "seed_arm_rows": len(seed_rows),
                "three_seed_arm_rows": len(arm_summary),
                "seed_length_change_rows": len(seed_length),
                "paired_length_direction_rows": len(paired_length),
                "seed_model_difference_rows": len(seed_models),
                "paired_model_direction_rows": len(paired_models),
            },
            "aggregation": {"run_metric": "sample_direct_mean", "seed_mean": "arithmetic_mean", "seed_sd": "sample_sd_ddof1", "pairing": "same_seed", "weighted_score": False, "p_value": False},
            "access": {"train_history_accessed": True, "validation_metrics_accessed": True, "validation_stored_summary_accessed": True, "checkpoint_content_read": False, "dataset_or_index_accessed": False, "signal_or_target_array_accessed": False, "model_training_used": False, "model_inference_used": False, "gpu_used": False, "research_test_accessed": False},
            "decision": decision,
            "artifacts": _artifact_records(temporary),
        }
        _write_json(temporary / "summary_receipt.json", receipt)
        _write_json(temporary / "artifact_manifest.json", {"schema_version": SUMMARY_SCHEMA_VERSION, "protocol_id": CENTER90_PROTOCOL_ID, "status": "complete", "read_only_validation_summary": True, "source_run_commit": FORMAL_RUN_COMMIT, "files": _artifact_records(temporary)})
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "summary_receipt.json"


def _audit_formal_arm(run_dir: Path, spec: ArmSpec, seed: int) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    lifecycle_path = run_dir / "lifecycle.json"
    manifest_path = run_dir / "artifact_manifest.json"
    lifecycle = _load_json(lifecycle_path)
    manifest = _load_json(manifest_path)
    if lifecycle != {"status": "complete"}:
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} lifecycle 未 complete")
    expected_manifest = {
        "protocol_id": CENTER90_PROTOCOL_ID,
        "task": "paper_center90_context_v1",
        "experiment_id": spec.experiment_id,
        "seed": seed,
        "parameter_count": spec.parameter_count,
        "selector": "full_validation_center90_rr_mae_bpm_strict_lower_tie_earlier",
        "train_access": True,
        "validation_access": True,
        "test_access": False,
        "test_cache_created": False,
        "outer_target_supervision": False,
        "checkpoint_initialization_used": False,
        "scientific_role": "independent_center90_output_context_scale_experiment",
        "representation": "reduced_w_49_scale" if spec.w_scale_count else "time_domain",
        "w_scale_count": spec.w_scale_count,
        "early_stopping": False,
        "resume": False,
        "runtime": EXPECTED_RUNTIME,
        "git": {"commit": FORMAL_RUN_COMMIT, "dirty": False, "error": None},
        "command": ["scripts/train_paper_center90_context_v1.py", "--config", f"configs/paper_evidence_v1/{spec.config_filename.format(seed=seed)}", "--confirm-formal-training"],
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"{spec.experiment_id}/seed_{seed} manifest {key} 漂移")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} artifacts 集合漂移")
    for filename, record in artifacts.items():
        path = run_dir / filename
        if Path(filename).name != filename or not path.is_file() or not isinstance(record, dict):
            raise RuntimeError(f"{spec.experiment_id}/seed_{seed} artifact 非法: {filename}")
        if record.get("size_bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"{spec.experiment_id}/seed_{seed} artifact hash/size 漂移: {filename}")
    cfg = load_center90_config(run_dir / "resolved_config.yaml")
    if center90_experiment_id(str(cfg.model.variant), int(cfg.window.input_samples)) != spec.experiment_id or str(cfg.model.variant) != spec.variant or int(cfg.window.input_sec) != spec.input_sec or int(cfg.training.seed) != seed:
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} resolved config identity 漂移")
    if _load_json(run_dir / "runtime_identity.json") != EXPECTED_RUNTIME:
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} runtime identity 漂移")
    _validate_data_identity(_load_json(run_dir / "data_identity.json"), cfg, spec)
    _validate_history(pd.read_csv(run_dir / "train_history.csv"), manifest, spec, seed)
    metrics = pd.read_csv(run_dir / "validation_center90_metrics.csv")
    identity = _validate_metrics(metrics, spec, seed)
    stored = pd.read_csv(run_dir / "validation_center90_metrics_summary.csv")
    recomputed = summarize_center90_metrics(metrics)
    _validate_stored_summary(stored, recomputed, spec, seed)
    summary_row = stored.iloc[0].to_dict()
    if not np.isclose(float(manifest["best_center90_rr_mae_bpm"]), float(summary_row["center90_rr_mae_bpm_mean"]), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} selector 与 validation summary 不一致")
    row = {"experiment_id": spec.experiment_id, "model_id": spec.model_id, "model_label": spec.model_label, "variant": spec.variant, "input_sec": spec.input_sec, "seed": seed, "parameter_count": spec.parameter_count, "w_scale_count": spec.w_scale_count, "best_epoch": int(manifest["best_epoch"]), "best_center90_rr_mae_bpm": float(manifest["best_center90_rr_mae_bpm"]), **summary_row}
    input_record = {"experiment_id": spec.experiment_id, "seed": seed, "run_dir": str(run_dir), "lifecycle_status": "complete", "lifecycle_sha256": sha256_file(lifecycle_path), "artifact_manifest_sha256": sha256_file(manifest_path), "validation_metrics_sha256": artifacts["validation_center90_metrics.csv"]["sha256"], "validation_summary_sha256": artifacts["validation_center90_metrics_summary.csv"]["sha256"], "checkpoint_content_read": False}
    return row, input_record, identity


def _validate_data_identity(data: Mapping[str, Any], cfg: Any, spec: ArmSpec) -> None:
    expected = {
        "train_rows": EXPECTED_TRAIN_ROWS, "val_rows": EXPECTED_VALIDATION_ROWS,
        "train_row_ids_sha256": EXPECTED_TRAIN_ROW_IDS_SHA256, "val_row_ids_sha256": EXPECTED_VALIDATION_ROW_IDS_SHA256,
        "row_overlap_count": 0, "samp_id_overlap_count": 0, "subject_session_overlap_count": 0,
        "views_share_parent_rows": True, "dataset_index_sha256": EXPECTED_DATASET_INDEX_SHA256,
        "input_samples": spec.input_sec * 100, "input_slice": {90: [4500, 13500], 135: [2250, 15750], 180: [0, 18000]}[spec.input_sec],
        "target_slice": [4500, 13500], "source_key_column": "bcg_rawish_segment_soft_z_key", "target_key_column": "target_waveform_segment_soft_z_key", "test_access": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"{spec.experiment_id} data identity {key} 漂移")
    if spec.w_scale_count:
        expected_hash = CENTER90_W_CACHE_MANIFEST_SHA256[spec.input_sec * 100]
        if data.get("center_w_cache_manifest_sha256") != expected_hash or Path(str(data.get("center_w_cache_path"))).resolve() != Path(str(cfg.data.center_w_cache_path)).resolve():
            raise RuntimeError(f"{spec.experiment_id} W cache identity 漂移")
    elif "center_w_cache_manifest_sha256" in data or "center_w_cache_path" in data:
        raise RuntimeError(f"{spec.experiment_id} C201 不得登记 W cache")


def _validate_history(history: pd.DataFrame, manifest: Mapping[str, Any], spec: ArmSpec, seed: int) -> None:
    required = {"epoch", "optimizer_update", "val_center90_rr_mae_bpm", "strictly_improved"}
    if len(history) != EXPECTED_EPOCHS or not required.issubset(history.columns) or history["epoch"].astype(int).tolist() != list(range(1, EXPECTED_EPOCHS + 1)):
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} train history 不完整")
    numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or history["optimizer_update"].astype(int).tolist() != list(range(80, EXPECTED_UPDATES + 1, 80)):
        raise FloatingPointError(f"{spec.experiment_id}/seed_{seed} history 非有限或 update identity 漂移")
    values = history["val_center90_rr_mae_bpm"].to_numpy(dtype=np.float64)
    expected_flags = np.zeros(len(values), dtype=bool)
    best = float("inf")
    for index, value in enumerate(values):
        if value < best:
            expected_flags[index] = True
            best = value
    flags = history["strictly_improved"].astype(str).str.lower().map({"true": True, "false": False})
    if flags.isna().any() or not np.array_equal(flags.to_numpy(dtype=bool), expected_flags):
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} strict selector trajectory 漂移")
    best_index = int(np.argmin(values))
    if int(manifest.get("best_epoch", -1)) != best_index + 1 or not np.isclose(float(manifest.get("best_center90_rr_mae_bpm", np.nan)), values[best_index], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} selector identity 漂移")


def _validate_metrics(metrics: pd.DataFrame, spec: ArmSpec, seed: int) -> pd.DataFrame:
    required = {"method", *METRICS_IDENTITY_COLUMNS, *CENTER90_PRIMARY_METRICS, "center90_ibi_coverage", "center90_ibi_target_eligible", "center90_ibi_interpretable"}
    if len(metrics) != EXPECTED_VALIDATION_ROWS or not required.issubset(metrics.columns):
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} validation metrics schema 漂移")
    if set(metrics["method"].astype(str)) != {spec.variant} or set(metrics["split"].astype(str)) != {"val"} or set(metrics["input_set"].astype(str)) != {"research_v2_waveform"}:
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} validation method/split/input_set 漂移")
    if dataset_row_ids_sha256(metrics["dataset_row_id"]) != EXPECTED_VALIDATION_ROW_IDS_SHA256:
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} validation row set 漂移")
    always_finite = [name for name in CENTER90_PRIMARY_METRICS if name != "center90_ibi_medae_sec"]
    if not np.isfinite(metrics[always_finite].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError(f"{spec.experiment_id}/seed_{seed} primary metrics 非有限")
    for column in ("center90_ibi_medae_sec", "center90_ibi_coverage"):
        if np.isinf(pd.to_numeric(metrics[column], errors="coerce").to_numpy(dtype=np.float64)).any():
            raise FloatingPointError(f"{spec.experiment_id}/seed_{seed} {column} 含 Inf")
    return metrics.loc[:, METRICS_IDENTITY_COLUMNS].sort_values("dataset_row_id").reset_index(drop=True)


def _validate_stored_summary(stored: pd.DataFrame, recomputed: pd.DataFrame, spec: ArmSpec, seed: int) -> None:
    if stored.shape != recomputed.shape or list(stored.columns) != list(recomputed.columns):
        raise RuntimeError(f"{spec.experiment_id}/seed_{seed} stored summary schema 漂移")
    for column in stored.columns:
        left, right = stored[column].iloc[0], recomputed[column].iloc[0]
        matches = int(left) == int(right) if isinstance(right, (int, np.integer)) else bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12, equal_nan=True))
        if not matches:
            raise RuntimeError(f"{spec.experiment_id}/seed_{seed} stored summary 重算不一致: {column}")


def _oriented_changes(from_value: float, to_value: float, metric: MetricSpec) -> tuple[float, float, float]:
    if not np.isfinite(from_value) or not np.isfinite(to_value) or from_value == 0.0:
        raise FloatingPointError(f"{metric.name} change 输入非法")
    absolute = from_value - to_value if metric.direction == "minimize" else to_value - from_value
    relative = absolute / abs(from_value)
    return relative, absolute, relative if metric.material_kind == "relative" else absolute


def _change_record(seed: int, model_id: str, from_sec: int, to_sec: int, metric: MetricSpec, from_value: float, to_value: float, relative: float, absolute: float, protocol_value: float) -> dict[str, Any]:
    return {"seed": seed, "model_id": model_id, "from_sec": from_sec, "to_sec": to_sec, "metric": metric.name, "direction": metric.direction, "from_value": from_value, "to_value": to_value, "raw_delta_to_minus_from": to_value - from_value, "oriented_relative_change": relative, "oriented_absolute_change": absolute, "material_kind": metric.material_kind, "material_threshold": metric.material_threshold, "protocol_material_value": protocol_value, "materially_improved": protocol_value >= metric.material_threshold, "materially_worsened": protocol_value <= -metric.material_threshold}


def _require_three_seeds(frame: pd.DataFrame, label: str) -> None:
    observed = tuple(int(value) for value in frame["seed"].sort_values())
    if observed != CENTER90_FORMAL_SEEDS:
        raise RuntimeError(f"{label} seeds 未闭合: {observed}")


def _assert_finite(*frames: pd.DataFrame) -> None:
    for frame in frames:
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all():
            raise FloatingPointError("center90 summary output 含 NaN/Inf")


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [{"filename": path.name, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)} for path in sorted(directory.iterdir()) if path.is_file()]


def _load_json(path: Path) -> dict[str, Any]:
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


__all__ = ["ARM_SPECS", "FORMAL_RUN_COMMIT", "METRIC_SPECS", "OUTPUT_FILENAMES", "SUMMARY_SCHEMA_VERSION", "audit_formal_runs", "build_paired_directions", "build_seed_length_changes", "build_seed_model_differences", "build_three_seed_arm_summary", "run_center90_validation_summary"]
