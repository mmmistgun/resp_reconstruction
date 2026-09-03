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
from omegaconf import OmegaConf

from resp_train.paper_evidence.center30_config import (
    CENTER30_P4S_OUTPUT_ROOT,
    CENTER30_PROTOCOL_ID,
    CENTER30_W_CACHE_MANIFEST_SHA256,
    CENTER30_W_CACHE_PATHS,
    load_center30_config,
)
from resp_train.paper_evidence.center30_experiment import center30_experiment_id
from resp_train.paper_evidence.center30_metrics import (
    CENTER30_PRIMARY_METRICS,
    summarize_center30_metrics,
)
P4S_SUMMARY_SCHEMA_VERSION = "paper-center30-context-p4s-single-seed-summary-v1"
P4S_RUN_COMMIT = "bead33307aa79139b8124bdf515700bf7e18379b"
P4S_SEED = 20260811
EXPECTED_VALIDATION_ROWS = 2675
EXPECTED_EPOCHS = 80
EXPECTED_UPDATES = 6400
EXPECTED_TRAIN_ROW_HASH = "f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e"
EXPECTED_VAL_ROW_HASH = "b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a"
EXPECTED_DATASET_HASH = "f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f"
DEFAULT_RUN_ROOT = Path(CENTER30_P4S_OUTPUT_ROOT)
DEFAULT_OUTPUT_DIR = Path("runs/paper_evidence_v1/center30_context/p4s_single_seed_summary")


@dataclass(frozen=True)
class ArmSpec:
    experiment_id: str
    config_filename: str
    model_id: str
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
        experiment_id=f"C30V1_{prefix}_{input_sec}",
        config_filename=f"p4s_c30v1_{short}_{input_sec}.yaml",
        model_id=model_id,
        variant=variant,
        input_sec=input_sec,
        parameter_count=parameter_count,
        w_scale_count=w_scale_count,
    )
    for prefix, short, model_id, variant, parameter_count, w_scale_count in (
        ("C201", "c201", "C201", "c201_center30", 1069802, 0),
        ("WR", "wr", "W-reduced", "w_reduced_center30", 1219850, 49),
    )
    for input_sec in (30, 45, 60, 90)
)
METRIC_SPECS = (
    MetricSpec("center30_rr_mae_bpm", "minimize", 0.005, "relative"),
    MetricSpec("center30_ibi_medae_sec", "minimize", 0.005, "relative"),
    MetricSpec("center30_envelope_trajectory_mae", "minimize", 0.005, "relative"),
    MetricSpec("center30_global_envelope_modulation_error", "minimize", 0.005, "relative"),
    MetricSpec("center30_lag_aware_signed_pcc", "maximize", 0.002, "absolute"),
)
CONTRASTS = ((30, 45), (30, 60), (30, 90), (45, 60), (60, 90))
EXPECTED_RUNTIME = {
    "device_type": "cuda",
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_benchmark": False,
    "amp_enabled": True,
    "amp_dtype": "bfloat16",
}
EXPECTED_INPUT_SLICES = {
    30: [7500, 10500],
    45: [6750, 11250],
    60: [6000, 12000],
    90: [4500, 13500],
}
REQUIRED_ARTIFACTS = {
    "checkpoint_best_center30_rr.pt",
    "checkpoint_final.pt",
    "data_identity.json",
    "optimizer_parameter_groups.json",
    "resolved_config.yaml",
    "runtime_identity.json",
    "train.log",
    "train_history.csv",
    "validation_center30_metrics.csv",
    "validation_center30_metrics_summary.csv",
}
IDENTITY_COLUMNS = ("dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id")
OUTPUT_FILENAMES = (
    "p4s_arm_validation_summary.csv",
    "p4s_relative_changes.csv",
    "p4s_companion_relative_changes.csv",
    "p4s_model_differences.csv",
    "p4s_decision.json",
    "summary_receipt.json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_p4s_runs(run_root: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = Path(run_root).resolve()
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None
    for spec in ARM_SPECS:
        run_dir = root / spec.experiment_id / f"seed_{P4S_SEED}"
        row, input_record, identity = audit_formal_arm(
            run_dir,
            spec,
            seed=P4S_SEED,
            expected_commit=P4S_RUN_COMMIT,
            config_filename=spec.config_filename,
            expected_stage="p4s_single_seed",
            expected_gate="p4s_formal",
            expected_output_root=CENTER30_P4S_OUTPUT_ROOT,
        )
        if reference_identity is None:
            reference_identity = identity
        elif not identity.equals(reference_identity):
            raise RuntimeError(f"{spec.experiment_id} validation row/metadata identity 不一致")
        rows.append(row)
        inputs.append(input_record)
    return pd.DataFrame.from_records(rows), inputs


def build_relative_changes(arm_summary: pd.DataFrame) -> pd.DataFrame:
    indexed = arm_summary.set_index(["model_id", "input_sec"])
    records: list[dict[str, Any]] = []
    for model_id in ("C201", "W-reduced"):
        for from_sec, to_sec in CONTRASTS:
            for metric in METRIC_SPECS:
                column = f"{metric.name}_mean"
                old = float(indexed.loc[(model_id, from_sec), column])
                new = float(indexed.loc[(model_id, to_sec), column])
                relative = _oriented_relative_change(old, new, metric.direction)
                absolute = (old - new) if metric.direction == "minimize" else (new - old)
                protocol_value = relative if metric.material_kind == "relative" else absolute
                records.append(
                    {
                        "model_id": model_id,
                        "from_sec": from_sec,
                        "to_sec": to_sec,
                        "metric": metric.name,
                        "direction": metric.direction,
                        "from_value": old,
                        "to_value": new,
                        "oriented_relative_change": relative,
                        "oriented_absolute_change": absolute,
                        "material_kind": metric.material_kind,
                        "material_threshold": metric.material_threshold,
                        "protocol_material_value": protocol_value,
                        "materially_improved": protocol_value >= metric.material_threshold,
                        "materially_worsened": protocol_value <= -metric.material_threshold,
                    }
                )
    return pd.DataFrame.from_records(records)


def build_companion_changes(arm_summary: pd.DataFrame) -> pd.DataFrame:
    indexed = arm_summary.set_index(["model_id", "input_sec"])
    records = []
    for model_id in ("C201", "W-reduced"):
        for from_sec, to_sec in CONTRASTS:
            for metric in ("center30_ibi_coverage_mean", "center30_ibi_interpretable_fraction"):
                old = float(indexed.loc[(model_id, from_sec), metric])
                new = float(indexed.loc[(model_id, to_sec), metric])
                records.append(
                    {
                        "model_id": model_id,
                        "from_sec": from_sec,
                        "to_sec": to_sec,
                        "metric": metric,
                        "from_value": old,
                        "to_value": new,
                        "relative_change": (new - old) / abs(old),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_model_differences(arm_summary: pd.DataFrame) -> pd.DataFrame:
    indexed = arm_summary.set_index(["model_id", "input_sec"])
    records = []
    for input_sec in (30, 45, 60, 90):
        for metric in METRIC_SPECS:
            column = f"{metric.name}_mean"
            c201 = float(indexed.loc[("C201", input_sec), column])
            wr = float(indexed.loc[("W-reduced", input_sec), column])
            records.append(
                {
                    "input_sec": input_sec,
                    "metric": metric.name,
                    "direction": metric.direction,
                    "c201_value": c201,
                    "w_reduced_value": wr,
                    "w_reduced_oriented_relative_change": _oriented_relative_change(
                        c201, wr, metric.direction
                    ),
                    "w_reduced_minus_c201": wr - c201,
                }
            )
    return pd.DataFrame.from_records(records)


def build_decision(arm_summary: pd.DataFrame, changes: pd.DataFrame) -> dict[str, Any]:
    rr = changes.loc[changes["metric"].eq("center30_rr_mae_bpm")].set_index(
        ["model_id", "from_sec", "to_sec"]
    )
    baseline_signals: dict[str, Any] = {}
    best_inputs: dict[str, int] = {}
    for model_id in ("C201", "W-reduced"):
        baseline = {
            str(to_sec): float(rr.loc[(model_id, 30, to_sec), "oriented_relative_change"])
            for to_sec in (45, 60, 90)
        }
        baseline_signals[model_id] = baseline
        model_rows = arm_summary.loc[arm_summary["model_id"].eq(model_id)]
        best_inputs[model_id] = int(
            model_rows.loc[model_rows["center30_rr_mae_bpm_mean"].idxmin(), "input_sec"]
        )
    all_longer_materially_better = all(
        value >= 0.005 for values in baseline_signals.values() for value in values.values()
    )
    return {
        "protocol_gate": "p4s_single_seed_directional_diagnostic",
        "rr_relative_improvement_from_30s": baseline_signals,
        "all_models_all_longer_inputs_materially_improve_rr_vs_30s": all_longer_materially_better,
        "best_rr_input_sec_by_model": best_inputs,
        "candidate_interpretation": (
            "30s_input_inferior_45_to_60_boundary_representation_dependent"
            if all_longer_materially_better and len(set(best_inputs.values())) > 1
            else "short_window_direction_requires_review"
        ),
        "single_seed_direction_only": True,
        "stable_minimum_context_claim_allowed": False,
        "p4s_three_seed_extension_recommended": all_longer_materially_better,
        "p4s_three_seed_extension_authorized": False,
        "additional_runs_if_authorized": 16,
        "model_or_length_selection_performed": False,
        "weighted_score_used": False,
        "research_test_authorized": False,
    }


def run_p4s_summary(
    *,
    repo_root: str | Path,
    command: str,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    require_clean_git: bool = True,
) -> Path:
    root = Path(repo_root).resolve()
    source = _resolve(root, run_root)
    output = _resolve(root, output_dir)
    if output.exists():
        raise FileExistsError(f"P4-S2 汇总输出禁止覆盖: {output}")
    git = _git_identity(root)
    if require_clean_git and (git["error"] is not None or git["dirty"] is not False):
        raise RuntimeError("P4-S2 冻结汇总要求干净 Git 工作树")
    arm_summary, inputs = audit_p4s_runs(source)
    changes = build_relative_changes(arm_summary)
    companion = build_companion_changes(arm_summary)
    differences = build_model_differences(arm_summary)
    decision = build_decision(arm_summary, changes)
    _assert_finite(arm_summary, changes, companion, differences)

    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        arm_summary.to_csv(temporary / "p4s_arm_validation_summary.csv", index=False)
        changes.to_csv(temporary / "p4s_relative_changes.csv", index=False)
        companion.to_csv(temporary / "p4s_companion_relative_changes.csv", index=False)
        differences.to_csv(temporary / "p4s_model_differences.csv", index=False)
        _write_json(temporary / "p4s_decision.json", decision)
        receipt = {
            "schema_version": P4S_SUMMARY_SCHEMA_VERSION,
            "protocol_id": CENTER30_PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": "single-seed validation directional short-window diagnostic",
            "execution": {"command": command, "cwd": str(root), "git": git, "run_commit": P4S_RUN_COMMIT},
            "inputs": inputs,
            "counts": {
                "expected_runs": 8,
                "actual_runs": len(arm_summary),
                "epochs_per_run": EXPECTED_EPOCHS,
                "validation_rows_per_run": EXPECTED_VALIDATION_ROWS,
                "validation_metric_rows_read": EXPECTED_VALIDATION_ROWS * len(arm_summary),
                "relative_change_rows": len(changes),
                "companion_change_rows": len(companion),
                "model_difference_rows": len(differences),
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
            "artifacts": _artifact_records(temporary),
        }
        _write_json(temporary / "summary_receipt.json", receipt)
        _write_json(
            temporary / "artifact_manifest.json",
            {
                "schema_version": P4S_SUMMARY_SCHEMA_VERSION,
                "protocol_id": CENTER30_PROTOCOL_ID,
                "status": "complete",
                "read_only_validation_summary": True,
                "source_run_commit": P4S_RUN_COMMIT,
                "files": _artifact_records(temporary),
            },
        )
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
    expected_stage: str,
    expected_gate: str,
    expected_output_root: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"P4-S2 run 缺失: {run_dir}")
    lifecycle_path = run_dir / "lifecycle.json"
    manifest_path = run_dir / "artifact_manifest.json"
    lifecycle = _load_json(lifecycle_path)
    manifest = _load_json(manifest_path)
    if lifecycle != {"status": "complete"}:
        raise RuntimeError(f"{spec.experiment_id} lifecycle 未 complete")
    expected = {
        "protocol_id": CENTER30_PROTOCOL_ID,
        "task": "paper_center30_context_v1",
        "experiment_id": spec.experiment_id,
        "seed": seed,
        "parameter_count": spec.parameter_count,
        "selector": "full_validation_center30_rr_mae_bpm_strict_lower_tie_earlier",
        "train_access": True,
        "validation_access": True,
        "test_access": False,
        "test_cache_created": False,
        "outer_target_supervision": False,
        "checkpoint_initialization_used": False,
        "scientific_role": "auxiliary_short_window_context_sensitivity_outside_main_model_experiment",
        "strict_w0_equivalent": False,
        "w_scale_count": spec.w_scale_count,
        "early_stopping": False,
        "resume": False,
        "runtime": EXPECTED_RUNTIME,
        "git": {"commit": expected_commit, "dirty": False, "error": None},
        "command": [
            "scripts/train_paper_center30_context_v1.py",
            "--config",
            f"configs/paper_evidence_v1/{config_filename}",
            "--confirm-formal-training",
        ],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"{spec.experiment_id} manifest {key} 漂移")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise RuntimeError(f"{spec.experiment_id} artifacts 不完整")
    for filename, record in artifacts.items():
        path = run_dir / filename
        if Path(filename).name != filename or not path.is_file():
            raise RuntimeError(f"{spec.experiment_id} artifact path 非法或缺失")
        if path.stat().st_size != int(record.get("size_bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"{spec.experiment_id} artifact hash/size 漂移: {filename}")
    cfg = load_center30_config(run_dir / "resolved_config.yaml")
    if center30_experiment_id(str(cfg.model.variant), int(cfg.window.input_samples)) != spec.experiment_id:
        raise RuntimeError(f"{spec.experiment_id} resolved config identity 漂移")
    if str(cfg.model.variant) != spec.variant or int(cfg.window.input_sec) != spec.input_sec:
        raise RuntimeError(f"{spec.experiment_id} model/length 漂移")
    for key, expected_value in {
        "protocol.stage": expected_stage,
        "protocol.execution_gate": expected_gate,
        "training.seed": seed,
        "model.initialization_seed": seed,
        "outputs.run_root": expected_output_root,
    }.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected_value:
            raise RuntimeError(f"{spec.experiment_id}/seed_{seed} resolved config {key} 漂移")
    if _load_json(run_dir / "runtime_identity.json") != EXPECTED_RUNTIME:
        raise RuntimeError(f"{spec.experiment_id} runtime identity 漂移")
    data = _load_json(run_dir / "data_identity.json")
    for key, value in {
        "train_rows": 10141,
        "val_rows": 2675,
        "train_row_ids_sha256": EXPECTED_TRAIN_ROW_HASH,
        "val_row_ids_sha256": EXPECTED_VAL_ROW_HASH,
        "row_overlap_count": 0,
        "samp_id_overlap_count": 0,
        "subject_session_overlap_count": 0,
        "views_share_parent_rows": True,
        "dataset_index_sha256": EXPECTED_DATASET_HASH,
        "input_samples": spec.input_sec * 100,
        "input_slice": EXPECTED_INPUT_SLICES[spec.input_sec],
        "target_slice": [7500, 10500],
        "test_access": False,
    }.items():
        if data.get(key) != value:
            raise RuntimeError(f"{spec.experiment_id} data identity {key} 漂移")
    if spec.variant == "w_reduced_center30":
        input_samples = spec.input_sec * 100
        if Path(str(data.get("center_w_cache_path"))).resolve() != Path(
            CENTER30_W_CACHE_PATHS[input_samples]
        ).resolve():
            raise RuntimeError(f"{spec.experiment_id} W cache path 漂移")
        if data.get("center_w_cache_manifest_sha256") != CENTER30_W_CACHE_MANIFEST_SHA256[input_samples]:
            raise RuntimeError(f"{spec.experiment_id} W cache manifest SHA-256 漂移")
    elif "center_w_cache_path" in data or "center_w_cache_manifest_sha256" in data:
        raise RuntimeError(f"{spec.experiment_id} C201 data identity 不得登记 W cache")
    history = pd.read_csv(run_dir / "train_history.csv")
    _validate_history(history, manifest, spec)
    metrics = pd.read_csv(run_dir / "validation_center30_metrics.csv")
    identity = _validate_metrics(metrics, spec)
    stored = pd.read_csv(run_dir / "validation_center30_metrics_summary.csv")
    recomputed = summarize_center30_metrics(metrics)
    _validate_stored_summary(stored, recomputed, spec)
    summary_row = stored.iloc[0].to_dict()
    if not np.isclose(
        float(manifest["best_center30_rr_mae_bpm"]),
        float(summary_row["center30_rr_mae_bpm_mean"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{spec.experiment_id} selector 与 stored summary 不一致")
    row = {
        "experiment_id": spec.experiment_id,
        "model_id": spec.model_id,
        "variant": spec.variant,
        "input_sec": spec.input_sec,
        "seed": seed,
        "parameter_count": spec.parameter_count,
        "w_scale_count": spec.w_scale_count,
        "best_epoch": int(manifest["best_epoch"]),
        **summary_row,
    }
    input_record = {
        "experiment_id": spec.experiment_id,
        "seed": seed,
        "run_dir": str(run_dir),
        "lifecycle_status": "complete",
        "lifecycle_sha256": sha256_file(lifecycle_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "validation_metrics_sha256": artifacts["validation_center30_metrics.csv"]["sha256"],
        "validation_summary_sha256": artifacts["validation_center30_metrics_summary.csv"]["sha256"],
        "checkpoint_content_read": False,
    }
    return row, input_record, identity


def _validate_history(history: pd.DataFrame, manifest: Mapping[str, Any], spec: ArmSpec) -> None:
    required = {"epoch", "optimizer_update", "val_center30_rr_mae_bpm"}
    if len(history) != EXPECTED_EPOCHS or not required.issubset(history.columns):
        raise RuntimeError(f"{spec.experiment_id} history 不完整")
    if history["epoch"].astype(int).tolist() != list(range(1, EXPECTED_EPOCHS + 1)):
        raise RuntimeError(f"{spec.experiment_id} epoch identity 漂移")
    numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or int(history.iloc[-1]["optimizer_update"]) != EXPECTED_UPDATES:
        raise FloatingPointError(f"{spec.experiment_id} history 非有限或 updates 未闭合")
    values = history["val_center30_rr_mae_bpm"].to_numpy(dtype=np.float64)
    best = int(np.argmin(values))
    if int(manifest["best_epoch"]) != best + 1 or not np.isclose(
        float(manifest["best_center30_rr_mae_bpm"]), values[best], rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(f"{spec.experiment_id} selector/history 不一致")


def _validate_metrics(metrics: pd.DataFrame, spec: ArmSpec) -> pd.DataFrame:
    required = {"method", *IDENTITY_COLUMNS, *CENTER30_PRIMARY_METRICS, "center30_ibi_coverage"}
    if len(metrics) != EXPECTED_VALIDATION_ROWS or not required.issubset(metrics.columns):
        raise RuntimeError(f"{spec.experiment_id} validation metrics schema/rows 漂移")
    if set(metrics["method"].astype(str)) != {spec.variant} or set(metrics["split"].astype(str)) != {"val"}:
        raise RuntimeError(f"{spec.experiment_id} method/split identity 漂移")
    if set(metrics["input_set"].astype(str)) != {"research_v2_waveform"}:
        raise RuntimeError(f"{spec.experiment_id} input_set 漂移")
    if metrics["dataset_row_id"].nunique(dropna=False) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(f"{spec.experiment_id} validation dataset_row_id 不唯一")
    row_ids = np.sort(metrics["dataset_row_id"].to_numpy(dtype=np.int64))
    if hashlib.sha256(row_ids.tobytes(order="C")).hexdigest() != EXPECTED_VAL_ROW_HASH:
        raise RuntimeError(f"{spec.experiment_id} validation row identity 漂移")
    always_finite = [metric for metric in CENTER30_PRIMARY_METRICS if metric != "center30_ibi_medae_sec"]
    if not np.isfinite(metrics[always_finite].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError(f"{spec.experiment_id} primary metrics 非有限")
    if np.isinf(pd.to_numeric(metrics["center30_ibi_medae_sec"], errors="coerce")).any():
        raise FloatingPointError(f"{spec.experiment_id} IBI 含 Inf")
    return metrics.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)


def _validate_stored_summary(stored: pd.DataFrame, recomputed: pd.DataFrame, spec: ArmSpec) -> None:
    if stored.shape != recomputed.shape or list(stored.columns) != list(recomputed.columns):
        raise RuntimeError(f"{spec.experiment_id} stored summary schema 漂移")
    for column in stored.columns:
        if not np.isclose(
            float(stored[column].iloc[0]),
            float(recomputed[column].iloc[0]),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(f"{spec.experiment_id} stored summary 重算不一致: {column}")


def _oriented_relative_change(old: float, new: float, direction: str) -> float:
    if not np.isfinite(old) or not np.isfinite(new) or old == 0.0:
        raise ValueError("相对变化要求有限且非零 baseline")
    return ((old - new) if direction == "minimize" else (new - old)) / abs(old)


def _assert_finite(*frames: pd.DataFrame) -> None:
    for frame in frames:
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all():
            raise FloatingPointError("P4-S2 summary output 含 NaN/Inf")


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {"filename": path.name, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: str | Path) -> Path:
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
    "CONTRASTS",
    "P4S_RUN_COMMIT",
    "P4S_SUMMARY_SCHEMA_VERSION",
    "audit_formal_arm",
    "audit_p4s_runs",
    "build_companion_changes",
    "build_decision",
    "build_model_differences",
    "build_relative_changes",
    "run_p4s_summary",
]
