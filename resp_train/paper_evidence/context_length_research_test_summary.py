from __future__ import annotations

import json
import os
import shlex
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

from resp_train.paper_evidence.center30_metrics import summarize_center30_metrics
from resp_train.paper_evidence.center30_summary import METRIC_SPECS as CENTER30_METRIC_SPECS
from resp_train.paper_evidence.center_context_metrics import summarize_center_metrics
from resp_train.paper_evidence.center_context_summary import METRIC_SPECS as CENTER60_METRIC_SPECS
from resp_train.paper_evidence.context_length_research_test import (
    EVALUATION_SCHEMA_VERSION,
    EXPECTED_RUNTIME,
    FROZEN_CACHE_MANIFEST_SHA256,
    FROZEN_CACHE_ROOT,
    FORMAL_SEEDS,
    OUTPUT_ROOT,
    REPO_ROOT,
    EvaluationSpec,
    _validate_metrics,
    expected_research_test_evaluations,
)
from resp_train.paper_evidence.context_length_research_test_cache import (
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


SUMMARY_SCHEMA_VERSION = "paper-context-length-research-test-summary-v1"
EVALUATION_COMMIT = "52d723891e34ad5051f18bdf93f2b0fbe4eb9d7f"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/paper_evidence_v1/context_length_research_test_summary"
MODEL_LABELS = {"c201": "C201", "wr": "W-reduced"}
TASK_CONFIG: dict[str, dict[str, Any]] = {
    "center30": {
        "output_sec": 30,
        "prefix": "center30",
        "metric_specs": CENTER30_METRIC_SPECS,
        "contrasts": ((30, 45), (30, 60), (30, 90), (45, 60), (60, 90)),
        "summarize": summarize_center30_metrics,
    },
    "center60": {
        "output_sec": 60,
        "prefix": "center",
        "metric_specs": CENTER60_METRIC_SPECS,
        "contrasts": ((60, 90), (90, 180), (60, 180)),
        "summarize": summarize_center_metrics,
    },
}
EXPECTED_EVALUATION_FILES = {
    "artifact_manifest.json",
    "checkpoint_identity.json",
    "evaluation_receipt.json",
    "lifecycle.json",
    "research_test_metrics.csv",
    "research_test_metrics_summary.csv",
    "resolved_evaluation_config.yaml",
    "runtime_identity.json",
    "test_data_identity.json",
}
MANIFEST_ARTIFACT_FILES = EXPECTED_EVALUATION_FILES - {"artifact_manifest.json", "lifecycle.json"}
RECEIPT_ARTIFACT_FILES = MANIFEST_ARTIFACT_FILES - {"evaluation_receipt.json"}
OUTPUT_FILENAMES = tuple(
    f"{task}_{suffix}"
    for task in ("center30", "center60")
    for suffix in (
        "seed_arm_test_summary.csv",
        "arm_three_seed_test_summary.csv",
        "seed_relative_changes.csv",
        "paired_seed_relative_directions.csv",
        "seed_companion_changes.csv",
        "paired_seed_companion_directions.csv",
        "decision.json",
    )
) + ("joint_decision.json", "summary_receipt.json")


def audit_research_test_evaluations(
    source_root: str | Path = OUTPUT_ROOT,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    source = Path(source_root).resolve()
    allowlist = expected_research_test_evaluations()
    task_rows: dict[str, list[dict[str, Any]]] = {"center30": [], "center60": []}
    inputs: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None
    reference_dependencies: Mapping[str, Any] | None = None

    for key in sorted(allowlist):
        spec = allowlist[key]
        run_dir = source / spec.task / spec.experiment_id / f"seed_{spec.seed}"
        row, input_record, identity, dependencies = _audit_evaluation(run_dir, spec)
        if reference_identity is None:
            reference_identity = identity
        elif not identity.equals(reference_identity):
            raise RuntimeError(f"{spec.identity} test row/metadata identity 与首项不一致")
        if reference_dependencies is None:
            reference_dependencies = dependencies
        elif dict(dependencies) != dict(reference_dependencies):
            raise RuntimeError(f"{spec.identity} dependency identity 漂移")
        task_rows[spec.task].append(row)
        inputs.append(input_record)

    frames = {
        task: pd.DataFrame.from_records(rows).sort_values(["model_id", "input_sec", "seed"])
        for task, rows in task_rows.items()
    }
    if len(frames["center30"]) != 24 or len(frames["center60"]) != 18 or len(inputs) != 42:
        raise RuntimeError("research-test evaluation 必须恰好闭合 center30=24、center60=18、总计42项")
    for task, frame in frames.items():
        if frame[["experiment_id", "seed"]].duplicated().any():
            raise RuntimeError(f"{task} evaluation identity 重复")
        for experiment_id, group in frame.groupby("experiment_id"):
            _require_three_seeds(group, label=f"{task}/{experiment_id}")
    return frames, inputs


def _audit_evaluation(
    run_dir: Path, spec: EvaluationSpec
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, Mapping[str, Any]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"research-test evaluation 缺失: {run_dir}")
    observed_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    if observed_files != EXPECTED_EVALUATION_FILES:
        raise RuntimeError(f"{spec.identity} evaluation 文件集合漂移: {sorted(observed_files)}")
    lifecycle_path = run_dir / "lifecycle.json"
    manifest_path = run_dir / "artifact_manifest.json"
    receipt_path = run_dir / "evaluation_receipt.json"
    lifecycle = _read_json(lifecycle_path)
    manifest = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    if lifecycle != {"status": "complete"}:
        raise RuntimeError(f"{spec.identity} lifecycle 未 complete")
    expected_manifest_identity = {
        "task": spec.task,
        "experiment_id": spec.experiment_id,
        "seed": spec.seed,
    }
    if (
        manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "complete"
        or manifest.get("identity") != expected_manifest_identity
    ):
        raise RuntimeError(f"{spec.identity} artifact manifest identity 漂移")
    manifest_records = _records_by_name(manifest.get("files"), label=f"{spec.identity}/manifest")
    if set(manifest_records) != MANIFEST_ARTIFACT_FILES:
        raise RuntimeError(f"{spec.identity} artifact manifest 文件集合不完整")
    _verify_artifact_records(run_dir, manifest_records)

    if (
        receipt.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "complete"
        or receipt.get("identity") != asdict(spec)
    ):
        raise RuntimeError(f"{spec.identity} evaluation receipt identity 漂移")
    receipt_records = _records_by_name(receipt.get("artifacts"), label=f"{spec.identity}/receipt")
    if set(receipt_records) != RECEIPT_ARTIFACT_FILES:
        raise RuntimeError(f"{spec.identity} evaluation receipt artifact 集合不完整")
    _verify_artifact_records(run_dir, receipt_records)
    for filename in RECEIPT_ARTIFACT_FILES:
        if receipt_records[filename] != manifest_records[filename]:
            raise RuntimeError(f"{spec.identity} receipt/manifest artifact 记录不一致: {filename}")

    execution = receipt.get("execution", {})
    if (
        execution.get("cwd") != str(REPO_ROOT)
        or execution.get("git_commit") != EVALUATION_COMMIT
        or execution.get("git_dirty") is not False
    ):
        raise RuntimeError(f"{spec.identity} evaluation Git/cwd identity 漂移")
    _validate_command(str(execution.get("command", "")), spec)
    runtime = execution.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError(f"{spec.identity} runtime identity 缺失")
    _validate_runtime(runtime, spec)
    if _read_json(run_dir / "runtime_identity.json") != dict(runtime):
        raise RuntimeError(f"{spec.identity} runtime receipt/file 不一致")
    dependencies = execution.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise RuntimeError(f"{spec.identity} dependency identity 缺失")

    data = receipt.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError(f"{spec.identity} test data identity 缺失")
    _validate_data_identity(data, spec)
    if _read_json(run_dir / "test_data_identity.json") != dict(data):
        raise RuntimeError(f"{spec.identity} test data receipt/file 不一致")
    if _read_json(run_dir / "checkpoint_identity.json") != asdict(spec):
        raise RuntimeError(f"{spec.identity} checkpoint identity file 漂移")
    _validate_resolved_config(run_dir / "resolved_evaluation_config.yaml", spec)

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
    if receipt.get("access") != expected_access or receipt.get("counts") != {
        "test_rows": EXPECTED_TEST_WINDOWS,
        "expected_test_rows": EXPECTED_TEST_WINDOWS,
    }:
        raise RuntimeError(f"{spec.identity} access/count identity 漂移")

    metrics_path = run_dir / "research_test_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    _validate_metrics(metrics, spec)
    identity = metrics[["dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id"]].copy()
    stored = pd.read_csv(run_dir / "research_test_metrics_summary.csv")
    summarize = TASK_CONFIG[spec.task]["summarize"]
    recomputed = summarize(metrics)
    _assert_summary_matches(stored, recomputed, label=str(spec.identity))
    summary_row = stored.iloc[0].to_dict()
    if int(summary_row["n_samples"]) != EXPECTED_TEST_WINDOWS:
        raise RuntimeError(f"{spec.identity} stored summary sample count 漂移")
    row = {
        "task": spec.task,
        "experiment_id": spec.experiment_id,
        "model_key": spec.model_key,
        "model_id": MODEL_LABELS[spec.model_key],
        "variant": spec.variant,
        "input_sec": spec.input_sec,
        "output_sec": spec.output_sec,
        "seed": spec.seed,
        "selected_epoch": spec.selected_epoch,
        "parameter_count": spec.parameter_count,
        "physical_gpu": str(runtime["cuda_visible_devices"]),
        **summary_row,
    }
    input_record = {
        "task": spec.task,
        "experiment_id": spec.experiment_id,
        "model_key": spec.model_key,
        "input_sec": spec.input_sec,
        "seed": spec.seed,
        "evaluation_dir": str(run_dir),
        "evaluation_commit": EVALUATION_COMMIT,
        "physical_gpu": str(runtime["cuda_visible_devices"]),
        "lifecycle_sha256": sha256_file(lifecycle_path),
        "evaluation_receipt_sha256": sha256_file(receipt_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "research_test_metrics_sha256": manifest_records["research_test_metrics.csv"]["sha256"],
        "research_test_summary_sha256": manifest_records["research_test_metrics_summary.csv"]["sha256"],
        "checkpoint_sha256": spec.checkpoint_sha256,
        "checkpoint_content_read_by_summary": False,
    }
    return row, input_record, identity, dependencies


def build_three_seed_arm_summary(task: str, seed_rows: pd.DataFrame) -> pd.DataFrame:
    config = TASK_CONFIG[task]
    metric_columns = [
        *(f"{metric.name}_mean" for metric in config["metric_specs"]),
        f"{config['prefix']}_ibi_coverage_mean",
        f"{config['prefix']}_ibi_interpretable_fraction",
    ]
    records: list[dict[str, Any]] = []
    for experiment_id, group in seed_rows.groupby("experiment_id", sort=False):
        group = group.sort_values("seed")
        _require_three_seeds(group, label=f"{task}/{experiment_id}")
        first = group.iloc[0]
        record: dict[str, Any] = {
            "task": task,
            "experiment_id": str(experiment_id),
            "model_id": str(first["model_id"]),
            "variant": str(first["variant"]),
            "input_sec": int(first["input_sec"]),
            "output_sec": int(first["output_sec"]),
            "parameter_count": int(first["parameter_count"]),
            "seeds": "|".join(str(int(value)) for value in group["seed"]),
            "selected_epochs": "|".join(str(int(value)) for value in group["selected_epoch"]),
        }
        for column in metric_columns:
            values = group[column].to_numpy(dtype=np.float64)
            record[f"{column}_seed_mean"] = float(np.mean(values))
            record[f"{column}_seed_sample_sd"] = float(np.std(values, ddof=1))
            record[f"{column}_seed_n"] = len(values)
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_seed_relative_changes(task: str, seed_rows: pd.DataFrame) -> pd.DataFrame:
    config = TASK_CONFIG[task]
    records: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        frame = seed_rows.loc[seed_rows["seed"].eq(seed)].set_index(["model_id", "input_sec"])
        for model_id in ("C201", "W-reduced"):
            for from_sec, to_sec in config["contrasts"]:
                for metric in config["metric_specs"]:
                    column = f"{metric.name}_mean"
                    old = float(frame.loc[(model_id, from_sec), column])
                    new = float(frame.loc[(model_id, to_sec), column])
                    relative = _oriented_relative_change(old, new, metric.direction)
                    absolute = old - new if metric.direction == "minimize" else new - old
                    protocol_value = relative if metric.material_kind == "relative" else absolute
                    records.append(
                        {
                            "task": task,
                            "seed": seed,
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


def build_paired_primary_directions(seed_changes: pd.DataFrame) -> pd.DataFrame:
    groups = ("task", "model_id", "from_sec", "to_sec", "metric")
    records: list[dict[str, Any]] = []
    for identity, group in seed_changes.groupby(list(groups), sort=False):
        group = group.sort_values("seed")
        _require_three_seeds(group, label=str(identity))
        protocol_values = group["protocol_material_value"].to_numpy(dtype=np.float64)
        relative_values = group["oriented_relative_change"].to_numpy(dtype=np.float64)
        absolute_values = group["oriented_absolute_change"].to_numpy(dtype=np.float64)
        threshold = float(group["material_threshold"].iloc[0])
        records.append(
            {
                **dict(zip(groups, identity, strict=True)),
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


def build_seed_companion_changes(task: str, seed_rows: pd.DataFrame) -> pd.DataFrame:
    prefix = TASK_CONFIG[task]["prefix"]
    metrics = (f"{prefix}_ibi_coverage_mean", f"{prefix}_ibi_interpretable_fraction")
    records: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        frame = seed_rows.loc[seed_rows["seed"].eq(seed)].set_index(["model_id", "input_sec"])
        for model_id in ("C201", "W-reduced"):
            for from_sec, to_sec in TASK_CONFIG[task]["contrasts"]:
                for metric in metrics:
                    old = float(frame.loc[(model_id, from_sec), metric])
                    new = float(frame.loc[(model_id, to_sec), metric])
                    records.append(
                        {
                            "task": task,
                            "seed": seed,
                            "model_id": model_id,
                            "from_sec": from_sec,
                            "to_sec": to_sec,
                            "metric": metric,
                            "from_value": old,
                            "to_value": new,
                            "raw_delta_to_minus_from": new - old,
                            "relative_change": (new - old) / abs(old),
                        }
                    )
    return pd.DataFrame.from_records(records)


def build_paired_companion_directions(seed_changes: pd.DataFrame) -> pd.DataFrame:
    groups = ("task", "model_id", "from_sec", "to_sec", "metric")
    records: list[dict[str, Any]] = []
    for identity, group in seed_changes.groupby(list(groups), sort=False):
        group = group.sort_values("seed")
        _require_three_seeds(group, label=str(identity))
        relative = group["relative_change"].to_numpy(dtype=np.float64)
        absolute = group["raw_delta_to_minus_from"].to_numpy(dtype=np.float64)
        records.append(
            {
                **dict(zip(groups, identity, strict=True)),
                "relative_change_seed_mean": float(np.mean(relative)),
                "relative_change_seed_sample_sd": float(np.std(relative, ddof=1)),
                "raw_delta_seed_mean": float(np.mean(absolute)),
                "raw_delta_seed_sample_sd": float(np.std(absolute, ddof=1)),
                "increased_seed_count": int(np.sum(absolute > 0.0)),
                "decreased_seed_count": int(np.sum(absolute < 0.0)),
                "equal_seed_count": int(np.sum(absolute == 0.0)),
            }
        )
    return pd.DataFrame.from_records(records)


def build_task_decision(task: str, seed_changes: pd.DataFrame) -> dict[str, Any]:
    rr_metric = "center30_rr_mae_bpm" if task == "center30" else "center_rr_mae_bpm"
    rr = seed_changes.loc[seed_changes["metric"].eq(rr_metric)]
    counts: dict[str, dict[str, int]] = {}
    for model_id in ("C201", "W-reduced"):
        model = rr.loc[rr["model_id"].eq(model_id)]
        counts[model_id] = {
            f"{from_sec}_to_{to_sec}": int(
                model.loc[
                    model["from_sec"].eq(from_sec) & model["to_sec"].eq(to_sec),
                    "materially_improved",
                ].sum()
            )
            for from_sec, to_sec in TASK_CONFIG[task]["contrasts"]
        }
    if task == "center30":
        first_gain = all(counts[model]["30_to_45"] == 3 for model in counts)
        continued = all(
            counts[model][contrast] == 3
            for model in counts
            for contrast in ("45_to_60", "60_to_90")
        )
        if first_gain and not continued:
            interpretation = "independent_test_supports_45s_rr_priority_reasonable_lower_bound"
        elif first_gain:
            interpretation = "test_confirms_30_to_45_rr_gain_but_gain_continues_beyond_45s"
        else:
            interpretation = "independent_test_does_not_confirm_45s_rr_priority_reasonable_lower_bound"
        return {
            "evidence_scope": "center30_three_seed_independent_test_context_sensitivity",
            "rr_materially_improved_seed_count": counts,
            "rr_30_to_45_materially_improved_in_both_models_all_seeds": first_gain,
            "rr_stepwise_gain_beyond_45_materially_improved_in_both_models_all_seeds": continued,
            "validation_claim": "45s_is_rr_priority_reasonable_input_lower_bound",
            "test_interpretation": interpretation,
            **_decision_guardrails(),
        }
    universal = [
        contrast
        for contrast in ("60_to_90", "90_to_180", "60_to_180")
        if all(counts[model][contrast] == 3 for model in counts)
    ]
    monotonic = all(
        counts[model][contrast] == 3
        for model in counts
        for contrast in ("60_to_90", "90_to_180")
    )
    if monotonic:
        interpretation = "independent_test_indicates_stable_stepwise_longer_input_rr_benefit"
    elif universal:
        interpretation = "independent_test_shows_uniform_rr_benefit_for_selected_longer_contrast_only"
    else:
        interpretation = "independent_test_supports_no_stable_longer_input_rr_benefit"
    return {
        "evidence_scope": "center60_three_seed_independent_test_context_sensitivity",
        "rr_materially_improved_seed_count": counts,
        "rr_uniformly_material_contrasts_both_models_all_seeds": universal,
        "rr_stable_stepwise_monotonic_gain_both_models_all_seeds": monotonic,
        "validation_claim": "longer_input_has_no_stable_rr_benefit",
        "test_interpretation": interpretation,
        **_decision_guardrails(),
    }


def run_research_test_summary(
    *,
    repo_root: str | Path,
    command: str,
    source_root: str | Path = OUTPUT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    require_clean_git: bool = True,
) -> Path:
    root = Path(repo_root).resolve()
    source = _resolve(root, source_root)
    output = _resolve(root, output_dir)
    if output.exists():
        raise FileExistsError(f"上下文 research-test 汇总输出禁止覆盖: {output}")
    git = _git_identity(root)
    if require_clean_git and (git["error"] is not None or git["dirty"] is not False):
        raise RuntimeError("上下文 research-test 冻结汇总要求干净 Git 工作树")

    seed_frames, inputs = audit_research_test_evaluations(source)
    products: dict[str, pd.DataFrame | dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for task, seed_rows in seed_frames.items():
        arm_summary = build_three_seed_arm_summary(task, seed_rows)
        seed_changes = build_seed_relative_changes(task, seed_rows)
        paired_changes = build_paired_primary_directions(seed_changes)
        seed_companion = build_seed_companion_changes(task, seed_rows)
        paired_companion = build_paired_companion_directions(seed_companion)
        decision = build_task_decision(task, seed_changes)
        _assert_finite(seed_rows, arm_summary, seed_changes, paired_changes, seed_companion, paired_companion)
        products.update(
            {
                f"{task}_seed_arm_test_summary.csv": seed_rows,
                f"{task}_arm_three_seed_test_summary.csv": arm_summary,
                f"{task}_seed_relative_changes.csv": seed_changes,
                f"{task}_paired_seed_relative_directions.csv": paired_changes,
                f"{task}_seed_companion_changes.csv": seed_companion,
                f"{task}_paired_seed_companion_directions.csv": paired_companion,
                f"{task}_decision.json": decision,
            }
        )
        decisions[task] = decision
    joint_decision = {
        "evidence_scope": "two_separate_independent_test_panels",
        "center30": decisions["center30"],
        "center60": decisions["center60"],
        "cross_task_absolute_metric_comparison_performed": False,
        "weighted_score_used": False,
        "model_or_length_reselection_performed": False,
        "additional_training_or_test_search_authorized": False,
    }
    products["joint_decision.json"] = joint_decision

    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        for filename, value in products.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(temporary / filename, index=False)
            else:
                _write_json(temporary / filename, value)
        receipt = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": "two separate three-seed independent-test context-sensitivity panels",
            "execution": {
                "command": command,
                "cwd": str(root),
                "git": git,
                "evaluation_commit": EVALUATION_COMMIT,
            },
            "inputs": inputs,
            "counts": {
                "expected_evaluations": 42,
                "actual_evaluations": len(inputs),
                "center30_evaluations": len(seed_frames["center30"]),
                "center60_evaluations": len(seed_frames["center60"]),
                "test_rows_per_evaluation": EXPECTED_TEST_WINDOWS,
                "test_metric_rows_read": EXPECTED_TEST_WINDOWS * len(inputs),
            },
            "aggregation": {
                "tasks_kept_separate": True,
                "run_metric": "sample_direct_mean",
                "seed_mean": "arithmetic_mean",
                "seed_sd": "sample_sd_ddof1",
                "pairing": "same_seed_within_task_and_model",
                "error_material_threshold": "relative_0.5_percent",
                "pcc_material_threshold": "absolute_0.002",
                "weighted_score": False,
                "p_value": False,
            },
            "access": {
                "research_test_metrics_accessed": True,
                "research_test_stored_summary_accessed": True,
                "test_signal_or_target_array_accessed": False,
                "checkpoint_content_read": False,
                "validation_summary_provenance_accessed": True,
                "validation_signal_target_or_prediction_read": False,
                "dataset_or_index_accessed": False,
                "model_training_used": False,
                "model_inference_used": False,
                "gpu_used": False,
            },
            "decision": joint_decision,
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


def _validate_command(command: str, spec: EvaluationSpec) -> None:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise RuntimeError(f"{spec.identity} command 无法解析") from exc
    expected = [
        "scripts/eval_paper_context_length_research_test_v1.py",
        "--task",
        spec.task,
        "--model",
        spec.model_key,
        "--input-sec",
        str(spec.input_sec),
        "--seed",
        str(spec.seed),
        "--device",
        "cuda:0",
        "--confirm-research-test",
    ]
    if len(argv) < 2 or argv[1:] != expected:
        raise RuntimeError(f"{spec.identity} evaluation command identity 漂移")


def _validate_runtime(runtime: Mapping[str, Any], spec: EvaluationSpec) -> None:
    for key, value in EXPECTED_RUNTIME.items():
        if runtime.get(key) != value:
            raise RuntimeError(f"{spec.identity} runtime {key} 漂移")
    if (
        runtime.get("logical_device") != "cuda:0"
        or str(runtime.get("cuda_visible_devices")) not in {"0", "1"}
        or int(runtime.get("cuda_visible_device_count", -1)) != 1
        or not str(runtime.get("cuda_device_name", "")).strip()
    ):
        raise RuntimeError(f"{spec.identity} logical/physical GPU identity 漂移")


def _validate_data_identity(data: Mapping[str, Any], spec: EvaluationSpec) -> None:
    expected = {
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
        "output_samples": spec.output_sec * 100,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"{spec.identity} test data {key} 漂移")
    if spec.model_key == "wr":
        if (
            Path(str(data.get("w_cache_path"))).resolve() != FROZEN_CACHE_ROOT.resolve()
            or data.get("w_cache_manifest_sha256") != FROZEN_CACHE_MANIFEST_SHA256
        ):
            raise RuntimeError(f"{spec.identity} W cache identity 漂移")
    elif data.get("w_cache_path") is not None or data.get("w_cache_manifest_sha256") is not None:
        raise RuntimeError(f"{spec.identity} C201 不得读取 W cache")


def _validate_resolved_config(path: Path, spec: EvaluationSpec) -> None:
    cfg = OmegaConf.load(path)
    expected = {
        "protocol.stage": "research_test",
        "protocol.run_role": "frozen_research_test_evaluation",
        "protocol.execution_gate": "research_test_only",
        "model.variant": spec.variant,
        "window.input_sec": spec.input_sec,
        "window.output_sec": spec.output_sec,
        "training.seed": spec.seed,
        "training.device": "cuda:0",
        "training.batch_size": 128,
        "training.num_workers": 0,
        "training.use_amp": True,
        "data.access_splits": ["test"],
    }
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise RuntimeError(f"{spec.identity} resolved evaluation config {key} 漂移")


def _assert_summary_matches(stored: pd.DataFrame, recomputed: pd.DataFrame, *, label: str) -> None:
    if len(stored) != 1 or list(stored.columns) != list(recomputed.columns):
        raise RuntimeError(f"{label} stored summary schema 漂移")
    for column in stored.columns:
        left = stored[column].iloc[0]
        right = recomputed[column].iloc[0]
        if pd.isna(left) and pd.isna(right):
            continue
        if isinstance(right, (int, float, np.integer, np.floating)):
            if not np.isclose(float(left), float(right), rtol=0.0, atol=1e-12):
                raise RuntimeError(f"{label} stored summary 数值漂移: {column}")
        elif left != right:
            raise RuntimeError(f"{label} stored summary 值漂移: {column}")


def _records_by_name(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} artifact records 必须为 list")
    records: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError(f"{label} artifact record 非 object")
        filename = str(item.get("filename", ""))
        if Path(filename).name != filename or filename in records:
            raise RuntimeError(f"{label} artifact filename 非法或重复")
        records[filename] = item
    return records


def _verify_artifact_records(directory: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    for filename, record in records.items():
        path = directory / filename
        if (
            not path.is_file()
            or int(record.get("size_bytes", -1)) != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"artifact hash/size 漂移: {path}")


def _oriented_relative_change(old: float, new: float, direction: str) -> float:
    if not np.isfinite([old, new]).all() or old == 0.0:
        raise ValueError("相对变化要求 finite 且 from_value 非零")
    return (old - new) / abs(old) if direction == "minimize" else (new - old) / abs(old)


def _decision_guardrails() -> dict[str, Any]:
    return {
        "reporting_disposition": "report_mean_sample_sd_and_paired_seed_relative_directions",
        "cross_task_absolute_comparison_performed": False,
        "weighted_score_used": False,
        "model_or_length_reselection_performed": False,
        "additional_training_or_test_search_authorized": False,
    }


def _require_three_seeds(frame: pd.DataFrame, *, label: str) -> None:
    observed = tuple(int(value) for value in frame["seed"].sort_values().tolist())
    if observed != FORMAL_SEEDS:
        raise RuntimeError(f"{label} seeds 未闭合: {observed}")


def _assert_finite(*frames: pd.DataFrame) -> None:
    for frame in frames:
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError("research-test summary output 含 NaN/Inf")


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {"filename": path.name, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]


def _read_json(path: Path) -> dict[str, Any]:
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
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "error": None if commit.returncode == 0 and status.returncode == 0 else "git identity failed",
        "python_version": sys.version.split()[0],
    }


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "EVALUATION_COMMIT",
    "OUTPUT_FILENAMES",
    "SUMMARY_SCHEMA_VERSION",
    "audit_research_test_evaluations",
    "build_paired_companion_directions",
    "build_paired_primary_directions",
    "build_seed_companion_changes",
    "build_seed_relative_changes",
    "build_task_decision",
    "build_three_seed_arm_summary",
    "run_research_test_summary",
]
