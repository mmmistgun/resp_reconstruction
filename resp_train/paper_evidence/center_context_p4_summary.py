from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from resp_train.paper_evidence.center_context_config import (
    CENTER_CONTEXT_FORMAL_SEEDS,
    CENTER_CONTEXT_P4_OUTPUT_ROOT,
    CENTER_CONTEXT_PROTOCOL_ID,
)
from resp_train.paper_evidence.center_context_summary import (
    ARM_SPECS,
    METRIC_SPECS,
    P3_RUN_COMMIT,
    audit_formal_arm,
    audit_p3_runs,
    build_length_changes,
    build_model_differences,
    evaluate_upgrade_signal,
    sha256_file,
)


P4_SUMMARY_SCHEMA_VERSION = "paper-center-context-p4-validation-summary-v1"
P4_RUN_COMMIT = "55d515e74adc551a760fdd5914f5c4a1c1ced0a8"
P4_SEEDS = CENTER_CONTEXT_FORMAL_SEEDS[1:]
DEFAULT_P3_RUN_ROOT = Path("runs/paper_evidence_v1/center_context/p3_single_seed")
DEFAULT_P4_RUN_ROOT = Path(CENTER_CONTEXT_P4_OUTPUT_ROOT)
DEFAULT_OUTPUT_DIR = Path("runs/paper_evidence_v1/center_context/p4_validation_summary")
THREE_SEED_METRIC_COLUMNS = (
    *(f"{spec.name}_mean" for spec in METRIC_SPECS),
    "center_ibi_coverage_mean",
    "center_ibi_interpretable_fraction",
)
OUTPUT_FILENAMES = (
    "p4_seed_arm_validation_summary.csv",
    "p4_arm_three_seed_summary.csv",
    "p4_seed_length_changes.csv",
    "p4_paired_seed_length_directions.csv",
    "p4_seed_model_differences.csv",
    "p4_paired_seed_model_directions.csv",
    "p4_decision.json",
    "summary_receipt.json",
)


def audit_p4_runs(run_root: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = Path(run_root).resolve()
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None
    for seed in P4_SEEDS:
        for spec in ARM_SPECS:
            config_filename = spec.config_filename.replace("p3_", "p4_").replace(
                ".yaml", f"_seed{seed}.yaml"
            )
            run_dir = root / spec.experiment_id / f"seed_{seed}"
            row, input_record, identity = audit_formal_arm(
                run_dir,
                spec,
                seed=seed,
                expected_commit=P4_RUN_COMMIT,
                config_filename=config_filename,
            )
            if reference_identity is None:
                reference_identity = identity
            elif not identity.equals(reference_identity):
                raise RuntimeError(f"{spec.experiment_id}/seed_{seed} validation identity 不一致")
            rows.append(row)
            inputs.append(input_record)
    return pd.DataFrame.from_records(rows), inputs


def build_seed_level_changes(seed_arm_summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for seed in CENTER_CONTEXT_FORMAL_SEEDS:
        frame = build_length_changes(seed_arm_summary.loc[seed_arm_summary["seed"].eq(seed)])
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_seed_level_model_differences(seed_arm_summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for seed in CENTER_CONTEXT_FORMAL_SEEDS:
        frame = build_model_differences(seed_arm_summary.loc[seed_arm_summary["seed"].eq(seed)])
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_three_seed_arm_summary(seed_arm_summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in ARM_SPECS:
        group = seed_arm_summary.loc[
            seed_arm_summary["experiment_id"].eq(spec.experiment_id)
        ].sort_values("seed")
        _require_three_seeds(group, label=spec.experiment_id)
        record: dict[str, Any] = {
            "experiment_id": spec.experiment_id,
            "model_id": spec.model_id,
            "model_label": spec.model_label,
            "input_sec": spec.input_sec,
            "parameter_count": spec.parameter_count,
            "seeds": "|".join(str(value) for value in group["seed"]),
            "best_epochs": "|".join(str(int(value)) for value in group["best_epoch"]),
        }
        for column in THREE_SEED_METRIC_COLUMNS:
            values = group[column].to_numpy(dtype=np.float64)
            record[f"{column}_seed_mean"] = float(np.mean(values))
            record[f"{column}_seed_sample_sd"] = float(np.std(values, ddof=1))
            record[f"{column}_seed_n"] = len(values)
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_paired_seed_directions(
    seed_values: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    improvement_column: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for identity, group in seed_values.groupby(list(group_columns), sort=False):
        group = group.sort_values("seed")
        _require_three_seeds(group, label=str(identity))
        values = group[improvement_column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"paired seed improvement 非有限: {identity}")
        metric_name = str(group["metric"].iloc[0])
        metric = next(spec for spec in METRIC_SPECS if spec.name == metric_name)
        if not isinstance(identity, tuple):
            identity = (identity,)
        record = dict(zip(group_columns, identity, strict=True))
        record.update(
            {
                "direction": metric.direction,
                "improvement_unit": metric.material_kind,
                "material_threshold": metric.material_threshold,
                "improvement_seed_mean": float(np.mean(values)),
                "improvement_seed_sample_sd": float(np.std(values, ddof=1)),
                "improved_seed_count": int(np.sum(values > 0.0)),
                "worsened_seed_count": int(np.sum(values < 0.0)),
                "equal_seed_count": int(np.sum(values == 0.0)),
                "materially_improved_seed_count": int(np.sum(values >= metric.material_threshold)),
                "materially_worsened_seed_count": int(np.sum(values <= -metric.material_threshold)),
                "within_materiality_seed_count": int(np.sum(np.abs(values) < metric.material_threshold)),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_descriptive_decision(seed_length_changes: pd.DataFrame) -> dict[str, Any]:
    seed_diagnostics = []
    condition_sets: dict[str, list[set[str]]] = {
        "monotonic_improvement": [],
        "candidate_90s_saturation": [],
        "model_by_context_interaction": [],
    }
    for seed in CENTER_CONTEXT_FORMAL_SEEDS:
        decision = evaluate_upgrade_signal(seed_length_changes.loc[seed_length_changes["seed"].eq(seed)])
        conditions = decision["triggered_conditions"]
        record: dict[str, Any] = {"seed": seed, "upgrade_signal_detected": decision["upgrade_signal_detected"]}
        for condition_name in condition_sets:
            values = {
                f"{item.get('model_id', 'cross_model')}:{item['metric']}"
                for item in conditions[condition_name]
            }
            condition_sets[condition_name].append(values)
            record[condition_name] = sorted(values)
        seed_diagnostics.append(record)
    common = {
        name: sorted(set.intersection(*sets)) if sets else []
        for name, sets in condition_sets.items()
    }
    return {
        "evidence_scope": "three-seed_validation_descriptive_context_sensitivity",
        "seed_level_p3_gate_diagnostics": seed_diagnostics,
        "conditions_present_in_all_three_seeds": common,
        "automatic_stability_gate_preregistered": False,
        "automatic_stable_effect_claim_allowed": False,
        "reporting_disposition": "report_mean_sample_sd_and_paired_seed_directions",
        "weighted_score_used": False,
        "model_or_length_selection_performed": False,
        "research_test_authorized": False,
    }


def run_p4_validation_summary(
    *,
    repo_root: str | Path,
    command: str,
    p3_run_root: str | Path = DEFAULT_P3_RUN_ROOT,
    p4_run_root: str | Path = DEFAULT_P4_RUN_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    require_clean_git: bool = True,
) -> Path:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    if output.exists():
        raise FileExistsError(f"P4 汇总输出禁止覆盖: {output}")
    git = _git_identity(root)
    if require_clean_git and (git["error"] is not None or git["dirty"] is not False):
        raise RuntimeError("P4 冻结汇总要求干净 Git 工作树")

    p3_rows, p3_inputs = audit_p3_runs(_resolve(root, p3_run_root))
    p4_rows, p4_inputs = audit_p4_runs(_resolve(root, p4_run_root))
    seed_rows = pd.concat([p3_rows, p4_rows], ignore_index=True).sort_values(
        ["model_id", "input_sec", "seed"]
    )
    if len(seed_rows) != 18 or seed_rows[["experiment_id", "seed"]].duplicated().any():
        raise RuntimeError("P4 汇总必须恰好闭合 6 arms × 3 seeds")
    for experiment_id, group in seed_rows.groupby("experiment_id"):
        _require_three_seeds(group, label=str(experiment_id))

    arm_summary = build_three_seed_arm_summary(seed_rows)
    seed_changes = build_seed_level_changes(seed_rows)
    paired_changes = build_paired_seed_directions(
        seed_changes,
        group_columns=("model_id", "from_sec", "to_sec", "metric"),
        improvement_column="oriented_improvement",
    )
    seed_model_differences = build_seed_level_model_differences(seed_rows)
    paired_model_differences = build_paired_seed_directions(
        seed_model_differences,
        group_columns=("input_sec", "metric"),
        improvement_column="w_reduced_oriented_improvement",
    )
    decision = build_descriptive_decision(seed_changes)
    _assert_finite(seed_rows, arm_summary, seed_changes, paired_changes, seed_model_differences, paired_model_differences)

    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        outputs = {
            "p4_seed_arm_validation_summary.csv": seed_rows,
            "p4_arm_three_seed_summary.csv": arm_summary,
            "p4_seed_length_changes.csv": seed_changes,
            "p4_paired_seed_length_directions.csv": paired_changes,
            "p4_seed_model_differences.csv": seed_model_differences,
            "p4_paired_seed_model_directions.csv": paired_model_differences,
        }
        for filename, frame in outputs.items():
            frame.to_csv(temporary / filename, index=False)
        _write_json(temporary / "p4_decision.json", decision)
        receipt = {
            "schema_version": P4_SUMMARY_SCHEMA_VERSION,
            "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": "three-seed validation descriptive context sensitivity",
            "execution": {
                "command": command,
                "cwd": str(root),
                "git": git,
                "p3_run_commit": P3_RUN_COMMIT,
                "p4_run_commit": P4_RUN_COMMIT,
            },
            "inputs": [*p3_inputs, *p4_inputs],
            "counts": {
                "expected_runs": 18,
                "actual_runs": len(seed_rows),
                "validation_rows_per_run": 2675,
                "validation_metric_rows_read": 2675 * len(seed_rows),
                "seed_arm_rows": len(seed_rows),
                "three_seed_arm_rows": len(arm_summary),
                "seed_length_change_rows": len(seed_changes),
                "paired_length_direction_rows": len(paired_changes),
                "seed_model_difference_rows": len(seed_model_differences),
                "paired_model_direction_rows": len(paired_model_differences),
            },
            "aggregation": {
                "seed_mean": "arithmetic_mean",
                "seed_sd": "sample_sd_ddof1",
                "pairing": "same_seed",
                "weighted_score": False,
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
                "schema_version": P4_SUMMARY_SCHEMA_VERSION,
                "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
                "status": "complete",
                "read_only_validation_summary": True,
                "source_run_commits": {"p3": P3_RUN_COMMIT, "p4": P4_RUN_COMMIT},
                "files": _artifact_records(temporary),
            },
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "summary_receipt.json"


def _require_three_seeds(frame: pd.DataFrame, *, label: str) -> None:
    seeds = tuple(int(value) for value in frame["seed"].tolist())
    if seeds != CENTER_CONTEXT_FORMAL_SEEDS:
        raise RuntimeError(f"{label} seed identity 必须为 {CENTER_CONTEXT_FORMAL_SEEDS}，实际 {seeds}")


def _assert_finite(*frames: pd.DataFrame) -> None:
    for frame in frames:
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError("P4 summary output 含 NaN/Inf")


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {"filename": path.name, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]


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
    "P4_RUN_COMMIT",
    "P4_SUMMARY_SCHEMA_VERSION",
    "audit_p4_runs",
    "build_descriptive_decision",
    "build_paired_seed_directions",
    "build_seed_level_changes",
    "build_seed_level_model_differences",
    "build_three_seed_arm_summary",
    "run_p4_validation_summary",
]
