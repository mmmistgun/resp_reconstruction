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

from resp_train.paper_evidence.center30_config import (
    CENTER30_FORMAL_SEEDS,
    CENTER30_P4S3_OUTPUT_ROOT,
    CENTER30_P4S_OUTPUT_ROOT,
    CENTER30_PROTOCOL_ID,
)
from resp_train.paper_evidence.center30_summary import (
    ARM_SPECS,
    METRIC_SPECS,
    P4S_RUN_COMMIT,
    audit_formal_arm,
    audit_p4s_runs,
    build_companion_changes,
    build_model_differences,
    build_relative_changes,
    sha256_file,
)


P4S3_SUMMARY_SCHEMA_VERSION = "paper-center30-context-p4s3-validation-summary-v1"
P4S3_RUN_COMMIT = "2107cf9935229b28224035aa7272515e91be0706"
P4S3_SEEDS = CENTER30_FORMAL_SEEDS[1:]
DEFAULT_P4S2_RUN_ROOT = Path(CENTER30_P4S_OUTPUT_ROOT)
DEFAULT_P4S3_RUN_ROOT = Path(CENTER30_P4S3_OUTPUT_ROOT)
DEFAULT_OUTPUT_DIR = Path("runs/paper_evidence_v1/center30_context/p4s3_validation_summary")
THREE_SEED_METRIC_COLUMNS = (
    *(f"{spec.name}_mean" for spec in METRIC_SPECS),
    "center30_ibi_coverage_mean",
    "center30_ibi_interpretable_fraction",
)
OUTPUT_FILENAMES = (
    "p4s3_seed_arm_validation_summary.csv",
    "p4s3_arm_three_seed_summary.csv",
    "p4s3_seed_relative_changes.csv",
    "p4s3_paired_seed_relative_directions.csv",
    "p4s3_seed_companion_relative_changes.csv",
    "p4s3_paired_seed_companion_directions.csv",
    "p4s3_seed_model_differences.csv",
    "p4s3_paired_seed_model_directions.csv",
    "p4s3_decision.json",
    "summary_receipt.json",
)


def audit_p4s3_runs(run_root: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = Path(run_root).resolve()
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None
    for seed in P4S3_SEEDS:
        for spec in ARM_SPECS:
            short = "c201" if spec.model_id == "C201" else "wr"
            config_filename = f"p4s3_c30v1_{short}_{spec.input_sec}_seed{seed}.yaml"
            run_dir = root / spec.experiment_id / f"seed_{seed}"
            row, input_record, identity = audit_formal_arm(
                run_dir,
                spec,
                seed=seed,
                expected_commit=P4S3_RUN_COMMIT,
                config_filename=config_filename,
                expected_stage="p4s_additional_seeds",
                expected_gate="p4s_three_seed_formal",
                expected_output_root=CENTER30_P4S3_OUTPUT_ROOT,
            )
            if reference_identity is None:
                reference_identity = identity
            elif not identity.equals(reference_identity):
                raise RuntimeError(f"{spec.experiment_id}/seed_{seed} validation identity 不一致")
            rows.append(row)
            inputs.append(input_record)
    return pd.DataFrame.from_records(rows), inputs


def build_three_seed_arm_summary(seed_arm_summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in ARM_SPECS:
        group = seed_arm_summary.loc[seed_arm_summary["experiment_id"].eq(spec.experiment_id)].sort_values("seed")
        _require_three_seeds(group, label=spec.experiment_id)
        record: dict[str, Any] = {
            "experiment_id": spec.experiment_id,
            "model_id": spec.model_id,
            "variant": spec.variant,
            "input_sec": spec.input_sec,
            "parameter_count": spec.parameter_count,
            "w_scale_count": spec.w_scale_count,
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


def build_seed_level_changes(seed_arm_summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for seed in CENTER30_FORMAL_SEEDS:
        frame = build_relative_changes(seed_arm_summary.loc[seed_arm_summary["seed"].eq(seed)])
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_seed_level_companion_changes(seed_arm_summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for seed in CENTER30_FORMAL_SEEDS:
        frame = build_companion_changes(seed_arm_summary.loc[seed_arm_summary["seed"].eq(seed)])
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_seed_level_model_differences(seed_arm_summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for seed in CENTER30_FORMAL_SEEDS:
        frame = build_model_differences(seed_arm_summary.loc[seed_arm_summary["seed"].eq(seed)])
        frame.insert(0, "seed", seed)
        absolute = []
        material_values = []
        material_kinds = []
        thresholds = []
        for row in frame.itertuples(index=False):
            metric = next(spec for spec in METRIC_SPECS if spec.name == row.metric)
            oriented_absolute = (
                -float(row.w_reduced_minus_c201)
                if metric.direction == "minimize"
                else float(row.w_reduced_minus_c201)
            )
            absolute.append(oriented_absolute)
            material_kinds.append(metric.material_kind)
            thresholds.append(metric.material_threshold)
            material_values.append(
                float(row.w_reduced_oriented_relative_change)
                if metric.material_kind == "relative"
                else oriented_absolute
            )
        frame["w_reduced_oriented_absolute_change"] = absolute
        frame["material_kind"] = material_kinds
        frame["material_threshold"] = thresholds
        frame["protocol_material_value"] = material_values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_paired_primary_directions(seed_changes: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groups = ("model_id", "from_sec", "to_sec", "metric")
    for identity, group in seed_changes.groupby(list(groups), sort=False):
        group = group.sort_values("seed")
        _require_three_seeds(group, label=str(identity))
        protocol_values = group["protocol_material_value"].to_numpy(dtype=np.float64)
        relative_values = group["oriented_relative_change"].to_numpy(dtype=np.float64)
        absolute_values = group["oriented_absolute_change"].to_numpy(dtype=np.float64)
        threshold = float(group["material_threshold"].iloc[0])
        record = dict(zip(groups, identity, strict=True))
        record.update(
            {
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
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_paired_companion_directions(seed_changes: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groups = ("model_id", "from_sec", "to_sec", "metric")
    for identity, group in seed_changes.groupby(list(groups), sort=False):
        group = group.sort_values("seed")
        _require_three_seeds(group, label=str(identity))
        values = group["relative_change"].to_numpy(dtype=np.float64)
        records.append(
            {
                **dict(zip(groups, identity, strict=True)),
                "relative_change_seed_mean": float(np.mean(values)),
                "relative_change_seed_sample_sd": float(np.std(values, ddof=1)),
                "increased_seed_count": int(np.sum(values > 0.0)),
                "decreased_seed_count": int(np.sum(values < 0.0)),
                "equal_seed_count": int(np.sum(values == 0.0)),
            }
        )
    return pd.DataFrame.from_records(records)


def build_paired_model_directions(seed_differences: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groups = ("input_sec", "metric")
    for identity, group in seed_differences.groupby(list(groups), sort=False):
        group = group.sort_values("seed")
        _require_three_seeds(group, label=str(identity))
        protocol_values = group["protocol_material_value"].to_numpy(dtype=np.float64)
        relative_values = group["w_reduced_oriented_relative_change"].to_numpy(dtype=np.float64)
        absolute_values = group["w_reduced_oriented_absolute_change"].to_numpy(dtype=np.float64)
        threshold = float(group["material_threshold"].iloc[0])
        records.append(
            {
                **dict(zip(groups, identity, strict=True)),
                "direction": str(group["direction"].iloc[0]),
                "material_kind": str(group["material_kind"].iloc[0]),
                "material_threshold": threshold,
                "w_reduced_oriented_relative_change_seed_mean": float(np.mean(relative_values)),
                "w_reduced_oriented_relative_change_seed_sample_sd": float(np.std(relative_values, ddof=1)),
                "w_reduced_oriented_absolute_change_seed_mean": float(np.mean(absolute_values)),
                "w_reduced_oriented_absolute_change_seed_sample_sd": float(np.std(absolute_values, ddof=1)),
                "w_reduced_better_seed_count": int(np.sum(protocol_values > 0.0)),
                "c201_better_seed_count": int(np.sum(protocol_values < 0.0)),
                "equal_seed_count": int(np.sum(protocol_values == 0.0)),
                "w_reduced_materially_better_seed_count": int(np.sum(protocol_values >= threshold)),
                "c201_materially_better_seed_count": int(np.sum(protocol_values <= -threshold)),
                "within_materiality_seed_count": int(np.sum(np.abs(protocol_values) < threshold)),
            }
        )
    return pd.DataFrame.from_records(records)


def build_descriptive_decision(
    seed_arm_summary: pd.DataFrame,
    seed_changes: pd.DataFrame,
) -> dict[str, Any]:
    rr = seed_changes.loc[seed_changes["metric"].eq("center30_rr_mae_bpm")]
    baseline_counts: dict[str, dict[str, int]] = {}
    best_inputs: dict[str, dict[str, int]] = {}
    best_input_counts: dict[str, dict[str, int]] = {}
    for model_id in ("C201", "W-reduced"):
        model_rr = rr.loc[rr["model_id"].eq(model_id)]
        baseline_counts[model_id] = {
            str(to_sec): int(
                model_rr.loc[
                    model_rr["from_sec"].eq(30) & model_rr["to_sec"].eq(to_sec),
                    "materially_improved",
                ].sum()
            )
            for to_sec in (45, 60, 90)
        }
        per_seed: dict[str, int] = {}
        for seed in CENTER30_FORMAL_SEEDS:
            rows = seed_arm_summary.loc[
                seed_arm_summary["model_id"].eq(model_id) & seed_arm_summary["seed"].eq(seed)
            ]
            per_seed[str(seed)] = int(
                rows.loc[rows["center30_rr_mae_bpm_mean"].idxmin(), "input_sec"]
            )
        best_inputs[model_id] = per_seed
        best_input_counts[model_id] = {
            str(input_sec): sum(value == input_sec for value in per_seed.values())
            for input_sec in (30, 45, 60, 90)
        }
    all_longer_materially_better = all(
        count == len(CENTER30_FORMAL_SEEDS)
        for values in baseline_counts.values()
        for count in values.values()
    )
    return {
        "evidence_scope": "three-seed_validation_short-window_context_sensitivity",
        "rr_materially_improved_seed_count_vs_30s": baseline_counts,
        "rr_best_input_sec_by_model_and_seed": best_inputs,
        "rr_best_input_sec_counts_by_model": best_input_counts,
        "all_models_all_seeds_all_longer_inputs_materially_improve_rr_vs_30s": all_longer_materially_better,
        "candidate_interpretation": (
            "rr_30s_inferior_all_models_all_seeds_boundary_requires_protocol_review"
            if all_longer_materially_better
            else "short_window_direction_not_uniform"
        ),
        "automatic_stable_minimum_context_gate_preregistered": False,
        "automatic_stable_minimum_context_claim_allowed": False,
        "reporting_disposition": "report_mean_sample_sd_and_paired_seed_directions",
        "weighted_score_used": False,
        "model_or_length_selection_performed": False,
        "research_test_authorized": False,
    }


def run_p4s3_validation_summary(
    *,
    repo_root: str | Path,
    command: str,
    p4s2_run_root: str | Path = DEFAULT_P4S2_RUN_ROOT,
    p4s3_run_root: str | Path = DEFAULT_P4S3_RUN_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    require_clean_git: bool = True,
) -> Path:
    root = Path(repo_root).resolve()
    output = _resolve(root, output_dir)
    if output.exists():
        raise FileExistsError(f"P4-S3 汇总输出禁止覆盖: {output}")
    git = _git_identity(root)
    if require_clean_git and (git["error"] is not None or git["dirty"] is not False):
        raise RuntimeError("P4-S3 冻结汇总要求干净 Git 工作树")

    p4s2_rows, p4s2_inputs = audit_p4s_runs(_resolve(root, p4s2_run_root))
    p4s3_rows, p4s3_inputs = audit_p4s3_runs(_resolve(root, p4s3_run_root))
    seed_rows = pd.concat([p4s2_rows, p4s3_rows], ignore_index=True).sort_values(
        ["model_id", "input_sec", "seed"]
    )
    if len(seed_rows) != 24 or seed_rows[["experiment_id", "seed"]].duplicated().any():
        raise RuntimeError("P4-S3 汇总必须恰好闭合 8 arms × 3 seeds")
    for experiment_id, group in seed_rows.groupby("experiment_id"):
        _require_three_seeds(group, label=str(experiment_id))

    arm_summary = build_three_seed_arm_summary(seed_rows)
    seed_changes = build_seed_level_changes(seed_rows)
    paired_changes = build_paired_primary_directions(seed_changes)
    seed_companion = build_seed_level_companion_changes(seed_rows)
    paired_companion = build_paired_companion_directions(seed_companion)
    seed_differences = build_seed_level_model_differences(seed_rows)
    paired_differences = build_paired_model_directions(seed_differences)
    decision = build_descriptive_decision(seed_rows, seed_changes)
    _assert_finite(
        seed_rows,
        arm_summary,
        seed_changes,
        paired_changes,
        seed_companion,
        paired_companion,
        seed_differences,
        paired_differences,
    )

    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        outputs = {
            "p4s3_seed_arm_validation_summary.csv": seed_rows,
            "p4s3_arm_three_seed_summary.csv": arm_summary,
            "p4s3_seed_relative_changes.csv": seed_changes,
            "p4s3_paired_seed_relative_directions.csv": paired_changes,
            "p4s3_seed_companion_relative_changes.csv": seed_companion,
            "p4s3_paired_seed_companion_directions.csv": paired_companion,
            "p4s3_seed_model_differences.csv": seed_differences,
            "p4s3_paired_seed_model_directions.csv": paired_differences,
        }
        for filename, frame in outputs.items():
            frame.to_csv(temporary / filename, index=False)
        _write_json(temporary / "p4s3_decision.json", decision)
        receipt = {
            "schema_version": P4S3_SUMMARY_SCHEMA_VERSION,
            "protocol_id": CENTER30_PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": "three-seed validation short-window context sensitivity",
            "execution": {
                "command": command,
                "cwd": str(root),
                "git": git,
                "p4s2_run_commit": P4S_RUN_COMMIT,
                "p4s3_run_commit": P4S3_RUN_COMMIT,
            },
            "inputs": [*p4s2_inputs, *p4s3_inputs],
            "counts": {
                "expected_runs": 24,
                "actual_runs": len(seed_rows),
                "validation_rows_per_run": 2675,
                "validation_metric_rows_read": 2675 * len(seed_rows),
                "seed_arm_rows": len(seed_rows),
                "three_seed_arm_rows": len(arm_summary),
                "seed_relative_change_rows": len(seed_changes),
                "paired_relative_direction_rows": len(paired_changes),
                "seed_companion_change_rows": len(seed_companion),
                "paired_companion_direction_rows": len(paired_companion),
                "seed_model_difference_rows": len(seed_differences),
                "paired_model_direction_rows": len(paired_differences),
            },
            "aggregation": {
                "run_metric": "sample_direct_mean",
                "seed_mean": "arithmetic_mean",
                "seed_sd": "sample_sd_ddof1",
                "paired_unit": "seed",
                "weighted_score": False,
                "p_value": False,
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
                "schema_version": P4S3_SUMMARY_SCHEMA_VERSION,
                "protocol_id": CENTER30_PROTOCOL_ID,
                "status": "complete",
                "read_only_validation_summary": True,
                "source_run_commits": {"p4s2": P4S_RUN_COMMIT, "p4s3": P4S3_RUN_COMMIT},
                "files": _artifact_records(temporary),
            },
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "summary_receipt.json"


def _require_three_seeds(frame: pd.DataFrame, *, label: str) -> None:
    observed = tuple(int(value) for value in frame["seed"].sort_values().tolist())
    if observed != CENTER30_FORMAL_SEEDS:
        raise RuntimeError(f"{label} seeds 未闭合: {observed}")


def _assert_finite(*frames: pd.DataFrame) -> None:
    for frame in frames:
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError("P4-S3 summary output 含 NaN/Inf")


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
    "OUTPUT_FILENAMES",
    "P4S3_RUN_COMMIT",
    "P4S3_SEEDS",
    "P4S3_SUMMARY_SCHEMA_VERSION",
    "audit_p4s3_runs",
    "build_descriptive_decision",
    "build_paired_companion_directions",
    "build_paired_model_directions",
    "build_paired_primary_directions",
    "build_seed_level_changes",
    "build_seed_level_companion_changes",
    "build_seed_level_model_differences",
    "build_three_seed_arm_summary",
    "run_p4s3_validation_summary",
]
