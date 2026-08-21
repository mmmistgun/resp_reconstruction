from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.metrics.task import summarize_task_metrics
from resp_train.temporal.formal import (
    EXPECTED_CANDIDATES,
    FORMAL_SEEDS,
    validate_formal_receipt,
)
from resp_train.temporal.gpu_engineering import sha256_file, validate_access_receipt as validate_gpu_receipt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/resp_temporal_v1/validation_summary_v1.yaml"
CONFIG_SCHEMA_VERSION = "rtm-v1-validation-summary-config-v1"
PROTOCOL_ID = "resp-temporal-v1-validation-summary-20260821"
RECEIPT_SCHEMA_VERSION = "rtm-v1-validation-summary-receipt-v1"
FROZEN_CONFIG_SHA256 = "21f723580c9cbbe39a3b85c68290e5ace26c6dadf0a9b37abdee9dd8a046816f"
FORMAL_EXECUTION_COMMIT = "24c54a88ea8b17ad183b48976bd1f690e3867706"
EXPECTED_VALIDATION_ROWS = 2675
EXPECTED_VALIDATION_SAMP_IDS = 7
EXPECTED_VALIDATION_ROW_HASH = "b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a"
OUTPUT_FILENAMES = (
    "resolved_config.json",
    "seed_summary.csv",
    "candidate_summary.csv",
    "paired_seed_directions.csv",
    "dominance_audit.csv",
    "quality_pareto.csv",
    "quality_efficiency_pareto.csv",
    "supplementary_metrics.csv",
    "decision.json",
    "artifact_manifest.json",
)
EXPECTED_METRICS_COLUMNS = (
    "evaluation_split",
    "method",
    "dataset_row_id",
    "split",
    "input_set",
    "samp_id",
    "coupling_state_id",
    "whole_rr_abs_error_bpm",
    "whole_rr_target_eligible",
    "local_rr_mae_bpm",
    "local_rr_prediction_valid_fraction",
    "local_rr_target_eligible",
    "local_rr_target_eligible_windows",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "target_envelope_modulation",
    "envelope_target_stratum",
    "target_stratified_envelope_spearman",
    "envelope_spearman_target_eligible",
    "envelope_spearman_prediction_degenerate",
    "lag_aware_signed_pcc",
    "best_lag_samples",
    "best_lag_sec",
    "joint_target_eligible",
    "joint_prediction_degenerate",
    "ibi_medae_sec",
    "ibi_coverage",
    "ibi_interpretable",
    "ibi_target_eligible",
)
STATIC_VALIDATION_COLUMNS = (
    "dataset_row_id",
    "split",
    "input_set",
    "samp_id",
    "coupling_state_id",
    "whole_rr_target_eligible",
    "local_rr_target_eligible",
    "local_rr_target_eligible_windows",
    "target_envelope_modulation",
    "envelope_target_stratum",
    "envelope_spearman_target_eligible",
    "joint_target_eligible",
    "ibi_target_eligible",
)
PRIMARY_SAMPLE_COLUMNS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
)


@dataclass(frozen=True)
class SummaryConfig:
    path: Path
    sha256: str
    raw: dict[str, Any]
    output_dir: Path


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{context}字段必须严格为{sorted(expected)}，实际为{sorted(actual)}")


def _repo_path(value: Any, *, context: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context}必须是仓库内相对路径")
    return REPO_ROOT / relative


def _is_sha256(value: Any) -> bool:
    rendered = str(value)
    return len(rendered) == 64 and all(character in "0123456789abcdef" for character in rendered)


def validate_summary_config(raw: Mapping[str, Any]) -> None:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "role",
            "evidence_label",
            "authorization",
            "provenance",
            "candidates",
            "seeds",
            "formal_runs",
            "aggregation",
            "primary_metrics",
            "quality_efficiency_metrics",
            "supplementary_metrics",
            "access",
            "output",
        },
        "summary config",
    )
    if (
        raw["schema_version"] != CONFIG_SCHEMA_VERSION
        or raw["protocol_id"] != PROTOCOL_ID
        or raw["role"] != "validation_summary_only"
        or raw["evidence_label"] != "validation-development evidence"
    ):
        raise ValueError("summary config identity/evidence漂移")
    if raw["authorization"] != {
        "validation_summary_implementation": True,
        "user_manual_summary_execution": True,
        "codex_summary_execution": False,
        "model_training": False,
        "model_inference": False,
        "research_test_evaluation": False,
    }:
        raise ValueError("summary authorization越界")
    provenance = raw["provenance"]
    _require_exact_keys(
        provenance,
        {
            "summary_protocol_path",
            "summary_protocol_sha256",
            "formal_dual_gpu_plan_path",
            "formal_dual_gpu_plan_sha256",
            "formal_dual_gpu_correction_path",
            "formal_dual_gpu_correction_sha256",
            "formal_dual_gpu_implementation_receipt_path",
            "formal_dual_gpu_implementation_receipt_sha256",
            "formal_execution_git_commit",
            "gpu_engineering_csv_path",
            "gpu_engineering_csv_sha256",
            "gpu_engineering_receipt_path",
            "gpu_engineering_receipt_sha256",
            "gpu_engineering_manifest_path",
            "gpu_engineering_manifest_sha256",
            "require_clean_git",
        },
        "summary provenance",
    )
    if provenance["formal_execution_git_commit"] != FORMAL_EXECUTION_COMMIT or not provenance["require_clean_git"]:
        raise ValueError("summary formal Git/clean contract漂移")
    for key, value in provenance.items():
        if key.endswith("_sha256") and not _is_sha256(value):
            raise ValueError(f"summary provenance hash无效: {key}")
    expected_candidates = (
        ("rtm_v1_t0_locked_stem_head", "no_temporal_trunk", "control", 59042),
        ("rtm_v1_tcn_d9_h384", "dilated_tcn", "representative", 729506),
        ("rtm_v1_bimamba2_d96_l6", "bimamba2", "representative", 1010426),
        ("rtm_v1_bilstm_h96_l2", "bilstm", "representative", 449474),
        ("rtm_v1_multiscale_10_2_1_h384", "feature_level_multiscale", "representative", 1129730),
    )
    observed_candidates = []
    for record in raw["candidates"]:
        _require_exact_keys(record, {"candidate_id", "family", "role", "trainable_parameters"}, "candidate")
        observed_candidates.append(
            (record["candidate_id"], record["family"], record["role"], int(record["trainable_parameters"]))
        )
    if tuple(observed_candidates) != expected_candidates or list(raw["seeds"]) != list(FORMAL_SEEDS):
        raise ValueError("summary candidate/seed matrix漂移")
    expected_identities = {(candidate, seed) for candidate in EXPECTED_CANDIDATES for seed in FORMAL_SEEDS}
    observed_identities: list[tuple[str, int]] = []
    groups: dict[tuple[str, str], int] = {}
    for record in raw["formal_runs"]:
        _require_exact_keys(
            record,
            {
                "candidate_id",
                "seed",
                "execution_group",
                "cuda_visible_devices",
                "receipt_path",
                "receipt_sha256",
            },
            "formal run",
        )
        identity = (str(record["candidate_id"]), int(record["seed"]))
        observed_identities.append(identity)
        expected_path = f"runs/resp_temporal_v1/formal/{identity[0]}/seed_{identity[1]}/formal_receipt.json"
        if record["receipt_path"] != expected_path or not _is_sha256(record["receipt_sha256"]):
            raise ValueError("summary formal run path/hash漂移")
        group_key = (str(record["execution_group"]), str(record["cuda_visible_devices"]))
        groups[group_key] = groups.get(group_key, 0) + 1
    if (
        len(observed_identities) != 15
        or len(set(observed_identities)) != 15
        or set(observed_identities) != expected_identities
        or groups != {("gpu_0", "0"): 8, ("gpu_1", "1"): 7}
    ):
        raise ValueError("summary formal 15-run/group closure漂移")
    if raw["aggregation"] != {
        "expected_formal_runs": 15,
        "expected_validation_rows_per_run": 2675,
        "expected_validation_samp_ids": 7,
        "expected_validation_row_ids_sha256": EXPECTED_VALIDATION_ROW_HASH,
        "candidate_seed_mean": "arithmetic",
        "candidate_seed_sd": "sample_ddof_1",
        "stored_summary_rtol": 1e-12,
        "stored_summary_atol": 1e-12,
    }:
        raise ValueError("summary aggregation contract漂移")
    expected_primary = [
        {"name": "whole_rr_abs_error_bpm_mean", "direction": "minimize", "tolerance_kind": "relative", "tolerance": 0.005},
        {"name": "local_rr_mae_bpm_mean", "direction": "minimize", "tolerance_kind": "relative", "tolerance": 0.005},
        {"name": "envelope_trajectory_mae_mean", "direction": "minimize", "tolerance_kind": "relative", "tolerance": 0.005},
        {"name": "global_envelope_modulation_error_mean", "direction": "minimize", "tolerance_kind": "relative", "tolerance": 0.005},
        {"name": "lag_aware_signed_pcc_mean", "direction": "maximize", "tolerance_kind": "absolute", "tolerance": 0.002},
    ]
    expected_resources = [
        {"name": "trainable_parameters", "direction": "minimize", "tolerance_kind": "exact", "tolerance": 0.0},
        {"name": "update_throughput_windows_per_second", "direction": "maximize", "tolerance_kind": "relative", "tolerance": 0.05},
        {"name": "update_peak_allocated_mib", "direction": "minimize", "tolerance_kind": "relative", "tolerance": 0.05},
    ]
    if raw["primary_metrics"] != expected_primary or raw["quality_efficiency_metrics"] != expected_resources:
        raise ValueError("summary Pareto metrics/tolerance漂移")
    if raw["supplementary_metrics"] != {
        "seed_aggregated": [
            "ibi_medae_sec_mean",
            "ibi_coverage_mean",
            "target_stratified_envelope_spearman_low_mean",
            "target_stratified_envelope_spearman_medium_mean",
            "target_stratified_envelope_spearman_high_mean",
            "joint_prediction_degenerate_fraction",
        ],
        "engineering": [
            "batch1_inference_p50_ms",
            "batch1_inference_p90_ms",
            "update_peak_reserved_mib",
        ],
    }:
        raise ValueError("summary supplementary metrics漂移")
    if raw["access"] != {
        "validation_metrics_allowed": True,
        "validation_summary_allowed": True,
        "formal_receipt_allowed": True,
        "lifecycle_and_manifest_allowed": True,
        "gpu_engineering_artifacts_allowed": True,
        "dataset_or_index_allowed": False,
        "signal_or_target_array_allowed": False,
        "checkpoint_content_allowed": False,
        "model_training_allowed": False,
        "model_inference_allowed": False,
        "gpu_allowed": False,
        "research_test_allowed": False,
    }:
        raise ValueError("summary access contract越界")
    if raw["output"] != {
        "directory": "runs/resp_temporal_v1/formal_validation_summary_v1",
        "allow_overwrite": False,
        "csv_float_format": "%.12g",
    }:
        raise ValueError("summary output contract漂移")


def load_summary_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SummaryConfig:
    config_path = Path(path).resolve()
    if config_path != DEFAULT_CONFIG_PATH.resolve():
        raise ValueError(f"RTM-v1 summary只允许冻结config: {DEFAULT_CONFIG_PATH}")
    if not config_path.is_file():
        raise FileNotFoundError(f"summary config不存在: {config_path}")
    actual_hash = sha256_file(config_path)
    if actual_hash != FROZEN_CONFIG_SHA256:
        raise ValueError(f"summary config SHA-256漂移: {actual_hash}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("summary config顶层必须是mapping")
    validate_summary_config(raw)
    return SummaryConfig(
        path=config_path,
        sha256=actual_hash,
        raw=raw,
        output_dir=_repo_path(raw["output"]["directory"], context="summary output"),
    )


def dimension_comparison(left: float, right: float, spec: Mapping[str, Any]) -> tuple[bool, bool]:
    """返回left相对right的(tolerance内不差, 实质更优)。"""

    left = float(left)
    right = float(right)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("Pareto输入必须finite")
    direction = str(spec["direction"])
    kind = str(spec["tolerance_kind"])
    tolerance = float(spec["tolerance"])
    if tolerance < 0.0 or direction not in {"minimize", "maximize"}:
        raise ValueError("Pareto direction/tolerance无效")
    if kind == "relative":
        if left <= 0.0 or right <= 0.0:
            raise ValueError("relative tolerance要求正值")
        if direction == "minimize":
            return left <= right * (1.0 + tolerance), left < right * (1.0 - tolerance)
        return left >= right * (1.0 - tolerance), left > right * (1.0 + tolerance)
    if kind == "absolute":
        if direction == "minimize":
            return left <= right + tolerance, left < right - tolerance
        return left >= right - tolerance, left > right + tolerance
    if kind == "exact":
        if tolerance != 0.0:
            raise ValueError("exact tolerance必须为0")
        if direction == "minimize":
            return left <= right, left < right
        return left >= right, left > right
    raise ValueError(f"未知tolerance_kind={kind!r}")


def _candidate_value(row: pd.Series, spec: Mapping[str, Any], *, primary_names: set[str]) -> float:
    name = str(spec["name"])
    column = f"{name}_seed_mean" if name in primary_names else name
    return float(row[column])


def build_candidate_summary(
    seed_summary: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    engineering: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = {
        "candidate_id",
        "family",
        "role",
        "seed",
        "execution_group",
        "cuda_visible_devices",
        "selected_epoch",
        "receipt_sha256",
        "metrics_sha256",
        "stored_summary_sha256",
    }
    numeric_columns = [
        column
        for column in seed_summary.select_dtypes(include=np.number).columns
        if column not in identity_columns and column != "trainable_parameters"
    ]
    rows: list[dict[str, Any]] = []
    expected_seeds = [int(seed) for seed in seeds]
    engineering_by_id = engineering.set_index("candidate_id")
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        selected = seed_summary.loc[seed_summary["candidate_id"].eq(candidate_id)].sort_values("seed")
        if selected["seed"].astype(int).tolist() != expected_seeds:
            raise ValueError(f"summary candidate seed不完整: {candidate_id}")
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "family": str(candidate["family"]),
            "role": str(candidate["role"]),
            "seed_count": 3,
            "trainable_parameters": int(candidate["trainable_parameters"]),
        }
        for column in numeric_columns:
            values = pd.to_numeric(selected[column], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{column}_seed_mean"] = float(np.mean(finite)) if finite.size else np.nan
            row[f"{column}_seed_sd"] = float(np.std(finite, ddof=1)) if finite.size >= 2 else np.nan
            row[f"{column}_seed_n"] = int(finite.size)
        if candidate_id not in engineering_by_id.index:
            raise ValueError(f"GPU engineering缺少candidate: {candidate_id}")
        engineering_row = engineering_by_id.loc[candidate_id]
        for column in (
            "update_throughput_windows_per_second",
            "update_peak_allocated_mib",
            "batch1_inference_p50_ms",
            "batch1_inference_p90_ms",
            "update_peak_reserved_mib",
        ):
            row[column] = float(engineering_row[column])
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_seed_directions(
    seed_summary: pd.DataFrame,
    candidate_order: Sequence[str],
    primary_specs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered_seeds = [int(seed) for seed in seeds]
    for left_id, right_id in combinations(candidate_order, 2):
        left = seed_summary.loc[seed_summary["candidate_id"].eq(left_id)].set_index("seed")
        right = seed_summary.loc[seed_summary["candidate_id"].eq(right_id)].set_index("seed")
        for spec in primary_specs:
            metric = str(spec["name"])
            row: dict[str, Any] = {
                "left_candidate_id": left_id,
                "right_candidate_id": right_id,
                "metric": metric,
                "direction": str(spec["direction"]),
                "tolerance_kind": str(spec["tolerance_kind"]),
                "tolerance": float(spec["tolerance"]),
                "left_raw_better_count": 0,
                "right_raw_better_count": 0,
                "raw_equal_count": 0,
                "left_materially_better_count": 0,
                "right_materially_better_count": 0,
                "within_tolerance_count": 0,
            }
            for seed in ordered_seeds:
                left_value = float(left.loc[seed, metric])
                right_value = float(right.loc[seed, metric])
                row[f"left_seed_{seed}"] = left_value
                row[f"right_seed_{seed}"] = right_value
                row[f"left_minus_right_seed_{seed}"] = left_value - right_value
                if left_value == right_value:
                    row["raw_equal_count"] += 1
                elif (spec["direction"] == "minimize" and left_value < right_value) or (
                    spec["direction"] == "maximize" and left_value > right_value
                ):
                    row["left_raw_better_count"] += 1
                else:
                    row["right_raw_better_count"] += 1
                _, left_material = dimension_comparison(left_value, right_value, spec)
                _, right_material = dimension_comparison(right_value, left_value, spec)
                if left_material:
                    row["left_materially_better_count"] += 1
                elif right_material:
                    row["right_materially_better_count"] += 1
                else:
                    row["within_tolerance_count"] += 1
            rows.append(row)
    return pd.DataFrame(rows)


def build_pareto_outputs(
    candidate_summary: pd.DataFrame,
    primary_specs: Sequence[Mapping[str, Any]],
    resource_specs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_names = {str(spec["name"]) for spec in primary_specs}
    candidate_order = candidate_summary["candidate_id"].astype(str).tolist()
    audit_rows: list[dict[str, Any]] = []
    pareto_frames: dict[str, pd.DataFrame] = {}
    for pareto_kind, specs in (
        ("quality", list(primary_specs)),
        ("quality_efficiency", [*primary_specs, *resource_specs]),
    ):
        dominated_by: dict[str, list[str]] = {candidate: [] for candidate in candidate_order}
        for left_id in candidate_order:
            left = candidate_summary.loc[candidate_summary["candidate_id"].eq(left_id)].iloc[0]
            for right_id in candidate_order:
                if left_id == right_id:
                    continue
                right = candidate_summary.loc[candidate_summary["candidate_id"].eq(right_id)].iloc[0]
                row: dict[str, Any] = {
                    "pareto_kind": pareto_kind,
                    "left_candidate_id": left_id,
                    "right_candidate_id": right_id,
                    "all_dimensions_no_worse": True,
                    "any_dimension_materially_better": False,
                }
                for spec in specs:
                    name = str(spec["name"])
                    left_value = _candidate_value(left, spec, primary_names=primary_names)
                    right_value = _candidate_value(right, spec, primary_names=primary_names)
                    no_worse, material = dimension_comparison(left_value, right_value, spec)
                    row[f"{name}_left"] = left_value
                    row[f"{name}_right"] = right_value
                    row[f"{name}_no_worse"] = bool(no_worse)
                    row[f"{name}_materially_better"] = bool(material)
                    row["all_dimensions_no_worse"] &= bool(no_worse)
                    row["any_dimension_materially_better"] |= bool(material)
                row["dominates"] = bool(
                    row["all_dimensions_no_worse"] and row["any_dimension_materially_better"]
                )
                if row["dominates"]:
                    dominated_by[right_id].append(left_id)
                audit_rows.append(row)
        pareto_frames[pareto_kind] = pd.DataFrame(
            [
                {
                    "candidate_id": candidate,
                    "pareto_kind": pareto_kind,
                    "nondominated": len(dominated_by[candidate]) == 0,
                    "dominated_by_count": len(dominated_by[candidate]),
                    "dominated_by": ";".join(dominated_by[candidate]),
                }
                for candidate in candidate_order
            ]
        )
    return pd.DataFrame(audit_rows), pareto_frames["quality"], pareto_frames["quality_efficiency"]


def _validate_engineering(config: SummaryConfig) -> pd.DataFrame:
    provenance = config.raw["provenance"]
    receipt_path = _repo_path(provenance["gpu_engineering_receipt_path"], context="GPU receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_gpu_receipt(receipt)
    if receipt["status"] != "complete" or receipt["counts"] != {
        "actual_candidates": 5,
        "expected_candidates": 5,
        "passed_candidates": 5,
        "unavailable_candidates": 0,
    }:
        raise RuntimeError("GPU engineering receipt不完整")
    path = _repo_path(provenance["gpu_engineering_csv_path"], context="GPU engineering CSV")
    csv_artifact = next(
        (item for item in receipt["artifacts"] if item["filename"] == "candidate_engineering.csv"),
        None,
    )
    if csv_artifact is None or csv_artifact["sha256"] != provenance["gpu_engineering_csv_sha256"]:
        raise RuntimeError("GPU engineering CSV artifact provenance漂移")
    engineering = pd.read_csv(path)
    if engineering["candidate_id"].astype(str).tolist() != list(EXPECTED_CANDIDATES):
        raise RuntimeError("GPU engineering candidate order漂移")
    candidate_records = {row["candidate_id"]: row for row in config.raw["candidates"]}
    for _, row in engineering.iterrows():
        candidate_id = str(row["candidate_id"])
        if (
            row["status"] != "passed"
            or int(row["physical_batch_size"]) != 128
            or int(row["accumulation_steps"]) != 1
            or int(row["effective_batch_size"]) != 128
            or int(row["trainable_parameters"]) != int(candidate_records[candidate_id]["trainable_parameters"])
        ):
            raise RuntimeError(f"GPU engineering identity/scheme漂移: {candidate_id}")
    required_numeric = [
        "update_throughput_windows_per_second",
        "update_peak_allocated_mib",
        "batch1_inference_p50_ms",
        "batch1_inference_p90_ms",
        "update_peak_reserved_mib",
    ]
    values = engineering[required_numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise FloatingPointError("GPU engineering summary resource values无效")
    return engineering


def _row_ids_sha256(metrics: pd.DataFrame) -> str:
    values = metrics["dataset_row_id"].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _validate_metrics(
    metrics: pd.DataFrame,
    *,
    candidate_id: str,
    reference_static: pd.DataFrame | None,
) -> pd.DataFrame:
    if tuple(metrics.columns) != EXPECTED_METRICS_COLUMNS:
        raise ValueError(f"validation metrics schema漂移: {candidate_id}")
    if len(metrics) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"validation metrics行数漂移: {candidate_id}: {len(metrics)}")
    if (
        not metrics["evaluation_split"].astype(str).eq("validation").all()
        or not metrics["split"].astype(str).eq("val").all()
        or not metrics["method"].astype(str).eq(candidate_id).all()
        or int(metrics["samp_id"].nunique()) != EXPECTED_VALIDATION_SAMP_IDS
        or int(metrics["dataset_row_id"].nunique()) != EXPECTED_VALIDATION_ROWS
        or _row_ids_sha256(metrics) != EXPECTED_VALIDATION_ROW_HASH
    ):
        raise RuntimeError(f"validation metrics identity/split/row hash漂移: {candidate_id}")
    primary = metrics[list(PRIMARY_SAMPLE_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(primary).all():
        raise FloatingPointError(f"validation primary包含NaN/Inf: {candidate_id}")
    numeric = metrics.select_dtypes(include=[np.number, "bool"]).to_numpy(dtype=np.float64, copy=True)
    if np.isinf(numeric).any():
        raise FloatingPointError(f"validation metrics包含Inf: {candidate_id}")
    static = metrics[list(STATIC_VALIDATION_COLUMNS)].reset_index(drop=True)
    if reference_static is not None:
        try:
            pd.testing.assert_frame_equal(
                static,
                reference_static,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise RuntimeError(f"validation row/target static identity跨run漂移: {candidate_id}") from exc
    return static


def _validate_formal_run(
    config: SummaryConfig,
    record: Mapping[str, Any],
    reference_static: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    candidate_id = str(record["candidate_id"])
    seed = int(record["seed"])
    receipt_path = _repo_path(record["receipt_path"], context="formal receipt")
    if sha256_file(receipt_path) != record["receipt_sha256"]:
        raise RuntimeError(f"formal receipt hash漂移: {candidate_id}/{seed}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_formal_receipt(receipt)
    if (
        receipt["candidate_id"] != candidate_id
        or int(receipt["seed"]) != seed
        or receipt["execution"]["git_commit"] != FORMAL_EXECUTION_COMMIT
        or receipt["execution"]["execution_group_id"] != record["execution_group"]
        or str(receipt["execution"]["cuda_visible_devices"]) != str(record["cuda_visible_devices"])
    ):
        raise RuntimeError(f"formal receipt identity/group/commit漂移: {candidate_id}/{seed}")
    run_dir = receipt_path.parent
    sidecar = (run_dir / "formal_receipt.sha256").read_text(encoding="utf-8").strip()
    if sidecar != f"{record['receipt_sha256']}  formal_receipt.json":
        raise RuntimeError(f"formal receipt sidecar漂移: {candidate_id}/{seed}")
    lifecycle = json.loads((run_dir / "lifecycle.json").read_text(encoding="utf-8"))
    if (
        lifecycle.get("status") != "complete"
        or int(lifecycle.get("last_completed_epoch", -1)) != 80
        or int(lifecycle.get("last_optimizer_update", -1)) != 6400
        or lifecycle.get("formal_receipt_sha256") != record["receipt_sha256"]
        or lifecycle.get("failure") is not None
    ):
        raise RuntimeError(f"formal lifecycle不完整: {candidate_id}/{seed}")
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        receipt["manifest_sha256"] != manifest_hash
        or manifest.get("artifacts") != receipt["artifacts"][:-1]
        or receipt["artifacts"][-1].get("filename") != "artifact_manifest.json"
        or receipt["artifacts"][-1].get("sha256") != manifest_hash
    ):
        raise RuntimeError(f"formal manifest/receipt closure漂移: {candidate_id}/{seed}")
    artifacts = {artifact["filename"]: artifact for artifact in receipt["artifacts"]}
    metrics_path = run_dir / "validation_metrics.csv"
    stored_summary_path = run_dir / "validation_metrics_summary.csv"
    for path in (metrics_path, stored_summary_path):
        artifact = artifacts[path.name]
        if sha256_file(path) != artifact["sha256"] or path.stat().st_size != int(artifact["size_bytes"]):
            raise RuntimeError(f"formal validation artifact hash/size漂移: {path}")
    if int(artifacts["validation_metrics.csv"]["rows"]) != EXPECTED_VALIDATION_ROWS or int(
        artifacts["validation_metrics_summary.csv"]["rows"]
    ) != 1:
        raise RuntimeError(f"formal validation artifact row receipt漂移: {candidate_id}/{seed}")
    metrics = pd.read_csv(metrics_path)
    static = _validate_metrics(metrics, candidate_id=candidate_id, reference_static=reference_static)
    recomputed = summarize_task_metrics(metrics)
    stored = pd.read_csv(stored_summary_path)
    try:
        pd.testing.assert_frame_equal(
            stored,
            recomputed,
            check_dtype=False,
            check_exact=False,
            rtol=float(config.raw["aggregation"]["stored_summary_rtol"]),
            atol=float(config.raw["aggregation"]["stored_summary_atol"]),
        )
    except AssertionError as exc:
        raise RuntimeError(f"stored summary重算不一致: {candidate_id}/{seed}") from exc
    candidate = next(item for item in config.raw["candidates"] if item["candidate_id"] == candidate_id)
    seed_row = {
        "candidate_id": candidate_id,
        "family": str(candidate["family"]),
        "role": str(candidate["role"]),
        "seed": seed,
        "execution_group": str(record["execution_group"]),
        "cuda_visible_devices": str(record["cuda_visible_devices"]),
        "selected_epoch": int(receipt["selector"]["best_epoch"]),
        "selected_validation_local_rr_mae": float(receipt["selector"]["best_validation_local_rr_mae"]),
        "trainable_parameters": int(candidate["trainable_parameters"]),
        "receipt_sha256": str(record["receipt_sha256"]),
        "metrics_sha256": str(artifacts["validation_metrics.csv"]["sha256"]),
        "stored_summary_sha256": str(artifacts["validation_metrics_summary.csv"]["sha256"]),
        **recomputed.iloc[0].to_dict(),
    }
    input_record = {
        "candidate_id": candidate_id,
        "seed": seed,
        "execution_group": str(record["execution_group"]),
        "cuda_visible_devices": str(record["cuda_visible_devices"]),
        "formal_receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
        "formal_receipt_sha256": str(record["receipt_sha256"]),
        "lifecycle_status": "complete",
        "artifact_manifest_sha256": manifest_hash,
        "validation_metrics_sha256": str(artifacts["validation_metrics.csv"]["sha256"]),
        "validation_summary_sha256": str(artifacts["validation_metrics_summary.csv"]["sha256"]),
        "validation_rows": len(metrics),
        "validation_samp_ids": int(metrics["samp_id"].nunique()),
        "checkpoint_content_read": False,
    }
    return seed_row, static, input_record


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("无法读取summary Git identity")
    return commit.stdout.strip(), bool(status.stdout.strip())


def _verify_provenance(config: SummaryConfig) -> tuple[str, dict[str, str]]:
    verified: dict[str, str] = {}
    provenance = config.raw["provenance"]
    for prefix in (
        "summary_protocol",
        "formal_dual_gpu_plan",
        "formal_dual_gpu_correction",
        "formal_dual_gpu_implementation_receipt",
        "gpu_engineering_csv",
        "gpu_engineering_receipt",
        "gpu_engineering_manifest",
    ):
        path = _repo_path(provenance[f"{prefix}_path"], context=prefix)
        actual = sha256_file(path)
        expected = str(provenance[f"{prefix}_sha256"])
        if actual != expected:
            raise RuntimeError(f"summary provenance hash漂移: {path}: {actual} != {expected}")
        verified[f"{prefix}_sha256"] = actual
    commit, dirty = _git_identity()
    if dirty:
        raise RuntimeError("RTM-v1 validation summary要求干净Git工作树")
    return commit, verified


def _frame_finite_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame.select_dtypes(include=[np.number, "bool"]).to_numpy(dtype=np.float64, copy=True)
    null = np.isnan(values)
    nonfinite = ~np.isfinite(values) & ~null
    return {
        "numeric_total": int(values.size),
        "numeric_finite": int(np.isfinite(values).sum()),
        "numeric_null": int(null.sum()),
        "numeric_nonfinite": int(nonfinite.sum()),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "rows": rows,
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scipy", "omegaconf"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_summary_receipt(receipt: Mapping[str, Any]) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "created_utc",
            "status",
            "evidence_label",
            "execution",
            "inputs",
            "counts",
            "finite_audit",
            "access",
            "decision",
            "artifacts",
            "manifest_sha256",
        },
        "summary receipt",
    )
    if (
        receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["protocol_id"] != PROTOCOL_ID
        or receipt["status"] != "complete"
        or receipt["evidence_label"] != "validation-development evidence"
    ):
        raise ValueError("summary receipt identity/status/evidence无效")
    execution = receipt["execution"]
    _require_exact_keys(
        execution,
        {
            "command",
            "cwd",
            "git_commit",
            "git_dirty",
            "python_version",
            "platform",
            "dependencies",
            "config_path",
            "config_sha256",
            "provenance",
        },
        "summary receipt.execution",
    )
    if (
        not str(execution["command"]).strip()
        or execution["cwd"] != str(REPO_ROOT)
        or bool(execution["git_dirty"])
        or execution["config_path"] != "configs/resp_temporal_v1/validation_summary_v1.yaml"
        or execution["config_sha256"] != FROZEN_CONFIG_SHA256
    ):
        raise ValueError("summary receipt execution provenance无效")
    if not isinstance(receipt["inputs"], list) or len(receipt["inputs"]) != 15:
        raise ValueError("summary receipt必须包含15项input")
    identities = {(item["candidate_id"], int(item["seed"])) for item in receipt["inputs"]}
    expected = {(candidate, seed) for candidate in EXPECTED_CANDIDATES for seed in FORMAL_SEEDS}
    if identities != expected or any(item.get("checkpoint_content_read") is not False for item in receipt["inputs"]):
        raise ValueError("summary receipt input identity/checkpoint access越界")
    if receipt["counts"] != {
        "expected_formal_runs": 15,
        "actual_formal_runs": 15,
        "validation_rows_per_run": 2675,
        "total_validation_metric_rows_read": 40125,
        "seed_summary_rows": 15,
        "candidate_summary_rows": 5,
        "paired_seed_direction_rows": 50,
        "dominance_audit_rows": 40,
        "quality_pareto_rows": 5,
        "quality_efficiency_pareto_rows": 5,
        "supplementary_rows": 5,
    }:
        raise ValueError("summary receipt count closure无效")
    finite = receipt["finite_audit"]
    _require_exact_keys(finite, {"components", "aggregate"}, "summary finite audit")
    if set(finite["components"]) != {
        "seed_summary",
        "candidate_summary",
        "paired_seed_directions",
        "dominance_audit",
        "quality_pareto",
        "quality_efficiency_pareto",
        "supplementary_metrics",
    }:
        raise ValueError("summary finite components漂移")
    for name, values in {**finite["components"], "aggregate": finite["aggregate"]}.items():
        _require_exact_keys(
            values,
            {"numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite"},
            f"summary finite {name}",
        )
        if (
            any(int(value) < 0 for value in values.values())
            or int(values["numeric_nonfinite"]) != 0
            or int(values["numeric_total"])
            != int(values["numeric_finite"]) + int(values["numeric_null"])
        ):
            raise ValueError(f"summary finite closure无效: {name}")
    if receipt["access"] != {
        "formal_receipts_accessed": True,
        "formal_lifecycle_and_manifest_accessed": True,
        "validation_metrics_accessed": True,
        "validation_stored_summary_accessed": True,
        "gpu_engineering_artifacts_accessed": True,
        "checkpoint_metadata_records_read": True,
        "checkpoint_content_read": False,
        "dataset_or_index_accessed": False,
        "signal_or_target_array_accessed": False,
        "model_training_used": False,
        "model_inference_used": False,
        "gpu_used": False,
        "research_test_accessed": False,
    }:
        raise ValueError("summary receipt access越界")
    decision = receipt["decision"]
    _require_exact_keys(
        decision,
        {
            "quality_pareto_set",
            "quality_efficiency_pareto_set",
            "stop_line",
            "secondary_used_for_quality_selection",
            "research_test_opened",
        },
        "summary receipt decision",
    )
    if decision["secondary_used_for_quality_selection"] or decision["research_test_opened"]:
        raise ValueError("summary receipt decision越界")
    if not isinstance(receipt["artifacts"], list) or len(receipt["artifacts"]) != len(OUTPUT_FILENAMES):
        raise ValueError("summary receipt artifact count无效")
    if [item["filename"] for item in receipt["artifacts"]] != list(OUTPUT_FILENAMES):
        raise ValueError("summary receipt artifact order漂移")
    for artifact in receipt["artifacts"]:
        _require_exact_keys(artifact, {"filename", "sha256", "size_bytes", "rows"}, "summary artifact")
        if not _is_sha256(artifact["sha256"]) or int(artifact["size_bytes"]) <= 0:
            raise ValueError("summary artifact hash/size无效")
    if receipt["manifest_sha256"] != receipt["artifacts"][-1]["sha256"]:
        raise ValueError("summary manifest hash closure无效")


def _supplementary_frame(
    candidate_summary: pd.DataFrame,
    supplementary: Mapping[str, Any],
) -> pd.DataFrame:
    columns = ["candidate_id", "family", "role"]
    for metric in supplementary["seed_aggregated"]:
        columns.extend([f"{metric}_seed_mean", f"{metric}_seed_sd", f"{metric}_seed_n"])
    columns.extend(str(metric) for metric in supplementary["engineering"])
    missing = [column for column in columns if column not in candidate_summary]
    if missing:
        raise ValueError(f"supplementary columns缺失: {missing}")
    return candidate_summary[columns].copy()


def _decision_payload(
    config: SummaryConfig,
    candidate_summary: pd.DataFrame,
    quality_pareto: pd.DataFrame,
    efficiency_pareto: pd.DataFrame,
) -> dict[str, Any]:
    quality_set = quality_pareto.loc[quality_pareto["nondominated"], "candidate_id"].astype(str).tolist()
    efficiency_set = efficiency_pareto.loc[
        efficiency_pareto["nondominated"], "candidate_id"
    ].astype(str).tolist()
    primary_names = {str(item["name"]) for item in config.raw["primary_metrics"]}
    t0_id = EXPECTED_CANDIDATES[0]
    t0 = candidate_summary.loc[candidate_summary["candidate_id"].eq(t0_id)].iloc[0]
    trunk_records: list[dict[str, Any]] = []
    for candidate_id in EXPECTED_CANDIDATES[1:]:
        candidate = candidate_summary.loc[candidate_summary["candidate_id"].eq(candidate_id)].iloc[0]
        improved_metrics = []
        for spec in config.raw["primary_metrics"]:
            _, material = dimension_comparison(
                _candidate_value(candidate, spec, primary_names=primary_names),
                _candidate_value(t0, spec, primary_names=primary_names),
                spec,
            )
            if material:
                improved_metrics.append(str(spec["name"]))
        trunk_records.append(
            {
                "candidate_id": candidate_id,
                "any_material_primary_improvement_vs_t0": bool(improved_metrics),
                "materially_improved_primary_metrics": improved_metrics,
            }
        )
    any_trunk = any(record["any_material_primary_improvement_vs_t0"] for record in trunk_records)
    return {
        "schema_version": "rtm-v1-validation-summary-decision-v1",
        "protocol_id": PROTOCOL_ID,
        "evidence_label": "validation-development evidence",
        "config_sha256": config.sha256,
        "quality_pareto_set": quality_set,
        "quality_efficiency_pareto_set": efficiency_set,
        "t0_role": "trunk_attribution_control",
        "trunk_vs_t0": trunk_records,
        "stop_line": (
            "at_least_one_trunk_material_primary_improvement"
            if any_trunk
            else "all_trunks_no_material_primary_improvement_stop"
        ),
        "quality_and_efficiency_pareto_separated": True,
        "secondary_used_for_quality_selection": False,
        "weighted_score_used": False,
        "confirmatory_p_value_used": False,
        "research_test_opened": False,
        "protocol_closes_after_summary": True,
    }


def run_validation_summary(*, config_path: str | Path = DEFAULT_CONFIG_PATH, command: str) -> Path:
    """只读冻结formal validation artifacts并原子发布一次性summary；用户手动调用。"""

    config = load_summary_config(config_path)
    if config.output_dir.exists():
        raise FileExistsError(f"RTM-v1 validation summary输出禁止覆盖: {config.output_dir}")
    git_commit, provenance = _verify_provenance(config)
    engineering = _validate_engineering(config)
    seed_rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    reference_static: pd.DataFrame | None = None
    for record in config.raw["formal_runs"]:
        seed_row, static, input_record = _validate_formal_run(config, record, reference_static)
        if reference_static is None:
            reference_static = static
        seed_rows.append(seed_row)
        input_records.append(input_record)
    seed_summary = pd.DataFrame(seed_rows)
    candidate_summary = build_candidate_summary(
        seed_summary,
        config.raw["candidates"],
        config.raw["seeds"],
        engineering,
    )
    paired = build_paired_seed_directions(
        seed_summary,
        list(EXPECTED_CANDIDATES),
        config.raw["primary_metrics"],
        config.raw["seeds"],
    )
    dominance, quality_pareto, efficiency_pareto = build_pareto_outputs(
        candidate_summary,
        config.raw["primary_metrics"],
        config.raw["quality_efficiency_metrics"],
    )
    supplementary = _supplementary_frame(candidate_summary, config.raw["supplementary_metrics"])
    decision = _decision_payload(config, candidate_summary, quality_pareto, efficiency_pareto)
    frames = {
        "seed_summary.csv": seed_summary,
        "candidate_summary.csv": candidate_summary,
        "paired_seed_directions.csv": paired,
        "dominance_audit.csv": dominance,
        "quality_pareto.csv": quality_pareto,
        "quality_efficiency_pareto.csv": efficiency_pareto,
        "supplementary_metrics.csv": supplementary,
    }
    expected_rows = {
        "seed_summary.csv": 15,
        "candidate_summary.csv": 5,
        "paired_seed_directions.csv": 50,
        "dominance_audit.csv": 40,
        "quality_pareto.csv": 5,
        "quality_efficiency_pareto.csv": 5,
        "supplementary_metrics.csv": 5,
    }
    for name, expected in expected_rows.items():
        if len(frames[name]) != expected:
            raise RuntimeError(f"summary output rows漂移: {name}: {len(frames[name])} != {expected}")
    finite_components = {name.removesuffix(".csv"): _frame_finite_counts(frame) for name, frame in frames.items()}
    finite_aggregate = {
        key: sum(component[key] for component in finite_components.values())
        for key in ("numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite")
    }
    if finite_aggregate["numeric_nonfinite"] != 0:
        raise FloatingPointError("summary output包含Inf")

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_dir.parent / f".{config.output_dir.name}.tmp.{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    failed_path: Path | None = None
    try:
        _write_json(temporary / "resolved_config.json", config.raw)
        float_format = str(config.raw["output"]["csv_float_format"])
        for name, frame in frames.items():
            frame.to_csv(temporary / name, index=False, float_format=float_format)
        _write_json(temporary / "decision.json", decision)
        artifact_specs = [
            ("resolved_config.json", None),
            *[(name, expected_rows[name]) for name in frames],
            ("decision.json", None),
        ]
        artifacts = [_artifact_record(temporary / name, rows=rows) for name, rows in artifact_specs]
        manifest = {
            "schema_version": "rtm-v1-validation-summary-manifest-v1",
            "protocol_id": PROTOCOL_ID,
            "config_sha256": config.sha256,
            "artifacts": artifacts,
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        manifest_record = _artifact_record(temporary / "artifact_manifest.json")
        artifacts.append(manifest_record)
        receipt_decision = {
            "quality_pareto_set": decision["quality_pareto_set"],
            "quality_efficiency_pareto_set": decision["quality_efficiency_pareto_set"],
            "stop_line": decision["stop_line"],
            "secondary_used_for_quality_selection": False,
            "research_test_opened": False,
        }
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": "validation-development evidence",
            "execution": {
                "command": command,
                "cwd": str(REPO_ROOT),
                "git_commit": git_commit,
                "git_dirty": False,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": _dependency_versions(),
                "config_path": str(config.path.relative_to(REPO_ROOT)),
                "config_sha256": config.sha256,
                "provenance": provenance,
            },
            "inputs": input_records,
            "counts": {
                "expected_formal_runs": 15,
                "actual_formal_runs": len(input_records),
                "validation_rows_per_run": EXPECTED_VALIDATION_ROWS,
                "total_validation_metric_rows_read": EXPECTED_VALIDATION_ROWS * len(input_records),
                "seed_summary_rows": len(seed_summary),
                "candidate_summary_rows": len(candidate_summary),
                "paired_seed_direction_rows": len(paired),
                "dominance_audit_rows": len(dominance),
                "quality_pareto_rows": len(quality_pareto),
                "quality_efficiency_pareto_rows": len(efficiency_pareto),
                "supplementary_rows": len(supplementary),
            },
            "finite_audit": {"components": finite_components, "aggregate": finite_aggregate},
            "access": {
                "formal_receipts_accessed": True,
                "formal_lifecycle_and_manifest_accessed": True,
                "validation_metrics_accessed": True,
                "validation_stored_summary_accessed": True,
                "gpu_engineering_artifacts_accessed": True,
                "checkpoint_metadata_records_read": True,
                "checkpoint_content_read": False,
                "dataset_or_index_accessed": False,
                "signal_or_target_array_accessed": False,
                "model_training_used": False,
                "model_inference_used": False,
                "gpu_used": False,
                "research_test_accessed": False,
            },
            "decision": receipt_decision,
            "artifacts": artifacts,
            "manifest_sha256": manifest_record["sha256"],
        }
        validate_summary_receipt(receipt)
        _write_json(temporary / "access_receipt.json", receipt)
        receipt_hash = sha256_file(temporary / "access_receipt.json")
        (temporary / "access_receipt.sha256").write_text(
            f"{receipt_hash}  access_receipt.json\n", encoding="utf-8"
        )
        if config.output_dir.exists():
            raise FileExistsError(f"summary输出在运行期间出现，拒绝覆盖: {config.output_dir}")
        os.replace(temporary, config.output_dir)
        return config.output_dir / "access_receipt.json"
    except BaseException as exc:
        if temporary.exists():
            _write_json(
                temporary / "failure_receipt.json",
                {
                    "schema_version": "rtm-v1-validation-summary-failure-v1",
                    "protocol_id": PROTOCOL_ID,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "config_sha256": config.sha256,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "research_test_accessed": False,
                    "checkpoint_content_read": False,
                },
            )
            failed_path = temporary.with_name(f".{config.output_dir.name}.failed.{uuid4().hex}")
            os.replace(temporary, failed_path)
        if failed_path is not None:
            print(f"RTM-v1 validation summary failure artifacts: {failed_path}", flush=True)
        raise
