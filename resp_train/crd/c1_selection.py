from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from resp_train.crd.candidate_lock import (
    DEFAULT_CANDIDATE_LOCK,
    REPO_ROOT,
    resolve_repo_path,
    sha256_file,
    verify_candidate_lock,
)
from resp_train.crd.config import CRD_CONTROLS_PROTOCOL_VERSION, FORMAL_SEEDS, load_crd_config
from resp_train.crd.s2_selection import BASE_VARIANT
from resp_train.metrics.task import summarize_task_metrics
from resp_train.utils.run import save_execution_manifest


C1_VARIANT = "crd_c101_b0_local_tcn"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs/crd_v1/crd_c101_b0_local_tcn"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_c1_validation_summary"
DEFAULT_FAILURE_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_failure_diagnostic"
DEFAULT_METADATA_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_failure_metadata_diagnostic"
DEFAULT_OBSERVABILITY_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_matched_observability_diagnostic"

SEED_SUMMARY_FILENAME = "c1_seed_summary.csv"
VARIANT_SUMMARY_FILENAME = "c1_variant_summary.csv"
PAIRED_WINDOW_FILENAME = "c1_paired_window_descriptives.csv"
PAIRED_SAMP_FILENAME = "c1_paired_samp_descriptives.csv"
FAILURE_STRATA_FILENAME = "c1_failure_strata_summary.csv"
DECISION_FILENAME = "c1_selection.json"
MANIFEST_FILENAME = "c1_selection_manifest.json"

REPORT_METRICS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
    "ibi_medae_sec",
    "ibi_coverage",
    "target_stratified_envelope_spearman",
)
SEED_SUMMARY_METRICS = REPORT_METRICS[:-1]
PRIMARY_METRICS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
)


def apply_c1_decision(seed_summary: pd.DataFrame) -> dict[str, Any]:
    """应用 C1 第 4.2 节冻结的 quality-superior/near 门槛。"""

    required = {"variant", "seed", *(f"{metric}_mean" for metric in SEED_SUMMARY_METRICS)}
    if not required.issubset(seed_summary.columns):
        raise ValueError(f"C1 seed summary 缺少字段: {sorted(required - set(seed_summary.columns))}")
    rows = seed_summary.copy()
    if set(rows["variant"]) != {BASE_VARIANT, C1_VARIANT}:
        raise ValueError("C1 decision 必须恰含 BASE 与 TCN")
    for variant in (BASE_VARIANT, C1_VARIANT):
        seeds = sorted(rows.loc[rows["variant"].eq(variant), "seed"].astype(int).tolist())
        if seeds != list(FORMAL_SEEDS):
            raise ValueError(f"C1 {variant} seeds 不完整: {seeds}")

    base = rows.loc[rows["variant"].eq(BASE_VARIANT)].sort_values("seed").reset_index(drop=True)
    candidate = rows.loc[rows["variant"].eq(C1_VARIANT)].sort_values("seed").reset_index(drop=True)
    base_local = float(base["local_rr_mae_bpm_mean"].mean())
    candidate_local = float(candidate["local_rr_mae_bpm_mean"].mean())
    base_pcc = float(base["lag_aware_signed_pcc_mean"].mean())
    candidate_pcc = float(candidate["lag_aware_signed_pcc_mean"].mean())
    base_trajectory = float(base["envelope_trajectory_mae_mean"].mean())
    candidate_trajectory = float(candidate["envelope_trajectory_mae_mean"].mean())
    values = (
        base_local,
        candidate_local,
        base_pcc,
        candidate_pcc,
        base_trajectory,
        candidate_trajectory,
    )
    if not all(math.isfinite(value) for value in values) or base_local <= 0.0 or base_trajectory <= 0.0:
        raise FloatingPointError("C1 decision 输入包含无效数值")

    local_improvement = (base_local - candidate_local) / base_local
    local_worsening = (candidate_local - base_local) / base_local
    paired_local_improved = int(
        np.sum(candidate["local_rr_mae_bpm_mean"].to_numpy() < base["local_rr_mae_bpm_mean"].to_numpy())
    )
    pcc_drop = base_pcc - candidate_pcc
    trajectory_worsening = (candidate_trajectory - base_trajectory) / base_trajectory
    quality_checks = {
        "local_rr_improvement_at_least_0_5pct": local_improvement >= 0.005,
        "paired_local_rr_improved_at_least_2_of_3": paired_local_improved >= 2,
        "pcc_drop_at_most_0_005": pcc_drop <= 0.005,
        "trajectory_worsening_at_most_1_5pct": trajectory_worsening <= 0.015,
    }
    quality_superior = bool(all(quality_checks.values()))
    quality_near_checks = {
        "local_rr_worsening_at_most_0_5pct": local_worsening <= 0.005,
        "pcc_drop_at_most_0_005": pcc_drop <= 0.005,
        "trajectory_worsening_at_most_1_5pct": trajectory_worsening <= 0.015,
    }
    quality_near_guardrails = bool(all(quality_near_checks.values()))
    if quality_superior:
        outcome = "quality_superior"
    elif quality_near_guardrails:
        outcome = "quality_near_requires_efficiency_benchmark"
    else:
        outcome = "mamba_retained_control_failure"
    return {
        "outcome": outcome,
        "quality_superior": quality_superior,
        "quality_checks": quality_checks,
        "quality_near_guardrails": quality_near_guardrails,
        "quality_near_checks": quality_near_checks,
        "efficiency_benchmark_required": bool(not quality_superior and quality_near_guardrails),
        "local_rr_relative_improvement": local_improvement,
        "paired_local_rr_improved_seed_count": paired_local_improved,
        "signed_pcc_absolute_drop": pcc_drop,
        "trajectory_relative_worsening": trajectory_worsening,
        "retain_crd_102": not quality_superior,
        "c2_may_be_activated_after_registration": True,
    }


def summarize_c1(
    *,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    failure_root: str | Path = DEFAULT_FAILURE_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    observability_root: str | Path = DEFAULT_OBSERVABILITY_ROOT,
) -> Path:
    """审计三个 C1 formal runs，重算冻结比较并生成不可覆盖 summary。"""

    _assert_clean_repository()
    verification = verify_candidate_lock(candidate_lock_path)
    base_records = sorted(
        (record for record in verification.records if record["variant"] == BASE_VARIANT),
        key=lambda item: int(item["seed"]),
    )
    if [int(record["seed"]) for record in base_records] != list(FORMAL_SEEDS):
        raise RuntimeError("candidate lock 中 CRD_102 seeds 不完整")

    metrics_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    seed_rows: list[dict[str, Any]] = []
    for record in base_records:
        seed = int(record["seed"])
        run_dir = resolve_repo_path(record["checkpoint_path"]).parent
        metrics, summary = _load_metrics(run_dir, BASE_VARIANT)
        metrics_by_key[(BASE_VARIANT, seed)] = metrics
        seed_rows.append(
            {
                "role": "anchor",
                "variant": BASE_VARIANT,
                "seed": seed,
                "selected_epoch": int(record["selected_epoch"]),
                "checkpoint_path": str(record["checkpoint_path"]),
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
                **summary,
            }
        )

    training_commits: set[str] = set()
    candidate_run_dirs: list[str] = []
    for seed in FORMAL_SEEDS:
        run_dir = _unique_run_dir(Path(runs_root), seed)
        audit, metrics, summary = _audit_candidate_run(run_dir, seed)
        training_commits.add(str(audit["training_commit"]))
        candidate_run_dirs.append(str(run_dir))
        metrics_by_key[(C1_VARIANT, seed)] = metrics
        seed_rows.append({"role": "candidate", **audit, **summary})
    if len(training_commits) != 1:
        raise RuntimeError(f"C1 formal runs training commit 不统一: {sorted(training_commits)}")

    seed_summary = pd.DataFrame(seed_rows).sort_values(["role", "variant", "seed"]).reset_index(drop=True)
    variant_summary = _variant_summary(seed_summary)
    paired_window, paired_samp = _paired_descriptives(metrics_by_key)
    failure_strata = _failure_strata_descriptives(
        metrics_by_key,
        failure_root=Path(failure_root),
        metadata_root=Path(metadata_root),
        observability_root=Path(observability_root),
    )
    decision = apply_c1_decision(seed_summary)

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"C1 summary 禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root.parent / f".{output_root.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        seed_summary.to_csv(temporary_dir / SEED_SUMMARY_FILENAME, index=False)
        variant_summary.to_csv(temporary_dir / VARIANT_SUMMARY_FILENAME, index=False)
        paired_window.to_csv(temporary_dir / PAIRED_WINDOW_FILENAME, index=False)
        paired_samp.to_csv(temporary_dir / PAIRED_SAMP_FILENAME, index=False)
        failure_strata.to_csv(temporary_dir / FAILURE_STRATA_FILENAME, index=False)
        (temporary_dir / DECISION_FILENAME).write_text(
            json.dumps(
                {
                    "protocol": CRD_CONTROLS_PROTOCOL_VERSION,
                    "candidate_lock_sha256": verification.lock_sha256,
                    "training_commit": next(iter(training_commits)),
                    "complete_candidate_run_count": 3,
                    "research_test_used": False,
                    "confirmatory_p_values_used": False,
                    **decision,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        save_execution_manifest(
            temporary_dir / MANIFEST_FILENAME,
            task="crd_c1_validation_summary",
            phase="c1_parameter_matched_tcn_control",
            protocol=CRD_CONTROLS_PROTOCOL_VERSION,
            candidate_lock_sha256=verification.lock_sha256,
            training_commit=next(iter(training_commits)),
            candidate_run_dirs=candidate_run_dirs,
            complete_candidate_run_count=3,
            decision_outcome=decision["outcome"],
            research_test_used=False,
        )
        os.replace(temporary_dir, output_root)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_root / DECISION_FILENAME


def _unique_run_dir(runs_root: Path, seed: int) -> Path:
    seed_root = runs_root / f"seed_{seed}"
    candidates = sorted(path for path in seed_root.iterdir() if path.is_dir()) if seed_root.exists() else []
    if len(candidates) != 1:
        raise RuntimeError(f"C1 seed={seed} 必须恰有一个 run，实际 {len(candidates)}: {candidates}")
    return candidates[0]


def _audit_candidate_run(run_dir: Path, seed: int) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    required = (
        "audit.csv",
        "checkpoint_best_local_rr.pt",
        "checkpoint_final.pt",
        "config.yaml",
        "metrics.csv",
        "metrics_summary.csv",
        "optimizer_parameter_groups.json",
        "run_manifest.json",
        "runtime_summary.json",
        "train_history.csv",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"C1 run 产物不完整 {run_dir}: {missing}")
    cfg = load_crd_config(run_dir / "config.yaml")
    if (
        str(cfg.model.variant) != C1_VARIANT
        or str(cfg.protocol.run_role) != "formal"
        or int(cfg.training.seed) != int(seed)
        or (int(cfg.training.epochs), int(cfg.training.batch_size), int(cfg.training.gradient_accumulation_steps))
        != (80, 128, 1)
    ):
        raise RuntimeError(f"C1 resolved config identity 不一致: {run_dir}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("git_dirty") is not False
        or manifest.get("protocol") != CRD_CONTROLS_PROTOCOL_VERSION
        or manifest.get("stage") != "c1"
        or manifest.get("run_role") != "formal"
    ):
        raise RuntimeError(f"C1 manifest identity 不一致: {run_dir}")

    history = pd.read_csv(run_dir / "train_history.csv")
    numeric_history = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if (
        len(history) != 80
        or history["epoch"].astype(int).tolist() != list(range(1, 81))
        or int(history.iloc[-1]["optimizer_update"]) != 6400
        or not np.isfinite(numeric_history).all()
    ):
        raise RuntimeError(f"C1 train history 不完整或非有限: {run_dir}")
    selected_epoch = int(history.loc[history["val_local_rr_mae"].idxmin(), "epoch"])
    best = torch.load(run_dir / "checkpoint_best_local_rr.pt", map_location="cpu")
    final = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu")
    if int(best["epoch"]) != selected_epoch or int(final["epoch"]) != 80:
        raise RuntimeError(f"C1 checkpoint selector/epoch 不一致: {run_dir}")
    for name, checkpoint, expected_update in (
        ("best", best, selected_epoch * 80),
        ("final", final, 6400),
    ):
        extra = checkpoint.get("extra_state", {})
        if int(extra.get("update_index", -1)) != expected_update or int(extra.get("total_updates", -1)) != 6400:
            raise RuntimeError(f"C1 {name} checkpoint update identity 不一致: {run_dir}")
        tensors = list(_iter_tensors(checkpoint["model_state_dict"])) + list(
            _iter_tensors(checkpoint["optimizer_state_dict"])
        )
        if not tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise FloatingPointError(f"C1 {name} checkpoint 包含 NaN/Inf: {run_dir}")

    metrics, summary = _load_metrics(run_dir, C1_VARIANT)
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    reserved_fraction = float(runtime.get("peak_reserved_fraction", math.nan))
    if not (0.0 < reserved_fraction <= 0.80):
        raise RuntimeError(f"C1 formal peak reserved fraction 越界: {run_dir}: {reserved_fraction}")
    return (
        {
            "variant": C1_VARIANT,
            "seed": int(seed),
            "selected_epoch": selected_epoch,
            "checkpoint_path": str(run_dir / "checkpoint_best_local_rr.pt"),
            "checkpoint_sha256": sha256_file(run_dir / "checkpoint_best_local_rr.pt"),
            "run_dir": str(run_dir),
            "training_commit": str(manifest.get("git_commit")),
            "peak_allocated_mib": float(runtime["peak_allocated_mib"]),
            "peak_reserved_mib": float(runtime["peak_reserved_mib"]),
            "peak_reserved_fraction": reserved_fraction,
        },
        metrics,
        summary,
    )


def _load_metrics(run_dir: Path, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    if (
        len(metrics) != 2675
        or metrics["dataset_row_id"].duplicated().any()
        or not metrics["split"].eq("val").all()
        or not metrics["evaluation_split"].eq("validation").all()
        or not metrics["method"].eq(variant).all()
    ):
        raise RuntimeError(f"C1 metrics identity 不一致: {run_dir}")
    primary = metrics.loc[:, list(PRIMARY_METRICS)].to_numpy(dtype=np.float64)
    if not np.isfinite(primary).all() or metrics["joint_prediction_degenerate"].astype(bool).any():
        raise FloatingPointError(f"C1 primary metrics 非有限或 prediction degenerate: {run_dir}")
    recomputed = summarize_task_metrics(metrics)
    stored = pd.read_csv(run_dir / "metrics_summary.csv")
    try:
        pd.testing.assert_frame_equal(
            recomputed,
            stored,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(f"C1 metrics_summary 与逐 sample 重算不一致: {run_dir}") from exc
    return metrics, recomputed.iloc[0].to_dict()


def _variant_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in seed_summary.groupby("variant", sort=True):
        row: dict[str, Any] = {"variant": variant, "seed_count": int(len(group))}
        for metric in SEED_SUMMARY_METRICS:
            column = f"{metric}_mean"
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_seed_mean"] = float(np.mean(finite)) if finite.size else math.nan
            row[f"{metric}_seed_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan
            row[f"{metric}_seed_n"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_descriptives(
    metrics_by_key: dict[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_parts: list[pd.DataFrame] = []
    for seed in FORMAL_SEEDS:
        base = metrics_by_key[(BASE_VARIANT, seed)]
        candidate = metrics_by_key[(C1_VARIANT, seed)]
        columns = ["dataset_row_id", "samp_id", *REPORT_METRICS]
        merged = base[columns].merge(candidate[columns], on=["dataset_row_id", "samp_id"], suffixes=("_base", "_tcn"))
        if len(merged) != 2675:
            raise RuntimeError(f"C1 seed={seed} paired window identity 不完整")
        result = merged[["dataset_row_id", "samp_id"]].copy()
        result.insert(0, "seed", seed)
        for metric in REPORT_METRICS:
            result[f"{metric}_base"] = merged[f"{metric}_base"]
            result[f"{metric}_tcn"] = merged[f"{metric}_tcn"]
            result[f"{metric}_tcn_minus_base"] = merged[f"{metric}_tcn"] - merged[f"{metric}_base"]
        window_parts.append(result)
    paired_window = pd.concat(window_parts, ignore_index=True)
    aggregate_columns = [column for column in paired_window.columns if column not in {"seed", "dataset_row_id", "samp_id"}]
    paired_samp = paired_window.groupby(["seed", "samp_id"], as_index=False)[aggregate_columns].mean()
    return paired_window, paired_samp


def _failure_strata_descriptives(
    metrics_by_key: dict[tuple[str, int], pd.DataFrame],
    *,
    failure_root: Path,
    metadata_root: Path,
    observability_root: Path,
) -> pd.DataFrame:
    consensus = pd.read_csv(failure_root / "crd102_window_consensus.csv")
    metadata = pd.read_csv(metadata_root / "crd102_failure_metadata_windows.csv")
    pairs = pd.read_csv(observability_root / "crd102_observability_match_pairs.csv")
    exact = pairs.loc[pairs["match_scheme"].eq("exact_state_primary")]
    matched_roles: dict[int, str] = {}
    for value in exact["case_dataset_row_id"]:
        matched_roles[int(value)] = "matched_case"
    for value in exact["control_dataset_row_id"]:
        matched_roles[int(value)] = "matched_control"
    frozen = consensus[
        ["dataset_row_id", "local_rr_mae_bpm_persistent_failure", "envelope_target_stratum"]
    ].merge(
        metadata[["dataset_row_id", "waveform_confidence_level"]],
        on="dataset_row_id",
        how="left",
        validate="one_to_one",
    )
    frozen["matched_observability_role"] = frozen["dataset_row_id"].map(matched_roles).fillna("not_matched")
    frozen["high_modulation_local_rr_failure"] = np.where(
        frozen["local_rr_mae_bpm_persistent_failure"].astype(bool)
        & frozen["envelope_target_stratum"].eq("high"),
        "yes",
        "no",
    )

    rows: list[dict[str, Any]] = []
    dimensions = {
        "local_rr_persistent_failure": "local_rr_mae_bpm_persistent_failure",
        "waveform_confidence": "waveform_confidence_level",
        "target_modulation": "envelope_target_stratum",
        "matched_observability": "matched_observability_role",
        "high_modulation_local_rr_failure": "high_modulation_local_rr_failure",
    }
    for seed in FORMAL_SEEDS:
        base = metrics_by_key[(BASE_VARIANT, seed)]
        candidate = metrics_by_key[(C1_VARIANT, seed)]
        columns = ["dataset_row_id", *REPORT_METRICS]
        merged = base[columns].merge(candidate[columns], on="dataset_row_id", suffixes=("_base", "_tcn"))
        merged = merged.merge(frozen, on="dataset_row_id", how="left", validate="one_to_one")
        for dimension, column in dimensions.items():
            for label, group in merged.groupby(column, dropna=False, sort=True):
                for metric in REPORT_METRICS:
                    base_values = pd.to_numeric(group[f"{metric}_base"], errors="coerce").to_numpy(dtype=np.float64)
                    tcn_values = pd.to_numeric(group[f"{metric}_tcn"], errors="coerce").to_numpy(dtype=np.float64)
                    valid = np.isfinite(base_values) & np.isfinite(tcn_values)
                    rows.append(
                        {
                            "seed": seed,
                            "dimension": dimension,
                            "label": str(label),
                            "metric": metric,
                            "n": int(np.sum(valid)),
                            "base_mean": float(np.mean(base_values[valid])) if np.any(valid) else math.nan,
                            "tcn_mean": float(np.mean(tcn_values[valid])) if np.any(valid) else math.nan,
                            "tcn_minus_base_mean": (
                                float(np.mean(tcn_values[valid] - base_values[valid])) if np.any(valid) else math.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_tensors(child)


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"无法检查 Git 工作树: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("C1 冻结汇总要求干净 Git 工作树")
