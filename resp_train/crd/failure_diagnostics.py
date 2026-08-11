from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK, REPO_ROOT, sha256_file, verify_candidate_lock
from resp_train.crd.config import CRD_102_FAILURE_DIAGNOSTIC_PROTOCOL_VERSION, FORMAL_SEEDS
from resp_train.crd.s2_selection import BASE_VARIANT, _load_and_validate_metrics
from resp_train.utils.run import save_execution_manifest


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_failure_diagnostic"

SEED_AUDIT_FILENAME = "crd102_seed_audit.csv"
THRESHOLD_FILENAME = "crd102_failure_thresholds.csv"
SEED_WINDOW_FILENAME = "crd102_seed_window_flags.csv"
WINDOW_FILENAME = "crd102_window_consensus.csv"
FAILURE_WINDOW_FILENAME = "crd102_failure_windows.csv"
STRATA_FILENAME = "crd102_strata_summary.csv"
SEED_STRATA_FILENAME = "crd102_seed_strata_summary.csv"
SIGNATURE_FILENAME = "crd102_failure_signature_summary.csv"
ASSOCIATION_FILENAME = "crd102_metric_associations.csv"
SEED_AGREEMENT_FILENAME = "crd102_seed_agreement.csv"
OVERVIEW_FILENAME = "crd102_failure_overview.json"
MANIFEST_FILENAME = "crd102_failure_diagnostic_manifest.json"

IDENTITY_COLUMNS = ("dataset_row_id", "samp_id", "coupling_state_id")
STATIC_COLUMNS = (
    "evaluation_split",
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

# direction 表示任务指标的优化方向；hard tail 始终转换为“越差越大”的含义。
DIAGNOSTIC_METRICS: dict[str, str] = {
    "whole_rr_abs_error_bpm": "minimize",
    "local_rr_mae_bpm": "minimize",
    "envelope_trajectory_mae": "minimize",
    "global_envelope_modulation_error": "minimize",
    "lag_aware_signed_pcc": "maximize",
    "ibi_medae_sec": "minimize",
    "ibi_coverage": "maximize",
    "target_stratified_envelope_spearman": "maximize",
}
CORE_FAILURE_METRICS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
)
STRATA_COLUMNS = (
    "samp_id",
    "coupling_state_id",
    "envelope_target_stratum",
    "ibi_interpretability_group",
    "lag_boundary_group",
)


def summarize_crd102_failure_diagnostics(
    *,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """审计冻结 CRD_102 validation 指标并生成探索性失败模式描述。"""

    _assert_clean_repository()
    verification = verify_candidate_lock(candidate_lock_path)
    records = sorted(
        (record for record in verification.records if record["variant"] == BASE_VARIANT),
        key=lambda item: int(item["seed"]),
    )
    if len(records) != len(FORMAL_SEEDS) or {int(record["seed"]) for record in records} != set(FORMAL_SEEDS):
        raise RuntimeError("candidate lock 中 CRD_102 三个固定 seed 不完整")

    metrics_by_seed: dict[int, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        checkpoint_path = Path(record["checkpoint_path"])
        run_dir = checkpoint_path.parent
        metrics, _ = _load_and_validate_metrics(run_dir, BASE_VARIANT)
        metrics_by_seed[seed] = metrics
        audit_rows.append(
            {
                "variant": BASE_VARIANT,
                "seed": seed,
                "selected_epoch": int(record["selected_epoch"]),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
                "metrics_path": str(run_dir / "metrics.csv"),
                "metrics_sha256": sha256_file(run_dir / "metrics.csv"),
                "metrics_rows": int(len(metrics)),
                "samp_id_count": int(metrics["samp_id"].nunique()),
            }
        )

    outputs = build_failure_diagnostics(metrics_by_seed)
    overview = _build_overview(outputs["window_consensus"], outputs["strata_summary"])
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"CRD_102 failure diagnostic 禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root.parent / f".{output_root.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        pd.DataFrame(audit_rows).to_csv(temporary_dir / SEED_AUDIT_FILENAME, index=False)
        outputs["thresholds"].to_csv(temporary_dir / THRESHOLD_FILENAME, index=False)
        outputs["seed_window_flags"].to_csv(temporary_dir / SEED_WINDOW_FILENAME, index=False)
        outputs["window_consensus"].to_csv(temporary_dir / WINDOW_FILENAME, index=False)
        outputs["failure_windows"].to_csv(temporary_dir / FAILURE_WINDOW_FILENAME, index=False)
        outputs["strata_summary"].to_csv(temporary_dir / STRATA_FILENAME, index=False)
        outputs["seed_strata_summary"].to_csv(temporary_dir / SEED_STRATA_FILENAME, index=False)
        outputs["signature_summary"].to_csv(temporary_dir / SIGNATURE_FILENAME, index=False)
        outputs["metric_associations"].to_csv(temporary_dir / ASSOCIATION_FILENAME, index=False)
        outputs["seed_agreement"].to_csv(temporary_dir / SEED_AGREEMENT_FILENAME, index=False)
        (temporary_dir / OVERVIEW_FILENAME).write_text(
            json.dumps(
                {
                    "protocol": CRD_102_FAILURE_DIAGNOSTIC_PROTOCOL_VERSION,
                    "candidate_lock": str(verification.lock_path),
                    "candidate_lock_sha256": verification.lock_sha256,
                    "checkpoint_count": len(records),
                    "research_test_used": False,
                    "checkpoint_reselection_allowed": False,
                    "confirmatory_p_values_used": False,
                    "exploratory_diagnostic_only": True,
                    **overview,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        save_execution_manifest(
            temporary_dir / MANIFEST_FILENAME,
            task="crd_102_validation_failure_diagnostic",
            phase="crd102_exploratory_failure_diagnostic",
            protocol=CRD_102_FAILURE_DIAGNOSTIC_PROTOCOL_VERSION,
            candidate_lock=str(verification.lock_path),
            candidate_lock_sha256=verification.lock_sha256,
            checkpoint_count=len(records),
            checkpoint_sha256_by_seed={str(row["seed"]): row["checkpoint_sha256"] for row in audit_rows},
            metrics_sha256_by_seed={str(row["seed"]): row["metrics_sha256"] for row in audit_rows},
            validation_window_count=int(len(outputs["window_consensus"])),
            research_test_used=False,
        )
        os.replace(temporary_dir, output_root)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_root / OVERVIEW_FILENAME


def build_failure_diagnostics(metrics_by_seed: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """纯计算入口：以 seed-specific worst decile 构造跨 seed 持续失败签名。"""

    if set(metrics_by_seed) != set(FORMAL_SEEDS):
        raise ValueError(f"CRD_102 diagnostic 要求固定 seeds={FORMAL_SEEDS}")
    sorted_frames = {
        seed: _sort_and_validate_frame(frame, seed=seed)
        for seed, frame in sorted(metrics_by_seed.items())
    }
    reference = sorted_frames[FORMAL_SEEDS[0]]
    for seed in FORMAL_SEEDS[1:]:
        _assert_cross_seed_identity(reference, sorted_frames[seed], seed=seed)

    seed_window_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    values_by_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in DIAGNOSTIC_METRICS}
    hard_by_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in DIAGNOSTIC_METRICS}
    eligible_by_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in DIAGNOSTIC_METRICS}
    lag_boundary_by_seed: list[np.ndarray] = []
    ibi_interpretable_by_seed: list[np.ndarray] = []

    for seed, frame in sorted_frames.items():
        seed_window = frame[list(IDENTITY_COLUMNS)].copy()
        seed_window.insert(0, "seed", seed)
        for metric, direction in DIAGNOSTIC_METRICS.items():
            values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=np.float64)
            eligible = _metric_eligibility(frame, metric)
            if not np.isfinite(values[eligible]).all():
                raise FloatingPointError(f"seed={seed} eligible {metric} 包含 NaN/Inf")
            finite_eligible = eligible & np.isfinite(values)
            tail_quantile = 0.90 if direction == "minimize" else 0.10
            threshold = float(np.quantile(values[finite_eligible], tail_quantile)) if finite_eligible.any() else math.nan
            if direction == "minimize":
                hard = finite_eligible & (values >= threshold)
            else:
                hard = finite_eligible & (values <= threshold)
            values_by_metric[metric].append(values)
            hard_by_metric[metric].append(hard)
            eligible_by_metric[metric].append(eligible)
            seed_window[metric] = values
            seed_window[f"{metric}_eligible"] = eligible
            seed_window[f"{metric}_worst_decile"] = hard
            threshold_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "direction": direction,
                    "tail_quantile": tail_quantile,
                    "threshold": threshold,
                    "eligible_n": int(finite_eligible.sum()),
                    "worst_decile_n_including_ties": int(hard.sum()),
                }
            )

        lag_boundary = np.isclose(
            np.abs(pd.to_numeric(frame["best_lag_sec"], errors="coerce").to_numpy(dtype=np.float64)),
            0.30,
            atol=1e-12,
        ) & _as_bool(frame["joint_target_eligible"]) & ~_as_bool(frame["joint_prediction_degenerate"])
        ibi_interpretable = _as_bool(frame["ibi_target_eligible"]) & _as_bool(frame["ibi_interpretable"])
        seed_window["lag_boundary"] = lag_boundary
        seed_window["ibi_interpretable"] = ibi_interpretable
        lag_boundary_by_seed.append(lag_boundary)
        ibi_interpretable_by_seed.append(ibi_interpretable)
        seed_window_frames.append(seed_window)

    consensus = reference[list(dict.fromkeys((*IDENTITY_COLUMNS, *STATIC_COLUMNS)))].copy()
    for metric in DIAGNOSTIC_METRICS:
        values = np.vstack(values_by_metric[metric])
        finite = np.isfinite(values)
        consensus[f"{metric}_seed_mean"] = _row_nanmean(values)
        consensus[f"{metric}_seed_sd"] = _row_nanstd(values)
        consensus[f"{metric}_eligible_seed_count"] = np.vstack(eligible_by_metric[metric]).sum(axis=0)
        hard_count = np.vstack(hard_by_metric[metric]).sum(axis=0)
        consensus[f"{metric}_worst_decile_seed_count"] = hard_count
        consensus[f"{metric}_persistent_failure"] = hard_count >= 2

    consensus["ibi_interpretable_seed_count"] = np.vstack(ibi_interpretable_by_seed).sum(axis=0)
    consensus["lag_boundary_seed_count"] = np.vstack(lag_boundary_by_seed).sum(axis=0)
    consensus["ibi_interpretability_group"] = np.select(
        [consensus["ibi_interpretable_seed_count"].eq(3), consensus["ibi_interpretable_seed_count"].eq(0)],
        ["all_interpretable", "none_interpretable"],
        default="mixed_interpretable",
    )
    consensus["lag_boundary_group"] = np.where(
        consensus["lag_boundary_seed_count"].ge(2), "persistent_boundary", "nonpersistent_boundary"
    )
    core_failure_columns = [f"{metric}_persistent_failure" for metric in CORE_FAILURE_METRICS]
    consensus["persistent_core_failure_count"] = consensus[core_failure_columns].sum(axis=1).astype(int)
    local_sd = consensus["local_rr_mae_bpm_seed_sd"].to_numpy(dtype=np.float64)
    local_sd_threshold = float(np.nanquantile(local_sd, 0.90))
    consensus["local_rr_seed_sd_top_decile"] = local_sd >= local_sd_threshold
    consensus["failure_signature"] = consensus.apply(_failure_signature, axis=1)
    consensus["any_persistent_failure"] = consensus["failure_signature"].ne("none")

    seed_window_flags = pd.concat(seed_window_frames, ignore_index=True)
    failure_windows = consensus.loc[consensus["any_persistent_failure"]].copy()
    failure_windows = failure_windows.sort_values(
        ["persistent_core_failure_count", "local_rr_mae_bpm_seed_mean", "dataset_row_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    strata = _strata_summary(consensus)
    seed_strata = _seed_strata_summary(seed_window_flags, reference)
    signatures = _signature_summary(consensus)
    associations = _metric_associations(consensus)
    agreements = _seed_agreement(sorted_frames, hard_by_metric)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds = pd.concat(
        [
            thresholds,
            pd.DataFrame(
                [
                    {
                        "seed": "cross_seed",
                        "metric": "local_rr_mae_bpm_seed_sd",
                        "direction": "variability",
                        "tail_quantile": 0.90,
                        "threshold": local_sd_threshold,
                        "eligible_n": int(np.isfinite(local_sd).sum()),
                        "worst_decile_n_including_ties": int((local_sd >= local_sd_threshold).sum()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return {
        "thresholds": thresholds,
        "seed_window_flags": seed_window_flags,
        "window_consensus": consensus,
        "failure_windows": failure_windows,
        "strata_summary": strata,
        "seed_strata_summary": seed_strata,
        "signature_summary": signatures,
        "metric_associations": associations,
        "seed_agreement": agreements,
    }


def _sort_and_validate_frame(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        *STATIC_COLUMNS,
        *DIAGNOSTIC_METRICS,
        "best_lag_sec",
        "joint_prediction_degenerate",
        "envelope_spearman_prediction_degenerate",
        "ibi_interpretable",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"seed={seed} metrics 缺少字段: {missing}")
    if len(frame) != 2675 or frame["dataset_row_id"].duplicated().any():
        raise ValueError(f"seed={seed} validation identity 数量或唯一性异常")
    if set(frame["evaluation_split"].astype(str)) != {"validation"}:
        raise ValueError(f"seed={seed} diagnostic 只允许 validation")
    return frame.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)


def _assert_cross_seed_identity(reference: pd.DataFrame, candidate: pd.DataFrame, *, seed: int) -> None:
    if not reference[list(IDENTITY_COLUMNS)].equals(candidate[list(IDENTITY_COLUMNS)]):
        raise RuntimeError(f"seed={seed} validation identity 未与 reference 逐行配对")
    for column in STATIC_COLUMNS:
        left = reference[column]
        right = candidate[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = np.array_equal(
                pd.to_numeric(left, errors="coerce").to_numpy(),
                pd.to_numeric(right, errors="coerce").to_numpy(),
                equal_nan=True,
            )
        else:
            equal = left.astype(str).equals(right.astype(str))
        if not equal:
            raise RuntimeError(f"seed={seed} target/static 字段跨 seed 不一致: {column}")


def _metric_eligibility(frame: pd.DataFrame, metric: str) -> np.ndarray:
    if metric == "whole_rr_abs_error_bpm":
        return _as_bool(frame["whole_rr_target_eligible"])
    if metric == "local_rr_mae_bpm":
        return _as_bool(frame["local_rr_target_eligible"])
    if metric in {"envelope_trajectory_mae", "global_envelope_modulation_error"}:
        return np.ones(len(frame), dtype=bool)
    if metric == "lag_aware_signed_pcc":
        return _as_bool(frame["joint_target_eligible"]) & ~_as_bool(frame["joint_prediction_degenerate"])
    if metric == "ibi_medae_sec":
        return _as_bool(frame["ibi_target_eligible"]) & _as_bool(frame["ibi_interpretable"])
    if metric == "ibi_coverage":
        return _as_bool(frame["ibi_target_eligible"])
    if metric == "target_stratified_envelope_spearman":
        return _as_bool(frame["envelope_spearman_target_eligible"]) & ~_as_bool(
            frame["envelope_spearman_prediction_degenerate"]
        )
    raise KeyError(metric)


def _failure_signature(row: pd.Series) -> str:
    signatures: list[str] = []
    if row["whole_rr_abs_error_bpm_persistent_failure"] or row["local_rr_mae_bpm_persistent_failure"]:
        signatures.append("rate")
    if row["envelope_trajectory_mae_persistent_failure"] or row[
        "global_envelope_modulation_error_persistent_failure"
    ]:
        signatures.append("envelope")
    if row["lag_aware_signed_pcc_persistent_failure"] or row["lag_boundary_seed_count"] >= 2:
        signatures.append("alignment")
    if row["target_stratified_envelope_spearman_persistent_failure"]:
        signatures.append("rank")
    if (
        row["ibi_medae_sec_persistent_failure"]
        or row["ibi_coverage_persistent_failure"]
        or row["ibi_interpretable_seed_count"] <= 1
    ):
        signatures.append("beat")
    return "+".join(signatures) if signatures else "none"


def _strata_summary(consensus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(consensus)
    for axis in STRATA_COLUMNS:
        for value, group in consensus.groupby(axis, sort=True, dropna=False):
            row: dict[str, Any] = {
                "stratum_axis": axis,
                "stratum_value": str(value),
                "window_n": int(len(group)),
                "window_fraction": float(len(group) / total),
                "samp_id_n": int(group["samp_id"].nunique()),
                "target_envelope_modulation_mean": float(group["target_envelope_modulation"].mean()),
                "persistent_core_failure_count_mean": float(group["persistent_core_failure_count"].mean()),
                "any_persistent_failure_fraction": float(group["any_persistent_failure"].mean()),
                "local_rr_seed_sd_top_decile_fraction": float(group["local_rr_seed_sd_top_decile"].mean()),
                "ibi_interpretable_seed_fraction": float(group["ibi_interpretable_seed_count"].mean() / 3.0),
                "lag_boundary_seed_fraction": float(group["lag_boundary_seed_count"].mean() / 3.0),
            }
            for metric in DIAGNOSTIC_METRICS:
                values = pd.to_numeric(group[f"{metric}_seed_mean"], errors="coerce")
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_median"] = float(values.median())
                row[f"{metric}_persistent_failure_fraction"] = float(
                    group[f"{metric}_persistent_failure"].mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _seed_strata_summary(seed_window: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    target = reference[list(IDENTITY_COLUMNS) + ["envelope_target_stratum"]].copy()
    merged = seed_window.merge(target, on=list(IDENTITY_COLUMNS), how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    axes = ("samp_id", "coupling_state_id", "envelope_target_stratum", "ibi_interpretable", "lag_boundary")
    for seed, seed_group in merged.groupby("seed", sort=True):
        for axis in axes:
            for value, group in seed_group.groupby(axis, sort=True, dropna=False):
                row: dict[str, Any] = {
                    "seed": int(seed),
                    "stratum_axis": axis,
                    "stratum_value": str(value),
                    "window_n": int(len(group)),
                    "window_fraction": float(len(group) / len(seed_group)),
                    "samp_id_n": int(group["samp_id"].nunique()),
                }
                for metric in DIAGNOSTIC_METRICS:
                    values = pd.to_numeric(group[metric], errors="coerce")
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_median"] = float(values.median())
                    row[f"{metric}_worst_decile_fraction"] = float(group[f"{metric}_worst_decile"].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def _signature_summary(consensus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signature, group in consensus.groupby("failure_signature", sort=True):
        row: dict[str, Any] = {
            "failure_signature": signature,
            "window_n": int(len(group)),
            "window_fraction": float(len(group) / len(consensus)),
            "samp_id_n": int(group["samp_id"].nunique()),
            "persistent_core_failure_count_mean": float(group["persistent_core_failure_count"].mean()),
            "target_envelope_modulation_mean": float(group["target_envelope_modulation"].mean()),
        }
        for metric in DIAGNOSTIC_METRICS:
            row[f"{metric}_mean"] = float(group[f"{metric}_seed_mean"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["window_n", "failure_signature"], ascending=[False, True])


def _metric_associations(consensus: pd.DataFrame) -> pd.DataFrame:
    aligned = pd.DataFrame(
        {
            "target_envelope_modulation": consensus["target_envelope_modulation"],
            "whole_rr_error": consensus["whole_rr_abs_error_bpm_seed_mean"],
            "local_rr_error": consensus["local_rr_mae_bpm_seed_mean"],
            "trajectory_error": consensus["envelope_trajectory_mae_seed_mean"],
            "global_modulation_error": consensus["global_envelope_modulation_error_seed_mean"],
            "pcc_error": -consensus["lag_aware_signed_pcc_seed_mean"],
            "ibi_medae_error": consensus["ibi_medae_sec_seed_mean"],
            "ibi_coverage_error": -consensus["ibi_coverage_seed_mean"],
            "envelope_rank_error": -consensus["target_stratified_envelope_spearman_seed_mean"],
            "local_rr_seed_sd": consensus["local_rr_mae_bpm_seed_sd"],
        }
    )
    rows: list[dict[str, Any]] = []
    for left, right in combinations(aligned.columns, 2):
        pair = aligned[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        rho = float(pair[left].corr(pair[right], method="spearman")) if len(pair) >= 2 else math.nan
        rows.append({"left": left, "right": right, "n": int(len(pair)), "spearman_rho": rho})
    return pd.DataFrame(rows)


def _seed_agreement(
    frames: dict[int, pd.DataFrame],
    hard_by_metric: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_to_position = {seed: index for index, seed in enumerate(FORMAL_SEEDS)}
    for metric in DIAGNOSTIC_METRICS:
        for seed_a, seed_b in combinations(FORMAL_SEEDS, 2):
            left = pd.to_numeric(frames[seed_a][metric], errors="coerce").to_numpy(dtype=np.float64)
            right = pd.to_numeric(frames[seed_b][metric], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(left) & np.isfinite(right)
            pair = pd.DataFrame({"left": left[finite], "right": right[finite]})
            rho = float(pair["left"].corr(pair["right"], method="spearman")) if len(pair) >= 2 else math.nan
            hard_a = hard_by_metric[metric][seed_to_position[seed_a]]
            hard_b = hard_by_metric[metric][seed_to_position[seed_b]]
            union = hard_a | hard_b
            jaccard = float((hard_a & hard_b).sum() / union.sum()) if union.any() else math.nan
            rows.append(
                {
                    "metric": metric,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "paired_n": int(finite.sum()),
                    "spearman_rho": rho,
                    "mean_absolute_difference": float(np.mean(np.abs(left[finite] - right[finite]))),
                    "worst_decile_jaccard": jaccard,
                }
            )
    return pd.DataFrame(rows)


def _build_overview(consensus: pd.DataFrame, strata: pd.DataFrame) -> dict[str, Any]:
    samp = strata.loc[strata["stratum_axis"].eq("samp_id")].sort_values(
        "local_rr_mae_bpm_mean", ascending=False
    )
    coupling = strata.loc[strata["stratum_axis"].eq("coupling_state_id")].sort_values(
        "local_rr_mae_bpm_mean", ascending=False
    )
    return {
        "validation_window_count": int(len(consensus)),
        "samp_id_count": int(consensus["samp_id"].nunique()),
        "coupling_state_count": int(consensus["coupling_state_id"].nunique()),
        "any_persistent_failure_count": int(consensus["any_persistent_failure"].sum()),
        "any_persistent_failure_fraction": float(consensus["any_persistent_failure"].mean()),
        "persistent_local_rr_failure_count": int(consensus["local_rr_mae_bpm_persistent_failure"].sum()),
        "persistent_local_rr_failure_fraction": float(consensus["local_rr_mae_bpm_persistent_failure"].mean()),
        "multimetric_core_failure_count": int(consensus["persistent_core_failure_count"].ge(2).sum()),
        "multimetric_core_failure_fraction": float(consensus["persistent_core_failure_count"].ge(2).mean()),
        "ibi_interpretable_all_seed_fraction": float(consensus["ibi_interpretable_seed_count"].eq(3).mean()),
        "persistent_lag_boundary_fraction": float(consensus["lag_boundary_seed_count"].ge(2).mean()),
        "worst_samp_id_by_local_rr": str(samp.iloc[0]["stratum_value"]),
        "worst_samp_id_local_rr_mae_bpm": float(samp.iloc[0]["local_rr_mae_bpm_mean"]),
        "worst_coupling_state_by_local_rr": str(coupling.iloc[0]["stratum_value"]),
        "worst_coupling_state_local_rr_mae_bpm": float(coupling.iloc[0]["local_rr_mae_bpm_mean"]),
    }


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.full(values.shape[1], np.nan), where=count > 0)


def _row_nanstd(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    mean = _row_nanmean(values)
    squared = np.where(finite, (values - mean) ** 2, 0.0).sum(axis=0)
    return np.sqrt(np.divide(squared, count - 1, out=np.full(values.shape[1], np.nan), where=count > 1))


def _as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(f"布尔字段包含未知值: {unexpected}")
    return normalized.eq("true").to_numpy(dtype=bool)


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"无法读取 git 状态: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("CRD_102 failure diagnostic 必须从干净 git commit 生成")
