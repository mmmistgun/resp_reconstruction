"""E9：完整18-cell validation汇总与预注册对比。"""

from __future__ import annotations

import fcntl
import json
import sys
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1 as e9
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_engineering as engineering
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_formal as formal
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import (
    ARMS,
    ARM_SPECS,
)


SUMMARY_PATH = Path(
    "resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_summary.py"
)
SCRIPT_PATH = Path("scripts/summarize_e9_latent_width_condition_refiner_v1.py")
TEST_PATH = Path("tests/test_e9_latent_width_condition_refiner_v1_summary.py")
E8_RESULTS_PATH = Path(
    "docs/experiments/e8_film_decoder_redesign_v1_validation_results_20260926.md"
)
ERRORS = tuple(e9.PRIMARY_METRICS[:4])
PCC = e9.PRIMARY_METRICS[4]
PRIMARY = (*ERRORS, PCC)
ERROR_RELATIVE_TOLERANCE = 0.005
PCC_ABSOLUTE_TOLERANCE = 0.002
REFERENCE_ARM = "e9a_d96_h65"

CONTRASTS = (
    ("e9a_planned", "h64_minus_h65", "e9a_d96_h64", "e9a_d96_h65"),
    ("e9a_planned", "h48_minus_h65", "e9a_d96_h48", "e9a_d96_h65"),
    ("e9a_planned", "h48_minus_h64", "e9a_d96_h48", "e9a_d96_h64"),
    ("e9b_planned", "d64_h64_minus_direct", "e9b_d64_h64", "e9b_d64_direct"),
    ("e9b_planned", "d64_h48_minus_direct", "e9b_d64_h48", "e9b_d64_direct"),
    ("e9b_planned", "d64_h48_minus_h64", "e9b_d64_h48", "e9b_d64_h64"),
    ("cross_width", "d64_direct_minus_d96_h65", "e9b_d64_direct", "e9a_d96_h65"),
    ("cross_width", "d64_direct_minus_d96_h64", "e9b_d64_direct", "e9a_d96_h64"),
    ("cross_width", "d64_h64_minus_d96_h65", "e9b_d64_h64", "e9a_d96_h65"),
    ("cross_width", "d64_h64_minus_d96_h64", "e9b_d64_h64", "e9a_d96_h64"),
    ("cross_width", "d64_h48_minus_d96_h65", "e9b_d64_h48", "e9a_d96_h65"),
    ("cross_width", "d64_h48_minus_d96_h64", "e9b_d64_h48", "e9a_d96_h64"),
)

SUMMARY_REQUIRED_FILES = {
    "seed_primary_metrics.csv",
    "arm_primary_summary.csv",
    "arm_reference_comparison.csv",
    "planned_contrasts_by_seed.csv",
    "planned_contrasts_across_seed.csv",
    "local_rr_tail_summary.csv",
    "local_rr_tail_across_seed.csv",
    "subject_stratified_metrics.csv",
    "subject_macro_by_seed.csv",
    "subject_macro_across_seed.csv",
    "metric_denominators.csv",
    "parameter_compute_memory.csv",
    "historical_e8_pointwise_reference.csv",
    "decision.json",
    "source_manifest.json",
    "summary_receipt.json",
    "access_receipt.json",
    "source_code.json",
}


def completed_formal_attempts() -> tuple[list[Path], dict[str, Any]]:
    status = formal.matrix_status()
    expected_counts = {"pending": 0, "running": 0, "failed": 0, "completed": 18}
    if status["counts"] != expected_counts:
        raise RuntimeError(f"E9 summary要求完整18-cell，当前={status['counts']}")
    attempts: list[Path] = []
    for cell in status["cells"]:
        if len(cell["completed"]) != 1 or cell["failed"] or cell["running"]:
            raise RuntimeError(
                f"E9 summary cell状态非法: {cell['arm']}/{cell['seed']}"
            )
        attempts.append(Path(cell["completed"][0]).resolve())
    if len(attempts) != 18 or len(set(attempts)) != 18:
        raise RuntimeError("E9 summary formal attempt集合重复或不完整")
    return attempts, {
        "matrix_status": status["counts"],
        "formal_runtime_amendment_sha256": status[
            "formal_runtime_amendment_sha256"
        ],
    }


def _metric_mask(frame: pd.DataFrame, metric: str) -> np.ndarray:
    flag = sf.ELIGIBILITY.get(metric)
    if flag is None:
        return np.ones(len(frame), dtype=bool)
    values = frame[flag]
    if values.isna().any() or not values.isin([True, False]).all():
        raise ValueError(f"E9 eligibility非法: {flag}")
    return values.to_numpy(dtype=bool)


def _require_complete_seed_frame(frame: pd.DataFrame) -> None:
    required = {"arm", "seed", *PRIMARY}
    if not required.issubset(frame.columns):
        raise ValueError("E9 seed frame字段不完整")
    expected = {(arm, seed) for arm in ARMS for seed in e9.SEEDS}
    observed = set(frame[["arm", "seed"]].itertuples(index=False, name=None))
    if observed != expected or len(frame) != len(expected):
        raise ValueError("E9 seed frame矩阵不完整")
    if not np.isfinite(frame[list(PRIMARY)].to_numpy(float)).all():
        raise FloatingPointError("E9 seed frame主指标非有限")


def seed_primary_metrics(
    window_metrics: pd.DataFrame,
    *,
    expected_rows: int = formal.COUNTS["val"],
) -> pd.DataFrame:
    expected_pairs = {(arm, seed) for arm in ARMS for seed in e9.SEEDS}
    observed_pairs = set(
        window_metrics[["arm", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_pairs != expected_pairs:
        raise ValueError("E9 seed-primary arm/seed矩阵不完整")
    rows: list[dict[str, Any]] = []
    for (arm, seed), group in window_metrics.groupby(["arm", "seed"], sort=False):
        if len(group) != expected_rows or group.dataset_row_id.duplicated().any():
            raise ValueError(f"E9 validation rows不完整: {arm}/{seed}")
        row: dict[str, Any] = {"arm": arm, "seed": int(seed)}
        for metric in PRIMARY:
            mask = _metric_mask(group, metric)
            values = pd.to_numeric(group.loc[mask, metric], errors="coerce").to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError(
                    f"E9 primary metric非有限: {arm}/{seed}/{metric}"
                )
            row[metric] = float(values.mean())
            row[metric + "_n"] = len(values)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    if len(result) != 18:
        raise ValueError("E9 seed-primary行数错误")
    return result


def arm_primary_summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        spec = ARM_SPECS[arm]
        group = seed_frame.loc[seed_frame.arm.eq(arm)].set_index("seed").loc[list(e9.SEEDS)]
        for metric in PRIMARY:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "arm": arm,
                    "family": spec.family,
                    "latent_channels": spec.latent_channels,
                    "condition_refiner": spec.condition_refiner,
                    "hidden_channels": spec.hidden_channels,
                    "metric": metric,
                    "seed_count": len(values),
                    "seed_mean": float(values.mean()),
                    "seed_sample_sd": float(values.std(ddof=1)),
                    "seed_min": float(values.min()),
                    "seed_max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _benefit(reference: float, candidate: float, metric: str) -> tuple[float, float]:
    if metric == PCC:
        oriented = candidate - reference
        return oriented, oriented
    if reference == 0:
        raise ZeroDivisionError(f"E9 reference error为零: {metric}")
    oriented = reference - candidate
    return oriented, oriented / abs(reference)


def arm_reference_comparison(seed_frame: pd.DataFrame) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame)
    indexed = seed_frame.set_index(["arm", "seed"])
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        for metric in PRIMARY:
            raw_values: list[float] = []
            normalized: list[float] = []
            for seed in e9.SEEDS:
                raw, norm = _benefit(
                    float(indexed.loc[(REFERENCE_ARM, seed), metric]),
                    float(indexed.loc[(arm, seed), metric]),
                    metric,
                )
                raw_values.append(raw)
                normalized.append(norm)
            threshold = (
                PCC_ABSOLUTE_TOLERANCE if metric == PCC else ERROR_RELATIVE_TOLERANCE
            )
            values = np.asarray(normalized)
            rows.append(
                {
                    "arm": arm,
                    "reference_arm": REFERENCE_ARM,
                    "metric": metric,
                    "benefit_unit": "absolute" if metric == PCC else "relative_fraction",
                    "material_threshold": threshold,
                    "oriented_benefit_seed_mean": float(np.mean(raw_values)),
                    "reference_normalized_benefit_seed_mean": float(values.mean()),
                    "reference_normalized_benefit_seed_sample_sd": float(
                        values.std(ddof=1)
                    ),
                    "benefit_positive_seeds": int((values > 0).sum()),
                    "materially_improved_seeds": int((values > threshold).sum()),
                    "materially_worsened_seeds": int((values < -threshold).sum()),
                    "seed_values": "|".join(f"{value:.17g}" for value in values),
                }
            )
    return pd.DataFrame(rows)


def planned_contrasts_by_seed(seed_frame: pd.DataFrame) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame)
    indexed = seed_frame.set_index(["arm", "seed"])
    rows: list[dict[str, Any]] = []
    for contrast_type, contrast, candidate_arm, reference_arm in CONTRASTS:
        for seed in e9.SEEDS:
            for metric in PRIMARY:
                reference = float(indexed.loc[(reference_arm, seed), metric])
                candidate = float(indexed.loc[(candidate_arm, seed), metric])
                raw, normalized = _benefit(reference, candidate, metric)
                rows.append(
                    {
                        "contrast_type": contrast_type,
                        "contrast": contrast,
                        "candidate_arm": candidate_arm,
                        "reference_arm": reference_arm,
                        "seed": seed,
                        "metric": metric,
                        "reference_value": reference,
                        "candidate_value": candidate,
                        "oriented_benefit": raw,
                        "reference_normalized_benefit": normalized,
                    }
                )
    return pd.DataFrame(rows)


def planned_contrasts_across_seed(by_seed: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "contrast_type",
        "contrast",
        "candidate_arm",
        "reference_arm",
        "metric",
    ]
    rows: list[dict[str, Any]] = []
    for group_key, group in by_seed.groupby(keys, sort=False):
        group = group.set_index("seed").loc[list(e9.SEEDS)]
        benefit = group.oriented_benefit.to_numpy(float)
        normalized = group.reference_normalized_benefit.to_numpy(float)
        metric = str(group_key[-1])
        threshold = (
            PCC_ABSOLUTE_TOLERANCE if metric == PCC else ERROR_RELATIVE_TOLERANCE
        )
        rows.append(
            {
                **dict(zip(keys, group_key)),
                "seed_count": len(group),
                "oriented_benefit_mean": float(benefit.mean()),
                "oriented_benefit_sample_sd": float(benefit.std(ddof=1)),
                "reference_normalized_benefit_mean": float(normalized.mean()),
                "reference_normalized_benefit_sample_sd": float(
                    normalized.std(ddof=1)
                ),
                "benefit_positive_seeds": int((normalized > 0).sum()),
                "materially_improved_seeds": int((normalized > threshold).sum()),
                "materially_worsened_seeds": int((normalized < -threshold).sum()),
                "material_threshold": threshold,
                "material_direction": (
                    "improved"
                    if normalized.mean() > threshold
                    else "worsened"
                    if normalized.mean() < -threshold
                    else "within_tolerance"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(CONTRASTS) * len(PRIMARY):
        raise ValueError("E9 planned contrast行数错误")
    return result


def tolerance_pareto(seed_frame: pd.DataFrame) -> dict[str, Any]:
    means = seed_frame.groupby("arm", sort=False)[list(PRIMARY)].mean()
    dominated_by: dict[str, list[str]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        current = means.loc[arm]
        for other in ARMS:
            if other == arm:
                continue
            candidate = means.loc[other]
            no_worse = True
            better = False
            for metric in PRIMARY:
                if metric == PCC:
                    no_worse &= bool(
                        candidate[metric] >= current[metric] - PCC_ABSOLUTE_TOLERANCE
                    )
                    better |= bool(
                        candidate[metric] > current[metric] + PCC_ABSOLUTE_TOLERANCE
                    )
                else:
                    no_worse &= bool(
                        candidate[metric]
                        <= current[metric] * (1.0 + ERROR_RELATIVE_TOLERANCE)
                    )
                    better |= bool(
                        candidate[metric]
                        < current[metric] * (1.0 - ERROR_RELATIVE_TOLERANCE)
                    )
            if no_worse and better:
                dominated_by[arm].append(other)
    return {
        "pareto_arms": [arm for arm in ARMS if not dominated_by[arm]],
        "dominated_by": dominated_by,
    }


def subject_macro_tables(
    subject_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_metrics = subject_metrics.loc[subject_metrics.metric.isin(PRIMARY)].copy()
    rows: list[dict[str, Any]] = []
    for (arm, seed, metric), group in subject_metrics.groupby(
        ["arm", "seed", "metric"], sort=False
    ):
        if (
            len(group) != formal.SAMP_IDS["val"]
            or group.samp_id.nunique() != formal.SAMP_IDS["val"]
        ):
            raise ValueError(f"E9 subject macro不完整: {arm}/{seed}/{metric}")
        values = group["mean"].to_numpy(float)
        if not np.isfinite(values).all():
            raise FloatingPointError("E9 subject macro非有限")
        rows.append(
            {
                "arm": arm,
                "seed": int(seed),
                "metric": metric,
                "subject_count": len(values),
                "subject_macro_mean": float(values.mean()),
                "subject_sample_sd": float(values.std(ddof=1)),
                "subject_min": float(values.min()),
                "subject_max": float(values.max()),
            }
        )
    by_seed = pd.DataFrame(rows)
    across: list[dict[str, Any]] = []
    for (arm, metric), group in by_seed.groupby(["arm", "metric"], sort=False):
        values = (
            group.set_index("seed")
            .loc[list(e9.SEEDS)]
            .subject_macro_mean.to_numpy(float)
        )
        across.append(
            {
                "arm": arm,
                "metric": metric,
                "seed_count": len(values),
                "subject_macro_seed_mean": float(values.mean()),
                "subject_macro_seed_sample_sd": float(values.std(ddof=1)),
            }
        )
    return by_seed, pd.DataFrame(across)


def local_rr_tail_across_seed(tails: pd.DataFrame) -> pd.DataFrame:
    value_columns = (
        "eligible_n",
        "mean",
        "median",
        "p90",
        "p95",
        "max",
        "gt_2_bpm_fraction",
        "gt_5_bpm_fraction",
    )
    rows: list[dict[str, Any]] = []
    for arm, group in tails.groupby("arm", sort=False):
        group = group.set_index("seed").loc[list(e9.SEEDS)]
        row: dict[str, Any] = {"arm": arm, "seed_count": len(e9.SEEDS)}
        for column in value_columns:
            values = group[column].to_numpy(float)
            row[column + "_seed_mean"] = float(values.mean())
            row[column + "_seed_sample_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_compute_memory_table(
    receipts: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        arm = str(receipt["arm"])
        spec = ARM_SPECS[arm]
        runtime = receipt["runtime"]
        rows.append(
            {
                "arm": arm,
                "seed": int(receipt["seed"]),
                "family": spec.family,
                "latent_channels": spec.latent_channels,
                "condition_refiner": spec.condition_refiner,
                "hidden_channels": spec.hidden_channels,
                "trainable_parameters": spec.trainable_parameters,
                "condition_refiner_macs": spec.condition_refiner_macs,
                "declared_covered_macs": spec.declared_covered_macs,
                "completed_epochs": int(receipt["completed_epochs"]),
                "completed_updates": int(receipt["completed_updates"]),
                "selected_epoch": int(receipt["selected_epoch"]),
                "formal_wall_seconds": float(receipt["formal_wall_seconds"]),
                "formal_peak_allocated_mib": float(runtime["peak_allocated_mib"]),
                "formal_peak_reserved_mib": float(runtime["peak_reserved_mib"]),
                "formal_peak_reserved_fraction": float(
                    runtime["peak_reserved_fraction"]
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    if len(frame) != 18 or frame[["arm", "seed"]].duplicated().any():
        raise ValueError("E9 resource cell不完整")
    return frame


def historical_e8_pointwise_reference() -> pd.DataFrame:
    rows = (
        ("e8_direct_pointwise", 1_207_274, 1_900_800, 0.492749, 0.545727, 0.152145, 0.190626, 0.861968),
        ("e8_fill65_pointwise", 1_219_850, 24_364_800, 0.500708, 0.553073, 0.152950, 0.191168, 0.865227),
        ("e8_res96_pointwise", 1_225_802, 35_078_400, 0.498948, 0.545092, 0.154157, 0.185980, 0.863997),
        ("e8_res192_pointwise", 1_244_234, 68_256_000, 0.490762, 0.544298, 0.154646, 0.193695, 0.865859),
    )
    return pd.DataFrame(
        rows,
        columns=(
            "historical_arm",
            "trainable_parameters",
            "e8_factor_covered_macs",
            *PRIMARY,
        ),
    )


def _audit_formal_inputs(
    attempts: Sequence[Path],
    lock: Mapping[str, Any],
    lock_hash: str,
    source_lock: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    observed: set[tuple[str, int]] = set()
    source_commits: dict[str, int] = {}
    for attempt in attempts:
        receipt = json.loads((attempt / "formal_receipt.json").read_text(encoding="utf-8"))
        arm, seed = str(receipt["arm"]), int(receipt["seed"])
        if arm not in ARMS or seed not in e9.SEEDS or (arm, seed) in observed:
            raise ValueError("E9 summary formal cell重复或越界")
        observed.add((arm, seed))
        manifest = formal.verify_formal_attempt(
            attempt,
            lock_hash=lock_hash,
            amendment_hash=amendment_hash,
            arm=arm,
            seed=seed,
        )
        run_dir = (attempt / receipt["run_dir"]).resolve()
        cfg = OmegaConf.load(run_dir / "config.yaml")
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        e9.validate_config(
            cfg,
            baseline,
            arm=arm,
            output_root=attempt / "training",
            device=str(cfg.training.device),
        )
        validation_rows = pd.read_csv(attempt / "val_rows.csv")
        checked = formal.validate_formal_run(
            run_dir,
            cfg,
            validation_rows,
            source_lock,
            arm=arm,
            initialization_path=attempt / "initialization.json",
        )
        for key, value in checked.items():
            if receipt.get(key) != value:
                raise ValueError(f"E9 summary formal receipt回放不一致: {arm}/{seed}/{key}")
        metrics = pd.read_csv(run_dir / "metrics.csv")
        frames.append(metrics)
        receipts.append(receipt)
        environment = json.loads((attempt / "environment.json").read_text(encoding="utf-8"))
        commit = str(environment["git"]["commit"])
        if environment["git"]["status_porcelain"]:
            raise ValueError("E9 formal source Git非clean")
        source_commits[commit] = source_commits.get(commit, 0) + 1
        key = f"{arm}/{seed}"
        sources[key] = {
            "arm": arm,
            "seed": seed,
            "attempt": str(attempt),
            "manifest": engineering._identity(attempt / "manifest.json"),
            "formal_receipt": engineering._identity(attempt / "formal_receipt.json"),
            "metrics": engineering._identity(run_dir / "metrics.csv"),
            "metrics_summary": engineering._identity(run_dir / "metrics_summary.csv"),
            "best_checkpoint": engineering._identity(
                run_dir / "checkpoint_best_local_rr.pt"
            ),
            "final_checkpoint": engineering._identity(run_dir / "checkpoint_final.pt"),
            "selected_epoch": int(receipt["selected_epoch"]),
            "completed_epochs": int(receipt["completed_epochs"]),
            "source_commit": commit,
            "formal_runtime_amendment_sha256": receipt[
                "formal_runtime_amendment_sha256"
            ],
            "manifest_file_count": len(manifest["files"]),
        }
    expected = {(arm, seed) for arm in ARMS for seed in e9.SEEDS}
    if observed != expected:
        raise ValueError("E9 summary formal 18-cell矩阵不完整")
    metrics = pd.concat(frames, ignore_index=True)
    if len(metrics) != len(expected) * formal.COUNTS["val"]:
        raise ValueError("E9 summary validation metrics总行数错误")
    return metrics, receipts, {
        "formal_source_commits": source_commits,
        "formal_runs": sources,
    }


def _classification(directions: Mapping[str, str]) -> str:
    values = set(directions.values())
    if "improved" in values and "worsened" not in values:
        return "tolerance_aware_pareto_improvement"
    if "improved" in values and "worsened" in values:
        return "attribute_tradeoff"
    if "worsened" in values:
        return "material_degradation"
    return "within_tolerance"


def build_decision(
    seed_frame: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> dict[str, Any]:
    contrast_decisions: dict[str, Any] = {}
    for contrast, group in contrasts.groupby("contrast", sort=False):
        directions = {row.metric: row.material_direction for row in group.itertuples()}
        first = group.iloc[0]
        contrast_decisions[str(contrast)] = {
            "contrast_type": str(first.contrast_type),
            "candidate_arm": str(first.candidate_arm),
            "reference_arm": str(first.reference_arm),
            "metric_directions": directions,
            "classification": _classification(directions),
            "paired_seed_benefit_positive": {
                row.metric: int(row.benefit_positive_seeds) for row in group.itertuples()
            },
        }

    h64_h65 = contrast_decisions["h64_minus_h65"]
    h48_h64 = contrast_decisions["h48_minus_h64"]
    h64_direct = contrast_decisions["d64_h64_minus_direct"]
    h48_direct = contrast_decisions["d64_h48_minus_direct"]
    h48_h64_d64 = contrast_decisions["d64_h48_minus_h64"]
    if h64_h65["classification"] in {
        "within_tolerance",
        "tolerance_aware_pareto_improvement",
    }:
        e9a_preference = "h64"
    else:
        e9a_preference = "h65"
    if h48_h64["classification"] == "tolerance_aware_pareto_improvement":
        e9a_h48_status = "retain_as_candidate"
    elif "worsened" in h48_h64["metric_directions"].values():
        e9a_h48_status = "not_preferred_over_h64"
    else:
        e9a_h48_status = "efficiency_candidate_with_quality_protection"

    direct_has_no_material_degradation = not any(
        "improved" in item["metric_directions"].values()
        for item in (h64_direct, h48_direct)
    )
    if direct_has_no_material_degradation:
        e9b_preference = "direct"
    elif h64_direct["classification"] == "tolerance_aware_pareto_improvement":
        e9b_preference = "h64"
    elif (
        h48_direct["classification"] == "tolerance_aware_pareto_improvement"
        and h48_h64_d64["classification"]
        in {"within_tolerance", "tolerance_aware_pareto_improvement"}
    ):
        e9b_preference = "h48"
    else:
        e9b_preference = "attribute_tradeoff"

    cross_width_protection: dict[str, bool] = {}
    for name, item in contrast_decisions.items():
        if item["contrast_type"] == "cross_width":
            cross_width_protection[name] = (
                "worsened" not in item["metric_directions"].values()
            )
    means = seed_frame.groupby("arm")[list(PRIMARY)].mean()
    best_by_metric = {
        metric: (
            str(means[metric].idxmax()) if metric == PCC else str(means[metric].idxmin())
        )
        for metric in PRIMARY
    }
    return {
        "protocol": e9.PROTOCOL,
        "evidence_scope": "three_seed_validation_descriptive_width_refiner_study",
        "contrast_decisions": contrast_decisions,
        "e9a": {
            "preferred_anchor": e9a_preference,
            "h48_status": e9a_h48_status,
        },
        "e9b": {
            "preferred_structure": e9b_preference,
            "direct_has_no_material_degradation": direct_has_no_material_degradation,
        },
        "cross_width_quality_protection": cross_width_protection,
        "tolerance_aware_pareto": tolerance_pareto(seed_frame),
        "best_arm_by_univariate_seed_mean": best_by_metric,
        "materiality": {
            "error_relative": ERROR_RELATIVE_TOLERANCE,
            "pcc_absolute": PCC_ABSOLUTE_TOLERANCE,
        },
        "statistical_unit": "seed describes optimization randomness; no population inference",
        "window_dependence": "overlapping windows are not independent samples",
        "research_test_accessed": False,
    }


def _summary_source_identity() -> dict[str, Any]:
    paths = (
        SUMMARY_PATH,
        SCRIPT_PATH,
        TEST_PATH,
        Path("resp_train/paper_evidence/e9_latent_width_condition_refiner_v1.py"),
        Path("resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_model.py"),
        formal.FORMAL_PATH,
        formal.FORMAL_AMENDMENT_PATH,
        E8_RESULTS_PATH,
    )
    return {
        "git": engineering._git_state(),
        "files": {str(path): engineering._identity(e9.ROOT / path) for path in paths},
    }


@contextmanager
def _summary_attempt(
    parent: Path, *, lock_hash: str, amendment_hash: str
) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".execution_{lock_hash}_{amendment_hash}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("相同E9 summary正在运行") from exc
        for receipt_path in parent.glob("*/freeze_receipt.json"):
            manifest_path = receipt_path.parent / "manifest.json"
            freeze = json.loads(receipt_path.read_text(encoding="utf-8"))
            if engineering._identity(manifest_path) != freeze.get("manifest"):
                raise RuntimeError("E9已完成summary身份错误")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("implementation_lock_sha256") == lock_hash
                and manifest.get("formal_runtime_amendment_sha256") == amendment_hash
                and manifest.get("status") == "completed"
            ):
                raise FileExistsError(f"相同E9 summary已完成: {receipt_path.parent}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = parent / f"summary_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
        output.mkdir(exist_ok=False)
        context = {
            "protocol": e9.PROTOCOL,
            "phase": "validation_summary",
            "implementation_lock_sha256": lock_hash,
            "formal_runtime_amendment_sha256": amendment_hash,
            "command": sys.argv,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        engineering._write_json(
            output / "lifecycle_started.json", {**context, "status": "running"}
        )
        try:
            yield output
            engineering._write_json(
                output / "lifecycle_completed.json",
                {
                    **context,
                    "status": "completed",
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            manifest = {
                **context,
                "status": "completed",
                "files": {
                    str(file.relative_to(output)): engineering._identity(file)
                    for file in sorted(output.rglob("*"))
                    if file.is_file()
                },
            }
            engineering._write_json(output / "manifest.json", manifest)
            engineering._write_json(
                output / "freeze_receipt.json",
                {"protocol": e9.PROTOCOL, "manifest": engineering._identity(output / "manifest.json")},
            )
        except BaseException as exc:
            engineering._write_json(
                output / "lifecycle_failed.json",
                {
                    **context,
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_summary() -> Path:
    lock, lock_hash, source_lock = formal.load_formal_lock()
    state = engineering._git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E9 validation summary要求干净Git工作树")
    attempts, lifecycle_summary = completed_formal_attempts()
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    parent = e9.SOURCE_ROOT / e9.SUMMARY_ROOT
    with _summary_attempt(parent, lock_hash=lock_hash, amendment_hash=amendment_hash) as output:
        metrics, receipts, sources = _audit_formal_inputs(
            attempts, lock, lock_hash, source_lock
        )
        seed_metrics = seed_primary_metrics(metrics)
        arm_summary = arm_primary_summary(seed_metrics)
        arm_comparison = arm_reference_comparison(seed_metrics)
        contrasts_seed = planned_contrasts_by_seed(seed_metrics)
        contrasts_across = planned_contrasts_across_seed(contrasts_seed)
        local_tail = sf.local_rr_tail_summary(metrics)
        local_tail_across = local_rr_tail_across_seed(local_tail)
        subject_metrics = sf.subject_stratified_metrics(metrics)
        subject_macro_seed, subject_macro_across = subject_macro_tables(subject_metrics)
        denominators = sf.metric_denominators(metrics)
        resources = parameter_compute_memory_table(receipts)
        historical = historical_e8_pointwise_reference()
        decision = build_decision(seed_metrics, contrasts_across)
        outputs = {
            "seed_primary_metrics.csv": seed_metrics,
            "arm_primary_summary.csv": arm_summary,
            "arm_reference_comparison.csv": arm_comparison,
            "planned_contrasts_by_seed.csv": contrasts_seed,
            "planned_contrasts_across_seed.csv": contrasts_across,
            "local_rr_tail_summary.csv": local_tail,
            "local_rr_tail_across_seed.csv": local_tail_across,
            "subject_stratified_metrics.csv": subject_metrics,
            "subject_macro_by_seed.csv": subject_macro_seed,
            "subject_macro_across_seed.csv": subject_macro_across,
            "metric_denominators.csv": denominators,
            "parameter_compute_memory.csv": resources,
            "historical_e8_pointwise_reference.csv": historical,
        }
        for filename, frame in outputs.items():
            frame.to_csv(output / filename, index=False, na_rep="NA")
        engineering._write_json(output / "decision.json", decision)
        source_identity = _summary_source_identity()
        engineering._write_json(
            output / "source_manifest.json",
            {
                "protocol": e9.PROTOCOL,
                "implementation_lock": {
                    "path": str(e9.FORMAL_LOCK_PATH),
                    "sha256": lock_hash,
                },
                "formal_runtime_amendment": {
                    "path": str(formal.FORMAL_AMENDMENT_PATH),
                    "sha256": amendment_hash,
                },
                "summary_source": source_identity,
                **sources,
            },
        )
        engineering._write_json(output / "source_code.json", source_identity)
        engineering._write_json(
            output / "access_receipt.json",
            {
                "formal_attempts_read": len(attempts),
                "validation_metric_rows_read": len(metrics),
                "checkpoints_replayed": 36,
                "validation_only": True,
                "gpu_used": False,
                "model_training_used": False,
                "model_inference_used": False,
                "research_test_accessed": False,
                **lifecycle_summary,
            },
        )
        engineering._write_json(
            output / "summary_receipt.json",
            {
                "protocol": e9.PROTOCOL,
                "status": "complete",
                "implementation_lock_sha256": lock_hash,
                "formal_runtime_amendment_sha256": amendment_hash,
                "summary_commit": state["commit"],
                "arms": list(ARMS),
                "seeds": list(e9.SEEDS),
                "formal_attempts": len(attempts),
                "validation_rows_per_attempt": formal.COUNTS["val"],
                "validation_metric_rows": len(metrics),
                "aggregation": {
                    "window_primary": "eligible sample direct mean within seed",
                    "seed": "arithmetic mean and sample SD ddof=1",
                    "subject_macro": "equal weight across seven validation samp_id",
                    "inference": "descriptive; no p-values",
                },
                "decision": decision,
            },
        )
    return output


def verify_summary_attempt(path: Path) -> dict[str, Any]:
    lock, lock_hash, _source = formal.load_formal_lock()
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    path = path.resolve()
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    manifest_path = path / "manifest.json"
    if engineering._identity(manifest_path) != freeze.get("manifest"):
        raise RuntimeError("E9 summary freeze receipt身份错误")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != e9.PROTOCOL
        or manifest.get("phase") != "validation_summary"
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
        or manifest.get("formal_runtime_amendment_sha256") != amendment_hash
        or not SUMMARY_REQUIRED_FILES.issubset(manifest.get("files", {}))
    ):
        raise ValueError("E9 summary manifest合同漂移")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path) or engineering._identity(target) != expected:
            raise RuntimeError(f"E9 summary文件身份漂移: {relative}")
    receipt = json.loads((path / "summary_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or receipt.get("formal_attempts") != 18
        or receipt.get("validation_metric_rows") != 18 * formal.COUNTS["val"]
    ):
        raise ValueError("E9 summary receipt计数漂移")
    return manifest

