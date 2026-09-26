"""E8 P5：完整 36-cell validation 汇总与预注册对比。"""

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

from resp_train.paper_evidence import e8_film_decoder_redesign_v1 as e8
from resp_train.paper_evidence import e8_film_decoder_redesign_v1_engineering as engineering
from resp_train.paper_evidence import e8_film_decoder_redesign_v1_formal as formal
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_model import (
    ARMS,
    ARM_SPECS,
    CONDITION_REFINERS,
    DECODERS,
    RECOMMENDED_ARM,
    REFERENCE_ARM,
)


SUMMARY_PATH = Path("resp_train/paper_evidence/e8_film_decoder_redesign_v1_summary.py")
ERRORS = tuple(e8.PRIMARY_METRICS[:4])
PCC = e8.PRIMARY_METRICS[4]
PRIMARY = (*ERRORS, PCC)
ERROR_RELATIVE_TOLERANCE = 0.005
PCC_ABSOLUTE_TOLERANCE = 0.002

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
    "decision.json",
    "source_manifest.json",
    "summary_receipt.json",
    "access_receipt.json",
    "source_code.json",
}


def completed_formal_attempts() -> tuple[list[Path], dict[str, Any]]:
    status = formal.matrix_status()
    expected_counts = {"pending": 0, "running": 0, "failed": 0, "completed": 36}
    if status["counts"] != expected_counts:
        raise RuntimeError(f"E8 P5 要求完整 36-cell，当前={status['counts']}")
    attempts: list[Path] = []
    failed_lifecycles: list[str] = []
    for cell in status["cells"]:
        if len(cell["completed"]) != 1:
            raise RuntimeError(
                f"E8 P5 cell 必须恰有一个成功 attempt: {cell['arm']}/{cell['seed']}"
            )
        attempts.append(Path(cell["completed"][0]).resolve())
        failed_lifecycles.extend(cell["failed"])
    if len(attempts) != 36 or len(set(attempts)) != 36:
        raise RuntimeError("E8 P5 formal attempt 集合重复或不完整")
    return attempts, {
        "matrix_status": status["counts"],
        "failed_lifecycle_count": len(failed_lifecycles),
        "failed_lifecycles": failed_lifecycles,
        "formal_runtime_amendment_sha256": status["formal_runtime_amendment_sha256"],
    }


def _metric_mask(frame: pd.DataFrame, metric: str) -> np.ndarray:
    flag = sf.ELIGIBILITY.get(metric)
    if flag is None:
        return np.ones(len(frame), dtype=bool)
    values = frame[flag]
    if values.isna().any() or not values.isin([True, False]).all():
        raise ValueError(f"E8 eligibility 非法: {flag}")
    return values.to_numpy(dtype=bool)


def seed_primary_metrics(
    window_metrics: pd.DataFrame,
    *,
    expected_rows: int = formal.COUNTS["val"],
) -> pd.DataFrame:
    expected_pairs = {(arm, seed) for arm in ARMS for seed in e8.SEEDS}
    observed_pairs = set(
        window_metrics[["arm", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_pairs != expected_pairs:
        raise ValueError("E8 seed-primary arm/seed 矩阵不完整")
    rows: list[dict[str, Any]] = []
    for (arm, seed), group in window_metrics.groupby(["arm", "seed"], sort=False):
        if len(group) != expected_rows or group.dataset_row_id.duplicated().any():
            raise ValueError(f"E8 validation rows 不完整: {arm}/{seed}")
        row: dict[str, Any] = {"arm": arm, "seed": int(seed)}
        for metric in PRIMARY:
            mask = _metric_mask(group, metric)
            values = pd.to_numeric(group.loc[mask, metric], errors="coerce").to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError(f"E8 primary metric 非有限: {arm}/{seed}/{metric}")
            row[metric] = float(values.mean())
            row[metric + "_n"] = len(values)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    if len(result) != 36:
        raise ValueError("E8 seed-primary 行数错误")
    return result


def arm_primary_summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        group = seed_frame.loc[seed_frame.arm.eq(arm)].set_index("seed").loc[list(e8.SEEDS)]
        for metric in PRIMARY:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "arm": arm,
                    "condition_refiner": ARM_SPECS[arm].condition_refiner,
                    "decoder": ARM_SPECS[arm].decoder,
                    "metric": metric,
                    "seed_count": len(values),
                    "seed_mean": float(values.mean()),
                    "seed_sample_sd": float(values.std(ddof=1)),
                    "seed_min": float(values.min()),
                    "seed_max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _require_complete_seed_frame(frame: pd.DataFrame) -> None:
    required = {"arm", "seed", *PRIMARY}
    if not required.issubset(frame.columns):
        raise ValueError("E8 seed frame 字段不完整")
    expected = {(arm, seed) for arm in ARMS for seed in e8.SEEDS}
    observed = set(frame[["arm", "seed"]].itertuples(index=False, name=None))
    if observed != expected or len(frame) != len(expected):
        raise ValueError("E8 seed frame 矩阵不完整")
    if not np.isfinite(frame[list(PRIMARY)].to_numpy(float)).all():
        raise FloatingPointError("E8 seed frame 主指标非有限")


def _benefit(reference: float, candidate: float, metric: str) -> tuple[float, float]:
    if metric == PCC:
        oriented = candidate - reference
        return oriented, oriented
    if reference == 0:
        raise ZeroDivisionError(f"E8 reference error 为零: {metric}")
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
            benefits: list[float] = []
            normalized: list[float] = []
            for seed in e8.SEEDS:
                reference = float(indexed.loc[(REFERENCE_ARM, seed), metric])
                candidate = float(indexed.loc[(arm, seed), metric])
                raw, norm = _benefit(reference, candidate, metric)
                benefits.append(raw)
                normalized.append(norm)
            threshold = PCC_ABSOLUTE_TOLERANCE if metric == PCC else ERROR_RELATIVE_TOLERANCE
            values = np.asarray(normalized)
            rows.append(
                {
                    "arm": arm,
                    "reference_arm": REFERENCE_ARM,
                    "metric": metric,
                    "benefit_unit": "absolute" if metric == PCC else "relative_fraction",
                    "material_threshold": threshold,
                    "oriented_benefit_seed_mean": float(np.mean(benefits)),
                    "reference_normalized_benefit_seed_mean": float(values.mean()),
                    "reference_normalized_benefit_seed_sample_sd": float(values.std(ddof=1)),
                    "benefit_positive_seeds": int((values > 0).sum()),
                    "materially_improved_seeds": int((values > threshold).sum()),
                    "materially_worsened_seeds": int((values < -threshold).sum()),
                    "seed_values": "|".join(f"{value:.17g}" for value in values),
                }
            )
    return pd.DataFrame(rows)


def _arm(condition: str, decoder: str) -> str:
    arm = f"e8_{condition}_{decoder}"
    if arm not in ARM_SPECS:
        raise KeyError(arm)
    return arm


def planned_contrasts_by_seed(seed_frame: pd.DataFrame) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame)
    indexed = seed_frame.set_index(["arm", "seed"])
    rows: list[dict[str, Any]] = []

    def append(
        *,
        seed: int,
        metric: str,
        contrast_type: str,
        contrast: str,
        reference_arms: Sequence[str],
        candidate_arms: Sequence[str],
        context: str,
    ) -> None:
        reference = float(np.mean([indexed.loc[(arm, seed), metric] for arm in reference_arms]))
        candidate = float(np.mean([indexed.loc[(arm, seed), metric] for arm in candidate_arms]))
        oriented, normalized = _benefit(reference, candidate, metric)
        rows.append(
            {
                "seed": seed,
                "metric": metric,
                "contrast_type": contrast_type,
                "contrast": contrast,
                "context": context,
                "reference_arms": "|".join(reference_arms),
                "candidate_arms": "|".join(candidate_arms),
                "reference_value": reference,
                "candidate_value": candidate,
                "raw_candidate_minus_reference": candidate - reference,
                "oriented_benefit": oriented,
                "reference_normalized_benefit": normalized,
            }
        )

    condition_pairs = (
        ("direct_vs_fill65", "fill65", "direct"),
        ("res96_vs_direct", "direct", "res96"),
        ("res192_vs_res96", "res96", "res192"),
    )
    decoder_pairs = (
        ("single_vs_pointwise", "pointwise", "single"),
        ("temporal_vs_pointwise", "pointwise", "temporal"),
        ("temporal_vs_single", "single", "temporal"),
    )
    for seed in e8.SEEDS:
        for metric in PRIMARY:
            for label, reference, candidate in condition_pairs:
                for decoder in DECODERS:
                    append(
                        seed=seed,
                        metric=metric,
                        contrast_type="condition_simple",
                        contrast=label,
                        reference_arms=[_arm(reference, decoder)],
                        candidate_arms=[_arm(candidate, decoder)],
                        context=f"decoder={decoder}",
                    )
                append(
                    seed=seed,
                    metric=metric,
                    contrast_type="condition_marginal",
                    contrast=label,
                    reference_arms=[_arm(reference, decoder) for decoder in DECODERS],
                    candidate_arms=[_arm(candidate, decoder) for decoder in DECODERS],
                    context="equal_mean_over_decoder",
                )
            for label, reference, candidate in decoder_pairs:
                for condition in CONDITION_REFINERS:
                    append(
                        seed=seed,
                        metric=metric,
                        contrast_type="decoder_simple",
                        contrast=label,
                        reference_arms=[_arm(condition, reference)],
                        candidate_arms=[_arm(condition, candidate)],
                        context=f"condition={condition}",
                    )
                append(
                    seed=seed,
                    metric=metric,
                    contrast_type="decoder_marginal",
                    contrast=label,
                    reference_arms=[_arm(condition, reference) for condition in CONDITION_REFINERS],
                    candidate_arms=[_arm(condition, candidate) for condition in CONDITION_REFINERS],
                    context="equal_mean_over_condition",
                )

            for condition in CONDITION_REFINERS[1:]:
                for decoder in DECODERS[1:]:
                    base = float(indexed.loc[(_arm("fill65", "pointwise"), seed), metric])
                    c_point = float(indexed.loc[(_arm(condition, "pointwise"), seed), metric])
                    f_decoder = float(indexed.loc[(_arm("fill65", decoder), seed), metric])
                    c_decoder = float(indexed.loc[(_arm(condition, decoder), seed), metric])
                    utility_sign = 1.0 if metric == PCC else -1.0
                    oriented = utility_sign * (
                        (c_decoder - c_point) - (f_decoder - base)
                    )
                    normalized = oriented if metric == PCC else oriented / abs(base)
                    rows.append(
                        {
                            "seed": seed,
                            "metric": metric,
                            "contrast_type": "interaction",
                            "contrast": f"{condition}_x_{decoder}",
                            "context": "reference=fill65_x_pointwise",
                            "reference_arms": (
                                f"{_arm('fill65', 'pointwise')}|{_arm(condition, 'pointwise')}|"
                                f"{_arm('fill65', decoder)}"
                            ),
                            "candidate_arms": _arm(condition, decoder),
                            "reference_value": base,
                            "candidate_value": c_decoder,
                            "raw_candidate_minus_reference": c_decoder - base,
                            "oriented_benefit": oriented,
                            "reference_normalized_benefit": normalized,
                        }
                    )
    result = pd.DataFrame(rows)
    expected_rows = len(e8.SEEDS) * len(PRIMARY) * (3 * 4 + 3 * 5 + 6)
    if len(result) != expected_rows or not np.isfinite(
        result[["oriented_benefit", "reference_normalized_benefit"]].to_numpy(float)
    ).all():
        raise ValueError("E8 planned contrasts 矩阵不完整或非有限")
    return result


def planned_contrasts_across_seed(by_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["contrast_type", "contrast", "context", "metric"]
    rows: list[dict[str, Any]] = []
    for group_key, group in by_seed.groupby(keys, sort=False):
        group = group.set_index("seed").loc[list(e8.SEEDS)]
        benefit = group.oriented_benefit.to_numpy(float)
        normalized = group.reference_normalized_benefit.to_numpy(float)
        metric = str(group_key[-1])
        threshold = PCC_ABSOLUTE_TOLERANCE if metric == PCC else ERROR_RELATIVE_TOLERANCE
        rows.append(
            {
                **dict(zip(keys, group_key)),
                "seed_count": len(group),
                "oriented_benefit_mean": float(benefit.mean()),
                "oriented_benefit_sample_sd": float(benefit.std(ddof=1)),
                "reference_normalized_benefit_mean": float(normalized.mean()),
                "reference_normalized_benefit_sample_sd": float(normalized.std(ddof=1)),
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
    return pd.DataFrame(rows)


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
                    no_worse &= bool(candidate[metric] >= current[metric] - PCC_ABSOLUTE_TOLERANCE)
                    better |= bool(candidate[metric] > current[metric] + PCC_ABSOLUTE_TOLERANCE)
                else:
                    no_worse &= bool(
                        candidate[metric] <= current[metric] * (1.0 + ERROR_RELATIVE_TOLERANCE)
                    )
                    better |= bool(
                        candidate[metric] < current[metric] * (1.0 - ERROR_RELATIVE_TOLERANCE)
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
    rows: list[dict[str, Any]] = []
    for (arm, seed, metric), group in subject_metrics.groupby(
        ["arm", "seed", "metric"], sort=False
    ):
        if len(group) != formal.SAMP_IDS["val"] or group.samp_id.nunique() != formal.SAMP_IDS["val"]:
            raise ValueError(f"E8 subject macro 不完整: {arm}/{seed}/{metric}")
        values = group["mean"].to_numpy(float)
        if not np.isfinite(values).all():
            raise FloatingPointError("E8 subject macro 非有限")
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
        values = group.set_index("seed").loc[list(e8.SEEDS)].subject_macro_mean.to_numpy(float)
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
        group = group.set_index("seed").loc[list(e8.SEEDS)]
        row: dict[str, Any] = {"arm": arm, "seed_count": len(e8.SEEDS)}
        for column in value_columns:
            values = group[column].to_numpy(float)
            row[column + "_seed_mean"] = float(values.mean())
            row[column + "_seed_sample_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_compute_memory_table(receipts: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    benchmark_path = e8.SOURCE_ROOT / formal.P2_BENCHMARK / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    measurements = {(item["arm"], item["mode"]): item for item in benchmark["measurements"]}
    expected = {(arm, mode) for arm in ARMS for mode in ("eval", "train")}
    if set(measurements) != expected:
        raise ValueError("E8 benchmark 矩阵不完整")
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        arm = str(receipt["arm"])
        spec = ARM_SPECS[arm]
        eval_item = measurements[(arm, "eval")]
        train_item = measurements[(arm, "train")]
        runtime = receipt["runtime"]
        rows.append(
            {
                "arm": arm,
                "seed": int(receipt["seed"]),
                "condition_refiner": spec.condition_refiner,
                "decoder": spec.decoder,
                "trainable_parameters": spec.trainable_parameters,
                "factor_covered_macs": spec.factor_covered_macs,
                "completed_epochs": int(receipt["completed_epochs"]),
                "completed_updates": int(receipt["completed_updates"]),
                "selected_epoch": int(receipt["selected_epoch"]),
                "formal_wall_seconds": float(receipt["formal_wall_seconds"]),
                "formal_peak_allocated_mib": float(runtime["peak_allocated_mib"]),
                "formal_peak_reserved_mib": float(runtime["peak_reserved_mib"]),
                "formal_peak_reserved_fraction": float(runtime["peak_reserved_fraction"]),
                "benchmark_eval_median_seconds": float(eval_item["median_seconds"]),
                "benchmark_eval_samples_per_second": float(
                    eval_item["throughput_samples_per_second"]
                ),
                "benchmark_train_median_seconds": float(train_item["median_seconds"]),
                "benchmark_train_samples_per_second": float(
                    train_item["throughput_samples_per_second"]
                ),
                "benchmark_train_peak_reserved_fraction": float(
                    train_item["peak_reserved_fraction"]
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    if len(frame) != 36 or frame[["arm", "seed"]].duplicated().any():
        raise ValueError("E8 resource cell 不完整")
    return frame


def _audit_formal_inputs(
    attempts: Sequence[Path],
    lock: Mapping[str, Any],
    lock_hash: str,
    source_lock: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    amendment = lock["_runtime_amendment"]
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    observed: set[tuple[str, int]] = set()
    source_commits: dict[str, int] = {}
    for attempt in attempts:
        receipt = json.loads((attempt / "formal_receipt.json").read_text(encoding="utf-8"))
        arm, seed = str(receipt["arm"]), int(receipt["seed"])
        if arm not in ARMS or seed not in e8.SEEDS or (arm, seed) in observed:
            raise ValueError("E8 P5 formal cell 重复或越界")
        observed.add((arm, seed))
        manifest = formal.verify_formal_attempt(
            attempt,
            lock_hash=lock_hash,
            arm=arm,
            seed=seed,
            amendment=amendment,
            amendment_hash=amendment_hash,
        )
        run_dir = (attempt / receipt["run_dir"]).resolve()
        cfg = OmegaConf.load(run_dir / "config.yaml")
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        e8.validate_config(
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
                raise ValueError(f"E8 P5 formal receipt 回放不一致: {arm}/{seed}/{key}")
        metrics = pd.read_csv(run_dir / "metrics.csv")
        frames.append(metrics)
        receipts.append(receipt)
        environment = json.loads((attempt / "environment.json").read_text(encoding="utf-8"))
        commit = str(environment["git"]["commit"])
        if environment["git"]["status_porcelain"]:
            raise ValueError("E8 formal source Git 非 clean")
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
            "selected_epoch": int(receipt["selected_epoch"]),
            "completed_epochs": int(receipt["completed_epochs"]),
            "source_commit": commit,
            "formal_runtime_amendment_sha256": receipt.get(
                "formal_runtime_amendment_sha256"
            ),
            "manifest_file_count": len(manifest["files"]),
        }
    expected = {(arm, seed) for arm in ARMS for seed in e8.SEEDS}
    if observed != expected:
        raise ValueError("E8 P5 formal 36-cell 矩阵不完整")
    metrics = pd.concat(frames, ignore_index=True)
    if len(metrics) != len(expected) * formal.COUNTS["val"]:
        raise ValueError("E8 P5 validation metrics 总行数错误")
    return metrics, receipts, {
        "formal_source_commits": source_commits,
        "formal_runs": sources,
    }


def build_decision(
    seed_frame: pd.DataFrame,
    arm_comparison: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> dict[str, Any]:
    marginal = contrasts.loc[
        contrasts.contrast_type.isin(["condition_marginal", "decoder_marginal"])
    ]
    marginal_decisions: list[dict[str, Any]] = []
    for (contrast_type, contrast), group in marginal.groupby(
        ["contrast_type", "contrast"], sort=False
    ):
        directions = {
            row.metric: row.material_direction for row in group.itertuples()
        }
        marginal_decisions.append(
            {
                "contrast_type": contrast_type,
                "contrast": contrast,
                "metric_directions": directions,
                "pareto_improvement": bool(
                    "improved" in directions.values() and "worsened" not in directions.values()
                ),
                "attribute_tradeoff": bool(
                    "improved" in directions.values() and "worsened" in directions.values()
                ),
            }
        )
    interactions = contrasts.loc[contrasts.contrast_type.eq("interaction")]
    material_interactions = [
        {
            "contrast": row.contrast,
            "metric": row.metric,
            "direction": row.material_direction,
            "reference_normalized_benefit_mean": row.reference_normalized_benefit_mean,
        }
        for row in interactions.itertuples()
        if row.material_direction != "within_tolerance"
    ]
    recommended = arm_comparison.loc[arm_comparison.arm.eq(RECOMMENDED_ARM)]
    recommended_directions = {
        row.metric: (
            "improved"
            if row.reference_normalized_benefit_seed_mean > row.material_threshold
            else "worsened"
            if row.reference_normalized_benefit_seed_mean < -row.material_threshold
            else "within_tolerance"
        )
        for row in recommended.itertuples()
    }
    means = seed_frame.groupby("arm")[list(PRIMARY)].mean()
    best_by_metric = {
        metric: (
            str(means[metric].idxmax()) if metric == PCC else str(means[metric].idxmin())
        )
        for metric in PRIMARY
    }
    return {
        "protocol": e8.PROTOCOL,
        "evidence_scope": "three_seed_validation_descriptive_factorial",
        "reference_arm": REFERENCE_ARM,
        "recommended_design_arm": RECOMMENDED_ARM,
        "recommended_vs_reference": {
            "metric_directions": recommended_directions,
            "pareto_improvement": bool(
                "improved" in recommended_directions.values()
                and "worsened" not in recommended_directions.values()
            ),
            "attribute_tradeoff": bool(
                "improved" in recommended_directions.values()
                and "worsened" in recommended_directions.values()
            ),
        },
        "marginal_contrasts": marginal_decisions,
        "material_interactions": material_interactions,
        "tolerance_aware_pareto": tolerance_pareto(seed_frame),
        "best_arm_by_univariate_seed_mean": best_by_metric,
        "materiality": {
            "error_relative": ERROR_RELATIVE_TOLERANCE,
            "pcc_absolute": PCC_ABSOLUTE_TOLERANCE,
        },
        "statistical_unit": "seed describes optimization randomness; no population inference",
        "window_dependence": "overlapping windows are not independent samples",
        "research_test_confirmation_claimed": False,
    }


def _summary_source_identity() -> dict[str, Any]:
    paths = (
        SUMMARY_PATH,
        Path("resp_train/paper_evidence/e8_film_decoder_redesign_v1.py"),
        Path("resp_train/paper_evidence/e8_film_decoder_redesign_v1_model.py"),
        formal.FORMAL_PATH,
        Path("scripts/summarize_e8_film_decoder_redesign_v1.py"),
        Path("tests/test_e8_film_decoder_redesign_v1_summary.py"),
    )
    return {
        "git": engineering._git_state(),
        "files": {
            str(path): engineering._identity(e8.ROOT / path) for path in paths
        },
    }


@contextmanager
def _summary_attempt(parent: Path, *, lock_hash: str, amendment_hash: str) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".execution_{lock_hash}_{amendment_hash}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("相同 E8 P5 summary 正在运行") from exc
        for receipt_path in parent.glob("*/freeze_receipt.json"):
            manifest_path = receipt_path.parent / "manifest.json"
            freeze = json.loads(receipt_path.read_text(encoding="utf-8"))
            if engineering._identity(manifest_path) != freeze.get("manifest"):
                raise RuntimeError("E8 P5 已完成 summary 身份错误")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("implementation_lock_sha256") == lock_hash
                and manifest.get("formal_runtime_amendment_sha256") == amendment_hash
                and manifest.get("status") == "completed"
            ):
                raise FileExistsError(f"相同 E8 P5 summary 已完成: {receipt_path.parent}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = parent / f"summary_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
        output.mkdir(exist_ok=False)
        context = {
            "protocol": e8.PROTOCOL,
            "phase": "summary",
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
                {"protocol": e8.PROTOCOL, "manifest": engineering._identity(output / "manifest.json")},
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
        raise RuntimeError("E8 P5 summary 要求干净 Git 工作树")
    attempts, lifecycle_summary = completed_formal_attempts()
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    parent = e8.SOURCE_ROOT / e8.OUTPUT_ROOT / "summary"
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
        decision = build_decision(
            seed_metrics,
            arm_comparison,
            contrasts_across,
        )
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
        }
        for filename, frame in outputs.items():
            frame.to_csv(output / filename, index=False, na_rep="NA")
        engineering._write_json(output / "decision.json", decision)
        source_identity = _summary_source_identity()
        engineering._write_json(
            output / "source_manifest.json",
            {
                "protocol": e8.PROTOCOL,
                "implementation_lock": {
                    "path": str(formal.FORMAL_LOCK_PATH),
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
                "checkpoints_replayed": 72,
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
                "protocol": e8.PROTOCOL,
                "status": "complete",
                "implementation_lock_sha256": lock_hash,
                "formal_runtime_amendment_sha256": amendment_hash,
                "summary_commit": state["commit"],
                "arms": list(ARMS),
                "seeds": list(e8.SEEDS),
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
        raise RuntimeError("E8 P5 summary freeze receipt 身份错误")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != e8.PROTOCOL
        or manifest.get("phase") != "summary"
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
        or manifest.get("formal_runtime_amendment_sha256") != amendment_hash
        or not SUMMARY_REQUIRED_FILES.issubset(manifest.get("files", {}))
    ):
        raise ValueError("E8 P5 summary manifest 合同漂移")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path) or engineering._identity(target) != expected:
            raise RuntimeError(f"E8 P5 summary 文件身份漂移: {relative}")
    receipt = json.loads((path / "summary_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or receipt.get("formal_attempts") != 36
        or receipt.get("validation_metric_rows") != 36 * formal.COUNTS["val"]
    ):
        raise ValueError("E8 P5 summary receipt 计数漂移")
    return manifest
