"""W0 三因素结构对照 P4：一次性 validation 汇总与预注册决策。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence import w0_structural_factorial_v1_formal as formal
from resp_train.paper_evidence.w0_structural_factorial_v1_model import ARMS, ARM_SPECS, REFERENCE_ARM


SUMMARY_SOURCE_PATH = Path("resp_train/paper_evidence/w0_structural_factorial_v1_summary.py")
SUMMARY_REQUIRED_FILES = {
    "seed_primary_metrics.csv",
    "arm_primary_summary.csv",
    "arm_reference_comparison.csv",
    "conditional_effects_by_seed.csv",
    "factorial_effects_by_seed.csv",
    "factorial_effects_across_seed.csv",
    "local_rr_tail_summary.csv",
    "local_rr_tail_across_seed.csv",
    "subject_stratified_metrics.csv",
    "subject_macro_by_seed.csv",
    "subject_macro_across_seed.csv",
    "subject_conditional_effects.csv",
    "subject_factorial_effects.csv",
    "metric_denominators.csv",
    "parameter_compute_memory.csv",
    "decision.json",
    "source_manifest.json",
    "summary_receipt.json",
    "access_receipt.json",
    "source_code.json",
}


def completed_formal_attempts() -> list[Path]:
    status = formal.matrix_status()
    if status["counts"] != {"pending": 0, "running": 0, "failed": 0, "completed": 24}:
        raise RuntimeError(f"P4 要求完整 24-cell 完成矩阵，当前={status['counts']}")
    attempts: list[Path] = []
    for cell in status["cells"]:
        if len(cell["completed"]) != 1:
            raise RuntimeError(f"P4 cell 必须恰有一个完成 attempt: {cell['arm']}/{cell['seed']}")
        attempts.append(Path(cell["completed"][0]).resolve())
    if len(attempts) != 24 or len(set(attempts)) != 24:
        raise RuntimeError("P4 formal attempt 集合重复或不完整")
    return attempts


def subject_macro_tables(
    subject_metrics: pd.DataFrame,
    *,
    expected_subjects: int = sf.SAMP_IDS["val"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (arm, seed, metric), group in subject_metrics.groupby(
        ["arm", "seed", "metric"], sort=False
    ):
        if len(group) != expected_subjects or group.samp_id.nunique() != expected_subjects:
            raise ValueError(f"subject macro samp_id 不完整: {arm}/{seed}/{metric}")
        values = group["mean"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FloatingPointError("subject macro 指标非有限")
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
    across_rows: list[dict[str, Any]] = []
    for (arm, metric), group in by_seed.groupby(["arm", "metric"], sort=False):
        group = group.set_index("seed").loc[list(sf.SEEDS)]
        values = group.subject_macro_mean.to_numpy(dtype=float)
        across_rows.append(
            {
                "arm": arm,
                "metric": metric,
                "seed_count": len(values),
                "subject_macro_seed_mean": float(values.mean()),
                "subject_macro_seed_sample_sd": float(values.std(ddof=1)),
            }
        )
    return by_seed, pd.DataFrame(across_rows)


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
        group = group.set_index("seed").loc[list(sf.SEEDS)]
        row: dict[str, Any] = {"arm": arm, "seed_count": len(sf.SEEDS)}
        for column in value_columns:
            values = group[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Local RR tail 非有限: {arm}/{column}")
            row[column + "_seed_mean"] = float(values.mean())
            row[column + "_seed_sample_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def arm_reference_comparison(seed_frame: pd.DataFrame) -> pd.DataFrame:
    sf._require_complete_seed_frame(seed_frame)
    indexed = seed_frame.set_index(["arm", "seed"])
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        for metric in sf.PRIMARY:
            reference = np.asarray(
                [indexed.loc[(REFERENCE_ARM, seed), metric] for seed in sf.SEEDS], dtype=float
            )
            candidate = np.asarray(
                [indexed.loc[(arm, seed), metric] for seed in sf.SEEDS], dtype=float
            )
            if metric == sf.PCC:
                benefit = candidate - reference
                threshold = sf.PCC_ABSOLUTE_TOLERANCE
                unit = "absolute"
            else:
                if np.any(reference == 0):
                    raise ZeroDivisionError(f"reference error 为零: {metric}")
                benefit = (reference - candidate) / reference
                threshold = sf.ERROR_RELATIVE_TOLERANCE
                unit = "relative_fraction"
            rows.append(
                {
                    "arm": arm,
                    "reference_arm": REFERENCE_ARM,
                    "metric": metric,
                    "unit": unit,
                    "material_threshold": threshold,
                    "oriented_benefit_seed_mean": float(benefit.mean()),
                    "oriented_benefit_seed_sample_sd": float(benefit.std(ddof=1)),
                    "benefit_positive_seeds": int((benefit > 0).sum()),
                    "materially_improved_seeds": int((benefit > threshold).sum()),
                    "materially_worsened_seeds": int((benefit < -threshold).sum()),
                    "seed_values": "|".join(f"{value:.17g}" for value in benefit),
                }
            )
    return pd.DataFrame(rows)


def parameter_compute_memory_table(
    receipts: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    benchmark_path = sf.SOURCE_ROOT / formal.P2_BENCHMARK / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    measurements = {
        (item["arm"], item["mode"]): item for item in benchmark["measurements"]
    }
    expected = {(arm, mode) for arm in ARMS for mode in ("eval", "train")}
    if set(measurements) != expected:
        raise ValueError("P4 benchmark arm/mode 矩阵不完整")
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        arm = str(receipt["arm"])
        eval_item = measurements[(arm, "eval")]
        train_item = measurements[(arm, "train")]
        spec = ARM_SPECS[arm]
        runtime = receipt["runtime"]
        rows.append(
            {
                "arm": arm,
                "seed": int(receipt["seed"]),
                "factors": "|".join(str(value) for value in spec.factors),
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
                "benchmark_eval_peak_allocated_bytes": int(eval_item["peak_allocated_bytes"]),
                "benchmark_train_median_seconds": float(train_item["median_seconds"]),
                "benchmark_train_samples_per_second": float(
                    train_item["throughput_samples_per_second"]
                ),
                "benchmark_train_peak_allocated_bytes": int(train_item["peak_allocated_bytes"]),
                "benchmark_train_peak_reserved_fraction": float(
                    train_item["peak_reserved_fraction"]
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    if len(frame) != 24 or frame[["arm", "seed"]].duplicated().any():
        raise ValueError("P4 parameter/compute/memory cell 不完整")
    return frame


def tolerance_pareto_arms(seed_frame: pd.DataFrame) -> dict[str, Any]:
    means = seed_frame.groupby("arm", sort=False)[list(sf.PRIMARY)].mean()
    dominated_by: dict[str, list[str]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        current = means.loc[arm]
        for other in ARMS:
            if arm == other:
                continue
            candidate = means.loc[other]
            no_worse = True
            materially_better = False
            for metric in sf.PRIMARY:
                if metric == sf.PCC:
                    no_worse &= bool(candidate[metric] >= current[metric] - sf.PCC_ABSOLUTE_TOLERANCE)
                    materially_better |= bool(candidate[metric] > current[metric] + sf.PCC_ABSOLUTE_TOLERANCE)
                else:
                    no_worse &= bool(
                        candidate[metric]
                        <= current[metric] * (1.0 + sf.ERROR_RELATIVE_TOLERANCE)
                    )
                    materially_better |= bool(
                        candidate[metric]
                        < current[metric] * (1.0 - sf.ERROR_RELATIVE_TOLERANCE)
                    )
            if no_worse and materially_better:
                dominated_by[arm].append(other)
    return {
        "pareto_arms": [arm for arm in ARMS if not dominated_by[arm]],
        "dominated_by": dominated_by,
        "definition": {
            "errors_relative_tolerance": sf.ERROR_RELATIVE_TOLERANCE,
            "pcc_absolute_tolerance": sf.PCC_ABSOLUTE_TOLERANCE,
            "dominance": "other is within tolerance on all metrics and materially better on at least one",
        },
    }


def factor_assessments(
    conditional: pd.DataFrame,
    effects_by_seed: pd.DataFrame,
    effects_across_seed: pd.DataFrame,
) -> list[dict[str, Any]]:
    contexts = {
        "A": {"B_level": 1, "C_level": 1},
        "B": {"A_level": 0, "C_level": 1},
        "C": {"A_level": 0, "B_level": 1},
    }
    interactions = {
        "A": ("AB", "AC", "ABC"),
        "B": ("AB", "BC", "ABC"),
        "C": ("AC", "BC", "ABC"),
    }
    parameter_reductions = {
        "A": 15_888,
        "B": 112_896,
        "C": 75_264,
    }
    rows: list[dict[str, Any]] = []
    for factor in ("A", "B", "C"):
        main = effects_across_seed.loc[effects_across_seed.effect.eq(factor)].set_index("metric")
        improved: list[str] = []
        worsened: list[str] = []
        unstable: list[str] = []
        main_evidence: dict[str, Any] = {}
        for metric in sf.PRIMARY:
            row = main.loc[metric]
            threshold = sf.PCC_ABSOLUTE_TOLERANCE if metric == sf.PCC else sf.ERROR_RELATIVE_TOLERANCE
            benefit = float(row.reference_normalized_benefit_mean)
            positives = int(row.reference_normalized_benefit_positive_seeds)
            if benefit > threshold and positives >= 2:
                improved.append(metric)
            if benefit < -threshold and positives <= 1:
                worsened.append(metric)
            if positives not in (0, 3):
                unstable.append(metric)
            main_evidence[metric] = {
                "oriented_benefit_mean": float(row.oriented_benefit_mean),
                "reference_normalized_benefit_mean": benefit,
                "positive_seeds": positives,
                "material_threshold": threshold,
            }

        neighborhood = conditional.loc[conditional.factor.eq(factor)].copy()
        for key, value in contexts[factor].items():
            neighborhood = neighborhood.loc[neighborhood[key].eq(value)]
        neighborhood_evidence: dict[str, Any] = {}
        neighborhood_adverse: list[str] = []
        neighborhood_high_benefit_material: list[str] = []
        for metric in sf.PRIMARY:
            values = neighborhood.loc[
                neighborhood.metric.eq(metric), "reference_normalized_benefit"
            ].to_numpy(dtype=float)
            if values.size != len(sf.SEEDS) or not np.isfinite(values).all():
                raise ValueError(f"W0 neighborhood simple effect 不完整: {factor}/{metric}")
            mean = float(values.mean())
            threshold = sf.PCC_ABSOLUTE_TOLERANCE if metric == sf.PCC else sf.ERROR_RELATIVE_TOLERANCE
            if mean < -threshold:
                neighborhood_adverse.append(metric)
            if mean > threshold:
                neighborhood_high_benefit_material.append(metric)
            neighborhood_evidence[metric] = {
                "reference_normalized_benefit_mean": mean,
                "positive_seeds": int((values > 0).sum()),
                "material_threshold": threshold,
            }

        related = effects_across_seed.loc[
            effects_across_seed.effect.isin(interactions[factor])
        ]
        material_interactions: list[str] = []
        for item in related.itertuples():
            threshold = (
                sf.PCC_ABSOLUTE_TOLERANCE
                if item.metric == sf.PCC
                else sf.ERROR_RELATIVE_TOLERANCE
            )
            if abs(float(item.reference_normalized_benefit_mean)) > threshold:
                material_interactions.append(f"{item.effect}:{item.metric}")

        sign_reversals: list[str] = []
        factor_conditionals = conditional.loc[conditional.factor.eq(factor)]
        for metric, group in factor_conditionals.groupby("metric", sort=False):
            context_means = group.groupby(
                list(contexts[factor]),
                sort=False,
            ).oriented_benefit.mean()
            if bool((context_means > 0).any() and (context_means < 0).any()):
                sign_reversals.append(str(metric))

        main_neighborhood_conflict = bool(
            set(improved) & set(neighborhood_adverse)
            or set(worsened) & set(neighborhood_high_benefit_material)
        )
        retain = bool(improved and not worsened and not neighborhood_adverse)
        simplify = bool(
            not improved
            and not neighborhood_high_benefit_material
            and parameter_reductions[factor] > 0
        )
        net_negative = bool(worsened and not improved)
        context_dependent = bool(
            material_interactions or sign_reversals or main_neighborhood_conflict
        )
        if context_dependent:
            classification = "context_dependent_continue_research"
        elif retain:
            classification = "supports_level1"
        elif simplify:
            classification = "supports_level0_simplification"
        elif net_negative:
            classification = "level1_net_negative"
        else:
            classification = "inconclusive_within_tolerance"
        rows.append(
            {
                "factor": factor,
                "classification": classification,
                "main_material_improvements": improved,
                "main_material_worsenings": worsened,
                "seed_direction_instability": unstable,
                "w0_neighborhood_adverse_metrics": neighborhood_adverse,
                "w0_neighborhood_material_level1_benefits": neighborhood_high_benefit_material,
                "material_interactions": sorted(material_interactions),
                "conditional_sign_reversals": sorted(sign_reversals),
                "main_neighborhood_conflict": main_neighborhood_conflict,
                "level0_parameter_reduction": parameter_reductions[factor],
                "main_effects": main_evidence,
                "w0_neighborhood_simple_effects": neighborhood_evidence,
            }
        )
    return rows


def build_decision(
    seed_frame: pd.DataFrame,
    arm_comparison: pd.DataFrame,
    conditional: pd.DataFrame,
    effects_by_seed: pd.DataFrame,
    effects_across_seed: pd.DataFrame,
    resources: pd.DataFrame,
) -> dict[str, Any]:
    simplifications = sf.quality_preserving_simplifications(seed_frame)
    reference_resource = resources.loc[resources.arm.eq(REFERENCE_ARM)].iloc[0]
    benchmark = resources.drop_duplicates("arm").set_index("arm")
    simplification_rows: list[dict[str, Any]] = []
    for row in simplifications.to_dict(orient="records"):
        item = benchmark.loc[row["arm"]]
        throughput_gain = (
            float(item.benchmark_train_samples_per_second)
            / float(reference_resource.benchmark_train_samples_per_second)
            - 1.0
        )
        memory_reduction = 1.0 - (
            float(item.benchmark_train_peak_allocated_bytes)
            / float(reference_resource.benchmark_train_peak_allocated_bytes)
        )
        row["benchmark_train_throughput_gain"] = throughput_gain
        row["benchmark_peak_allocated_reduction"] = memory_reduction
        row["measured_efficiency_simplification"] = bool(
            row["quality_preserving"]
            and (throughput_gain >= 0.10 or memory_reduction >= 0.15)
        )
        simplification_rows.append(row)

    tradeoffs: list[dict[str, Any]] = []
    for arm, group in arm_comparison.groupby("arm", sort=False):
        improved = group.loc[
            group.oriented_benefit_seed_mean > group.material_threshold, "metric"
        ].tolist()
        worsened = group.loc[
            group.oriented_benefit_seed_mean < -group.material_threshold, "metric"
        ].tolist()
        if improved and worsened:
            tradeoffs.append({"arm": arm, "improved_metrics": improved, "worsened_metrics": worsened})

    return {
        "protocol": sf.PROTOCOL,
        "evidence_scope": "three_seed_validation_descriptive_factorial",
        "statistical_unit": "seed describes training randomness; no population inference",
        "window_dependence": "overlapping windows are not independent samples",
        "local_rr_scope": "window-level spectral-peak error, not breath-phase tracking",
        "reference_arm": REFERENCE_ARM,
        "factor_assessments": factor_assessments(
            conditional, effects_by_seed, effects_across_seed
        ),
        "structural_simplifications": simplification_rows,
        "attribute_tradeoffs": tradeoffs,
        "tolerance_aware_pareto": tolerance_pareto_arms(seed_frame),
        "materiality": {
            "error_relative": sf.ERROR_RELATIVE_TOLERANCE,
            "pcc_absolute": sf.PCC_ABSOLUTE_TOLERANCE,
        },
        "research_test_confirmation_claimed": False,
    }


def _source_identity() -> dict[str, Any]:
    paths = tuple(
        dict.fromkeys(
            (*sf.critical_paths(), formal.FORMAL_SOURCE_PATH, SUMMARY_SOURCE_PATH, formal.P2_RECEIPT_PATH)
        )
    )
    return {
        "git": sf.git_state(),
        "files": {str(path): sf.identity(sf.ROOT / path) for path in paths},
    }


def _audit_formal_inputs(
    attempts: Sequence[Path],
    lock: Mapping[str, Any],
    lock_hash: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    formal_source_identity: str | None = None
    formal_commit: str | None = None
    observed: set[tuple[str, int]] = set()
    for attempt in attempts:
        receipt = json.loads((attempt / "formal_receipt.json").read_text(encoding="utf-8"))
        arm, seed = str(receipt["arm"]), int(receipt["seed"])
        if arm not in ARMS or seed not in sf.SEEDS or (arm, seed) in observed:
            raise ValueError("P4 formal cell 重复或越界")
        observed.add((arm, seed))
        manifest = formal.verify_formal_attempt(
            attempt, lock_hash=lock_hash, arm=arm, seed=seed
        )
        run_dir = (attempt / receipt["run_dir"]).resolve()
        cfg = OmegaConf.load(run_dir / "config.yaml")
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        sf.validate_config(
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
            arm=arm,
            initialization_path=attempt / "initialization.json",
        )
        for key, value in checked.items():
            if receipt.get(key) != value:
                raise ValueError(f"P4 formal receipt 回放不一致: {arm}/{seed}/{key}")
        metrics = pd.read_csv(run_dir / "metrics.csv")
        formal.validate_anchor_rows(metrics, lock, seed)
        frames.append(metrics)
        receipts.append(receipt)

        source_code = json.loads((attempt / "source_code.json").read_text(encoding="utf-8"))
        encoded_files = json.dumps(source_code["files"], sort_keys=True, separators=(",", ":"))
        commit = source_code["git"]["commit"]
        if source_code["git"]["status_porcelain"]:
            raise ValueError("P4 formal source Git 非 clean")
        if formal_source_identity is None:
            formal_source_identity = encoded_files
            formal_commit = commit
        elif encoded_files != formal_source_identity or commit != formal_commit:
            raise ValueError("P4 formal runs 未使用同一源码身份")
        key = f"{arm}/{seed}"
        sources[key] = {
            "arm": arm,
            "seed": seed,
            "attempt": str(attempt),
            "manifest": sf.identity(attempt / "manifest.json"),
            "formal_receipt": sf.identity(attempt / "formal_receipt.json"),
            "metrics": sf.identity(run_dir / "metrics.csv"),
            "metrics_summary": sf.identity(run_dir / "metrics_summary.csv"),
            "selected_epoch": int(receipt["selected_epoch"]),
            "completed_epochs": int(receipt["completed_epochs"]),
            "source_commit": commit,
            "manifest_file_count": len(manifest["files"]),
        }
    expected = {(arm, seed) for arm in ARMS for seed in sf.SEEDS}
    if observed != expected or set(sources) != {f"{arm}/{seed}" for arm, seed in expected}:
        raise ValueError("P4 formal 24-cell 矩阵不完整")
    metrics = pd.concat(frames, ignore_index=True)
    if len(metrics) != len(expected) * sf.COUNTS["val"]:
        raise ValueError("P4 validation metrics 总行数错误")
    return metrics, receipts, {
        "formal_source_commit": formal_commit,
        "formal_runs": sources,
    }


def run_summary() -> Path:
    lock, lock_hash = formal.load_formal_contract()
    state = sf.git_state()
    if state["status_porcelain"]:
        raise RuntimeError("P4 冻结汇总要求干净 Git 工作树")
    attempts = completed_formal_attempts()
    parent = sf.SOURCE_ROOT / sf.OUTPUT_ROOT / "summary"
    with sf.exclusive_attempt(
        parent,
        phase="summary",
        lock_hash=lock_hash,
        reject_completed=True,
    ) as output:
        metrics, receipts, sources = _audit_formal_inputs(attempts, lock, lock_hash)
        seed_metrics = sf.seed_primary_metrics(metrics)
        arm_summary = sf.arm_primary_summary(seed_metrics)
        arm_comparison = arm_reference_comparison(seed_metrics)
        conditional = sf.conditional_effects_by_seed(seed_metrics)
        factorial_by_seed = sf.factorial_effects_by_seed(seed_metrics)
        factorial_across_seed = sf.factorial_effects_across_seed(factorial_by_seed)
        local_tail = sf.local_rr_tail_summary(metrics)
        local_tail_across = local_rr_tail_across_seed(local_tail)
        subject_metrics = sf.subject_stratified_metrics(metrics)
        subject_macro_seed, subject_macro_across = subject_macro_tables(subject_metrics)
        subject_wide = subject_metrics.pivot(
            index=["arm", "seed", "samp_id"], columns="metric", values="mean"
        ).reset_index()
        subject_conditional = sf.conditional_effects_by_seed(
            subject_wide, group_columns=("seed", "samp_id")
        )
        subject_factorial = sf.subject_factorial_effects(subject_metrics)
        denominators = sf.metric_denominators(metrics)
        resources = parameter_compute_memory_table(receipts)
        decision = build_decision(
            seed_metrics,
            arm_comparison,
            conditional,
            factorial_by_seed,
            factorial_across_seed,
            resources,
        )

        outputs = {
            "seed_primary_metrics.csv": seed_metrics,
            "arm_primary_summary.csv": arm_summary,
            "arm_reference_comparison.csv": arm_comparison,
            "conditional_effects_by_seed.csv": conditional,
            "factorial_effects_by_seed.csv": factorial_by_seed,
            "factorial_effects_across_seed.csv": factorial_across_seed,
            "local_rr_tail_summary.csv": local_tail,
            "local_rr_tail_across_seed.csv": local_tail_across,
            "subject_stratified_metrics.csv": subject_metrics,
            "subject_macro_by_seed.csv": subject_macro_seed,
            "subject_macro_across_seed.csv": subject_macro_across,
            "subject_conditional_effects.csv": subject_conditional,
            "subject_factorial_effects.csv": subject_factorial,
            "metric_denominators.csv": denominators,
            "parameter_compute_memory.csv": resources,
        }
        for filename, frame in outputs.items():
            frame.to_csv(output / filename, index=False, na_rep="NA")
        sf.write_json(output / "decision.json", decision)
        sf.write_json(
            output / "source_manifest.json",
            {
                "protocol": sf.PROTOCOL,
                "implementation_lock": {
                    "path": str(formal.P2_LOCK_PATH),
                    "sha256": lock_hash,
                },
                "summary_source": _source_identity(),
                **sources,
            },
        )
        sf.write_json(output / "source_code.json", _source_identity())
        sf.write_json(
            output / "access_receipt.json",
            {
                "formal_attempts_read": len(attempts),
                "validation_metric_rows_read": len(metrics),
                "checkpoints_replayed": 48,
                "validation_only": True,
                "gpu_used": False,
                "model_training_used": False,
                "model_inference_used": False,
                "research_test_accessed": False,
            },
        )
        sf.write_json(
            output / "summary_receipt.json",
            {
                "protocol": sf.PROTOCOL,
                "status": "complete",
                "implementation_lock_sha256": lock_hash,
                "source_formal_commit": sources["formal_source_commit"],
                "summary_commit": state["commit"],
                "arms": list(ARMS),
                "seeds": list(sf.SEEDS),
                "formal_attempts": len(attempts),
                "validation_rows_per_attempt": sf.COUNTS["val"],
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


def verify_summary_attempt(path: Path, *, lock_hash: str = formal.P2_LOCK_SHA256) -> dict[str, Any]:
    path = path.resolve()
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    sf.verify_identity(path / "manifest.json", freeze["manifest"])
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != sf.PROTOCOL
        or manifest.get("phase") != "summary"
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
        or manifest.get("arm") is not None
        or manifest.get("seed") is not None
    ):
        raise ValueError("P4 summary lifecycle identity 漂移")
    if not SUMMARY_REQUIRED_FILES.issubset(manifest["files"]):
        raise ValueError("P4 summary 缺少必需产物")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path):
            raise ValueError("P4 summary manifest 路径越界")
        sf.verify_identity(target, expected)
    receipt = json.loads((path / "summary_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or receipt.get("formal_attempts") != 24
        or receipt.get("validation_metric_rows") != 24 * sf.COUNTS["val"]
    ):
        raise ValueError("P4 summary receipt 计数漂移")
    return manifest


__all__ = [
    "SUMMARY_REQUIRED_FILES",
    "SUMMARY_SOURCE_PATH",
    "arm_reference_comparison",
    "build_decision",
    "completed_formal_attempts",
    "factor_assessments",
    "local_rr_tail_across_seed",
    "parameter_compute_memory_table",
    "run_summary",
    "subject_macro_tables",
    "tolerance_pareto_arms",
    "verify_summary_attempt",
]
