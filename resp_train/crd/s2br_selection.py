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
from omegaconf import OmegaConf

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK, REPO_ROOT, sha256_file, verify_candidate_lock
from resp_train.crd.config import CRD_S2BR_PROTOCOL_VERSION, FORMAL_SEEDS
from resp_train.crd.s2_selection import (
    BASE_VARIANT,
    PAIRED_METRICS,
    REPORT_METRICS,
    _discover_single_run,
    _load_and_validate_metrics,
)
from resp_train.utils.run import save_execution_manifest


S2BR_VARIANTS = (
    "crd_205_base_em_static",
    "crd_206_base_am_static",
    "crd_207_base_cap_em",
    "crd_208_base_cap_am",
)
S2A_SINGLE_VARIANTS = (
    "crd_202_base_legacy_energy",
    "crd_203_base_analytic_am",
    "crd_204_base_morphology",
)
COMBO_SPEC = {
    "crd_205_base_em_static": {
        "control": "crd_207_base_cap_em",
        "best_constituent": "crd_202_base_legacy_energy",
        "energy": "crd_202_base_legacy_energy",
        "morphology": "crd_204_base_morphology",
    },
    "crd_206_base_am_static": {
        "control": "crd_208_base_cap_am",
        "best_constituent": BASE_VARIANT,
        "energy": "crd_203_base_analytic_am",
        "morphology": "crd_204_base_morphology",
    },
}

DEFAULT_S2BR_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_s2br_validation_summary"
DEFAULT_S2A_SUMMARY_ROOT = REPO_ROOT / "runs/crd_v1/crd_s2a_validation_summary"
SEED_SUMMARY_FILENAME = "s2br_seed_summary.csv"
VARIANT_SUMMARY_FILENAME = "s2br_variant_summary.csv"
PAIRED_FILENAME = "s2br_paired_descriptives.csv"
FACTORIAL_SEED_FILENAME = "s2br_factorial_seed_summary.csv"
FACTORIAL_PAIRED_FILENAME = "s2br_factorial_paired_descriptives.csv"
DECISION_FILENAME = "s2br_decision.json"
MANIFEST_FILENAME = "s2br_summary_manifest.json"


def summarize_crd_s2br(
    *,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    runs_root: str | Path = REPO_ROOT / "runs/crd_v1",
    s2a_summary_root: str | Path = DEFAULT_S2A_SUMMARY_ROOT,
    output_root: str | Path = DEFAULT_S2BR_OUTPUT_ROOT,
) -> Path:
    """审计 S2B-R formal 集合并应用附件第 22.3 节冻结规则。"""

    _assert_clean_repository()
    _validate_s2a_frozen(Path(s2a_summary_root))
    verification = verify_candidate_lock(candidate_lock_path)
    runs_root = Path(runs_root)
    metrics_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    seed_rows: list[dict[str, Any]] = []

    base_records = [record for record in verification.records if record["variant"] == BASE_VARIANT]
    if len(base_records) != 3:
        raise RuntimeError("candidate lock 中 CRD_102 BASE seed 集不完整")
    for record in sorted(base_records, key=lambda item: int(item["seed"])):
        seed = int(record["seed"])
        run_dir = Path(record["checkpoint_path"]).parent
        metrics, summary = _load_and_validate_metrics(run_dir, BASE_VARIANT)
        metrics_by_key[(BASE_VARIANT, seed)] = metrics
        seed_rows.append(
            {
                "role": "base",
                "variant": BASE_VARIANT,
                "seed": seed,
                "selected_epoch": int(record["selected_epoch"]),
                "checkpoint_path": str(record["checkpoint_path"]),
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
                **summary,
            }
        )

    for variant in S2A_SINGLE_VARIANTS:
        for seed in FORMAL_SEEDS:
            run_dir = _discover_single_run(runs_root, variant, seed)
            metrics, summary = _load_and_validate_metrics(run_dir, variant)
            checkpoint_path = run_dir / "checkpoint_best_local_rr.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            metrics_by_key[(variant, seed)] = metrics
            seed_rows.append(
                {
                    "role": "single",
                    "variant": variant,
                    "seed": seed,
                    "selected_epoch": int(checkpoint["epoch"]),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    **summary,
                }
            )

    training_commits: set[str] = set()
    for variant in S2BR_VARIANTS:
        for seed in FORMAL_SEEDS:
            run_dir = _discover_single_run(runs_root, variant, seed)
            audit, metrics, summary = _audit_s2br_run(run_dir, variant=variant, seed=seed)
            training_commits.add(str(audit["training_commit"]))
            metrics_by_key[(variant, seed)] = metrics
            seed_rows.append({"role": "combo" if variant in COMBO_SPEC else "control", **audit, **summary})
    if len(training_commits) != 1:
        raise RuntimeError(f"S2B-R training commit 不统一: {sorted(training_commits)}")

    seed_summary = pd.DataFrame(seed_rows)
    variant_summary, decision = apply_s2br_decision(seed_summary)
    paired = _combo_paired_descriptives(metrics_by_key)
    factorial_seed, factorial_paired = _factorial_descriptives(metrics_by_key)

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"S2B-R summary 产物禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root.parent / f".{output_root.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        seed_summary.to_csv(temporary_dir / SEED_SUMMARY_FILENAME, index=False)
        variant_summary.to_csv(temporary_dir / VARIANT_SUMMARY_FILENAME, index=False)
        paired.to_csv(temporary_dir / PAIRED_FILENAME, index=False)
        factorial_seed.to_csv(temporary_dir / FACTORIAL_SEED_FILENAME, index=False)
        factorial_paired.to_csv(temporary_dir / FACTORIAL_PAIRED_FILENAME, index=False)
        (temporary_dir / DECISION_FILENAME).write_text(
            json.dumps(
                {
                    "protocol": CRD_S2BR_PROTOCOL_VERSION,
                    "candidate_lock": str(verification.lock_path),
                    "candidate_lock_sha256": verification.lock_sha256,
                    "training_commit": next(iter(training_commits)),
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
            task="crd_v1_s2br_validation_selection",
            phase="s2br_frozen_validation_selection",
            protocol=CRD_S2BR_PROTOCOL_VERSION,
            candidate_lock=str(verification.lock_path),
            candidate_lock_sha256=verification.lock_sha256,
            complete_checkpoint_count=12,
            training_commit=next(iter(training_commits)),
            passing_combinations=decision["passing_combinations"],
            selected_model=decision["selected_model"],
            s3_activated=decision["s3_activated"],
        )
        os.replace(temporary_dir, output_root)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_root / DECISION_FILENAME


def apply_s2br_decision(seed_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_variants = {BASE_VARIANT, *S2A_SINGLE_VARIANTS, *S2BR_VARIANTS}
    required = {"variant", "seed", *REPORT_METRICS}
    missing = sorted(required - set(seed_summary.columns))
    if missing:
        raise ValueError(f"S2B-R seed summary 缺少字段: {missing}")
    if set(seed_summary["variant"].astype(str)) != expected_variants or len(seed_summary) != 24:
        raise ValueError("S2B-R decision 要求 BASE、三个 singles、四个 S2B-R 各三个 seed")
    for variant in expected_variants:
        rows = seed_summary.loc[seed_summary["variant"].eq(variant)]
        if len(rows) != 3 or set(pd.to_numeric(rows["seed"]).astype(int)) != set(FORMAL_SEEDS):
            raise ValueError(f"S2B-R {variant} seed 集不完整")

    variant_rows: list[dict[str, Any]] = []
    for variant in (BASE_VARIANT, *S2A_SINGLE_VARIANTS, *S2BR_VARIANTS):
        rows = seed_summary.loc[seed_summary["variant"].eq(variant)].sort_values("seed")
        row: dict[str, Any] = {"variant": variant}
        for metric in REPORT_METRICS:
            values = pd.to_numeric(rows[metric], errors="coerce")
            row[f"{metric}_seed_mean"] = float(values.mean())
            row[f"{metric}_seed_sd"] = float(values.std(ddof=1))
        variant_rows.append(row)
    variant_summary = pd.DataFrame(variant_rows)

    eligibility = {
        combo: _combo_eligibility(seed_summary, combo, spec)
        for combo, spec in COMBO_SPEC.items()
    }
    passing = [combo for combo in COMBO_SPEC if eligibility[combo]["eligible"]]
    comparison: dict[str, Any]
    if len(passing) == 2:
        comparison = _am_replaces_em(seed_summary)
        selected_model = (
            "crd_206_base_am_static" if comparison["am_replaces_em"] else "crd_205_base_em_static"
        )
    elif len(passing) == 1:
        comparison = {"applicable": False, "reason": "only_one_combination_passed"}
        selected_model = passing[0]
    else:
        comparison = {"applicable": False, "reason": "no_combination_passed"}
        selected_model = BASE_VARIANT

    for combo, details in eligibility.items():
        variant_summary.loc[variant_summary["variant"].eq(combo), "eligible"] = bool(details["eligible"])
    return variant_summary, {
        "complete_checkpoint_count": 12,
        "combination_eligibility": eligibility,
        "passing_combinations": passing,
        "combination_comparison": comparison,
        "selected_model": selected_model,
        "outcome": "open_s3" if passing else "retain_base",
        "s3_activated": bool(passing),
        "research_test_used": False,
        "confirmatory_p_values_used": False,
        "result_informed_exploratory": True,
    }


def _combo_eligibility(
    seed_summary: pd.DataFrame,
    combo: str,
    spec: dict[str, str],
) -> dict[str, Any]:
    local = "local_rr_mae_bpm_mean"
    base_mean = _mean(seed_summary, BASE_VARIANT, local)
    combo_mean = _mean(seed_summary, combo, local)
    control_mean = _mean(seed_summary, spec["control"], local)
    best_mean = _mean(seed_summary, spec["best_constituent"], local)
    base_improvement = _relative_improvement(base_mean, combo_mean)
    control_improvement = _relative_improvement(control_mean, combo_mean)
    best_improvement = _relative_improvement(best_mean, combo_mean)
    base_paired = _paired_seed_count(seed_summary, combo, BASE_VARIANT, local, "minimize")
    control_paired = _paired_seed_count(seed_summary, combo, spec["control"], local, "minimize")
    best_paired = _paired_seed_count(seed_summary, combo, spec["best_constituent"], local, "minimize")
    pcc_drop = _mean(seed_summary, BASE_VARIANT, "lag_aware_signed_pcc_mean") - _mean(
        seed_summary, combo, "lag_aware_signed_pcc_mean"
    )
    trajectory_worsening = _relative_worsening(
        _mean(seed_summary, BASE_VARIANT, "envelope_trajectory_mae_mean"),
        _mean(seed_summary, combo, "envelope_trajectory_mae_mean"),
    )
    coverage_drop = _mean(seed_summary, BASE_VARIANT, "ibi_coverage_mean") - _mean(
        seed_summary, combo, "ibi_coverage_mean"
    )
    checks = {
        "base_local_improvement_pass": base_improvement >= 0.005,
        "base_paired_seed_pass": base_paired >= 2,
        "control_local_improvement_pass": control_improvement >= 0.0025,
        "control_paired_seed_pass": control_paired >= 2,
        "best_constituent_local_improvement_pass": best_improvement >= 0.0025,
        "best_constituent_paired_seed_pass": best_paired >= 2,
        "signed_pcc_drop_pass": pcc_drop <= 0.003,
        "trajectory_worsening_pass": trajectory_worsening <= 0.015,
        "ibi_coverage_drop_pass": coverage_drop <= 0.01,
    }
    return {
        "eligible": bool(all(checks.values())),
        "control_variant": spec["control"],
        "best_constituent_variant": spec["best_constituent"],
        "base_local_relative_improvement": base_improvement,
        "base_local_paired_seed_improvement_count": base_paired,
        "control_local_relative_improvement": control_improvement,
        "control_local_paired_seed_improvement_count": control_paired,
        "best_constituent_local_relative_improvement": best_improvement,
        "best_constituent_local_paired_seed_improvement_count": best_paired,
        "signed_pcc_absolute_drop_vs_base": pcc_drop,
        "trajectory_relative_worsening_vs_base": trajectory_worsening,
        "ibi_coverage_absolute_drop_vs_base": coverage_drop,
        **checks,
    }


def _am_replaces_em(seed_summary: pd.DataFrame) -> dict[str, Any]:
    em = "crd_205_base_em_static"
    am = "crd_206_base_am_static"
    local = "local_rr_mae_bpm_mean"
    improvement = _relative_improvement(_mean(seed_summary, em, local), _mean(seed_summary, am, local))
    paired = _paired_seed_count(seed_summary, am, em, local, "minimize")
    pcc_drop = _mean(seed_summary, em, "lag_aware_signed_pcc_mean") - _mean(
        seed_summary, am, "lag_aware_signed_pcc_mean"
    )
    trajectory_worsening = _relative_worsening(
        _mean(seed_summary, em, "envelope_trajectory_mae_mean"),
        _mean(seed_summary, am, "envelope_trajectory_mae_mean"),
    )
    coverage_drop = _mean(seed_summary, em, "ibi_coverage_mean") - _mean(
        seed_summary, am, "ibi_coverage_mean"
    )
    checks = {
        "local_improvement_pass": improvement >= 0.0025,
        "paired_seed_pass": paired >= 2,
        "signed_pcc_drop_pass": pcc_drop <= 0.003,
        "trajectory_worsening_pass": trajectory_worsening <= 0.015,
        "ibi_coverage_drop_pass": coverage_drop <= 0.01,
    }
    return {
        "applicable": True,
        "am_replaces_em": bool(all(checks.values())),
        "local_relative_improvement": improvement,
        "local_paired_seed_improvement_count": paired,
        "signed_pcc_absolute_drop": pcc_drop,
        "trajectory_relative_worsening": trajectory_worsening,
        "ibi_coverage_absolute_drop": coverage_drop,
        **checks,
    }


def _audit_s2br_run(
    run_dir: Path,
    *,
    variant: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    required_files = (
        "config.yaml",
        "run_manifest.json",
        "train_history.csv",
        "checkpoint_best_local_rr.pt",
        "checkpoint_final.pt",
        "metrics.csv",
        "metrics_summary.csv",
        "runtime_summary.json",
        "optimizer_parameter_groups.json",
    )
    missing = [name for name in required_files if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"S2B-R run 缺失产物 {run_dir}: {missing}")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    expected = {
        "protocol.name": CRD_S2BR_PROTOCOL_VERSION,
        "protocol.stage": "s2br",
        "protocol.run_role": "formal",
        "model.variant": variant,
        "training.seed": seed,
        "model.initialization_seed": seed,
        "training.epochs": 80,
        "training.batch_size": 128,
        "training.gradient_accumulation_steps": 1,
    }
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise RuntimeError(f"S2B-R {run_dir} config {key} 不一致")
    for key in ("data.max_train_windows", "data.max_val_windows", "data.max_test_windows"):
        if OmegaConf.select(cfg, key) is not None:
            raise RuntimeError(f"S2B-R formal {key} 必须为 null")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("git_dirty") is not False or manifest.get("run_role") != "formal":
        raise RuntimeError(f"S2B-R manifest 不干净或非 formal: {run_dir}")
    history = pd.read_csv(run_dir / "train_history.csv")
    if len(history) != 80 or not np.array_equal(history["epoch"].to_numpy(), np.arange(1, 81)):
        raise RuntimeError(f"S2B-R history epoch 不完整: {run_dir}")
    if not np.array_equal(history["optimizer_update"].to_numpy(), np.arange(1, 81) * 80):
        raise RuntimeError(f"S2B-R optimizer update 不满足每 epoch 80: {run_dir}")
    if not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        raise FloatingPointError(f"S2B-R history 包含非有限值: {run_dir}")
    has_proto = "train_loss_proto" in history.columns
    if has_proto != (variant in COMBO_SPEC):
        raise RuntimeError(f"S2B-R prototype regularizer provenance 与 variant 不一致: {run_dir}")
    best_path = run_dir / "checkpoint_best_local_rr.pt"
    best = torch.load(best_path, map_location="cpu")
    final = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu")
    if int(final["epoch"]) != 80 or not (1 <= int(best["epoch"]) <= 80):
        raise RuntimeError(f"S2B-R checkpoint epoch 异常: {run_dir}")
    for name, checkpoint in (("best", best), ("final", final)):
        if not all(
            bool(torch.isfinite(value).all())
            for value in checkpoint["model_state_dict"].values()
            if torch.is_tensor(value)
        ):
            raise FloatingPointError(f"S2B-R {name} checkpoint 包含 NaN/Inf: {run_dir}")
    metrics, summary = _load_and_validate_metrics(run_dir, variant)
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    if not (0.0 < float(runtime["peak_reserved_fraction"]) <= 0.80):
        raise RuntimeError(f"S2B-R formal peak reserved fraction 越界: {run_dir}")
    return (
        {
            "variant": variant,
            "seed": seed,
            "selected_epoch": int(best["epoch"]),
            "checkpoint_path": str(best_path),
            "checkpoint_sha256": sha256_file(best_path),
            "run_dir": str(run_dir),
            "training_commit": str(manifest.get("git_commit")),
            "peak_reserved_fraction": float(runtime["peak_reserved_fraction"]),
        },
        metrics,
        summary,
    )


def _combo_paired_descriptives(metrics_by_key: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for combo, spec in COMBO_SPEC.items():
        comparators = (BASE_VARIANT, spec["control"], spec["best_constituent"])
        for comparator in dict.fromkeys(comparators):
            for seed in FORMAL_SEEDS:
                candidate = metrics_by_key[(combo, seed)]
                reference = metrics_by_key[(comparator, seed)]
                _assert_paired_identity(candidate, reference, combo, comparator, seed)
                for metric, direction in PAIRED_METRICS.items():
                    rows.extend(
                        _difference_rows(
                            candidate,
                            reference,
                            variant=combo,
                            comparator=comparator,
                            seed=seed,
                            metric=metric,
                            direction=direction,
                        )
                    )
    return pd.DataFrame(rows)


def _factorial_descriptives(
    metrics_by_key: dict[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for combo, spec in COMBO_SPEC.items():
        for seed in FORMAL_SEEDS:
            frames = {
                "combo": metrics_by_key[(combo, seed)],
                "energy": metrics_by_key[(spec["energy"], seed)],
                "morphology": metrics_by_key[(spec["morphology"], seed)],
                "base": metrics_by_key[(BASE_VARIANT, seed)],
            }
            for name, frame in frames.items():
                if name != "base":
                    _assert_paired_identity(frame, frames["base"], combo, name, seed)
            summary_row: dict[str, Any] = {"variant": combo, "seed": seed}
            for metric, direction in PAIRED_METRICS.items():
                for unit in ("window", "samp_id"):
                    arrays = {
                        name: _metric_array(frame, metric, unit)
                        for name, frame in frames.items()
                    }
                    finite = np.logical_and.reduce([np.isfinite(values) for values in arrays.values()])
                    contrast = (
                        arrays["combo"][finite]
                        - arrays["energy"][finite]
                        - arrays["morphology"][finite]
                        + arrays["base"][finite]
                    )
                    beneficial = contrast < 0.0 if direction == "minimize" else contrast > 0.0
                    paired_rows.append(
                        {
                            "variant": combo,
                            "seed": seed,
                            "metric": metric,
                            "direction": direction,
                            "unit": unit,
                            "n_pairs": int(len(contrast)),
                            "factorial_interaction_mean": float(np.mean(contrast)) if len(contrast) else math.nan,
                            "factorial_interaction_median": float(np.median(contrast)) if len(contrast) else math.nan,
                            "factorial_interaction_sd": (
                                float(np.std(contrast, ddof=1)) if len(contrast) > 1 else math.nan
                            ),
                            "beneficial_count": int(beneficial.sum()),
                            "beneficial_fraction": float(beneficial.mean()) if len(beneficial) else math.nan,
                        }
                    )
                    if unit == "window":
                        summary_row[f"{metric}_factorial_interaction_mean"] = (
                            float(np.mean(contrast)) if len(contrast) else math.nan
                        )
            seed_rows.append(summary_row)
    return pd.DataFrame(seed_rows), pd.DataFrame(paired_rows)


def _difference_rows(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    variant: str,
    comparator: str,
    seed: int,
    metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in ("window", "samp_id"):
        left = _metric_array(reference, metric, unit)
        right = _metric_array(candidate, metric, unit)
        finite = np.isfinite(left) & np.isfinite(right)
        differences = right[finite] - left[finite]
        improved = differences < 0.0 if direction == "minimize" else differences > 0.0
        rows.append(
            {
                "variant": variant,
                "comparator": comparator,
                "seed": seed,
                "metric": metric,
                "direction": direction,
                "unit": unit,
                "n_pairs": int(len(differences)),
                "candidate_minus_comparator_mean": float(np.mean(differences)) if len(differences) else math.nan,
                "candidate_minus_comparator_median": float(np.median(differences)) if len(differences) else math.nan,
                "candidate_minus_comparator_sd": (
                    float(np.std(differences, ddof=1)) if len(differences) > 1 else math.nan
                ),
                "improved_count": int(improved.sum()),
                "improved_fraction": float(improved.mean()) if len(improved) else math.nan,
            }
        )
    return rows


def _metric_array(frame: pd.DataFrame, metric: str, unit: str) -> np.ndarray:
    if unit == "window":
        return pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=np.float64)
    if unit == "samp_id":
        return frame.groupby("samp_id", sort=True)[metric].mean().to_numpy(dtype=np.float64)
    raise ValueError(f"未知 paired unit={unit}")


def _assert_paired_identity(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    seed: int,
) -> None:
    identity = ["dataset_row_id", "samp_id", "coupling_state_id"]
    if not left[identity].equals(right[identity]):
        raise RuntimeError(f"{left_name} vs {right_name}/seed_{seed} validation identity 未逐行配对")


def _validate_s2a_frozen(root: Path) -> None:
    decision_path = root / "s2a_decision.json"
    manifest_path = root / "s2a_summary_manifest.json"
    if not decision_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("S2B-R summary 要求已冻结的 S2A decision/manifest")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        decision.get("selected_model_after_s2a") != BASE_VARIANT
        or decision.get("s2b_activated") is not False
        or manifest.get("git_dirty") is not False
    ):
        raise RuntimeError("S2A frozen identity 不符合 S2B-R result-informed 起点")


def _mean(seed_summary: pd.DataFrame, variant: str, metric: str) -> float:
    rows = seed_summary.loc[seed_summary["variant"].eq(variant), metric]
    return float(pd.to_numeric(rows).mean())


def _paired_seed_count(
    seed_summary: pd.DataFrame,
    candidate: str,
    comparator: str,
    metric: str,
    direction: str,
) -> int:
    left = seed_summary.loc[seed_summary["variant"].eq(comparator), ["seed", metric]].set_index("seed")[metric]
    right = seed_summary.loc[seed_summary["variant"].eq(candidate), ["seed", metric]].set_index("seed")[metric]
    seeds = sorted(FORMAL_SEEDS)
    if direction == "minimize":
        return int((right.loc[seeds] < left.loc[seeds]).sum())
    if direction == "maximize":
        return int((right.loc[seeds] > left.loc[seeds]).sum())
    raise ValueError(f"未知方向: {direction}")


def _relative_improvement(comparator: float, candidate: float) -> float:
    if comparator <= 0.0:
        raise ValueError("相对改善 comparator 必须为正")
    return (comparator - candidate) / comparator


def _relative_worsening(comparator: float, candidate: float) -> float:
    if comparator <= 0.0:
        raise ValueError("相对恶化 comparator 必须为正")
    return (candidate - comparator) / comparator


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(status.stderr.strip() or "无法确认 Git 工作树状态")
    if status.stdout.strip():
        raise RuntimeError("S2B-R frozen summary 要求干净 Git 工作树")


__all__ = ["apply_s2br_decision", "summarize_crd_s2br"]
