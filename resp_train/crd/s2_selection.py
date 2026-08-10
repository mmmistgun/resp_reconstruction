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
from resp_train.crd.config import CRD_S2A_PROTOCOL_VERSION, FORMAL_SEEDS
from resp_train.crd.prototype_diagnostics import (
    DEFAULT_PROTOTYPE_OUTPUT_ROOT,
    MANIFEST_FILENAME as PROTOTYPE_MANIFEST_FILENAME,
    SAMP_FILENAME as PROTOTYPE_SAMP_FILENAME,
    SUMMARY_FILENAME as PROTOTYPE_SUMMARY_FILENAME,
    WINDOW_FILENAME as PROTOTYPE_WINDOW_FILENAME,
    summarize_prototype_frame,
)
from resp_train.metrics.task import summarize_task_metrics
from resp_train.utils.run import save_execution_manifest


S2_VARIANTS = (
    "crd_202_base_legacy_energy",
    "crd_203_base_analytic_am",
    "crd_204_base_morphology",
)
BASE_VARIANT = "crd_102_b0_local_mamba"
DEFAULT_S2_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_s2a_validation_summary"

SEED_SUMMARY_FILENAME = "s2a_seed_summary.csv"
VARIANT_SUMMARY_FILENAME = "s2a_variant_summary.csv"
PAIRED_FILENAME = "s2a_paired_descriptives.csv"
PROTOTYPE_SEED_FILENAME = "s2a_prototype_seed_summary.csv"
DECISION_FILENAME = "s2a_decision.json"
MANIFEST_FILENAME = "s2a_summary_manifest.json"

REPORT_METRICS = (
    "whole_rr_abs_error_bpm_mean",
    "local_rr_mae_bpm_mean",
    "local_rr_prediction_valid_fraction_mean",
    "envelope_trajectory_mae_mean",
    "global_envelope_modulation_error_mean",
    "lag_aware_signed_pcc_mean",
    "ibi_medae_sec_mean",
    "ibi_coverage_mean",
    "ibi_interpretable_fraction",
    "target_stratified_envelope_spearman_low_mean",
    "target_stratified_envelope_spearman_medium_mean",
    "target_stratified_envelope_spearman_high_mean",
    "joint_prediction_degenerate_fraction",
)

PAIRED_METRICS = {
    "whole_rr_abs_error_bpm": "minimize",
    "local_rr_mae_bpm": "minimize",
    "envelope_trajectory_mae": "minimize",
    "global_envelope_modulation_error": "minimize",
    "lag_aware_signed_pcc": "maximize",
    "ibi_medae_sec": "minimize",
    "ibi_coverage": "maximize",
    "target_stratified_envelope_spearman": "maximize",
}


def summarize_crd_s2a(
    *,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    runs_root: str | Path = REPO_ROOT / "runs/crd_v1",
    prototype_root: str | Path = DEFAULT_PROTOTYPE_OUTPUT_ROOT,
    output_root: str | Path = DEFAULT_S2_OUTPUT_ROOT,
) -> Path:
    """审计 S2A formal/diagnostic 集合并应用附件第 21.4–21.7 节冻结规则。"""

    _assert_clean_repository()
    verification = verify_candidate_lock(candidate_lock_path)
    base_records = [record for record in verification.records if record["variant"] == BASE_VARIANT]
    if len(base_records) != len(FORMAL_SEEDS):
        raise RuntimeError("candidate lock 中 CRD_102 BASE seed 集不完整")

    runs_root = Path(runs_root)
    metrics_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    seed_rows: list[dict[str, Any]] = []
    checkpoint_by_seed: dict[int, Path] = {}
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

    training_commits: set[str] = set()
    for variant in S2_VARIANTS:
        for seed in FORMAL_SEEDS:
            run_dir = _discover_single_run(runs_root, variant, seed)
            audit, metrics, summary = _audit_s2_run(run_dir, variant=variant, seed=seed)
            training_commits.add(str(audit["training_commit"]))
            metrics_by_key[(variant, seed)] = metrics
            if variant == "crd_204_base_morphology":
                checkpoint_by_seed[seed] = Path(audit["checkpoint_path"])
            seed_rows.append({"role": "candidate", **audit, **summary})
    if len(training_commits) != 1:
        raise RuntimeError(f"S2A training commit 不统一: {sorted(training_commits)}")

    seed_summary = pd.DataFrame(seed_rows)
    variant_summary, decision = apply_s2a_decision(seed_summary)
    paired = _paired_descriptives(metrics_by_key)
    prototype_seed = _load_prototype_diagnostics(
        Path(prototype_root),
        checkpoint_by_seed=checkpoint_by_seed,
    )

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"S2A summary 产物禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root.parent / f".{output_root.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        seed_summary.to_csv(temporary_dir / SEED_SUMMARY_FILENAME, index=False)
        variant_summary.to_csv(temporary_dir / VARIANT_SUMMARY_FILENAME, index=False)
        paired.to_csv(temporary_dir / PAIRED_FILENAME, index=False)
        prototype_seed.to_csv(temporary_dir / PROTOTYPE_SEED_FILENAME, index=False)
        (temporary_dir / DECISION_FILENAME).write_text(
            json.dumps(
                {
                    "protocol": CRD_S2A_PROTOCOL_VERSION,
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
            task="crd_v1_s2a_validation_selection",
            phase="s2a_frozen_validation_selection",
            protocol=CRD_S2A_PROTOCOL_VERSION,
            candidate_lock=str(verification.lock_path),
            candidate_lock_sha256=verification.lock_sha256,
            complete_checkpoint_count=9,
            training_commit=next(iter(training_commits)),
            selected_energy=decision["energy_selection"]["selected_energy"],
            morphology_eligible=decision["morphology_eligibility"]["eligible"],
            s2b_activated=decision["s2b_activated"],
        )
        os.replace(temporary_dir, output_root)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_root / DECISION_FILENAME


def apply_s2a_decision(seed_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"variant", "seed", *REPORT_METRICS}
    missing = sorted(required - set(seed_summary.columns))
    if missing:
        raise ValueError(f"S2A seed summary 缺少字段: {missing}")
    expected_variants = {BASE_VARIANT, *S2_VARIANTS}
    if set(seed_summary["variant"].astype(str)) != expected_variants or len(seed_summary) != 12:
        raise ValueError("S2A decision 要求 BASE+202/203/204 各三个固定 seed")
    expected_seeds = set(FORMAL_SEEDS)
    for variant in expected_variants:
        rows = seed_summary.loc[seed_summary["variant"].eq(variant)]
        if set(pd.to_numeric(rows["seed"]).astype(int)) != expected_seeds or len(rows) != 3:
            raise ValueError(f"S2A {variant} seed 集不完整")

    variant_rows: list[dict[str, Any]] = []
    for variant in (BASE_VARIANT, *S2_VARIANTS):
        rows = seed_summary.loc[seed_summary["variant"].eq(variant)].sort_values("seed")
        row: dict[str, Any] = {"variant": variant}
        for metric in REPORT_METRICS:
            values = pd.to_numeric(rows[metric], errors="coerce")
            row[f"{metric}_seed_mean"] = float(values.mean())
            row[f"{metric}_seed_sd"] = float(values.std(ddof=1))
        variant_rows.append(row)
    variant_summary = pd.DataFrame(variant_rows)

    energy_eligibility = {
        variant: _energy_eligibility(seed_summary, variant)
        for variant in ("crd_202_base_legacy_energy", "crd_203_base_analytic_am")
    }
    e_ok = bool(energy_eligibility["crd_202_base_legacy_energy"]["eligible"])
    a_ok = bool(energy_eligibility["crd_203_base_analytic_am"]["eligible"])
    energy_comparison: dict[str, Any]
    if e_ok and a_ok:
        energy_comparison = _energy_pair_comparison(seed_summary)
        selected_energy = "A" if energy_comparison["a_replaces_e"] else "E"
    elif e_ok:
        energy_comparison = {"applicable": False, "reason": "only_E_eligible"}
        selected_energy = "E"
    elif a_ok:
        energy_comparison = {"applicable": False, "reason": "only_A_eligible"}
        selected_energy = "A"
    else:
        energy_comparison = {"applicable": False, "reason": "neither_eligible"}
        selected_energy = "none"

    morphology = _morphology_eligibility(seed_summary)
    morphology_ok = bool(morphology["eligible"])
    s2b_activated = selected_energy != "none" and morphology_ok
    if selected_energy == "none" and not morphology_ok:
        selected_model = BASE_VARIANT
        outcome = "retain_base"
    elif selected_energy != "none" and not morphology_ok:
        selected_model = (
            "crd_202_base_legacy_energy" if selected_energy == "E" else "crd_203_base_analytic_am"
        )
        outcome = "select_energy_only"
    elif selected_energy == "none" and morphology_ok:
        selected_model = "crd_204_base_morphology"
        outcome = "select_morphology_only"
    else:
        selected_model = None
        outcome = "s2b_required_before_final_selection"

    for variant, details in energy_eligibility.items():
        mask = variant_summary["variant"].eq(variant)
        variant_summary.loc[mask, "energy_eligible"] = bool(details["eligible"])
    mask = variant_summary["variant"].eq("crd_204_base_morphology")
    variant_summary.loc[mask, "morphology_eligible"] = morphology_ok
    return variant_summary, {
        "complete_checkpoint_count": 9,
        "energy_eligibility": energy_eligibility,
        "energy_selection": {
            "selected_energy": selected_energy,
            "comparison": energy_comparison,
        },
        "morphology_eligibility": morphology,
        "s2b_activated": s2b_activated,
        "s3_activated": False,
        "selected_model_after_s2a": selected_model,
        "outcome": outcome,
        "research_test_used": False,
        "confirmatory_p_values_used": False,
    }


def _energy_eligibility(seed_summary: pd.DataFrame, variant: str) -> dict[str, Any]:
    base = _variant_mean(seed_summary, BASE_VARIANT)
    candidate = _variant_mean(seed_summary, variant)
    trajectory_improvement = _relative_improvement(
        base["envelope_trajectory_mae_mean"], candidate["envelope_trajectory_mae_mean"]
    )
    local_worsening = _relative_worsening(base["local_rr_mae_bpm_mean"], candidate["local_rr_mae_bpm_mean"])
    pcc_drop = base["lag_aware_signed_pcc_mean"] - candidate["lag_aware_signed_pcc_mean"]
    global_worsening = _relative_worsening(
        base["global_envelope_modulation_error_mean"],
        candidate["global_envelope_modulation_error_mean"],
    )
    paired_count = _paired_seed_count(
        seed_summary,
        variant,
        "envelope_trajectory_mae_mean",
        direction="minimize",
    )
    checks = {
        "trajectory_improvement_pass": trajectory_improvement >= 0.01,
        "trajectory_paired_seed_pass": paired_count >= 2,
        "local_rr_worsening_pass": local_worsening <= 0.01,
        "signed_pcc_drop_pass": pcc_drop <= 0.003,
        "global_envelope_worsening_pass": global_worsening <= 0.015,
    }
    return {
        "eligible": bool(all(checks.values())),
        "trajectory_relative_improvement": trajectory_improvement,
        "trajectory_paired_seed_improvement_count": paired_count,
        "local_rr_relative_worsening": local_worsening,
        "signed_pcc_absolute_drop": pcc_drop,
        "global_envelope_relative_worsening": global_worsening,
        **checks,
    }


def _energy_pair_comparison(seed_summary: pd.DataFrame) -> dict[str, Any]:
    e_variant = "crd_202_base_legacy_energy"
    a_variant = "crd_203_base_analytic_am"
    energy = _variant_mean(seed_summary, e_variant)
    analytic = _variant_mean(seed_summary, a_variant)
    trajectory_improvement = _relative_improvement(
        energy["envelope_trajectory_mae_mean"], analytic["envelope_trajectory_mae_mean"]
    )
    paired_count = _paired_seed_count(
        seed_summary,
        a_variant,
        "envelope_trajectory_mae_mean",
        direction="minimize",
        comparator=e_variant,
    )
    local_worsening = _relative_worsening(energy["local_rr_mae_bpm_mean"], analytic["local_rr_mae_bpm_mean"])
    pcc_drop = energy["lag_aware_signed_pcc_mean"] - analytic["lag_aware_signed_pcc_mean"]
    global_worsening = _relative_worsening(
        energy["global_envelope_modulation_error_mean"],
        analytic["global_envelope_modulation_error_mean"],
    )
    checks = {
        "trajectory_improvement_pass": trajectory_improvement >= 0.01,
        "trajectory_paired_seed_pass": paired_count >= 2,
        "local_rr_worsening_pass": local_worsening <= 0.01,
        "signed_pcc_drop_pass": pcc_drop <= 0.003,
        "global_envelope_worsening_pass": global_worsening <= 0.015,
    }
    return {
        "applicable": True,
        "a_replaces_e": bool(all(checks.values())),
        "trajectory_relative_improvement": trajectory_improvement,
        "trajectory_paired_seed_improvement_count": paired_count,
        "local_rr_relative_worsening": local_worsening,
        "signed_pcc_absolute_drop": pcc_drop,
        "global_envelope_relative_worsening": global_worsening,
        **checks,
    }


def _morphology_eligibility(seed_summary: pd.DataFrame) -> dict[str, Any]:
    variant = "crd_204_base_morphology"
    base = _variant_mean(seed_summary, BASE_VARIANT)
    candidate = _variant_mean(seed_summary, variant)
    pcc_increase = candidate["lag_aware_signed_pcc_mean"] - base["lag_aware_signed_pcc_mean"]
    paired_count = _paired_seed_count(
        seed_summary,
        variant,
        "lag_aware_signed_pcc_mean",
        direction="maximize",
    )
    local_worsening = _relative_worsening(base["local_rr_mae_bpm_mean"], candidate["local_rr_mae_bpm_mean"])
    coverage_drop = base["ibi_coverage_mean"] - candidate["ibi_coverage_mean"]
    trajectory_worsening = _relative_worsening(
        base["envelope_trajectory_mae_mean"], candidate["envelope_trajectory_mae_mean"]
    )
    checks = {
        "signed_pcc_increase_pass": pcc_increase >= 0.002,
        "signed_pcc_paired_seed_pass": paired_count >= 2,
        "local_rr_worsening_pass": local_worsening <= 0.01,
        "ibi_coverage_drop_pass": coverage_drop <= 0.01,
        "trajectory_worsening_pass": trajectory_worsening <= 0.015,
    }
    return {
        "eligible": bool(all(checks.values())),
        "signed_pcc_absolute_increase": pcc_increase,
        "signed_pcc_paired_seed_improvement_count": paired_count,
        "local_rr_relative_worsening": local_worsening,
        "ibi_coverage_absolute_drop": coverage_drop,
        "trajectory_relative_worsening": trajectory_worsening,
        **checks,
    }


def _discover_single_run(runs_root: Path, variant: str, seed: int) -> Path:
    manifests = sorted((runs_root / variant / f"seed_{seed}").glob("*/run_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"{variant}/seed_{seed} 要求恰好一个 formal run，实际 {len(manifests)}")
    return manifests[0].parent


def _audit_s2_run(
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
        raise FileNotFoundError(f"S2A run 缺失产物 {run_dir}: {missing}")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    expected = {
        "protocol.name": CRD_S2A_PROTOCOL_VERSION,
        "protocol.stage": "s2",
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
            raise RuntimeError(f"S2A {run_dir} config {key} 不一致")
    for key in ("data.max_train_windows", "data.max_val_windows", "data.max_test_windows"):
        if OmegaConf.select(cfg, key) is not None:
            raise RuntimeError(f"S2A formal {key} 必须为 null")

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("git_dirty") is not False or manifest.get("run_role") != "formal":
        raise RuntimeError(f"S2A manifest 不干净或非 formal: {run_dir}")
    history = pd.read_csv(run_dir / "train_history.csv")
    if len(history) != 80 or not np.array_equal(history["epoch"].to_numpy(), np.arange(1, 81)):
        raise RuntimeError(f"S2A history epoch 不完整: {run_dir}")
    if not np.array_equal(history["optimizer_update"].to_numpy(), np.arange(1, 81) * 80):
        raise RuntimeError(f"S2A optimizer update 不满足每 epoch 80: {run_dir}")
    if not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        raise FloatingPointError(f"S2A history 包含非有限值: {run_dir}")
    if variant == "crd_204_base_morphology":
        required_proto = {
            "train_loss_proto",
            "train_loss_proto_weighted",
            "prototype_regularizer_weight",
            "regularizer_ramp_first",
            "regularizer_ramp_last",
        }
        if not required_proto.issubset(history.columns):
            raise RuntimeError("CRD_204 history 缺少 prototype regularizer provenance")

    best = torch.load(run_dir / "checkpoint_best_local_rr.pt", map_location="cpu")
    final = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu")
    if int(final["epoch"]) != 80 or not (1 <= int(best["epoch"]) <= 80):
        raise RuntimeError(f"S2A checkpoint epoch 异常: {run_dir}")
    for name, checkpoint in (("best", best), ("final", final)):
        if not all(
            bool(torch.isfinite(value).all())
            for value in checkpoint["model_state_dict"].values()
            if torch.is_tensor(value)
        ):
            raise FloatingPointError(f"S2A {name} checkpoint 包含 NaN/Inf: {run_dir}")
    metrics, summary = _load_and_validate_metrics(run_dir, variant)
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    if not (0.0 < float(runtime["peak_reserved_fraction"]) <= 0.80):
        raise RuntimeError(f"S2A formal peak reserved fraction 越界: {run_dir}")
    return (
        {
            "variant": variant,
            "seed": seed,
            "selected_epoch": int(best["epoch"]),
            "checkpoint_path": str(run_dir / "checkpoint_best_local_rr.pt"),
            "checkpoint_sha256": sha256_file(run_dir / "checkpoint_best_local_rr.pt"),
            "run_dir": str(run_dir),
            "training_commit": str(manifest.get("git_commit")),
            "peak_reserved_fraction": float(runtime["peak_reserved_fraction"]),
        },
        metrics,
        summary,
    )


def _load_and_validate_metrics(run_dir: Path, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    if len(metrics) != 2675 or int(metrics["samp_id"].nunique()) != 7:
        raise RuntimeError(f"validation identity 数量异常: {run_dir}")
    if set(metrics["evaluation_split"].astype(str)) != {"validation"}:
        raise RuntimeError(f"S2A metrics 非 validation: {run_dir}")
    if set(metrics["method"].astype(str)) != {variant}:
        raise RuntimeError(f"S2A metrics method 不一致: {run_dir}")
    primary = [
        "whole_rr_abs_error_bpm",
        "local_rr_mae_bpm",
        "local_rr_prediction_valid_fraction",
        "envelope_trajectory_mae",
        "global_envelope_modulation_error",
        "lag_aware_signed_pcc",
        "ibi_coverage",
    ]
    if not np.isfinite(metrics[primary].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError(f"S2A eligible primary metrics 包含 NaN/Inf: {run_dir}")
    recomputed = summarize_task_metrics(metrics)
    stored = pd.read_csv(run_dir / "metrics_summary.csv")
    try:
        pd.testing.assert_frame_equal(
            stored,
            recomputed,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(f"S2A metrics_summary 与逐 sample 重算不一致: {run_dir}") from exc
    summary = recomputed.iloc[0].to_dict()
    missing = sorted(set(REPORT_METRICS) - set(summary))
    if missing:
        raise RuntimeError(f"S2A summary 缺少报告指标: {missing}")
    return metrics, summary


def _paired_descriptives(metrics_by_key: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in S2_VARIANTS:
        for seed in FORMAL_SEEDS:
            base = metrics_by_key[(BASE_VARIANT, seed)]
            candidate = metrics_by_key[(variant, seed)]
            identity = ["dataset_row_id", "samp_id", "coupling_state_id"]
            if not base[identity].equals(candidate[identity]):
                raise RuntimeError(f"{variant}/seed_{seed} 与 BASE validation identity 未逐行配对")
            for metric, direction in PAIRED_METRICS.items():
                for unit in ("window", "samp_id"):
                    if unit == "window":
                        left = pd.to_numeric(base[metric], errors="coerce").to_numpy(dtype=np.float64)
                        right = pd.to_numeric(candidate[metric], errors="coerce").to_numpy(dtype=np.float64)
                    else:
                        left = base.groupby("samp_id", sort=True)[metric].mean().to_numpy(dtype=np.float64)
                        right = candidate.groupby("samp_id", sort=True)[metric].mean().to_numpy(dtype=np.float64)
                    finite = np.isfinite(left) & np.isfinite(right)
                    differences = right[finite] - left[finite]
                    improved = differences < 0.0 if direction == "minimize" else differences > 0.0
                    rows.append(
                        {
                            "variant": variant,
                            "comparator": BASE_VARIANT,
                            "seed": seed,
                            "metric": metric,
                            "direction": direction,
                            "unit": unit,
                            "n_pairs": int(len(differences)),
                            "candidate_minus_base_mean": float(np.mean(differences)) if len(differences) else math.nan,
                            "candidate_minus_base_median": float(np.median(differences)) if len(differences) else math.nan,
                            "candidate_minus_base_sd": (
                                float(np.std(differences, ddof=1)) if len(differences) > 1 else math.nan
                            ),
                            "improved_count": int(improved.sum()),
                            "improved_fraction": float(improved.mean()) if len(improved) else math.nan,
                        }
                    )
    return pd.DataFrame(rows)


def _load_prototype_diagnostics(
    prototype_root: Path,
    *,
    checkpoint_by_seed: dict[int, Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        directory = prototype_root / "crd_204_base_morphology" / f"seed_{seed}"
        paths = {
            "window": directory / PROTOTYPE_WINDOW_FILENAME,
            "samp": directory / PROTOTYPE_SAMP_FILENAME,
            "summary": directory / PROTOTYPE_SUMMARY_FILENAME,
            "manifest": directory / PROTOTYPE_MANIFEST_FILENAME,
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("S2A prototype diagnostics 尚未齐备: " + ", ".join(missing))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        expected_checkpoint = checkpoint_by_seed[seed]
        if manifest.get("git_dirty") is not False:
            raise RuntimeError(f"prototype diagnostic manifest 必须来自干净工作树: {directory}")
        if (
            manifest.get("phase") != "validation_prototype_diagnostics"
            or manifest.get("protocol") != CRD_S2A_PROTOCOL_VERSION
            or manifest.get("evaluation_split") != "validation"
            or manifest.get("variant") != "crd_204_base_morphology"
            or int(manifest.get("seed", -1)) != seed
        ):
            raise RuntimeError(f"prototype diagnostic identity 不一致: {directory}")
        if manifest.get("checkpoint_sha256") != sha256_file(expected_checkpoint):
            raise RuntimeError(f"prototype diagnostic checkpoint hash 不一致: {directory}")
        frame = pd.read_csv(paths["window"])
        recomputed, samp = summarize_prototype_frame(frame, prototype_count=16)
        stored_samp = pd.read_csv(paths["samp"])
        if len(samp) != 7 or len(stored_samp) != 7:
            raise RuntimeError(f"prototype samp summary 数量异常: {directory}")
        try:
            pd.testing.assert_frame_equal(
                stored_samp,
                samp,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise RuntimeError(f"prototype samp summary 与 window 重算不一致: {directory}") from exc
        for key in (
            "n_windows",
            "n_samp_ids",
            "hard_usage_entropy_normalized",
            "soft_usage_entropy_normalized",
            "mean_token_entropy_normalized",
            "dominant_hard_fraction",
            "dominant_soft_fraction",
        ):
            if not math.isclose(float(summary[key]), float(recomputed[key]), rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError(f"prototype summary 重算不一致 {key}: {directory}")
        if int(summary["selected_epoch"]) != int(manifest["selected_epoch"]):
            raise RuntimeError(f"prototype selected epoch 不一致: {directory}")
        rows.append(
            {
                "variant": "crd_204_base_morphology",
                "seed": seed,
                "selected_epoch": int(summary["selected_epoch"]),
                "n_windows": int(summary["n_windows"]),
                "n_samp_ids": int(summary["n_samp_ids"]),
                "prototype_loss": float(summary["prototype_loss"]),
                "hard_usage_entropy_normalized": float(summary["hard_usage_entropy_normalized"]),
                "soft_usage_entropy_normalized": float(summary["soft_usage_entropy_normalized"]),
                "mean_token_entropy_normalized": float(summary["mean_token_entropy_normalized"]),
                "dominant_hard_prototype": int(summary["dominant_hard_prototype"]),
                "dominant_hard_fraction": float(summary["dominant_hard_fraction"]),
                "dominant_soft_prototype": int(summary["dominant_soft_prototype"]),
                "dominant_soft_fraction": float(summary["dominant_soft_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def _variant_mean(seed_summary: pd.DataFrame, variant: str) -> dict[str, float]:
    rows = seed_summary.loc[seed_summary["variant"].eq(variant)]
    return {metric: float(pd.to_numeric(rows[metric]).mean()) for metric in REPORT_METRICS}


def _paired_seed_count(
    seed_summary: pd.DataFrame,
    variant: str,
    metric: str,
    *,
    direction: str,
    comparator: str = BASE_VARIANT,
) -> int:
    base = seed_summary.loc[seed_summary["variant"].eq(comparator), ["seed", metric]].set_index("seed")[metric]
    candidate = seed_summary.loc[seed_summary["variant"].eq(variant), ["seed", metric]].set_index("seed")[metric]
    seeds = sorted(FORMAL_SEEDS)
    if direction == "minimize":
        return int((candidate.loc[seeds] < base.loc[seeds]).sum())
    if direction == "maximize":
        return int((candidate.loc[seeds] > base.loc[seeds]).sum())
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
        raise RuntimeError("S2A frozen summary 要求干净 Git 工作树")


__all__ = ["apply_s2a_decision", "summarize_crd_s2a"]
