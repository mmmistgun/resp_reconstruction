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

from resp_train.crd.c1_selection import (
    REPORT_METRICS,
    SEED_SUMMARY_METRICS,
    _iter_tensors,
    _load_metrics,
    _variant_summary,
)
from resp_train.crd.candidate_lock import (
    DEFAULT_CANDIDATE_LOCK,
    REPO_ROOT,
    resolve_repo_path,
    sha256_file,
    verify_candidate_lock,
)
from resp_train.crd.config import CRD_CONTROLS_PROTOCOL_VERSION, FORMAL_SEEDS, load_crd_config
from resp_train.crd.decoder_diagnostics import MANIFEST_FILENAME as RESIDUAL_MANIFEST_FILENAME
from resp_train.crd.decoder_diagnostics import SUMMARY_FILENAME as RESIDUAL_SUMMARY_FILENAME
from resp_train.crd.decoder_diagnostics import WINDOW_FILENAME as RESIDUAL_WINDOW_FILENAME
from resp_train.crd.s2_selection import BASE_VARIANT
from resp_train.utils.run import save_execution_manifest


C201_VARIANT = "crd_c201_decoder_10hz_cap"
C202_VARIANT = "crd_c202_decoder_100hz"
C2_VARIANTS = (C201_VARIANT, C202_VARIANT)
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs/crd_v1"
DEFAULT_RESIDUAL_ROOT = REPO_ROOT / "runs/crd_v1/crd_c2_decoder_diagnostics"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_c2_validation_summary"
DEFAULT_FAILURE_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_failure_diagnostic"
DEFAULT_METADATA_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_failure_metadata_diagnostic"
DEFAULT_OBSERVABILITY_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_matched_observability_diagnostic"

SEED_SUMMARY_FILENAME = "c2_seed_summary.csv"
VARIANT_SUMMARY_FILENAME = "c2_variant_summary.csv"
PAIRED_WINDOW_FILENAME = "c2_paired_window_descriptives.csv"
PAIRED_SAMP_FILENAME = "c2_paired_samp_descriptives.csv"
FAILURE_STRATA_FILENAME = "c2_failure_strata_summary.csv"
RESIDUAL_SEED_FILENAME = "c2_residual_seed_summary.csv"
DECISION_FILENAME = "c2_selection.json"
MANIFEST_FILENAME = "c2_selection_manifest.json"


def apply_c2_decision(seed_summary: pd.DataFrame) -> dict[str, Any]:
    """应用附件第 5.2 节的两个基本资格与 C202-vs-C201 placement 门槛。"""

    required = {"variant", "seed", *(f"{metric}_mean" for metric in SEED_SUMMARY_METRICS)}
    if not required.issubset(seed_summary.columns):
        raise ValueError(f"C2 seed summary 缺少字段: {sorted(required - set(seed_summary.columns))}")
    rows = seed_summary.copy()
    if set(rows["variant"]) != {BASE_VARIANT, C201_VARIANT, C202_VARIANT}:
        raise ValueError("C2 decision 必须恰含 BASE/C201/C202")
    grouped: dict[str, pd.DataFrame] = {}
    for variant in (BASE_VARIANT, C201_VARIANT, C202_VARIANT):
        group = rows.loc[rows["variant"].eq(variant)].sort_values("seed").reset_index(drop=True)
        if group["seed"].astype(int).tolist() != list(FORMAL_SEEDS):
            raise ValueError(f"C2 {variant} seeds 不完整")
        grouped[variant] = group

    base = grouped[BASE_VARIANT]
    eligibility = {
        variant: _comparison_gate(grouped[variant], base, local_threshold=0.005, pcc_limit=0.005)
        for variant in C2_VARIANTS
    }
    placement = _comparison_gate(
        grouped[C202_VARIANT],
        grouped[C201_VARIANT],
        local_threshold=0.0025,
        pcc_limit=0.003,
    )
    if eligibility[C202_VARIANT]["passed"] and placement["passed"]:
        selected = C202_VARIANT
        outcome = "100hz_nonlinear_placement_supported"
    elif eligibility[C201_VARIANT]["passed"]:
        selected = C201_VARIANT
        outcome = "decoder_capacity_supported_100hz_placement_not_supported"
    elif eligibility[C202_VARIANT]["passed"]:
        selected = C202_VARIANT
        outcome = "c202_package_supported_without_100hz_attribution"
    else:
        selected = BASE_VARIANT
        outcome = "retain_original_10hz_fourier_decoder"
    return {
        "outcome": outcome,
        "selected_variant": selected,
        "basic_eligibility": eligibility,
        "c202_vs_c201_placement": placement,
        "retain_crd_102_backbone": True,
        "research_test_allowed": False,
        "tcn_decoder_combination_allowed": False,
    }


def _comparison_gate(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    local_threshold: float,
    pcc_limit: float,
) -> dict[str, Any]:
    candidate_local = float(candidate["local_rr_mae_bpm_mean"].mean())
    comparator_local = float(comparator["local_rr_mae_bpm_mean"].mean())
    candidate_pcc = float(candidate["lag_aware_signed_pcc_mean"].mean())
    comparator_pcc = float(comparator["lag_aware_signed_pcc_mean"].mean())
    candidate_trajectory = float(candidate["envelope_trajectory_mae_mean"].mean())
    comparator_trajectory = float(comparator["envelope_trajectory_mae_mean"].mean())
    values = (
        candidate_local,
        comparator_local,
        candidate_pcc,
        comparator_pcc,
        candidate_trajectory,
        comparator_trajectory,
    )
    if not all(math.isfinite(value) for value in values) or comparator_local <= 0 or comparator_trajectory <= 0:
        raise FloatingPointError("C2 gate 输入包含无效数值")
    local_improvement = (comparator_local - candidate_local) / comparator_local
    paired = int(
        np.sum(candidate["local_rr_mae_bpm_mean"].to_numpy() < comparator["local_rr_mae_bpm_mean"].to_numpy())
    )
    pcc_drop = comparator_pcc - candidate_pcc
    trajectory_worsening = (candidate_trajectory - comparator_trajectory) / comparator_trajectory
    checks = {
        "local_rr_improvement": local_improvement >= local_threshold,
        "paired_local_rr_improved_at_least_2_of_3": paired >= 2,
        "pcc_drop": pcc_drop <= pcc_limit,
        "trajectory_worsening_at_most_1_5pct": trajectory_worsening <= 0.015,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "local_rr_relative_improvement": local_improvement,
        "local_rr_improvement_threshold": local_threshold,
        "paired_local_rr_improved_seed_count": paired,
        "signed_pcc_absolute_drop": pcc_drop,
        "signed_pcc_drop_limit": pcc_limit,
        "trajectory_relative_worsening": trajectory_worsening,
    }


def summarize_c2(
    *,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    residual_root: str | Path = DEFAULT_RESIDUAL_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    failure_root: str | Path = DEFAULT_FAILURE_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    observability_root: str | Path = DEFAULT_OBSERVABILITY_ROOT,
) -> Path:
    """审计 C201/C202 六个 runs 与 residual diagnostics，生成不可覆盖冻结汇总。"""

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
    checkpoint_by_seed: dict[int, Path] = {}
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
    runs_root = Path(runs_root)
    for variant in C2_VARIANTS:
        for seed in FORMAL_SEEDS:
            run_dir = _unique_run_dir(runs_root / variant, seed)
            audit, metrics, summary = _audit_candidate_run(run_dir, variant, seed)
            training_commits.add(str(audit["training_commit"]))
            candidate_run_dirs.append(str(run_dir))
            metrics_by_key[(variant, seed)] = metrics
            seed_rows.append({"role": "candidate", **audit, **summary})
            if variant == C202_VARIANT:
                checkpoint_by_seed[seed] = run_dir / "checkpoint_best_local_rr.pt"
    if len(training_commits) != 1:
        raise RuntimeError(f"C2 training commit 不统一: {sorted(training_commits)}")

    residual_seed = _audit_residual_diagnostics(Path(residual_root), checkpoint_by_seed)
    seed_summary = pd.DataFrame(seed_rows).sort_values(["role", "variant", "seed"]).reset_index(drop=True)
    variant_summary = _variant_summary(seed_summary)
    paired_window, paired_samp = _paired_descriptives(metrics_by_key)
    failure_strata = _failure_strata_descriptives(
        metrics_by_key,
        failure_root=Path(failure_root),
        metadata_root=Path(metadata_root),
        observability_root=Path(observability_root),
    )
    decision = apply_c2_decision(seed_summary)

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"C2 summary 禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root.parent / f".{output_root.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        seed_summary.to_csv(temporary_dir / SEED_SUMMARY_FILENAME, index=False)
        variant_summary.to_csv(temporary_dir / VARIANT_SUMMARY_FILENAME, index=False)
        paired_window.to_csv(temporary_dir / PAIRED_WINDOW_FILENAME, index=False)
        paired_samp.to_csv(temporary_dir / PAIRED_SAMP_FILENAME, index=False)
        failure_strata.to_csv(temporary_dir / FAILURE_STRATA_FILENAME, index=False)
        residual_seed.to_csv(temporary_dir / RESIDUAL_SEED_FILENAME, index=False)
        (temporary_dir / DECISION_FILENAME).write_text(
            json.dumps(
                {
                    "protocol": CRD_CONTROLS_PROTOCOL_VERSION,
                    "candidate_lock_sha256": verification.lock_sha256,
                    "training_commit": next(iter(training_commits)),
                    "complete_candidate_run_count": 6,
                    "complete_residual_diagnostic_count": 3,
                    "research_test_used": False,
                    "confirmatory_p_values_used": False,
                    "c202_out_of_band_energy_fraction_seed_mean": float(
                        residual_seed["out_of_band_energy_fraction_mean"].mean()
                    ),
                    "c202_out_of_band_energy_fraction_across_seed_sd": float(
                        residual_seed["out_of_band_energy_fraction_mean"].std(ddof=1)
                    ),
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
            task="crd_c2_validation_summary",
            phase="c2_decoder_capacity_placement_controls",
            protocol=CRD_CONTROLS_PROTOCOL_VERSION,
            candidate_lock_sha256=verification.lock_sha256,
            training_commit=next(iter(training_commits)),
            candidate_run_dirs=candidate_run_dirs,
            complete_candidate_run_count=6,
            complete_residual_diagnostic_count=3,
            selected_variant=decision["selected_variant"],
            decision_outcome=decision["outcome"],
            research_test_used=False,
        )
        os.replace(temporary_dir, output_root)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_root / DECISION_FILENAME


def _unique_run_dir(variant_root: Path, seed: int) -> Path:
    seed_root = variant_root / f"seed_{seed}"
    candidates = sorted(path for path in seed_root.iterdir() if path.is_dir()) if seed_root.exists() else []
    if len(candidates) != 1:
        raise RuntimeError(f"C2 {variant_root.name} seed={seed} 必须恰有一个 run: {candidates}")
    return candidates[0]


def _audit_candidate_run(
    run_dir: Path,
    variant: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    required = (
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
        raise FileNotFoundError(f"C2 run 产物不完整 {run_dir}: {missing}")
    cfg = load_crd_config(run_dir / "config.yaml")
    if (
        str(cfg.model.variant) != variant
        or str(cfg.protocol.run_role) != "formal"
        or str(cfg.protocol.stage) != "c2"
        or int(cfg.training.seed) != int(seed)
        or (int(cfg.training.epochs), int(cfg.training.batch_size), int(cfg.training.gradient_accumulation_steps))
        != (80, 128, 1)
    ):
        raise RuntimeError(f"C2 resolved config identity 不一致: {run_dir}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("git_dirty") is not False
        or manifest.get("protocol") != CRD_CONTROLS_PROTOCOL_VERSION
        or manifest.get("stage") != "c2"
        or manifest.get("run_role") != "formal"
    ):
        raise RuntimeError(f"C2 manifest identity 不一致: {run_dir}")
    history = pd.read_csv(run_dir / "train_history.csv")
    if (
        len(history) != 80
        or history["epoch"].astype(int).tolist() != list(range(1, 81))
        or int(history.iloc[-1]["optimizer_update"]) != 6400
        or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError(f"C2 history 不完整或非有限: {run_dir}")
    selected_epoch = int(history.loc[history["val_local_rr_mae"].idxmin(), "epoch"])
    best = torch.load(run_dir / "checkpoint_best_local_rr.pt", map_location="cpu")
    final = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu")
    if int(best["epoch"]) != selected_epoch or int(final["epoch"]) != 80:
        raise RuntimeError(f"C2 checkpoint selector/epoch 不一致: {run_dir}")
    for name, checkpoint, expected_update in (
        ("best", best, selected_epoch * 80),
        ("final", final, 6400),
    ):
        extra = checkpoint.get("extra_state", {})
        if int(extra.get("update_index", -1)) != expected_update or int(extra.get("total_updates", -1)) != 6400:
            raise RuntimeError(f"C2 {name} checkpoint update identity 不一致: {run_dir}")
        tensors = list(_iter_tensors(checkpoint["model_state_dict"])) + list(
            _iter_tensors(checkpoint["optimizer_state_dict"])
        )
        if not tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise FloatingPointError(f"C2 {name} checkpoint 包含 NaN/Inf: {run_dir}")
    metrics, summary = _load_metrics(run_dir, variant)
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    reserved = float(runtime.get("peak_reserved_fraction", math.nan))
    if not (0.0 < reserved <= 0.80):
        raise RuntimeError(f"C2 peak reserved fraction 越界: {run_dir}: {reserved}")
    return (
        {
            "variant": variant,
            "seed": int(seed),
            "selected_epoch": selected_epoch,
            "checkpoint_path": str(run_dir / "checkpoint_best_local_rr.pt"),
            "checkpoint_sha256": sha256_file(run_dir / "checkpoint_best_local_rr.pt"),
            "run_dir": str(run_dir),
            "training_commit": str(manifest.get("git_commit")),
            "peak_allocated_mib": float(runtime["peak_allocated_mib"]),
            "peak_reserved_mib": float(runtime["peak_reserved_mib"]),
            "peak_reserved_fraction": reserved,
        },
        metrics,
        summary,
    )


def _audit_residual_diagnostics(residual_root: Path, checkpoints: dict[int, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    commits: set[str] = set()
    for seed in FORMAL_SEEDS:
        root = residual_root / C202_VARIANT / f"seed_{seed}"
        paths = {
            "summary": root / RESIDUAL_SUMMARY_FILENAME,
            "manifest": root / RESIDUAL_MANIFEST_FILENAME,
            "window": root / RESIDUAL_WINDOW_FILENAME,
        }
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"C202 residual diagnostic 不完整 seed={seed}: {missing}")
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        checkpoint = checkpoints[seed]
        expected_hash = sha256_file(checkpoint)
        if (
            summary.get("protocol") != CRD_CONTROLS_PROTOCOL_VERSION
            or manifest.get("protocol") != CRD_CONTROLS_PROTOCOL_VERSION
            or int(summary.get("seed", -1)) != seed
            or int(manifest.get("seed", -1)) != seed
            or summary.get("checkpoint_sha256") != expected_hash
            or manifest.get("checkpoint_sha256") != expected_hash
            or summary.get("research_test_used") is not False
            or manifest.get("research_test_used") is not False
            or manifest.get("git_dirty") is not False
        ):
            raise RuntimeError(f"C202 residual diagnostic identity 不一致 seed={seed}")
        frame = pd.read_csv(paths["window"])
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ratio = frame["residual_out_of_band_energy_fraction"].to_numpy(dtype=np.float64)
        if (
            len(frame) != 2675
            or frame["dataset_row_id"].nunique() != 2675
            or frame["samp_id"].nunique() != 7
            or not np.isfinite(numeric).all()
            or not ((ratio >= 0.0) & (ratio <= 1.0)).all()
            or not math.isclose(float(np.mean(ratio)), float(summary["out_of_band_energy_fraction_mean"]), rel_tol=1e-12)
        ):
            raise RuntimeError(f"C202 residual diagnostic rows/summary 不一致 seed={seed}")
        commits.add(str(manifest.get("git_commit")))
        rows.append(summary)
    if len(commits) != 1:
        raise RuntimeError(f"C202 residual diagnostic commit 不统一: {sorted(commits)}")
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _paired_descriptives(
    metrics_by_key: dict[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = ((C201_VARIANT, BASE_VARIANT), (C202_VARIANT, BASE_VARIANT), (C202_VARIANT, C201_VARIANT))
    parts: list[pd.DataFrame] = []
    for candidate, comparator in comparisons:
        for seed in FORMAL_SEEDS:
            left = metrics_by_key[(comparator, seed)]
            right = metrics_by_key[(candidate, seed)]
            columns = ["dataset_row_id", "samp_id", *REPORT_METRICS]
            merged = left[columns].merge(right[columns], on=["dataset_row_id", "samp_id"], suffixes=("_comparator", "_candidate"))
            if len(merged) != 2675:
                raise RuntimeError(f"C2 {candidate} vs {comparator} seed={seed} paired identity 不完整")
            result = merged[["dataset_row_id", "samp_id"]].copy()
            result.insert(0, "seed", seed)
            result.insert(0, "comparator", comparator)
            result.insert(0, "candidate", candidate)
            for metric in REPORT_METRICS:
                result[f"{metric}_comparator"] = merged[f"{metric}_comparator"]
                result[f"{metric}_candidate"] = merged[f"{metric}_candidate"]
                result[f"{metric}_candidate_minus_comparator"] = (
                    merged[f"{metric}_candidate"] - merged[f"{metric}_comparator"]
                )
            parts.append(result)
    window = pd.concat(parts, ignore_index=True)
    identifiers = {"candidate", "comparator", "seed", "dataset_row_id", "samp_id"}
    values = [column for column in window.columns if column not in identifiers]
    samp = window.groupby(["candidate", "comparator", "seed", "samp_id"], as_index=False)[values].mean()
    return window, samp


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
    roles = {int(value): "matched_case" for value in exact["case_dataset_row_id"]}
    roles.update({int(value): "matched_control" for value in exact["control_dataset_row_id"]})
    frozen = consensus[
        ["dataset_row_id", "local_rr_mae_bpm_persistent_failure", "envelope_target_stratum"]
    ].merge(
        metadata[["dataset_row_id", "waveform_confidence_level"]],
        on="dataset_row_id",
        how="left",
        validate="one_to_one",
    )
    frozen["matched_observability_role"] = frozen["dataset_row_id"].map(roles).fillna("not_matched")
    frozen["high_modulation_local_rr_failure"] = np.where(
        frozen["local_rr_mae_bpm_persistent_failure"].astype(bool)
        & frozen["envelope_target_stratum"].eq("high"),
        "yes",
        "no",
    )
    dimensions = {
        "local_rr_persistent_failure": "local_rr_mae_bpm_persistent_failure",
        "waveform_confidence": "waveform_confidence_level",
        "target_modulation": "envelope_target_stratum",
        "matched_observability": "matched_observability_role",
        "high_modulation_local_rr_failure": "high_modulation_local_rr_failure",
    }
    rows: list[dict[str, Any]] = []
    for variant in C2_VARIANTS:
        for seed in FORMAL_SEEDS:
            base = metrics_by_key[(BASE_VARIANT, seed)]
            candidate = metrics_by_key[(variant, seed)]
            columns = ["dataset_row_id", *REPORT_METRICS]
            merged = base[columns].merge(candidate[columns], on="dataset_row_id", suffixes=("_base", "_candidate"))
            merged = merged.merge(frozen, on="dataset_row_id", how="left", validate="one_to_one")
            for dimension, column in dimensions.items():
                for label, group in merged.groupby(column, dropna=False, sort=True):
                    for metric in REPORT_METRICS:
                        base_values = pd.to_numeric(group[f"{metric}_base"], errors="coerce").to_numpy(dtype=np.float64)
                        candidate_values = pd.to_numeric(
                            group[f"{metric}_candidate"], errors="coerce"
                        ).to_numpy(dtype=np.float64)
                        valid = np.isfinite(base_values) & np.isfinite(candidate_values)
                        rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "dimension": dimension,
                                "label": str(label),
                                "metric": metric,
                                "n": int(np.sum(valid)),
                                "base_mean": float(np.mean(base_values[valid])) if np.any(valid) else math.nan,
                                "candidate_mean": (
                                    float(np.mean(candidate_values[valid])) if np.any(valid) else math.nan
                                ),
                                "candidate_minus_base_mean": (
                                    float(np.mean(candidate_values[valid] - base_values[valid]))
                                    if np.any(valid)
                                    else math.nan
                                ),
                            }
                        )
    return pd.DataFrame(rows)


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
        raise RuntimeError("C2 冻结汇总要求干净 Git 工作树")

