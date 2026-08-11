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

from resp_train.crd.candidate_lock import (
    DEFAULT_CANDIDATE_LOCK,
    REPO_ROOT,
    resolve_repo_path,
    sha256_file,
    verify_candidate_lock,
)
from resp_train.crd.config import CRD_CONTROLS_PROTOCOL_VERSION, FORMAL_SEEDS
from resp_train.crd.s2_selection import BASE_VARIANT
from resp_train.crd.spectral_ops import fourier_interpolate
from resp_train.data.research_v2 import ResearchV2WindowDataset, adapt_research_v2_index
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.protocols.respiration import canonicalize_numpy
from resp_train.utils.run import save_execution_manifest


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_decoder_roundtrip_audit"

METRICS_FILENAME = "decoder_roundtrip_metrics.csv"
SUMMARY_FILENAME = "decoder_roundtrip_metrics_summary.csv"
NUMERICAL_AUDIT_FILENAME = "decoder_roundtrip_numerical_audit.json"
DECISION_FILENAME = "decoder_roundtrip_decision.json"
MANIFEST_FILENAME = "decoder_roundtrip_manifest.json"

METHOD = "CRD_102_TARGET_10HZ_FOURIER_ROUNDTRIP"
EXPECTED_VALIDATION_WINDOWS = 2675
EXPECTED_VALIDATION_SAMP_IDS = 7
CHUNK_SIZE = 64
BOUNDARY_SECONDS = 15.0

THRESHOLDS = {
    "max_abs_error": 1e-4,
    "rmse": 1e-5,
    "boundary_rmse": 1e-5,
    "whole_rr_abs_error_bpm_mean": 1e-3,
    "local_rr_mae_bpm_mean": 1e-3,
    "envelope_trajectory_mae_mean": 1e-4,
    "global_envelope_modulation_error_mean": 1e-4,
    "lag_aware_signed_pcc_mean": 0.9999,
}


def roundtrip_target_batch(
    targets: np.ndarray,
    *,
    sample_rate: float = 100.0,
    low_hz: float = 0.05,
    high_hz: float = 0.70,
    scale_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """执行 C0 固定变换，返回 raw round-trip、target Pi 与 round-trip Pi。"""

    values = np.asarray(targets)
    if values.ndim != 2 or values.shape[1] != 18000:
        raise ValueError(f"C0 target 期望 (B,18000)，实际 {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError("C0 target 包含 NaN/Inf")
    if float(sample_rate) != 100.0:
        raise ValueError("C0 固定使用 100 Hz target")

    _, target_canonical = canonicalize_numpy(
        values,
        fs=sample_rate,
        low_hz=low_hz,
        high_hz=high_hz,
        scale_eps=scale_eps,
    )
    source_10hz = np.ascontiguousarray(target_canonical[..., ::10], dtype=np.float32)
    if source_10hz.shape[1] != 1800:
        raise RuntimeError(f"C0 10-Hz 长度异常: {source_10hz.shape}")
    with torch.no_grad():
        reconstructed = fourier_interpolate(torch.from_numpy(source_10hz), target_length=18000).cpu().numpy()
    _, reconstructed_canonical = canonicalize_numpy(
        reconstructed,
        fs=sample_rate,
        low_hz=low_hz,
        high_hz=high_hz,
        scale_eps=scale_eps,
    )
    if not np.isfinite(reconstructed).all() or not np.isfinite(reconstructed_canonical).all():
        raise FloatingPointError("C0 round-trip 输出包含 NaN/Inf")
    return reconstructed, target_canonical, reconstructed_canonical


def numerical_error_rows(
    target_canonical: np.ndarray,
    reconstructed_canonical: np.ndarray,
    *,
    sample_rate: float = 100.0,
    boundary_seconds: float = BOUNDARY_SECONDS,
) -> pd.DataFrame:
    """计算逐窗口全局与首尾边界误差。"""

    target = np.asarray(target_canonical, dtype=np.float64)
    reconstructed = np.asarray(reconstructed_canonical, dtype=np.float64)
    if target.shape != reconstructed.shape or target.ndim != 2:
        raise ValueError(f"C0 canonical shape 不一致: {target.shape} vs {reconstructed.shape}")
    difference = reconstructed - target
    if not np.isfinite(difference).all():
        raise FloatingPointError("C0 canonical difference 包含 NaN/Inf")
    boundary = int(round(float(sample_rate) * float(boundary_seconds)))
    if boundary <= 0 or 2 * boundary > difference.shape[1]:
        raise ValueError("C0 boundary 长度无效")
    return pd.DataFrame(
        {
            "roundtrip_max_abs_error": np.max(np.abs(difference), axis=1),
            "roundtrip_rmse": np.sqrt(np.mean(np.square(difference), axis=1)),
            "roundtrip_start_boundary_rmse": np.sqrt(np.mean(np.square(difference[:, :boundary]), axis=1)),
            "roundtrip_end_boundary_rmse": np.sqrt(np.mean(np.square(difference[:, -boundary:]), axis=1)),
        }
    )


def summarize_numerical_errors(rows: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "roundtrip_max_abs_error",
        "roundtrip_rmse",
        "roundtrip_start_boundary_rmse",
        "roundtrip_end_boundary_rmse",
    }
    if not required.issubset(rows.columns) or rows.empty:
        raise ValueError("C0 numerical rows 不完整")
    values = rows.loc[:, sorted(required)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("C0 numerical rows 包含 NaN/Inf")
    return {
        "n_samples": int(len(rows)),
        "max_abs_error": float(rows["roundtrip_max_abs_error"].max()),
        "rmse": float(np.sqrt(np.mean(np.square(rows["roundtrip_rmse"].to_numpy(dtype=np.float64))))),
        "start_boundary_rmse": float(
            np.sqrt(np.mean(np.square(rows["roundtrip_start_boundary_rmse"].to_numpy(dtype=np.float64))))
        ),
        "end_boundary_rmse": float(
            np.sqrt(np.mean(np.square(rows["roundtrip_end_boundary_rmse"].to_numpy(dtype=np.float64))))
        ),
    }


def apply_roundtrip_decision(
    numerical_audit: dict[str, Any],
    metric_summary: dict[str, Any],
) -> dict[str, Any]:
    """应用附件第 3.4 节冻结的 C0 判定。"""

    metric_keys = (
        "whole_rr_abs_error_bpm_mean",
        "local_rr_mae_bpm_mean",
        "envelope_trajectory_mae_mean",
        "global_envelope_modulation_error_mean",
        "lag_aware_signed_pcc_mean",
    )
    primary_values = {key: float(metric_summary.get(key, math.nan)) for key in metric_keys}
    primary_finite = all(math.isfinite(value) for value in primary_values.values())
    degeneracy = float(metric_summary.get("joint_prediction_degenerate_fraction", math.nan))
    checks = {
        "max_abs_error": float(numerical_audit.get("max_abs_error", math.inf)) <= THRESHOLDS["max_abs_error"],
        "rmse": float(numerical_audit.get("rmse", math.inf)) <= THRESHOLDS["rmse"],
        "start_boundary_rmse": float(numerical_audit.get("start_boundary_rmse", math.inf))
        <= THRESHOLDS["boundary_rmse"],
        "end_boundary_rmse": float(numerical_audit.get("end_boundary_rmse", math.inf))
        <= THRESHOLDS["boundary_rmse"],
        "whole_rr": primary_values["whole_rr_abs_error_bpm_mean"]
        <= THRESHOLDS["whole_rr_abs_error_bpm_mean"],
        "local_rr": primary_values["local_rr_mae_bpm_mean"] <= THRESHOLDS["local_rr_mae_bpm_mean"],
        "trajectory": primary_values["envelope_trajectory_mae_mean"]
        <= THRESHOLDS["envelope_trajectory_mae_mean"],
        "global_envelope": primary_values["global_envelope_modulation_error_mean"]
        <= THRESHOLDS["global_envelope_modulation_error_mean"],
        "signed_pcc": primary_values["lag_aware_signed_pcc_mean"]
        >= THRESHOLDS["lag_aware_signed_pcc_mean"],
        "primary_finite": primary_finite,
        "joint_prediction_nondegenerate": math.isfinite(degeneracy) and degeneracy == 0.0,
    }
    return {
        "roundtrip_negligible": bool(all(checks.values())),
        "checks": checks,
        "thresholds": dict(THRESHOLDS),
        "numerical_audit": numerical_audit,
        "primary_metric_summary": primary_values,
        "joint_prediction_degenerate_fraction": degeneracy,
        "c2_resolution_recovery_claim_allowed": False,
        "c2_capacity_placement_framing_required": bool(all(checks.values())),
    }


def run_decoder_roundtrip_audit(
    *,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """在完整 validation 上生成一次性 C0 审计产物。"""

    _assert_clean_repository()
    verification = verify_candidate_lock(candidate_lock_path)
    records = sorted(
        (record for record in verification.records if record["variant"] == BASE_VARIANT),
        key=lambda item: int(item["seed"]),
    )
    if [int(record["seed"]) for record in records] != list(FORMAL_SEEDS):
        raise RuntimeError("candidate lock 中 CRD_102 formal seeds 不完整")

    config_paths = [resolve_repo_path(record["checkpoint_path"]).parent / "config.yaml" for record in records]
    configs = [OmegaConf.load(path) for path in config_paths]
    identities = {
        (
            str(cfg.data.dataset_root),
            str(cfg.data.index_csv),
            str(cfg.data.val_split),
            int(cfg.window.duration_samples),
            float(cfg.window.target_fs),
            float(cfg.loss.band_low_hz),
            float(cfg.loss.band_high_hz),
            float(cfg.loss.scale_eps),
        )
        for cfg in configs
    }
    if len(identities) != 1:
        raise RuntimeError(f"CRD_102 三 seed 的 C0 数据/指标 identity 不一致: {sorted(identities)}")
    cfg = OmegaConf.create(OmegaConf.to_container(configs[0], resolve=True))
    cfg.data.preload_windows = False
    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    raw_index = pd.read_csv(index_path)
    adapted = adapt_research_v2_index(raw_index, cfg)
    rows = adapted.loc[adapted["split"].eq(str(cfg.data.val_split))].sort_values("dataset_row_id").reset_index(drop=True)
    dataset = ResearchV2WindowDataset(index_path, rows, cfg, preload_windows=False)
    if len(dataset) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(f"C0 validation windows 必须为 {EXPECTED_VALIDATION_WINDOWS}，实际 {len(dataset)}")
    if int(dataset.rows["samp_id"].nunique()) != EXPECTED_VALIDATION_SAMP_IDS:
        raise RuntimeError("C0 validation samp_id 数不符")

    metric_parts: list[pd.DataFrame] = []
    numerical_parts: list[pd.DataFrame] = []
    for start in range(0, len(dataset), CHUNK_SIZE):
        items = [dataset[index] for index in range(start, min(start + CHUNK_SIZE, len(dataset)))]
        targets = np.stack([item["target"].numpy().reshape(-1) for item in items]).astype(np.float32)
        reconstructed, target_canonical, reconstructed_canonical = roundtrip_target_batch(
            targets,
            sample_rate=float(cfg.window.target_fs),
            low_hz=float(cfg.loss.band_low_hz),
            high_hz=float(cfg.loss.band_high_hz),
            scale_eps=float(cfg.loss.scale_eps),
        )
        predictions = {
            "r_tho_hat": reconstructed,
            "tho_ref": targets,
            "dataset_row_id": np.asarray([item["meta"]["dataset_row_id"] for item in items], dtype=np.int64),
            "split": np.asarray([item["meta"]["split"] for item in items]),
            "input_set": np.asarray([item["meta"]["input_set"] for item in items]),
            "samp_id": np.asarray([item["meta"]["samp_id"] for item in items], dtype=np.int64),
            "coupling_state_id": np.asarray(
                [item["meta"]["coupling_state_id"] for item in items], dtype=np.int64
            ),
        }
        metrics = evaluate_task_predictions(predictions, cfg, method=METHOD)
        errors = numerical_error_rows(
            target_canonical,
            reconstructed_canonical,
            sample_rate=float(cfg.window.target_fs),
        )
        if len(metrics) != len(errors):
            raise RuntimeError("C0 metric/numerical rows 数量不一致")
        metrics = pd.concat([metrics.reset_index(drop=True), errors], axis=1)
        metric_parts.append(metrics)
        numerical_parts.append(errors)

    all_metrics = pd.concat(metric_parts, ignore_index=True)
    all_numerical = pd.concat(numerical_parts, ignore_index=True)
    if len(all_metrics) != EXPECTED_VALIDATION_WINDOWS or not all_metrics["split"].eq("val").all():
        raise RuntimeError("C0 metrics identity 不完整")
    summary = summarize_task_metrics(all_metrics)
    numerical_audit = summarize_numerical_errors(all_numerical)
    decision = apply_roundtrip_decision(numerical_audit, summary.iloc[0].to_dict())

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"C0 输出禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root.parent / f".{output_root.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        all_metrics.to_csv(temporary_dir / METRICS_FILENAME, index=False)
        summary.to_csv(temporary_dir / SUMMARY_FILENAME, index=False)
        (temporary_dir / NUMERICAL_AUDIT_FILENAME).write_text(
            json.dumps(numerical_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary_dir / DECISION_FILENAME).write_text(
            json.dumps(
                {
                    "protocol": CRD_CONTROLS_PROTOCOL_VERSION,
                    "candidate_lock_sha256": verification.lock_sha256,
                    "dataset_index_sha256": sha256_file(index_path),
                    "research_test_used": False,
                    "model_inference_used": False,
                    "checkpoint_tensor_read": False,
                    "checkpoint_reselection_allowed": False,
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
            task="crd_102_decoder_roundtrip_audit",
            phase="c0_target_10hz_fourier_roundtrip",
            protocol=CRD_CONTROLS_PROTOCOL_VERSION,
            candidate_lock_sha256=verification.lock_sha256,
            dataset_index=str(index_path),
            dataset_index_sha256=sha256_file(index_path),
            validation_window_count=int(len(all_metrics)),
            validation_samp_id_count=int(all_metrics["samp_id"].nunique()),
            roundtrip_negligible=decision["roundtrip_negligible"],
            research_test_used=False,
            model_inference_used=False,
            checkpoint_tensor_read=False,
        )
        os.replace(temporary_dir, output_root)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_root / DECISION_FILENAME


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
        raise RuntimeError("C0 正式审计要求干净 Git 工作树")

