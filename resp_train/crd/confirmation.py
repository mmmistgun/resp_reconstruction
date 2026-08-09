from __future__ import annotations

import os
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from resp_train.crd.candidate_lock import (
    DEFAULT_CANDIDATE_LOCK,
    CandidateLockVerification,
    resolve_repo_path,
    verify_candidate_lock,
)
from resp_train.crd.config import crd_dependency_versions, load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.metrics.task import summarize_task_metrics
from resp_train.utils.run import save_execution_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
S1C_PROTOCOL_VERSION = "crd-v1.1-s1c-research-20260809"
S1C_CANDIDATE_LOCK_SHA256 = "9a14db8be8af22e1ce1c5a332b4912ab5c13c7fb03cdf1894fc5c6ed6ff7f8cc"
S1C_EXPECTED_TEST_WINDOWS = 2310
S1C_EXPECTED_TEST_SAMP_IDS = 8
DEFAULT_S1C_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_s1c_research_confirmation"

ACCESS_RECEIPT_FILENAME = "research_confirmation_access_receipt.json"
METRICS_FILENAME = "research_confirmation_metrics.csv"
SUMMARY_FILENAME = "research_confirmation_metrics_summary.csv"
MANIFEST_FILENAME = "research_confirmation_metrics_manifest.json"

_REQUIRED_METRIC_COLUMNS = {
    "evaluation_split",
    "method",
    "dataset_row_id",
    "split",
    "samp_id",
    "whole_rr_abs_error_bpm",
    "whole_rr_target_eligible",
    "local_rr_mae_bpm",
    "local_rr_target_eligible",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
    "joint_target_eligible",
    "respiratory_band_coherence",
    "constrained_ndtw",
}


def evaluate_crd_s1c_checkpoint(
    *,
    checkpoint_path: str | Path,
    device: str,
    confirm_research_test: bool,
    lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    output_root: str | Path = DEFAULT_S1C_OUTPUT_ROOT,
    expected_test_windows: int = S1C_EXPECTED_TEST_WINDOWS,
    expected_test_samp_ids: int = S1C_EXPECTED_TEST_SAMP_IDS,
) -> Path:
    """对 candidate lock 中的单个 checkpoint 执行一次受控 S1C research-test 评价。"""

    if not confirm_research_test:
        raise ValueError("S1C research-test 必须显式传入 --confirm-research-test")

    verification = verify_candidate_lock(lock_path, checkpoint_path=checkpoint_path)
    _validate_s1c_lock(verification)
    record = verification.matched_record
    if record is None:  # verify_candidate_lock 已保证；保留显式防线。
        raise RuntimeError("candidate lock 未返回目标 checkpoint 记录")

    checkpoint = resolve_repo_path(record["checkpoint_path"])
    config_path = checkpoint.parent / "config.yaml"
    cfg = load_crd_config(
        config_path,
        overrides=[f"training.device={device}", "training.show_progress=false"],
    )
    _validate_s1c_config(cfg, record)
    _assert_clean_repository()

    output_dir = Path(output_root) / str(record["variant"]) / f"seed_{int(record['seed'])}"
    final_paths = {
        "access_receipt": output_dir / ACCESS_RECEIPT_FILENAME,
        "metrics": output_dir / METRICS_FILENAME,
        "summary": output_dir / SUMMARY_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing:
        raise FileExistsError("S1C 禁止覆盖或重复评价，已存在: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)

    # receipt 在首次读取 test 前排他创建；若进程中断，保留它以阻止静默重跑。
    access_temp = output_dir / f".{ACCESS_RECEIPT_FILENAME}.{uuid4().hex}.tmp"
    try:
        save_execution_manifest(
            access_temp,
            task="crd_v1_s1c_research_confirmation",
            phase="research_test_access_started",
            protocol=S1C_PROTOCOL_VERSION,
            evaluation_split="test",
            evidence_boundary="development_research_confirmation_not_unbiased_heldout",
            candidate_lock=str(verification.lock_path),
            candidate_lock_sha256=verification.lock_sha256,
            checkpoint=str(checkpoint),
            checkpoint_sha256=str(record["checkpoint_sha256"]),
            checkpoint_role=str(record["role"]),
            variant=str(record["variant"]),
            seed=int(record["seed"]),
            selected_epoch=int(record["selected_epoch"]),
            dependency_versions=crd_dependency_versions(),
        )
        try:
            os.link(access_temp, final_paths["access_receipt"])
        except FileExistsError as exc:
            raise FileExistsError(f"S1C 已有访问记录，禁止重复评价: {final_paths['access_receipt']}") from exc
    finally:
        access_temp.unlink(missing_ok=True)

    metrics = CRDExperiment(cfg).evaluate_checkpoint(checkpoint, split="test", metrics_output=None)
    _validate_confirmation_metrics(
        metrics,
        variant=str(record["variant"]),
        expected_test_windows=expected_test_windows,
        expected_test_samp_ids=expected_test_samp_ids,
    )
    summary = summarize_task_metrics(metrics)

    token = uuid4().hex
    temporary_paths = {
        name: output_dir / f".{path.name}.{token}.tmp"
        for name, path in final_paths.items()
        if name != "access_receipt"
    }
    try:
        metrics.to_csv(temporary_paths["metrics"], index=False)
        summary.to_csv(temporary_paths["summary"], index=False)
        save_execution_manifest(
            temporary_paths["manifest"],
            task="crd_v1_s1c_research_confirmation",
            phase="s1c_research_confirmation",
            protocol=S1C_PROTOCOL_VERSION,
            evaluation_split="test",
            evaluation_role="research_confirmation_development_evidence",
            evidence_boundary="not_unbiased_heldout",
            research_test_previously_observed=True,
            candidate_lock=str(verification.lock_path),
            candidate_lock_id=str(verification.payload["lock_id"]),
            candidate_lock_sha256=verification.lock_sha256,
            selection_rule=verification.payload["selection_rule"],
            checkpoint=str(checkpoint),
            checkpoint_sha256=str(record["checkpoint_sha256"]),
            checkpoint_role=str(record["role"]),
            variant=str(record["variant"]),
            seed=int(record["seed"]),
            selected_epoch=int(record["selected_epoch"]),
            training_commit=str(record["training_commit"]),
            training_protocol=str(record["training_protocol"]),
            config=str(config_path.resolve()),
            expected_test_windows=int(expected_test_windows),
            observed_test_windows=int(len(metrics)),
            expected_test_samp_ids=int(expected_test_samp_ids),
            observed_test_samp_ids=int(metrics["samp_id"].nunique(dropna=False)),
            dependency_versions=crd_dependency_versions(),
            overwrite_allowed=False,
        )
        for name in ("metrics", "summary", "manifest"):
            os.replace(temporary_paths[name], final_paths[name])
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
    return final_paths["metrics"]


def _validate_s1c_lock(verification: CandidateLockVerification) -> None:
    payload = verification.payload
    if verification.lock_sha256 != S1C_CANDIDATE_LOCK_SHA256:
        raise ValueError(
            "S1C candidate lock SHA-256 与激活协议不一致: "
            f"{verification.lock_sha256} != {S1C_CANDIDATE_LOCK_SHA256}"
        )
    if payload.get("lock_id") != "crd-v1.1-candidate-lock-20260809":
        raise ValueError("S1C candidate lock ID 不一致")


def _validate_s1c_config(cfg, record: dict[str, object]) -> None:
    if str(cfg.protocol.run_role) != "formal":
        raise ValueError("S1C 只允许 frozen formal checkpoint 配置")
    if str(cfg.model.variant) != str(record["variant"]):
        raise ValueError("S1C checkpoint variant 与 candidate lock 不一致")
    if int(cfg.training.seed) != int(record["seed"]):
        raise ValueError("S1C checkpoint seed 与 candidate lock 不一致")
    if str(cfg.data.test_split) != "test":
        raise ValueError("S1C data.test_split 必须为 test")
    if cfg.data.get("max_test_windows") is not None:
        raise ValueError("S1C 必须评价完整 research-test，max_test_windows 必须为 null")


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        detail = status.stderr.strip() or "git status failed"
        raise RuntimeError(f"S1C 无法确认 Git 工作树状态: {detail}")
    if status.stdout.strip():
        raise RuntimeError("S1C research-test 要求干净 Git 工作树；请先提交或移出全部改动")


def _validate_confirmation_metrics(
    metrics: pd.DataFrame,
    *,
    variant: str,
    expected_test_windows: int,
    expected_test_samp_ids: int,
) -> None:
    if len(metrics) != int(expected_test_windows):
        raise RuntimeError(f"S1C test 行数异常: {len(metrics)} != {expected_test_windows}")
    missing = sorted(_REQUIRED_METRIC_COLUMNS - set(metrics.columns))
    if missing:
        raise RuntimeError(f"S1C metrics 缺少字段: {missing}")
    if set(metrics["evaluation_split"].astype(str)) != {"test"}:
        raise RuntimeError("S1C evaluation_split 必须全部为 test")
    if set(metrics["split"].astype(str)) != {"test"}:
        raise RuntimeError("S1C sample split 必须全部为 test")
    if set(metrics["method"].astype(str)) != {variant}:
        raise RuntimeError("S1C method 与 candidate lock variant 不一致")
    if metrics["dataset_row_id"].duplicated().any():
        raise RuntimeError("S1C dataset_row_id 存在重复")
    observed_samp_ids = int(metrics["samp_id"].nunique(dropna=False))
    if observed_samp_ids != int(expected_test_samp_ids):
        raise RuntimeError(f"S1C test samp_id 数异常: {observed_samp_ids} != {expected_test_samp_ids}")

    numeric = metrics.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)
    if np.isinf(numeric).any():
        raise FloatingPointError("S1C metrics 包含 Inf")
    _require_finite(metrics, "envelope_trajectory_mae")
    _require_finite(metrics, "global_envelope_modulation_error")
    _require_finite(metrics, "whole_rr_abs_error_bpm", eligible="whole_rr_target_eligible")
    _require_finite(metrics, "local_rr_mae_bpm", eligible="local_rr_target_eligible")
    for column in ("lag_aware_signed_pcc", "respiratory_band_coherence", "constrained_ndtw"):
        _require_finite(metrics, column, eligible="joint_target_eligible")


def _require_finite(metrics: pd.DataFrame, column: str, *, eligible: str | None = None) -> None:
    values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.ones(len(metrics), dtype=bool) if eligible is None else metrics[eligible].astype(bool).to_numpy()
    if not np.isfinite(values[mask]).all():
        raise FloatingPointError(f"S1C eligible {column} 包含缺失或非有限值")
