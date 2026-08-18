from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from resp_train.crd.tf_w_v2 import (
    CANDIDATE_LOCK_SHA256,
    P3_VARIANT,
    P3_VARIANTS,
    PROTOCOL,
    SOURCE_CACHE_MANIFEST_SHA256,
    verify_p1_source_identity,
)
from resp_train.crd.tf_w_v2_p1 import (
    PRIMARY_COLUMNS,
    _all_checkpoint_tensors_finite,
    _benchmark_p1_model,
)
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
P3_ENGINEERING_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/engineering"
P3_FORMAL_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/formal"
P3_STRESS_RECEIPT = "p3_stress_receipt.json"
P3_TRAINABLE_PARAMETERS = 902_722
P3_LOCAL_BLOCK_COUNT = 4


def validate_p3_training_preflight(cfg: Any) -> str:
    """P3 stress/formal 只允许从干净 commit 与冻结 full-12V source 启动。"""

    variant = str(cfg.model.variant).strip().lower()
    role = str(cfg.protocol.run_role).strip().lower()
    if variant not in P3_VARIANTS:
        raise ValueError(f"P3 training preflight 未知 variant={variant!r}")
    verify_p1_source_identity()
    commit = _require_clean_git()
    if role == "stress":
        parent = P3_ENGINEERING_ROOT / variant
        if Path(str(cfg.outputs.run_root)).resolve() != parent.resolve():
            raise RuntimeError(f"P3 stress output root 漂移: {cfg.outputs.run_root}")
        existing = sorted(path for path in parent.glob("20*") if path.is_dir()) if parent.is_dir() else []
        if existing:
            raise RuntimeError(f"P3 isolation stress 已存在，拒绝重复运行: {existing}")
    elif role == "formal":
        parent = P3_FORMAL_ROOT / variant / f"seed_{int(cfg.training.seed)}"
        if Path(str(cfg.outputs.run_root)).resolve() != parent.resolve():
            raise RuntimeError(f"P3 formal output root 漂移: {cfg.outputs.run_root}")
        existing = sorted(path for path in parent.glob("20*") if path.is_dir()) if parent.is_dir() else []
        if existing:
            raise RuntimeError(f"P3 formal seed 已存在，拒绝重复运行: {existing}")
        validate_p3_formal_preflight(variant)
    elif role != "smoke":
        raise ValueError(f"P3 training preflight 不接受 run_role={role!r}")
    return commit


def validate_p3_formal_preflight(variant: str = P3_VARIANT) -> dict[str, Any]:
    """任一 D4 formal 前要求唯一 P3 stress receipt 通过且 commit 相同。"""

    normalized = str(variant).strip().lower()
    if normalized not in P3_VARIANTS:
        raise ValueError(f"P3 formal preflight 未知 variant={variant!r}")
    verify_p1_source_identity()
    current_commit = _git_commit()
    parent = P3_ENGINEERING_ROOT / normalized
    run_dirs = sorted(path for path in parent.glob("20*") if path.is_dir()) if parent.is_dir() else []
    if len(run_dirs) != 1:
        raise RuntimeError(f"P3 formal 要求唯一 isolation stress run，当前数量={len(run_dirs)}")
    receipt_path = run_dirs[0] / P3_STRESS_RECEIPT
    if not receipt_path.is_file():
        raise RuntimeError(f"P3 formal 缺少 stress receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _validate_p3_stress_receipt(receipt, normalized)
    if receipt.get("git_commit") != current_commit:
        raise RuntimeError("P3 formal 与 isolation stress 不是同一 commit")
    return receipt


def _validate_p3_stress_receipt(receipt: Mapping[str, Any], variant: str = P3_VARIANT) -> None:
    normalized = str(variant).strip().lower()
    if normalized not in P3_VARIANTS:
        raise ValueError(f"P3 stress receipt 未知 variant={variant!r}")
    required_exact = {
        "protocol": PROTOCOL,
        "phase": "p3_depth_stress",
        "variant": normalized,
        "status": "passed",
        "complete": True,
        "run_role": "stress",
        "git_dirty": False,
        "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
        "source_cache_manifest_sha256": SOURCE_CACHE_MANIFEST_SHA256,
        "physical_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "trainable_parameters": P3_TRAINABLE_PARAMETERS,
        "local_bimamba2_blocks": P3_LOCAL_BLOCK_COUNT,
        "benchmark_batch_size": 1,
        "all_benchmark_input_gradients_finite": True,
        "all_benchmark_parameter_gradients_finite": True,
        "all_history_finite": True,
        "all_checkpoint_finite": True,
        "all_optimizer_finite": True,
        "all_primary_finite": True,
        "prediction_degenerate_fraction": 0.0,
        "validation_peak_prediction_degenerate_fraction": 0.0,
        "oom": False,
    }
    mismatched = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in required_exact.items()
        if receipt.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"P3 stress receipt 冻结字段不合格: {mismatched}")
    git_commit = str(receipt.get("git_commit", ""))
    if len(git_commit) != 40 or any(character not in "0123456789abcdef" for character in git_commit):
        raise RuntimeError("P3 stress receipt git_commit 必须为完整小写 SHA-1")
    if int(receipt.get("optimizer_updates", -1)) != 400:
        raise RuntimeError("P3 stress optimizer updates 必须恰为 400")
    if int(receipt.get("full_validation_count", -1)) != 5:
        raise RuntimeError("P3 stress 完整 validation 必须恰为 5 次")
    peak_reserved_fraction = float(receipt.get("peak_reserved_fraction", math.nan))
    if not math.isfinite(peak_reserved_fraction) or not 0.0 <= peak_reserved_fraction <= 0.90:
        raise RuntimeError("P3 stress peak reserved fraction 非有限或超过 90%")
    expected_memory_status = "passed" if peak_reserved_fraction < 0.85 else "warning_passed"
    if receipt.get("memory_status") != expected_memory_status:
        raise RuntimeError("P3 stress memory_status 与 peak reserved fraction 不一致")
    for key in (
        "warm_train_samples_per_second",
        "forward_latency_ms",
        "forward_backward_latency_ms",
        "training_peak_allocated_mib",
        "training_peak_reserved_mib",
        "validation_peak_allocated_mib",
        "validation_peak_reserved_mib",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "total_device_memory_mib",
    ):
        value = float(receipt.get(key, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"P3 stress receipt {key} 必须为正的有限值")
    expected_peak_allocated = max(
        float(receipt["training_peak_allocated_mib"]),
        float(receipt["validation_peak_allocated_mib"]),
    )
    expected_peak_reserved = max(
        float(receipt["training_peak_reserved_mib"]),
        float(receipt["validation_peak_reserved_mib"]),
    )
    expected_fraction = expected_peak_reserved / float(receipt["total_device_memory_mib"])
    if not math.isclose(
        float(receipt["peak_allocated_mib"]), expected_peak_allocated, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("P3 stress peak allocated 未取 training/validation 最大值")
    if not math.isclose(
        float(receipt["peak_reserved_mib"]), expected_peak_reserved, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("P3 stress peak reserved 未取 training/validation 最大值")
    if not math.isclose(peak_reserved_fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("P3 stress peak reserved fraction 计算不一致")


def write_p3_stress_receipt(
    *,
    run_dir: Path,
    cfg: Any,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Any,
    history: list[dict[str, float | int]],
    metrics: pd.DataFrame,
    device: torch.device,
    runtime_summary: Mapping[str, Any],
) -> Path:
    """在 D4 isolation stress 末尾写入不可覆盖的资源与数值 receipt。"""

    variant = str(cfg.model.variant).strip().lower()
    if variant not in P3_VARIANTS or str(cfg.protocol.run_role) != "stress":
        raise ValueError("P3 stress receipt 只接受 P3 stress config")
    if device.type != "cuda":
        raise ValueError("P3 stress receipt 必须在 CUDA 上生成")
    receipt_path = Path(run_dir) / P3_STRESS_RECEIPT
    if receipt_path.exists():
        raise FileExistsError(f"P3 stress receipt 已存在，拒绝覆盖: {receipt_path}")

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    local_blocks = len(getattr(getattr(model, "base", None), "local_blocks", ()))
    history_frame = pd.DataFrame(history)
    numeric_history = history_frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    all_history_finite = bool(np.isfinite(numeric_history).all())
    optimizer_updates = int(history_frame["optimizer_update"].iloc[-1])
    warm_sps = history_frame.loc[history_frame["epoch"].astype(int) >= 2, "train_samples_per_second"]
    warm_train_samples_per_second = float(warm_sps.mean())

    summary = summarize_task_metrics(metrics).iloc[0]
    primary_values = np.asarray([float(summary[column]) for column in PRIMARY_COLUMNS], dtype=np.float64)
    checkpoint_finite = _all_checkpoint_tensors_finite(Path(run_dir))
    # P1/P3 使用完全相同的 batch-1 benchmark，复用 helper 保持 latency/gradient 口径一致。
    benchmark = _benchmark_p1_model(model, loss_fn, optimizer, data.train.loader, device=device, use_amp=True)

    torch.cuda.reset_peak_memory_stats(device)
    validation_predictions = collect_predictions(
        model,
        data.val.loader,
        device=device,
        max_windows=len(data.val.loader.dataset),
        use_amp=True,
    )
    validation_metrics = evaluate_task_predictions(
        validation_predictions,
        cfg,
        include_test_only=False,
        method=f"{variant}__p3_stress_validation_peak",
    )
    validation_summary = summarize_task_metrics(validation_metrics).iloc[0]
    validation_primary = np.asarray(
        [float(validation_summary[column]) for column in PRIMARY_COLUMNS], dtype=np.float64
    )
    validation_peak_allocated_mib = float(torch.cuda.max_memory_allocated(device)) / float(1024**2)
    validation_peak_reserved_mib = float(torch.cuda.max_memory_reserved(device)) / float(1024**2)

    training_peak_allocated_mib = float(runtime_summary["peak_allocated_mib"])
    training_peak_reserved_mib = float(runtime_summary["peak_reserved_mib"])
    peak_allocated_mib = max(training_peak_allocated_mib, validation_peak_allocated_mib)
    peak_reserved_mib = max(training_peak_reserved_mib, validation_peak_reserved_mib)
    total_device_memory_mib = float(runtime_summary["total_device_memory_mib"])
    peak_reserved_fraction = peak_reserved_mib / total_device_memory_mib
    memory_status = "passed" if peak_reserved_fraction < 0.85 else "warning_passed"

    ending_commit = _require_clean_git()
    run_manifest = json.loads((Path(run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
    contract = run_manifest.get("tf_w_v2_p3_contract", {})
    if run_manifest.get("git_commit") != ending_commit or run_manifest.get("git_dirty") is not False:
        raise RuntimeError("P3 stress 起止 Git identity 不一致")
    if (
        contract.get("variant") != variant
        or int(contract.get("local_bimamba2_blocks", -1)) != P3_LOCAL_BLOCK_COUNT
        or int(contract.get("trainable_parameters", -1)) != P3_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("P3 stress run manifest 的 D4 contract 不合格")

    receipt: dict[str, Any] = {
        "protocol": PROTOCOL,
        "phase": "p3_depth_stress",
        "variant": variant,
        "status": "passed",
        "complete": True,
        "run_role": "stress",
        "git_commit": ending_commit,
        "git_dirty": False,
        "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
        "source_cache_manifest_sha256": SOURCE_CACHE_MANIFEST_SHA256,
        "physical_batch_size": int(cfg.training.batch_size),
        "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
        "use_amp": bool(cfg.training.use_amp),
        "amp_dtype": str(cfg.training.amp_dtype),
        "trainable_parameters": int(trainable_parameters),
        "local_bimamba2_blocks": int(local_blocks),
        "optimizer_updates": optimizer_updates,
        "full_validation_count": int(len(history)),
        "all_history_finite": all_history_finite,
        "all_checkpoint_finite": checkpoint_finite,
        "all_optimizer_finite": checkpoint_finite,
        "all_primary_finite": bool(np.isfinite(primary_values).all() and np.isfinite(validation_primary).all()),
        "prediction_degenerate_fraction": float(summary["joint_prediction_degenerate_fraction"]),
        "validation_peak_prediction_degenerate_fraction": float(
            validation_summary["joint_prediction_degenerate_fraction"]
        ),
        "oom": False,
        "training_peak_allocated_mib": training_peak_allocated_mib,
        "training_peak_reserved_mib": training_peak_reserved_mib,
        "validation_peak_allocated_mib": validation_peak_allocated_mib,
        "validation_peak_reserved_mib": validation_peak_reserved_mib,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "total_device_memory_mib": total_device_memory_mib,
        "peak_reserved_fraction": peak_reserved_fraction,
        "memory_status": memory_status,
        "warm_train_samples_per_second": warm_train_samples_per_second,
        **benchmark,
    }
    try:
        _validate_p3_stress_receipt(receipt, variant)
    except BaseException as exc:
        failed = {
            **receipt,
            "status": "failed",
            "complete": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        receipt_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def _require_clean_git() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise RuntimeError("P3 stress/formal 要求干净 Git 工作树")
    return _git_commit()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
