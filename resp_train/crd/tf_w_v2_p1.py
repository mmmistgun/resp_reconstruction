from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from resp_train.crd.tf_v1_data import batch_tf_to_device
from resp_train.crd.tf_w_v2 import (
    CANDIDATE_LOCK_SHA256,
    P1_VARIANTS,
    PROTOCOL,
    SOURCE_CACHE_MANIFEST_SHA256,
    verify_p1_source_identity,
)
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
P1_ENGINEERING_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/engineering"
P1_FORMAL_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/formal"
PRIMARY_COLUMNS = (
    "whole_rr_abs_error_bpm_mean",
    "local_rr_mae_bpm_mean",
    "envelope_trajectory_mae_mean",
    "global_envelope_modulation_error_mean",
    "lag_aware_signed_pcc_mean",
)


def validate_p1_training_preflight(cfg: Any) -> str:
    """P1 stress/formal 只允许从干净 commit 与冻结 source 启动。"""

    variant = str(cfg.model.variant).strip().lower()
    role = str(cfg.protocol.run_role).strip().lower()
    if variant not in P1_VARIANTS:
        raise ValueError(f"P1 training preflight 未知 variant={variant!r}")
    verify_p1_source_identity()
    commit = _require_clean_git()
    if role == "stress":
        parent = P1_ENGINEERING_ROOT / variant
        if Path(str(cfg.outputs.run_root)).resolve() != parent.resolve():
            raise RuntimeError(f"P1 stress output root 漂移: {cfg.outputs.run_root}")
        existing = sorted(path for path in parent.glob("20*") if path.is_dir()) if parent.is_dir() else []
        if existing:
            raise RuntimeError(f"P1 isolation stress 已存在，拒绝重复运行: {existing}")
    elif role == "formal":
        parent = P1_FORMAL_ROOT / variant / f"seed_{int(cfg.training.seed)}"
        if Path(str(cfg.outputs.run_root)).resolve() != parent.resolve():
            raise RuntimeError(f"P1 formal output root 漂移: {cfg.outputs.run_root}")
        existing = sorted(path for path in parent.glob("20*") if path.is_dir()) if parent.is_dir() else []
        if existing:
            raise RuntimeError(f"P1 formal seed 已存在，拒绝重复运行: {existing}")
        validate_p1_formal_preflight(variant)
    elif role != "smoke":
        raise ValueError(f"P1 training preflight 不接受 run_role={role!r}")
    return commit


def validate_p1_formal_preflight(variant: str) -> dict[str, Any]:
    """任一 formal 前要求三个 P1 isolation stress 均完整通过。"""

    normalized = str(variant).strip().lower()
    if normalized not in P1_VARIANTS:
        raise ValueError(f"P1 formal preflight 未知 variant={variant!r}")
    verify_p1_source_identity()
    current_commit = _git_commit()
    receipts: dict[str, dict[str, Any]] = {}
    for candidate in P1_VARIANTS:
        parent = P1_ENGINEERING_ROOT / candidate
        run_dirs = sorted(path for path in parent.glob("20*") if path.is_dir()) if parent.is_dir() else []
        if len(run_dirs) != 1:
            raise RuntimeError(f"P1 formal 要求唯一 isolation stress run，当前 {candidate} 数量={len(run_dirs)}")
        receipt_path = run_dirs[0] / "p1_stress_receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"P1 formal 缺少 stress receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _validate_p1_stress_receipt(receipt, candidate)
        if receipt.get("git_commit") != current_commit:
            raise RuntimeError("P1 formal 与三个 isolation stress 不是同一 commit")
        receipts[candidate] = receipt
    return receipts[normalized]


def _validate_p1_stress_receipt(receipt: Mapping[str, Any], variant: str) -> None:
    normalized = str(variant).strip().lower()
    if normalized not in P1_VARIANTS:
        raise ValueError(f"P1 stress receipt 未知 variant={variant!r}")
    required_exact = {
        "protocol": PROTOCOL,
        "phase": "p1_isolation_stress",
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
        "benchmark_batch_size": 1,
        "all_benchmark_input_gradients_finite": True,
        "all_benchmark_parameter_gradients_finite": True,
        "all_history_finite": True,
        "all_checkpoint_finite": True,
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
        raise RuntimeError(f"P1 stress receipt 冻结字段不合格: {mismatched}")
    if int(receipt.get("optimizer_updates", -1)) != 400:
        raise RuntimeError("P1 stress optimizer updates 必须恰为 400")
    if int(receipt.get("full_validation_count", -1)) != 5:
        raise RuntimeError("P1 stress 完整 validation 必须恰为 5 次")
    peak_reserved_fraction = float(receipt.get("peak_reserved_fraction", math.nan))
    if not math.isfinite(peak_reserved_fraction) or peak_reserved_fraction < 0.0 or peak_reserved_fraction > 0.90:
        raise RuntimeError("P1 stress peak reserved fraction 非有限或超过 90%")
    for key in (
        "warm_train_samples_per_second",
        "forward_latency_ms",
        "forward_backward_latency_ms",
        "validation_peak_allocated_mib",
        "validation_peak_reserved_mib",
    ):
        value = float(receipt.get(key, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"P1 stress receipt {key} 必须为正的有限值")


def write_p1_stress_receipt(
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
    """在 isolation stress 末尾补齐冻结的数值、显存与 latency receipt。"""

    variant = str(cfg.model.variant).strip().lower()
    if variant not in P1_VARIANTS or str(cfg.protocol.run_role) != "stress":
        raise ValueError("P1 stress receipt 只接受 P1 stress config")
    if device.type != "cuda":
        raise ValueError("P1 stress receipt 必须在 CUDA 上生成")
    receipt_path = Path(run_dir) / "p1_stress_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"P1 stress receipt 已存在，拒绝覆盖: {receipt_path}")

    history_frame = pd.DataFrame(history)
    numeric_history = history_frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    all_history_finite = bool(np.isfinite(numeric_history).all())
    optimizer_updates = int(history_frame["optimizer_update"].iloc[-1])
    warm_sps = history_frame.loc[history_frame["epoch"].astype(int) >= 2, "train_samples_per_second"]
    warm_train_samples_per_second = float(warm_sps.mean())

    summary = summarize_task_metrics(metrics).iloc[0]
    primary_values = np.asarray([float(summary[column]) for column in PRIMARY_COLUMNS], dtype=np.float64)
    all_primary_finite = bool(np.isfinite(primary_values).all())
    degeneracy = float(summary["joint_prediction_degenerate_fraction"])
    all_checkpoint_finite = _all_checkpoint_tensors_finite(Path(run_dir))
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
        method=f"{variant}__p1_stress_validation_peak",
    )
    validation_summary = summarize_task_metrics(validation_metrics).iloc[0]
    validation_primary = np.asarray(
        [float(validation_summary[column]) for column in PRIMARY_COLUMNS], dtype=np.float64
    )
    validation_peak_allocated_mib = float(torch.cuda.max_memory_allocated(device)) / float(1024**2)
    validation_peak_reserved_mib = float(torch.cuda.max_memory_reserved(device)) / float(1024**2)

    peak_reserved_fraction = float(runtime_summary["peak_reserved_fraction"])
    memory_status = "passed" if peak_reserved_fraction < 0.85 else "warning_passed"
    ending_commit = _require_clean_git()
    run_manifest = json.loads((Path(run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("git_commit") != ending_commit or run_manifest.get("git_dirty") is not False:
        raise RuntimeError("P1 stress 起止 Git identity 不一致")
    receipt: dict[str, Any] = {
        "protocol": PROTOCOL,
        "phase": "p1_isolation_stress",
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
        "optimizer_updates": optimizer_updates,
        "full_validation_count": int(len(history)),
        "all_history_finite": all_history_finite,
        "all_checkpoint_finite": all_checkpoint_finite,
        "all_primary_finite": bool(all_primary_finite and np.isfinite(validation_primary).all()),
        "prediction_degenerate_fraction": degeneracy,
        "validation_peak_prediction_degenerate_fraction": float(
            validation_summary["joint_prediction_degenerate_fraction"]
        ),
        "oom": False,
        "peak_allocated_mib": float(runtime_summary["peak_allocated_mib"]),
        "peak_reserved_mib": float(runtime_summary["peak_reserved_mib"]),
        "total_device_memory_mib": float(runtime_summary["total_device_memory_mib"]),
        "peak_reserved_fraction": peak_reserved_fraction,
        "memory_status": memory_status,
        "warm_train_samples_per_second": warm_train_samples_per_second,
        "validation_peak_allocated_mib": validation_peak_allocated_mib,
        "validation_peak_reserved_mib": validation_peak_reserved_mib,
        **benchmark,
    }
    try:
        _validate_p1_stress_receipt(receipt, variant)
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


def _benchmark_p1_model(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Any,
    *,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    batch = next(iter(loader))
    sensor = batch["x"][:1].to(device).detach().requires_grad_(True)
    target = batch["target"][:1].to(device)
    tf = batch_tf_to_device({"tf": {key: value[:1] for key, value in batch["tf"].items()}}, device, non_blocking=True)
    if tf is None:
        raise RuntimeError("P1 benchmark 缺少 TF batch")
    amp_enabled = bool(use_amp and device.type == "cuda")

    def forward_only() -> None:
        with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            model(sensor.detach(), tf=tf)

    model.eval()
    for _ in range(2):
        forward_only()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(5):
        forward_only()
    torch.cuda.synchronize(device)
    forward_latency_ms = (time.perf_counter() - started) * 1000.0 / 5.0

    model.train()

    def forward_backward() -> tuple[bool, bool]:
        optimizer.zero_grad(set_to_none=True)
        sensor.grad = None
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            prediction = model(sensor, tf=tf)
        with torch.amp.autocast(device.type, enabled=False):
            loss, _ = loss_fn(prediction, target.float())
        loss.backward()
        input_finite = sensor.grad is not None and bool(torch.isfinite(sensor.grad).all())
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        parameter_finite = bool(gradients) and all(
            gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        return input_finite, parameter_finite

    input_checks: list[bool] = []
    parameter_checks: list[bool] = []
    forward_backward()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(3):
        input_finite, parameter_finite = forward_backward()
        input_checks.append(input_finite)
        parameter_checks.append(parameter_finite)
    torch.cuda.synchronize(device)
    forward_backward_latency_ms = (time.perf_counter() - started) * 1000.0 / 3.0
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {
        "benchmark_batch_size": 1,
        "forward_latency_ms": float(forward_latency_ms),
        "forward_backward_latency_ms": float(forward_backward_latency_ms),
        "all_benchmark_input_gradients_finite": bool(all(input_checks)),
        "all_benchmark_parameter_gradients_finite": bool(all(parameter_checks)),
    }


def _all_checkpoint_tensors_finite(run_dir: Path) -> bool:
    for name in ("checkpoint_best_local_rr.pt", "checkpoint_final.pt"):
        payload = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        tensors: list[torch.Tensor] = list(payload["model_state_dict"].values())
        for state in payload["optimizer_state_dict"].get("state", {}).values():
            tensors.extend(value for value in state.values() if isinstance(value, torch.Tensor))
        if not tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            return False
    return True


def _require_clean_git() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise RuntimeError("P1 stress/formal 要求干净 Git 工作树")
    return _git_commit()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
