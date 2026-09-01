from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from resp_train.crd.config import check_crd_dependencies, crd_dependency_versions
from resp_train.crd.training import build_crd_optimizer
from resp_train.paper_evidence.center_context_config import load_center_context_config
from resp_train.paper_evidence.center_context_loss import CenterContextLoss
from resp_train.paper_evidence.center_context_model import CenterContextModel


P2_ACCEPTANCE_SCHEMA_VERSION = "paper-center-context-p2-gpu-acceptance-v1"
P2_ACCEPTANCE_ARMS = (
    ("c201_center60", 6000),
    ("c201_center60", 9000),
    ("c201_center60", 18000),
    ("w_reduced_center60", 6000),
    ("w_reduced_center60", 9000),
    ("w_reduced_center60", 18000),
)
P2_BATCH_ACCEPTANCE_ARM = ("w_reduced_center60", 18000)


def make_synthetic_center_batch(
    batch_size: int,
    input_samples: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """生成固定解析输入、中心目标与 49-scale 工程特征，不读取数据或 cache。"""

    batch = int(batch_size)
    samples = int(input_samples)
    if batch <= 0 or samples not in {6000, 9000, 18000}:
        raise ValueError("synthetic batch 要求 batch_size>0 且 input_samples 为 6000/9000/18000")
    resolved = torch.device(device)
    phase = torch.arange(batch, device=resolved, dtype=torch.float32).view(-1, 1, 1) * 0.037

    input_time = (
        torch.arange(samples, device=resolved, dtype=torch.float32).view(1, 1, -1)
        - (samples - 1) / 2.0
    ) / 100.0
    target_time = (
        torch.arange(6000, device=resolved, dtype=torch.float32).view(1, 1, -1) - 2999.5
    ) / 100.0

    def respiratory(time: torch.Tensor) -> torch.Tensor:
        effort = 1.0 + 0.22 * torch.sin(2.0 * math.pi * 0.06 * time + 0.3 * phase)
        return effort * torch.sin(2.0 * math.pi * 0.24 * time + phase)

    center_target = respiratory(target_time) + 0.04 * torch.sin(
        2.0 * math.pi * 0.11 * target_time + 0.5 * phase
    )
    respiration = respiratory(input_time)
    carrier = 0.20 * (1.0 + 0.30 * respiration) * torch.sin(
        2.0 * math.pi * 2.2 * input_time + 0.4 * phase
    )
    sensor = respiration + carrier + 0.07 * torch.sin(2.0 * math.pi * 5.1 * input_time)

    context = samples // 50
    tf_time = torch.arange(context, device=resolved, dtype=torch.float32).view(1, 1, -1) / 2.0
    scale = torch.arange(49, device=resolved, dtype=torch.float32).view(1, -1, 1)
    w = torch.log1p(
        torch.square(
            torch.sin(2.0 * math.pi * (0.05 + 0.015 * scale) * tf_time + phase)
        )
    )
    if sensor.shape != (batch, 1, samples) or center_target.shape != (batch, 1, 6000):
        raise RuntimeError("synthetic sensor/target shape 错误")
    if w.shape != (batch, 49, context):
        raise RuntimeError("synthetic W shape 错误")
    if not bool(torch.isfinite(sensor).all() and torch.isfinite(center_target).all() and torch.isfinite(w).all()):
        raise FloatingPointError("synthetic tensors 包含 NaN/Inf")
    return sensor.contiguous(), center_target.contiguous(), {"w": w.contiguous()}


def exercise_synthetic_arm(
    cfg: Any,
    *,
    variant: str,
    input_samples: int,
    batch_size: int,
    device: torch.device | str,
    optimizer_step: bool,
    mamba_factory: Callable[..., nn.Module] | None = None,
) -> dict[str, Any]:
    """执行一次真实计算图验收；测试可注入 shape-preserving Mamba factory。"""

    resolved = torch.device(device)
    model = CenterContextModel(
        variant,
        int(cfg.training.seed),
        mamba_factory=mamba_factory,
    ).to(resolved).train()
    loss_fn = CenterContextLoss(cfg).to(resolved)
    optimizer = build_crd_optimizer(model, cfg)[0] if optimizer_step else None
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)

    sensor, target, w = make_synthetic_center_batch(batch_size, input_samples, device=resolved)
    if not optimizer_step:
        sensor.requires_grad_(True)
    tf = w if variant == "w_reduced_center60" else None
    with torch.amp.autocast(
        resolved.type,
        dtype=torch.bfloat16,
        enabled=resolved.type == "cuda",
    ):
        prediction = model(sensor, tf=tf)
    with torch.amp.autocast(resolved.type, enabled=False):
        total, parts = loss_fn(prediction, target.float())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("synthetic total loss 非有限")
    total.backward()

    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameter_gradients or any(
        gradient is None or not bool(torch.isfinite(gradient).all())
        for gradient in parameter_gradients
    ):
        raise FloatingPointError("synthetic parameter gradient 缺失或非有限")
    if not optimizer_step and (sensor.grad is None or not bool(torch.isfinite(sensor.grad).all())):
        raise FloatingPointError("synthetic input gradient 缺失或非有限")
    if optimizer is not None:
        optimizer.step()
        _assert_optimizer_finite(model, optimizer)

    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
        peak_allocated = torch.cuda.max_memory_allocated(resolved) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(resolved) / (1024**2)
    else:
        peak_allocated = peak_reserved = None
    return {
        "variant": variant,
        "input_samples": int(input_samples),
        "input_sec": int(input_samples) // 100,
        "batch_size": int(batch_size),
        "optimizer_step": bool(optimizer_step),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "waveform_shape": list(prediction["waveform"].shape),
        "waveform_10hz_shape": list(prediction["waveform_10hz"].shape),
        "loss_total": float(total.detach().cpu()),
        "loss_sync": float(parts["loss_sync"].detach().cpu()),
        "loss_effort": float(parts["loss_effort"].detach().cpu()),
        "finite": True,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
    }


def run_p2_gpu_acceptance(
    *,
    config_path: str | Path,
    output_root: str | Path,
    device: str,
    command: list[str],
) -> Path:
    """运行单一 P2 GPU harness；只使用 synthetic tensors。"""

    cfg = load_center_context_config(config_path)
    if str(cfg.protocol.run_role) != "implementation":
        raise ValueError("P2 acceptance 必须从冻结 implementation config 派生")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("P2 acceptance 依赖不满足: " + "; ".join(problems))
    commit = _assert_clean_git()
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P2 acceptance 要求显式可用 CUDA device")
    torch.cuda.set_device(resolved)
    device_info = _cuda_identity(resolved)

    output = Path(output_root).resolve() / commit[:12]
    if output.exists():
        raise FileExistsError(f"P2 acceptance 输出禁止覆盖: {output}")
    output.mkdir(parents=True, exist_ok=False)
    lifecycle_path = output / "lifecycle.json"
    _write_json(lifecycle_path, {"status": "running", "git_commit": commit})

    results: list[dict[str, Any]] = []
    failure: dict[str, str] | None = None
    decision = "batch128_accepted"
    try:
        for variant, samples in P2_ACCEPTANCE_ARMS:
            results.append(
                exercise_synthetic_arm(
                    cfg,
                    variant=variant,
                    input_samples=samples,
                    batch_size=1,
                    device=resolved,
                    optimizer_step=False,
                )
            )
            _release_cuda()
        try:
            results.append(
                exercise_synthetic_arm(
                    cfg,
                    variant=P2_BATCH_ACCEPTANCE_ARM[0],
                    input_samples=P2_BATCH_ACCEPTANCE_ARM[1],
                    batch_size=128,
                    device=resolved,
                    optimizer_step=True,
                )
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            decision = "batch128_fallback_required"
            failure = {"type": type(exc).__name__, "message": str(exc), "stage": "w_reduced_180_batch128"}
            _release_cuda()

        receipt = _receipt(
            cfg=cfg,
            config_path=Path(config_path).resolve(),
            commit=commit,
            command=command,
            device_info=device_info,
            results=results,
            decision=decision,
            failure=failure,
        )
        _write_json(output / "acceptance_receipt.json", receipt)
        _write_json(lifecycle_path, {"status": "complete", "decision": decision, "git_commit": commit})
        _write_manifest(output)
    except BaseException as exc:
        _write_json(
            lifecycle_path,
            {
                "status": "failed",
                "git_commit": commit,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        _write_manifest(output)
        raise
    return output / "acceptance_receipt.json"


def _receipt(
    *,
    cfg: Any,
    config_path: Path,
    commit: str,
    command: list[str],
    device_info: dict[str, Any],
    results: list[dict[str, Any]],
    decision: str,
    failure: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "schema_version": P2_ACCEPTANCE_SCHEMA_VERSION,
        "protocol_id": str(cfg.protocol.name),
        "role": "engineering_only",
        "scientific_evidence": False,
        "decision": decision,
        "git_commit": commit,
        "git_dirty": False,
        "command": list(command),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "device": device_info,
        "dependencies": _dependency_versions(),
        "access": {
            "dataset_accessed": False,
            "cache_accessed": False,
            "train_split_accessed": False,
            "validation_split_accessed": False,
            "test_split_accessed": False,
            "checkpoint_accessed": False,
            "synthetic_tensor_used": True,
            "optimizer_step_used": any(bool(item.get("optimizer_step")) for item in results),
        },
        "batch1_expected_arms": len(P2_ACCEPTANCE_ARMS),
        "batch1_completed_arms": sum(
            int(item["batch_size"] == 1) for item in results
        ),
        "batch128_arm": {
            "variant": P2_BATCH_ACCEPTANCE_ARM[0],
            "input_samples": P2_BATCH_ACCEPTANCE_ARM[1],
        },
        "batch128_completed": any(int(item.get("batch_size", 0)) == 128 for item in results),
        "results": results,
        "failure": failure,
    }


def _assert_optimizer_finite(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("optimizer step 后 model parameter 非有限")
    tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    if not tensors or any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("optimizer state 缺失或非有限")


def _cuda_identity(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "logical_device": str(device),
        "name": properties.name,
        "total_memory_mib": properties.total_memory / (1024**2),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _assert_clean_git() -> str:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("无法读取 P2 Git identity")
    if status.stdout.strip():
        raise RuntimeError("P2 acceptance 要求干净 Git 工作树")
    return commit.stdout.strip()


def _write_manifest(output: Path) -> None:
    files = {
        path.name: {"size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(
        output / "artifact_manifest.json",
        {"schema_version": P2_ACCEPTANCE_SCHEMA_VERSION, "files": files},
    )


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = dict(crd_dependency_versions())
    for distribution in ("torch", "numpy", "omegaconf", "mamba-ssm", "causal-conv1d"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    versions["python"] = platform.python_version()
    return versions


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "P2_ACCEPTANCE_ARMS",
    "P2_ACCEPTANCE_SCHEMA_VERSION",
    "P2_BATCH_ACCEPTANCE_ARM",
    "exercise_synthetic_arm",
    "make_synthetic_center_batch",
    "run_p2_gpu_acceptance",
]
