from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import check_crd_dependencies
from resp_train.losses.task import RespirationTaskLoss
from resp_train.temporal.config import load_resp_temporal_config
from resp_train.temporal.model import build_resp_temporal_model, trainable_parameter_count


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/resp_temporal_v1/gpu_engineering_v1.yaml"
CONFIG_SCHEMA_VERSION = "rtm-v1-gpu-engineering-config-v1"
RECEIPT_SCHEMA_VERSION = "rtm-v1-gpu-engineering-receipt-v1"
MANIFEST_SCHEMA_VERSION = "rtm-v1-gpu-engineering-manifest-v1"
PROTOCOL_ID = "resp-temporal-v1-gpu-engineering-20260820"
FROZEN_CONFIG_SHA256 = "7ec9c26481a4369e7fea56a5c9c14f2bae1167423365466ce7ce4325bec35300"
EXPECTED_CANDIDATES = (
    "rtm_v1_t0_locked_stem_head",
    "rtm_v1_tcn_d9_h384",
    "rtm_v1_bimamba2_d96_l6",
    "rtm_v1_bilstm_h96_l2",
    "rtm_v1_multiscale_10_2_1_h384",
)
SUMMARY_FIELDS = (
    "candidate_id",
    "family",
    "trainable_parameters",
    "status",
    "batch1_inference_p50_ms",
    "batch1_inference_p90_ms",
    "fixed_batch_size",
    "fixed_forward_p50_ms",
    "fixed_forward_p90_ms",
    "fixed_forward_backward_p50_ms",
    "fixed_forward_backward_p90_ms",
    "fixed_peak_allocated_mib",
    "fixed_peak_reserved_mib",
    "physical_batch_size",
    "accumulation_steps",
    "effective_batch_size",
    "update_p50_seconds",
    "update_p90_seconds",
    "update_throughput_windows_per_second",
    "update_peak_allocated_mib",
    "update_peak_reserved_mib",
    "compute_only_6400_updates_hours",
    "planning_6400_updates_hours",
    "failure_stage",
    "oom_attempt_count",
)


@dataclass(frozen=True)
class GPUConfig:
    path: Path
    sha256: str
    raw: dict[str, Any]
    output_dir: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{context} 字段必须严格为 {sorted(expected)}，实际为 {sorted(actual)}")


def _repo_path(value: Any, *, context: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} 必须是仓库内相对路径")
    return REPO_ROOT / relative


def validate_gpu_engineering_config(raw: Mapping[str, Any]) -> None:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "role",
            "scientific_evidence",
            "authorization",
            "provenance",
            "candidates",
            "device",
            "synthetic",
            "benchmark",
            "optimizer",
            "output",
        },
        "gpu engineering config",
    )
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION or raw["protocol_id"] != PROTOCOL_ID:
        raise ValueError("GPU engineering config schema/protocol 不匹配")
    if raw["role"] != "engineering_only" or bool(raw["scientific_evidence"]):
        raise ValueError("GPU engineering 只能是 non-scientific engineering_only")

    authorization = raw["authorization"]
    _require_exact_keys(
        authorization,
        {"gpu_engineering", "formal_training", "validation_evaluation", "research_test_evaluation"},
        "authorization",
    )
    if authorization != {
        "gpu_engineering": True,
        "formal_training": False,
        "validation_evaluation": False,
        "research_test_evaluation": False,
    }:
        raise ValueError("GPU engineering authorization 越界")

    provenance = raw["provenance"]
    _require_exact_keys(
        provenance,
        {
            "protocol_path",
            "protocol_sha256",
            "implementation_receipt_path",
            "implementation_receipt_sha256",
            "verify_implementation_source_artifacts",
            "require_clean_git",
        },
        "provenance",
    )
    expected_provenance = {
        "protocol_path": "docs/experiments/resp_temporal_v1_gpu_engineering_protocol_20260820.md",
        "protocol_sha256": "7b1b946564989b4d3e8b7244d475349a4efe0ecefb023d1df5a53a6d5d5669fa",
        "implementation_receipt_path": "docs/experiments/resp_temporal_v1_cpu_implementation_receipt_20260820.json",
        "implementation_receipt_sha256": "6ef3ca0d48e1ac819ff55bab4247d542c8bae0adfddb6951c8a026e81fea7872",
        "verify_implementation_source_artifacts": True,
        "require_clean_git": True,
    }
    if dict(provenance) != expected_provenance:
        raise ValueError("GPU engineering provenance 漂移")

    candidates = raw["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError("GPU engineering 必须恰好包含五项 locked candidates")
    expected_candidate_records = [
        {
            "candidate_id": "rtm_v1_t0_locked_stem_head",
            "config_path": "configs/resp_temporal_v1/rtm_v1_t0_locked_stem_head.yaml",
            "config_sha256": "eb7cab872305459f62dac7eb0eb15e699690ea03e5ff7d33b581666ed7a31be6",
        },
        {
            "candidate_id": "rtm_v1_tcn_d9_h384",
            "config_path": "configs/resp_temporal_v1/rtm_v1_tcn_d9_h384.yaml",
            "config_sha256": "b838396563580bfe19d211bc9cfd86ad09228a1f72a28084c25e003b67077519",
        },
        {
            "candidate_id": "rtm_v1_bimamba2_d96_l6",
            "config_path": "configs/resp_temporal_v1/rtm_v1_bimamba2_d96_l6.yaml",
            "config_sha256": "18c98c28240fc5f983feef301e174b13b3be5829f7cb723acd032e412e7917f5",
        },
        {
            "candidate_id": "rtm_v1_bilstm_h96_l2",
            "config_path": "configs/resp_temporal_v1/rtm_v1_bilstm_h96_l2.yaml",
            "config_sha256": "05dde416538ea0920f9f4f01b6462baf8e95575551ad1a71c2089315d17b9bb7",
        },
        {
            "candidate_id": "rtm_v1_multiscale_10_2_1_h384",
            "config_path": "configs/resp_temporal_v1/rtm_v1_multiscale_10_2_1_h384.yaml",
            "config_sha256": "cdef3bd3ac3dd7a0e259d8182bfe7d4aa91944cc5391b945835bfe29bf5a7968",
        },
    ]
    if candidates != expected_candidate_records:
        raise ValueError("GPU engineering candidate identity/order/hash 漂移")

    device = raw["device"]
    _require_exact_keys(
        device,
        {
            "name",
            "expected_device_name",
            "minimum_total_memory_gib",
            "amp_dtype",
            "allow_tf32",
            "cudnn_benchmark",
        },
        "device",
    )
    if device != {
        "name": "cuda:0",
        "expected_device_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
        "minimum_total_memory_gib": 15.0,
        "amp_dtype": "bfloat16",
        "allow_tf32": False,
        "cudnn_benchmark": False,
    }:
        raise ValueError("GPU engineering device contract 漂移")

    synthetic = raw["synthetic"]
    _require_exact_keys(
        synthetic,
        {"seed", "sample_rate_hz", "duration_samples", "duration_sec", "input_channels", "generator"},
        "synthetic",
    )
    if synthetic != {
        "seed": 20260820,
        "sample_rate_hz": 100,
        "duration_samples": 18000,
        "duration_sec": 180,
        "input_channels": 1,
        "generator": "fixed_analytic_bcg_resp_v1",
    }:
        raise ValueError("GPU engineering synthetic contract 漂移")

    benchmark = raw["benchmark"]
    _require_exact_keys(
        benchmark,
        {
            "batch1_inference",
            "fixed_forward",
            "fixed_forward_backward",
            "physical_batch_schemes",
            "update_warmup",
            "update_repeats",
            "effective_batch_size",
            "formal_optimizer_updates",
            "planning_wall_time_multiplier",
            "latency_percentiles",
        },
        "benchmark",
    )
    if benchmark["batch1_inference"] != {"batch_size": 1, "warmup": 10, "repeats": 50}:
        raise ValueError("batch1 inference contract 漂移")
    if benchmark["fixed_forward"] != {"batch_size": 8, "warmup": 5, "repeats": 20}:
        raise ValueError("fixed forward contract 漂移")
    if benchmark["fixed_forward_backward"] != {"batch_size": 8, "warmup": 3, "repeats": 10}:
        raise ValueError("fixed forward/backward contract 漂移")
    schemes = benchmark["physical_batch_schemes"]
    if schemes != [
        {"physical_batch_size": 128, "accumulation_steps": 1},
        {"physical_batch_size": 64, "accumulation_steps": 2},
        {"physical_batch_size": 32, "accumulation_steps": 4},
    ]:
        raise ValueError("physical batch ladder 漂移")
    if any(int(item["physical_batch_size"]) * int(item["accumulation_steps"]) != 128 for item in schemes):
        raise ValueError("physical batch ladder 未保持 effective batch=128")
    if {
        "update_warmup": benchmark["update_warmup"],
        "update_repeats": benchmark["update_repeats"],
        "effective_batch_size": benchmark["effective_batch_size"],
        "formal_optimizer_updates": benchmark["formal_optimizer_updates"],
        "planning_wall_time_multiplier": benchmark["planning_wall_time_multiplier"],
        "latency_percentiles": benchmark["latency_percentiles"],
    } != {
        "update_warmup": 1,
        "update_repeats": 3,
        "effective_batch_size": 128,
        "formal_optimizer_updates": 6400,
        "planning_wall_time_multiplier": 1.25,
        "latency_percentiles": [50, 90],
    }:
        raise ValueError("GPU update/wall-time contract 漂移")

    optimizer = raw["optimizer"]
    _require_exact_keys(
        optimizer,
        {"name", "learning_rate", "betas", "eps", "weight_decay", "grad_clip_norm"},
        "optimizer",
    )
    if optimizer != {
        "name": "adamw",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
    }:
        raise ValueError("GPU engineering optimizer contract 漂移")

    output = raw["output"]
    _require_exact_keys(output, {"directory", "allow_overwrite", "csv_float_format"}, "output")
    if output != {
        "directory": "runs/resp_temporal_v1/gpu_engineering/rtm_v1_gpu_engineering_v1",
        "allow_overwrite": False,
        "csv_float_format": "%.12g",
    }:
        raise ValueError("GPU engineering output contract 漂移")


def load_gpu_engineering_config(path: str | Path = DEFAULT_CONFIG_PATH) -> GPUConfig:
    config_path = Path(path).resolve()
    if config_path != DEFAULT_CONFIG_PATH.resolve():
        raise ValueError(f"GPU engineering 只允许冻结配置: {DEFAULT_CONFIG_PATH}")
    if not config_path.is_file():
        raise FileNotFoundError(f"GPU engineering 配置不存在: {config_path}")
    actual_sha = sha256_file(config_path)
    if actual_sha != FROZEN_CONFIG_SHA256:
        raise ValueError(f"GPU engineering config SHA-256 漂移: {actual_sha}")
    container = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(container, dict):
        raise ValueError("GPU engineering config 顶层必须是 mapping")
    validate_gpu_engineering_config(container)
    return GPUConfig(
        path=config_path,
        sha256=actual_sha,
        raw=container,
        output_dir=_repo_path(container["output"]["directory"], context="output.directory"),
    )


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("无法读取 Git commit/dirty state")
    return commit.stdout.strip(), bool(status.stdout.strip())


def verify_frozen_provenance(config: GPUConfig) -> dict[str, Any]:
    provenance = config.raw["provenance"]
    protocol_path = _repo_path(provenance["protocol_path"], context="protocol_path")
    receipt_path = _repo_path(provenance["implementation_receipt_path"], context="implementation_receipt_path")
    for path, expected in (
        (protocol_path, provenance["protocol_sha256"]),
        (receipt_path, provenance["implementation_receipt_sha256"]),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"冻结 provenance hash 漂移: {path}: {actual} != {expected}")

    implementation_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_hashes = implementation_receipt.get("source_artifacts_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("CPU implementation receipt 缺少 source_artifacts_sha256")
    for relative, expected in source_hashes.items():
        source_path = _repo_path(relative, context="implementation source artifact")
        actual = sha256_file(source_path)
        if actual != expected:
            raise RuntimeError(f"CPU implementation source hash 漂移: {relative}: {actual} != {expected}")

    loaded_ids: list[str] = []
    config_hashes: dict[str, str] = {}
    for record in config.raw["candidates"]:
        candidate_path = _repo_path(record["config_path"], context="candidate config")
        actual = sha256_file(candidate_path)
        if actual != record["config_sha256"]:
            raise RuntimeError(f"candidate config hash 漂移: {candidate_path}")
        candidate_cfg = load_resp_temporal_config(candidate_path)
        candidate_id = str(candidate_cfg.model.variant)
        if candidate_id != record["candidate_id"]:
            raise RuntimeError(f"candidate config identity 漂移: {candidate_path}")
        loaded_ids.append(candidate_id)
        config_hashes[candidate_id] = actual
    if tuple(loaded_ids) != EXPECTED_CANDIDATES:
        raise RuntimeError("GPU engineering candidate order 漂移")

    git_commit, git_dirty = _git_identity()
    if git_dirty:
        raise RuntimeError("GPU engineering benchmark 要求干净 Git 工作树")
    implementation_commit = str(implementation_receipt.get("implementation_commit", ""))
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, git_commit],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("CPU implementation commit 不是当前 benchmark commit 的祖先")
    return {
        "git_commit": git_commit,
        "git_dirty": False,
        "protocol_path": str(protocol_path.relative_to(REPO_ROOT)),
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
        "implementation_receipt_sha256": sha256_file(receipt_path),
        "implementation_commit": implementation_commit,
        "verified_source_artifact_count": len(source_hashes),
        "candidate_config_sha256": config_hashes,
    }


def make_synthetic_batch(batch_size: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """不读取数据的固定解析工程波形；包含 displacement、carrier modulation 与 effort 变化。"""

    if int(batch_size) <= 0:
        raise ValueError("synthetic batch_size 必须为正")
    time = torch.arange(18000, device=device, dtype=torch.float32).view(1, 1, -1) / 100.0
    phase = torch.arange(batch_size, device=device, dtype=torch.float32).view(-1, 1, 1) * 0.037
    effort = 1.0 + 0.25 * torch.sin(2.0 * math.pi * 0.06 * time + 0.5 * phase)
    respiratory = effort * torch.sin(2.0 * math.pi * 0.24 * time + phase)
    carrier_low = 0.22 * (1.0 + 0.35 * respiratory) * torch.sin(2.0 * math.pi * 2.2 * time + 0.3 * phase)
    carrier_high = 0.14 * (1.0 + 0.30 * respiratory) * torch.sin(2.0 * math.pi * 5.2 * time + 0.7 * phase)
    bcg = respiratory + carrier_low + carrier_high + 0.06 * torch.sin(2.0 * math.pi * 0.09 * time)
    target = respiratory + 0.04 * torch.sin(2.0 * math.pi * 0.11 * time + phase)
    if bcg.shape != (batch_size, 1, 18000) or target.shape != (batch_size, 1, 18000):
        raise RuntimeError("synthetic batch shape 异常")
    if not bool(torch.isfinite(bcg).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError("synthetic batch 包含 NaN/Inf")
    return bcg.contiguous(), target.contiguous()


def summarize_latencies(milliseconds: Sequence[float]) -> dict[str, float]:
    values = np.asarray(list(milliseconds), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("latency samples 必须是非空有限正数")
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        "repeat_count": int(values.size),
    }


def _time_cuda_operation(
    operation,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup/repeats 无效")
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise FloatingPointError(f"CUDA timing 非有限/非正: {elapsed}")
        samples.append(elapsed)
    summary = summarize_latencies(samples)
    summary.update(
        {
            "warmup_count": int(warmup),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
        }
    )
    return summary


def _loss_config(candidate_cfg) -> Any:
    return OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "loss": OmegaConf.to_container(candidate_cfg.loss, resolve=True),
        }
    )


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def _assert_output_finite(output: Mapping[str, torch.Tensor], *, context: str) -> None:
    if set(output) != {"waveform", "waveform_10hz"}:
        raise RuntimeError(f"{context} output keys 漂移: {sorted(output)}")
    for name, value in output.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{context} {name} 包含 NaN/Inf")


def _assert_model_finite(model: torch.nn.Module, *, gradients: bool) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"模型参数非有限: {name}")
        if gradients and parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"模型梯度非有限: {name}")


def _build_model(candidate_cfg, device: torch.device):
    torch.manual_seed(int(candidate_cfg.model.initialization_seed))
    model = build_resp_temporal_model(candidate_cfg).to(device)
    observed = trainable_parameter_count(model)
    expected = int(candidate_cfg.model.expected_trainable_parameters)
    if observed != expected:
        raise RuntimeError(f"candidate 参数数漂移: {observed} != {expected}")
    return model


def _standard_benchmarks(candidate_cfg, device: torch.device, contract: Mapping[str, Any]) -> dict[str, Any]:
    model = _build_model(candidate_cfg, device)
    loss_fn = RespirationTaskLoss(_loss_config(candidate_cfg)).to(device)
    batch1, _ = make_synthetic_batch(1, device=device)
    fixed_batch_size = int(contract["fixed_forward"]["batch_size"])
    fixed_input, fixed_target = make_synthetic_batch(fixed_batch_size, device=device)
    last_output: dict[str, torch.Tensor] = {}
    last_loss: torch.Tensor | None = None

    model.eval()

    def inference_operation() -> None:
        nonlocal last_output
        with torch.no_grad(), _autocast():
            last_output = model(batch1)

    inference = _time_cuda_operation(
        inference_operation,
        device=device,
        warmup=int(contract["batch1_inference"]["warmup"]),
        repeats=int(contract["batch1_inference"]["repeats"]),
    )
    _assert_output_finite(last_output, context="batch1 inference")

    def forward_operation() -> None:
        nonlocal last_output
        with torch.no_grad(), _autocast():
            last_output = model(fixed_input)

    forward = _time_cuda_operation(
        forward_operation,
        device=device,
        warmup=int(contract["fixed_forward"]["warmup"]),
        repeats=int(contract["fixed_forward"]["repeats"]),
    )
    _assert_output_finite(last_output, context="fixed forward")

    model.train()

    def backward_operation() -> None:
        nonlocal last_output, last_loss
        model.zero_grad(set_to_none=True)
        with _autocast():
            last_output = model(fixed_input)
            last_loss, _ = loss_fn(last_output, fixed_target)
        last_loss.backward()

    forward_backward = _time_cuda_operation(
        backward_operation,
        device=device,
        warmup=int(contract["fixed_forward_backward"]["warmup"]),
        repeats=int(contract["fixed_forward_backward"]["repeats"]),
    )
    if last_loss is None or not bool(torch.isfinite(last_loss)):
        raise FloatingPointError("fixed forward/backward loss 非有限")
    _assert_output_finite(last_output, context="fixed forward/backward")
    _assert_model_finite(model, gradients=True)
    return {
        "batch1_inference": inference,
        "fixed_forward": forward,
        "fixed_forward_backward": forward_backward,
        "fixed_batch_size": fixed_batch_size,
        "last_loss": float(last_loss.detach().float().cpu()),
    }


def _optimizer(model: torch.nn.Module, optimizer_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["eps"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )


def _update_benchmark(
    candidate_cfg,
    *,
    device: torch.device,
    scheme: Mapping[str, Any],
    benchmark_cfg: Mapping[str, Any],
    optimizer_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    physical = int(scheme["physical_batch_size"])
    accumulation = int(scheme["accumulation_steps"])
    if physical * accumulation != int(benchmark_cfg["effective_batch_size"]):
        raise RuntimeError("update scheme 未保持 effective batch")
    model = _build_model(candidate_cfg, device).train()
    loss_fn = RespirationTaskLoss(_loss_config(candidate_cfg)).to(device)
    optimizer = _optimizer(model, optimizer_cfg)
    inputs, targets = make_synthetic_batch(physical, device=device)
    target_counts = loss_fn.target_component_counts(targets)
    sync_count = target_counts["loss_sync_count"].float() * accumulation
    effort_count = target_counts["loss_effort_count"].float() * accumulation
    if not bool(sync_count > 0) or not bool(effort_count > 0):
        raise RuntimeError("synthetic target 未形成完整 loss eligibility")
    last_loss: torch.Tensor | None = None
    last_output: dict[str, torch.Tensor] = {}

    def update_operation() -> None:
        nonlocal last_loss, last_output
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(accumulation):
            with _autocast():
                last_output = model(inputs)
                sums = loss_fn.differentiable_component_sums(last_output, targets)
                micro_loss = (
                    float(candidate_cfg.loss.sync_weight) * sums["loss_sync_sum"] / sync_count
                    + float(candidate_cfg.loss.effort_weight) * sums["loss_effort_sum"] / effort_count
                )
            micro_loss.backward()
            total_loss = total_loss + micro_loss.detach().float()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimizer_cfg["grad_clip_norm"]))
        optimizer.step()
        last_loss = total_loss

    timing = _time_cuda_operation(
        update_operation,
        device=device,
        warmup=int(benchmark_cfg["update_warmup"]),
        repeats=int(benchmark_cfg["update_repeats"]),
    )
    if last_loss is None or not bool(torch.isfinite(last_loss)):
        raise FloatingPointError("synthetic update loss 非有限")
    _assert_output_finite(last_output, context="synthetic update")
    _assert_model_finite(model, gradients=True)
    p50_seconds = float(timing["p50_ms"]) / 1000.0
    updates = int(benchmark_cfg["formal_optimizer_updates"])
    compute_hours = p50_seconds * updates / 3600.0
    return {
        "physical_batch_size": physical,
        "accumulation_steps": accumulation,
        "effective_batch_size": physical * accumulation,
        "timing": timing,
        "throughput_windows_per_second": float(physical * accumulation / p50_seconds),
        "compute_only_6400_updates_hours": float(compute_hours),
        "planning_6400_updates_hours": float(
            compute_hours * float(benchmark_cfg["planning_wall_time_multiplier"])
        ),
        "last_loss": float(last_loss.detach().cpu()),
    }


def _is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return isinstance(error, RuntimeError) and "cuda" in message and "out of memory" in message


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _empty_summary(candidate_id: str, family: str, parameters: int) -> dict[str, Any]:
    return {field: None for field in SUMMARY_FIELDS} | {
        "candidate_id": candidate_id,
        "family": family,
        "trainable_parameters": int(parameters),
        "status": "not_evaluated_on_current_hardware",
        "fixed_batch_size": 8,
        "effective_batch_size": 128,
        "oom_attempt_count": 0,
    }


def _summary_from_detail(detail: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(detail["candidate_id"])
    family = str(detail["family"])
    parameters = int(detail["trainable_parameters"])
    summary = _empty_summary(candidate_id, family, parameters)
    summary["status"] = detail["status"]
    summary["failure_stage"] = detail["failure_stage"]
    summary["oom_attempt_count"] = len(detail["oom_attempts"])
    standard = detail.get("standard")
    if isinstance(standard, Mapping):
        summary.update(
            {
                "batch1_inference_p50_ms": standard["batch1_inference"]["p50_ms"],
                "batch1_inference_p90_ms": standard["batch1_inference"]["p90_ms"],
                "fixed_batch_size": standard["fixed_batch_size"],
                "fixed_forward_p50_ms": standard["fixed_forward"]["p50_ms"],
                "fixed_forward_p90_ms": standard["fixed_forward"]["p90_ms"],
                "fixed_forward_backward_p50_ms": standard["fixed_forward_backward"]["p50_ms"],
                "fixed_forward_backward_p90_ms": standard["fixed_forward_backward"]["p90_ms"],
                "fixed_peak_allocated_mib": max(
                    standard["fixed_forward"]["peak_allocated_mib"],
                    standard["fixed_forward_backward"]["peak_allocated_mib"],
                ),
                "fixed_peak_reserved_mib": max(
                    standard["fixed_forward"]["peak_reserved_mib"],
                    standard["fixed_forward_backward"]["peak_reserved_mib"],
                ),
            }
        )
    selected = detail.get("selected_update_scheme")
    if isinstance(selected, Mapping):
        timing = selected["timing"]
        summary.update(
            {
                "physical_batch_size": selected["physical_batch_size"],
                "accumulation_steps": selected["accumulation_steps"],
                "effective_batch_size": selected["effective_batch_size"],
                "update_p50_seconds": timing["p50_ms"] / 1000.0,
                "update_p90_seconds": timing["p90_ms"] / 1000.0,
                "update_throughput_windows_per_second": selected["throughput_windows_per_second"],
                "update_peak_allocated_mib": timing["peak_allocated_mib"],
                "update_peak_reserved_mib": timing["peak_reserved_mib"],
                "compute_only_6400_updates_hours": selected["compute_only_6400_updates_hours"],
                "planning_6400_updates_hours": selected["planning_6400_updates_hours"],
            }
        )
    return summary


def benchmark_candidate(record: Mapping[str, Any], config: GPUConfig, device: torch.device) -> dict[str, Any]:
    candidate_path = _repo_path(record["config_path"], context="candidate config")
    candidate_cfg = load_resp_temporal_config(candidate_path)
    candidate_id = str(candidate_cfg.model.variant)
    detail: dict[str, Any] = {
        "candidate_id": candidate_id,
        "family": str(candidate_cfg.model.family),
        "trainable_parameters": int(candidate_cfg.model.expected_trainable_parameters),
        "status": "pending",
        "failure_stage": None,
        "standard": None,
        "oom_attempts": [],
        "selected_update_scheme": None,
    }
    try:
        detail["standard"] = _standard_benchmarks(candidate_cfg, device, config.raw["benchmark"])
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not _is_cuda_oom(exc):
            raise
        detail["status"] = "not_evaluated_on_current_hardware"
        detail["failure_stage"] = "standard_batch1_or_fixed_batch"
        detail["oom_attempts"].append(
            {"stage": detail["failure_stage"], "error_type": type(exc).__name__, "message": str(exc)[:500]}
        )
        _cleanup_cuda()
        detail["summary"] = _summary_from_detail(detail)
        return detail
    finally:
        _cleanup_cuda()

    for scheme in config.raw["benchmark"]["physical_batch_schemes"]:
        try:
            detail["selected_update_scheme"] = _update_benchmark(
                candidate_cfg,
                device=device,
                scheme=scheme,
                benchmark_cfg=config.raw["benchmark"],
                optimizer_cfg=config.raw["optimizer"],
            )
            detail["status"] = "passed"
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_oom(exc):
                raise
            detail["oom_attempts"].append(
                {
                    "stage": "physical_batch_update",
                    "physical_batch_size": int(scheme["physical_batch_size"]),
                    "accumulation_steps": int(scheme["accumulation_steps"]),
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                }
            )
        finally:
            _cleanup_cuda()
    if detail["status"] != "passed":
        detail["status"] = "not_evaluated_on_current_hardware"
        detail["failure_stage"] = "all_physical_batch_schemes_oom"
    detail["summary"] = _summary_from_detail(detail)
    return detail


def _cuda_identity(raw: Mapping[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU engineering 要求 CUDA available")
    device = torch.device(str(raw["device"]["name"]))
    if device.type != "cuda":
        raise RuntimeError("GPU engineering device 必须是 CUDA")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if properties.name != str(raw["device"]["expected_device_name"]):
        raise RuntimeError(
            f"GPU device 不匹配: {properties.name!r} != {raw['device']['expected_device_name']!r}"
        )
    total_gib = float(properties.total_memory / (1024**3))
    if total_gib < float(raw["device"]["minimum_total_memory_gib"]):
        raise RuntimeError(f"GPU 显存不足15 GiB合同: {total_gib:.3f} GiB")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("目标 GPU/环境不支持 bfloat16")
    torch.backends.cuda.matmul.allow_tf32 = bool(raw["device"]["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(raw["device"]["allow_tf32"])
    torch.backends.cudnn.benchmark = bool(raw["device"]["cudnn_benchmark"])
    return device, {
        "device": str(device),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "total_memory_gib": total_gib,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "bf16_supported": True,
        "amp_dtype": "bfloat16",
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "torch", "omegaconf", "mamba-ssm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["torch_cuda"] = torch.version.cuda
    versions["cudnn"] = str(torch.backends.cudnn.version())
    return versions


def _finite_counts(value: Any) -> dict[str, int]:
    counts = {"numeric_total": 0, "numeric_finite": 0, "numeric_null": 0, "numeric_nonfinite": 0}

    def visit(item: Any) -> None:
        if item is None:
            counts["numeric_total"] += 1
            counts["numeric_null"] += 1
        elif isinstance(item, bool):
            return
        elif isinstance(item, (int, float)):
            counts["numeric_total"] += 1
            if math.isfinite(float(item)):
                counts["numeric_finite"] += 1
            else:
                counts["numeric_nonfinite"] += 1
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for nested in item:
                visit(nested)

    visit(value)
    return counts


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, float_format: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            rendered = {}
            for field in SUMMARY_FIELDS:
                value = row[field]
                rendered[field] = float_format % value if isinstance(value, float) else value
            writer.writerow(rendered)


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "rows": rows,
    }


def validate_access_receipt(receipt: Mapping[str, Any]) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "created_utc",
            "status",
            "scientific_evidence",
            "execution",
            "device",
            "benchmark_contract",
            "access",
            "counts",
            "candidate_summary",
            "finite_audit",
            "artifacts",
            "manifest_sha256",
            "failure",
        },
        "GPU engineering receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION or receipt["protocol_id"] != PROTOCOL_ID:
        raise ValueError("GPU engineering receipt schema/protocol 不匹配")
    if bool(receipt["scientific_evidence"]):
        raise ValueError("GPU engineering receipt 不得标为科学证据")
    if receipt["status"] not in {"complete", "complete_with_unavailable_candidates", "failed"}:
        raise ValueError("GPU engineering receipt status 无效")
    execution = receipt["execution"]
    _require_exact_keys(
        execution,
        {
            "command",
            "cwd",
            "git_commit",
            "git_dirty",
            "python_version",
            "platform",
            "dependencies",
            "config_path",
            "config_sha256",
            "provenance",
        },
        "receipt.execution",
    )
    if bool(execution["git_dirty"]) or execution["config_sha256"] != FROZEN_CONFIG_SHA256:
        raise ValueError("GPU engineering receipt Git/config provenance 无效")
    access = receipt["access"]
    _require_exact_keys(
        access,
        {
            "dataset_accessed",
            "index_accessed",
            "train_split_accessed",
            "validation_accessed",
            "validation_target_accessed",
            "validation_prediction_accessed",
            "research_test_accessed",
            "checkpoint_accessed",
            "formal_training_used",
            "validation_evaluation_used",
            "research_test_evaluation_used",
            "synthetic_tensor_used",
            "synthetic_training_like_update_used",
            "gpu_used",
        },
        "receipt.access",
    )
    forbidden = {
        "dataset_accessed",
        "index_accessed",
        "train_split_accessed",
        "validation_accessed",
        "validation_target_accessed",
        "validation_prediction_accessed",
        "research_test_accessed",
        "checkpoint_accessed",
        "formal_training_used",
        "validation_evaluation_used",
        "research_test_evaluation_used",
    }
    if any(bool(access[key]) for key in forbidden):
        raise ValueError("GPU engineering receipt 显示越界访问/执行")
    if receipt["status"] != "failed" and not all(
        bool(access[key]) for key in ("synthetic_tensor_used", "synthetic_training_like_update_used", "gpu_used")
    ):
        raise ValueError("成功 GPU receipt 必须登记 synthetic/GPU 使用")
    counts = receipt["counts"]
    _require_exact_keys(
        counts,
        {"expected_candidates", "actual_candidates", "passed_candidates", "unavailable_candidates"},
        "receipt.counts",
    )
    if int(counts["expected_candidates"]) != 5:
        raise ValueError("GPU engineering expected candidates 必须为5")
    if receipt["status"] != "failed" and int(counts["actual_candidates"]) != 5:
        raise ValueError("成功 GPU engineering receipt 必须包含五项候选")
    summaries = receipt["candidate_summary"]
    if not isinstance(summaries, list):
        raise ValueError("GPU engineering candidate_summary 必须是 list")
    if receipt["status"] != "failed" and tuple(row["candidate_id"] for row in summaries) != EXPECTED_CANDIDATES:
        raise ValueError("成功 GPU engineering candidate_summary identity/order 不完整")
    for row in summaries:
        _require_exact_keys(row, set(SUMMARY_FIELDS), "receipt.candidate_summary")
        if row["status"] not in {"passed", "not_evaluated_on_current_hardware"}:
            raise ValueError("GPU engineering candidate status 无效")
    finite = receipt["finite_audit"]
    _require_exact_keys(
        finite,
        {"numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite"},
        "receipt.finite_audit",
    )
    if int(finite["numeric_nonfinite"]) != 0 or (
        int(finite["numeric_finite"]) + int(finite["numeric_null"]) != int(finite["numeric_total"])
    ):
        raise ValueError("GPU engineering numeric finite audit 不闭合")
    if not isinstance(receipt["artifacts"], list) or not receipt["artifacts"]:
        raise ValueError("GPU engineering receipt artifacts 不能为空")
    for artifact in receipt["artifacts"]:
        _require_exact_keys(artifact, {"filename", "sha256", "size_bytes", "rows"}, "receipt.artifact")
        if len(str(artifact["sha256"])) != 64:
            raise ValueError("GPU engineering artifact SHA-256 无效")
    if receipt["status"] == "failed" and not isinstance(receipt["failure"], Mapping):
        raise ValueError("failed GPU receipt 必须记录 failure")
    if receipt["status"] != "failed" and receipt["failure"] is not None:
        raise ValueError("成功 GPU receipt failure 必须为 null")


def _finalize_output(
    temporary: Path,
    *,
    config: GPUConfig,
    command: str,
    provenance: Mapping[str, Any],
    device_info: Mapping[str, Any] | None,
    details: Sequence[Mapping[str, Any]],
    failure: BaseException | None,
) -> Path:
    summaries = [dict(detail["summary"]) for detail in details]
    _json_write(temporary / "candidate_engineering.json", list(details))
    _write_csv(
        temporary / "candidate_engineering.csv",
        summaries,
        float_format=str(config.raw["output"]["csv_float_format"]),
    )
    artifacts = [
        _artifact_record(temporary / "resolved_config.json"),
        _artifact_record(temporary / "candidate_engineering.json", rows=len(details)),
        _artifact_record(temporary / "candidate_engineering.csv", rows=len(summaries)),
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifacts": artifacts,
    }
    _json_write(temporary / "artifact_manifest.json", manifest)
    manifest_record = _artifact_record(temporary / "artifact_manifest.json")
    artifacts.append(manifest_record)
    passed = sum(detail["status"] == "passed" for detail in details)
    unavailable = sum(detail["status"] == "not_evaluated_on_current_hardware" for detail in details)
    if failure is not None:
        status = "failed"
    elif unavailable:
        status = "complete_with_unavailable_candidates"
    else:
        status = "complete"
    finite_audit = _finite_counts(details)
    access = {
        "dataset_accessed": False,
        "index_accessed": False,
        "train_split_accessed": False,
        "validation_accessed": False,
        "validation_target_accessed": False,
        "validation_prediction_accessed": False,
        "research_test_accessed": False,
        "checkpoint_accessed": False,
        "formal_training_used": False,
        "validation_evaluation_used": False,
        "research_test_evaluation_used": False,
        "synthetic_tensor_used": device_info is not None,
        "synthetic_training_like_update_used": any(
            detail.get("standard") is not None or detail.get("selected_update_scheme") is not None
            for detail in details
        ),
        "gpu_used": device_info is not None,
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scientific_evidence": False,
        "execution": {
            "command": command,
            "cwd": str(REPO_ROOT),
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "config_path": str(config.path.relative_to(REPO_ROOT)),
            "config_sha256": config.sha256,
            "provenance": dict(provenance),
        },
        "device": None if device_info is None else dict(device_info),
        "benchmark_contract": config.raw["benchmark"],
        "access": access,
        "counts": {
            "expected_candidates": 5,
            "actual_candidates": len(details),
            "passed_candidates": passed,
            "unavailable_candidates": unavailable,
        },
        "candidate_summary": summaries,
        "finite_audit": finite_audit,
        "artifacts": artifacts,
        "manifest_sha256": manifest_record["sha256"],
        "failure": None
        if failure is None
        else {"error_type": type(failure).__name__, "message": str(failure)[:1000]},
    }
    validate_access_receipt(receipt)
    _json_write(temporary / "access_receipt.json", receipt)
    receipt_hash = sha256_file(temporary / "access_receipt.json")
    (temporary / "access_receipt.sha256").write_text(
        f"{receipt_hash}  access_receipt.json\n", encoding="utf-8"
    )
    if config.output_dir.exists():
        raise FileExistsError(f"GPU engineering 输出在运行期间出现，拒绝覆盖: {config.output_dir}")
    os.replace(temporary, config.output_dir)
    return config.output_dir / "access_receipt.json"


def run_gpu_engineering(*, config_path: str | Path = DEFAULT_CONFIG_PATH, command: str) -> Path:
    """运行冻结 synthetic GPU engineering；调用者必须显式手动执行。"""

    config = load_gpu_engineering_config(config_path)
    if config.output_dir.exists():
        raise FileExistsError(f"GPU engineering 输出禁止覆盖: {config.output_dir}")
    provenance = verify_frozen_provenance(config)
    dependency_problems = check_crd_dependencies()
    if dependency_problems:
        raise RuntimeError("; ".join(dependency_problems))
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_dir.parent / f".{config.output_dir.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    details: list[dict[str, Any]] = []
    device_info: dict[str, Any] | None = None
    try:
        _json_write(
            temporary / "resolved_config.json",
            {
                "protocol_id": PROTOCOL_ID,
                "config": config.raw,
                "config_sha256": config.sha256,
                "provenance": provenance,
            },
        )
        try:
            device, device_info = _cuda_identity(config.raw)
            torch.manual_seed(int(config.raw["synthetic"]["seed"]))
            torch.cuda.manual_seed_all(int(config.raw["synthetic"]["seed"]))
            for index, candidate in enumerate(config.raw["candidates"], start=1):
                print(f"RTM-v1 GPU engineering {index}/5: {candidate['candidate_id']}", flush=True)
                detail = benchmark_candidate(candidate, config, device)
                details.append(detail)
                _json_write(temporary / "candidate_engineering.json", details)
        except BaseException as exc:
            receipt_path = _finalize_output(
                temporary,
                config=config,
                command=command,
                provenance=provenance,
                device_info=device_info,
                details=details,
                failure=exc,
            )
            print(f"GPU engineering failure receipt: {receipt_path}", flush=True)
            raise
        receipt_path = _finalize_output(
            temporary,
            config=config,
            command=command,
            provenance=provenance,
            device_info=device_info,
            details=details,
            failure=None,
        )
        return receipt_path
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        _cleanup_cuda()
