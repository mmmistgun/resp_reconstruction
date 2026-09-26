"""W0 三因素结构对照的 synthetic GPU 验收与独立进程 benchmark。"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import check_crd_dependencies, crd_dependency_versions
from resp_train.crd.training import build_crd_optimizer, train_crd_one_epoch
from resp_train.losses.task import RespirationTaskLoss
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.w0_structural_factorial_v1_model import (
    ARMS,
    ARM_SPECS,
    build_w0_structural_factorial_model,
)
from resp_train.utils.run import set_seed


MAX_RESOURCE_ARM = "sfv1_conv20_tm3_ref2"
BATCH1_UPDATES = 1
PHYSICAL_BATCH_UPDATES = 3
LIFECYCLE_TRAIN_WINDOWS = 128
LIFECYCLE_VALIDATION_WINDOWS = 32
BENCHMARK_WARMUP = 3
BENCHMARK_REPEATS = 10
MEMORY_LIMIT_FRACTION = 0.80


def finite_tree(value: Any, *, label: str = "value") -> None:
    """递归拒绝 tensor/array/标量中的 NaN 与 Inf。"""

    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{label} 包含非有限 tensor")
        return
    if isinstance(value, np.ndarray):
        if not bool(np.isfinite(value).all()):
            raise FloatingPointError(f"{label} 包含非有限 array")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            finite_tree(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, label=f"{label}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError(f"{label} 包含非有限标量")


def runtime_preflight(device: str) -> dict[str, Any]:
    state = sf.git_state()
    if state["status_porcelain"]:
        raise RuntimeError("P2 GPU 工程验收要求干净 Git 工作树")
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None or not torch.cuda.is_available():
        raise RuntimeError("P2 GPU 工程验收要求显式可用的 cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("CRD 依赖不满足: " + "; ".join(problems))
    torch.cuda.set_device(resolved)
    properties = torch.cuda.get_device_properties(resolved)
    return {
        "git": state,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "dependencies": crd_dependency_versions(),
        "device": str(resolved),
        "device_name": properties.name,
        "device_total_bytes": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "amp_dtype": "bfloat16",
    }


def synthetic_batch(
    batch_size: int,
    seed: int,
    *,
    device: str | torch.device = "cpu",
    split: str = "synthetic",
    row_offset: int = 0,
) -> dict[str, Any]:
    """构造有稳定呼吸主频与幅度调制的 180 s 波形和冻结 W shape。"""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("synthetic batch_size 必须为正")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    time_grid = torch.arange(18_000, dtype=torch.float32).view(1, 1, -1) / 100.0
    phase = torch.rand(batch_size, 1, 1, generator=generator) * (2.0 * torch.pi)
    frequency = 0.16 + 0.18 * torch.rand(batch_size, 1, 1, generator=generator)
    effort = 1.0 + 0.30 * torch.sin(2.0 * torch.pi * 0.025 * time_grid + 0.4 * phase)
    target = effort * torch.sin(2.0 * torch.pi * frequency * time_grid + phase)
    carrier = 0.15 * effort * torch.sin(2.0 * torch.pi * 4.0 * time_grid + 0.25 * phase)
    sensor = target + carrier + 0.04 * torch.randn(target.shape, generator=generator)

    tf_time = torch.arange(360, dtype=torch.float32).view(1, 1, -1) / 2.0
    scale = torch.arange(97, dtype=torch.float32).view(1, -1, 1)
    w = torch.log1p(
        torch.square(
            torch.sin(2.0 * torch.pi * (0.035 + 0.004 * scale) * tf_time + phase)
        )
    )
    resolved = torch.device(device)
    batch = {
        "x": sensor.contiguous().to(resolved),
        "target": target.contiguous().to(resolved),
        "tf": {"w": w.contiguous().to(resolved)},
        "meta": {
            "dataset_row_id": torch.arange(row_offset, row_offset + batch_size, dtype=torch.int64),
            "split": [str(split)] * batch_size,
            "input_set": ["synthetic"] * batch_size,
            "samp_id": torch.arange(batch_size, dtype=torch.int64) % 4,
            "coupling_state_id": torch.arange(batch_size, dtype=torch.int64),
            "residual_quality_class": ["synthetic"] * batch_size,
        },
    }
    finite_tree(batch, label="synthetic_batch")
    if batch["x"].shape != (batch_size, 1, 18_000) or batch["tf"]["w"].shape != (
        batch_size,
        97,
        360,
    ):
        raise RuntimeError("synthetic batch shape 漂移")
    return batch


class _SingleBatchDataset:
    def __init__(self, count: int):
        self.count = int(count)

    def __len__(self) -> int:
        return self.count


class _SingleBatchLoader:
    """仅用于 P2：保持原生 trainer 的 loader 长度和 dataset 计数语义。"""

    def __init__(self, batch: dict[str, Any]):
        self.batch = batch
        self.dataset = _SingleBatchDataset(int(batch["x"].shape[0]))

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield self.batch


def synthetic_data_bundle(seed: int) -> tuple[Any, pd.DataFrame]:
    train_batch = synthetic_batch(
        LIFECYCLE_TRAIN_WINDOWS,
        int(seed) + 4100,
        split="synthetic_train",
        row_offset=0,
    )
    val_batch = synthetic_batch(
        LIFECYCLE_VALIDATION_WINDOWS,
        int(seed) + 4200,
        split="synthetic_validation",
        row_offset=1000,
    )
    validation_rows = pd.DataFrame(
        {
            "dataset_row_id": val_batch["meta"]["dataset_row_id"].numpy(),
            "samp_id": val_batch["meta"]["samp_id"].numpy(),
            "split": val_batch["meta"]["split"],
        }
    )
    audit = pd.DataFrame(
        [
            {"split": "synthetic_train", "windows": LIFECYCLE_TRAIN_WINDOWS},
            {"split": "synthetic_validation", "windows": LIFECYCLE_VALIDATION_WINDOWS},
        ]
    )
    bundle = SimpleNamespace(
        train=SimpleNamespace(loader=_SingleBatchLoader(train_batch)),
        val=SimpleNamespace(loader=_SingleBatchLoader(val_batch)),
        audit_summary=audit,
    )
    return bundle, validation_rows


class _SyntheticLifecycleExperiment(sf.StructuralFactorialExperiment):
    def __init__(self, cfg: DictConfig, data: Any, validation_rows: pd.DataFrame, arm: str):
        self._synthetic_data = data
        super().__init__(cfg, validation_rows, arm)

    def _build_data(self):
        return self._synthetic_data


def engineering_config(
    lock: Mapping[str, Any], *, arm: str, output_root: Path, device: str
) -> DictConfig:
    baseline = OmegaConf.create(lock["baselines"][str(sf.SEEDS[0])])
    cfg = sf.derived_config(
        baseline,
        arm=arm,
        output_root=output_root,
        device=device,
    )
    return cfg


def _native_update(
    model: torch.nn.Module,
    batch: dict[str, Any],
    loss_fn: RespirationTaskLoss,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    index: int,
) -> dict[str, float]:
    summary, completed = train_crd_one_epoch(
        model,
        [batch],
        loss_fn,
        optimizer,
        device=str(cfg.training.device),
        accumulation_steps=1,
        update_index=index,
        total_updates=sf.PLANNED_UPDATES,
        max_learning_rate=float(cfg.training.max_learning_rate),
        min_learning_rate=float(cfg.training.min_learning_rate),
        warmup_fraction=float(cfg.training.warmup_fraction),
        grad_clip_norm=float(cfg.training.grad_clip_norm),
        use_amp=True,
        show_progress=False,
    )
    if completed != index + 1 or not np.isclose(
        summary["loss"],
        summary["loss_sync"] + 0.25 * summary["loss_effort"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("P2 synthetic update/loss 合同错误")
    return summary


def _gradient_receipt(model: torch.nn.Module) -> dict[str, Any]:
    missing: list[str] = []
    nonfinite: list[str] = []
    nonzero = 0
    zero = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        if bool(parameter.grad.ne(0).any()):
            nonzero += 1
        else:
            zero += 1
    if missing or nonfinite or nonzero == 0:
        raise FloatingPointError(
            f"parameter gradient 异常: missing={missing[:5]} nonfinite={nonfinite[:5]} nonzero={nonzero}"
        )
    return {
        "trainable_parameter_tensors": nonzero + zero,
        "nonzero_gradient_tensors": nonzero,
        "zero_gradient_tensors": zero,
        "missing_gradient_tensors": 0,
        "nonfinite_gradient_tensors": 0,
    }


def _memory_receipt(device: str, *, enforce_limit: bool = True) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = reserved / total
    if enforce_limit and fraction > MEMORY_LIMIT_FRACTION:
        raise RuntimeError(
            f"peak reserved fraction={fraction:.6f} 超过 {MEMORY_LIMIT_FRACTION:.2f}"
        )
    return {
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "device_total_bytes": total,
        "peak_reserved_fraction": fraction,
        "limit_fraction": MEMORY_LIMIT_FRACTION,
    }


def multistep_acceptance(cfg: DictConfig, *, batch_size: int, updates: int) -> dict[str, Any]:
    arm = str(cfg.model.w0_structural_factorial_v1.arm)
    seed = int(cfg.training.seed)
    device = str(cfg.training.device)
    set_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_w0_structural_factorial_model(cfg).to(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, partition = build_crd_optimizer(model, cfg)
    active_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(partition.decay_names) | set(partition.no_decay_names) != active_names:
        raise RuntimeError("P2 optimizer active parameter 集合不完整")
    batch = synthetic_batch(batch_size, seed + 4300, device=device)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    summaries: list[dict[str, float]] = []
    gradients: list[dict[str, Any]] = []
    for index in range(int(updates)):
        summary = _native_update(model, batch, loss_fn, optimizer, cfg, index)
        finite_tree(summary, label=f"summary[{index}]")
        finite_tree(model.state_dict(), label="model")
        finite_tree(optimizer.state_dict(), label="optimizer")
        gradients.append(_gradient_receipt(model))
        summaries.append(summary)
    changed = sum(
        int(not torch.equal(parameter.detach(), initial[name]))
        for name, parameter in model.named_parameters()
    )
    if changed == 0:
        raise RuntimeError("P2 optimizer update 后没有参数变化")
    return {
        "arm": arm,
        "seed": seed,
        "batch_size": int(batch_size),
        "updates": int(updates),
        "trainable_parameters": ARM_SPECS[arm].trainable_parameters,
        "factor_covered_macs": ARM_SPECS[arm].factor_covered_macs,
        "optimizer_active_parameter_tensors": len(active_names),
        "changed_parameter_tensors": changed,
        "gradients": gradients,
        "summaries": summaries,
        **_memory_receipt(device),
    }


def run_synthetic_lifecycle(
    lock: Mapping[str, Any], *, output: Path, device: str
) -> dict[str, Any]:
    data, rows = synthetic_data_bundle(sf.SEEDS[0])
    cfg = engineering_config(
        lock,
        arm=MAX_RESOURCE_ARM,
        output_root=output / "native_lifecycle",
        device=device,
    )
    cfg.training.epochs = 1
    cfg.training.early_stopping_enabled = False
    cfg.training.batch_size = LIFECYCLE_TRAIN_WINDOWS
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    run_dir = _SyntheticLifecycleExperiment(cfg, data, rows, MAX_RESOURCE_ARM).train()
    history = pd.read_csv(run_dir / "train_history.csv")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    required = {
        "config.yaml",
        "run_manifest.json",
        "audit.csv",
        "optimizer_parameter_groups.json",
        "train_history.csv",
        "checkpoint_best_local_rr.pt",
        "checkpoint_final.pt",
        "metrics.csv",
        "metrics_summary.csv",
        "runtime_summary.json",
    }
    present = {path.name for path in run_dir.iterdir() if path.is_file()}
    if len(history) != 1 or int(history.iloc[0].optimizer_update) != 1:
        raise RuntimeError("P2 native lifecycle update/history 不完整")
    if len(metrics) != LIFECYCLE_VALIDATION_WINDOWS or not required.issubset(present):
        raise RuntimeError("P2 native lifecycle validation/checkpoint 产物不完整")
    sf.validate_metrics(metrics, rows)
    for checkpoint_name in ("checkpoint_best_local_rr.pt", "checkpoint_final.pt"):
        checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu")
        finite_tree(checkpoint["model_state_dict"], label=checkpoint_name)
        finite_tree(checkpoint["optimizer_state_dict"], label=checkpoint_name)
    return {
        "arm": MAX_RESOURCE_ARM,
        "train_windows": LIFECYCLE_TRAIN_WINDOWS,
        "validation_windows": LIFECYCLE_VALIDATION_WINDOWS,
        "physical_batch": LIFECYCLE_TRAIN_WINDOWS,
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "optimizer_updates": 1,
        "run_dir": str(run_dir),
        "required_files": sorted(required),
        "validation_metric_rows": len(metrics),
        "validation_degeneracy": sf.validate_metrics(metrics, rows),
        **_memory_receipt(device),
    }


def run_gpu_acceptance(device: str = "cuda:0") -> Path:
    lock, lock_hash = sf.load_lock()
    parent = sf.SOURCE_ROOT / sf.OUTPUT_ROOT / "gpu_acceptance"
    with sf.exclusive_attempt(parent, phase="gpu_acceptance", lock_hash=lock_hash) as output:
        sf.write_json(output / "environment.json", runtime_preflight(device))
        sf.write_json(
            output / "access_receipt.json",
            {
                "input": "deterministic synthetic tensors",
                "checkpoint": "fresh initialization",
                "dataset_index_read": False,
                "waveform_or_cache_read": False,
                "test_read": False,
            },
        )
        batch1: list[dict[str, Any]] = []
        for arm in ARMS:
            cfg = engineering_config(lock, arm=arm, output_root=output / "unused", device=device)
            receipt = multistep_acceptance(cfg, batch_size=1, updates=BATCH1_UPDATES)
            sf.write_json(output / f"batch1_{arm}.json", receipt)
            batch1.append(receipt)
        cfg = engineering_config(
            lock,
            arm=MAX_RESOURCE_ARM,
            output_root=output / "unused",
            device=device,
        )
        physical = multistep_acceptance(
            cfg,
            batch_size=LIFECYCLE_TRAIN_WINDOWS,
            updates=PHYSICAL_BATCH_UPDATES,
        )
        sf.write_json(output / "max_resource_batch128.json", physical)
        lifecycle = run_synthetic_lifecycle(lock, output=output, device=device)
        sf.write_json(output / "native_lifecycle.json", lifecycle)
        sf.write_json(output / "parameter_compute_report.json", sf.parameter_compute_report())
        sf.write_json(
            output / "gpu_acceptance.json",
            {
                "protocol": sf.PROTOCOL,
                "passed": True,
                "implementation_lock_sha256": lock_hash,
                "batch1": batch1,
                "max_resource_batch128": physical,
                "native_lifecycle": lifecycle,
            },
        )
    return output


def benchmark_worker(*, arm: str, mode: str, device: str, destination: Path) -> Path:
    lock, lock_hash = sf.load_lock()
    if arm not in ARMS or mode not in {"eval", "train"}:
        raise ValueError("P2 benchmark arm/mode 非法")
    destination = destination.resolve()
    parent = destination.parent
    benchmark_root = (sf.SOURCE_ROOT / sf.OUTPUT_ROOT / "benchmark").resolve()
    if destination.exists() or not parent.is_relative_to(benchmark_root):
        raise ValueError("P2 benchmark worker 输出 identity 非法或已存在")
    context = json.loads((parent / "lifecycle_started.json").read_text(encoding="utf-8"))
    if (
        context.get("phase") != "benchmark"
        or context.get("implementation_lock_sha256") != lock_hash
        or (parent / "lifecycle_completed.json").exists()
        or (parent / "lifecycle_failed.json").exists()
    ):
        raise RuntimeError("P2 benchmark parent lifecycle 非 active lock identity")
    environment = runtime_preflight(device)
    cfg = engineering_config(lock, arm=arm, output_root=parent / "unused", device=device)
    set_seed(sf.SEEDS[0])
    torch.cuda.empty_cache()
    model = build_w0_structural_factorial_model(cfg).to(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, _ = build_crd_optimizer(model, cfg)
    batch_size = 1 if mode == "eval" else 128
    batch = synthetic_batch(batch_size, sf.SEEDS[0] + 4400, device=device)
    times: list[float] = []
    total_iterations = BENCHMARK_WARMUP + BENCHMARK_REPEATS
    for index in range(total_iterations):
        if index == BENCHMARK_WARMUP:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if mode == "eval":
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch["x"], tf=batch["tf"])
        else:
            result = _native_update(model, batch, loss_fn, optimizer, cfg, index)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        finite_tree(result, label="benchmark_result")
        if index >= BENCHMARK_WARMUP:
            times.append(elapsed)
    if len(times) != BENCHMARK_REPEATS or not np.isfinite(times).all() or min(times) <= 0:
        raise RuntimeError("P2 benchmark timing 非法")
    memory = _memory_receipt(device)
    median = float(np.median(times))
    sf.write_json(
        destination,
        {
            "arm": arm,
            "mode": mode,
            "batch_size": batch_size,
            "warmup": BENCHMARK_WARMUP,
            "repeats": BENCHMARK_REPEATS,
            "seconds": times,
            "median_seconds": median,
            "iqr_seconds": float(np.quantile(times, 0.75) - np.quantile(times, 0.25)),
            "throughput_samples_per_second": float(batch_size / median),
            "trainable_parameters": ARM_SPECS[arm].trainable_parameters,
            "factor_covered_macs": ARM_SPECS[arm].factor_covered_macs,
            "environment": environment,
            "implementation_lock_sha256": lock_hash,
            **memory,
        },
    )
    return destination


def run_benchmark(device: str = "cuda:0") -> Path:
    _lock, lock_hash = sf.load_lock()
    parent = sf.SOURCE_ROOT / sf.OUTPUT_ROOT / "benchmark"
    with sf.exclusive_attempt(parent, phase="benchmark", lock_hash=lock_hash) as output:
        sf.write_json(output / "environment.json", runtime_preflight(device))
        sf.write_json(
            output / "access_receipt.json",
            {
                "input": "deterministic synthetic tensors",
                "dataset_index_read": False,
                "waveform_or_cache_read": False,
                "test_read": False,
                "process_isolation": "one fresh subprocess per arm and mode",
            },
        )
        measurements: list[dict[str, Any]] = []
        for mode in ("eval", "train"):
            ordered_arms = ARMS if mode == "eval" else tuple(reversed(ARMS))
            for arm in ordered_arms:
                target = output / f"{mode}_{arm}.json"
                command = [
                    sys.executable,
                    str(sf.ROOT / sf.SCRIPT_PATH),
                    "_benchmark-worker",
                    "--arm",
                    arm,
                    "--mode",
                    mode,
                    "--device",
                    device,
                    "--output",
                    str(target),
                ]
                with target.with_suffix(".log").open("x", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        cwd=sf.ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                measurements.append(json.loads(target.read_text(encoding="utf-8")))
        sf.write_json(
            output / "benchmark.json",
            {
                "protocol": sf.PROTOCOL,
                "implementation_lock_sha256": lock_hash,
                "scope": "synthetic cached-input model; eval batch1 and native train batch128; fresh subprocess per measurement",
                "measurements": measurements,
            },
        )
        sf.write_json(output / "parameter_compute_report.json", sf.parameter_compute_report())
    return output


__all__ = [
    "BATCH1_UPDATES",
    "BENCHMARK_REPEATS",
    "BENCHMARK_WARMUP",
    "LIFECYCLE_TRAIN_WINDOWS",
    "LIFECYCLE_VALIDATION_WINDOWS",
    "MAX_RESOURCE_ARM",
    "PHYSICAL_BATCH_UPDATES",
    "benchmark_worker",
    "engineering_config",
    "finite_tree",
    "multistep_acceptance",
    "run_benchmark",
    "run_gpu_acceptance",
    "run_synthetic_lifecycle",
    "runtime_preflight",
    "synthetic_batch",
    "synthetic_data_bundle",
]
