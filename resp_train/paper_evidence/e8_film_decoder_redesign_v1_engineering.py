"""E8 的 synthetic GPU acceptance 与独立进程 benchmark。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from resp_train.crd.config import check_crd_dependencies, crd_dependency_versions
from resp_train.crd.training import build_crd_optimizer, train_crd_one_epoch
from resp_train.losses.task import RespirationTaskLoss
from resp_train.paper_evidence import e8_film_decoder_redesign_v1 as e8
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_model import (
    ARMS,
    ARM_SPECS,
    build_e8_film_decoder_redesign_model,
)
from resp_train.utils.run import set_seed


MAX_RESOURCE_ARM = "e8_res192_temporal"
BATCH1_UPDATES = 3
PHYSICAL_BATCH_UPDATES = 3
LIFECYCLE_TRAIN_WINDOWS = 128
LIFECYCLE_VALIDATION_WINDOWS = 32
BENCHMARK_WARMUP = 5
BENCHMARK_REPEATS = 20
MEMORY_LIMIT_FRACTION = 0.80

ENGINEERING_PATH = Path(
    "resp_train/paper_evidence/e8_film_decoder_redesign_v1_engineering.py"
)
SCRIPT_PATH = Path("scripts/run_e8_film_decoder_redesign_v1.py")
MODEL_PATH = Path("resp_train/paper_evidence/e8_film_decoder_redesign_v1_model.py")
CONTROL_PATH = Path("resp_train/paper_evidence/e8_film_decoder_redesign_v1.py")
TEST_PATH = Path("tests/test_e8_film_decoder_redesign_v1.py")


def finite_tree(value: Any, *, label: str = "value") -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=e8.ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def engineering_identity() -> dict[str, Any]:
    """以 clean commit 和关键文件身份固定 P2，不额外建立正式实验锁。"""

    paths = (
        e8.SPEC_PATH,
        e8.PROTOCOL_PATH,
        MODEL_PATH,
        CONTROL_PATH,
        ENGINEERING_PATH,
        SCRIPT_PATH,
        TEST_PATH,
        e8.W0_CONFIG_PATH,
        Path("resp_train/crd/model.py"),
        Path("resp_train/crd/tf_v1_model.py"),
        Path("resp_train/crd/blocks.py"),
        Path("resp_train/crd/experiment.py"),
        Path("resp_train/crd/training.py"),
        Path("resp_train/losses/task.py"),
    )
    missing = [str(path) for path in paths if not (e8.ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"E8 P2 缺少关键文件: {missing}")
    state = _git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E8 P2 要求干净 Git 工作树")
    files = {str(path): _identity(e8.ROOT / path) for path in paths}
    digest = hashlib.sha256()
    for relative, identity in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(identity["sha256"]))
    return {
        "protocol": e8.PROTOCOL,
        "git": state,
        "files": files,
        "identity_sha256": digest.hexdigest(),
    }


@contextmanager
def exclusive_attempt(parent: Path, *, phase: str, identity_hash: str) -> Iterator[Path]:
    """排他创建不可覆盖 attempt；成功冻结，失败保留现场。"""

    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".execution_{identity_hash}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("相同 E8 P2 identity 正在运行") from exc
        for receipt_path in parent.glob("*/freeze_receipt.json"):
            completed_path = receipt_path.parent
            freeze = json.loads(receipt_path.read_text(encoding="utf-8"))
            manifest_path = completed_path / "manifest.json"
            if _identity(manifest_path) != freeze.get("manifest"):
                raise RuntimeError(f"E8 P2 freeze receipt 身份错误: {completed_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("protocol") == e8.PROTOCOL
                and manifest.get("phase") == phase
                and manifest.get("engineering_identity_sha256") == identity_hash
                and manifest.get("status") == "completed"
            ):
                raise FileExistsError(f"相同 E8 P2 {phase} 已完成: {completed_path}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = parent / f"{phase}_{identity_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
        path.mkdir(exist_ok=False)
        context = {
            "protocol": e8.PROTOCOL,
            "phase": phase,
            "engineering_identity_sha256": identity_hash,
            "command": sys.argv,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(path / "lifecycle_started.json", {**context, "status": "running"})
        try:
            yield path
            _write_json(
                path / "lifecycle_completed.json",
                {
                    **context,
                    "status": "completed",
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            manifest = {
                **context,
                "status": "completed",
                "files": {
                    str(file.relative_to(path)): _identity(file)
                    for file in sorted(path.rglob("*"))
                    if file.is_file()
                },
            }
            _write_json(path / "manifest.json", manifest)
            _write_json(
                path / "freeze_receipt.json",
                {"protocol": e8.PROTOCOL, "manifest": _identity(path / "manifest.json")},
            )
        except BaseException as exc:
            _write_json(
                path / "lifecycle_failed.json",
                {
                    **context,
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def runtime_preflight(device: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    current = engineering_identity()
    if current["identity_sha256"] != identity["identity_sha256"]:
        raise RuntimeError("E8 P2 engineering identity 在执行中漂移")
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None or not torch.cuda.is_available():
        raise RuntimeError("E8 P2 要求显式可用的 cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("CRD 依赖不满足: " + "; ".join(problems))
    torch.cuda.set_device(resolved)
    properties = torch.cuda.get_device_properties(resolved)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("E8 P2 GPU 必须支持 BF16")
    return {
        "engineering_identity": identity,
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
    """生成确定性的 180 s 输入、目标与冻结 W shape。"""

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
        torch.square(torch.sin(2.0 * torch.pi * (0.035 + 0.004 * scale) * tf_time + phase))
    )
    resolved = torch.device(device)
    batch = {
        "x": sensor.contiguous().to(resolved),
        "target": target.contiguous().to(resolved),
        "tf": {"w": w.contiguous().to(resolved)},
        "meta": {
            "dataset_row_id": torch.arange(row_offset, row_offset + batch_size),
            "split": [str(split)] * batch_size,
            "input_set": ["synthetic"] * batch_size,
            "samp_id": torch.arange(batch_size) % 4,
            "coupling_state_id": torch.arange(batch_size),
            "residual_quality_class": ["synthetic"] * batch_size,
        },
    }
    finite_tree(batch, label="synthetic_batch")
    if batch["x"].shape != (batch_size, 1, 18_000) or batch["tf"]["w"].shape != (
        batch_size,
        97,
        360,
    ):
        raise RuntimeError("E8 synthetic batch shape 漂移")
    return batch


class _SingleBatchDataset:
    def __init__(self, count: int):
        self.count = int(count)

    def __len__(self) -> int:
        return self.count


class _SingleBatchLoader:
    def __init__(self, batch: dict[str, Any]):
        self.batch = batch
        self.dataset = _SingleBatchDataset(int(batch["x"].shape[0]))

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield self.batch


def _engineering_config(*, arm: str, seed: int, output_root: Path, device: str) -> DictConfig:
    baseline = e8.load_w0_baseline(seed)
    cfg = e8.derived_config(
        baseline,
        arm=arm,
        output_root=output_root,
        device=device,
    )
    e8.validate_config(
        cfg,
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
        total_updates=e8.PLANNED_UPDATES,
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
        raise RuntimeError("E8 P2 native update/loss 合同错误")
    return summary


def _memory_receipt(device: str) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = reserved / total
    if fraction > MEMORY_LIMIT_FRACTION:
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


def _factor_prefixes(arm: str) -> dict[str, str]:
    spec = ARM_SPECS[arm]
    prefixes = {"film_projection": "branches.w.final_projection."}
    if spec.condition_refiner != "direct":
        prefixes["condition_refiner"] = "branches.w.parameter_fill."
    if spec.decoder != "single":
        prefixes["decoder_residual"] = "base.decoder_residual."
    return prefixes


def multistep_acceptance(cfg: DictConfig, *, batch_size: int, updates: int) -> dict[str, Any]:
    arm = str(cfg.model.e8_film_decoder_redesign_v1.arm)
    seed = int(cfg.training.seed)
    device = str(cfg.training.device)
    set_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_e8_film_decoder_redesign_model(cfg).to(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, partition = build_crd_optimizer(model, cfg)
    active = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(partition.decay_names) | set(partition.no_decay_names) != active:
        raise RuntimeError("E8 P2 optimizer parameter 集合不完整")
    batch = synthetic_batch(batch_size, seed + 4300, device=device)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    summaries: list[dict[str, float]] = []
    gradient_steps: list[dict[str, int]] = []
    for index in range(int(updates)):
        summary = _native_update(model, batch, loss_fn, optimizer, cfg, index)
        finite_tree(summary, label=f"summary[{index}]")
        finite_tree(model.state_dict(), label="model")
        finite_tree(optimizer.state_dict(), label="optimizer")
        gradient_steps.append(
            {
                label: sum(
                    int(parameter.grad is not None and bool(parameter.grad.ne(0).any()))
                    for name, parameter in model.named_parameters()
                    if name.startswith(prefix)
                )
                for label, prefix in _factor_prefixes(arm).items()
            }
        )
        summaries.append(summary)
    changed_by_factor = {
        label: sum(
            int(not torch.equal(parameter.detach(), initial[name]))
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        )
        for label, prefix in _factor_prefixes(arm).items()
    }
    if any(count == 0 for count in changed_by_factor.values()):
        raise RuntimeError(f"E8 P2 factor 参数未更新: {changed_by_factor}")
    if any(count == 0 for count in gradient_steps[-1].values()):
        raise RuntimeError(f"E8 P2 最后一步 factor 梯度未开启: {gradient_steps[-1]}")
    return {
        "arm": arm,
        "seed": seed,
        "batch_size": int(batch_size),
        "updates": int(updates),
        "trainable_parameters": ARM_SPECS[arm].trainable_parameters,
        "factor_covered_macs": ARM_SPECS[arm].factor_covered_macs,
        "optimizer_active_parameter_tensors": len(active),
        "gradient_steps": gradient_steps,
        "changed_by_factor": changed_by_factor,
        "summaries": summaries,
        **_memory_receipt(device),
    }


class _SyntheticLifecycleExperiment(e8.E8FilmDecoderExperiment):
    def __init__(self, cfg: DictConfig, data: Any):
        self._synthetic_data = data
        super().__init__(cfg)

    def _build_data(self):
        return self._synthetic_data


def run_synthetic_lifecycle(*, output: Path, device: str) -> dict[str, Any]:
    seed = e8.SEEDS[0]
    train_batch = synthetic_batch(
        LIFECYCLE_TRAIN_WINDOWS,
        seed + 4100,
        split="synthetic_train",
    )
    val_batch = synthetic_batch(
        LIFECYCLE_VALIDATION_WINDOWS,
        seed + 4200,
        split="synthetic_validation",
        row_offset=1000,
    )
    audit = pd.DataFrame(
        [
            {"split": "synthetic_train", "windows": LIFECYCLE_TRAIN_WINDOWS},
            {"split": "synthetic_validation", "windows": LIFECYCLE_VALIDATION_WINDOWS},
        ]
    )
    data = SimpleNamespace(
        train=SimpleNamespace(loader=_SingleBatchLoader(train_batch)),
        val=SimpleNamespace(loader=_SingleBatchLoader(val_batch)),
        audit_summary=audit,
    )
    cfg = _engineering_config(
        arm=MAX_RESOURCE_ARM,
        seed=seed,
        output_root=output / "native_lifecycle",
        device=device,
    )
    cfg.training.epochs = 1
    cfg.training.early_stopping_enabled = False
    cfg.training.batch_size = LIFECYCLE_TRAIN_WINDOWS
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    run_dir = _SyntheticLifecycleExperiment(cfg, data).train()
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
    if (
        len(history) != 1
        or int(history.iloc[0].optimizer_update) != 1
        or len(metrics) != LIFECYCLE_VALIDATION_WINDOWS
        or not required.issubset(present)
    ):
        raise RuntimeError("E8 P2 native lifecycle 产物不完整")
    for checkpoint_name in ("checkpoint_best_local_rr.pt", "checkpoint_final.pt"):
        checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu")
        finite_tree(checkpoint["model_state_dict"], label=checkpoint_name)
        finite_tree(checkpoint["optimizer_state_dict"], label=checkpoint_name)
    return {
        "arm": MAX_RESOURCE_ARM,
        "train_windows": LIFECYCLE_TRAIN_WINDOWS,
        "validation_windows": LIFECYCLE_VALIDATION_WINDOWS,
        "optimizer_updates": 1,
        "run_dir": str(run_dir),
        "required_files": sorted(required),
        "validation_metric_rows": len(metrics),
        **_memory_receipt(device),
    }


def run_gpu_acceptance(device: str = "cuda:0") -> Path:
    identity = engineering_identity()
    identity_hash = str(identity["identity_sha256"])
    parent = e8.SOURCE_ROOT / e8.OUTPUT_ROOT / "gpu_acceptance"
    with exclusive_attempt(parent, phase="gpu_acceptance", identity_hash=identity_hash) as output:
        _write_json(output / "environment.json", runtime_preflight(device, identity))
        _write_json(
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
        for seed in e8.SEEDS:
            for arm in ARMS:
                cfg = _engineering_config(
                    arm=arm,
                    seed=seed,
                    output_root=output / "unused",
                    device=device,
                )
                receipt = multistep_acceptance(
                    cfg,
                    batch_size=1,
                    updates=BATCH1_UPDATES,
                )
                _write_json(output / f"batch1_{arm}_seed_{seed}.json", receipt)
                batch1.append(receipt)
        cfg = _engineering_config(
            arm=MAX_RESOURCE_ARM,
            seed=e8.SEEDS[0],
            output_root=output / "unused",
            device=device,
        )
        physical = multistep_acceptance(
            cfg,
            batch_size=LIFECYCLE_TRAIN_WINDOWS,
            updates=PHYSICAL_BATCH_UPDATES,
        )
        _write_json(output / "max_resource_batch128.json", physical)
        lifecycle = run_synthetic_lifecycle(output=output, device=device)
        _write_json(output / "native_lifecycle.json", lifecycle)
        _write_json(
            output / "gpu_acceptance.json",
            {
                "protocol": e8.PROTOCOL,
                "passed": True,
                "engineering_identity": identity,
                "batch1": batch1,
                "max_resource_batch128": physical,
                "native_lifecycle": lifecycle,
            },
        )
    return output


def benchmark_worker(*, arm: str, mode: str, device: str, destination: Path) -> Path:
    if arm not in ARMS or mode not in {"eval", "train"}:
        raise ValueError("E8 benchmark arm/mode 非法")
    identity = engineering_identity()
    identity_hash = str(identity["identity_sha256"])
    destination = destination.resolve()
    parent = destination.parent
    benchmark_root = (e8.SOURCE_ROOT / e8.OUTPUT_ROOT / "benchmark").resolve()
    if destination.exists() or not parent.is_relative_to(benchmark_root):
        raise ValueError("E8 benchmark worker 输出 identity 非法或已存在")
    context = json.loads((parent / "lifecycle_started.json").read_text(encoding="utf-8"))
    if (
        context.get("phase") != "benchmark"
        or context.get("engineering_identity_sha256") != identity_hash
        or (parent / "lifecycle_completed.json").exists()
        or (parent / "lifecycle_failed.json").exists()
    ):
        raise RuntimeError("E8 benchmark parent lifecycle 非 active identity")
    environment = runtime_preflight(device, identity)
    cfg = _engineering_config(
        arm=arm,
        seed=e8.SEEDS[0],
        output_root=parent / "unused",
        device=device,
    )
    set_seed(e8.SEEDS[0])
    torch.cuda.empty_cache()
    model = build_e8_film_decoder_redesign_model(cfg).to(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, _partition = build_crd_optimizer(model, cfg)
    batch_size = 1 if mode == "eval" else 128
    batch = synthetic_batch(batch_size, e8.SEEDS[0] + 4400, device=device)
    times: list[float] = []
    for index in range(BENCHMARK_WARMUP + BENCHMARK_REPEATS):
        if index == BENCHMARK_WARMUP:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if mode == "eval":
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch["x"], tf=batch["tf"])
        else:
            model.train()
            result = _native_update(model, batch, loss_fn, optimizer, cfg, index)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        finite_tree(result, label="benchmark_result")
        if index >= BENCHMARK_WARMUP:
            times.append(elapsed)
    if len(times) != BENCHMARK_REPEATS or not np.isfinite(times).all() or min(times) <= 0:
        raise RuntimeError("E8 benchmark timing 非法")
    median = float(np.median(times))
    _write_json(
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
            "engineering_identity_sha256": identity_hash,
            **_memory_receipt(device),
        },
    )
    return destination


def run_benchmark(device: str = "cuda:0") -> Path:
    identity = engineering_identity()
    identity_hash = str(identity["identity_sha256"])
    parent = e8.SOURCE_ROOT / e8.OUTPUT_ROOT / "benchmark"
    with exclusive_attempt(parent, phase="benchmark", identity_hash=identity_hash) as output:
        _write_json(output / "environment.json", runtime_preflight(device, identity))
        _write_json(
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
                    str(e8.ROOT / SCRIPT_PATH),
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
                        cwd=e8.ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                measurements.append(json.loads(target.read_text(encoding="utf-8")))
        _write_json(
            output / "benchmark.json",
            {
                "protocol": e8.PROTOCOL,
                "engineering_identity": identity,
                "scope": (
                    "synthetic cached-input model; eval batch1 and native train batch128; "
                    "fresh subprocess per arm/mode"
                ),
                "measurements": measurements,
            },
        )
    return output
