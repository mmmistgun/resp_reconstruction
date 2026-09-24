"""E7 P3 synthetic GPU acceptance 与独立进程 benchmark。"""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.model import build_crd_model
from resp_train.crd.tf_v1_model import TF_BRANCH_CHECKPOINT_BATCH_CHUNK
from resp_train.crd.training import build_crd_optimizer, train_crd_one_epoch
from resp_train.losses.task import RespirationTaskLoss
from resp_train.paper_evidence import e7_scale_encoding_aggregation as e7
from resp_train.paper_evidence.e1_scale_topology_runtime import environment
from resp_train.paper_evidence.e7_scale_encoding_aggregation_model import ARMS, build_e7_model
from resp_train.utils.run import set_seed


P3_PROTOCOL = "e7-scale-encoding-aggregation-p3-engineering-v1-20260924"
P1_LOCK_SHA256 = "299448a12c4006f2f918f56645000838f83f044fbbf31d7cc281faa6540b11e1"
P3_LOCK_PATH = Path(
    "docs/experiments/e7_scale_encoding_aggregation_p3_engineering_lock_20260924.json"
)
P3_PROTOCOL_PATH = Path(
    "docs/experiments/e7_scale_encoding_aggregation_p3_engineering_protocol_20260924.md"
)
ENGINEERING_PATH = Path(
    "resp_train/paper_evidence/e7_scale_encoding_aggregation_engineering.py"
)
P3_SCRIPT_PATH = Path("scripts/run_e7_scale_encoding_aggregation_p3.py")
P3_TEST_PATH = Path("tests/test_e7_scale_encoding_aggregation_engineering.py")
ACCEPTANCE_MIN_UPDATES = 5
ACCEPTANCE_MAX_UPDATES = 20
MEMORY_LIMIT_FRACTION = 0.8
BENCHMARK_WARMUP = 5
BENCHMARK_REPEATS = 20
BENCHMARK_GROUPS = 3


def p3_contract() -> dict[str, Any]:
    return {
        "p3_protocol": P3_PROTOCOL,
        "p1_implementation_lock_sha256": P1_LOCK_SHA256,
        "arms": list(ARMS),
        "seeds": list(e7.SEEDS),
        "acceptance": {
            "batch1_cells": len(ARMS) * len(e7.SEEDS),
            "batch128_cells": len(ARMS),
            "minimum_updates": ACCEPTANCE_MIN_UPDATES,
            "maximum_updates": ACCEPTANCE_MAX_UPDATES,
            "physical_batch": 128,
            "amp_dtype": "bfloat16",
            "branch_checkpoint_batch_chunk": TF_BRANCH_CHECKPOINT_BATCH_CHUNK,
            "memory_limit_fraction": MEMORY_LIMIT_FRACTION,
        },
        "benchmark": {
            "arms": list(ARMS),
            "modes": ["eval", "train"],
            "eval_batch": 1,
            "train_batch": 128,
            "warmup": BENCHMARK_WARMUP,
            "repeats": BENCHMARK_REPEATS,
            "groups": BENCHMARK_GROUPS,
            "processes": len(ARMS) * 2 * BENCHMARK_GROUPS,
        },
    }


def p3_critical_paths() -> tuple[Path, ...]:
    paths = (ENGINEERING_PATH, P3_SCRIPT_PATH, P3_TEST_PATH, P3_PROTOCOL_PATH)
    missing = [str(path) for path in paths if not (e7.ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"E7 P3 缺少关键文件: {missing}")
    return paths


def prepare_p3_lock(root: Path = e7.ROOT) -> Path:
    destination = root / P3_LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"E7 P3 engineering lock 已存在: {destination}")
    state = e7.git_state(root)
    if state.get("status_porcelain"):
        raise RuntimeError("E7 P3 engineering lock 要求干净工作树")
    _p1_lock, p1_digest = e7.load_implementation_lock(root)
    if p1_digest != P1_LOCK_SHA256:
        raise ValueError("E7 P3 引用的 P1 implementation lock 漂移")
    lock = {
        "schema_version": 1,
        "protocol": P3_PROTOCOL,
        "status": "engineering_implemented_not_run",
        "contract": p3_contract(),
        "p1_implementation_lock": {
            "path": str(e7.LOCK_PATH),
            **e7.identity(root / e7.LOCK_PATH),
        },
        "code_files": {
            str(path): e7.identity(root / path)
            for path in p3_critical_paths()
        },
        "preparation_git": state,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    e7.write_json(destination, lock)
    return destination


def load_p3_lock(root: Path = e7.ROOT) -> tuple[dict[str, Any], str]:
    path = root / P3_LOCK_PATH
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != P3_PROTOCOL
        or lock.get("status") != "engineering_implemented_not_run"
        or lock.get("contract") != p3_contract()
        or lock.get("p1_implementation_lock", {}).get("path") != str(e7.LOCK_PATH)
    ):
        raise ValueError("E7 P3 engineering lock 合同漂移")
    e7.verify(root / e7.LOCK_PATH, lock["p1_implementation_lock"])
    _p1_lock, p1_digest = e7.load_implementation_lock(root)
    if p1_digest != P1_LOCK_SHA256:
        raise ValueError("E7 P3 的 P1 implementation lock identity 漂移")
    for relative, expected in lock["code_files"].items():
        e7.verify(root / relative, expected)
    return lock, e7.sha256_file(path)


@contextmanager
def phase_guard(parent: Path, lock_hash: str, phase: str) -> Iterator[None]:
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / f".execution_{lock_hash}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("E7 P3 相同身份正在运行") from exc
        try:
            for receipt in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((receipt.parent / "manifest.json").read_text())
                if manifest.get("p3_engineering_lock_sha256") == lock_hash:
                    verify_attempt(receipt.parent, phase=phase, lock_hash=lock_hash)
                    raise FileExistsError(f"E7 P3 相同身份阶段已完成: {receipt.parent}")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def attempt(parent: Path, lock_hash: str, phase: str) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    path.mkdir(exist_ok=False)
    context = {
        "protocol": P3_PROTOCOL,
        "phase": phase,
        "p3_engineering_lock_sha256": lock_hash,
        "p1_implementation_lock_sha256": P1_LOCK_SHA256,
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    e7.write_json(path / "lifecycle_started.json", {**context, "status": "running"})
    try:
        yield path
        e7.write_json(
            path / "lifecycle_completed.json",
            {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
        )
        manifest = {
            **context,
            "status": "completed",
            "files": {
                str(file.relative_to(path)): e7.identity(file)
                for file in sorted(path.rglob("*"))
                if file.is_file()
            },
        }
        e7.write_json(path / "manifest.json", manifest)
        e7.write_json(
            path / "freeze_receipt.json",
            {"protocol": P3_PROTOCOL, "manifest": e7.identity(path / "manifest.json")},
        )
    except BaseException as exc:
        e7.write_json(
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


def verify_attempt(path: Path, *, phase: str, lock_hash: str) -> dict[str, Any]:
    path = path.resolve()
    if (path / "lifecycle_failed.json").exists():
        raise ValueError(f"E7 P3 attempt 失败: {path}")
    freeze = json.loads((path / "freeze_receipt.json").read_text())
    e7.verify(path / "manifest.json", freeze["manifest"])
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("protocol") != P3_PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("p3_engineering_lock_sha256") != lock_hash
        or manifest.get("p1_implementation_lock_sha256") != P1_LOCK_SHA256
    ):
        raise ValueError("E7 P3 attempt identity 漂移")
    required = {
        "gpu_acceptance": {
            "lifecycle_completed.json",
            "environment.json",
            "access_receipt.json",
            "gpu_acceptance.json",
            "parameter_compute_report.json",
        },
        "benchmark": {
            "lifecycle_completed.json",
            "environment.json",
            "access_receipt.json",
            "benchmark.json",
            "parameter_compute_report.json",
        },
    }[phase]
    if not required.issubset(manifest["files"]):
        raise ValueError("E7 P3 attempt 缺少必需产物")
    for relative, expected in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path):
            raise ValueError("E7 P3 manifest 路径越界")
        e7.verify(file, expected)
    return manifest


def runtime_preflight(device: str) -> dict[str, Any]:
    state = e7.git_state(e7.ROOT)
    if state.get("status_porcelain"):
        raise RuntimeError("E7 P3 GPU 执行要求干净提交")
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("E7 P3 GPU 阶段需要 CUDA")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("E7 P3 原生依赖检查失败: " + "; ".join(problems))
    return {**environment(device), "git": state}


def finite_tree(value: Any) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("E7 P3 tensor 非有限")
    elif isinstance(value, Mapping):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError("E7 P3 scalar 非有限")


def synthetic_batch(batch_size: int, seed: int, device: str = "cpu") -> dict[str, Any]:
    generator = torch.Generator().manual_seed(int(seed))
    time_grid = torch.arange(18000, dtype=torch.float32)[None, None] / 100.0
    phase = torch.rand(batch_size, 1, 1, generator=generator) * (2 * torch.pi)
    frequency = 0.12 + 0.45 * torch.rand(batch_size, 1, 1, generator=generator)
    effort = 1.0 + 0.35 * torch.sin(2 * torch.pi * 0.02 * time_grid + phase)
    target = effort * torch.sin(2 * torch.pi * frequency * time_grid + phase)
    carrier = 0.12 * effort * torch.sin(2 * torch.pi * 6.0 * time_grid + 0.5 * phase)
    sensor = target + carrier + 0.05 * torch.randn(target.shape, generator=generator)
    return {
        "x": sensor.to(device),
        "target": target.to(device),
        "tf": {"w": torch.rand(batch_size, 97, 360, generator=generator).to(device)},
    }


def update(model, batch, loss_fn, optimizer, cfg, index: int) -> dict[str, float]:
    summary, completed = train_crd_one_epoch(
        model,
        [batch],
        loss_fn,
        optimizer,
        device=str(cfg.training.device),
        accumulation_steps=1,
        update_index=index,
        total_updates=e7.EPOCHS * e7.UPDATES_PER_EPOCH,
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
        rtol=0,
    ):
        raise RuntimeError("E7 P3 synthetic update/loss 合同错误")
    return summary


def tracked_parameters(model) -> dict[str, torch.nn.Parameter]:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("branches.w.scale_encoder.")
        or name.startswith("branches.w.aggregation.")
    }
    if not parameters:
        parameters = {
            "branches.w.final_projection.weight": model.branches["w"].final_projection.weight
        }
    return parameters


def initial_identity(reference, candidate, batch, device: str) -> dict[str, Any]:
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    if not all(
        name in candidate_state and torch.equal(value, candidate_state[name])
        for name, value in reference_state.items()
    ):
        raise RuntimeError("E7 P3 公共 W0 初始 state 不一致")
    reference.to(device).eval()
    candidate.to(device).eval()
    with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        features = candidate.e7_branch.forward_features(batch["tf"])
        expected = reference(batch["x"], tf=batch["tf"])
        observed = candidate(batch["x"], tf=batch["tf"])
    finite_tree(features)
    finite_tree(expected)
    finite_tree(observed)
    if expected.keys() != observed.keys():
        raise RuntimeError("E7 P3 waveform 输出字段漂移")
    errors = []
    for key in expected:
        torch.testing.assert_close(observed[key], expected[key], rtol=1e-5, atol=1e-6)
        errors.append(float((observed[key].float() - expected[key].float()).abs().max()))
    if features["x0"].shape[1:] != (96, 97, 360) or features["aggregated"].shape[1:] != (96, 360):
        raise RuntimeError("E7 P3 聚合前后 shape 漂移")
    return {
        "common_state_equal": True,
        "feature_shapes": {
            "x0": list(features["x0"].shape),
            "xe": list(features["xe"].shape),
            "aggregated": list(features["aggregated"].shape),
        },
        "waveform_max_abs_error": max(errors),
        "rtol": 1e-5,
        "atol": 1e-6,
    }


def acceptance_updates(
    model,
    batch,
    loss_fn,
    optimizer,
    cfg,
    record: dict[str, Any],
    diagnostic_dir: Path | None = None,
) -> None:
    tracked = tracked_parameters(model)
    initial = {name: parameter.detach().clone() for name, parameter in tracked.items()}
    record.update(
        tracked_parameter_count=len(tracked),
        updates=0,
        summaries=[],
        tracked_gradient_norms=[],
        tracked_max_abs_changes=[],
        tracked_parameters_updated={},
    )
    hook = model.register_forward_hook(lambda _module, _args, output: finite_tree(output))
    try:
        for index in range(ACCEPTANCE_MAX_UPDATES):
            record["stage"] = f"update_{index + 1}"
            summary = update(model, batch, loss_fn, optimizer, cfg, index)
            finite_tree(summary)
            finite_tree(model.state_dict())
            finite_tree(optimizer.state_dict())
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"E7 P3 gradient 通路缺失: {name}")
                finite_tree(parameter.grad)
            gradients = {
                name: float(parameter.grad.double().norm())
                for name, parameter in tracked.items()
            }
            changes = {
                name: float(
                    (parameter.detach().double() - initial[name].double()).abs().max()
                )
                for name, parameter in tracked.items()
            }
            changed = {name: value > 0 for name, value in changes.items()}
            record["updates"] = index + 1
            record["summaries"].append(summary)
            record["tracked_gradient_norms"].append(gradients)
            record["tracked_max_abs_changes"].append(changes)
            record["tracked_parameters_updated"] = changed
            if diagnostic_dir is not None:
                e7.write_json(
                    diagnostic_dir / f"step_{index + 1:03d}.json",
                    {
                        "arm": record["arm"],
                        "seed": record["seed"],
                        "batch_size": record["batch_size"],
                        "update": index + 1,
                        "summary": summary,
                        "gradient_norms": gradients,
                        "max_abs_changes": changes,
                        "parameters_updated": changed,
                    },
                )
            if (
                index + 1 >= ACCEPTANCE_MIN_UPDATES
                and all(changed.values())
                and all(value > 0 for value in gradients.values())
            ):
                return
        stalled = [name for name, changed_value in changed.items() if not changed_value]
        zero = [name for name, value in gradients.items() if value == 0]
        raise RuntimeError(
            f"E7 P3 {ACCEPTANCE_MAX_UPDATES} 步验收失败: 未更新参数={stalled}; 零梯度参数={zero}"
        )
    finally:
        hook.remove()


def acceptance_case(
    cfg,
    batch_size: int,
    check_initial: bool,
    *,
    diagnostic_dir: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "arm": str(cfg.model.e7_factorial.arm),
        "seed": int(cfg.training.seed),
        "batch_size": int(batch_size),
        "minimum_updates": ACCEPTANCE_MIN_UPDATES,
        "maximum_updates": ACCEPTANCE_MAX_UPDATES,
        "status": "running",
        "stage": "initialization",
    }
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(exist_ok=False)
    try:
        _acceptance_case(cfg, batch_size, check_initial, record, diagnostic_dir)
        record.update(status="passed", stage="completed")
    except BaseException as exc:
        record.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        exc.add_note(
            f"E7 P3 arm={record['arm']} seed={record['seed']} batch={batch_size} "
            f"stage={record['stage']}; diagnostics={diagnostic_dir}"
        )
        raise
    finally:
        if diagnostic_dir is not None:
            e7.write_json(diagnostic_dir / "case.json", record)
    return record


def _acceptance_case(cfg, batch_size, check_initial, record, diagnostic_dir) -> None:
    seed = int(cfg.training.seed)
    device = str(cfg.training.device)
    set_seed(seed)
    model = build_e7_model(cfg)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    batch = synthetic_batch(batch_size, seed + 2400, device)
    equality = None
    if check_initial:
        baseline = e7.baseline_config(seed)
        baseline.training.device = device
        reference = build_crd_model(baseline)
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
            cuda_rng, torch.cuda.get_rng_state(device)
        ):
            raise RuntimeError("E7 P3 与 W0 构造后的 CPU/CUDA RNG 不一致")
        equality = {
            **initial_identity(reference, model, batch, device),
            "cpu_rng_equal": True,
            "cuda_rng_equal": True,
        }
        del reference
    model.to(device)
    record["initial_identity"] = equality
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, partition = build_crd_optimizer(model, cfg)
    tracked_names = set(tracked_parameters(model))
    optimizer_names = set(partition.decay_names) | set(partition.no_decay_names)
    if not tracked_names.issubset(optimizer_names):
        raise RuntimeError("E7 P3 tracked 参数未全部进入 optimizer")
    acceptance_updates(model, batch, loss_fn, optimizer, cfg, record, diagnostic_dir)
    record["stage"] = "memory"
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    record.update(
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
        device_total_bytes=total,
        peak_reserved_fraction=reserved / total,
    )
    if reserved / total > MEMORY_LIMIT_FRACTION:
        raise RuntimeError(
            f"E7 P3 batch={batch_size} reserved fraction={reserved / total:.6f} "
            f"超过 {MEMORY_LIMIT_FRACTION}"
        )


def run_gpu_acceptance(device: str = "cuda:0") -> Path:
    p3_lock, p3_digest = load_p3_lock()
    p1_lock, _p1_digest = e7.load_implementation_lock()
    parent = e7.ROOT / e7.OUTPUT / "gpu_acceptance"
    with phase_guard(parent, p3_digest, "gpu_acceptance"):
        with attempt(parent, p3_digest, "gpu_acceptance") as output:
            e7.write_json(output / "environment.json", runtime_preflight(device))
            e7.write_json(
                output / "access_receipt.json",
                {
                    "waveforms": "synthetic",
                    "checkpoint": "fresh_initialization",
                    "real_data_read": False,
                    "research_test_read": False,
                },
            )
            e7.write_json(output / "p3_engineering_lock.json", p3_lock)
            records = []
            for arm in ARMS:
                for seed in e7.SEEDS:
                    cfg = OmegaConf.create(p1_lock["resolved_templates"][arm][str(seed)])
                    cfg.training.device = device
                    cfg.outputs.run_root = str(output / "synthetic")
                    record = acceptance_case(
                        cfg,
                        1,
                        True,
                        diagnostic_dir=output / f"{arm}_{seed}_batch1_diagnostics",
                    )
                    e7.write_json(output / f"{arm}_{seed}_batch1.json", record)
                    records.append(record)
                cfg = OmegaConf.create(
                    p1_lock["resolved_templates"][arm][str(e7.SEEDS[0])]
                )
                cfg.training.device = device
                cfg.outputs.run_root = str(output / "synthetic")
                record = acceptance_case(
                    cfg,
                    128,
                    False,
                    diagnostic_dir=output / f"{arm}_batch128_diagnostics",
                )
                e7.write_json(output / f"{arm}_batch128.json", record)
                records.append(record)
            e7.write_json(
                output / "gpu_acceptance.json",
                {
                    "protocol": P3_PROTOCOL,
                    "passed": True,
                    "arms": list(ARMS),
                    "seeds": list(e7.SEEDS),
                    "physical_batch": 128,
                    "p3_engineering_lock_sha256": p3_digest,
                    "p1_implementation_lock_sha256": P1_LOCK_SHA256,
                    "records": records,
                },
            )
            e7.write_json(
                output / "parameter_compute_report.json", e7.parameter_compute_report()
            )
    return output


def benchmark_worker(*, arm: str, mode: str, device: str, destination: Path) -> Path:
    _p3_lock, p3_digest = load_p3_lock()
    p1_lock, _p1_digest = e7.load_implementation_lock()
    parent = destination.resolve().parent
    benchmark_root = (e7.ROOT / e7.OUTPUT / "benchmark").resolve()
    if destination.exists():
        raise FileExistsError(f"E7 P3 benchmark measurement 已存在: {destination}")
    if not parent.is_relative_to(benchmark_root):
        raise ValueError("E7 P3 benchmark worker 输出路径越界")
    context = json.loads((parent / "lifecycle_started.json").read_text())
    if (
        context.get("phase") != "benchmark"
        or context.get("p3_engineering_lock_sha256") != p3_digest
    ):
        raise ValueError("E7 P3 benchmark worker 父 identity 漂移")
    if (parent / "lifecycle_completed.json").exists() or (parent / "lifecycle_failed.json").exists():
        raise ValueError("E7 P3 benchmark 父 attempt 已结束")
    if arm not in ARMS or mode not in {"eval", "train"}:
        raise ValueError("E7 P3 benchmark arm/mode 非法")
    runtime = runtime_preflight(device)
    cfg = OmegaConf.create(p1_lock["resolved_templates"][arm][str(e7.SEEDS[0])])
    cfg.training.device = device
    cfg.outputs.run_root = str(parent / "synthetic")
    set_seed(e7.SEEDS[0])
    model = build_e7_model(cfg).to(device)
    batch_size = 1 if mode == "eval" else 128
    batch = synthetic_batch(batch_size, 20260924, device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, _partition = build_crd_optimizer(model, cfg)
    times: list[float] = []
    for index in range(BENCHMARK_WARMUP + BENCHMARK_REPEATS):
        if index == BENCHMARK_WARMUP:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if mode == "eval":
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch["x"], tf=batch["tf"])
        else:
            result = update(model, batch, loss_fn, optimizer, cfg, index)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= BENCHMARK_WARMUP:
            times.append(elapsed)
        finite_tree(result)
    finite_tree(model.state_dict())
    finite_tree(optimizer.state_dict())
    if len(times) != BENCHMARK_REPEATS or not np.isfinite(times).all() or min(times) <= 0:
        raise RuntimeError("E7 P3 benchmark 计时无效")
    median_seconds = float(np.median(times))
    e7.write_json(
        destination,
        {
            "arm": arm,
            "mode": mode,
            "batch_size": batch_size,
            "environment": runtime,
            "warmup": BENCHMARK_WARMUP,
            "repeats": BENCHMARK_REPEATS,
            "seconds": times,
            "median_seconds": median_seconds,
            "iqr_seconds": float(np.quantile(times, 0.75) - np.quantile(times, 0.25)),
            "throughput_samples_per_second": batch_size / median_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "p3_engineering_lock_sha256": p3_digest,
            "p1_implementation_lock_sha256": P1_LOCK_SHA256,
        },
    )
    return destination


def benchmark_order(group: int) -> tuple[str, ...]:
    if int(group) not in range(BENCHMARK_GROUPS):
        raise ValueError("E7 P3 benchmark group 越界")
    if group == 0:
        return ARMS
    if group == 1:
        return tuple(reversed(ARMS))
    return ARMS[2:] + ARMS[:2]


def run_benchmark(device: str = "cuda:0") -> Path:
    _p3_lock, p3_digest = load_p3_lock()
    parent = e7.ROOT / e7.OUTPUT / "benchmark"
    with phase_guard(parent, p3_digest, "benchmark"):
        with attempt(parent, p3_digest, "benchmark") as output:
            e7.write_json(output / "environment.json", runtime_preflight(device))
            e7.write_json(
                output / "access_receipt.json",
                {
                    "waveforms": "synthetic",
                    "real_data_read": False,
                    "research_test_read": False,
                },
            )
            measurements = []
            for group in range(BENCHMARK_GROUPS):
                for mode in ("eval", "train"):
                    for arm in benchmark_order(group):
                        target = output / f"group{group}_{mode}_{arm}.json"
                        command = [
                            sys.executable,
                            str(e7.ROOT / P3_SCRIPT_PATH),
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
                        with target.with_suffix(".log").open("x") as log:
                            subprocess.run(
                                command,
                                cwd=e7.ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                        measurements.append(
                            {"group": group, **json.loads(target.read_text())}
                        )
            if len(measurements) != len(ARMS) * 2 * BENCHMARK_GROUPS:
                raise RuntimeError("E7 P3 benchmark measurement 矩阵不完整")
            e7.write_json(
                output / "benchmark.json",
                {
                    "protocol": P3_PROTOCOL,
                    "measurements": measurements,
                    "scope": "synthetic cached-input model-only; eval batch1 and native train batch128; independent process per measurement",
                    "p3_engineering_lock_sha256": p3_digest,
                    "p1_implementation_lock_sha256": P1_LOCK_SHA256,
                },
            )
            e7.write_json(
                output / "parameter_compute_report.json", e7.parameter_compute_report()
            )
    return output
