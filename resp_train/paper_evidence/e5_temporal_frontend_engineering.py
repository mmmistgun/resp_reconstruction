"""E5 用户执行的 synthetic GPU 验收和独立进程效率测量。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from resp_train.crd.model import build_crd_model
from resp_train.crd.training import build_crd_optimizer, train_crd_one_epoch
from resp_train.losses.task import RespirationTaskLoss
from resp_train.paper_evidence import e5_temporal_frontend as e5
from resp_train.paper_evidence.e5_temporal_frontend_model import (
    ARM,
    FRONTEND_CONTRACT,
    E5TemporalFrontend,
    build_e5_model,
)
from resp_train.utils.run import set_seed


def synthetic_batch(batch_size: int, seed: int, device: str = "cpu") -> dict:
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
        "tf": {
            "w": torch.rand(batch_size, 97, 360, generator=generator).to(device)
        },
    }


def update(model, batch, loss_fn, optimizer, cfg, index: int) -> dict:
    summary, completed = train_crd_one_epoch(
        model,
        [batch],
        loss_fn,
        optimizer,
        device=str(cfg.training.device),
        accumulation_steps=1,
        update_index=index,
        total_updates=e5.EPOCHS * e5.UPDATES_PER_EPOCH,
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
        raise RuntimeError("E5 synthetic update/loss 合同错误")
    return summary


def check_initial_structure(reference, candidate, batch, device: str) -> dict:
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    shared_reference = {
        key: value
        for key, value in reference_state.items()
        if not key.startswith("base.frontend.")
    }
    shared_candidate = {
        key: value
        for key, value in candidate_state.items()
        if not key.startswith("base.frontend.")
    }
    if shared_reference.keys() != shared_candidate.keys() or not all(
        torch.equal(value, shared_candidate[key])
        for key, value in shared_reference.items()
    ):
        raise RuntimeError("E5 公共模块初始 state 与 W0 不一致")
    reference.to(device).eval()
    candidate.to(device).eval()
    frontend = candidate.base.frontend
    if not isinstance(frontend, E5TemporalFrontend):
        raise RuntimeError("E5 frontend 类型错误")
    captured: dict[str, torch.Tensor] = {}
    hook = frontend.local_embedding.register_forward_hook(
        lambda _module, _args, value: captured.update(embedded=value.detach().clone())
    )
    try:
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            candidate_frontend = frontend(batch["x"])
            reference_output = reference(batch["x"], tf=batch["tf"])
            candidate_output = candidate(batch["x"], tf=batch["tf"])
    finally:
        hook.remove()
    e5.finite_tree(reference_output)
    e5.finite_tree(candidate_output)
    if not torch.equal(candidate_frontend, captured["embedded"]):
        raise RuntimeError("E5 zero-init residual 未严格退化为局部 embedding")
    if reference_output.keys() != candidate_output.keys():
        raise RuntimeError("E5 waveform 输出键漂移")
    return {
        "shared_state_equal": True,
        "shared_state_tensor_count": len(shared_reference),
        "zero_residual_exact_identity": True,
        "frontend_output_shape": list(candidate_frontend.shape),
        "reference_output_finite": True,
        "candidate_output_finite": True,
    }


def multistep_acceptance(cfg, batch_size: int, *, initial_check: bool) -> dict:
    seed = int(cfg.training.seed)
    device = str(cfg.training.device)
    set_seed(seed)
    model = build_e5_model(cfg)
    batch = synthetic_batch(batch_size, seed + 2300, device)
    initial = None
    if initial_check:
        reference_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        reference_cfg.protocol.name = "crd-tf-v1-research-informed-20260812"
        reference_cfg.protocol.execution_gate = "p4_formal"
        del reference_cfg.model.e5_temporal_frontend
        initial = check_initial_structure(
            build_crd_model(reference_cfg), model, batch, device
        )
    model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, partition = build_crd_optimizer(model, cfg)
    expected_new = {
        "base.frontend.local_embedding.weight",
        "base.frontend.local_residual.weight",
    }
    if not expected_new.issubset(partition.decay_names):
        raise RuntimeError("E5 前端参数未进入原生 decay 分组")
    summaries: list[dict] = []
    gradients: list[dict[str, float]] = []
    for index in range(3):
        summary = update(model, batch, loss_fn, optimizer, cfg, index)
        e5.finite_tree(summary)
        e5.finite_tree(model.state_dict())
        e5.finite_tree(optimizer.state_dict())
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"E5 synthetic 缺少梯度通路: {name}")
            e5.finite_tree(parameter.grad)
        frontend = model.base.frontend
        gradients.append(
            {
                "local_embedding": float(frontend.local_embedding.weight.grad.float().norm()),
                "local_residual": float(frontend.local_residual.weight.grad.float().norm()),
            }
        )
        summaries.append(summary)
    frontend = model.base.frontend
    if (
        not all(item["local_embedding"] > 0 for item in gradients)
        or not all(item["local_residual"] > 0 for item in gradients)
        or not bool(frontend.local_residual.weight.detach().ne(0).any())
    ):
        raise RuntimeError("E5 三步后前端梯度或参数尚未打开")
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    if reserved / total > 0.8:
        raise RuntimeError(f"E5 acceptance reserved fraction={reserved / total:.6f} 超过 0.8")
    return {
        "seed": seed,
        "batch_size": batch_size,
        "updates": 3,
        "initial_structure": initial,
        "frontend_gradient_norms": gradients,
        "frontend_parameters_updated": True,
        "summaries": summaries,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "device_total_bytes": total,
        "peak_reserved_fraction": reserved / total,
    }


def run_gpu_acceptance(device: str = "cuda:0") -> Path:
    lock, lock_hash = e5.load_lock()
    parent = e5.ROOT / e5.OUTPUT / "gpu_acceptance"
    with e5.phase_guard(parent, lock_hash, completed_phase="gpu_acceptance"), e5.attempt(
        parent, lock_hash, "gpu_acceptance"
    ) as output:
        e5.write_json(output / "environment.json", e5.runtime_preflight(device))
        e5.write_json(
            output / "access_receipt.json",
            {
                "waveforms": "synthetic",
                "checkpoint": "fresh initialization",
                "real_data_read": False,
                "research_test_read": False,
            },
        )
        receipts = []
        for seed in e5.SEEDS:
            baseline = OmegaConf.create(lock["baselines"][str(seed)])
            cfg = e5.derived_config(
                baseline, output_root=output / "synthetic", device=device
            )
            receipt = multistep_acceptance(cfg, 1, initial_check=True)
            receipts.append(receipt)
            e5.write_json(output / f"seed_{seed}_batch1.json", receipt)
        baseline = OmegaConf.create(lock["baselines"][str(e5.SEEDS[0])])
        cfg = e5.derived_config(
            baseline, output_root=output / "synthetic", device=device
        )
        physical = multistep_acceptance(cfg, 128, initial_check=False)
        e5.write_json(output / "batch128.json", physical)
        e5.write_json(output / "parameter_compute_report.json", e5.parameter_compute_report())
        e5.write_json(
            output / "gpu_acceptance.json",
            {
                "protocol": e5.PROTOCOL,
                "passed": True,
                "frontend_contract": FRONTEND_CONTRACT,
                "seeds": list(e5.SEEDS),
                "physical_batch": 128,
                "batch1_receipts": receipts,
                "batch128_receipt": physical,
            },
        )
    return output


def benchmark_worker(
    *, arm: str, mode: str, device: str, destination: Path
) -> Path:
    lock, lock_hash = e5.load_lock()
    parent = destination.resolve().parent
    if destination.exists():
        raise FileExistsError(f"E5 benchmark measurement 已存在: {destination}")
    if not parent.is_relative_to((e5.ROOT / e5.OUTPUT / "benchmark").resolve()):
        raise ValueError("E5 benchmark worker 输出须位于 benchmark attempt")
    context = json.loads((parent / "lifecycle_started.json").read_text())
    if context["phase"] != "benchmark" or context["implementation_lock_sha256"] != lock_hash:
        raise ValueError("E5 benchmark worker 父 identity 不一致")
    runtime = e5.runtime_preflight(device)
    baseline = OmegaConf.create(lock["baselines"][str(e5.SEEDS[0])])
    cfg = e5.derived_config(baseline, output_root=parent / "synthetic", device=device)
    set_seed(e5.SEEDS[0])
    if arm not in ("W0_FULL", ARM) or mode not in ("eval", "train"):
        raise ValueError("E5 benchmark arm/mode 非法")
    model = (
        build_crd_model(baseline) if arm == "W0_FULL" else build_e5_model(cfg)
    ).to(device)
    batch = synthetic_batch(1 if mode == "eval" else 128, 20260923, device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, _ = build_crd_optimizer(model, cfg)
    times: list[float] = []
    for index in range(25):
        if index == 5:
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
        if index >= 5:
            times.append(elapsed)
        e5.finite_tree(result)
    if len(times) != 20 or not np.isfinite(times).all() or min(times) <= 0:
        raise RuntimeError("E5 benchmark 计时无效")
    e5.write_json(
        destination,
        {
            "arm": arm,
            "mode": mode,
            "environment": runtime,
            "warmup": 5,
            "repeats": 20,
            "seconds": times,
            "median_seconds": float(np.median(times)),
            "iqr_seconds": float(np.quantile(times, 0.75) - np.quantile(times, 0.25)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "implementation_lock_sha256": lock_hash,
        },
    )
    return destination


def run_benchmark(device: str = "cuda:0") -> Path:
    _lock, lock_hash = e5.load_lock()
    parent = e5.ROOT / e5.OUTPUT / "benchmark"
    with e5.phase_guard(parent, lock_hash, completed_phase="benchmark"), e5.attempt(
        parent, lock_hash, "benchmark"
    ) as output:
        e5.write_json(output / "environment.json", e5.runtime_preflight(device))
        e5.write_json(
            output / "access_receipt.json",
            {"waveforms": "synthetic", "real_data_read": False, "research_test_read": False},
        )
        measurements = []
        for group in range(3):
            for mode in ("eval", "train"):
                arms = ("W0_FULL", ARM) if group % 2 == 0 else (ARM, "W0_FULL")
                for arm in arms:
                    target = output / f"group{group}_{mode}_{arm}.json"
                    command = [
                        sys.executable,
                        str(e5.ROOT / e5.SCRIPT_PATH),
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
                            cwd=e5.ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    measurements.append(
                        {"group": group, **json.loads(target.read_text())}
                    )
        e5.write_json(
            output / "benchmark.json",
            {
                "protocol": e5.PROTOCOL,
                "measurements": measurements,
                "scope": "synthetic cached-input model only; eval batch1 / native training batch128; separate process per measurement",
            },
        )
        e5.write_json(output / "parameter_compute_report.json", e5.parameter_compute_report())
    return output
