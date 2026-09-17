"""E4 用户执行的 synthetic GPU 验收及独立进程效率测量。"""

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
from resp_train.paper_evidence import e4_scale_aggregation as e4
from resp_train.paper_evidence.e4_scale_aggregation_model import AGGREGATION_CONTRACT, build_e4_model
from resp_train.utils.run import set_seed

GPU_RTOL = 1e-5
GPU_ATOL = 1e-6


def synthetic_batch(batch_size: int, seed: int, device: str = "cpu") -> dict:
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(18000, dtype=torch.float32)[None, None] / 100
    phase = torch.rand(batch_size, 1, 1, generator=generator) * (2 * torch.pi)
    frequency = 0.18 + 0.12 * torch.rand(batch_size, 1, 1, generator=generator)
    target = (1 + 0.3 * torch.sin(2 * torch.pi * 0.02 * t + phase)) * torch.sin(2 * torch.pi * frequency * t + phase)
    sensor = target + 0.05 * torch.randn(target.shape, generator=generator)
    return {"x": sensor.to(device), "target": target.to(device),
            "tf": {"w": torch.rand(batch_size, 97, 360, generator=generator).to(device)}}


def update(model, batch, loss_fn, optimizer, cfg, index: int) -> dict:
    summary, completed = train_crd_one_epoch(
        model, [batch], loss_fn, optimizer, device=str(cfg.training.device), accumulation_steps=1,
        update_index=index, total_updates=e4.EPOCHS * e4.UPDATES_PER_EPOCH,
        max_learning_rate=float(cfg.training.max_learning_rate), min_learning_rate=float(cfg.training.min_learning_rate),
        warmup_fraction=float(cfg.training.warmup_fraction), grad_clip_norm=float(cfg.training.grad_clip_norm),
        use_amp=True, show_progress=False,
    )
    if completed != index + 1 or not np.isclose(summary["loss"], summary["loss_sync"] + 0.25 * summary["loss_effort"], atol=1e-12, rtol=0):
        raise RuntimeError("E4 synthetic update/loss 合同错误")
    return summary


def check_initial_identity(reference, candidate, batch, device: str) -> dict:
    ref = reference.state_dict()
    cur = candidate.state_dict()
    if set(cur) - set(ref) != {"branches.w.aggregation.projection.weight"} or not all(torch.equal(v, cur[k]) for k, v in ref.items()):
        raise RuntimeError("E4 公共模块初始 state 与 W0 不一致")
    reference.to(device).eval()
    candidate.to(device).eval()
    contexts = {}
    hooks = [reference.branches["w"].temporal.register_forward_pre_hook(lambda _, args: contexts.update(reference=args[0].detach().clone())),
             candidate.branches["w"].temporal.register_forward_pre_hook(lambda _, args: contexts.update(candidate=args[0].detach().clone()))]
    errors = []
    def check_aggregation(_, args, output):
        expected = args[0].mean(dim=2)
        if not torch.equal(output, expected):
            raise RuntimeError("E4 零残差聚合未严格还原全尺度均值")
    hooks.append(candidate.branches["w"].aggregation.register_forward_hook(check_aggregation))
    try:
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            full = reference(batch["x"], tf=batch["tf"])
            changed = candidate(batch["x"], tf=batch["tf"])
        e4.finite_tree(full)
        e4.finite_tree(changed)
        if not torch.equal(contexts["reference"], contexts["candidate"]):
            raise RuntimeError("E4 temporal 输入与 W0 不一致")
        if full.keys() != changed.keys():
            raise RuntimeError("E4 waveform 输出键漂移")
        for name in full:
            torch.testing.assert_close(full[name], changed[name], rtol=GPU_RTOL, atol=GPU_ATOL)
            errors.append(float((full[name].float() - changed[name].float()).abs().max()))
    finally:
        for hook in hooks:
            hook.remove()
    return {"common_state_equal": True, "aggregation_equal": True, "temporal_input_equal": True,
            "waveform_max_abs_error": max(errors), "rtol": GPU_RTOL, "atol": GPU_ATOL}


def multistep_acceptance(cfg, batch_size: int, *, initial_identity: bool) -> dict:
    seed, device = int(cfg.training.seed), str(cfg.training.device)
    set_seed(seed)
    model = build_e4_model(cfg)
    batch = synthetic_batch(batch_size, seed + 1700, device)
    initial = None
    if initial_identity:
        reference_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        initial = check_initial_identity(build_crd_model(reference_cfg), model, batch, device)
    model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, partition = build_crd_optimizer(model, cfg)
    if "branches.w.aggregation.projection.weight" not in partition.decay_names:
        raise RuntimeError("E4 新投影未进入原生 decay 分组")
    summaries, norms = [], []
    # 原生 FiLM 从零打开，明确记录每一步梯度；第 1 步上游零梯度是预期现象。
    hook = model.register_forward_hook(lambda _, args, value: e4.finite_tree(value))
    try:
        for index in range(3):
            summary = update(model, batch, loss_fn, optimizer, cfg, index)
            e4.finite_tree(summary)
            e4.finite_tree(model.state_dict())
            e4.finite_tree(optimizer.state_dict())
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"E4 synthetic 缺少梯度通路: {name}")
                e4.finite_tree(parameter.grad)
            grad = model.branches["w"].aggregation.projection.weight.grad
            norms.append([float(part.float().norm()) for part in grad.split(96, dim=1)])
            summaries.append(summary)
    finally:
        hook.remove()
    weight = model.branches["w"].aggregation.projection.weight.detach()
    if not all(value > 0 for value in norms[-1]) or not all(bool(part.ne(0).any()) for part in weight.split(96, dim=1)):
        raise RuntimeError("E4 三步后聚合四区梯度/参数尚未打开")
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    if reserved / total > 0.8:
        raise RuntimeError(f"E4 acceptance reserved fraction={reserved / total:.6f} 超过 0.8")
    return {"seed": seed, "batch_size": batch_size, "updates": 3, "initial_identity": initial,
            "projection_gradient_norms": norms, "projection_updated_all_regions": True,
            "summaries": summaries, "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
            "device_total_bytes": total, "peak_reserved_fraction": reserved / total}


def run_gpu_acceptance(device: str = "cuda:0") -> Path:
    lock, lock_hash = e4.load_lock()
    parent = e4.ROOT / e4.OUTPUT / "gpu_acceptance"
    with e4.phase_guard(parent, lock_hash, completed_phase="gpu_acceptance"), e4.attempt(parent, lock_hash, "gpu_acceptance") as output:
        e4.write_json(output / "environment.json", e4.runtime_preflight(device))
        e4.write_json(output / "access_receipt.json", {"waveforms": "synthetic", "checkpoint": "fresh initialization",
                      "real_data_read": False, "test_read": False})
        receipts = []
        for seed in e4.SEEDS:
            baseline = OmegaConf.create(lock["baselines"][str(seed)])
            cfg = e4.derived_config(baseline, output_root=output / "synthetic", device=device)
            receipt = multistep_acceptance(cfg, 1, initial_identity=True)
            receipts.append(receipt)
            e4.write_json(output / f"seed_{seed}_batch1.json", receipt)
        baseline = OmegaConf.create(lock["baselines"][str(e4.SEEDS[0])])
        cfg = e4.derived_config(baseline, output_root=output / "synthetic", device=device)
        physical = multistep_acceptance(cfg, 128, initial_identity=False)
        e4.write_json(output / "batch128.json", physical)
        e4.write_json(output / "parameter_compute_report.json", e4.parameter_compute_report())
        e4.write_json(output / "gpu_acceptance.json", {"protocol": e4.PROTOCOL, "passed": True,
                      "aggregation_contract": AGGREGATION_CONTRACT, "seeds": list(e4.SEEDS), "physical_batch": 128,
                      "batch1_receipts": receipts, "batch128_receipt": physical})
    return output


def benchmark_worker(*, arm: str, mode: str, device: str, destination: Path) -> Path:
    lock, lock_hash = e4.load_lock()
    parent = destination.resolve().parent
    if destination.exists():
        raise FileExistsError(f"E4 benchmark measurement 已存在: {destination}")
    if not parent.is_relative_to((e4.ROOT / e4.OUTPUT / "benchmark").resolve()):
        raise ValueError("E4 benchmark worker 输出须在父 attempt 内")
    context = json.loads((parent / "lifecycle_started.json").read_text())
    if context["phase"] != "benchmark" or context["implementation_lock_sha256"] != lock_hash:
        raise ValueError("E4 benchmark worker 父 identity 不一致")
    if (parent / "lifecycle_completed.json").exists() or (parent / "lifecycle_failed.json").exists():
        raise ValueError("E4 benchmark worker 父 attempt 已结束")
    runtime = e4.runtime_preflight(device)
    baseline = OmegaConf.create(lock["baselines"][str(e4.SEEDS[0])])
    cfg = e4.derived_config(baseline, output_root=parent / "synthetic", device=device)
    set_seed(e4.SEEDS[0])
    if arm not in ("W0_FULL", e4.ARM) or mode not in ("eval", "train"):
        raise ValueError("E4 benchmark arm/mode 非法")
    model = (build_crd_model(baseline) if arm == "W0_FULL" else build_e4_model(cfg)).to(device)
    batch = synthetic_batch(1 if mode == "eval" else 128, 20260917, device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, _ = build_crd_optimizer(model, cfg)
    times = []
    for index in range(25):
        if index == 5:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        if mode == "eval":
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch["x"], tf=batch["tf"])
        else:
            summary = update(model, batch, loss_fn, optimizer, cfg, index)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if index >= 5:
            times.append(elapsed)
        e4.finite_tree(prediction if mode == "eval" else summary)
    e4.finite_tree(model.state_dict())
    e4.finite_tree(optimizer.state_dict())
    if len(times) != 20 or not np.isfinite(times).all() or min(times) <= 0:
        raise RuntimeError("E4 benchmark 计时无效")
    e4.write_json(destination, {"arm": arm, "mode": mode, "environment": runtime, "warmup": 5, "repeats": 20,
                  "seconds": times, "median_seconds": float(np.median(times)),
                  "iqr_seconds": float(np.quantile(times, .75) - np.quantile(times, .25)),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
                  "implementation_lock_sha256": lock_hash})
    return destination


def run_benchmark(device: str = "cuda:0") -> Path:
    lock, lock_hash = e4.load_lock()
    parent = e4.ROOT / e4.OUTPUT / "benchmark"
    with e4.phase_guard(parent, lock_hash, completed_phase="benchmark"), e4.attempt(parent, lock_hash, "benchmark") as output:
        e4.write_json(output / "environment.json", e4.runtime_preflight(device))
        e4.write_json(output / "access_receipt.json", {"waveforms": "synthetic", "real_data_read": False, "test_read": False})
        measurements = []
        for group in range(3):
            for mode in ("eval", "train"):
                arms = ("W0_FULL", e4.ARM) if group % 2 == 0 else (e4.ARM, "W0_FULL")
                for arm in arms:
                    target = output / f"group{group}_{mode}_{arm}.json"
                    command = [sys.executable, str(e4.ROOT / e4.SCRIPT_PATH), "_benchmark-worker",
                               "--arm", arm, "--mode", mode, "--device", device, "--output", str(target)]
                    with target.with_suffix(".log").open("x") as log:
                        subprocess.run(command, cwd=e4.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    measurements.append({"group": group, **json.loads(target.read_text())})
        e4.write_json(output / "benchmark.json", {"protocol": e4.PROTOCOL, "measurements": measurements,
                      "scope": "synthetic cached-input model only; eval batch1 / native training batch128; separate process per measurement"})
        e4.write_json(output / "parameter_compute_report.json", e4.parameter_compute_report())
    return output
