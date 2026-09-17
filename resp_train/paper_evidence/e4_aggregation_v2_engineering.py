"""四候选的 synthetic GPU 验收和匹配效率测量。"""

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
from resp_train.crd.training import build_crd_optimizer
from resp_train.losses.task import RespirationTaskLoss
from resp_train.utils.run import set_seed
from resp_train.paper_evidence import e4_aggregation_v2 as experiment
from resp_train.paper_evidence.e4_aggregation_v2_model import ARMS, build_model
from resp_train.paper_evidence.e4_scale_aggregation_engineering import synthetic_batch, update

ACCEPTANCE_MIN_UPDATES = 5
ACCEPTANCE_MAX_UPDATES = 20


def initial_identity(reference, candidate, batch, device):
    original = reference.state_dict()
    current = candidate.state_dict()
    if not all(name in current and torch.equal(value, current[name]) for name, value in original.items()):
        raise RuntimeError("E4-v2 公共初始 state 不一致")
    reference.to(device).eval(); candidate.to(device).eval()
    contexts = {}
    hooks = [reference.branches["w"].temporal.register_forward_pre_hook(lambda _, args: contexts.update(reference=args[0].detach().clone())),
             candidate.branches["w"].temporal.register_forward_pre_hook(lambda _, args: contexts.update(candidate=args[0].detach().clone()))]
    def check_mean(_, args, output):
        if not torch.equal(args[0].mean(dim=2), output):
            raise RuntimeError("E4-v2 初始聚合不等于 W0 mean")
    hooks.append(candidate.branches["w"].aggregation.register_forward_hook(check_mean))
    try:
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            expected = reference(batch["x"], tf=batch["tf"])
            observed = candidate(batch["x"], tf=batch["tf"])
        experiment.finite_tree(expected); experiment.finite_tree(observed)
        if not torch.equal(contexts["reference"], contexts["candidate"]):
            raise RuntimeError("E4-v2 temporal 初始输入不一致")
        if expected.keys() != observed.keys():
            raise RuntimeError("E4-v2 输出字段漂移")
        errors = []
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=1e-5, atol=1e-6)
            errors.append(float((observed[key].float() - expected[key].float()).abs().max()))
    finally:
        for hook in hooks:
            hook.remove()
    return {"common_state_equal": True, "aggregation_equal": True, "temporal_input_equal": True,
            "waveform_max_abs_error": max(errors), "rtol": 1e-5, "atol": 1e-6}


def acceptance_updates(model, batch, loss_fn, optimizer, cfg, record, diagnostic_dir=None):
    """在原生 warm-up 内检验通路及实际更新，并保存每步的可诊断证据。"""
    aggregation = model.branches["w"].aggregation
    initial = {name: value.detach().clone() for name, value in aggregation.named_parameters()}
    record.update(updates=0, summaries=[], new_parameter_gradient_norms=[],
                  new_parameter_max_abs_changes=[], new_parameters_updated={})
    hook = model.register_forward_hook(lambda _, args, result: experiment.finite_tree(result))
    try:
        for index in range(ACCEPTANCE_MAX_UPDATES):
            record["stage"] = f"update_{index + 1}"
            summary = update(model, batch, loss_fn, optimizer, cfg, index)
            experiment.finite_tree(summary)
            experiment.finite_tree(model.state_dict()); experiment.finite_tree(optimizer.state_dict())
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"E4-v2 gradient 通路缺失: {name}")
                experiment.finite_tree(parameter.grad)
            # FP64 范数避免极小但非零的梯度在平方求和时下溢为零。
            gradients = {name: float(parameter.grad.double().norm()) for name, parameter in aggregation.named_parameters()}
            changes = {name: float((parameter.detach().double() - initial[name].double()).abs().max())
                       for name, parameter in aggregation.named_parameters()}
            changed = {name: value > 0 for name, value in changes.items()}
            record["updates"] = index + 1
            record["summaries"].append(summary)
            record["new_parameter_gradient_norms"].append(gradients)
            record["new_parameter_max_abs_changes"].append(changes)
            record["new_parameters_updated"] = changed
            if diagnostic_dir is not None:
                experiment.write_json(diagnostic_dir / f"step_{index + 1:03d}.json", {
                    "arm": record["arm"], "seed": record["seed"], "batch_size": record["batch_size"],
                    "update": index + 1, "summary": summary, "gradient_norms": gradients,
                    "max_abs_changes": changes, "parameters_updated": changed})
            # 两个零末层依次打开；梯度非零与 FP32 参数发生可见变化分别验证。
            if index + 1 >= ACCEPTANCE_MIN_UPDATES and all(changed.values()) and all(v > 0 for v in gradients.values()):
                return
        stalled = [name for name, changed in record["new_parameters_updated"].items() if not changed]
        zero = [name for name, value in record["new_parameter_gradient_norms"][-1].items() if value == 0]
        raise RuntimeError(f"E4-v2 {ACCEPTANCE_MAX_UPDATES} 步验收失败: 未更新参数={stalled}; 零梯度参数={zero}")
    finally:
        hook.remove()


def acceptance_case(cfg, batch_size, check_initial, *, diagnostic_dir=None):
    record = {"arm": str(cfg.model.aggregation_v2.arm), "seed": int(cfg.training.seed),
              "batch_size": batch_size, "minimum_updates": ACCEPTANCE_MIN_UPDATES,
              "maximum_updates": ACCEPTANCE_MAX_UPDATES, "status": "running", "stage": "initialization"}
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(exist_ok=False)
    try:
        _acceptance_case(cfg, batch_size, check_initial, record, diagnostic_dir)
        record.update(status="passed", stage="completed")
    except BaseException as exc:
        record.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        exc.add_note(f"E4-v2 arm={record['arm']} seed={record['seed']} batch={batch_size} stage={record['stage']}; diagnostics={diagnostic_dir}")
        raise
    finally:
        if diagnostic_dir is not None:
            experiment.write_json(diagnostic_dir / "case.json", record)
    return record


def _acceptance_case(cfg, batch_size, check_initial, record, diagnostic_dir):
    seed, device = int(cfg.training.seed), str(cfg.training.device)
    set_seed(seed)
    model = build_model(cfg)
    batch = synthetic_batch(batch_size, seed + 1700, device)
    equality = None
    if check_initial:
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        reference = build_crd_model(cfg)
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
            raise RuntimeError("E4-v2 与 W0 构造后的 CPU/CUDA RNG 不一致")
        equality = {**initial_identity(reference, model, batch, device), "cpu_rng_equal": True, "cuda_rng_equal": True}
        del reference
    model.to(device)
    record["initial_identity"] = equality
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, _ = build_crd_optimizer(model, cfg)
    acceptance_updates(model, batch, loss_fn, optimizer, cfg, record, diagnostic_dir)
    record["stage"] = "memory"
    allocated, reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    record.update(peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                  device_total_bytes=total, peak_reserved_fraction=reserved / total)
    if reserved / total > .8:
        raise RuntimeError(f"E4-v2 batch={batch_size} reserved fraction={reserved / total} 超过0.8")


def run_gpu_acceptance(device="cuda:0"):
    lock, digest = experiment.load_lock()
    parent = experiment.ROOT / experiment.OUTPUT / "gpu_acceptance"
    with experiment.phase_guard(parent, digest):
        experiment.reject_completed(parent, "gpu_acceptance", digest)
        with experiment.attempt(parent, digest, "gpu_acceptance") as output:
            experiment.write_json(output / "environment.json", experiment.runtime_preflight(device))
            experiment.write_json(output / "access_receipt.json", {"waveforms": "synthetic", "checkpoint": "fresh initialization", "real_data_read": False})
            records = []
            for arm in ARMS:
                for seed in experiment.SEEDS:
                    baseline = OmegaConf.create(lock["baselines"][str(seed)])
                    cfg = experiment.derived_config(baseline, arm, lock["frequency"]["values_hz"], output_root=output / "synthetic", device=device)
                    record = acceptance_case(cfg, 1, True, diagnostic_dir=output / f"{arm}_{seed}_batch1_diagnostics")
                    experiment.write_json(output / f"{arm}_{seed}_batch1.json", record); records.append(record)
                baseline = OmegaConf.create(lock["baselines"][str(experiment.SEEDS[0])])
                cfg = experiment.derived_config(baseline, arm, lock["frequency"]["values_hz"], output_root=output / "synthetic", device=device)
                record = acceptance_case(cfg, 128, False, diagnostic_dir=output / f"{arm}_batch128_diagnostics")
                experiment.write_json(output / f"{arm}_batch128.json", record); records.append(record)
            experiment.write_json(output / "gpu_acceptance.json", {"protocol": experiment.PROTOCOL, "passed": True,
                "arms": list(ARMS), "seeds": list(experiment.SEEDS), "physical_batch": 128, "records": records})
            experiment.write_json(output / "parameter_compute_report.json", experiment.parameter_compute_report())
    return output


def benchmark_worker(arm: str, mode: str, device: str, destination: Path):
    lock, digest = experiment.load_lock()
    parent = destination.resolve().parent
    if destination.exists() or not parent.is_relative_to((experiment.ROOT / experiment.OUTPUT / "benchmark").resolve()):
        raise ValueError("E4-v2 measurement 已存在或路径越界")
    context = json.loads((parent / "lifecycle_started.json").read_text())
    if context["phase"] != "benchmark" or context["implementation_lock_sha256"] != digest:
        raise ValueError("E4-v2 measurement 父 identity 漂移")
    if (parent / "lifecycle_completed.json").exists() or (parent / "lifecycle_failed.json").exists():
        raise ValueError("E4-v2 measurement 父 attempt 已结束")
    environment = experiment.runtime_preflight(device)
    baseline = OmegaConf.create(lock["baselines"][str(experiment.SEEDS[0])])
    if arm not in ("W0_FULL", *ARMS) or mode not in ("eval", "train"):
        raise ValueError("E4-v2 benchmark arm/mode 错误")
    cfg = baseline if arm == "W0_FULL" else experiment.derived_config(baseline, arm, lock["frequency"]["values_hz"], output_root=parent / "synthetic", device=device)
    cfg.training.device = device
    set_seed(experiment.SEEDS[0])
    model = (build_crd_model(cfg) if arm == "W0_FULL" else build_model(cfg)).to(device)
    batch = synthetic_batch(1 if mode == "eval" else 128, 20260917, device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, _ = build_crd_optimizer(model, cfg)
    times = []
    for index in range(25):
        if index == 5:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device); start = time.perf_counter()
        if mode == "eval":
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch["x"], tf=batch["tf"])
        else:
            result = update(model, batch, loss_fn, optimizer, cfg, index)
        torch.cuda.synchronize(device); elapsed = time.perf_counter() - start
        if index >= 5:
            times.append(elapsed)
        experiment.finite_tree(result)
    experiment.finite_tree(model.state_dict()); experiment.finite_tree(optimizer.state_dict())
    if not np.isfinite(times).all() or min(times) <= 0:
        raise RuntimeError("E4-v2 measurement 计时无效")
    experiment.write_json(destination, {"arm": arm, "mode": mode, "environment": environment,
        "warmup": 5, "repeats": 20, "seconds": times, "median_seconds": float(np.median(times)),
        "iqr_seconds": float(np.quantile(times, .75) - np.quantile(times, .25)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_total_bytes": torch.cuda.get_device_properties(device).total_memory, "implementation_lock_sha256": digest})
    return destination


def run_benchmark(device="cuda:0"):
    _, digest = experiment.load_lock()
    parent = experiment.ROOT / experiment.OUTPUT / "benchmark"
    with experiment.phase_guard(parent, digest):
        experiment.reject_completed(parent, "benchmark", digest)
        with experiment.attempt(parent, digest, "benchmark") as output:
            experiment.write_json(output / "environment.json", experiment.runtime_preflight(device))
            experiment.write_json(output / "access_receipt.json", {"waveforms": "synthetic", "real_data_read": False})
            measurements = []
            arms = ("W0_FULL", *ARMS)
            for group in range(3):
                order = arms if group % 2 == 0 else tuple(reversed(arms))
                for mode in ("eval", "train"):
                    for arm in order:
                        path = output / f"group{group}_{mode}_{arm}.json"
                        command = [sys.executable, str(experiment.ROOT / experiment.SCRIPT_PATH), "_benchmark-worker",
                                   "--arm", arm, "--mode", mode, "--device", device, "--output", str(path)]
                        with path.with_suffix(".log").open("x") as log:
                            subprocess.run(command, cwd=experiment.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                        measurements.append({"group": group, **json.loads(path.read_text())})
            experiment.write_json(output / "benchmark.json", {"protocol": experiment.PROTOCOL, "measurements": measurements,
                "scope": "synthetic model-only; eval batch1 / native training batch128; independent process per measurement"})
            experiment.write_json(output / "parameter_compute_report.json", experiment.parameter_compute_report())
    return output
