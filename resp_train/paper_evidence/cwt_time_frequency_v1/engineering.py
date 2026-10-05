"""FP32/BF16 等价、多步梯度和实际 batch 验收，全部使用合成输入。"""
from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import torch
from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.tf_v1_model import CRDTfV1Model
from resp_train.crd.training import build_crd_optimizer
from resp_train.losses.task import RespirationTaskLoss
from resp_train.paper_evidence.w0_structural_factorial_v1_engineering import finite_tree
from . import artifacts as io
from .model import build_model
from .spec import ARMS, SEEDS, config, OUTPUT_ROOT, ROOT, PROTOCOL, load_spec, plan

MAX_RESERVED_FRACTION = .95
ELEVATED_RESERVED_FRACTION = .90
CALIBRATION_REUSE_ALLOWED_CHANGES = frozenset({
    "resp_train/paper_evidence/cwt_time_frequency_v1/engineering.py",
    "scripts/run_cwt_time_frequency_v1_engineering.py",
    "tests/test_cwt_time_frequency_v1.py",
    "docs/experiments/cwt_time_frequency_v1_protocol_20261001.md",
})


def memory_report(peak_reserved_bytes, peak_allocated_bytes, total_memory_bytes):
    reserved, allocated, total = map(int, (peak_reserved_bytes, peak_allocated_bytes, total_memory_bytes))
    if total <= 0 or min(reserved, allocated) < 0 or allocated > reserved:
        raise ValueError("显存统计无效")
    fraction = reserved / total
    return {"peak_reserved_bytes": reserved, "peak_allocated_bytes": allocated, "total_memory_bytes": total,
            "peak_reserved_fraction": fraction, "peak_allocated_fraction": allocated / total,
            "max_reserved_fraction": MAX_RESERVED_FRACTION, "elevated_memory": fraction > ELEVATED_RESERVED_FRACTION,
            "memory_passed": fraction <= MAX_RESERVED_FRACTION}


def device_memory_report(device):
    return memory_report(torch.cuda.max_memory_reserved(device), torch.cuda.max_memory_allocated(device),
                         torch.cuda.get_device_properties(device).total_memory)


def require_memory_passed(record):
    expected = memory_report(record["peak_reserved_bytes"], record["peak_allocated_bytes"], record["total_memory_bytes"])
    if any(record.get(k) != value for k, value in expected.items()):
        raise ValueError("显存记录/验收策略不一致")
    if not expected["memory_passed"]:
        raise RuntimeError(f"显存 reserved峰值 {expected['peak_reserved_fraction']:.2%} 超过 {MAX_RESERVED_FRACTION:.0%}；"
                           f"allocated峰值 {expected['peak_allocated_fraction']:.2%}")


def audit_calibration_reuse(payload, current_files):
    if payload["spec"] != load_spec() or payload["provenance"]["dependencies"] != io.dependencies():
        raise ValueError("校准复用要求变换规格和依赖环境不变")
    old_files = payload["provenance"]["files"]
    if set(old_files) != set(current_files):
        raise ValueError("校准来源集合发生变化，不能按显存策略修订直接复用")
    changed = {p: {"old": old_files[p], "new": current_files[p]} for p in old_files if old_files[p] != current_files[p]}
    forbidden = set(changed) - CALIBRATION_REUSE_ALLOWED_CHANGES
    if forbidden:
        raise ValueError(f"校准实际依赖发生变化，禁止复用: {sorted(forbidden)}")
    if payload.get("numerical_checks_passed") is not True or set(payload["representations"]) != set(ARMS):
        raise ValueError("来源校准未完成完整矩阵")
    return changed


def prepare_reusing_calibration(calibration):
    """显存策略修订的新合成会话；原校准文件、源码快照和数学定义均保留。"""
    calibration = Path(calibration).resolve()
    io.verify_stage(calibration)
    payload = io.read_json(calibration / "calibration.json")
    current_files = {str(p.relative_to(ROOT)): io.identity(p) for p in io.source_files()}
    changed = audit_calibration_reuse(payload, current_files)
    output = OUTPUT_ROOT / ("session_" + io.stamp())
    output.mkdir(parents=True, exist_ok=False)
    try:
        provenance = io.provenance(output)
        if provenance["files"] != current_files:
            raise RuntimeError("准备过程中源码改变")
        io.write_json(output / "session.json", {"protocol": PROTOCOL, "spec": load_spec(), "plan": plan(),
                      "representations": payload["representations"], "provenance": provenance,
                      "calibration": {"path": str(calibration), **io.identity(calibration / "manifest.json")},
                      "calibration_reuse": {"reason": "仅修改GPU工程显存策略与相应入口/测试/说明；变换和校准计算源码字节不变",
                                             "source_changes": changed, "recomputed": False},
                      "engineering_memory_policy": {"max_reserved_fraction": MAX_RESERVED_FRACTION,
                                                    "elevated_reserved_fraction": ELEVATED_RESERVED_FRACTION},
                      "execution_scope": "synthetic_only", "parameter_review": None, "research_test_open": False})
        io.write_json(output / "session_receipt.json", io.identity(output / "session.json"))
    except BaseException:
        import traceback
        io.write_json(output / "failed.json", {"traceback": traceback.format_exc()})
        raise
    return output


def environment(device):
    device = torch.device(device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ValueError("要求显式可用 cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("; ".join(problems))
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU 不支持 BF16")
    p = torch.cuda.get_device_properties(device)
    return {"dependencies": io.dependencies(), "device": str(device), "name": p.name,
            "total_memory": p.total_memory, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def synthetic_batch(rep, size, device):
    generator = torch.Generator(device="cpu").manual_seed(20260930)
    t = torch.arange(18000).float() / 100
    phase = torch.arange(size).float()[:, None] / 7
    target = ((1 + .3 * torch.sin(2 * torch.pi * .035 * t)) * torch.sin(2 * torch.pi * .23 * t + phase))[:, None]
    x = target + .2 * torch.randn(size, 1, 18000, generator=generator)
    w = torch.rand(size, *rep["shape"], generator=generator)
    return {"x": x.to(device), "target": target.to(device), "tf": {"w": w.to(device)}}


def equivalence(rep, seed, device):
    native = CRDTfV1Model("crd_tf102_w", seed).to(device).eval()
    candidate = build_model(seed, rep).to(device).eval()
    for key, value in native.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    # 零初始化 FiLM 会屏蔽条件差异；用同一非零投影再检验实际条件路径。
    with torch.no_grad():
        weights = native.branches["w"].final_projection.weight
        weights.copy_(torch.linspace(-.03, .03, weights.numel(), device=device).reshape_as(weights))
        candidate.load_state_dict(native.state_dict(), strict=True)
    batch = synthetic_batch(rep, 1, device)
    errors = {}
    for amp in (False, True):
        with torch.no_grad(), torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            x = native(batch["x"], tf=batch["tf"])["waveform"]
            y = candidate(batch["x"], tf=batch["tf"])["waveform"]
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
        errors["bf16" if amp else "fp32"] = float((x.float() - y.float()).abs().max())
    return errors


def engineering_cell(arm, rep, seed, size, device):
    cfg = config(arm, seed, OUTPUT_ROOT / "synthetic_unused", device)
    model = build_model(seed, rep).to(device).train()
    optimizer, _ = build_crd_optimizer(model, cfg)
    task = RespirationTaskLoss(cfg).to(device)
    batch = synthetic_batch(rep, size, device)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(batch["x"], tf=batch["tf"])
            loss, _ = task(prediction, batch["target"])
        finite_tree(prediction, label="prediction")
        finite_tree(loss, label="loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"缺少梯度: {name}")
            finite_tree(parameter.grad, label=name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    groups = {}
    for name, p in model.named_parameters():
        group = name.rsplit(".", 1)[0]
        groups[group] = groups.get(group, False) or (bool(p.grad.ne(0).any()) and not torch.equal(p, initial[name]))
    if not all(groups.values()):
        raise RuntimeError(f"第三步仍不活动的模块: {[k for k, v in groups.items() if not v]}")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(batch["x"], tf=batch["tf"])
        loss, _ = task(prediction, batch["target"])
    finite_tree(prediction, label="eval")
    finite_tree(loss, label="eval_loss")
    if prediction["waveform"].shape != (size, 1, 18000):
        raise ValueError("输出 shape 不正确")
    return {"arm": arm, "seed": seed, "batch_size": size, "updates": 3,
            "eval_loss": float(loss), "active_modules": len(groups), **device_memory_report(device)}


def run_gpu(session, device, retry=False):
    frozen = io.load_session(session)
    if "engineering_reference" in frozen:
        return require_gpu(session, device)
    current = environment(device)
    key = io.binding(session, "gpu", device=str(torch.device(device)))
    parent = session / "engineering" / str(torch.device(device)).replace(":", "_")
    prior = io.completed(parent, key)
    if prior:
        if io.read_json(prior / "environment.json") != current:
            raise ValueError("设备执行环境已改变")
        return prior
    with io.mutex(OUTPUT_ROOT / f".device_{torch.device(device).index}.mutex"):
        with io.attempt(parent, key, retry) as output:
            io.write_json(output / "environment.json", current)
            io.write_json(output / "memory_policy.json", {"max_reserved_fraction": MAX_RESERVED_FRACTION,
                          "elevated_reserved_fraction": ELEVATED_RESERVED_FRACTION, "measurement": "peak_reserved/total_device_memory"})
            eq = {str(seed): equivalence(frozen["representations"]["B"], seed, device) for seed in SEEDS}
            io.write_json(output / "equivalence.json", eq)
            # 条件机制也需在原生 CUDA/BF16 下验收，CPU 公式不能代替 GPU 重放。
            from .interventions import paired_batch, CONDITIONS
            gn_records = []
            for seed in SEEDS:
                for size in (1, 128):
                    probe = build_model(seed, frozen["representations"]["B"]).to(device).eval()
                    with torch.no_grad():
                        probe.branches["w"].final_projection.weight.fill_(.001)
                    batch = synthetic_batch(frozen["representations"]["B"], size, device)
                    shifts = torch.tensor([60, 120, 240], device=device).expand(size, -1)
                    observed = []
                    torch.cuda.reset_peak_memory_stats(device)
                    for condition, *rest in paired_batch(probe, batch["x"], batch["tf"]["w"],
                                                         frozen["representations"]["B"]["frequencies_hz"], shifts):
                        observed.append(condition)
                        del rest
                    record = {"seed": seed, "batch_size": size, "conditions": observed, **device_memory_report(device)}
                    io.write_json(output / f"gn_{seed}_b{size}.json", record)
                    if observed != list(CONDITIONS):
                        raise RuntimeError("GN 条件矩阵不完整")
                    require_memory_passed(record)
                    gn_records.append(record)
                    del probe, batch
                    gc.collect()
                    torch.cuda.empty_cache()
            io.write_json(output / "gn_acceptance.json", {"passed": True, "cases": gn_records})
            records = []
            for arm in ARMS:
                for seed, size in [*( (seed, 1) for seed in SEEDS), (SEEDS[0], 128)]:
                    print(f"synthetic {arm}/{seed}/batch{size}", flush=True)
                    try:
                        record = engineering_cell(arm, frozen["representations"][arm], seed, size, device)
                    except torch.OutOfMemoryError:
                        io.write_json(output / f"{arm}_{seed}_b{size}_oom.json", {"arm": arm, "seed": seed,
                                      "batch_size": size, "failure": "cuda_oom", **device_memory_report(device)})
                        raise
                    io.write_json(output / f"{arm}_{seed}_b{size}.json", record)
                    require_memory_passed(record)
                    records.append(record)
                    gc.collect()
                    torch.cuda.empty_cache()
            io.write_json(output / "acceptance.json", {"passed": True, "synthetic_only": True, "cells": records})
            io.verify_provenance(frozen["provenance"], session)
    return output


def require_gpu(session, device):
    current = environment(device)
    frozen = io.load_session(session)
    if "engineering_reference" in frozen:
        from .formal_launch import engineering_reference
        path = engineering_reference(session, str(torch.device(device)))
    else:
        path = io.completed(session / "engineering" / str(torch.device(device)).replace(":", "_"),
                            io.binding(session, "gpu", device=str(torch.device(device))))
    if path is None or io.read_json(path / "environment.json") != current:
        raise RuntimeError("先在同一 GPU/环境完成工程验收")
    report = io.read_json(path / "acceptance.json")
    gn = io.read_json(path / "gn_acceptance.json")
    if gn.get("passed") is not True or {(x["seed"], x["batch_size"]) for x in gn["cases"]} != {(s, b) for s in SEEDS for b in (1, 128)}:
        raise ValueError("GN GPU 验收不完整")
    for record in [*gn["cases"], *report["cells"]]:
        require_memory_passed(record)
    expected = {(a, s, 1) for a in ARMS for s in SEEDS} | {(a, SEEDS[0], 128) for a in ARMS}
    if (report["passed"] is not True or len(report["cells"]) != len(expected)
            or {(c["arm"], c["seed"], c["batch_size"]) for c in report["cells"]} != expected
            or any(c["updates"] != 3 or not np.isfinite(c["eval_loss"]) or c["active_modules"] <= 0 for c in report["cells"])):
        raise ValueError("GPU 验收矩阵不完整")
    return path
