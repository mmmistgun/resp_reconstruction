"""APOR独立GELU/H64/Direct对照：复用B0三seed，新增九次训练。"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import check_crd_dependencies, crd_dependency_versions, load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.training import build_crd_optimizer, train_crd_one_epoch
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import summarize_task_metrics, evaluate_task_predictions
from resp_train.engine import collect_predictions
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence import patch_apor_v1 as legacy
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_engineering import synthetic_batch
from scripts.apor_gelu_refiner_v1_model import (
    ARMS, ARM_SPECS, PROTOCOL, SEEDS, TRAIN_ARMS, REFERENCE_ARMS,
    GeluRefinerModel, build_model, model_contract,
)
from resp_train.paper_evidence.w0_structural_factorial_v1_engineering import finite_tree
from resp_train.paper_evidence.w0_structural_factorial_v1_formal import (
    P2_LOCK_PATH, P2_LOCK_SHA256, array_hash, state_dict_identity,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path("configs/apor_gelu_refiner_v1/experiment.yaml")
PROTOCOL_PATH = Path("docs/experiments/apor_gelu_refiner_v1_protocol_20260930.md")
STRUCTURE_DECISION = ROOT / "runs/apor_grid_decoder_v2/session_20260930T093614Z_511f2ff06f0f/summary/attempt_20260930T111411Z_d03d6929f568/validation_decision.json"
STRUCTURE_SHA = "4e69b062349e4139746cc83c7b7aff4f2e305b27afbdf10b2cdc9fd3a4810a8c"
SOURCE_SESSION = ROOT / "runs/patch_apor_v1/session_20260929T181607Z_77e89165fef0"
SOURCE_SHA256 = "cdf6337db77d73724f93508a3568669863a17011066e7d332cf69b18b4e98fa2"
OUTPUT_ROOT = ROOT / "runs/apor_gelu_refiner_v1"
COUNTS = {"train": 10141, "val": 2675}
SAMP_IDS = {"train": 32, "val": 7}
UPDATES_PER_EPOCH = 80
PLANNED_UPDATES = 6400
COMPARISONS = tuple((arm, "B0") for arm in TRAIN_ARMS)


def write_json(path: Path, value: Any) -> None:
    """排他写入，避免成功产物或失败现场被重新写入。"""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_spec():
    spec = OmegaConf.to_container(OmegaConf.load(ROOT / SPEC), resolve=True)
    expected = {
        "schema_version": 1, "protocol": PROTOCOL, "arms": list(ARMS), "train_arms": list(TRAIN_ARMS), "seeds": list(SEEDS),
        "reference_arms": {a: ARM_SPECS[a].source_arm for a in REFERENCE_ARMS},
        "source_session": str(SOURCE_SESSION.relative_to(ROOT)), "source_session_sha256": SOURCE_SHA256,
        "structure_decision_sha256": STRUCTURE_SHA,
        "output_root": "runs/apor_gelu_refiner_v1",
        "training": {"epochs": 80, "min_epoch": 30, "patience": 15, "min_delta": 0.0,
                     "planned_updates": 6400, "batch_size": 128, "accumulation": 1, "amp_dtype": "bfloat16",
                     "selector": "full_validation_local_rr_strict_minimum_earliest_tie"},
        "data": {"splits": ["train", "val"], "windows": COUNTS, "samp_ids": SAMP_IDS},
        "engineering": {"updates": 3, "physical_batch": 128, "max_reserved_fraction": 0.85},
        "research_test": "not_enabled",
    }
    if spec != expected:
        raise ValueError("实验矩阵/配置合同漂移")
    return spec


def config(arm: str, seed: int, output: Path, device: str = "cpu"):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("arm/seed不属于12-cell独立对照矩阵")
    load_spec()
    cfg = legacy.config(ARM_SPECS[arm].source_arm, seed, output, device)
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "apor_gelu_refiner_v1"
    cfg.protocol.execution_gate = "session_gpu_acceptance"
    cfg.model.apor_gelu_refiner_v1 = model_contract(arm)
    cfg.training.early_stopping_enabled = True
    cfg.training.early_stopping_min_epoch = 30
    cfg.training.early_stopping_patience = 15
    cfg.training.early_stopping_min_delta = 0.0
    cfg.training.device = str(device)
    cfg.training.show_progress = False
    cfg.outputs.run_root = str(output)
    return cfg


def plan():
    return [{"arm": arm, "seed": seed, "reference": "B0"}
            for seed in SEEDS for arm in TRAIN_ARMS]


def scientific_plan():
    return [{"arm": arm, "seed": seed, "reused": arm in REFERENCE_ARMS} for seed in SEEDS for arm in ARMS]


def shard_plan(index: int, count: int):
    """按固定矩阵位置交错分片，每个cell恰好分配一次。"""
    if not 1 <= count <= len(plan()) or not 0 <= index < count:
        raise ValueError("非法分片编号/数量")
    return plan()[index::count]


def parallel_layout(devices):
    normalized = [str(torch.device(device)) for device in devices]
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise ValueError("并行执行至少需要两张不同GPU")
    if any(torch.device(device).type != "cuda" or torch.device(device).index is None for device in normalized):
        raise ValueError("并行设备要求显式cuda:<index>")
    return {"protocol": PROTOCOL, "shard_count": len(normalized), "workers": [
        {"index": index, "device": device, "cells": shard_plan(index, len(normalized))}
        for index, device in enumerate(normalized)]}


def source_files() -> list[Path]:
    # 保存全部仓库Python依赖，避免只记录模型文件而漏掉数据/损失的传递依赖。
    paths = legacy.source_files()
    paths += list((ROOT / "configs/crd_tf_v1").glob("*.yaml"))
    paths += [ROOT / SPEC, ROOT / PROTOCOL_PATH, ROOT / "scripts/run_apor_gelu_refiner_v1.py",
              ROOT / "scripts/apor_gelu_refiner_v1_model.py", ROOT / "scripts/apor_gelu_refiner_v1_runtime.py",
              ROOT / "tests/test_apor_gelu_refiner_v1.py", ROOT / P2_LOCK_PATH]
    return sorted(set(paths))


def dependencies():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "packages": crd_dependency_versions()}


def scientific_config(cfg):
    """仅消去本轮命名与操作性路径，完整比较其余训练合同。"""
    value = OmegaConf.to_container(cfg, resolve=True)
    value["protocol"].pop("name", None)
    value["protocol"].pop("stage", None)
    value["model"].pop("apor_gelu_refiner_v1", None)
    value["outputs"].pop("run_root", None)
    for key in ("device", "show_progress"):
        value["training"].pop(key, None)
    return value


def audit_references():
    if sf.sha256_file(STRUCTURE_DECISION) != STRUCTURE_SHA:
        raise ValueError("APOR的validation结构选择身份漂移")
    decision = read_json(STRUCTURE_DECISION)
    if decision["candidate"] != "N0" or decision["test_used"] is not False:
        raise ValueError("本阶段只允许已固定N0结构")
    if sf.sha256_file(SOURCE_SESSION / "session.json") != SOURCE_SHA256:
        raise ValueError("历史session身份漂移")
    original = legacy.load_session(SOURCE_SESSION)
    if original["dependencies"] != dependencies():
        raise ValueError("历史与当前依赖环境不一致")
    references = {}
    for arm in REFERENCE_ARMS:
        source_arm = ARM_SPECS[arm].source_arm
        for seed in SEEDS:
            path = legacy.completed(SOURCE_SESSION / "formal" / source_arm / f"seed_{seed}",
                                    SOURCE_SESSION, "formal", arm=source_arm, seed=seed)
            if path is None:
                raise ValueError(f"缺少冻结参照{source_arm}/{seed}")
            result = read_json(path / "result.json")
            run = path / result["run_dir"]
            source_cfg = OmegaConf.load(run / "config.yaml")
            if scientific_config(source_cfg) != scientific_config(config(arm, seed, Path("REFERENCE"))):
                raise ValueError("复用参照的训练合同不等价")
            best = sf.validate_history(pd.read_csv(run / "train_history.csv"), source_cfg)
            if best != result["best_epoch"] or result["validation_rows"] != COUNTS["val"]:
                raise ValueError("历史checkpoint选择/分母不一致")
            references[f"{arm}/{seed}"] = {
                "arm": arm, "seed": seed, "source_arm": source_arm, "attempt": str(path), "run": str(run),
                "manifest": sf.identity(path / "manifest.json"), "selected_epoch": best,
                "checkpoint": {"path": str(run / "checkpoint_best_local_rr.pt"), **sf.identity(run / "checkpoint_best_local_rr.pt")},
                "metrics": {"path": str(run / "metrics.csv"), **sf.identity(run / "metrics.csv")},
                "initialization": read_json(path / "initialization.json"),
                "environment": read_json(path / "environment.json"),
            }
    for seed in SEEDS:
        frozen = next(c for c in decision["candidate_checkpoints"] if c["seed"] == seed)
        expected = references[f"B0/{seed}"]["checkpoint"]
        if {k: v for k,v in frozen.items() if k != "seed"} != expected:
            raise ValueError("B0来源与已固定结构候选checkpoint不一致")
    return references


def verify_references(payload, *, checkpoint=False):
    expected = {f"{a}/{s}" for a in REFERENCE_ARMS for s in SEEDS}
    if set(payload["references"]) != expected:
        raise ValueError("复用参照集合不完整")
    for ref in payload["references"].values():
        sf.verify_identity(Path(ref["attempt"]) / "manifest.json", ref["manifest"])
        sf.verify_identity(Path(ref["metrics"]["path"]), ref["metrics"])
        if checkpoint:
            sf.verify_identity(Path(ref["checkpoint"]["path"]), ref["checkpoint"])


def prepare() -> Path:
    load_spec()
    # 只读既有训练产物和源码；不打开真实信号或cache数组。
    if sf.sha256_file(ROOT / P2_LOCK_PATH) != P2_LOCK_SHA256:
        raise ValueError("历史训练数据来源身份漂移")
    references = audit_references()
    output = OUTPUT_ROOT / ("session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in source_files():
        relative = path.relative_to(ROOT)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        files[str(relative)] = sf.identity(destination)
    templates = {f"{arm}/{seed}": OmegaConf.to_container(config(arm, seed, Path("FORMAL_OUTPUT")), resolve=True)
                 for seed in SEEDS for arm in ARMS}
    payload = {"protocol": PROTOCOL, "spec": load_spec(), "contracts": {arm: model_contract(arm) for arm in ARMS},
               "files": files, "templates": templates, "dependencies": dependencies(), "git": sf.git_state(ROOT),
               "created_at": datetime.now(timezone.utc).isoformat(), "source_root": str(ROOT),
               "execution_mode": "reuse_three_train_nine", "source_session_sha256": SOURCE_SHA256,
               "references": references, "test_access": False,
               "structure_decision": {"path": str(STRUCTURE_DECISION), **sf.identity(STRUCTURE_DECISION)}}
    write_json(output / "session.json", payload)
    write_json(output / "session_receipt.json", sf.identity(output / "session.json"))
    return output


def load_session(session: Path) -> dict:
    session = session.resolve()
    sf.verify_identity(session / "session.json", read_json(session / "session_receipt.json"))
    payload = read_json(session / "session.json")
    if payload["protocol"] != PROTOCOL or payload["spec"] != load_spec() or payload["dependencies"] != dependencies():
        raise ValueError("会话合同或运行环境漂移")
    if payload["contracts"] != {arm: model_contract(arm) for arm in ARMS}:
        raise ValueError("模型定义漂移")
    expected_paths = {str(path.relative_to(ROOT)) for path in source_files()}
    if set(payload["files"]) != expected_paths:
        raise ValueError("会话源码集合漂移")
    for relative, expected in payload["files"].items():
        sf.verify_identity(ROOT / relative, expected)
        sf.verify_identity(session / "source_snapshot" / relative, expected)
    templates = {f"{arm}/{seed}": OmegaConf.to_container(config(arm, seed, Path("FORMAL_OUTPUT")), resolve=True)
                 for seed in SEEDS for arm in ARMS}
    if templates != payload["templates"]:
        raise ValueError("resolved配置漂移")
    if payload["execution_mode"] != "reuse_three_train_nine" or payload["source_session_sha256"] != SOURCE_SHA256:
        raise ValueError("参照复用模式漂移")
    sf.verify_identity(Path(payload["structure_decision"]["path"]), payload["structure_decision"])
    verify_references(payload)
    return payload


@contextmanager
def file_mutex(path: Path, *, shared=False, wait_timeout=0.0):
    """执行互斥默认立即失败；共享参考写入允许有界等待。"""
    with path.open("a") as handle:
        deadline = time.monotonic() + wait_timeout
        while True:
            try:
                mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(handle, mode | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"执行锁等待超时或已被占用: {path.name}") from exc
                time.sleep(min(.05, max(0., deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def session_mutex(session: Path, *, shared=False):
    return file_mutex(session / ".execution.lock", shared=shared)


def save_validation_reference(root: Path, target: np.ndarray):
    """原子发布完整参考；并发写入者等待后核对，不把锁竞争当作训练失败。"""
    reference = root / "validation_reference.npy"
    if target.dtype != np.float32 or not np.isfinite(target).all():
        raise ValueError("validation参考必须为有限float32数组")
    with file_mutex(root / ".validation_reference.lock", wait_timeout=120.):
        if reference.exists():
            saved = np.load(reference, mmap_mode="r", allow_pickle=False)
            if saved.dtype != target.dtype or not np.array_equal(saved, target):
                raise ValueError("跨cell validation参考波形不一致")
        else:
            temporary = root / f".validation_reference_{uuid4().hex}.npy"
            try:
                with temporary.open("xb") as stream:
                    np.save(stream, target, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                # 同目录硬链接原子发布；目标已存在时拒绝覆盖。
                os.link(temporary, reference)
            finally:
                temporary.unlink(missing_ok=True)
    return reference


@contextmanager
def attempt(parent: Path, session: Path, phase: str, **fields):
    output = parent / ("attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    context = {"protocol": PROTOCOL, "phase": phase, "session_sha256": sf.sha256_file(session / "session.json"), **fields}
    write_json(output / "started.json", context)
    try:
        yield output
        files = {str(path.relative_to(output)): sf.identity(path) for path in sorted(output.rglob("*")) if path.is_file()}
        write_json(output / "manifest.json", {**context, "status": "completed", "files": files})
        write_json(output / "freeze_receipt.json", sf.identity(output / "manifest.json"))
    except BaseException:
        write_json(output / "failed.json", {**context, "traceback": traceback.format_exc()})
        raise


def verify_attempt(path: Path, session: Path, phase: str, **fields):
    sf.verify_identity(path / "manifest.json", read_json(path / "freeze_receipt.json"))
    manifest = read_json(path / "manifest.json")
    expected = {"protocol": PROTOCOL, "phase": phase, "status": "completed",
                "session_sha256": sf.sha256_file(session / "session.json"), **fields}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("产物来源/阶段身份不符")
    for relative, identity in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("manifest路径越界")
        sf.verify_identity(target, identity)
    return manifest


def completed(parent: Path, session: Path, phase: str, retry_failed: bool = False, **fields):
    attempts = sorted(parent.glob("attempt_*"))
    successes = [p for p in attempts if (p / "freeze_receipt.json").is_file()]
    if len(successes) > 1:
        raise RuntimeError("同一cell存在多个成功attempt")
    if successes:
        verify_attempt(successes[0], session, phase, **fields)
        return successes[0]
    if attempts and not retry_failed:
        raise RuntimeError(f"保留失败/中断现场，请检查后用--retry-failed创建新attempt: {parent}")
    return None


def cuda_environment(device: str):
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None or not torch.cuda.is_available():
        raise RuntimeError("运行要求显式可用的cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("依赖检查失败: " + "; ".join(problems))
    torch.cuda.set_device(resolved)
    props = torch.cuda.get_device_properties(resolved)
    return {"dependencies": dependencies(), "device": device, "device_name": props.name, "total_memory": props.total_memory,
            "device_uuid": str(props.uuid), "cpu_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def equivalence_cell(arm, seed, device):
    """非零FiLM下验证B0与冻结A0的严格前后向等价。"""
    source_arm = ARM_SPECS[arm].source_arm
    original = legacy.build_model(legacy.config(source_arm, seed, Path("SYNTHETIC"), device)).to(device).eval()
    reference = GeluRefinerModel(arm, seed).to(device).eval()
    initial = state_dict_identity(reference.state_dict())
    if set(reference.state_dict()) != set(original.state_dict()):
        raise RuntimeError("参照state key不等价")
    for name, value in original.state_dict().items():
        if not torch.equal(value, reference.state_dict()[name]):
            raise RuntimeError(f"公共初始化不等价: {name}")
    torch.manual_seed(seed + 921)
    with torch.no_grad():
        original.branches["w"].final_projection.weight.normal_(std=.01)
        original.branches["w"].final_projection.bias.normal_(std=.01)
    reference.load_state_dict(original.state_dict(), strict=True)
    batch = synthetic_batch(1, seed + 923, device=device)
    for amp in (False, True):
        grads, outputs = [], []
        for model in (original, reference):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                output = model(batch["x"], tf=batch["tf"])["waveform"]
                loss = output.float().square().mean()
            loss.backward()
            outputs.append(output.detach())
            grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters()})
        if not torch.equal(outputs[0], outputs[1]):
            raise RuntimeError(f"参照前向不等价: {arm}/{seed}/amp={amp}")
        for name in grads[0]:
            if not torch.equal(grads[0][name], grads[1][name]):
                raise RuntimeError(f"公共梯度不等价: {arm}/{seed}/{name}/amp={amp}")
    return {"arm": arm, "seed": seed, "fp32": "exact", "bf16": "exact", "initialization": initial}


def engineering_cell(arm: str, seed: int, batch_size: int, device: str):
    cfg = config(arm, seed, Path("SYNTHETIC"), device)
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg).to(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, partition = build_crd_optimizer(model, cfg)
    batch = synthetic_batch(batch_size, seed + 7300, device=device)
    if not ARM_SPECS[arm].condition:
        batch.pop("tf")
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    for update in range(3):
        train_crd_one_epoch(model, [batch], loss_fn, optimizer, device=torch.device(device),
                            accumulation_steps=1, update_index=update, total_updates=6400,
                            max_learning_rate=3e-4, min_learning_rate=3e-5, warmup_fraction=0.05,
                            grad_clip_norm=1.0, use_amp=True, show_progress=False, epoch=1, total_epochs=80)
        finite_tree(model.state_dict(), label="model")
        finite_tree(optimizer.state_dict(), label="optimizer")
    named = dict(model.named_parameters())
    if set(partition.decay_names) | set(partition.no_decay_names) != set(named):
        raise RuntimeError("optimizer覆盖错误")
    missing = [name for name, p in named.items() if p.grad is None or not torch.isfinite(p.grad).all()]
    if missing:
        raise RuntimeError(f"活跃参数梯度缺失/非有限: {missing}")
    # FiLM零初始化导致条件内部梯度延迟开启，第三次更新后逐模块验证。
    groups = sorted({name.rsplit(".", 1)[0] for name in named})
    inactive = [group for group in groups if not any(
        name.rsplit(".", 1)[0] == group and bool(p.grad.ne(0).any()) and not torch.equal(p.detach(), initial[name])
        for name, p in named.items())]
    if inactive:
        raise RuntimeError(f"第三步仍无梯度/更新的模块: {inactive}")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(batch["x"], tf=batch.get("tf"))
        loss, _ = loss_fn(prediction, batch["target"])
    finite_tree(prediction, label="eval_prediction")
    if prediction["waveform"].shape != (batch_size, 1, 18000) or not torch.isfinite(loss):
        raise RuntimeError("eval路径shape/loss错误")
    peak = torch.cuda.max_memory_reserved(device)
    fraction = peak / torch.cuda.get_device_properties(device).total_memory
    if fraction > 0.85:
        raise RuntimeError(f"验收显存比例超限: {fraction:.4f}")
    return {"arm": arm, "seed": seed, "batch_size": batch_size, "updates": 3,
            "parameters": sum(p.numel() for p in named.values()), "modules_with_gradient_and_update": len(groups),
            "eval_loss": float(loss), "peak_reserved_bytes": peak, "peak_reserved_fraction": fraction}


def run_equivalence(device, output):
    """仅在隔离进程启用确定性反向，避免改动正式训练的算子选择。"""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("严格GPU反向复核要求独立CUBLAS确定性环境")
    torch.use_deterministic_algorithms(True)
    records = []
    for arm in REFERENCE_ARMS:
        for seed in SEEDS:
            print(f"GPU deterministic equivalence {arm}/{seed} FP32+BF16", flush=True)
            records.append(equivalence_cell(arm, seed, device))
    write_json(output, {"passed": True, "mode": "isolated_deterministic_backward", "cells": records})


def run_gpu(session: Path, device: str, retry_failed: bool = False) -> Path:
    # 两张卡各自验收并绑定自己的环境回执，避免并发写入同一验收目录。
    parent = session / "engineering" / str(torch.device(device)).replace(":", "_")
    existing = completed(parent, session, "engineering", retry_failed)
    if existing:
        return existing
    with attempt(parent, session, "engineering") as output:
        environment = cuda_environment(device)
        payload = load_session(session)
        for ref in payload["references"].values():
            historical = ref["environment"]
            if (environment["device_name"] != historical["device_name"]
                    or environment["dependencies"] != historical["dependencies"]):
                raise RuntimeError("GPU/环境与冻结参照不匹配，禁止进入正式训练")
        write_json(output / "environment.json", environment)
        write_json(output / "access.json", {"input": "synthetic", "real_data": False, "test": False})
        if environment["deterministic_algorithms"]:
            raise RuntimeError("正式运行须保持历史默认算法设置")
        command = [sys.executable, str(ROOT / "scripts/run_apor_gelu_refiner_v1.py"), "equivalence",
                   "--device", device, "--output", str(output / "reference_equivalence.json"), "--confirm-synthetic-gpu"]
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        equivalent = read_json(output / "reference_equivalence.json")
        for record in equivalent["cells"]:
            if record["initialization"] != payload["references"][f"{record['arm']}/{record['seed']}"]["initialization"]:
                raise ValueError("GPU构造初始化与历史训练初始化不一致")
        results = []
        for arm in ARMS:
            for seed in SEEDS:
                print(f"GPU synthetic {arm}/{seed}/batch1", flush=True)
                result = engineering_cell(arm, seed, 1, device)
                write_json(output / f"{arm}_{seed}_b1.json", result)
                results.append(result)
            print(f"GPU synthetic {arm}/batch128 train+eval", flush=True)
            result = engineering_cell(arm, SEEDS[0], 128, device)
            write_json(output / f"{arm}_b128.json", result)
            results.append(result)
        write_json(output / "acceptance.json", {"passed": True, "cells": results})
    return output


def verify_gpu(path: Path, session: Path):
    verify_attempt(path, session, "engineering")
    receipt = read_json(path / "acceptance.json")
    equivalence = read_json(path / "reference_equivalence.json")
    if (equivalence.get("passed") is not True or equivalence.get("mode") != "isolated_deterministic_backward"
            or len(equivalence.get("cells", [])) != 3
            or {(r["arm"], r["seed"]) for r in equivalence["cells"]} != {(a,s) for a in REFERENCE_ARMS for s in SEEDS}
            or any(r["fp32"] != "exact" or r["bf16"] != "exact" for r in equivalence["cells"])):
        raise ValueError("参照等价性验收不完整")
    expected = {(a, s, 1) for a in ARMS for s in SEEDS} | {(a, SEEDS[0], 128) for a in ARMS}
    cells = receipt.get("cells", [])
    if (receipt.get("passed") is not True or len(cells) != len(expected)
            or {(c["arm"], c["seed"], c["batch_size"]) for c in cells} != expected
            or any(c["updates"] != 3 or c["modules_with_gradient_and_update"] <= 0
                   or not 0 < c["peak_reserved_fraction"] <= .85 or not np.isfinite(c["eval_loss"]) for c in cells)):
        raise ValueError("GPU验收矩阵/回执不完整")


def audit_sources(cfg, output: Path) -> dict[str, pd.DataFrame]:
    source = read_json(ROOT / P2_LOCK_PATH)
    index = source["dataset_index"]
    cache = source["cache_lock"]
    write_json(output / "access_started.json", {"splits": ["train", "val"], "test": False, "index": index,
                                                "cache_read": bool(cfg.model.tf_representations)})
    if Path(cfg.data.dataset_root) / cfg.data.index_csv != Path(index["path"]):
        raise ValueError("索引路径漂移")
    if sf.sha256_file(Path(index["path"])) != index["sha256"]:
        raise ValueError("索引身份漂移")
    if cfg.model.tf_representations:
        if Path(cfg.data.tf_cache_path).resolve() != (ROOT / cache["root"]).resolve():
            raise ValueError("cache路径漂移")
        for relative, identity in source["source_files"].items():
            if relative.startswith(cache["root"] + "/"):
                sf.verify_identity(ROOT / relative, identity)
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = {}
    for split in COUNTS:
        frame = filter_index(audited, cfg, split=split, max_windows=None,
                             sample_strategy=cfg.data[f"{split}_sample_strategy"], sample_seed=cfg.data[f"{split}_sample_seed"])
        if (len(frame) != COUNTS[split] or frame.samp_id.nunique() != SAMP_IDS[split]
                or frame.dataset_row_id.duplicated().any() or set(frame.split) != {split}
                or array_hash(np.sort(frame.dataset_row_id.to_numpy())) != cache["row_identity"][f"{split}_row_content_sha256"]):
            raise ValueError(f"{split}样本身份漂移")
        frame.to_csv(output / f"{split}_rows.csv", index=False)
        rows[split] = frame
    if set(rows["train"].samp_id) & set(rows["val"].samp_id):
        raise ValueError("subject隔离失败")
    write_json(output / "access_receipt.json", {"counts": COUNTS, "samp_ids": SAMP_IDS,
               "row_order": {k: array_hash(v.dataset_row_id.to_numpy()) for k, v in rows.items()},
               "sample_seeds": {k: int(cfg.data[f"{k}_sample_seed"]) for k in rows}, "source": sf.identity(ROOT / P2_LOCK_PATH)})
    return rows


class ModuleExperiment(CRDExperiment):
    task_name = "apor_gelu_refiner_v1"

    def __init__(self, cfg, rows=None, initialization_path=None, session=None):
        super().__init__(cfg)
        self.expected_rows = rows
        self.initialization_path = initialization_path
        self.session = session

    def _build_model(self):
        model = build_model(self.cfg)
        if self.initialization_path is not None:
            write_json(self.initialization_path, state_dict_identity(model.state_dict()))
        return model

    def _build_data(self):
        data = super()._build_data()
        if self.expected_rows is not None:
            for split in COUNTS:
                bundle = getattr(data, split)
                if not bundle.rows.equals(self.expected_rows[split]):
                    raise ValueError(f"实际loader与审计样本不符: {split}")
            if len(data.train.loader) != 80 or data.train.loader.batch_size != 128:
                raise ValueError("train loader预算错误")
        return data

    def _evaluate_model(self, model, loader, **kwargs):
        if kwargs.get("include_test_only", False) or kwargs.get("evaluation_split", "validation") != "validation":
            raise ValueError("GELU/refiner训练入口只允许validation评价")
        overlap = []

        def record_overlap(module, inputs, patches):
            a, b = patches.detach().float()[:, :-1, 128:], patches.detach().float()[:, 1:, :128]
            energy = .5 * (a.square().mean((1, 2)) + b.square().mean((1, 2)))
            value = (a - b).square().mean((1, 2)) / (energy + 1e-8)
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("重叠诊断非有限")
            overlap.append(torch.stack((value, energy), 1).cpu().numpy())

        handle = model.patch_head.register_forward_hook(record_overlap)
        try:
            predictions = collect_predictions(model, loader, device=self.device,
                                              max_windows=len(loader.dataset), use_amp=bool(self.cfg.training.use_amp))
        finally:
            handle.remove()
        frame = evaluate_task_predictions(predictions, self.cfg, include_test_only=False,
                                          method=self.cfg.model.apor_gelu_refiner_v1.arm)
        frame.insert(0, "evaluation_split", "validation")
        if self.expected_rows is not None:
            sf.validate_metrics(frame, self.expected_rows["val"])
        if set(frame.split) != {"val"}:
            raise ValueError("预测集合包含非validation样本")
        diagnostic = np.concatenate(overlap)
        if diagnostic.shape != (len(frame), 2):
            raise ValueError("重叠诊断与预测行数不一致")
        frame["overlap_relative_mse"] = diagnostic[:, 0]
        frame["overlap_energy"] = diagnostic[:, 1]
        frame["overlap_low_energy"] = diagnostic[:, 1] <= 1e-8
        frame.insert(0, "arm", self.cfg.model.apor_gelu_refiner_v1.arm)
        frame.insert(0, "seed", int(self.cfg.training.seed))
        frame["method"] = self.cfg.model.apor_gelu_refiner_v1.arm
        for key in ("r_tho_hat", "tho_ref"):
            if not np.isfinite(predictions[key]).all():
                raise FloatingPointError("保存的validation波形非有限")
        with (self.run_dir / "validation_prediction.npy").open("xb") as stream:
            np.save(stream, predictions["r_tho_hat"].astype(np.float32, copy=False), allow_pickle=False)
        frame[["dataset_row_id", "samp_id", "split"]].to_csv(self.run_dir / "prediction_rows.csv", index=False)
        reference_root = self.session if self.session is not None else self.run_dir
        target = predictions["tho_ref"].astype(np.float32, copy=False)
        reference = save_validation_reference(reference_root, target)
        write_json(self.run_dir / "reference_source.json", {"path": str(reference), **sf.identity(reference)})
        return frame


def validate_run(run: Path, cfg, rows: pd.DataFrame, initialization: Path):
    if OmegaConf.to_container(OmegaConf.load(run / "config.yaml"), resolve=True) != OmegaConf.to_container(cfg, resolve=True):
        raise ValueError("保存配置漂移")
    history = pd.read_csv(run / "train_history.csv")
    best = sf.validate_history(history, cfg)
    model = build_model(cfg)
    if state_dict_identity(model.state_dict()) != read_json(initialization):
        raise ValueError("初始化身份漂移")
    optimizer, partition = build_crd_optimizer(model, cfg)
    if read_json(run / "optimizer_parameter_groups.json") != {
        "weight_decay": float(cfg.training.weight_decay), "decay": list(partition.decay_names), "no_decay": list(partition.no_decay_names)
    }:
        raise ValueError("optimizer分组漂移")
    for filename, epoch in (("checkpoint_best_local_rr.pt", best), ("checkpoint_final.pt", len(history))):
        ckpt = torch.load(run / filename, map_location="cpu", weights_only=False)
        finite_tree(ckpt, label=filename)
        extra = ckpt["extra_state"]
        if (ckpt["epoch"] != epoch or ckpt["config"] != OmegaConf.to_container(cfg, resolve=True)
                or extra["protocol"] != PROTOCOL or extra["update_index"] != epoch * UPDATES_PER_EPOCH or extra["total_updates"] != PLANNED_UPDATES):
            raise ValueError("checkpoint来源/更新数漂移")
        record = history.loc[history.epoch.eq(epoch)].iloc[0]
        for key, value in record.items():
            if key not in ckpt["metrics"] or not np.isclose(float(value), float(ckpt["metrics"][key]), rtol=1e-12, atol=1e-12):
                raise ValueError(f"checkpoint/history不一致: {key}")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for p in model.parameters():
            state = optimizer.state[p]
            if (not {"step", "exp_avg", "exp_avg_sq"}.issubset(state) or float(state["step"]) != epoch * UPDATES_PER_EPOCH
                    or state["exp_avg"].shape != p.shape or state["exp_avg_sq"].shape != p.shape):
                raise ValueError("optimizer state不完整")
    frame = pd.read_csv(run / "metrics.csv")
    degenerate = sf.validate_metrics(frame, rows)
    if not frame.arm.eq(cfg.model.apor_gelu_refiner_v1.arm).all() or not frame.seed.eq(cfg.training.seed).all():
        raise ValueError("metrics arm/seed漂移")
    saved = pd.read_csv(run / "metrics_summary.csv")
    expected = summarize_task_metrics(frame)
    for metric in sf.PRIMARY:
        if len(saved) != 1 or not np.isclose(saved.iloc[0][metric + "_mean"], expected.iloc[0][metric + "_mean"], rtol=0, atol=1e-12):
            raise ValueError("指标汇总漂移")
        if saved.iloc[0][metric + "_n"] != expected.iloc[0][metric + "_n"]:
            raise ValueError("指标分母漂移")
    return {"best_epoch": best, "completed_epochs": len(history), "validation_rows": len(frame), "degeneracy": degenerate}


def run_formal(session: Path, arm: str, seed: int, device: str, acceptance: Path, retry_failed=False):
    if arm not in TRAIN_ARMS or seed not in SEEDS:
        raise ValueError("formal只允许九个新增cell；参照禁止重训")
    load_session(session)
    verify_gpu(acceptance, session)
    runtime = cuda_environment(device)
    accepted = read_json(acceptance / "environment.json")
    if runtime["dependencies"] != accepted["dependencies"] or runtime["device_name"] != accepted["device_name"] or runtime["total_memory"] < accepted["total_memory"]:
        raise ValueError("formal环境与GPU验收不兼容")
    parent = session / "formal" / arm / f"seed_{seed}"
    prior = completed(parent, session, "formal", retry_failed, arm=arm, seed=seed)
    if prior:
        print(f"复用已完成 {arm}/{seed}: {prior}", flush=True)
        return prior
    with attempt(parent, session, "formal", arm=arm, seed=seed) as output:
        write_json(output / "environment.json", runtime)
        write_json(output / "engineering_source.json", {"path": str(acceptance), **sf.identity(acceptance / "manifest.json")})
        cfg = config(arm, seed, output / "training", device)
        rows = audit_sources(cfg, output)
        experiment = ModuleExperiment(cfg, rows, output / "initialization.json", session=session)
        print(f"开始 formal {arm}/{seed}: {output}", flush=True)
        run = experiment.train()
        receipt = validate_run(run, cfg, rows["val"], output / "initialization.json")
        write_json(output / "result.json", {**receipt, "run_dir": str(run.relative_to(output)), "arm": arm, "seed": seed})
    return output


def status(session: Path):
    load_session(session)
    cells = []
    for cell in plan():
        parent = session / "formal" / cell["arm"] / f"seed_{cell['seed']}"
        paths = sorted(parent.glob("attempt_*"))
        success = [p for p in paths if (p / "freeze_receipt.json").is_file()]
        if len(success) > 1:
            raise RuntimeError("同cell多个成功attempt")
        state = "completed" if success else "failed" if paths and all((p / "failed.json").is_file() for p in paths) else "running_or_interrupted" if paths else "pending"
        cells.append({**cell, "status": state, "attempts": [str(p) for p in paths]})
    return {"cells": cells, "reused_reference_cells": 3,
            "counts": {s: sum(c["status"] == s for c in cells) for s in ("completed", "failed", "running_or_interrupted", "pending")}}


def summary_tables(window_metrics: pd.DataFrame):
    expected = {(a, s) for a in ARMS for s in SEEDS}
    observed = set(window_metrics[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("汇总要求完整12-cell科学矩阵")
    reference_rows = None
    seed_rows = []
    for (arm, seed), frame in window_metrics.groupby(["arm", "seed"], sort=False):
        identity = frame[["dataset_row_id", "samp_id", "split"]].reset_index(drop=True)
        sf.validate_metrics(frame.reset_index(drop=True), identity)
        if len(frame) != COUNTS["val"] or set(frame.split) != {"val"}:
            raise ValueError("validation分母/split错误")
        target_flags = sorted(set(sf.ELIGIBILITY.values()) | {c for c in frame if c.endswith("_target_eligible")})
        target_identity = frame[["dataset_row_id", "samp_id", "split", *target_flags]].reset_index(drop=True)
        if reference_rows is None:
            reference_rows = target_identity
        elif not target_identity.equals(reference_rows):
            raise ValueError("跨cell样本或target资格不一致")
        for metric in sf.PRIMARY:
            mask = sf._metric_mask(frame, metric)
            values = frame.loc[mask, metric].to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError("主指标缺失/非有限")
            seed_rows.append({"arm": arm, "seed": seed, "metric": metric, "mean": values.mean(), "n": len(values)})
    per_seed = pd.DataFrame(seed_rows)
    across = per_seed.groupby(["arm", "metric"], sort=False)["mean"].agg(seed_mean="mean", seed_sd="std").reset_index()
    indexed = per_seed.set_index(["arm", "seed", "metric"])
    deltas = []
    for arm, reference in COMPARISONS:
        for seed in SEEDS:
            for metric in sf.PRIMARY:
                value = indexed.loc[(arm, seed, metric), "mean"]
                base = indexed.loc[(reference, seed, metric), "mean"]
                deltas.append({"arm": arm, "reference": reference, "seed": seed, "metric": metric,
                               "delta": value - base, "improvement": value - base if metric == sf.PCC else base - value})
    paired = pd.DataFrame(deltas)
    paired_summary = paired.groupby(["arm", "reference", "metric"], sort=False)["improvement"].agg(
        mean="mean", seed_sd="std", improved_seeds=lambda x: int((x > 0).sum())).reset_index()
    subjects = sf.subject_stratified_metrics(window_metrics)
    macro = subjects.groupby(["arm", "seed", "metric"], sort=False)["mean"].mean().reset_index()
    return {"per_seed": per_seed, "across_seed": across, "paired_delta": paired, "paired_summary": paired_summary,
            "per_subject": subjects, "subject_macro": macro,
            "local_rr_tail": sf.local_rr_tail_summary(window_metrics), "denominators": sf.metric_denominators(window_metrics)}


def validation_decision(tables):
    selected = ["B0"]
    checks = []
    for arm, ref in COMPARISONS:
        records, protected = [], True
        for scope in ("per_seed", "subject_macro"):
            data = tables[scope].set_index(["arm", "seed", "metric"])["mean"]
            for metric in sf.PRIMARY:
                values = np.array([data.loc[arm, s, metric] for s in SEEDS])
                base = np.array([data.loc[ref, s, metric] for s in SEEDS])
                a, b = float(values.mean()), float(base.mean())
                good = int(((values > base) if metric == sf.PCC else (values < base)).sum())
                change = a - b if metric == sf.PCC else (a / b - 1 if b else None)
                passed = a >= b - .002 if metric == sf.PCC else a <= b * 1.005
                protected = protected and passed
                records.append({"scope": scope, "metric": metric, "change": change,
                                "improved_seeds": good, "protected": bool(passed)})
        passed = protected
        checks.append({"arm": arm, "reference": ref, "eligible": bool(passed), "metrics": records})
        if passed:
            selected.append(arm)

    return {"protocol": PROTOCOL, "selection_split": "full_validation", "reference": "B0",
            "eligible_arms": selected, "checks": checks, "test_used": False,
            "comparison": "three_independent_single_factor_contrasts", "replaces_main_model": False,
            "selection_rule": "各候选独立对B0检查保护线，不合并因素或强选单一赢家"}


def summarize(session: Path, retry_failed=False):
    payload = load_session(session)
    prior = completed(session / "summary", session, "summary", retry_failed)
    if prior:
        return prior
    frames, sources = [], []
    for ref in payload["references"].values():
        frame = pd.read_csv(ref["metrics"]["path"])
        if not frame.arm.eq(ref["source_arm"]).all() or not frame.seed.eq(ref["seed"]).all():
            raise ValueError("历史参照指标标签错误")
        frame = frame.copy()
        frame["arm"] = ref["arm"]
        frame["method"] = ref["arm"]
        frames.append(frame)
        sources.append({"arm": ref["arm"], "seed": ref["seed"], "reused": True,
                        "path": ref["attempt"], **ref["manifest"]})
    for cell in plan():
        parent = session / "formal" / cell["arm"] / f"seed_{cell['seed']}"
        path = completed(parent, session, "formal", arm=cell["arm"], seed=cell["seed"])
        if path is None:
            raise RuntimeError("九个新增cell未全部完成")
        result = read_json(path / "result.json")
        frame = pd.read_csv(path / result["run_dir"] / "metrics.csv")
        if not frame.arm.eq(cell["arm"]).all() or not frame.seed.eq(cell["seed"]).all():
            raise ValueError("cell指标标签漂移")
        frames.append(frame)
        sources.append({"path": str(path), **sf.identity(path / "manifest.json")})
    tables = summary_tables(pd.concat(frames, ignore_index=True))
    with attempt(session / "summary", session, "summary") as output:
        write_json(output / "sources.json", sources)
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=False)
        decision = validation_decision(tables)
        decision["fixed_checkpoints"] = []
        for arm in ARMS:
            for seed in SEEDS:
                if arm in REFERENCE_ARMS:
                    identity = payload["references"][f"{arm}/{seed}"]["checkpoint"]
                else:
                    path = completed(session / "formal" / arm / f"seed_{seed}", session, "formal", arm=arm, seed=seed)
                    checkpoint = path / read_json(path / "result.json")["run_dir"] / "checkpoint_best_local_rr.pt"
                    identity = {"path": str(checkpoint), **sf.identity(checkpoint)}
                decision["fixed_checkpoints"].append({"arm": arm, "seed": seed, **identity})
        write_json(output / "validation_decision.json", decision)
    return output


def execute(session: Path, device: str, *, phase: str, arm=None, seed=None, retry_failed=False):
    session = session.resolve()
    load_session(session)
    with session_mutex(session):
        if phase == "summarize":
            return summarize(session)
        acceptance = run_gpu(session, device, retry_failed)
        if phase == "gpu-acceptance":
            return acceptance
        cells = plan() if phase == "run-all" else [{"arm": arm, "seed": seed}]
        for cell in cells:
            run_formal(session, cell["arm"], cell["seed"], device, acceptance, retry_failed)
        return summarize(session, retry_failed) if phase == "run-all" else status(session)


def execute_shard(session: Path, device: str, index: int, count: int, retry_failed=False):
    session = session.resolve()
    load_session(session)
    assignment = read_json(session / "parallel_plan.json")
    expected = parallel_layout([worker["device"] for worker in assignment["workers"]])
    if (assignment != {**expected, "session_sha256": sf.sha256_file(session / "session.json")}
            or count != assignment["shard_count"] or not 0 <= index < count
            or str(torch.device(device)) != assignment["workers"][index]["device"]):
        raise ValueError("worker分片/设备与固定调度表不一致")
    # 共享会话锁允许不同worker并行；同片、同设备独占锁禁止重复运行。
    with session_mutex(session, shared=True), file_mutex(session / f".shard_{index}.lock"), \
            file_mutex(session / f".device_{str(torch.device(device)).replace(':', '_')}.lock"):
        acceptance = run_gpu(session, device, retry_failed)
        for worker in assignment["workers"]:
            parent = session / "engineering" / worker["device"].replace(":", "_")
            other = completed(parent, session, "engineering")
            if other is None:
                raise RuntimeError("双卡验收未全部完成，禁止读取真实数据训练")
            verify_gpu(other, session)
        for cell in shard_plan(index, count):
            run_formal(session, cell["arm"], cell["seed"], device, acceptance, retry_failed)
        return {"shard_index": index, "completed_cells": len(shard_plan(index, count))}


def _stop_workers(processes):
    """只停止本次调度创建的进程组，先SIGINT让训练器保存失败现场。"""
    def send(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            # poll与发信号之间进程已退出，仍由下面wait回收退出状态。
            pass

    for process in processes:
        if process.poll() is None:
            send(process, signal.SIGINT)
    for process in processes:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            send(process, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                send(process, signal.SIGKILL)
                process.wait()


def run_parallel(session: Path, devices, retry_failed=False):
    """两个独立CUDA进程共享一份矩阵，全部成功后由父进程统一汇总。"""
    session = session.resolve()
    load_session(session)
    layout = parallel_layout(devices)
    with file_mutex(session / ".parallel_controller.lock"):
        with session_mutex(session):
            environments = [cuda_environment(worker["device"]) for worker in layout["workers"]]
            if len({env["device_uuid"] for env in environments}) != len(environments):
                raise ValueError("设备列表映射到了相同物理GPU")
            assignment = {**layout, "session_sha256": sf.sha256_file(session / "session.json")}
            plan_path = session / "parallel_plan.json"
            if plan_path.exists():
                if read_json(plan_path) != assignment:
                    raise ValueError("已有并行分片定义不可更改")
            else:
                write_json(plan_path, assignment)
            # 两卡全部验收成功后才允许任何真实数据训练，避免一边失败另一边已经训练。
            for worker in layout["workers"]:
                accepted = run_gpu(session, worker["device"], retry_failed)
                verify_gpu(accepted, session)
                torch.cuda.set_device(worker["device"])
                torch.cuda.empty_cache()
        with attempt(session / "dispatch", session, "dispatch") as output, ExitStack() as stack:
            processes = []
            try:
                for worker in layout["workers"]:
                    command = [sys.executable, str(ROOT / "scripts/run_apor_gelu_refiner_v1.py"), "run-shard",
                               "--session", str(session), "--device", worker["device"],
                               "--shard-index", str(worker["index"]), "--shard-count", str(layout["shard_count"]),
                               "--confirm-training"]
                    if retry_failed:
                        command.append("--retry-failed")
                    log_path = output / f"worker_{worker['index']}.log"
                    log = stack.enter_context(log_path.open("x"))
                    env = os.environ.copy()
                    # 限制两个进程的BLAS/OpenMP线程，避免CPU validation互相抢占。
                    env.update({key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
                    env["PYTHONUNBUFFERED"] = "1"
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    processes.append(process)
                    write_json(output / f"worker_{worker['index']}.json", {**worker, "pid": process.pid,
                               "command": command, "log": str(log_path), "cpu_threads": 4})
                    print(f"GPU {worker['device']}: {len(worker['cells'])} cells, PID={process.pid}, LOG={log_path}", flush=True)
                while any(process.poll() is None for process in processes):
                    if any(process.poll() not in (None, 0) for process in processes):
                        raise RuntimeError(f"并行worker失败，请检查日志: {output}")
                    time.sleep(5)
                if any(process.returncode != 0 for process in processes):
                    raise RuntimeError(f"并行worker失败，请检查日志: {output}")
            except BaseException:
                _stop_workers(processes)
                raise
            with session_mutex(session):
                result = summarize(session, retry_failed)
            write_json(output / "result.json", {"summary": str(result), "worker_exit_codes": [p.returncode for p in processes]})
    return result
