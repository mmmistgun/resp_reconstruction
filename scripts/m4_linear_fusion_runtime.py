"""M4 linear fusion：三 seed 单投影训练与冻结双投影基线的 validation 比较。"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train import patch_aligned_tf_training as legacy
from resp_train.crd.config import load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.tf_v1_features import cwt_magnitude_features
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch
from resp_train.data.factory import build_tho_data
from resp_train.engine import validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from scripts import m4_cross_time_bias_runtime as base
from scripts.m4_linear_fusion_model import ARMS, REFERENCES, TRAIN_ARMS, LinearFusionModel, architecture, band_indices, build_model, contract, frequency_metadata
from scripts.m4_components_runtime import verified_receipt, artifact_path, delta_summary

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "m4-linear-fusion-v1-20261019"
SPEC = ROOT / "configs/m4_linear_fusion_v1/experiment.yaml"
PROTOCOL_PATH = ROOT / "docs/experiments/m4_linear_fusion_v1_protocol_20261019.md"
SEEDS = (20260811, 20260812, 20260813)
COMPARISONS = {"L1": "L0"}
REFERENCE_SESSION = ROOT / "runs/m4_cross_time_bias_v1/formal_r2_20261018"
REFERENCE_SHA = "96975f061d9e1e0e51cbd75d8f0154f383a07389535d6423c0b9853d68c5ee43"
REFERENCE_SUMMARY = REFERENCE_SESSION / "summary/10d3d10d991d4a18a57386c755e4d3fe"
REFERENCE_SUMMARY_SHA = "e3f6206ef63157aaf9ebde106d6c9ffb10f5e5d2c350107e047866c44e1f0458"
sha, read_json, write_json, mutex = legacy.sha, legacy.read_json, legacy.write_json, legacy.mutex


def spec():
    value = OmegaConf.to_container(OmegaConf.load(SPEC), resolve=True)
    expected = {"protocol": PROTOCOL, "arms": list(ARMS), "seeds": list(SEEDS), "patch_seconds": 1,
        "normalization": "groupnorm_representation_variant", "baseline": "configs/crd_tf_v1/crd_tf102_w_formal.yaml",
        "training": {"epochs": 80, "batch_size": 32, "gradient_accumulation_steps": 4,
            "early_stopping_min_epoch": 30, "early_stopping_patience": 15, "early_stopping_min_delta": 0., "amp_dtype": "bfloat16"},
        "data": {"train_windows": 10141, "val_windows": 2675, "train_subjects": 32, "val_subjects": 7},
        "selector": "full_validation_local_rr_strict_minimum_earliest_tie", "comparisons": COMPARISONS,
        "references": REFERENCES, "reference_session_sha256": REFERENCE_SHA,
        "mean_delta_threshold": .01,
        "threshold_rule": "absolute_mean_delta_lt_threshold_is_no_obvious_mean_difference",
        "selection_aggregation": "window_weighted", "diagnostic_aggregation": "subject_macro"}
    if value != expected:
        raise ValueError("linear fusion 矩阵/预算/比较合同不符")
    return value


def plan(shard_index=0, shard_count=1, *, include_references=False):
    arms = ARMS if include_references else TRAIN_ARMS
    if not 1 <= shard_count <= len(arms) * len(SEEDS) or not 0 <= shard_index < shard_count:
        raise ValueError("非法分片")
    return [{"arm": a, "name": ARMS[a].name, "seed": s, "cell": f"{a}_seed{s}",
             "role": "reference" if a in REFERENCES else "train"}
            for s in SEEDS for a in arms][shard_index::shard_count]


def config(arm, seed, output, device="cpu"):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("未知 linear fusion cell")
    value = spec()
    cfg = load_crd_config(ROOT / value["baseline"], overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"])
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "m4_linear_fusion_v1"
    cfg.protocol.run_role = "formal"
    cfg.protocol.execution_gate = "m4_linear_fusion_synthetic_task_loss_acceptance"
    cfg.model.name = "m4_linear_fusion_v1"
    cfg.model.variant = ARMS[arm].name
    cfg.model.m4_linear_fusion = contract(arm)
    cfg.model.architecture = asdict(architecture())
    cfg.model.tf_representations = [] if ARMS[arm].band == "none" else ["w"]
    if ARMS[arm].band == "none":
        cfg.data.tf_cache_path = None
    cfg.data.preload_windows = False
    for key, v in value["training"].items():
        cfg.training[key] = v
    cfg.training.device = str(device)
    cfg.training.show_progress = False
    cfg.training.early_stopping_enabled = True
    cfg.outputs.run_root = str(Path(output).resolve())
    return cfg


def code_identity():
    # 前序冻结源码作共享依赖，新实验只在独立目录与 scripts 中扩展。
    result = base.code_identity()
    files = [Path(__file__), ROOT / "scripts/m4_linear_fusion_model.py", ROOT / "scripts/run_m4_linear_fusion.py",
             ROOT / "scripts/run_m4_linear_fusion_parallel.sh", SPEC, PROTOCOL_PATH,
             ROOT / "tests/test_m4_linear_fusion.py", ROOT / "tests/test_m4_components.py",
             ROOT / "scripts/m4_components_model.py", ROOT / "scripts/m4_components_runtime.py"]
    for path in files:
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


def state_hash(state):
    """初始化摘要忽略另由冻结 cache 校验的频率坐标，覆盖所有可学习参数。"""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name == "condition.frequencies_hz":
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def audit_initialization(*, cfg=None, mamba_factory=None):
    """核对基线完整状态，候选仅合并输出投影；其他参数逐张量配对。"""
    from resp_train.crd.tf_v1_features import morlet_scales_and_frequencies
    from scripts.m4_cross_time_bias_model import CrossTimeBiasModel
    _, frequencies = morlet_scales_and_frequencies(12, 97)
    frequencies = np.sort(frequencies, kind="stable")
    records = {}
    for seed in SEEDS:
        original = CrossTimeBiasModel("A1", seed, frequencies, cfg=cfg, mamba_factory=mamba_factory)
        full = LinearFusionModel("L0", seed, frequencies, cfg=cfg, mamba_factory=mamba_factory)
        removed = LinearFusionModel("L1", seed, frequencies, cfg=cfg, mamba_factory=mamba_factory)
        old, left, right = original.state_dict(), full.state_dict(), removed.state_dict()
        if old.keys() != left.keys() or any(not torch.equal(old[k], left[k]) for k in old):
            raise ValueError("L0 与冻结简化基线初始化不同")
        expected_removed = {"fusion.projection.weight", "fusion.projection.bias"}
        changed = {"condition.output.weight", "condition.output.bias"}
        if (not expected_removed or set(left) - set(right) != expected_removed
                or set(right) - set(left)
                or any(not torch.equal(v, left[k]) for k, v in right.items() if k not in changed)
                or any(torch.count_nonzero(right[k]) != 0 for k in changed)):
            raise ValueError("单投影消融改变了约定以外的初始化或结构")
        records[str(seed)] = {arm: {"parameters": sum(p.numel() for p in net.parameters()),
            "state_sha256": state_hash(net.state_dict()),
            "time_bias_parameters": sum(p.numel() for p in net.condition.time_bias.parameters()),
            "frequency_bias_parameters": sum(p.numel() for p in net.condition.frequency_bias.parameters()),
            "fusion_projection_parameters": sum(p.numel() for p in net.fusion.parameters()),
            "attention_output_zero": all(torch.count_nonzero(p) == 0 for p in net.condition.output.parameters())}
            for arm, net in (("L0", full), ("L1", removed))}
        records[str(seed)].update(common_exact_except_output=True, reference_exact=True,
            removed_keys=sorted(expected_removed), zero_initialized_keys=sorted(changed),
            mamba="official_mamba2" if mamba_factory is None else "fixture")
    return records


def audit_references():
    """核验三个冻结双投影简化基线的来源、初始化与科学配置。"""
    session = REFERENCE_SESSION.resolve()
    if sha(session / "session.json") != REFERENCE_SHA:
        raise ValueError("简化基线 session 身份变化")
    source = base.load_session(session)
    if sha(REFERENCE_SUMMARY / "receipt.json") != REFERENCE_SUMMARY_SHA:
        raise ValueError("简化基线 validation 汇总身份变化")
    summary = read_json(REFERENCE_SUMMARY / "receipt.json")
    for name, digest in summary["artifacts"].items():
        if sha(artifact_path(REFERENCE_SUMMARY, name)) != digest:
            raise ValueError("简化基线 validation 汇总产物变化")
    cells = {}
    for arm, old_arm in REFERENCES.items():
        for seed in SEEDS:
            directory = session / "cells" / f"{old_arm}_seed{seed}"
            receipt = verified_receipt(directory, old_arm, seed)
            previous = summary["sources"][f"{old_arm}_seed{seed}"]
            if previous["completed_sha256"] != sha(directory / "completed.json") or Path(previous["path"]).resolve() != directory.resolve():
                raise ValueError("简化基线 cell 与冻结 validation 来源不符")
            saved = OmegaConf.load(directory / "config.yaml")
            expected = base.config(old_arm, seed, saved.outputs.run_root, saved.training.device)
            if Path(saved.outputs.run_root).resolve() != directory.resolve():
                raise ValueError("参照输出路径未解析到原 cell")
            expected.outputs.run_root = saved.outputs.run_root
            if legacy.config_identity(saved) != legacy.config_identity(expected):
                raise ValueError("简化基线科学配置变化")
            current = config(arm, seed, saved.outputs.run_root, saved.training.device)
            for section in ("data", "window", "loss", "evaluation", "training"):
                if current[section] != saved[section]:
                    raise ValueError(f"新旧科学合同不一致: {section}")
            if current.model.architecture != saved.model.architecture:
                raise ValueError("新旧模型几何不一致")
            if receipt["parameters"] != 1056257:
                raise ValueError("简化基线参数量变化")
            environment = directory / "environment.json"
            initial_path = artifact_path(directory, receipt["metrics"]).parent / "initialization.json"
            if str(initial_path.relative_to(directory)) not in receipt["artifacts"]:
                raise ValueError("简化基线缺少初始化身份")
            initial_hash = source["paired_initialization"][str(seed)][old_arm]["state_sha256"]
            if read_json(initial_path)["state_sha256"] != initial_hash:
                raise ValueError("简化基线初始化记录不同")
            cells[f"{arm}_seed{seed}"] = {"path": str(directory), "source_arm": old_arm,
                "completed_sha256": sha(directory / "completed.json"),
                "initialization_sha256": initial_hash,
                "environment_sha256": sha(environment), "parameters": receipt["parameters"]}
    return {"session": str(session), "session_sha256": REFERENCE_SHA,
        "validation_summary": {"path": str(REFERENCE_SUMMARY), "receipt_sha256": REFERENCE_SUMMARY_SHA}, "cells": cells}


def prepare(session):
    session = Path(session).resolve()
    session.mkdir(parents=True, exist_ok=True)
    with mutex(session / ".prepare.lock"):
        if (session / "session.json").exists():
            load_session(session)
            return session
        if set(p.name for p in session.iterdir()) - {".prepare.lock"}:
            raise FileExistsError("session 路径已有其他产物")
        references = audit_references()
        initialization = audit_initialization()
        for seed in SEEDS:
            if initialization[str(seed)]["L0"]["state_sha256"] != references["cells"][f"L0_seed{seed}"]["initialization_sha256"]:
                raise ValueError("本轮 L0 构造与冻结简化基线的初始化不同")
        resolved = session / "resolved_configs"
        resolved.mkdir()
        configs = {}
        for cell in plan():
            path = resolved / f"{cell['cell']}.yaml"
            with path.open("x") as stream:
                OmegaConf.save(config(cell["arm"], cell["seed"], session / "cells" / cell["cell"], "cuda:0"), stream)
            configs[str(path.relative_to(session))] = sha(path)
        sources = code_identity()
        with tarfile.open(session / "source_snapshot.tar.gz", "x:gz") as archive:
            for name in sources:
                archive.add(ROOT / name, arcname=name)
        if sources != code_identity():
            raise RuntimeError("保存快照期间源码变化")
        write_json(session / "session.json", {"protocol": PROTOCOL, "spec": spec(), "cells": plan(include_references=True),
            "references": references,
            "paired_initialization": initialization, "resolved_configs": configs,
            "data_source": {"path": str(legacy.P2_LOCK_PATH), "sha256": legacy.P2_LOCK_SHA256},
            "arms": {a: contract(a) for a in ARMS}, "architecture": asdict(architecture()),
            "code_sha256": sources, "snapshot_sha256": sha(session / "source_snapshot.tar.gz"),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
            "data_array_read": False, "splits": ["train", "val"], "research_test": False})
    return session


def load_session(session):
    session = Path(session)
    value = read_json(session / "session.json")
    if (value["protocol"] != PROTOCOL or value["spec"] != spec() or value["cells"] != plan(include_references=True)
            or value["arms"] != {a: contract(a) for a in ARMS}
            or value["architecture"] != json.loads(json.dumps(asdict(architecture())))
            or value["references"] != audit_references()
            or value["code_sha256"] != code_identity()
            or value["snapshot_sha256"] != sha(session / "source_snapshot.tar.gz")):
        raise ValueError("session 合同或来源身份变化")
    for name, digest in value["resolved_configs"].items():
        if sha(artifact_path(session, name)) != digest:
            raise ValueError("resolved config 身份变化")
    if (value["data_source"] != {"path": str(legacy.P2_LOCK_PATH), "sha256": legacy.P2_LOCK_SHA256}
            or sha(ROOT / legacy.P2_LOCK_PATH) != legacy.P2_LOCK_SHA256):
        raise ValueError("冻结数据来源变化")
    return value


class BandDataset(Dataset):
    """读取既有 W cache 后，在数据层固定选择频带。"""
    def __init__(self, base, indices):
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        if set(item.get("tf", {})) != {"w"}:
            raise ValueError("条件数据要求冻结 W cache")
        full = item["tf"]["w"]
        if full.shape != (97, 360) or full.dtype != torch.float32:
            raise ValueError("W cache shape/dtype 不符")
        if not torch.isfinite(full).all():
            raise FloatingPointError("W cache 包含非有限值")
        item["tf"] = {"w": full.index_select(0, self.indices)}
        return item


def condition_data(data, cfg):
    arm = cfg.model.m4_linear_fusion.arm
    if ARMS[arm].band == "none":
        return data
    frequencies = frequency_metadata(cfg.data.tf_cache_path)
    indices = band_indices(frequencies, ARMS[arm].band)
    bundles = {}
    for split in ("train", "val"):
        bundle = getattr(data, split)
        dataset = BandDataset(bundle.dataset, indices)
        loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), shuffle=split == "train",
            num_workers=0, pin_memory=str(cfg.training.device).startswith("cuda"), drop_last=False)
        bundles[split] = replace(bundle, dataset=dataset, loader=loader)
    return replace(data, **bundles)


def build_data(cfg, output):
    if sha(ROOT / legacy.P2_LOCK_PATH) != legacy.P2_LOCK_SHA256:
        raise ValueError("数据来源记录身份变化")
    source = read_json(ROOT / legacy.P2_LOCK_PATH)
    index = Path(cfg.data.dataset_root) / cfg.data.index_csv
    if index.resolve() != Path(source["dataset_index"]["path"]).resolve() or sha(index) != source["dataset_index"]["sha256"]:
        raise ValueError("dataset index 身份变化")
    write_json(output / "access_started.json", {"splits": ["train", "val"], "test": False,
        "cache_read": ARMS[cfg.model.m4_linear_fusion.arm].band != "none", "index": str(index)})
    data = condition_data(build_tho_data(cfg), cfg)
    counts, identities = spec()["data"], {}
    for split in ("train", "val"):
        rows = getattr(data, split).rows
        ids = rows.dataset_row_id.to_numpy(np.int64)
        row_hash = hashlib.sha256(np.sort(ids).tobytes()).hexdigest()
        if (len(rows) != counts[f"{split}_windows"] or rows.samp_id.nunique() != counts[f"{split}_subjects"]
                or set(rows.split) != {split} or len(set(ids)) != len(ids)
                or row_hash != source["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]):
            raise ValueError(f"{split} 样本身份/分母变化")
        rows.to_csv(output / f"{split}_rows.csv", index=False)
        identities[split] = row_hash
    if set(data.train.rows.samp_id) & set(data.val.rows.samp_id):
        raise ValueError("train/val 受试者重叠")
    if optimizer_updates_per_epoch(len(data.train.loader), int(cfg.training.gradient_accumulation_steps)) != 80:
        raise ValueError("每 epoch optimizer 更新数必须为 80")
    write_json(output / "data_receipt.json", {"row_hashes": identities,
        "source_sha256": legacy.P2_LOCK_SHA256, "index_sha256": sha(index)})
    return data


def smoke(session, arm, device, *, mamba_factory=None, batch_size=32):
    load_session(session)
    output = Path(session) / "engineering" / f"{arm}_{str(device).replace(':', '_')}_{uuid4().hex[:12]}"
    output.mkdir(parents=True)
    cfg = config(arm, SEEDS[0], output, device)
    legacy.set_seed(SEEDS[0])
    t = np.arange(18000, dtype=np.float32) / 100
    signal = ((1 + .2 * np.sin(2 * np.pi * .25 * t)) * np.sin(2 * np.pi * 2 * t)).astype(np.float32)
    batch = {"x": torch.from_numpy(signal)[None, None].repeat(batch_size, 1, 1),
        "target": torch.from_numpy(np.sin(2 * np.pi * .25 * t).astype(np.float32))[None, None].repeat(batch_size, 1, 1)}
    frequencies = None
    if ARMS[arm].band != "none":
        w, frequencies = cwt_magnitude_features(signal)
        selected = w[band_indices(frequencies, ARMS[arm].band)]
        batch["tf"] = {"w": torch.from_numpy(selected)[None].repeat(batch_size, 1, 1)}
    model = LinearFusionModel(arm, SEEDS[0], frequencies, cfg=architecture(), mamba_factory=mamba_factory).to(device)
    loss = RespirationTaskLoss(cfg).to(device)
    optimizer, _ = build_crd_optimizer(model, cfg)
    watch = model.waveform_encoder.project.weight if model.condition is None else model.condition.encoder[1].weight
    before = watch.detach().clone()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    training_summary, updates = legacy._train_epoch(model, [batch] * 8, loss, optimizer, cfg, 0, 6400, 1)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    legacy.finite_tree(training_summary)
    legacy.finite_tree(model.state_dict())
    legacy.finite_tree(optimizer.state_dict())
    if updates != 2 or watch.grad is None or not torch.isfinite(watch.grad).all() or watch.grad.abs().sum() == 0 or torch.equal(watch, before):
        raise RuntimeError("正式 loss 两步累积后分支梯度或参数更新无效")
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"模块梯度缺失或非有限: {name}")
    val_summary = validate(model, [batch], loss, device=device, use_amp=True, show_progress=False)
    legacy.finite_tree(val_summary)
    write_json(output / "receipt.json", {"passed": True, "arm": arm, "contract": contract(arm),
        "batch_size": batch_size, "accumulation": 4, "updates": 2,
        "mamba": "official_mamba2" if ARMS[arm].mamba and mamba_factory is None else "fixture" if ARMS[arm].mamba else "identity",
        "environment": legacy.runtime(device), "parameters": sum(p.numel() for p in model.parameters()),
        "train_elapsed_seconds": elapsed, "timing_scope": "two_update_engineering_check_including_initial_kernel_cost",
        "training": training_summary, "validation": val_summary,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if torch.device(device).type == "cuda" else None})
    write_json(output / "completed.json", {"receipt_sha256": sha(output / "receipt.json"),
        "session_sha256": sha(Path(session) / "session.json")})
    return output


def ensure_smoke(session, arm, device):
    """复用同源码、同设备环境且满足正式批量的已完成验收。"""
    load_session(session)
    expected_mamba = "official_mamba2" if ARMS[arm].mamba else "identity"
    environment = legacy.runtime(device)
    for marker in sorted((Path(session) / "engineering").glob(f"{arm}_{str(device).replace(':', '_')}_*/completed.json")):
        completed = read_json(marker)
        receipt_path = marker.parent / "receipt.json"
        if completed["receipt_sha256"] != sha(receipt_path) or completed["session_sha256"] != sha(Path(session) / "session.json"):
            raise ValueError("GPU 验收身份变化")
        receipt = read_json(receipt_path)
        if (receipt["passed"] is True and receipt["arm"] == arm and receipt["contract"] == contract(arm)
                and receipt["batch_size"] == 32 and receipt["accumulation"] == 4 and receipt["updates"] == 2
                and receipt["mamba"] == expected_mamba and receipt["environment"] == environment):
            return marker.parent
    return smoke(session, arm, device)


def register_training_environment(session, environment):
    """核对冻结参照的软件/GPU 型号，短临界区保证多分片登记一致。"""
    session = Path(session)
    references = read_json(session / "session.json")["references"]["cells"]
    for reference in references.values():
        previous = read_json(Path(reference["path"]) / "environment.json")
        if any(previous[k] != environment[k] for k in ("torch", "cuda", "packages", "gpu")):
            raise ValueError("当前训练环境与双投影简化基线不同")
    with (session / ".environment.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            path = session / "training_environment.json"
            if path.exists():
                previous = read_json(path)
                if any(previous[k] != environment[k] for k in ("torch", "cuda", "packages", "gpu")):
                    raise ValueError("配对 cell 的训练软件或 GPU 型号不同")
            else:
                write_json(path, environment)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_cell(session, arm, seed, device, *, resume=False):
    if arm not in TRAIN_ARMS:
        raise ValueError("L0 为冻结参照，不进入训练队列")
    session = Path(session).resolve()
    load_session(session)
    if seed not in SEEDS or torch.device(device).type != "cuda":
        raise ValueError("正式运行要求合法 cell 与 CUDA 设备")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU 不支持 BF16")
    cell = session / "cells" / f"{arm}_seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    with mutex(cell / ".run.lock"):
        if (cell / "completed.json").exists():
            result = read_json(cell / "completed.json")
            for name, digest in result["artifacts"].items():
                if sha(cell / name) != digest:
                    raise ValueError("完成产物身份变化")
            print(f"复用已完成 {cell.name}", flush=True)
            return result
        if (cell / "config.yaml").exists() and not resume:
            raise FileExistsError("已有 cell；恢复请使用 --resume")
        cfg = config(arm, seed, cell, device)
        if (cell / "config.yaml").exists():
            if legacy.config_identity(OmegaConf.load(cell / "config.yaml")) != legacy.config_identity(cfg):
                raise ValueError("恢复配置变化")
        else:
            with (cell / "config.yaml").open("x") as stream:
                OmegaConf.save(cfg, stream)
        output = cell / "attempts" / uuid4().hex
        output.mkdir(parents=True)
        try:
            environment = legacy.runtime(device)
            write_json(output / "environment.json", {**environment, "command": sys.argv})
            register_training_environment(session, environment)
            if (cell / "environment.json").exists():
                previous = read_json(cell / "environment.json")
                if any(previous[k] != environment[k] for k in ("torch", "cuda", "packages", "gpu")):
                    raise ValueError("恢复环境的软件版本或 GPU 型号变化")
            else:
                write_json(cell / "environment.json", environment)
            acceptance = ensure_smoke(session, arm, device)
            write_json(output / "engineering.json", {"path": str(acceptance), "sha256": sha(acceptance / "receipt.json")})
            legacy.set_seed(seed)
            data = build_data(cfg, output)
            model = build_model(cfg)
            initial = state_hash(model.state_dict())
            expected = load_session(session)["paired_initialization"][str(seed)][arm]
            if initial != expected["state_sha256"] or sum(p.numel() for p in model.parameters()) != expected["parameters"]:
                raise ValueError("正式构造与 prepare 的初始化身份不同")
            write_json(output / "initialization.json", {"state_sha256": initial,
                "parameters": expected["parameters"], "time_bias_parameters": expected["time_bias_parameters"],
                "frequency_bias_parameters": expected["frequency_bias_parameters"],
                "fusion_projection_parameters": expected["fusion_projection_parameters"],
                "attention_output_zero": expected["attention_output_zero"]})
            model = model.to(device)
            loss = RespirationTaskLoss(cfg).to(device)
            optimizer, partition = build_crd_optimizer(model, cfg)
            write_json(output / "optimizer_groups.json", {"decay": partition.decay_names, "no_decay": partition.no_decay_names})
            result = legacy.run_epochs(model, data, loss, optimizer, cfg, cell, resume=resume, log_path=output / "train.log")
            selected = torch.load(cell / result["best_checkpoint"], map_location="cpu", weights_only=False)
            legacy.finite_tree(selected["model_state_dict"])
            model.load_state_dict(selected["model_state_dict"], strict=True)
            evaluator = CRDExperiment(cfg)
            evaluator.device = torch.device(device)
            metrics = evaluator._evaluate_model(model, data.val.loader)
            sf.validate_metrics(metrics, data.val.rows)
            metrics.insert(0, "arm", arm)
            metrics.insert(1, "seed", seed)
            metrics.to_csv(output / "validation_metrics.csv", index=False)
            summarize_task_metrics(metrics).to_csv(output / "validation_summary.csv", index=False)
            pd.json_normalize([read_json(p / "record.json") for _, p in legacy.completed_epochs(cell)]).to_csv(output / "train_history.csv", index=False)
            load_session(session)
            result.update(arm=arm, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                metrics=str((output / "validation_metrics.csv").relative_to(cell)),
                artifacts={str(p.relative_to(cell)): sha(p) for p in [cell / result["best_checkpoint"],
                    cell / result["final_checkpoint"], output / "validation_metrics.csv", output / "validation_summary.csv",
                    output / "train_history.csv", cell / "config.yaml", cell / "environment.json",
                    output / "environment.json", output / "initialization.json", output / "optimizer_groups.json",
                    output / "engineering.json", output / "access_started.json", output / "data_receipt.json",
                    output / "train_rows.csv", output / "val_rows.csv"]})
            write_json(cell / "completed.json", result)
            return result
        except BaseException as exc:
            write_json(output / "failed.json", {"error": repr(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()})
            raise
        finally:
            gc.collect()


def status(session):
    load_session(session)
    return [{**c, "status": "reference_verified" if c["role"] == "reference" else "completed" if (Path(session) / "cells" / c["cell"] / "completed.json").exists()
             else "partial" if (Path(session) / "cells" / c["cell"] / "config.yaml").exists() else "pending"}
            for c in plan(include_references=True)]


def summary_tables(frame):
    expected = {(a, s) for a in ARMS for s in SEEDS}
    if set(frame[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None)) != expected:
        raise ValueError("汇总要求完整六个 cell（L0/L1 × 三 seed）")
    first, records = None, []
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=False):
        group = group.reset_index(drop=True)
        identity_cols = ["dataset_row_id", "samp_id", "split"]
        sf.validate_metrics(group, group[identity_cols])
        identity = group[identity_cols + sorted(c for c in group if c.endswith("_target_eligible") or c == "local_rr_target_eligible_windows")]
        if first is None:
            first = identity
        elif not identity.equals(first):
            raise ValueError("跨 cell 样本或 target 资格变化")
        for metric in sf.PRIMARY:
            values = group.loc[sf._metric_mask(group, metric), metric].to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError("主指标缺少有效有限值")
            records.append({"arm": arm, "name": ARMS[arm].name, "seed": seed, "metric": metric,
                "mean": float(values.mean()), "n": len(values)})
    per_seed = pd.DataFrame(records)
    across = per_seed.groupby(["arm", "name", "metric"])["mean"].agg(seed_mean="mean", seed_sd="std").reset_index()
    indexed = per_seed.set_index(["arm", "seed", "metric"])["mean"]
    paired = []
    for arm, reference in COMPARISONS.items():
        for seed in SEEDS:
            for metric in sf.PRIMARY:
                delta = indexed[arm, seed, metric] - indexed[reference, seed, metric]
                paired.append({"arm": arm, "reference": reference, "seed": seed, "metric": metric,
                    "delta": delta, "improvement": delta if metric == sf.PCC else -delta})
    subjects = sf.subject_stratified_metrics(frame)
    paired = pd.DataFrame(paired)
    macro = subjects.groupby(["arm", "seed", "metric"])["mean"].mean().reset_index()
    macro_pairs = macro.pivot(index=["seed", "metric"], columns="arm", values="mean").reset_index()
    macro_pairs["arm"], macro_pairs["reference"] = "L1", "L0"
    macro_pairs["delta"] = macro_pairs.L1 - macro_pairs.L0
    macro_pairs["improvement"] = np.where(macro_pairs.metric.eq(sf.PCC), macro_pairs.delta, -macro_pairs.delta)
    return {"per_seed": per_seed, "across_seed": across, "paired_delta": paired,
        "paired_summary": delta_summary(paired, ["arm", "reference", "metric"]),
        "threshold_summary": threshold_summary(paired),
        "per_subject": subjects, "subject_macro": macro,
        "subject_macro_across_seed": macro.groupby(["arm", "metric"])["mean"].agg(seed_mean="mean", seed_sd="std").reset_index(),
        "subject_macro_paired_delta": macro_pairs,
        "subject_macro_paired_summary": delta_summary(macro_pairs, ["arm", "reference", "metric"]),
        "subject_macro_threshold_summary": threshold_summary(macro_pairs),
        "local_rr_tail": sf.local_rr_tail_summary(frame), "denominators": sf.metric_denominators(frame)}


def threshold_summary(frame):
    """0.01 仅用于描述均值幅度，另存逐 seed 达阈值方向，保留原始差值。"""
    threshold = spec()["mean_delta_threshold"]
    rows = []
    for metric, group in frame.groupby("metric", sort=False):
        value = float(group.improvement.mean())
        rows.append({"metric": metric, "threshold": threshold, "delta_mean": float(group.delta.mean()),
            "improvement_mean": value,
            "mean_classification": "no_obvious_mean_difference" if abs(value) < threshold else "improved" if value > 0 else "worsened",
            "improved_at_threshold_seeds": int((group.improvement >= threshold).sum()),
            "worsened_at_threshold_seeds": int((group.improvement <= -threshold).sum()),
            "below_threshold_seeds": int((group.improvement.abs() < threshold).sum())})
    return pd.DataFrame(rows)


def write_report(output, tables):
    """完整三 seed 后比较单/双投影，以窗口加权阈值形成选择建议。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"local_rr_mae_bpm": "Local RR MAE (bpm; lower)",
        "whole_rr_abs_error_bpm": "Whole RR AE (bpm; lower)",
        "envelope_trajectory_mae": "Envelope trajectory MAE (lower)",
        "global_envelope_modulation_error": "Global envelope modulation error (lower)",
        "lag_aware_signed_pcc": "Lag-aware signed PCC (higher)"}
    translate = {"no_obvious_mean_difference": "均值无明显差异", "improved": "单投影后改善", "worsened": "单投影后变差"}
    adverse = tables["threshold_summary"].query("mean_classification == 'worsened'")
    conclusion = ("五项窗口加权主指标均未出现达到 0.01 的均值退步，按本次判据支持单投影作为简化候选。"
        if adverse.empty else "存在达到 0.01 的窗口加权主指标均值退步，应保留双投影或明确相应指标取舍。")
    write_json(output / "decision.json", {"selection_aggregation": "window_weighted", "threshold": .01,
        "recommendation": "supports_single_projection" if adverse.empty else "tradeoff_exceeds_threshold",
        "adverse_metrics": adverse.to_dict("records"), "diagnostic_aggregation": "subject_macro",
        "statistical_equivalence_test": False})
    lines = ["# M4 linear fusion 单/双投影 validation 对照", "", conclusion, "",
        "L0 为三个冻结双投影简化基线；L1 使用单一零初始化 attention.output，随后直接残差相加。",
        "完整三 seed，validation 为 2675 窗口/7 人；checkpoint 按完整 validation Local RR 严格最小、并列最早选择。",
        "delta=L1−L0，误差负值/PCC 正值为改善；绝对均值差<0.01 记为均值无明显差异。",
        "窗口加权为性能选择口径，受试者宏平均作为诊断；sample SD 表示 seed 波动。",
        "两种配置均保留交叉频率 bias 和时域注意力池化，初始条件残差均为零。本阶段仅含 train/validation。", ""]
    for prefix, title in (("", "窗口加权"), ("subject_macro_", "受试者宏平均")):
        table = tables[prefix + "across_seed"].set_index(["arm", "metric"])
        paired = tables[prefix + "paired_summary"].set_index("metric")
        decision = tables[prefix + "threshold_summary"].set_index("metric")
        lines += [f"## {title}", "", "| 指标 | L0 mean±SD | L1 mean±SD | delta mean±SD | 改善 seed 数 | 0.01 判据 |",
            "|---|---:|---:|---:|---:|---|"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        detail = tables["subject_macro" if prefix else "per_seed"]
        for axis, (metric, label) in zip(axes.flat, labels.items()):
            values = [f"{table.loc[(arm, metric), 'seed_mean']:.6f} ± {table.loc[(arm, metric), 'seed_sd']:.6f}" for arm in ARMS]
            row = paired.loc[metric]
            lines.append(f"| {metric} | {values[0]} | {values[1]} | {row.delta_mean:+.6f} ± {row.delta_sd:.6f} | {int(row.improved_seeds)}/3 | {translate[decision.loc[metric, 'mean_classification']]} |")
            for seed in SEEDS:
                selected = detail[(detail.seed == seed) & (detail.metric == metric)].set_index("arm")
                axis.plot([0, 1], [selected.loc[a, "mean"] for a in ARMS], marker="o", label=str(seed))
            axis.set_xticks([0, 1], ["Dual projection", "Single projection"])
            axis.set_title(label, fontsize=10)
            axis.grid(axis="y", alpha=.2)
        axes.flat[0].legend(fontsize=8)
        axes.flat[5].axis("off")
        axes.flat[5].text(.05, .95, "Three paired training seeds\nFixed validation-selected checkpoints\nZero initial conditional residual\nFrequency bias and alignment retained", va="top")
        fig.savefig(output / f"{prefix}paired_validation.png", dpi=160)
        plt.close(fig)
        lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(session):
    session = Path(session)
    manifest = load_session(session)
    if any(not (session / "cells" / cell["cell"] / "completed.json").is_file() for cell in plan()):
        raise RuntimeError("三组新训练尚未全部完成")
    frames, sources, parameters, native = [], {}, [], []
    for cell in plan(include_references=True):
        reference = manifest["references"]["cells"].get(cell["cell"])
        directory = Path(reference["path"]) if reference else session / "cells" / cell["cell"]
        source_arm = reference["source_arm"] if reference else cell["arm"]
        receipt = verified_receipt(directory, source_arm, cell["seed"])
        path = artifact_path(directory, receipt["metrics"])
        frame = pd.read_csv(path)
        if (len(frame) != spec()["data"]["val_windows"] or frame.samp_id.nunique() != spec()["data"]["val_subjects"]
                or set(frame.split) != {"val"} or not frame.arm.eq(source_arm).all() or not frame.seed.eq(cell["seed"]).all()):
            raise ValueError("validation 分母或 cell 标签不符")
        frame["arm"] = cell["arm"]
        frames.append(frame)
        sources[cell["cell"]] = {"path": str(directory), "source_arm": source_arm,
            "completed_sha256": sha(directory / "completed.json"), "role": cell["role"]}
        parameters.append({"arm": cell["arm"], "seed": cell["seed"], "parameters": receipt["parameters"]})
        detail = summarize_task_metrics(frame)
        detail.insert(0, "arm", cell["arm"])
        detail.insert(1, "seed", cell["seed"])
        native.append(detail)
    tables = summary_tables(pd.concat(frames, ignore_index=True))
    tables["parameters"] = pd.DataFrame(parameters)
    tables["sources"] = pd.DataFrame([{"cell": cell, **source} for cell, source in sources.items()])
    tables["native_summary"] = pd.concat(native, ignore_index=True)
    parent = session / "summary"
    parent.mkdir(exist_ok=True)
    with mutex(parent / ".summary.lock"):
        for marker in parent.glob("*/receipt.json"):
            receipt = read_json(marker)
            if receipt["sources"] == sources:
                for name, digest in receipt["artifacts"].items():
                    if sha(marker.parent / name) != digest:
                        raise ValueError("汇总产物身份变化")
                return marker.parent
        output = parent / uuid4().hex
        output.mkdir()
        try:
            for name, frame in tables.items():
                frame.to_csv(output / f"{name}.csv", index=False)
            write_report(output, tables)
            write_json(output / "receipt.json", {"sources": sources,
                "session_sha256": sha(session / "session.json"),
                "artifacts": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
        except BaseException:
            write_json(output / "failed.json", {"traceback": traceback.format_exc()})
            raise
        return output
