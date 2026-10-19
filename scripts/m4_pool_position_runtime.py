"""M4 池化位置 A/B 六组配对重训练及 validation 生命周期。"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train import patch_aligned_tf_training as legacy
from resp_train.crd.config import load_crd_config
from resp_train.crd.tf_v1_features import cwt_magnitude_features
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch
from resp_train.data.factory import build_tho_data
from resp_train.engine import validate, collect_predictions
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import summarize_task_metrics, evaluate_task_predictions
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from scripts.m4_pool_position_model import ARMS, PoolPositionModel, architecture, band_indices, build_model, contract, frequency_metadata
from scripts import p1_components_runtime as p1
from scripts.m4_components_runtime import verified_receipt, delta_summary

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "m4-pool-position-v1-20261015"
SPEC = ROOT / "configs/m4_pool_position_v1/experiment.yaml"
PROTOCOL_PATH = ROOT / "docs/experiments/m4_pool_position_v1_protocol_20261015.md"
SEEDS = (20260811, 20260812, 20260813)
COMPARISONS = {"B": "A"}
sha, read_json, write_json, mutex = legacy.sha, legacy.read_json, legacy.write_json, legacy.mutex


def spec():
    value = OmegaConf.to_container(OmegaConf.load(SPEC), resolve=True)
    expected = {"protocol": PROTOCOL, "arms": list(ARMS), "seeds": list(SEEDS), "patch_seconds": 1,
        "normalization": "groupnorm_representation_variant", "baseline": "configs/crd_tf_v1/crd_tf102_w_formal.yaml",
        "training": {"epochs": 80, "batch_size": 32, "gradient_accumulation_steps": 4,
            "early_stopping_min_epoch": 30, "early_stopping_patience": 15, "early_stopping_min_delta": 0., "amp_dtype": "bfloat16"},
        "data": {"train_windows": 10141, "val_windows": 2675, "train_subjects": 32, "val_subjects": 7},
        "selector": "full_validation_local_rr_strict_minimum_earliest_tie", "comparisons": COMPARISONS}
    decision = value.get("decision")
    expected["decision"] = decision
    if value != expected or not isinstance(decision, dict) or set(decision) != {"local_rr_margin_bpm", "confirmed", "rule"}:
        raise ValueError("池化位置矩阵/预算/比较合同不符")
    margin = decision["local_rr_margin_bpm"]
    valid_margin = margin is None and decision["confirmed"] is False
    valid_margin = valid_margin or (type(margin) in (int, float) and np.isfinite(margin) and margin > 0)
    if (not valid_margin or type(decision["confirmed"]) is not bool
            or decision["rule"] != "mean_gain_at_least_margin_and_all_three_seeds_positive"):
        raise ValueError("实际意义阈值或判断规则无效")
    return value


def plan(shard_index=0, shard_count=1):
    if not 1 <= shard_count <= 6 or not 0 <= shard_index < shard_count:
        raise ValueError("非法分片")
    return [{"arm": a, "name": ARMS[a].name, "seed": s, "cell": f"{a}_seed{s}"}
            for s in SEEDS for a in ARMS][shard_index::shard_count]


def config(arm, seed, output, device="cpu"):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("未知池化位置第一阶段 cell")
    value = spec()
    cfg = load_crd_config(ROOT / value["baseline"], overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"])
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "m4_pool_position_v1"
    cfg.protocol.run_role = "formal"
    cfg.protocol.execution_gate = "m4_pool_position_synthetic_task_loss_acceptance"
    cfg.model.name = "m4_pool_position_v1"
    cfg.model.variant = ARMS[arm].name
    cfg.model.m4_pool_position = contract(arm)
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
    result = p1.code_identity()
    files = [Path(__file__), ROOT / "scripts/m4_pool_position_model.py", ROOT / "scripts/run_m4_pool_position.py",
             ROOT / "scripts/m4_components_runtime.py", ROOT / "scripts/m4_components_model.py", SPEC, PROTOCOL_PATH,
             ROOT / "tests/test_m4_pool_position.py"]
    for path in files:
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


def state_hash(state):
    """参数和结构 buffer 的确定性摘要；真实频率坐标由冻结 cache 单独核验。"""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name == "condition.frequencies_hz":
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def audit_initialization(*, cfg=None, mamba_factory=None):
    """只在 CPU 构造模型；逐 seed 核验共有权重和原始 M4 的完整状态。"""
    from resp_train.crd.tf_v1_features import morlet_scales_and_frequencies
    from scripts.p1_components_model import P1ComponentsModel

    _, frequencies = morlet_scales_and_frequencies(12, 97)
    frequencies = np.sort(frequencies, kind="stable")
    records = {}
    for seed in SEEDS:
        original = P1ComponentsModel("M4", seed, frequencies, cfg=cfg, mamba_factory=mamba_factory)
        a = PoolPositionModel("A", seed, frequencies, cfg=cfg, mamba_factory=mamba_factory)
        b = PoolPositionModel("B", seed, frequencies, cfg=cfg, mamba_factory=mamba_factory)
        old, left, right = original.state_dict(), a.state_dict(), b.state_dict()
        if old.keys() != left.keys() or any(not torch.equal(old[k], left[k]) for k in old):
            raise ValueError("A 与冻结 P1-M4 初始化不一致")
        removed = set(left) - set(right)
        expected_removed = {k for k in left if k.startswith("waveform_encoder.position.")}
        if removed != expected_removed or set(right) - set(left) or any(not torch.equal(v, left[k]) for k, v in right.items()):
            raise ValueError("A/B 共有初始化不一致或移除了非位置参数")
        records[str(seed)] = {arm: {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "position_parameters": sum(p.numel() for n, p in model.named_parameters() if n.startswith("waveform_encoder.position.")),
            "state_sha256": state_hash(model.state_dict())} for arm, model in (("A", a), ("B", b))}
        records[str(seed)].update(common_exact=True, original_m4_exact=True, removed_keys=sorted(removed),
            common_sha256=state_hash(right), mamba="official_mamba2" if mamba_factory is None else "fixture")
    return records


def prepare(session):
    session = Path(session).resolve()
    session.mkdir(parents=True, exist_ok=True)
    with mutex(session / ".prepare.lock"):
        if (session / "session.json").exists():
            load_session(session)
            return session
        if set(p.name for p in session.iterdir()) - {".prepare.lock"}:
            raise FileExistsError("session 路径已有其他产物")
        sources = code_identity()
        initialization = audit_initialization()
        resolved = session / "resolved_configs"
        resolved.mkdir()
        configs = {}
        for cell in plan():
            path = resolved / f"{cell['cell']}.yaml"
            with path.open("x") as stream:
                OmegaConf.save(config(cell["arm"], cell["seed"], session / "cells" / cell["cell"], "cuda:0"), stream)
            configs[str(path.relative_to(session))] = sha(path)
        with tarfile.open(session / "source_snapshot.tar.gz", "x:gz") as archive:
            for name in sources:
                archive.add(ROOT / name, arcname=name)
        if sources != code_identity():
            raise RuntimeError("保存快照期间源码变化")
        write_json(session / "session.json", {"protocol": PROTOCOL, "spec": spec(), "cells": plan(),
            "paired_initialization": initialization,
            "resolved_configs": configs,
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
    if (value["protocol"] != PROTOCOL or value["spec"] != spec() or value["cells"] != plan()
            or value["arms"] != {a: contract(a) for a in ARMS}
            or value["code_sha256"] != code_identity()
            or value["snapshot_sha256"] != sha(session / "source_snapshot.tar.gz")):
        raise ValueError("session 合同或来源身份变化")
    for name, digest in value["resolved_configs"].items():
        if Path(name).is_absolute() or not (session / name).resolve().is_relative_to(session.resolve()) or sha(session / name) != digest:
            raise ValueError("resolved config 身份或路径变化")
    if (value["data_source"] != {"path": str(legacy.P2_LOCK_PATH), "sha256": legacy.P2_LOCK_SHA256}
            or sha(ROOT / legacy.P2_LOCK_PATH) != legacy.P2_LOCK_SHA256):
        raise ValueError("数据来源记录变化")
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
    arm = cfg.model.m4_pool_position.arm
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
        "cache_read": ARMS[cfg.model.m4_pool_position.arm].band != "none", "index": str(index)})
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
    model = PoolPositionModel(arm, SEEDS[0], frequencies, cfg=architecture(), mamba_factory=mamba_factory).to(device)
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
    """短临界区串行登记软件与 GPU 型号，其他分片等待登记完成后校验。"""
    session = Path(session)
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
    session = Path(session).resolve()
    load_session(session)
    if arm not in ARMS or seed not in SEEDS or torch.device(device).type != "cuda":
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
            initial_hash = state_hash(model.state_dict())
            expected_hash = load_session(session)["paired_initialization"][str(seed)][arm]["state_sha256"]
            if initial_hash != expected_hash:
                raise ValueError("正式模型初始化与 prepare 核验不一致")
            write_json(output / "initialization.json", {"state_sha256": initial_hash,
                "parameters": sum(p.numel() for p in model.parameters()),
                "position_parameters": sum(p.numel() for n, p in model.named_parameters() if n.startswith("waveform_encoder.position."))})
            model = model.to(device)
            loss = RespirationTaskLoss(cfg).to(device)
            optimizer, partition = build_crd_optimizer(model, cfg)
            write_json(output / "optimizer_groups.json", {"decay": partition.decay_names, "no_decay": partition.no_decay_names})
            result = legacy.run_epochs(model, data, loss, optimizer, cfg, cell, resume=resume, log_path=output / "train.log")
            selected = torch.load(cell / result["best_checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(selected["model_state_dict"], strict=True)
            predictions = collect_predictions(model, data.val.loader, device=device,
                max_windows=len(data.val.dataset), use_amp=bool(cfg.training.use_amp))
            metrics = evaluate_task_predictions(predictions, cfg, include_test_only=False, method=str(cfg.model.variant))
            metrics.insert(0, "evaluation_split", "validation")
            sf.validate_metrics(metrics, data.val.rows)
            save_cases(output / "cases.npz", data.val, predictions)
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
                    output / "train_history.csv", output / "initialization.json", output / "data_receipt.json",
                    output / "environment.json", output / "optimizer_groups.json", output / "engineering.json",
                    output / "train_rows.csv", output / "val_rows.csv", output / "cases.npz", cell / "config.yaml"]})
            write_json(cell / "completed.json", result)
            return result
        except BaseException as exc:
            write_json(output / "failed.json", {"error": repr(exc), "type": type(exc).__name__})
            raise
        finally:
            gc.collect()


def status(session):
    load_session(session)
    return [{**c, "status": "completed" if (Path(session) / "cells" / c["cell"] / "completed.json").exists()
             else "partial" if (Path(session) / "cells" / c["cell"] / "config.yaml").exists() else "pending"}
            for c in plan()]


def save_cases(path, bundle, predictions):
    """按每人最小 row ID 固定窗口；预测直接复用同一次完整 validation。"""
    rows = bundle.rows.reset_index(drop=True)
    if not np.array_equal(rows.dataset_row_id.to_numpy(), predictions["dataset_row_id"]):
        raise ValueError("案例预测与 validation 行顺序不一致")
    indices = rows.sort_values(["samp_id", "dataset_row_id"]).groupby("samp_id", sort=True).head(1).index.to_numpy()
    inputs = np.stack([bundle.dataset[int(i)]["x"].numpy() for i in indices])
    payload = {"x": inputs, "prediction": predictions["r_tho_hat"][indices],
        "target": predictions["tho_ref"][indices], "dataset_row_id": predictions["dataset_row_id"][indices],
        "samp_id": predictions["samp_id"][indices]}
    if any(not np.isfinite(v).all() for v in payload.values()):
        raise FloatingPointError("波形案例包含非有限值")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **payload)


def plot_cases(session, output):
    """仅从六个已完成 cell 的小型案例产物绘制 PNG，无额外模型推理。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = {}
    for seed in SEEDS:
        cases = {}
        for arm in ARMS:
            directory = session / "cells" / f"{arm}_seed{seed}"
            receipt = verified_receipt(directory, arm, seed)
            source = directory / Path(receipt["metrics"]).parent / "cases.npz"
            if str(source.relative_to(directory)) not in receipt["artifacts"]:
                raise ValueError("案例未列入完成记录")
            with np.load(source, allow_pickle=False) as archive:
                cases[arm] = dict(archive)
        a, b = cases["A"], cases["B"]
        for key in ("x", "target", "dataset_row_id", "samp_id"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError("配对案例的窗口/输入/目标不同")
        for i, row in enumerate(a["dataset_row_id"]):
            fig, axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)
            time_axis = np.arange(a["x"].shape[-1]) / 100
            axes[0].plot(time_axis, a["x"][i].ravel(), lw=.5)
            axes[0].set_title("Model input BCG: bcg_rawish_segment_soft_z_key (100 Hz)")
            for axis in axes[1:]:
                axis.plot(time_axis, a["target"][i].ravel(), color="black", lw=1, label="Target")
                for arm, color in (("A", "tab:blue"), ("B", "tab:orange")):
                    axis.plot(time_axis, cases[arm]["prediction"][i].ravel(), color=color, lw=.8, label=arm)
                axis.legend()
                axis.set_xlabel("Time (s)")
            axes[2].set_xlim(60, 120)
            axes[1].set_title(f"Subject {a['samp_id'][i]}, row {row}, seed {seed}; original timing")
            name = f"case_row{row}_seed{seed}.png"
            fig.savefig(output / name, dpi=150)
            plt.close(fig)
            artifacts[name] = sha(output / name)
    return artifacts


def summary_tables(frame):
    expected = {(a, s) for a in ARMS for s in SEEDS}
    if set(frame[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None)) != expected:
        raise ValueError("汇总要求完整 A/B × 三 seed 六个 cell")
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
    subject_pairs = subjects.pivot(index=["seed", "samp_id", "metric"], columns="arm", values="mean").reset_index()
    subject_pairs["delta_B_minus_A"] = subject_pairs.B - subject_pairs.A
    return {"per_seed": per_seed, "across_seed": across, "paired_delta": pd.DataFrame(paired),
        "paired_summary": delta_summary(paired, ["arm", "reference", "metric"]),
        "per_subject": subjects, "subject_paired_delta": subject_pairs,
        "subject_macro": subjects.groupby(["arm", "seed", "metric"])["mean"].mean().reset_index(),
        "local_rr_tail": sf.local_rr_tail_summary(frame), "denominators": sf.metric_denominators(frame)}


def decision_summary(paired, decision):
    """报告实测效应；只有已确认阈值的 session 才计算相应门槛。"""
    gains = paired.query("metric == 'local_rr_mae_bpm'").delta.to_numpy()
    margin = decision["local_rr_margin_bpm"]
    gate = bool(gains.mean() >= margin and (gains > 0).all()) if decision["confirmed"] else None
    return pd.DataFrame([{"gain_A_over_B_bpm": float(gains.mean()),
        "paired_sd_bpm": float(gains.std(ddof=1)), "positive_seeds": int((gains > 0).sum()),
        "margin_bpm": margin, "threshold_confirmed": decision["confirmed"],
        "stage2_discussion_gate": gate,
        "interpretation": "开发阶段描述性结果；须结合受试者与辅助指标取舍，第二阶段另行确认"}])


def summarize(session):
    session = Path(session)
    load_session(session)
    frames, sources, parameters, selections, auxiliary = [], {}, [], [], []
    for cell in plan():
        directory = session / "cells" / cell["cell"]
        receipt = verified_receipt(directory, cell["arm"], cell["seed"])
        if receipt["arm"] != cell["arm"] or receipt["seed"] != cell["seed"]:
            raise ValueError("完成 cell 标签不符")
        path = directory / receipt["metrics"]
        if sha(path) != receipt["artifacts"][receipt["metrics"]]:
            raise ValueError("metrics 身份变化")
        frame = pd.read_csv(path)
        if (len(frame) != spec()["data"]["val_windows"] or frame.samp_id.nunique() != spec()["data"]["val_subjects"]
                or set(frame.split) != {"val"} or not frame.arm.eq(cell["arm"]).all() or not frame.seed.eq(cell["seed"]).all()):
            raise ValueError("validation 分母或 cell 标签不符")
        frames.append(frame)
        sources[cell["cell"]] = sha(directory / "completed.json")
        parameters.append({"arm": cell["arm"], "seed": cell["seed"], "parameters": receipt["parameters"]})
        selections.append({"arm": cell["arm"], "seed": cell["seed"], **{k: v for k, v in receipt.items()
            if k not in {"artifacts", "metrics"} and not isinstance(v, (dict, list))}})
        detail = summarize_task_metrics(frame)
        detail.insert(0, "arm", cell["arm"])
        detail.insert(1, "seed", cell["seed"])
        auxiliary.append(detail)
    tables = summary_tables(pd.concat(frames, ignore_index=True))
    tables["parameters"] = pd.DataFrame(parameters)
    tables["selection"] = pd.DataFrame(selections)
    tables["all_metrics_per_seed"] = pd.concat(auxiliary, ignore_index=True)
    all_metrics = tables["all_metrics_per_seed"]
    columns = [c for c in all_metrics if c.endswith(("_mean", "_fraction", "_sec"))]
    tables["all_metrics_across_seed"] = all_metrics.melt(id_vars=["arm", "seed"], value_vars=columns,
        var_name="metric", value_name="value").groupby(["arm", "metric"]).value.agg(seed_mean="mean", seed_sd="std").reset_index()
    tables["decision"] = decision_summary(tables["paired_delta"], spec()["decision"])
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
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=False)
        plots = plot_cases(session, output)
        write_json(output / "receipt.json", {"sources": sources,
            "artifacts": {**{f"{name}.csv": sha(output / f"{name}.csv") for name in tables}, **plots}})
        return output
