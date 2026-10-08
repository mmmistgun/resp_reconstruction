"""M4-v2 十八组 residual 条件消融的 train/validation 生命周期。"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
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
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.tf_v1_features import cwt_magnitude_features
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch
from resp_train.data.factory import build_tho_data
from resp_train.engine import validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from scripts.m4_residual_model import ARMS, M4ResidualModel, architecture, band_indices, build_model, contract, frequency_metadata

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "m4-residual-v2-20261007"
SPEC = ROOT / "configs/m4_residual_v2/experiment.yaml"
PROTOCOL_PATH = ROOT / "docs/experiments/m4_residual_v2_protocol_20261007.md"
SEEDS = (20260811, 20260812, 20260813)
COMPARISONS = {"A1": "A0", "A2": "A0", "A3": "A0", "S1": "A0", "S2": "A0"}
INCREMENTS = {"tf_information": ("A2", "A1"), "content_attention": ("S2", "A2"),
              "physical_coordinates": ("A0", "S2")}
sha, read_json, write_json, mutex = legacy.sha, legacy.read_json, legacy.write_json, legacy.mutex


def spec():
    value = OmegaConf.to_container(OmegaConf.load(SPEC), resolve=True)
    expected = {"protocol": PROTOCOL, "arms": list(ARMS), "seeds": list(SEEDS), "patch_seconds": 1,
        "normalization": "groupnorm_full_w", "baseline": "configs/crd_tf_v1/crd_tf102_w_formal.yaml",
        "training": {"epochs": 80, "batch_size": 32, "gradient_accumulation_steps": 4,
            "early_stopping_min_epoch": 30, "early_stopping_patience": 15, "early_stopping_min_delta": 0., "amp_dtype": "bfloat16"},
        "data": {"train_windows": 10141, "val_windows": 2675, "train_subjects": 32, "val_subjects": 7},
        "selector": "full_validation_local_rr_strict_minimum_earliest_tie", "comparisons": COMPARISONS,
        "increments": {name: list(pair) for name, pair in INCREMENTS.items()}}
    if value != expected:
        raise ValueError("M4-v2 矩阵/预算/比较合同不符")
    return value


def plan(shard_index=0, shard_count=1):
    if not 1 <= shard_count <= 18 or not 0 <= shard_index < shard_count:
        raise ValueError("非法分片")
    return [{"arm": a, "name": ARMS[a].name, "seed": s, "cell": f"{a}_seed{s}"}
            for s in SEEDS for a in ARMS][shard_index::shard_count]


def config(arm, seed, output, device="cpu"):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("未知 M4-v2 cell")
    value = spec()
    cfg = load_crd_config(ROOT / value["baseline"], overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"])
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "m4_residual_v2"
    cfg.protocol.run_role = "formal"
    cfg.protocol.execution_gate = "m4_residual_synthetic_task_loss_acceptance"
    cfg.model.name = "m4_residual_v2"
    cfg.model.variant = ARMS[arm].name
    cfg.model.m4_residual = contract(arm)
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
    result = legacy.source_identity()
    files = [Path(__file__), ROOT / "scripts/m4_residual_model.py", ROOT / "scripts/run_m4_residual.py",
             ROOT / "scripts/run_m4_residual_shard.sh", SPEC, PROTOCOL_PATH,
             ROOT / "tests/test_m4_residual.py", ROOT / "scripts/p1_components_model.py"]
    for path in files:
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


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
        with tarfile.open(session / "source_snapshot.tar.gz", "x:gz") as archive:
            for name in sources:
                archive.add(ROOT / name, arcname=name)
        if sources != code_identity():
            raise RuntimeError("保存快照期间源码变化")
        write_json(session / "session.json", {"protocol": PROTOCOL, "spec": spec(), "cells": plan(),
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
    arm = cfg.model.m4_residual.arm
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
        "cache_read": ARMS[cfg.model.m4_residual.arm].band != "none", "index": str(index)})
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
    model = M4ResidualModel(arm, SEEDS[0], frequencies, cfg=architecture(), mamba_factory=mamba_factory).to(device)
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
        environment = legacy.runtime(device)
        if (cell / "environment.json").exists():
            previous = read_json(cell / "environment.json")
            if any(previous[k] != environment[k] for k in ("torch", "cuda", "packages", "gpu")):
                raise ValueError("恢复环境的软件版本或 GPU 型号变化")
        else:
            write_json(cell / "environment.json", environment)
        write_json(output / "environment.json", {**environment, "command": sys.argv})
        try:
            acceptance = ensure_smoke(session, arm, device)
            write_json(output / "engineering.json", {"path": str(acceptance), "sha256": sha(acceptance / "receipt.json")})
            legacy.set_seed(seed)
            data = build_data(cfg, output)
            model = build_model(cfg).to(device)
            loss = RespirationTaskLoss(cfg).to(device)
            optimizer, partition = build_crd_optimizer(model, cfg)
            write_json(output / "optimizer_groups.json", {"decay": partition.decay_names, "no_decay": partition.no_decay_names})
            result = legacy.run_epochs(model, data, loss, optimizer, cfg, cell, resume=resume, log_path=output / "train.log")
            selected = torch.load(cell / result["best_checkpoint"], map_location="cpu", weights_only=False)
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
                    output / "train_history.csv", cell / "config.yaml"]})
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


def summary_tables(frame):
    expected = {(a, s) for a in ARMS for s in SEEDS}
    if set(frame[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None)) != expected:
        raise ValueError("汇总要求完整 18-cell")
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
    paired, increments = [], []
    for arm, reference in COMPARISONS.items():
        for seed in SEEDS:
            for metric in sf.PRIMARY:
                delta = indexed[arm, seed, metric] - indexed[reference, seed, metric]
                paired.append({"arm": arm, "reference": reference, "seed": seed, "metric": metric,
                    "delta": delta, "improvement": delta if metric == sf.PCC else -delta})
    for stage, (arm, reference) in INCREMENTS.items():
        for seed in SEEDS:
            for metric in sf.PRIMARY:
                delta = indexed[arm, seed, metric] - indexed[reference, seed, metric]
                increments.append({"stage": stage, "arm": arm, "reference": reference, "seed": seed,
                    "metric": metric, "delta": delta, "improvement": delta if metric == sf.PCC else -delta})
    subjects = sf.subject_stratified_metrics(frame)
    return {"per_seed": per_seed, "across_seed": across, "paired_delta": pd.DataFrame(paired),
        "incremental_delta": pd.DataFrame(increments), "per_subject": subjects,
        "subject_macro": subjects.groupby(["arm", "seed", "metric"])["mean"].mean().reset_index(),
        "local_rr_tail": sf.local_rr_tail_summary(frame), "denominators": sf.metric_denominators(frame)}


def summarize(session):
    session = Path(session)
    load_session(session)
    frames, sources, parameters = [], {}, []
    for cell in plan():
        directory = session / "cells" / cell["cell"]
        receipt = read_json(directory / "completed.json")
        if receipt["arm"] != cell["arm"] or receipt["seed"] != cell["seed"]:
            raise ValueError("完成 cell 标签不符")
        path = directory / receipt["metrics"]
        if sha(path) != receipt["artifacts"][receipt["metrics"]]:
            raise ValueError("metrics 身份变化")
        frame = pd.read_csv(path)
        if len(frame) != spec()["data"]["val_windows"] or set(frame.split) != {"val"} or not frame.arm.eq(cell["arm"]).all() or not frame.seed.eq(cell["seed"]).all():
            raise ValueError("validation 分母或 cell 标签不符")
        frames.append(frame)
        sources[cell["cell"]] = sha(directory / "completed.json")
        parameters.append({"arm": cell["arm"], "seed": cell["seed"], "parameters": receipt["parameters"]})
    tables = summary_tables(pd.concat(frames, ignore_index=True))
    tables["parameters"] = pd.DataFrame(parameters)
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
        write_json(output / "receipt.json", {"sources": sources,
            "artifacts": {f"{name}.csv": sha(output / f"{name}.csv") for name in tables}})
        return output
