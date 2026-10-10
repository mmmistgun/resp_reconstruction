"""Patch 时长三 seed 实验：独立 cell、epoch 边界恢复与 validation 汇总。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import load_crd_config, crd_dependency_versions
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.training import build_crd_optimizer, train_crd_one_epoch, optimizer_updates_per_epoch
from resp_train.data.factory import build_tho_data
from resp_train.engine import validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import validation_local_rr_mean, summarize_task_metrics
from resp_train.models.patch_aligned_tf_mamba import (
    PatchAlignedTFMamba, PatchTFConfig, build_patch_aligned_tf_mamba, h_cwt_features,
)
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.w0_structural_factorial_v1_formal import P2_LOCK_PATH, P2_LOCK_SHA256
from resp_train.utils.run import set_seed

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/patch_aligned_tf_mamba/experiment.yaml"
MODEL_CONFIG = ROOT / "configs/patch_aligned_tf_mamba/model.yaml"
PROTOCOL = "patch-aligned-tf-mamba-v1-20261005"
PATCH_SECONDS = (1, 2, 4)
SEEDS = (20260811, 20260812, 20260813)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def finite_tree(value):
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise FloatingPointError("checkpoint/optimizer 包含非有限张量")
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise FloatingPointError("运行状态包含非有限值")
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)


@contextmanager
def mutex(path):
    with Path(path).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"运行已被其他进程占用: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def spec():
    value = OmegaConf.to_container(OmegaConf.load(SPEC), resolve=True)
    if (value["protocol"] != PROTOCOL or value["patch_seconds"] != list(PATCH_SECONDS)
            or value["seeds"] != list(SEEDS)
            or value["baseline"] != "configs/crd_tf_v1/crd_tf102_w_formal.yaml"
            or value["selector"] != "full_validation_local_rr_strict_minimum_earliest_tie"):
        raise ValueError("实验矩阵或 selector 与协议不符")
    if value["training"] != {"epochs": 80, "batch_size": 64, "gradient_accumulation_steps": 2,
            "early_stopping_min_epoch": 30, "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0, "amp_dtype": "bfloat16"}:
        raise ValueError("训练预算与协议不符")
    if value["data"] != {"train_windows": 10141, "val_windows": 2675, "train_subjects": 32, "val_subjects": 7}:
        raise ValueError("数据分母与协议不符")
    return value


def plan(shard_index=0, shard_count=1):
    if not 1 <= shard_count <= 9 or not 0 <= shard_index < shard_count:
        raise ValueError("非法分片编号/数量")
    return [{"patch_seconds": p, "seed": s, "cell": f"p{p}s_seed{s}"}
            for s in SEEDS for p in PATCH_SECONDS][shard_index::shard_count]


def config(patch_seconds, seed, output, device="cpu"):
    if patch_seconds not in PATCH_SECONDS or seed not in SEEDS:
        raise ValueError("cell 不属于 1/2/4 秒 × 三 seed 矩阵")
    contract = spec()
    cfg = load_crd_config(ROOT / contract["baseline"], overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"])
    architecture = OmegaConf.to_container(OmegaConf.load(MODEL_CONFIG).model.patch_aligned_tf_mamba, resolve=True)
    if PatchTFConfig(**architecture) != PatchTFConfig():
        raise ValueError("默认模型结构与本轮时长敏感性协议不符")
    architecture["patch_samples"] = patch_seconds * 100
    PatchTFConfig(**architecture)
    cfg.model.name = "patch_aligned_tf_mamba"
    cfg.model.variant = f"patch_aligned_tf_mamba_p{patch_seconds}s"
    cfg.model.patch_aligned_tf_mamba = architecture
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "patch_aligned_tf_mamba"
    cfg.protocol.run_role = "formal"
    cfg.protocol.execution_gate = "synthetic_task_loss_acceptance"
    for key, value in contract["training"].items():
        cfg.training[key] = value
    cfg.training.early_stopping_enabled = True
    cfg.training.device = str(device)
    cfg.training.show_progress = False
    cfg.data.preload_windows = False
    cfg.outputs.run_root = str(Path(output).resolve())
    return cfg


def config_identity(cfg):
    value = OmegaConf.to_container(cfg, resolve=True)
    # 设备位置不是科学配置，允许恢复到另一张同型号 GPU。
    value["training"]["device"] = "cuda"
    return value


def source_identity():
    paths = list((ROOT / "resp_train").rglob("*.py"))
    paths += list((ROOT / "configs/crd_tf_v1").glob("*.yaml"))
    paths += list((ROOT / "configs/patch_aligned_tf_mamba").glob("*.yaml"))
    paths += [ROOT / "scripts/run_patch_aligned_tf_mamba.py",
              ROOT / "docs/experiments/patch_aligned_tf_mamba_training_v1_20261005.md", ROOT / P2_LOCK_PATH]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def prepare(session):
    session = Path(session).resolve()
    session.mkdir(parents=True, exist_ok=True)
    with mutex(session / ".prepare.lock"):
        if (session / "session.json").exists():
            verify_session(session)
            return session
        if set(p.name for p in session.iterdir()) - {".prepare.lock"}:
            raise FileExistsError("session 路径已有其他产物，请使用新路径")
        contract = spec()
        sources = source_identity()
        with tarfile.open(session / "source_snapshot.tar.gz", "x:gz") as archive:
            for name in sources:
                archive.add(ROOT / name, arcname=name)
        if source_identity() != sources:
            raise RuntimeError("保存期间源码发生变化")
        write_json(session / "session.json", {"protocol": PROTOCOL, "spec": contract, "cells": plan(),
            "source_sha256": sources, "snapshot_sha256": sha(session / "source_snapshot.tar.gz"),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
            "splits": ["train", "val"], "research_test": False})
    return session


def verify_session(session):
    payload = read_json(Path(session) / "session.json")
    if payload["protocol"] != PROTOCOL or payload["spec"] != spec() or payload["source_sha256"] != source_identity():
        raise ValueError("session 的源码或配置身份变化，请恢复来源或新建 session")
    if sha(Path(session) / "source_snapshot.tar.gz") != payload["snapshot_sha256"]:
        raise ValueError("源码快照身份变化")
    return payload


def runtime(device):
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
        "packages": crd_dependency_versions(), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "cpu_fixture",
        "LD_LIBRARY_PATH_unset": "LD_LIBRARY_PATH" not in os.environ,
        "LD_PRELOAD_unset": "LD_PRELOAD" not in os.environ}


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(value, device):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"].cpu(), device)


def _train_epoch(model, loader, loss, optimizer, cfg, update, total, epoch):
    return train_crd_one_epoch(model, loader, loss, optimizer,
        device=cfg.training.device, accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        update_index=update, total_updates=total,
        max_learning_rate=float(cfg.training.max_learning_rate), min_learning_rate=float(cfg.training.min_learning_rate),
        warmup_fraction=float(cfg.training.warmup_fraction), grad_clip_norm=float(cfg.training.grad_clip_norm),
        use_amp=bool(cfg.training.use_amp), show_progress=False, epoch=epoch, total_epochs=int(cfg.training.epochs))


def smoke(session, patch_seconds, device, *, mamba_factory=None, batch_size=64):
    """与正式入口相同的 task loss、AdamW 分组及两微批累积；输入均为合成。"""
    session = Path(session)
    verify_session(session)
    root = session / "engineering"
    root.mkdir(exist_ok=True)
    output = root / f"p{patch_seconds}s_{str(device).replace(':', '_')}_{uuid4().hex[:12]}"
    output.mkdir()
    cfg = config(patch_seconds, SEEDS[0], output, device)
    set_seed(SEEDS[0])
    t = np.arange(18000, dtype=np.float32) / 100
    x = ((1 + .2 * np.sin(2 * np.pi * .25 * t)) * np.sin(2 * np.pi * 2 * t)).astype(np.float32)
    w, frequencies = h_cwt_features(x)
    model = PatchAlignedTFMamba(frequencies, config=PatchTFConfig(**dict(cfg.model.patch_aligned_tf_mamba)),
        initialization_seed=SEEDS[0], mamba_factory=mamba_factory).to(device)
    loss = RespirationTaskLoss(cfg).to(device)
    optimizer, _ = build_crd_optimizer(model, cfg)
    batch = {"x": torch.from_numpy(x)[None, None].repeat(batch_size, 1, 1),
        "target": torch.from_numpy(np.sin(2 * np.pi * .25 * t).astype(np.float32))[None, None].repeat(batch_size, 1, 1),
        "tf": {"w": torch.from_numpy(w)[None].repeat(batch_size, 1, 1)}}
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = model.condition.value.weight.detach().clone()
    summary, updates = _train_epoch(model, [batch] * 4, loss, optimizer, cfg, 0, 6400, 1)
    finite_tree(model.state_dict())
    finite_tree(optimizer.state_dict())
    finite_tree(summary)
    condition_grad = model.condition.value.weight.grad
    if (updates != 2 or torch.equal(before, model.condition.value.weight)
            or condition_grad is None or not torch.isfinite(condition_grad).all()
            or condition_grad.abs().sum() == 0):
        raise RuntimeError("两步更新后条件分支未更新")
    val_summary = validate(model, [batch], loss, device=device, use_amp=True, show_progress=False)
    finite_tree(val_summary)
    report = {"passed": True, "patch_seconds": patch_seconds, "batch_size": batch_size,
        "accumulation": 2, "updates": updates, "training": summary, "validation": val_summary,
        "official_mamba": mamba_factory is None, "condition_gradient_nonzero": True, "environment": runtime(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if torch.device(device).type == "cuda" else None}
    write_json(output / "receipt.json", report)
    return output


def build_data(cfg, output):
    if sha(ROOT / P2_LOCK_PATH) != P2_LOCK_SHA256:
        raise ValueError("既有数据来源记录身份变化")
    source = read_json(ROOT / P2_LOCK_PATH)
    index = Path(cfg.data.dataset_root) / cfg.data.index_csv
    if index.resolve() != Path(source["dataset_index"]["path"]).resolve() or sha(index) != source["dataset_index"]["sha256"]:
        raise ValueError("dataset index 身份不符")
    write_json(output / "access_started.json", {"splits": ["train", "val"], "research_test": False,
        "index": str(index), "cache": str(cfg.data.tf_cache_path)})
    data = build_tho_data(cfg)
    spec_data = spec()["data"]
    identities = {}
    for split in ("train", "val"):
        bundle = getattr(data, split)
        rows = bundle.rows
        ids = rows.dataset_row_id.to_numpy(np.int64)
        row_hash = hashlib.sha256(np.sort(ids).tobytes()).hexdigest()
        if (len(rows) != spec_data[f"{split}_windows"] or rows.samp_id.nunique() != spec_data[f"{split}_subjects"]
                or set(rows.split) != {split} or len(set(ids)) != len(ids)
                or row_hash != source["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]):
            raise ValueError(f"{split} 样本身份/分母变化")
        rows.to_csv(output / f"{split}_rows.csv", index=False)
        identities[split] = row_hash
    if set(data.train.rows.samp_id) & set(data.val.rows.samp_id):
        raise ValueError("train/val 受试者重叠")
    if optimizer_updates_per_epoch(len(data.train.loader), 2) != 80:
        raise ValueError("正式训练应为每 epoch 80 次更新")
    write_json(output / "data_receipt.json", {"row_hashes": identities,
        "source_sha256": P2_LOCK_SHA256, "index_sha256": sha(index)})
    return data


def completed_epochs(cell):
    records = []
    for receipt in sorted((Path(cell) / "epochs").glob("*/receipt.json")):
        value = read_json(receipt)
        directory = receipt.parent
        if sha(directory / "checkpoint.pt") != value["checkpoint_sha256"]:
            raise ValueError("epoch checkpoint 身份变化")
        records.append((int(value["epoch"]), directory))
    records.sort()
    if [e for e, _ in records] != list(range(1, len(records) + 1)):
        raise ValueError("epoch 回执存在重复或缺口")
    return records


def run_epochs(model, data, loss, optimizer, cfg, cell, *, resume=False, log_path=None):
    """每个完整 epoch 为恢复边界；checkpoint 和回执均追加写入。"""
    cell = Path(cell)
    (cell / "epochs").mkdir(exist_ok=True)
    prior = completed_epochs(cell)
    if prior and not resume:
        raise FileExistsError("已有 epoch，请显式使用 --resume")
    total = int(cfg.training.epochs) * optimizer_updates_per_epoch(len(data.train.loader), int(cfg.training.gradient_accumulation_steps))
    update, history, best, best_epoch, wait, stopped = 0, [], float("inf"), None, 0, False
    if prior:
        state = torch.load(prior[-1][1] / "checkpoint.pt", map_location="cpu", weights_only=False)
        finite_tree(state)
        if state["config"] != config_identity(cfg) or state["total_updates"] != total:
            raise ValueError("恢复 checkpoint 配置或预算不一致")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        update, history, best, best_epoch, wait, stopped = (state[k] for k in
            ("update_index", "history", "best_local_rr", "best_epoch", "wait", "stopped"))
        if len(history) != len(prior) or history[-1]["epoch"] != prior[-1][0]:
            raise ValueError("checkpoint/history/epoch 不一致")
        expected_updates = len(history) * optimizer_updates_per_epoch(len(data.train.loader), int(cfg.training.gradient_accumulation_steps))
        if update != expected_updates or best_epoch not in range(1, len(history) + 1):
            raise ValueError("checkpoint 更新计数或 best epoch 不一致")
        restore_rng(state["rng"], cfg.training.device)
    for epoch in range(len(history) + 1, int(cfg.training.epochs) + 1):
        if stopped:
            break
        started = time.perf_counter()
        train_summary, update = _train_epoch(model, data.train.loader, loss, optimizer, cfg, update, total, epoch)
        val_summary, predictions = validate(model, data.val.loader, loss, device=cfg.training.device,
            use_amp=bool(cfg.training.use_amp), show_progress=False, return_predictions=True)
        score = float(validation_local_rr_mean(predictions, cfg))
        finite_tree((train_summary, val_summary, score))
        improved = score < best - float(cfg.training.early_stopping_min_delta)
        if improved:
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        stopped = epoch >= int(cfg.training.early_stopping_min_epoch) and wait >= int(cfg.training.early_stopping_patience)
        record = {"epoch": epoch, "optimizer_update": update, "train": train_summary,
            "validation": val_summary, "val_local_rr_mae": score, "best_epoch": best_epoch,
            "wait": wait, "stopped": stopped, "elapsed_seconds": time.perf_counter() - started}
        history.append(record)
        state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "config": config_identity(cfg), "epoch": epoch, "update_index": update, "total_updates": total,
            "history": history, "best_local_rr": best, "best_epoch": best_epoch, "wait": wait,
            "stopped": stopped, "rng": rng_state(cfg.training.device)}
        finite_tree(state)
        directory = cell / "epochs" / f"epoch_{epoch:04d}_{uuid4().hex[:12]}"
        directory.mkdir()
        with (directory / "checkpoint.pt").open("xb") as stream:
            torch.save(state, stream)
        write_json(directory / "record.json", record)
        write_json(directory / "receipt.json", {"epoch": epoch, "checkpoint_sha256": sha(directory / "checkpoint.pt")})
        line = f"epoch={epoch} update={update}/{total} val_local_rr={score:.6f} best_epoch={best_epoch}"
        print(line, flush=True)
        if log_path is not None:
            with Path(log_path).open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
    if best_epoch is None:
        raise RuntimeError("没有有效 validation checkpoint")
    epochs = dict(completed_epochs(cell))
    return {"best_epoch": best_epoch, "best_local_rr": best,
        "best_checkpoint": str((epochs[best_epoch] / "checkpoint.pt").relative_to(cell)),
        "final_checkpoint": str((epochs[len(history)] / "checkpoint.pt").relative_to(cell)),
        "completed_epochs": len(history), "optimizer_updates": update, "early_stopped": stopped}


def run_cell(session, patch_seconds, seed, device, *, resume=False):
    session = Path(session).resolve()
    verify_session(session)
    if torch.device(device).type != "cuda":
        raise ValueError("正式入口要求显式 CUDA 设备")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("正式入口要求 BF16")
    cells = session / "cells"
    cells.mkdir(exist_ok=True)
    cell = cells / f"p{patch_seconds}s_seed{seed}"
    cell.mkdir(exist_ok=True)
    with mutex(cell / ".run.lock"):
        if (cell / "completed.json").exists():
            result = read_json(cell / "completed.json")
            for name, digest in result["artifacts"].items():
                if sha(cell / name) != digest:
                    raise ValueError("完成产物身份变化")
            print(f"已完成，复用: {cell.name}", flush=True)
            return result
        if (cell / "config.yaml").exists() and not resume:
            raise FileExistsError("cell 已存在；恢复请使用 --resume")
        cfg = config(patch_seconds, seed, cell, device)
        if (cell / "config.yaml").exists():
            if config_identity(OmegaConf.load(cell / "config.yaml")) != config_identity(cfg):
                raise ValueError("cell 配置变化")
        else:
            with (cell / "config.yaml").open("x") as stream:
                OmegaConf.save(cfg, stream)
        attempts = cell / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = attempts / uuid4().hex
        attempt.mkdir()
        environment = runtime(device)
        environment_path = cell / "environment.json"
        if environment_path.exists():
            previous = read_json(environment_path)
            if any(previous[k] != environment[k] for k in ("torch", "cuda", "packages", "gpu")):
                raise ValueError("恢复环境的软件版本或 GPU 型号变化")
        else:
            write_json(environment_path, environment)
        write_json(attempt / "environment.json", {**environment, "command": sys.argv})
        try:
            # 每次进程启动先验证该时长在正式 loss/累积路径下可运行。
            acceptance = smoke(session, patch_seconds, device)
            write_json(attempt / "engineering.json", {"path": str(acceptance), "sha256": sha(acceptance / "receipt.json")})
            set_seed(seed)
            data = build_data(cfg, attempt)
            model = build_patch_aligned_tf_mamba(cfg).to(device)
            loss = RespirationTaskLoss(cfg).to(device)
            optimizer, partition = build_crd_optimizer(model, cfg)
            write_json(attempt / "optimizer_groups.json", {"decay": partition.decay_names, "no_decay": partition.no_decay_names})
            result = run_epochs(model, data, loss, optimizer, cfg, cell, resume=resume, log_path=attempt / "train.log")
            state = torch.load(cell / result["best_checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state_dict"], strict=True)
            evaluator = CRDExperiment(cfg)
            evaluator.device = torch.device(device)
            metrics = evaluator._evaluate_model(model, data.val.loader)
            sf.validate_metrics(metrics, data.val.rows)
            metrics.insert(0, "patch_seconds", patch_seconds)
            metrics.insert(0, "seed", seed)
            metrics_path = attempt / "validation_metrics.csv"
            metrics.to_csv(metrics_path, index=False)
            summarize_task_metrics(metrics).to_csv(attempt / "validation_summary.csv", index=False)
            history_path = attempt / "train_history.csv"
            pd.json_normalize([read_json(p / "record.json") for _, p in completed_epochs(cell)]).to_csv(history_path, index=False)
            verify_session(session)
            result.update(patch_seconds=patch_seconds, seed=seed,
                metrics=str(metrics_path.relative_to(cell)),
                artifacts={str(p.relative_to(cell)): sha(p) for p in
                    [cell / result["best_checkpoint"], cell / result["final_checkpoint"], metrics_path,
                     attempt / "validation_summary.csv", history_path, cell / "config.yaml"]})
            write_json(cell / "completed.json", result)
            return result
        except BaseException as exc:
            write_json(attempt / "failed.json", {"error": repr(exc), "type": type(exc).__name__})
            raise


def status(session):
    verify_session(session)
    return [{**cell, "status": "completed" if (Path(session) / "cells" / cell["cell"] / "completed.json").exists()
             else "partial" if (Path(session) / "cells" / cell["cell"] / "config.yaml").exists() else "pending"}
            for cell in plan()]


def summarize(session):
    session = Path(session)
    verify_session(session)
    rows, sources, reference = [], {}, None
    for cell in plan():
        directory = session / "cells" / cell["cell"]
        result = read_json(directory / "completed.json")
        path = directory / result["metrics"]
        if sha(path) != result["artifacts"][result["metrics"]]:
            raise ValueError("validation metrics 身份变化")
        frame = pd.read_csv(path)
        if (len(frame) != spec()["data"]["val_windows"] or not frame.seed.eq(cell["seed"]).all()
                or not frame.patch_seconds.eq(cell["patch_seconds"]).all()):
            raise ValueError("validation cell/分母不符")
        identity = frame[["dataset_row_id", "samp_id", "split"] + sorted(c for c in frame if c.endswith("_target_eligible"))]
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError("跨 cell 样本或 target 资格变化")
        summary = summarize_task_metrics(frame)
        if len(summary) != 1:
            raise ValueError("每个 cell 必须有唯一 validation 汇总")
        summary.insert(0, "seed", cell["seed"])
        summary.insert(0, "patch_seconds", cell["patch_seconds"])
        rows.append(summary)
        sources[cell["cell"]] = sha(directory / "completed.json")
    parent = session / "summary"
    parent.mkdir(exist_ok=True)
    with mutex(parent / ".summary.lock"):
        for receipt in parent.glob("*/receipt.json"):
            saved = read_json(receipt)
            if saved["sources"] == sources:
                for name, digest in saved["artifacts"].items():
                    if sha(receipt.parent / name) != digest:
                        raise ValueError("summary 产物身份变化")
                return receipt.parent
        return _write_summary(parent, rows, sources)


def _write_summary(parent, rows, sources):
    output = parent / uuid4().hex
    output.mkdir()
    per_seed = pd.concat(rows, ignore_index=True)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    means = [c for c in per_seed if c.endswith("_mean")]
    per_seed.groupby("patch_seconds")[means].agg(["mean", "std"]).to_csv(output / "across_seed.csv")
    write_json(output / "receipt.json", {"sources": sources,
        "artifacts": {name: sha(output / name) for name in ("per_seed.csv", "across_seed.csv")}})
    return output
