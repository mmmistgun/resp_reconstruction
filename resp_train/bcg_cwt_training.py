"""BCG-CWT 独立 train/val 入口，复用任务损失、优化器和评价实现。"""
from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from contextlib import contextmanager
from uuid import uuid4

import numpy as np
from omegaconf import OmegaConf
import torch

from resp_train.crd.training import build_crd_optimizer, train_crd_one_epoch, optimizer_updates_per_epoch
from resp_train.data.factory import build_tho_data
from resp_train.engine import validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.models.registry import build_model
from resp_train.models.bcg_cwt_patch_mamba import BCGCWTConfig, MODEL_NAME, environment_versions
from resp_train.utils.run import set_seed

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path("/mnt/disk_code/marques/resp_reconstruction")


def load_training_config(path, overrides=()):
    path = Path(path).resolve()
    custom = OmegaConf.load(path)
    if "task_base" in custom or "model_config" in custom:
        base = OmegaConf.load(path.parent / custom.pop("task_base"))
        model = OmegaConf.load(path.parent / custom.pop("model_config"))
        # 基础文件只提供数据、损失、评价和优化器定义；模型与协议有独立身份。
        del base["model"]
        del base["protocol"]
        cfg = OmegaConf.merge(base, model, custom)
    else:
        # 恢复可直接使用运行目录的 resolved_config.yaml，避免重新拼接原始覆盖项。
        cfg = custom
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    validate_training_config(cfg)
    return cfg


def validate_training_config(cfg):
    BCGCWTConfig(cfg.model.architecture)
    if cfg.model.name != MODEL_NAME or list(cfg.model.tf_representations) != ["w"]:
        raise ValueError("训练配置要求 bcg_cwt_patch_mamba_v1 和 tf_representations=[w]")
    if (cfg.window.target_fs, cfg.window.duration_samples) != (100, 18000):
        raise ValueError("window 时间几何与模型不一致")
    if (cfg.data.train_split, cfg.data.val_split) != ("train", "val"):
        raise ValueError("训练入口只允许 train/val")
    if cfg.data.format != "research_v2" or cfg.data.drop_nonfinite_windows:
        raise ValueError("要求 research_v2 且 drop_nonfinite_windows=false")
    if not cfg.data.tf_cache_path:
        raise ValueError("data.tf_cache_path 必填")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps"):
        if type(cfg.training[key]) is not int or cfg.training[key] <= 0:
            raise ValueError(f"training.{key} 必须为正整数")
    if cfg.training.amp_dtype != "bfloat16" or cfg.training.optimizer != "adamw":
        raise ValueError("现有训练循环支持 AdamW 和 bfloat16 AMP")
    if cfg.training.drop_last:
        raise ValueError("此入口要求 drop_last=false；保存每个完整 epoch")
    if type(cfg.training.resume) is not bool:
        raise ValueError("training.resume 必须为 bool")
    if type(cfg.training.early_stopping_enabled) is not bool:
        raise ValueError("training.early_stopping_enabled 必须为 bool")
    for key in ("early_stopping_min_epoch", "early_stopping_patience"):
        if type(cfg.training[key]) is not int or cfg.training[key] <= 0:
            raise ValueError(f"training.{key} 必须为正整数")
    delta = cfg.training.early_stopping_min_delta
    if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
        raise ValueError("training.early_stopping_min_delta 必须为有限非负数")
    RespirationTaskLoss(cfg)


def write_json(path, value):
    # 回执的出现表示写入完成；中断产生的临时文件保留，已有产物不能覆盖。
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.link(temporary, path)
    temporary.unlink()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_sources(output):
    """保存当前源码内容，涵盖尚未提交的新文件。"""
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
                             "resp_train", "scripts", "configs", "docs/models"], cwd=ROOT, check=True, capture_output=True)
    files = sorted(set(result.stdout.decode().strip("\0").split("\0")))
    sources = {}
    with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
        for name in files:
            path = ROOT / name
            if path.is_file():
                archive.add(path, arcname=name, recursive=False)
                sources[name] = sha256(path)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    write_json(output / "source_manifest.json", {"git_commit": git, "files": sources})


def train_epoch(model, loader, loss, optimizer, cfg, update, total, epoch):
    return train_crd_one_epoch(model, loader, loss, optimizer,
        device=cfg.training.device, accumulation_steps=cfg.training.gradient_accumulation_steps,
        update_index=update, total_updates=total, max_learning_rate=cfg.training.max_learning_rate,
        min_learning_rate=cfg.training.min_learning_rate, warmup_fraction=cfg.training.warmup_fraction,
        grad_clip_norm=cfg.training.grad_clip_norm, use_amp=cfg.training.use_amp,
        show_progress=cfg.training.show_progress, epoch=epoch, total_epochs=cfg.training.epochs)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_identity(cfg):
    value = OmegaConf.to_container(cfg, resolve=True)
    # 恢复开关和终端进度不改变训练身份；预算、设备和其他字段保持一致。
    value["training"].pop("resume", None)
    value["training"].pop("show_progress", None)
    return value


def local_rr_from_metrics(metrics):
    """按原 selector 的逐窗口均值聚合，复用本轮已计算的 Local RR。"""
    eligible = metrics["local_rr_target_eligible"].to_numpy(dtype=bool)
    values = metrics["local_rr_mae_bpm"].to_numpy(dtype=np.float64)
    if not eligible.any():
        raise ValueError("完整 validation 没有 target-eligible Local RR sample")
    if not np.isfinite(values[eligible]).all():
        raise FloatingPointError("validation Local RR selector 的 eligible 指标非有限")
    return float(np.mean(values[eligible]))


@contextmanager
def run_mutex(output):
    """仅在进程存活期间互斥，避免两个恢复进程同时更新同一运行。"""
    with (output / ".training.lock").open("ab") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("此运行目录已有训练进程") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def completed_epochs(output):
    complete = {}
    for directory in sorted(output.glob("epoch_*")):
        path = directory / "receipt.json"
        if not path.is_file():
            continue
        receipt = read_json(path)
        if "epoch" not in receipt or "sha256" not in receipt:
            raise ValueError("旧版本 epoch 回执缺少恢复身份；此入口只恢复包含完整恢复状态的新运行")
        epoch = receipt["epoch"]
        if type(epoch) is not int or epoch < 1 or epoch in complete:
            raise ValueError("epoch 回执编号无效或存在重复完整 epoch")
        for name in ("checkpoint.pt", "record.json", "validation_metrics.csv"):
            if sha256(directory / name) != receipt["sha256"][name]:
                raise ValueError(f"epoch {epoch} {name} SHA-256 不一致")
        if read_json(directory / "record.json")["epoch"] != epoch:
            raise ValueError("record/receipt epoch 不一致")
        complete[epoch] = directory
    if sorted(complete) != list(range(1, len(complete) + 1)):
        raise ValueError("完整 epoch 序列不连续")
    return complete


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def finite_tensors(value, name="checkpoint"):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} 包含 NaN/Inf")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tensors(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_tensors(child, name)


def train(cfg):
    validate_training_config(cfg)
    output = Path(cfg.outputs.run_root).resolve()
    if not output.is_relative_to(STORAGE.resolve()):
        raise ValueError(f"outputs.run_root 必须位于 {STORAGE}")
    resume = cfg.training.resume
    if resume:
        if not (output / "resolved_config.yaml").is_file():
            raise ValueError("恢复目录缺少 resolved_config.yaml")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_mutex(output):
        return _train_locked(cfg, output, resume)


def _train_locked(cfg, output, resume):
    attempt = output
    if resume:
        attempt = output / "attempts" / uuid4().hex
        attempt.mkdir(parents=True)
    try:
        if resume:
            if config_identity(OmegaConf.load(output / "resolved_config.yaml")) != config_identity(cfg):
                raise ValueError("恢复配置与原运行不一致（包括 batch、累积、预算、早停和数据定义）")
            if read_json(output / "started.json")["environment"] != environment_versions():
                raise ValueError("恢复环境版本与原运行不一致")
            for name, expected in read_json(output / "source_manifest.json")["files"].items():
                if name.endswith(".py") and sha256(ROOT / name) != expected:
                    raise ValueError(f"恢复源码身份不一致: {name}")
        else:
            with (output / "resolved_config.yaml").open("x", encoding="utf-8") as stream:
                OmegaConf.save(cfg, stream, resolve=True)
            save_sources(output)
        write_json(attempt / "started.json", {"command": sys.argv, "environment": environment_versions(),
            "training_seed": cfg.training.seed, "splits": ["train", "val"], "resume": resume})
        prior = completed_epochs(output)
        if (output / "completed.json").is_file():
            result = read_json(output / "completed.json")
            if result["epochs"] != len(prior):
                raise ValueError("完成回执与完整 epoch 数不一致")
            if (sha256(output / result["best_loss_checkpoint"]) != result["best_loss_checkpoint_sha256"]
                    or sha256(output / result["best_loss_source_checkpoint"]) != result["best_loss_checkpoint_sha256"]):
                raise ValueError("已完成的最低 loss 模型 SHA-256 不一致")
            write_json(attempt / "completed.json", {"reused_completed_run": True, **result})
            return result
        set_seed(cfg.training.seed)
        index = Path(cfg.data.dataset_root) / cfg.data.index_csv
        access = {"splits": ["train", "val"], "research_test": False, "index": str(index),
                  "index_sha256": sha256(index), "cache": str(cfg.data.tf_cache_path),
                  "cache_manifest_sha256": sha256(Path(cfg.data.tf_cache_path) / "cache_manifest.json")}
        if resume and read_json(output / "access_started.json") != access:
            raise ValueError("恢复数据 index/cache 身份不一致")
        write_json(attempt / "access_started.json", access)
        model = build_model(cfg).to(cfg.training.device)
        write_json(attempt / "model_metadata.json", model.get_extra_state())
        data = build_tho_data(cfg)
        row_hashes = {}
        for split in ("train", "val"):
            rows = getattr(data, split).rows
            if set(rows.split) != {split} or rows.dataset_row_id.duplicated().any():
                raise ValueError(f"{split} 行身份无效")
            content = rows.to_csv(index=False).encode("utf-8")
            row_hashes[split] = hashlib.sha256(content).hexdigest()
            if resume and sha256(output / f"{split}_rows.csv") != row_hashes[split]:
                raise ValueError(f"恢复 {split} 行身份或顺序不一致")
            with (attempt / f"{split}_rows.csv").open("xb") as stream:
                stream.write(content)
        if set(data.train.rows.samp_id) & set(data.val.rows.samp_id):
            raise ValueError("train/val 受试者重叠")
        if set(data.train.rows.dataset_row_id) & set(data.val.rows.dataset_row_id):
            raise ValueError("train/val 样本重叠")
        loss = RespirationTaskLoss(cfg).to(cfg.training.device)
        optimizer, partition = build_crd_optimizer(model, cfg)
        write_json(attempt / "optimizer_groups.json", {"decay": partition.decay_names, "no_decay": partition.no_decay_names})
        updates_per_epoch = optimizer_updates_per_epoch(len(data.train.loader), cfg.training.gradient_accumulation_steps)
        total = cfg.training.epochs * updates_per_epoch
        update, best, best_epoch, history = 0, math.inf, None, []
        best_val_loss, best_loss_epoch = math.inf, None
        early_stopping_best, wait, stopped = math.inf, 0, False
        if prior:
            state = torch.load(prior[len(prior)] / "checkpoint.pt", map_location="cpu", weights_only=False)
            finite_tensors(state)
            if state.get("training_state_version") != 2:
                raise ValueError("checkpoint 不包含当前版本的恢复状态")
            if (state["config_identity"] != config_identity(cfg) or state["data_identity"] != row_hashes
                    or state["total_updates"] != total or state["updates_per_epoch"] != updates_per_epoch):
                raise ValueError("checkpoint 配置、数据或更新预算不一致")
            history, update = state["history"], state["update_index"]
            if (len(history) != len(prior) or state["epoch"] != len(prior)
                    or update != len(prior) * updates_per_epoch):
                raise ValueError("checkpoint epoch/history/update_index 不一致")
            for epoch, directory in prior.items():
                if history[epoch - 1] != read_json(directory / "record.json"):
                    raise ValueError("checkpoint history 与已完成记录不一致")
            best, best_epoch = state["best_local_rr"], state["best_epoch"]
            best_val_loss, best_loss_epoch = state["best_val_loss"], state["best_loss_epoch"]
            early_stopping_best, wait, stopped = state["early_stopping_best"], state["wait"], state["stopped"]
            if best_epoch not in prior or best_loss_epoch not in prior:
                raise ValueError("checkpoint 最优 epoch 不存在")
            model.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            # 数据加载器、模型与优化器重建完成后恢复 RNG，下一轮 shuffle/dropout 接续原状态。
            restore_rng(state["rng"], cfg.training.device)
        for epoch in range(len(history) + 1, cfg.training.epochs + 1):
            if stopped:
                break
            directory = output / f"epoch_{epoch:04d}"
            if directory.exists():
                directory = output / f"epoch_{epoch:04d}_retry_{uuid4().hex}"
            directory.mkdir()
            started = time.perf_counter()
            training, update = train_epoch(model, data.train.loader, loss, optimizer, cfg, update, total, epoch)
            validation, predictions = validate(model, data.val.loader, loss, device=cfg.training.device,
                use_amp=cfg.training.use_amp, show_progress=cfg.training.show_progress, return_predictions=True)
            metrics = evaluate_task_predictions(predictions, cfg, include_test_only=False, method=MODEL_NAME)
            score = local_rr_from_metrics(metrics)
            if not math.isfinite(score):
                raise FloatingPointError("validation Local RR selector 非有限")
            val_loss = float(validation["loss"])
            if not math.isfinite(val_loss):
                raise FloatingPointError("validation loss selector 非有限")
            if score < best:
                best, best_epoch = score, epoch
            # 早停参考值仅由 RR 的有效改善更新；独立于严格最小 RR 和最低 loss 的保存。
            if score < early_stopping_best - cfg.training.early_stopping_min_delta:
                early_stopping_best, wait = score, 0
            else:
                wait += 1
            stopped = bool(cfg.training.early_stopping_enabled
                           and epoch >= cfg.training.early_stopping_min_epoch
                           and wait >= cfg.training.early_stopping_patience)
            # 辅助选择独立记录，只比较 validation 总 loss，平局保留较早 epoch。
            if val_loss < best_val_loss:
                best_val_loss, best_loss_epoch = val_loss, epoch
            metrics.to_csv(directory / "validation_metrics.csv", index=False)
            record = {"epoch": epoch, "optimizer_updates": update, "train": training, "validation": validation,
                      "val_local_rr_mae": score, "best_epoch": best_epoch,
                      "best_val_loss": best_val_loss, "best_loss_epoch": best_loss_epoch,
                      "early_stopping_best": early_stopping_best, "wait": wait, "stopped": stopped,
                      "seconds": time.perf_counter() - started}
            history.append(record)
            write_json(directory / "record.json", record)
            state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                     "config": OmegaConf.to_container(cfg, resolve=True), "history": history,
                     "training_state_version": 2, "config_identity": config_identity(cfg), "data_identity": row_hashes,
                     "epoch": epoch, "update_index": update, "total_updates": total, "updates_per_epoch": updates_per_epoch,
                     "best_local_rr": best, "best_epoch": best_epoch,
                     "best_val_loss": best_val_loss, "best_loss_epoch": best_loss_epoch,
                     "early_stopping_best": early_stopping_best, "wait": wait, "stopped": stopped,
                     "rng": rng_state(cfg.training.device)}
            finite_tensors(state)
            with (directory / "checkpoint.pt").open("xb") as stream:
                torch.save(state, stream)
            write_json(directory / "receipt.json", {"epoch": epoch, "sha256": {
                name: sha256(directory / name) for name in ("checkpoint.pt", "record.json", "validation_metrics.csv")}})
            prior[epoch] = directory
            line = (f"epoch={epoch} val_local_rr_mae={score:.6f} best_epoch={best_epoch} "
                    f"val_loss={val_loss:.6f} best_loss_epoch={best_loss_epoch} wait={wait} stopped={stopped}")
            print(line, flush=True)
            with (attempt / "train.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            if stopped:
                break
        # 每轮 checkpoint 已独立保存；完成时复制选中产物，避免重新序列化改变来源身份。
        best_loss_source = str((prior[best_loss_epoch] / "checkpoint.pt").relative_to(output))
        best_loss_hash = sha256(output / best_loss_source)
        if (output / "best_loss.pt").exists():
            if sha256(output / "best_loss.pt") != best_loss_hash:
                raise ValueError("已有 best_loss.pt 与选中 checkpoint 不一致")
        else:
            pending = attempt / f"best_loss_{uuid4().hex}.pending.pt"
            with (output / best_loss_source).open("rb") as source, pending.open("xb") as target:
                shutil.copyfileobj(source, target)
            os.link(pending, output / "best_loss.pt")
            pending.unlink()
        result = {"status": "completed", "epochs": len(history), "best_epoch": best_epoch,
                  "early_stopped": stopped, "early_stopping_best": early_stopping_best, "wait": wait,
                  "best_local_rr_mae": best, "best_checkpoint": str((prior[best_epoch] / "checkpoint.pt").relative_to(output)),
                  "best_val_loss": best_val_loss, "best_loss_epoch": best_loss_epoch,
                  "best_loss_checkpoint": "best_loss.pt", "best_loss_source_checkpoint": best_loss_source,
                  "best_loss_checkpoint_sha256": best_loss_hash}
        write_json(output / "completed.json", result)
        return result
    except BaseException as exc:
        write_json(attempt / "failed.json", {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise
