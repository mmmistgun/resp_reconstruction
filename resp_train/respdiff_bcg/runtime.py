"""固定更新训练、完整父窗口验证及不可覆盖的运行记录。"""

import hashlib
import json
import logging
import random
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.respdiff.model import RespDiffSpec, finite
from .config import ROOT
from .model import RespDiffBCG
from .signal import TAPS_SHA256, parent_noise, restore_parent

LOGGER = logging.getLogger("respdiff_bcg")


@contextmanager
def run_logging(output):
    """终端显示常规进度，文件额外保存异常堆栈；退出时恢复 logger。"""
    previous = (LOGGER.handlers[:], LOGGER.level, LOGGER.propagate)
    stream = logging.StreamHandler(sys.stderr)
    logfile = logging.FileHandler(output / "run.log", mode="x", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    formatter.converter = lambda stamp: datetime.fromtimestamp(stamp, ZoneInfo("Asia/Shanghai")).timetuple()
    stream.setLevel(logging.INFO)
    logfile.setLevel(logging.DEBUG)
    for handler in (stream, logfile):
        handler.setFormatter(formatter)
    LOGGER.handlers = [stream, logfile]
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    try:
        yield
    finally:
        for handler in (stream, logfile):
            handler.close()
        LOGGER.handlers, LOGGER.level, LOGGER.propagate = previous


class Progress:
    """首项、末项以及定期间隔输出，重定向后仍是可读的逐行日志。"""
    def __init__(self, label, total, *, every=25, unit="batch"):
        self.label, self.total, self.every, self.unit = label, total, every, unit
        self.started = self.last_logged = time.monotonic()

    def update(self, completed, detail=""):
        now = time.monotonic()
        if completed not in (1, self.total) and completed % self.every and now - self.last_logged < 30:
            return
        elapsed = max(now - self.started, 1e-9)
        eta = elapsed / completed * (self.total - completed)
        LOGGER.info("%s %d/%d (%.1f%%) | %s | %.2f %s/s | elapsed=%s ETA=%s",
                    self.label, completed, self.total, 100 * completed / self.total,
                    detail, completed / elapsed, self.unit,
                    timedelta(seconds=int(elapsed)), timedelta(seconds=int(eta)))
        self.last_logged = now


def cuda_memory_detail(device):
    if device.type != "cuda":
        return ""
    return (f" | 显存 allocated={torch.cuda.memory_allocated(device) / 2**30:.2f} GiB"
            f" reserved={torch.cuda.memory_reserved(device) / 2**30:.2f} GiB")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_identity(cfg):
    paths = sorted((ROOT / "resp_train").rglob("*.py"))
    paths += [ROOT / "scripts/run_respdiff_bcg_v1.py", ROOT / "configs/respdiff_bcg_v1/experiment.yaml"]
    return {"files": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
            "config": OmegaConf.to_container(cfg, resolve=True), "aa_taps_sha256": TAPS_SHA256}


def environment():
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "command": sys.argv}


def save_source_snapshot(output, identity):
    # 工作树允许未提交实现；保存实际源码，避免仅有 commit/hash 而无法恢复。
    with tarfile.open(output / "source_snapshot.tar.gz", "x:gz") as archive:
        for relative in identity["files"]:
            archive.add(ROOT / relative, arcname=relative, recursive=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(cfg, *, tiny=False):
    spec = RespDiffSpec(8, 1, 4) if tiny else RespDiffSpec(**OmegaConf.to_container(cfg.model))
    return RespDiffBCG(spec)


def train_updates(model, dataset, cfg, output, *, updates=6400, batch_size=None,
                  checkpoint_schema="respdiff-bcg-v1", checkpoint_metadata=None):
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(int(cfg.training.seed))
    loader = DataLoader(dataset, batch_size=batch_size or int(cfg.training.batch_size),
                        shuffle=True, drop_last=False, num_workers=0, generator=generator)
    if not len(loader):
        raise ValueError("训练集为空")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.learning_rate),
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.training.milestones), gamma=float(cfg.training.gamma))
    LOGGER.info("训练开始：updates=%d batch=%d chunks=%d；逐步记录：%s",
                updates, loader.batch_size, len(dataset), output / "history.jsonl")
    progress = Progress("训练 update", updates, unit="update")
    iterator, epoch = iter(loader), 0
    model.train()
    with (output / "history.jsonl").open("x") as history:
        for update in range(1, updates + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_loss(batch["x"].to(device), batch["target"].to(device))
            losses["loss"].backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    finite(f"gradient {name}", parameter.grad)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            for name, parameter in model.named_parameters():
                finite(f"parameter {name}", parameter)
            scheduler.step()
            record = {"update": update, "epoch_zero_based": epoch, "lr": lr,
                      "chunk_indices": batch["index"].tolist(),
                      **{k: float(v.detach()) for k, v in losses.items()}}
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            detail = " ".join(f"{key}={record[key]:.6g}" for key in losses)
            progress.update(update, f"{detail} lr={lr:.3g}" + cuda_memory_detail(device))
    for name, value in model.state_dict().items():
        finite(f"checkpoint {name}", value)
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                finite(f"optimizer {name}", value)
    LOGGER.info("保存最终 checkpoint：%s", output / "final.pt")
    checkpoint = {"schema": checkpoint_schema, "update": updates,
                "selector": "final_update", "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "model_spec": vars(model.spec)}
    if checkpoint_metadata is not None:
        checkpoint["experiment"] = checkpoint_metadata
    torch.save(checkpoint, output / "final.pt")


def check_primary_metrics(metrics):
    gates = {"whole_rr_abs_error_bpm": "whole_rr_target_eligible",
             "local_rr_mae_bpm": "local_rr_target_eligible",
             "lag_aware_signed_pcc": "joint_target_eligible",
             "envelope_trajectory_mae": None, "global_envelope_modulation_error": None}
    for metric, flag in gates.items():
        values = metrics[metric].to_numpy(dtype=float)
        eligible = metrics[flag].to_numpy(dtype=bool) if flag else np.ones(len(metrics), dtype=bool)
        if np.isinf(values).any() or not np.isfinite(values[eligible]).all():
            raise FloatingPointError(f"关键指标 {metric} 出现非有限值")


@torch.inference_mode()
def validate(model, dataset, cfg, output, *, batch_size=None, method="RespDiff-BCG-v1"):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size or int(cfg.inference.batch_size),
                        shuffle=False, drop_last=False, num_workers=0)
    LOGGER.info("验证开始：parents=%d chunks=%d batch=%d DDIM=6 N=1",
                len(dataset.rows), len(dataset), loader.batch_size)
    sampling_progress = Progress("验证采样 batch", len(loader))
    chunks, seen = [], []
    for batch_index, batch in enumerate(loader, start=1):
        indices = batch["index"].tolist()
        noises = []
        for index in indices:
            parent, chunk = divmod(index, 13)
            row = dataset.rows.iloc[parent]
            noises.append(parent_noise(split=str(row.split), row_id=int(row.dataset_row_id),
                                       seed=int(cfg.inference.noise_seed))[chunk])
        prediction = model.sample_chunks(batch["x"].to(device), torch.stack(noises).to(device))
        chunks.append(prediction.cpu().numpy())
        seen.extend(indices)
        sampling_progress.update(batch_index, f"chunks={len(seen)}/{len(dataset)}" + cuda_memory_detail(device))
    if seen != list(range(len(dataset))):
        raise ValueError("验证片段缺失、重复或错序")
    predictions = np.concatenate(chunks).reshape(len(dataset.rows), 13, 1, 600)
    lows, highs, targets = [], [], []
    LOGGER.info("重建父窗口：OLA 与 100 Hz 插值")
    reconstruction_progress = Progress("父窗口重建", len(predictions), every=100, unit="parent")
    for index, parent in enumerate(predictions):
        low, high = restore_parent(parent, chunk_indices=range(13))
        lows.append(low)
        highs.append(high)
        targets.append(dataset.parents[index]["target"].numpy().reshape(18000))
        reconstruction_progress.update(index + 1)
    payload = {"r_tho_hat": np.stack(highs), "tho_ref": np.stack(targets),
               "split": dataset.rows["split"].to_numpy(dtype=str),
               **{k: dataset.rows[k].to_numpy(dtype=np.int64) for k in ("dataset_row_id", "samp_id")}}
    LOGGER.info("计算公共指标：parents=%d", len(dataset.rows))
    metric_progress = Progress("指标父窗口", len(dataset.rows), every=64, unit="parent")
    metric_parts = []
    # 公共实现按 64 个父窗口独立计算；外层使用相同边界以报告指标阶段进度。
    for start in range(0, len(dataset.rows), 64):
        stop = min(start + 64, len(dataset.rows))
        part = {key: value[start:stop] for key, value in payload.items()}
        metric_parts.append(evaluate_task_predictions(part, cfg, method=method, include_test_only=False))
        metric_progress.update(stop)
    metrics = pd.concat(metric_parts, ignore_index=True)
    check_primary_metrics(metrics)
    summary = summarize_task_metrics(metrics)
    LOGGER.info("Validation 汇总：%s", " ".join(
        f"{key}={summary.iloc[0][key]:.6g}" for key in summary.columns if key.endswith("_mean")))
    # 共同指标合同允许 target 不合格时为空；CSV 同时保留 eligibility 与计数。
    metrics.to_csv(output / "validation_per_parent.csv", index=False)
    summary.to_csv(output / "validation_summary.csv", index=False)
    LOGGER.info("保存验证波形与指标：%s", output)
    np.savez(output / "validation_waveforms.npz", prediction_20hz=np.stack(lows), **payload)
    write_json(output / "validation_batches.json", {
        "batch_size": loader.batch_size, "ordered_chunk_indices": seen,
        "last_batch_size": len(seen) % loader.batch_size or loader.batch_size,
        "n_parents": len(dataset.rows), "n_samples": 1, "nfe_per_chunk": 6,
        "metric_operator": "resp_train.metrics.task.evaluate_task_predictions"})


def artifact_manifest(output):
    return {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "receipt.json"}


def source_file_inventory(rows):
    paths = sorted(set(rows.source_npz) | set(rows.target_source_npz))
    return {p: {"size": Path(p).stat().st_size, "mtime_ns": Path(p).stat().st_mtime_ns}
            for p in paths}
