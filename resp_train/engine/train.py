from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    """执行一个训练 epoch，并返回平均 loss 摘要。"""
    resolved_device = torch.device(device)
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    model.to(resolved_device)
    model.train()
    meter = _LossMeter()

    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        sensor, target = _move_batch(batch, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred = model(sensor)
            loss, parts = loss_fn(pred, target)
        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

        meter.update(loss, parts, batch_size=sensor.size(0))
        progress.set_postfix(loss=f"{meter.summary()['loss']:.4f}")

    return meter.summary()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    """执行验证循环，并返回平均 loss 摘要。"""
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    meter = _LossMeter()

    progress = tqdm(dataloader, desc="val", leave=False)
    for batch in progress:
        sensor, target = _move_batch(batch, resolved_device)
        pred = model(sensor)
        loss, parts = loss_fn(pred, target)
        meter.update(loss, parts, batch_size=sensor.size(0))
        progress.set_postfix(loss=f"{meter.summary()['loss']:.4f}")

    return meter.summary()


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    cfg: DictConfig | None = None,
) -> None:
    """保存模型、优化器和当前指标，供后续评价脚本加载。"""
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if cfg is not None:
        payload["config"] = OmegaConf.to_container(cfg, resolve=True)
    torch.save(payload, Path(path))


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    max_windows: int,
    pred_key: str = "r_tho_hat",
    target_key: str = "tho_ref",
) -> dict[str, np.ndarray]:
    """收集少量或完整验证预测，并展开 DataLoader 默认 collate 后的 meta。"""
    if int(max_windows) <= 0:
        raise ValueError("max_windows 必须大于 0")

    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    meta_records: list[dict[str, Any]] = []

    for batch in loader:
        if "meta" not in batch:
            raise KeyError("batch 必须包含 meta")
        x = batch["x"].to(resolved_device)
        pred = model(x).detach().cpu().numpy()
        target = batch["target"].detach().cpu().numpy()
        preds.append(pred)
        targets.append(target)

        for idx in range(pred.shape[0]):
            meta_records.append(_extract_meta(batch["meta"], idx))
        if len(meta_records) >= int(max_windows):
            break

    if not preds:
        raise RuntimeError("没有可收集的预测窗口")

    pred_arr = np.concatenate(preds, axis=0)[: int(max_windows)]
    target_arr = np.concatenate(targets, axis=0)[: int(max_windows)]
    meta_records = meta_records[: int(max_windows)]
    return {
        pred_key: pred_arr,
        target_key: target_arr,
        "dataset_row_id": np.asarray([int(m.get("dataset_row_id", -1)) for m in meta_records], dtype=np.int64),
        "split": np.asarray([str(m.get("split", "")) for m in meta_records]),
        "input_set": np.asarray([str(m.get("input_set", "")) for m in meta_records]),
        "residual_quality_class": np.asarray([str(m.get("residual_quality_class", "")) for m in meta_records]),
    }


def _extract_meta(meta: Mapping[str, Any], idx: int) -> dict[str, Any]:
    """从默认 collate 后的 meta 字典中取出单个样本的标量元数据。"""
    result: dict[str, Any] = {}
    for key, value in meta.items():
        if torch.is_tensor(value):
            item = value[idx].item() if value.ndim > 0 else value.item()
        elif isinstance(value, np.ndarray):
            item = value[idx].item() if value.ndim > 0 else value.item()
        elif isinstance(value, (list, tuple)):
            item = value[idx]
        else:
            item = value
        result[key] = item
    return result


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        sensor = batch["x"]
        target = batch["target"]
    except KeyError as exc:
        raise KeyError("batch 必须包含 x 和 target") from exc
    return sensor.to(device), target.to(device)


class _LossMeter:
    """按样本数累加 loss，避免最后一个小 batch 影响均值。"""

    def __init__(self) -> None:
        self.total_weight = 0
        self.totals: dict[str, float] = {}

    def update(self, loss: torch.Tensor, parts: Mapping[str, Any], batch_size: int) -> None:
        self.total_weight += batch_size
        self.totals["loss"] = self.totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
        for name, value in parts.items():
            if torch.is_tensor(value):
                item = float(value.detach().cpu())
            else:
                item = float(value)
            self.totals[name] = self.totals.get(name, 0.0) + item * batch_size

    def summary(self) -> dict[str, float]:
        if self.total_weight == 0:
            raise ValueError("没有可用 batch，无法计算平均 loss")
        return {name: value / self.total_weight for name, value in self.totals.items()}
