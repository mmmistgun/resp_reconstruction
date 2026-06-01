from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> dict[str, float]:
    """执行一个训练 epoch，并返回平均 loss 摘要。"""
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.train()
    meter = _LossMeter()

    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        sensor, target = _move_batch(batch, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(sensor)
        loss, parts = loss_fn(pred, target)
        loss.backward()
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
