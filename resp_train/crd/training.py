from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from tqdm.auto import tqdm

from resp_train.crd.blocks import CustomRMSNorm
from resp_train.losses.task import RespirationTaskLoss


@dataclass(frozen=True)
class ParameterPartition:
    decay: tuple[nn.Parameter, ...]
    no_decay: tuple[nn.Parameter, ...]
    decay_names: tuple[str, ...]
    no_decay_names: tuple[str, ...]


def partition_weight_decay_parameters(model: nn.Module) -> ParameterPartition:
    """按 CRD-v1.1 的显式 no-decay 规则无遗漏划分参数。"""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    seen: set[int] = set()
    normalization_types = (nn.GroupNorm, nn.LayerNorm, CustomRMSNorm)
    special_no_decay = {"A_log", "D", "dt_bias", "center_logits", "bandwidth_logits", "P"}

    for module_name, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            leaf = local_name.rsplit(".", 1)[-1]
            is_no_decay = (
                leaf == "bias"
                or isinstance(module, normalization_types)
                or module.__class__.__name__.lower() == "rmsnorm"
                or leaf in special_no_decay
                or full_name.endswith(".center_logits")
                or full_name.endswith(".bandwidth_logits")
            )
            if is_no_decay:
                no_decay.append(parameter)
                no_decay_names.append(full_name)
            else:
                decay.append(parameter)
                decay_names.append(full_name)

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError("AdamW 参数分组存在遗漏或重复")
    return ParameterPartition(
        decay=tuple(decay),
        no_decay=tuple(no_decay),
        decay_names=tuple(decay_names),
        no_decay_names=tuple(no_decay_names),
    )


def build_crd_optimizer(model: nn.Module, cfg: DictConfig) -> tuple[torch.optim.AdamW, ParameterPartition]:
    partition = partition_weight_decay_parameters(model)
    groups = [
        {"params": partition.decay, "weight_decay": float(cfg.training.weight_decay)},
        {"params": partition.no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(cfg.training.max_learning_rate),
        betas=tuple(float(value) for value in cfg.training.adam_betas),
        eps=float(cfg.training.adam_eps),
    )
    return optimizer, partition


def optimizer_updates_per_epoch(microbatch_count: int, accumulation_steps: int) -> int:
    if int(microbatch_count) <= 0 or int(accumulation_steps) <= 0:
        raise ValueError("microbatch_count 与 accumulation_steps 必须为正")
    return math.ceil(int(microbatch_count) / int(accumulation_steps))


def warmup_update_count(total_updates: int, warmup_fraction: float) -> int:
    if int(total_updates) <= 0:
        raise ValueError("total_updates 必须为正")
    raw = math.floor(float(warmup_fraction) * int(total_updates))
    # 正式完整数据下 floor(0.05U)>=1；该下界仅让 acceptance/smoke 仍有定义。
    return max(1, raw)


def crd_learning_rate(
    update_index: int,
    *,
    total_updates: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_fraction: float = 0.05,
) -> float:
    """CRD-v1.1 在 optimizer.step 前使用的精确 update-index LR。"""

    update_index = int(update_index)
    total_updates = int(total_updates)
    if not (0 <= update_index < total_updates):
        raise ValueError(f"update_index={update_index} 超出 [0,{total_updates})")
    maximum = float(max_learning_rate)
    minimum = float(min_learning_rate)
    if not (0.0 < minimum <= maximum):
        raise ValueError("学习率必须满足 0 < min <= max")
    warmup = warmup_update_count(total_updates, warmup_fraction)
    if update_index < warmup:
        return maximum * float(update_index + 1) / float(warmup)
    denominator = total_updates - warmup - 1
    if denominator <= 0:
        return minimum
    progress = float(update_index - warmup) / float(denominator)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def train_crd_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    loss_fn: RespirationTaskLoss,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    accumulation_steps: int,
    update_index: int,
    total_updates: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_fraction: float,
    grad_clip_norm: float,
    use_amp: bool,
    show_progress: bool | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[dict[str, float], int]:
    """按 accumulation-group eligible 总数精确归一化执行一个 epoch。"""

    resolved_device = torch.device(device)
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    non_blocking = resolved_device.type == "cuda"
    accumulation_steps = int(accumulation_steps)
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps 必须为正")
    model.to(resolved_device)
    model.train()
    loss_fn.train()
    meter = _ComponentMeter(sync_weight=loss_fn.sync_weight, effort_weight=loss_fn.effort_weight)

    source = iter(dataloader)
    try:
        microbatch_total = len(dataloader)  # type: ignore[arg-type]
    except TypeError:
        microbatch_total = None
    group_total = (
        optimizer_updates_per_epoch(microbatch_total, accumulation_steps)
        if microbatch_total is not None and microbatch_total > 0
        else None
    )
    groups = _accumulation_groups(source, accumulation_steps)
    progress = tqdm(
        groups,
        total=group_total,
        desc=_progress_description(epoch, total_epochs),
        leave=False,
        disable=not _should_show_progress(show_progress),
    )

    for raw_group in progress:
        prepared = [_prepare_batch(batch, resolved_device, non_blocking=non_blocking) for batch in raw_group]
        group_counts = {"loss_sync_count": 0, "loss_effort_count": 0}
        prepared_counts: list[dict[str, int]] = []
        for _, target, _ in prepared:
            with torch.amp.autocast(resolved_device.type, enabled=False):
                counts = loss_fn.target_component_counts(target.float())
            micro_counts = {key: int(counts[key].item()) for key in group_counts}
            prepared_counts.append(micro_counts)
            for key in group_counts:
                group_counts[key] += micro_counts[key]

        optimizer.zero_grad(set_to_none=True)
        for (sensor, target, sst), expected_counts in zip(prepared, prepared_counts):
            with torch.amp.autocast(
                resolved_device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                prediction = model(sensor, sst=sst) if sst is not None else model(sensor)
            with torch.amp.autocast(resolved_device.type, enabled=False):
                components = loss_fn.differentiable_component_sums(prediction, target.float())
                _validate_component_counts(components, expected_counts)
                sync_term = _normalized_component(
                    components["loss_sync_sum"],
                    group_counts["loss_sync_count"],
                )
                effort_term = _normalized_component(
                    components["loss_effort_sum"],
                    group_counts["loss_effort_count"],
                )
                objective = loss_fn.sync_weight * sync_term + loss_fn.effort_weight * effort_term
            objective.backward()
            meter.update(components)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(grad_clip_norm),
            error_if_nonfinite=True,
        )
        learning_rate = crd_learning_rate(
            update_index,
            total_updates=total_updates,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_fraction=warmup_fraction,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.step()
        update_index += 1
        meter.record_update(learning_rate)
        progress.set_postfix(loss=f"{meter.summary()['loss']:.4f}", lr=f"{learning_rate:.2e}")

    if meter.optimizer_updates == 0:
        raise ValueError("没有可用 microbatch，无法训练")
    return meter.summary(), update_index


def _accumulation_groups(iterator: Any, accumulation_steps: int):
    while True:
        group = list(islice(iterator, accumulation_steps))
        if not group:
            return
        yield group


def _prepare_batch(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    non_blocking: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    try:
        sensor = batch["x"].to(device, non_blocking=non_blocking)
        target = batch["target"].to(device, non_blocking=non_blocking)
    except KeyError as exc:
        raise KeyError("batch 必须包含 x 和 target") from exc
    sst = batch.get("sst")
    if sst is not None:
        sst = sst.to(device, non_blocking=non_blocking)
    return sensor, target, sst


def _normalized_component(numerator: torch.Tensor, count: int) -> torch.Tensor:
    if count > 0:
        return numerator / float(count)
    return numerator


def _validate_component_counts(components: Mapping[str, torch.Tensor], expected_counts: Mapping[str, int]) -> None:
    for name in ("loss_sync", "loss_effort"):
        micro_count = int(components[f"{name}_count"].item())
        expected = int(expected_counts[f"{name}_count"])
        if micro_count != expected:
            raise RuntimeError(f"{name} eligibility count 与 target-only 预计算不一致: {micro_count} != {expected}")


def _should_show_progress(value: bool | None) -> bool:
    return sys.stderr.isatty() if value is None else bool(value)


def _progress_description(epoch: int | None, total_epochs: int | None) -> str:
    if epoch is None or total_epochs is None:
        return "crd train"
    return f"epoch {int(epoch)}/{int(total_epochs)} crd train"


class _ComponentMeter:
    def __init__(self, *, sync_weight: float, effort_weight: float) -> None:
        self.sync_weight = float(sync_weight)
        self.effort_weight = float(effort_weight)
        self.sums = {"loss_sync": 0.0, "loss_effort": 0.0}
        self.counts = {"loss_sync": 0, "loss_effort": 0}
        self.optimizer_updates = 0
        self.first_learning_rate: float | None = None
        self.last_learning_rate: float | None = None

    def update(self, components: Mapping[str, torch.Tensor]) -> None:
        for name in ("loss_sync", "loss_effort"):
            self.sums[name] += float(components[f"{name}_sum"].detach().cpu())
            self.counts[name] += int(components[f"{name}_count"].detach().cpu())

    def record_update(self, learning_rate: float) -> None:
        if self.first_learning_rate is None:
            self.first_learning_rate = float(learning_rate)
        self.last_learning_rate = float(learning_rate)
        self.optimizer_updates += 1

    def summary(self) -> dict[str, float]:
        sync = self.sums["loss_sync"] / self.counts["loss_sync"] if self.counts["loss_sync"] else 0.0
        effort = self.sums["loss_effort"] / self.counts["loss_effort"] if self.counts["loss_effort"] else 0.0
        return {
            "loss": self.sync_weight * sync + self.effort_weight * effort,
            "loss_sync": sync,
            "loss_effort": effort,
            "eligible_sync": float(self.counts["loss_sync"]),
            "eligible_effort": float(self.counts["loss_effort"]),
            "optimizer_updates": float(self.optimizer_updates),
            "first_learning_rate": float(self.first_learning_rate or 0.0),
            "last_learning_rate": float(self.last_learning_rate or 0.0),
        }
