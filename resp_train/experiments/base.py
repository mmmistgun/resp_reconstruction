from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig

from resp_train.engine import save_checkpoint, train_one_epoch, validate
from resp_train.utils.run import create_run_dir, resolve_device, save_config, set_seed, setup_logger


@dataclass(frozen=True)
class ExperimentData:
    """实验构建阶段产出的通用数据包。"""

    train_loader: Any
    val_loader: Any
    audit_frame: pd.DataFrame
    audit_summary: pd.DataFrame
    extras: dict[str, Any]


class BaseExperiment:
    """公共实验基类，封装训练生命周期和可覆写钩子。"""

    task_name = "base"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run_dir: Path | None = None
        self.device: torch.device | None = None

    def train(self) -> Path:
        """执行一次完整训练，并在最佳 checkpoint 上运行最终评价。"""

        run_dir = create_run_dir(self.cfg.outputs.run_root)
        self.run_dir = run_dir
        save_config(self.cfg, run_dir)
        logger = setup_logger(run_dir)
        set_seed(int(self.cfg.training.seed))
        device = resolve_device(str(self.cfg.training.device))
        self.device = device
        logger.info("task=%s device=%s", self.task_name, device)

        # 数据构建、审计和 baseline 是任务侧可扩展的前置阶段。
        data = self.build_data()
        data.audit_summary.to_csv(run_dir / "audit.csv", index=False)
        self.run_baseline(data, run_dir)

        model = self.build_model().to(device)
        loss_fn = self.build_loss()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)

        history_records = []
        best_loss = float("inf")
        best_checkpoint_loss = float("inf")
        best_epoch = 0
        has_gated_checkpoint = False
        stale_epochs = 0
        patience = int(self.cfg.training.get("patience", 0))
        min_delta = float(self.cfg.training.get("min_delta", 0.0))
        show_progress = self._resolve_show_progress()

        # 训练循环只关注通用 loss，不包含具体任务指标或数据逻辑。
        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            if hasattr(loss_fn, "set_epoch"):
                loss_fn.set_epoch(epoch)
            train_metrics = train_one_epoch(
                model,
                data.train_loader,
                loss_fn,
                optimizer,
                device=device,
                grad_clip_norm=self.cfg.training.get("grad_clip_norm"),
                use_amp=bool(self.cfg.training.get("use_amp", False)),
                show_progress=show_progress,
            )
            val_metrics = validate(model, data.val_loader, loss_fn, device=device, show_progress=show_progress)
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "loss"},
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
            }
            record["checkpoint_gate_passed"] = self._checkpoint_gate_allows(record)
            history_records.append(record)
            logger.info(self._format_epoch_log(record))

            improved = record["val_loss"] < (best_loss - min_delta)
            checkpoint_improved = record["val_loss"] < (best_checkpoint_loss - min_delta)
            if improved:
                best_loss = record["val_loss"]
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1

            if record["checkpoint_gate_passed"] and checkpoint_improved:
                best_checkpoint_loss = record["val_loss"]
                has_gated_checkpoint = True
                save_checkpoint(
                    run_dir / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=self.cfg,
                )
            elif improved and not has_gated_checkpoint:
                best_checkpoint_loss = record["val_loss"]
                save_checkpoint(
                    run_dir / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=self.cfg,
                )
            if not improved and patience > 0 and stale_epochs >= patience:
                logger.info("early_stop epoch=%s best_epoch=%s best_val_loss=%.6f", epoch, best_epoch, best_loss)
                break
            if scheduler is not None:
                scheduler.step()

        pd.DataFrame(history_records).to_csv(run_dir / "train_history.csv", index=False)
        # 最终评价统一基于验证集最优模型，避免使用最后一个 epoch 的权重。
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.evaluate_best(model, data, run_dir)
        return run_dir

    def build_data(self) -> ExperimentData:
        raise NotImplementedError

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def build_loss(self) -> torch.nn.Module:
        raise NotImplementedError

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=float(self.cfg.training.learning_rate))

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        name = str(self.cfg.training.get("lr_scheduler", "none"))
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.cfg.training.epochs))
        raise ValueError(f"未知 lr_scheduler: {name}")

    def run_baseline(self, data: ExperimentData, run_dir: Path) -> None:
        return None

    def evaluate_best(self, model: torch.nn.Module, data: ExperimentData, run_dir: Path) -> None:
        raise NotImplementedError

    def _format_epoch_log(self, record: dict[str, Any]) -> str:
        """构造每个 epoch 的核心 loss 和训练/验证分项指标日志。"""
        return (
            f"epoch={int(record['epoch'])} | "
            f"train: {self._format_metric_group(record, prefix='train')} | "
            f"val: {self._format_metric_group(record, prefix='val')}"
        )

    def _format_metric_group(self, record: dict[str, Any], *, prefix: str) -> str:
        """按 loss 优先、分项指标排序的形式格式化一个训练阶段。"""
        parts = [f"loss={float(record[f'{prefix}_loss']):.6f}"]
        metric_prefix = f"{prefix}_"
        metric_keys = sorted(
            key
            for key in record
            if key.startswith(metric_prefix) and key != f"{prefix}_loss"
        )
        for key in metric_keys:
            display_key = key.removeprefix(metric_prefix)
            parts.append(f"{display_key}={float(record[key]):.6f}")
        return " ".join(parts)

    def _resolve_show_progress(self) -> bool | None:
        """解析训练进度条开关；None 表示由训练引擎按终端类型自动判断。"""
        value = self.cfg.training.get("show_progress", None)
        if value in (None, "auto"):
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
            raise ValueError(f"training.show_progress 只能是 true/false/auto，当前为: {value}")
        return bool(value)

    def _checkpoint_gate_allows(self, record: dict[str, Any]) -> bool:
        """根据验证分项决定当前 epoch 是否可作为最终 checkpoint。"""
        gate = self.cfg.training.get("checkpoint_gate", None)
        if not gate:
            return True
        metric = str(gate.get("metric", "")).strip()
        if not metric:
            return True
        if metric not in record:
            raise ValueError(f"checkpoint_gate.metric 不存在于训练记录: {metric}")
        value = float(record[metric])
        if math.isnan(value):
            return False
        min_value = gate.get("min", None)
        max_value = gate.get("max", None)
        if min_value is None and max_value is None:
            raise ValueError("checkpoint_gate 至少需要设置 min 或 max")
        if min_value is not None and value < float(min_value):
            return False
        if max_value is not None and value > float(max_value):
            return False
        return True
