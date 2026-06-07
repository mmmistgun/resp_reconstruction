from __future__ import annotations

from dataclasses import dataclass
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
        best_epoch = 0
        stale_epochs = 0
        patience = int(self.cfg.training.get("patience", 0))
        min_delta = float(self.cfg.training.get("min_delta", 0.0))

        # 训练循环只关注通用 loss，不包含具体任务指标或数据逻辑。
        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            train_metrics = train_one_epoch(
                model,
                data.train_loader,
                loss_fn,
                optimizer,
                device=device,
                grad_clip_norm=self.cfg.training.get("grad_clip_norm"),
                use_amp=bool(self.cfg.training.get("use_amp", False)),
            )
            val_metrics = validate(model, data.val_loader, loss_fn, device=device)
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "loss"},
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
            }
            history_records.append(record)
            logger.info("epoch=%s train_loss=%.6f val_loss=%.6f", epoch, record["train_loss"], record["val_loss"])

            improved = record["val_loss"] < (best_loss - min_delta)
            if improved:
                best_loss = record["val_loss"]
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(
                    run_dir / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=self.cfg,
                )
            else:
                stale_epochs += 1
                if patience > 0 and stale_epochs >= patience:
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
