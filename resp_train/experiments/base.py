from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import sys
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
        show_progress = self._resolve_show_progress()
        friendly_output = self._friendly_output_enabled(show_progress)

        # 数据构建、审计和 baseline 是任务侧可扩展的前置阶段。
        if friendly_output:
            logger.info("data: building train/val loaders")
        data = self.build_data()
        if friendly_output:
            logger.info(self._format_data_log(data))
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
        checkpoint_gate_enabled = self._checkpoint_gate_enabled()
        total_epochs = int(self.cfg.training.epochs)
        checkpoint_top_k = max(1, int(self.cfg.training.get("checkpoint_top_k", 3)))
        top_checkpoints: list[dict[str, Any]] = []

        # 训练循环只关注通用 loss，不包含具体任务指标或数据逻辑。
        for epoch in range(1, total_epochs + 1):
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
                epoch=epoch,
                total_epochs=total_epochs,
            )
            val_metrics = validate(
                model,
                data.val_loader,
                loss_fn,
                device=device,
                show_progress=show_progress,
                epoch=epoch,
                total_epochs=total_epochs,
            )
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "loss"},
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
            }
            record["checkpoint_gate_passed"] = self._checkpoint_gate_allows(record)

            improved = record["val_loss"] < (best_loss - min_delta)
            checkpoint_improved = record["val_loss"] < (best_checkpoint_loss - min_delta)
            if improved:
                best_loss = record["val_loss"]
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1

            history_records.append(record)
            logger.info(
                self._format_epoch_log(
                    record,
                    friendly=friendly_output,
                    total_epochs=total_epochs,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    patience=patience,
                )
            )

            if record["checkpoint_gate_passed"] and checkpoint_improved:
                best_checkpoint_loss = record["val_loss"]
                if checkpoint_gate_enabled:
                    has_gated_checkpoint = True
                save_checkpoint(
                    run_dir / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=self.cfg,
                )
            elif improved and not checkpoint_gate_enabled:
                best_checkpoint_loss = record["val_loss"]
                save_checkpoint(
                    run_dir / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=self.cfg,
                )
            if record["checkpoint_gate_passed"]:
                top_checkpoints = self._update_top_checkpoints(
                    top_checkpoints,
                    run_dir=run_dir,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    record=record,
                    top_k=checkpoint_top_k,
                )
            if not improved and patience > 0 and stale_epochs >= patience:
                logger.info("early_stop epoch=%s best_epoch=%s best_val_loss=%.6f", epoch, best_epoch, best_loss)
                break
            if scheduler is not None:
                scheduler.step()

        pd.DataFrame(history_records).to_csv(run_dir / "train_history.csv", index=False)
        if checkpoint_gate_enabled and not has_gated_checkpoint:
            raise ValueError(
                "没有 epoch 满足 checkpoint_gate；未保存最终 checkpoint。"
                "请放宽 gate 或检查方向约束。"
            )
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

    def _format_epoch_log(
        self,
        record: dict[str, Any],
        *,
        friendly: bool = False,
        total_epochs: int | None = None,
        best_loss: float | None = None,
        best_epoch: int | None = None,
        stale_epochs: int | None = None,
        patience: int | None = None,
    ) -> str:
        """构造每个 epoch 的核心 loss 和训练/验证分项指标日志。"""
        if friendly:
            return self._format_friendly_epoch_log(
                record,
                total_epochs=total_epochs,
                best_loss=best_loss,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                patience=patience,
            )
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

    def _format_friendly_epoch_log(
        self,
        record: dict[str, Any],
        *,
        total_epochs: int | None,
        best_loss: float | None,
        best_epoch: int | None,
        stale_epochs: int | None,
        patience: int | None,
    ) -> str:
        """面向交互训练的短摘要：先看关键状态，再看 loss 分项。"""
        epoch = int(record["epoch"])
        total = int(total_epochs) if total_epochs is not None else epoch
        gate = "pass" if bool(record.get("checkpoint_gate_passed", True)) else "blocked"
        best_value = float(best_loss) if best_loss is not None else float(record["val_loss"])
        best_at = int(best_epoch) if best_epoch is not None else epoch
        stale = int(stale_epochs or 0)
        patience_text = "off" if not patience or int(patience) <= 0 else f"{stale}/{int(patience)}"
        return "\n".join(
            [
                (
                    f"epoch {epoch}/{total} | "
                    f"best={best_value:.6f}@{best_at} | "
                    f"gate={gate} | patience={patience_text}"
                ),
                f"  loss: train={float(record['train_loss']):.6f} val={float(record['val_loss']):.6f}",
                self._format_metric_parts_table(record),
            ]
        )

    def _format_metric_parts_table(self, record: dict[str, Any]) -> str:
        """用共享表头对齐 train/val loss 分项，便于观察跨 epoch 变化。"""
        train_names = self._metric_part_names(record, prefix="train")
        val_names = self._metric_part_names(record, prefix="val")
        names = sorted(train_names | val_names)
        if not names:
            return "  parts: none"
        lines = ["  parts: metric train val"]
        for name in names:
            train_value = self._format_optional_metric(record, f"train_{name}")
            val_value = self._format_optional_metric(record, f"val_{name}")
            lines.append(f"    {name} {train_value} {val_value}")
        return "\n".join(lines)

    def _metric_part_names(self, record: dict[str, Any], *, prefix: str) -> set[str]:
        """提取除总 loss 外的分项名。"""
        metric_prefix = f"{prefix}_"
        return {
            key.removeprefix(metric_prefix)
            for key in record
            if key.startswith(metric_prefix) and key != f"{prefix}_loss"
        }

    def _format_optional_metric(self, record: dict[str, Any], key: str) -> str:
        """缺失分项用短横线占位，避免 train/val 表头错位。"""
        if key not in record:
            return "-"
        return f"{float(record[key]):.6f}"

    def _format_data_log(self, data: ExperimentData) -> str:
        """汇总数据加载结果，便于长训练启动阶段判断是否按预期。"""
        train_windows = self._safe_len(getattr(data.train_loader, "dataset", None))
        val_windows = self._safe_len(getattr(data.val_loader, "dataset", None))
        train_batches = self._safe_len(data.train_loader)
        val_batches = self._safe_len(data.val_loader)
        parts = [
            f"data: train windows={train_windows} batches={train_batches}",
            f"val windows={val_windows} batches={val_batches}",
        ]
        if self._data_uses_sst_cache(data):
            parts.append("sst_cache=enabled")
        return " | ".join(parts)

    def _update_top_checkpoints(
        self,
        current: list[dict[str, Any]],
        *,
        run_dir: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        record: dict[str, Any],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """保存并维护 val_loss 前 top_k 的 checkpoint。"""
        candidate_path = run_dir / f"checkpoint_epoch_{int(epoch):04d}.pt"
        save_checkpoint(
            candidate_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=record,
            cfg=self.cfg,
        )
        candidates = [
            *current,
            {
                "epoch": int(epoch),
                "val_loss": float(record["val_loss"]),
                "path": candidate_path,
            },
        ]
        ranked = sorted(candidates, key=lambda item: (float(item["val_loss"]), int(item["epoch"])))
        kept = ranked[: int(top_k)]
        kept_paths = {Path(item["path"]) for item in kept}
        for item in ranked[int(top_k) :]:
            path = Path(item["path"])
            if path not in kept_paths and path.exists():
                path.unlink()
        self._write_top_checkpoint_files(run_dir, kept, top_k=top_k)
        return kept

    def _write_top_checkpoint_files(self, run_dir: Path, checkpoints: list[dict[str, Any]], *, top_k: int) -> None:
        """将内部 epoch checkpoint 物化为稳定的 checkpoint_topN.pt 和 manifest。"""
        rows: list[dict[str, Any]] = []
        for rank in range(1, int(top_k) + 1):
            target = run_dir / f"checkpoint_top{rank}.pt"
            if rank <= len(checkpoints):
                item = checkpoints[rank - 1]
                shutil.copy2(Path(item["path"]), target)
                rows.append(
                    {
                        "rank": rank,
                        "epoch": int(item["epoch"]),
                        "val_loss": float(item["val_loss"]),
                        "checkpoint": target.name,
                    }
                )
            elif target.exists():
                target.unlink()
        pd.DataFrame(rows, columns=["rank", "epoch", "val_loss", "checkpoint"]).to_csv(
            run_dir / "checkpoint_topk.csv",
            index=False,
        )

    def _safe_len(self, value: Any) -> str:
        """len 可能对部分 iterable 不可用；日志里用 unknown 表示。"""
        try:
            return str(len(value))
        except TypeError:
            return "unknown"

    def _data_uses_sst_cache(self, data: ExperimentData) -> bool:
        """检测数据集是否已经挂载 SST 缓存，不触发实际取样。"""
        for loader in (data.train_loader, data.val_loader):
            dataset = getattr(loader, "dataset", None)
            if getattr(dataset, "_sst_cache", None) is not None:
                return True
        return False

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

    def _friendly_output_enabled(self, show_progress: bool | None) -> bool:
        """只在进度条实际启用时输出交互友好日志；批量模式保持安静。"""
        if show_progress is not None:
            return bool(show_progress)
        return sys.stderr.isatty()

    def _checkpoint_gate_enabled(self) -> bool:
        """判断是否配置了实际生效的 checkpoint gate。"""
        gate = self.cfg.training.get("checkpoint_gate", None)
        if not gate:
            return False
        return bool(str(gate.get("metric", "")).strip())

    def _checkpoint_gate_allows(self, record: dict[str, Any]) -> bool:
        """根据验证分项决定当前 epoch 是否可作为最终 checkpoint。"""
        gate = self.cfg.training.get("checkpoint_gate", None)
        if not gate:
            return True
        metric = self._resolve_checkpoint_gate_metric(str(gate.get("metric", "")).strip())
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

    def _resolve_checkpoint_gate_metric(self, metric: str) -> str:
        """解析 checkpoint gate 指标，并阻止未启用 loss 分项假通过。"""
        if metric == "auto_direction":
            return self._auto_checkpoint_direction_metric()
        self._validate_checkpoint_gate_metric_is_active(metric)
        return metric

    def _auto_checkpoint_direction_metric(self) -> str:
        """选择当前配置中真正启用的 signed 方向约束指标。"""
        if self._loss_weight_active("signed_corr_weight") or self._signed_schedule_active("signed_corr_schedule"):
            return "val_signed_corr"
        if self._signed_cosine_active():
            return "val_signed_cosine"
        raise ValueError(
            "checkpoint_gate.metric=auto_direction 需要启用 "
            "loss.signed_corr_weight 或 loss.signed_cosine_weight"
        )

    def _validate_checkpoint_gate_metric_is_active(self, metric: str) -> None:
        """显式 gate 到 signed 分项时，确保对应 loss 分项真的启用。"""
        if not metric or "loss" not in self.cfg:
            return
        if metric == "val_signed_corr" and not self._loss_weight_active("signed_corr_weight"):
            raise ValueError("checkpoint_gate.metric=val_signed_corr，但 loss.signed_corr_weight 未启用")
        if metric == "val_signed_cosine" and not self._signed_cosine_active():
            raise ValueError("checkpoint_gate.metric=val_signed_cosine，但 loss.signed_cosine_weight 未启用")

    def _loss_weight_active(self, name: str) -> bool:
        """判断一个普通 loss 权重是否为正。"""
        loss_cfg = self.cfg.get("loss", {})
        return float(loss_cfg.get(name, 0.0)) > 0.0

    def _signed_cosine_active(self) -> bool:
        """判断 signed cosine 是否通过固定权重或 schedule 启用。"""
        if self._loss_weight_active("signed_cosine_weight"):
            return True
        return self._signed_schedule_active("signed_cosine_schedule")

    def _signed_schedule_active(self, name: str) -> bool:
        """判断 signed schedule 是否在任一阶段提供正权重。"""
        loss_cfg = self.cfg.get("loss", {})
        schedule = loss_cfg.get(name, None)
        if not schedule:
            return False
        mode = str(schedule.get("mode", "none")).strip().lower()
        if mode == "none":
            return False
        return (
            float(schedule.get("start_weight", 0.0)) > 0.0
            or float(schedule.get("end_weight", 0.0)) > 0.0
        )
