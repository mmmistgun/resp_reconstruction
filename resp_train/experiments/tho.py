from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data, build_window_data
from resp_train.engine import collect_predictions, save_checkpoint, train_one_epoch, validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics, validation_local_rr_mean
from resp_train.models.registry import build_model
from resp_train.utils.run import create_run_dir, resolve_device, save_config, save_execution_manifest, set_seed, setup_logger


class ThoExperiment:
    """当前唯一 THO 主线：新 loss/metrics 与 Local RR checkpoint。"""

    task_name = "tho_restart"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run_dir: Path | None = None
        self.device: torch.device | None = None

    def train(self) -> Path:
        run_dir = create_run_dir(self.cfg.outputs.run_root)
        self.run_dir = run_dir
        save_config(self.cfg, run_dir)
        save_execution_manifest(run_dir / "run_manifest.json", task=self.task_name, phase="train")
        logger = setup_logger(run_dir)
        set_seed(int(self.cfg.training.seed))
        device = resolve_device(str(self.cfg.training.device))
        self.device = device

        data = build_tho_data(self.cfg)
        data.audit_summary.to_csv(run_dir / "audit.csv", index=False)
        model = build_model(self.cfg).to(device)
        loss_fn = RespirationTaskLoss(self.cfg).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.cfg.training.learning_rate),
            betas=tuple(float(v) for v in self.cfg.training.get("adam_betas", [0.9, 0.999])),
            eps=float(self.cfg.training.get("adam_eps", 1e-8)),
            weight_decay=float(self.cfg.training.get("weight_decay", 0.0)),
        )
        total_epochs = int(self.cfg.training.epochs)
        show_progress = _resolve_show_progress(self.cfg)

        history: list[dict[str, float | int]] = []
        best_local_rr = float("inf")
        best_epoch: int | None = None
        for epoch in range(1, total_epochs + 1):
            train_summary = train_one_epoch(
                model,
                data.train.loader,
                loss_fn,
                optimizer,
                device=device,
                grad_clip_norm=self.cfg.training.get("grad_clip_norm"),
                use_amp=bool(self.cfg.training.use_amp),
                show_progress=show_progress,
                epoch=epoch,
                total_epochs=total_epochs,
            )
            val_summary, val_predictions = validate(
                model,
                data.val.loader,
                loss_fn,
                device=device,
                show_progress=show_progress,
                epoch=epoch,
                total_epochs=total_epochs,
                return_predictions=True,
            )
            val_core_loss = float(self.cfg.loss.sync_weight) * float(
                val_summary["loss_sync"]
            ) + float(self.cfg.loss.effort_weight) * float(val_summary["loss_effort"])
            val_local_rr = validation_local_rr_mean(val_predictions, self.cfg)
            if not np.isfinite(val_core_loss) or not np.isfinite(val_local_rr):
                raise ValueError("validation core loss 或 Local RR 非有限")

            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss_total": float(train_summary["loss"]),
                "train_loss_sync": float(train_summary["loss_sync"]),
                "train_loss_effort": float(train_summary["loss_effort"]),
                "val_core_loss": val_core_loss,
                "val_local_rr_mae": val_local_rr,
            }
            history.append(record)
            pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
            logger.info(
                "epoch=%d/%d train_total=%.6f val_core=%.6f val_local_rr=%.6f best_local_rr=%.6f@%s",
                epoch,
                total_epochs,
                record["train_loss_total"],
                val_core_loss,
                val_local_rr,
                min(best_local_rr, val_local_rr),
                epoch if val_local_rr < best_local_rr else best_epoch,
            )

            if val_local_rr < best_local_rr:
                best_local_rr = val_local_rr
                best_epoch = epoch
                save_checkpoint(
                    run_dir / "checkpoint_best_local_rr.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=self.cfg,
                )

        if best_epoch is None:
            raise RuntimeError("训练结束但没有产生 Local RR checkpoint")
        save_checkpoint(
            run_dir / "checkpoint_final.pt",
            model=model,
            optimizer=optimizer,
            epoch=total_epochs,
            metrics=history[-1],
            cfg=self.cfg,
        )

        checkpoint = torch.load(run_dir / "checkpoint_best_local_rr.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = self._evaluate_model(model, data.val.loader, split="validation", include_test_only=False)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        summary = summarize_task_metrics(metrics)
        summary.to_csv(run_dir / "metrics_summary.csv", index=False)
        signed_pcc = float(summary.iloc[0].get("lag_aware_signed_pcc_mean", np.nan))
        if np.isfinite(signed_pcc) and signed_pcc < 0.0:
            logger.warning("selected checkpoint validation signed PCC < 0：方向/极性科学失败")
        return run_dir

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        loader,
        *,
        split: str,
        include_test_only: bool,
    ) -> pd.DataFrame:
        if self.device is None:
            raise RuntimeError("device 尚未初始化")
        predictions = collect_predictions(
            model,
            loader,
            device=self.device,
            max_windows=len(loader.dataset),
        )
        frame = evaluate_task_predictions(
            predictions,
            self.cfg,
            include_test_only=include_test_only,
            method=str(self.cfg.model.get("experiment_variant", self.cfg.model.name)),
        )
        frame.insert(0, "evaluation_split", split)
        return frame

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        split: str = "val",
        metrics_output: str | Path | None = None,
    ) -> pd.DataFrame:
        device = resolve_device(str(self.cfg.training.device))
        self.device = device
        model = build_model(self.cfg).to(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=device)
        _validate_checkpoint_config(checkpoint.get("config"), self.cfg)
        model.load_state_dict(checkpoint["model_state_dict"])

        normalized_split = str(split).strip().lower()
        if normalized_split == "val":
            split_name = str(self.cfg.data.val_split)
            max_windows = self.cfg.data.get("max_val_windows")
            strategy = str(self.cfg.data.val_sample_strategy)
            sample_seed = int(self.cfg.data.val_sample_seed)
            include_test_only = False
        elif normalized_split == "test":
            split_name = str(self.cfg.data.test_split)
            max_windows = self.cfg.data.get("max_test_windows")
            strategy = str(self.cfg.data.test_sample_strategy)
            sample_seed = int(self.cfg.data.test_sample_seed)
            include_test_only = True
        else:
            raise ValueError("split 必须是 val 或 test")

        window_data = build_window_data(
            self.cfg,
            split=split_name,
            max_windows=max_windows,
            sample_strategy=strategy,
            sample_seed=sample_seed,
            shuffle=False,
        )
        metrics = self._evaluate_model(
            model,
            window_data.loader,
            split=normalized_split,
            include_test_only=include_test_only,
        )
        if metrics_output is not None:
            output_path = Path(metrics_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(output_path, index=False)
            summarize_task_metrics(metrics).to_csv(output_path.with_name(f"{output_path.stem}_summary.csv"), index=False)
        return metrics


def evaluate_tho_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    metrics_output_path: str | Path | None,
    overrides: list[str] | None = None,
    split: str = "val",
    confirm_research_test: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    normalized_split = str(split).strip().lower()
    if normalized_split == "test" and not confirm_research_test:
        raise ValueError("research test 必须显式确认")
    config_path = _resolve_config_path(config_path, checkpoint_path)
    cfg = load_config(
        config_path,
        overrides=overrides,
        allow_legacy_envelope_metric_migration=True,
    )
    output_path = (
        Path(metrics_output_path)
        if metrics_output_path is not None
        else checkpoint_path.parent / ("research_test_metrics.csv" if normalized_split == "test" else "metrics.csv")
    )
    experiment = ThoExperiment(cfg)
    experiment.evaluate_checkpoint(
        checkpoint_path,
        split=split,
        metrics_output=output_path,
    )
    save_execution_manifest(
        output_path.with_name(f"{output_path.stem}_manifest.json"),
        task=ThoExperiment.task_name,
        phase="evaluation",
        split=normalized_split,
        evaluation_role="research_test" if normalized_split == "test" else "validation",
        checkpoint=str(checkpoint_path.resolve()),
        config=str(Path(config_path).resolve()),
    )
    return output_path


def _resolve_config_path(config_path: str | Path | None, checkpoint_path: Path) -> Path:
    if config_path is not None and str(config_path):
        return Path(config_path)
    sidecar = checkpoint_path.parent / "config.yaml"
    if sidecar.exists():
        return sidecar
    raise FileNotFoundError("未指定 --config，且 checkpoint 同目录不存在 config.yaml")


def _validate_checkpoint_config(checkpoint_config: dict | None, cfg: DictConfig) -> None:
    if checkpoint_config is None:
        raise ValueError("checkpoint 缺少训练配置")
    checkpoint_cfg = OmegaConf.create(checkpoint_config)
    # 复评只允许改变设备等运行参数；数据、模型和科学协议必须与训练时完全一致。
    mismatched: list[str] = []
    for section in ("data", "window", "model", "loss", "evaluation"):
        left = OmegaConf.to_container(OmegaConf.select(checkpoint_cfg, section), resolve=True)
        right = OmegaConf.to_container(OmegaConf.select(cfg, section), resolve=True)
        if section == "evaluation":
            evaluation_only_keys = {
                "envelope_quantile_method",
                "envelope_strata_low",
                "envelope_strata_high",
            }
            left = {key: value for key, value in dict(left or {}).items() if key not in evaluation_only_keys}
            right = {key: value for key, value in dict(right or {}).items() if key not in evaluation_only_keys}
        if left != right:
            mismatched.append(section)
    if mismatched:
        raise ValueError("checkpoint 配置与当前科学协议不一致: " + ", ".join(mismatched))


def _resolve_show_progress(cfg: DictConfig) -> bool | None:
    value: Any = cfg.training.get("show_progress", None)
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
