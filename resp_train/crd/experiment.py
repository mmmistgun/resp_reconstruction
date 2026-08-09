from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import crd_dependency_versions, load_crd_config
from resp_train.crd.model import build_crd_model
from resp_train.crd.training import (
    PROTOTYPE_REGULARIZER_WEIGHT,
    build_crd_optimizer,
    optimizer_updates_per_epoch,
    train_crd_one_epoch,
)
from resp_train.data.factory import build_tho_data, build_window_data
from resp_train.engine import collect_predictions, save_checkpoint, validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics, validation_local_rr_mean
from resp_train.utils.run import create_run_dir, resolve_device, save_config, save_execution_manifest, set_seed, setup_logger


class CRDExperiment:
    """冻结的 CRD-v1.1 S0/S1/S2 训练与 validation checkpoint 流程。"""

    task_name = "crd_v1"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run_dir: Path | None = None
        self.device: torch.device | None = None

    def train(self) -> Path:
        run_dir = create_run_dir(self.cfg.outputs.run_root)
        self.run_dir = run_dir
        save_config(self.cfg, run_dir)
        save_execution_manifest(
            run_dir / "run_manifest.json",
            task=self.task_name,
            phase="train",
            protocol=str(self.cfg.protocol.name),
            stage=str(self.cfg.protocol.stage),
            run_role=str(self.cfg.protocol.run_role),
            dependency_versions=crd_dependency_versions(),
            resume_supported=False,
        )
        logger = setup_logger(run_dir)
        set_seed(int(self.cfg.training.seed))
        device = resolve_device(str(self.cfg.training.device))
        self.device = device

        data = build_tho_data(self.cfg)
        data.audit_summary.to_csv(run_dir / "audit.csv", index=False)
        model = build_crd_model(self.cfg).to(device)
        loss_fn = RespirationTaskLoss(self.cfg).to(device)
        optimizer, partition = build_crd_optimizer(model, self.cfg)
        (run_dir / "optimizer_parameter_groups.json").write_text(
            json.dumps(
                {
                    "weight_decay": float(self.cfg.training.weight_decay),
                    "decay": list(partition.decay_names),
                    "no_decay": list(partition.no_decay_names),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        total_epochs = int(self.cfg.training.epochs)
        accumulation_steps = int(self.cfg.training.gradient_accumulation_steps)
        updates_per_epoch = optimizer_updates_per_epoch(len(data.train.loader), accumulation_steps)
        total_updates = total_epochs * updates_per_epoch
        update_index = 0
        show_progress = _resolve_show_progress(self.cfg)
        use_amp = bool(self.cfg.training.use_amp)

        history: list[dict[str, float | int]] = []
        best_local_rr = float("inf")
        best_epoch: int | None = None
        for epoch in range(1, total_epochs + 1):
            train_summary, update_index = train_crd_one_epoch(
                model,
                data.train.loader,
                loss_fn,
                optimizer,
                device=device,
                accumulation_steps=accumulation_steps,
                update_index=update_index,
                total_updates=total_updates,
                max_learning_rate=float(self.cfg.training.max_learning_rate),
                min_learning_rate=float(self.cfg.training.min_learning_rate),
                warmup_fraction=float(self.cfg.training.warmup_fraction),
                grad_clip_norm=float(self.cfg.training.grad_clip_norm),
                use_amp=use_amp,
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
                use_amp=use_amp,
            )
            val_core_loss = float(self.cfg.loss.sync_weight) * float(val_summary["loss_sync"]) + float(
                self.cfg.loss.effort_weight
            ) * float(val_summary["loss_effort"])
            val_local_rr = validation_local_rr_mean(val_predictions, self.cfg)
            if not np.isfinite(val_core_loss) or not np.isfinite(val_local_rr):
                raise ValueError("CRD validation core loss 或 Local RR 非有限")

            record: dict[str, float | int] = {
                "epoch": epoch,
                "optimizer_update": update_index,
                "train_loss_total": float(train_summary["loss"]),
                "train_loss_sync": float(train_summary["loss_sync"]),
                "train_loss_effort": float(train_summary["loss_effort"]),
                "first_learning_rate": float(train_summary["first_learning_rate"]),
                "last_learning_rate": float(train_summary["last_learning_rate"]),
                "val_core_loss": val_core_loss,
                "val_local_rr_mae": val_local_rr,
            }
            if "loss_proto" in train_summary:
                record.update(
                    {
                        "train_loss_proto": float(train_summary["loss_proto"]),
                        "train_loss_proto_weighted": float(train_summary["loss_proto_weighted"]),
                        "prototype_regularizer_weight": float(train_summary["prototype_regularizer_weight"]),
                        "regularizer_ramp_first": float(train_summary["regularizer_ramp_first"]),
                        "regularizer_ramp_last": float(train_summary["regularizer_ramp_last"]),
                    }
                )
            history.append(record)
            pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
            logger.info(
                "epoch=%d/%d update=%d/%d train_total=%.6f val_core=%.6f val_local_rr=%.6f",
                epoch,
                total_epochs,
                update_index,
                total_updates,
                record["train_loss_total"],
                val_core_loss,
                val_local_rr,
            )
            checkpoint_extra = {
                "protocol": str(self.cfg.protocol.name),
                "update_index": update_index,
                "total_updates": total_updates,
                "resume_supported": False,
                "dependency_versions": crd_dependency_versions(),
            }
            if "loss_proto" in train_summary:
                checkpoint_extra["structural_regularizer"] = {
                    "name": "prototype_orthogonality",
                    "weight": PROTOTYPE_REGULARIZER_WEIGHT,
                    "ramp": "optimizer_update_5S_to_15S",
                    "updates_per_epoch": updates_per_epoch,
                }
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
                    extra_state=checkpoint_extra,
                )

        if update_index != total_updates:
            raise RuntimeError(f"optimizer update 数不一致: {update_index} != {total_updates}")
        if best_epoch is None:
            raise RuntimeError("训练结束但没有产生 Local RR checkpoint")
        save_checkpoint(
            run_dir / "checkpoint_final.pt",
            model=model,
            optimizer=optimizer,
            epoch=total_epochs,
            metrics=history[-1],
            cfg=self.cfg,
            extra_state={
                "protocol": str(self.cfg.protocol.name),
                "update_index": update_index,
                "total_updates": total_updates,
                "resume_supported": False,
                "dependency_versions": crd_dependency_versions(),
                **(
                    {
                        "structural_regularizer": {
                            "name": "prototype_orthogonality",
                            "weight": PROTOTYPE_REGULARIZER_WEIGHT,
                            "ramp": "optimizer_update_5S_to_15S",
                            "updates_per_epoch": updates_per_epoch,
                        }
                    }
                    if "train_loss_proto" in history[-1]
                    else {}
                ),
            },
        )

        checkpoint = torch.load(run_dir / "checkpoint_best_local_rr.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = self._evaluate_model(model, data.val.loader)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        summarize_task_metrics(metrics).to_csv(run_dir / "metrics_summary.csv", index=False)
        return run_dir

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        loader,
        *,
        evaluation_split: str = "validation",
        include_test_only: bool = False,
    ) -> pd.DataFrame:
        if self.device is None:
            raise RuntimeError("device 尚未初始化")
        predictions = collect_predictions(
            model,
            loader,
            device=self.device,
            max_windows=len(loader.dataset),
            use_amp=bool(self.cfg.training.use_amp),
        )
        frame = evaluate_task_predictions(
            predictions,
            self.cfg,
            include_test_only=include_test_only,
            method=str(self.cfg.model.variant),
        )
        frame.insert(0, "evaluation_split", evaluation_split)
        return frame

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        split: str = "val",
        metrics_output: str | Path | None = None,
    ) -> pd.DataFrame:
        """复评指定 split；research-test 授权必须由受控的公共入口完成。"""

        device = resolve_device(str(self.cfg.training.device))
        self.device = device
        model = build_crd_model(self.cfg).to(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=device)
        _validate_checkpoint_config(checkpoint.get("config"), self.cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        normalized_split = str(split).strip().lower()
        if normalized_split == "val":
            split_name = str(self.cfg.data.val_split)
            max_windows = self.cfg.data.get("max_val_windows")
            strategy = str(self.cfg.data.val_sample_strategy)
            sample_seed = int(self.cfg.data.val_sample_seed)
            evaluation_split = "validation"
            include_test_only = False
        elif normalized_split == "test":
            split_name = str(self.cfg.data.test_split)
            max_windows = self.cfg.data.get("max_test_windows")
            strategy = str(self.cfg.data.test_sample_strategy)
            sample_seed = int(self.cfg.data.test_sample_seed)
            evaluation_split = "test"
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
            evaluation_split=evaluation_split,
            include_test_only=include_test_only,
        )
        if metrics_output is not None:
            output_path = Path(metrics_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(output_path, index=False)
            summarize_task_metrics(metrics).to_csv(
                output_path.with_name(f"{output_path.stem}_summary.csv"),
                index=False,
            )
        return metrics


def evaluate_crd_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    metrics_output_path: str | Path | None,
    overrides: list[str] | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    resolved_config = Path(config_path) if config_path else checkpoint_path.parent / "config.yaml"
    if not resolved_config.exists():
        raise FileNotFoundError("未指定 --config，且 checkpoint 同目录不存在 config.yaml")
    cfg = load_crd_config(resolved_config, overrides=overrides)
    output_path = (
        Path(metrics_output_path)
        if metrics_output_path
        else checkpoint_path.parent / "validation_reeval_metrics.csv"
    )
    CRDExperiment(cfg).evaluate_checkpoint(checkpoint_path, split="val", metrics_output=output_path)
    save_execution_manifest(
        output_path.with_name(f"{output_path.stem}_manifest.json"),
        task=CRDExperiment.task_name,
        phase="validation_evaluation",
        protocol=str(cfg.protocol.name),
        checkpoint=str(checkpoint_path.resolve()),
        config=str(resolved_config.resolve()),
        dependency_versions=crd_dependency_versions(),
    )
    return output_path


def _validate_checkpoint_config(checkpoint_config: dict | None, cfg: DictConfig) -> None:
    if checkpoint_config is None:
        raise ValueError("checkpoint 缺少训练配置")
    checkpoint_cfg = OmegaConf.create(checkpoint_config)
    mismatched: list[str] = []
    for section in ("protocol", "data", "window", "model", "loss", "evaluation"):
        left = OmegaConf.to_container(OmegaConf.select(checkpoint_cfg, section), resolve=True)
        right = OmegaConf.to_container(OmegaConf.select(cfg, section), resolve=True)
        if left != right:
            mismatched.append(section)
    operational_training_keys = {"device", "num_workers", "persistent_workers", "prefetch_factor", "show_progress"}
    left_training = dict(OmegaConf.to_container(checkpoint_cfg.training, resolve=True))
    right_training = dict(OmegaConf.to_container(cfg.training, resolve=True))
    for key in operational_training_keys:
        left_training.pop(key, None)
        right_training.pop(key, None)
    if left_training != right_training:
        mismatched.append("training")
    if mismatched:
        raise ValueError("checkpoint 配置与 CRD 科学协议不一致: " + ", ".join(mismatched))


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
