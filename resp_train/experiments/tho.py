from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data
from resp_train.engine import collect_predictions
from resp_train.experiments.base import BaseExperiment, ExperimentData
from resp_train.losses.weak import WeakSyncLoss
from resp_train.metrics.baseline import evaluate_baseline_dataset
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.models.registry import build_model
from resp_train.utils.run import resolve_device


class ThoExperiment(BaseExperiment):
    """THO 训练任务的语义层，负责串联任务专属组件。"""

    task_name = "tho"

    def build_data(self) -> ExperimentData:
        tho_data = build_tho_data(self.cfg)
        return ExperimentData(
            train_loader=tho_data.train.loader,
            val_loader=tho_data.val.loader,
            audit_frame=tho_data.audited,
            audit_summary=tho_data.audit_summary,
            extras={"tho_data": tho_data},
        )

    def build_model(self):
        return build_model(self.cfg)

    def build_loss(self):
        return WeakSyncLoss(self.cfg)

    def run_baseline(self, data: ExperimentData, run_dir: Path) -> None:
        if not _baseline_enabled(self.cfg):
            return

        tho_data = data.extras["tho_data"]
        evaluate_baseline_dataset(tho_data.val.dataset, self.cfg).to_csv(
            run_dir / "baseline_metrics.csv",
            index=False,
        )

    def evaluate_best(self, model: torch.nn.Module, data: ExperimentData, run_dir: Path) -> None:
        if self.device is None:
            raise RuntimeError("device 尚未初始化，请通过 train() 运行实验。")

        tho_data = data.extras["tho_data"]
        eval_preds = collect_predictions(
            model,
            tho_data.val.loader,
            device=self.device,
            max_windows=len(tho_data.val.dataset),
        )
        evaluate_prediction_dict(eval_preds, self.cfg, method=str(self.cfg.model.name)).to_csv(
            run_dir / "metrics.csv",
            index=False,
        )

    def evaluate_checkpoint(self, checkpoint_path: Path, *, metrics_output: Path | None) -> None:
        device = self.device or resolve_device(str(self.cfg.training.device))
        model = self.build_model().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        _validate_checkpoint_config(checkpoint.get("config"), self.cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        data = self.build_data()
        tho_data = data.extras["tho_data"]
        if metrics_output is not None:
            metrics_output.parent.mkdir(parents=True, exist_ok=True)
            eval_preds = collect_predictions(
                model,
                tho_data.val.loader,
                device=device,
                max_windows=len(tho_data.val.dataset),
            )
            evaluate_prediction_dict(eval_preds, self.cfg, method=str(self.cfg.model.name)).to_csv(
                metrics_output,
                index=False,
            )


def evaluate_tho_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    metrics_output_path: str | Path | None,
    overrides: list[str] | None = None,
) -> Path:
    resolved_checkpoint = Path(checkpoint_path)
    resolved_config = _resolve_config_path(config_path, resolved_checkpoint)
    cfg = load_config(resolved_config, overrides=overrides)
    experiment = ThoExperiment(cfg)
    experiment.evaluate_checkpoint(
        resolved_checkpoint,
        metrics_output=Path(metrics_output_path) if metrics_output_path else None,
    )
    return Path(metrics_output_path) if metrics_output_path else resolved_checkpoint.parent / "metrics.csv"


def _resolve_config_path(config_path: str | Path | None, checkpoint_path: Path) -> Path:
    """解析评价配置；默认复用训练 run 目录中的配置快照。"""
    if config_path is not None and str(config_path):
        return Path(config_path)
    sidecar = checkpoint_path.parent / "config.yaml"
    if sidecar.exists():
        return sidecar
    raise FileNotFoundError("未指定 --config，且 checkpoint 同目录不存在 config.yaml")


def _baseline_enabled(cfg: DictConfig) -> bool:
    value = OmegaConf.select(cfg, "baseline.enabled")
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no", "off"}:
            return False
        if normalized in {"true", "1", "yes", "on"}:
            return True
        raise ValueError(f"baseline.enabled 只能是 true/false，当前为: {value}")
    return bool(value)


def _validate_checkpoint_config(
    checkpoint_config: dict | None,
    cfg: DictConfig,
    *,
    keys: tuple[str, ...] | list[str] = (
        "data.dataset_root",
        "data.index_csv",
        "data.input_set",
        "data.val_split",
        "data.max_val_windows",
        "data.val_sample_strategy",
        "data.val_sample_seed",
        "data.stratify_column",
        "data.filter_unusable",
        "data.valid_ratio_min",
        "data.input_finite_ratio_min",
        "data.target_finite_ratio_min",
        "data.unusable_residual_classes",
        "window.target_fs",
        "window.duration_samples",
        "model.name",
        "model.in_channels",
        "model.out_channels",
        "model.base_channels",
        "loss.envelope_window_sec",
        "loss.spectrum_low_hz",
        "loss.spectrum_high_hz",
    ),
) -> None:
    """校验评价配置和 checkpoint 记录的训练配置是否一致。"""
    if checkpoint_config is None:
        raise ValueError("checkpoint 缺少训练配置，无法校验评价配置一致性")
    checkpoint_cfg = OmegaConf.create(checkpoint_config)
    mismatched: list[str] = []
    for key in keys:
        checkpoint_value = _plain_value(OmegaConf.select(checkpoint_cfg, key))
        current_value = _plain_value(OmegaConf.select(cfg, key))
        if checkpoint_value != current_value:
            mismatched.append(f"{key}: checkpoint={checkpoint_value!r} current={current_value!r}")
    if mismatched:
        details = "; ".join(mismatched)
        raise ValueError(f"checkpoint 配置与当前配置不一致: {details}")


def _plain_value(value):
    """将 OmegaConf 容器转成普通 Python 值，便于稳定比较。"""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value
