from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.config import load_config
from resp_train.data.factory import WindowDataBundle, build_tho_data, build_window_data
from resp_train.engine import collect_predictions, save_checkpoint
from resp_train.experiments.base import BaseExperiment, ExperimentData
from resp_train.experiments.selection import EPOCH_TASK_SELECTION_COLUMNS
from resp_train.losses.weak import WeakSyncLoss
from resp_train.metrics.baseline import evaluate_baseline_dataset
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.metrics.parallel import evaluate_predictions_chunked, load_or_build_target_feature_cache
from resp_train.models.registry import build_model
from resp_train.utils.run import resolve_device


class ThoExperiment(BaseExperiment):
    """THO 训练任务的语义层，负责串联任务专属组件。"""

    task_name = "tho"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._epoch_metric_records: list[dict[str, float | int]] = []
        self._best_rr_value: float | None = None
        self._best_task_key: tuple[float, ...] | None = None

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

        primary_label = self.final_checkpoint_label()
        metrics = self._evaluate_current_model(model, data, run_dir)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        metrics.to_csv(run_dir / f"metrics_{primary_label}.csv", index=False)
        summary = _format_metrics_summary(metrics)
        if summary:
            logging.getLogger("resp_train").info(summary)

        if self.epoch_metrics_enabled():
            for label, checkpoint_path in self._task_checkpoint_paths(run_dir).items():
                if label == primary_label or not checkpoint_path.exists():
                    continue
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                self._evaluate_current_model(model, data, run_dir).to_csv(
                    run_dir / f"metrics_{label}.csv",
                    index=False,
                )

    def final_checkpoint_label(self) -> str:
        label = str(self.cfg.training.get("final_checkpoint", "val_loss")).strip()
        if label not in {"val_loss", "best_rr", "best_task"}:
            raise ValueError(f"training.final_checkpoint 未知: {label}")
        return label

    def final_checkpoint_path(self, run_dir: Path) -> Path:
        label = self.final_checkpoint_label()
        checkpoint_path = self._task_checkpoint_paths(run_dir)[label]
        if not checkpoint_path.exists():
            raise ValueError(f"最终评价 checkpoint 不存在: {checkpoint_path}")
        return checkpoint_path

    def _task_checkpoint_paths(self, run_dir: Path) -> dict[str, Path]:
        return {
            "val_loss": run_dir / "checkpoint.pt",
            "best_rr": run_dir / "checkpoint_best_rr.pt",
            "best_task": run_dir / "checkpoint_best_task.pt",
        }

    def _evaluate_current_model(self, model: torch.nn.Module, data: ExperimentData, run_dir: Path) -> pd.DataFrame:
        tho_data = data.extras["tho_data"]
        eval_preds = collect_predictions(
            model,
            tho_data.val.loader,
            device=self.device,
            max_windows=len(tho_data.val.dataset),
        )
        show_progress = self._friendly_output_enabled(self._resolve_show_progress())
        if self.epoch_metrics_enabled():
            options = self._epoch_metrics_options()
            target_features = load_or_build_target_feature_cache(
                eval_preds,
                self.cfg,
                cache_dir=run_dir / "target_feature_cache",
                show_progress=show_progress,
                target_workers=options["target_workers"],
                target_chunk_size=options["target_chunk_size"],
            )
            metrics = evaluate_predictions_chunked(
                eval_preds,
                self.cfg,
                method=str(self.cfg.model.name),
                metrics_workers=options["metrics_workers"],
                metrics_chunk_size=options["metrics_chunk_size"],
                target_features=target_features,
                show_progress=show_progress,
            )
        else:
            metrics = evaluate_prediction_dict(
                eval_preds,
                self.cfg,
                method=str(self.cfg.model.name),
                show_progress=show_progress,
            )
        return metrics

    def epoch_metrics_enabled(self) -> bool:
        metrics_cfg = self.cfg.training.get("epoch_metrics", {})
        if not metrics_cfg:
            return False
        return bool(metrics_cfg.get("enabled", False))

    def evaluate_epoch_metrics(
        self,
        *,
        predictions: dict[str, np.ndarray],
        run_dir: Path,
        epoch: int,
        show_progress: bool,
    ) -> dict[str, float]:
        options = self._epoch_metrics_options()
        target_cache_dir = run_dir / "target_feature_cache"
        target_features = load_or_build_target_feature_cache(
            predictions,
            self.cfg,
            cache_dir=target_cache_dir,
            show_progress=show_progress,
            target_workers=options["target_workers"],
            target_chunk_size=options["target_chunk_size"],
        )
        metrics = evaluate_predictions_chunked(
            predictions,
            self.cfg,
            method=str(self.cfg.model.name),
            metrics_workers=options["metrics_workers"],
            metrics_chunk_size=options["metrics_chunk_size"],
            target_features=target_features,
            show_progress=show_progress,
        )
        summary = _summarize_epoch_metrics(metrics)
        row = {"epoch": int(epoch), **summary}
        self._epoch_metric_records.append(row)
        pd.DataFrame(self._epoch_metric_records).to_csv(run_dir / "epoch_metrics.csv", index=False)
        return {f"val_{key}": value for key, value in summary.items()}

    def update_task_checkpoints(
        self,
        *,
        run_dir: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        record: dict[str, object],
    ) -> None:
        if not self.epoch_metrics_enabled():
            return
        if not bool(record.get("checkpoint_gate_passed", True)):
            return
        rr_value = _finite_float(record.get("val_rr_peak_band_robust_abs_error_mean"))
        if rr_value is not None and (self._best_rr_value is None or rr_value < self._best_rr_value):
            self._best_rr_value = rr_value
            save_checkpoint(
                run_dir / "checkpoint_best_rr.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=record,
                cfg=self.cfg,
            )
        task_key = _task_selection_key(record)
        if task_key is not None and (self._best_task_key is None or task_key < self._best_task_key):
            self._best_task_key = task_key
            save_checkpoint(
                run_dir / "checkpoint_best_task.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=record,
                cfg=self.cfg,
            )

    def _epoch_metrics_options(self) -> dict[str, int]:
        metrics_cfg = self.cfg.training.get("epoch_metrics", {})
        metrics_workers = _resolve_epoch_metric_workers(metrics_cfg.get("metrics_workers", 32), metrics_cfg)
        metrics_chunk_size = int(metrics_cfg.get("metrics_chunk_size", 64))
        return {
            "metrics_workers": metrics_workers,
            "metrics_chunk_size": metrics_chunk_size,
            "target_workers": _resolve_epoch_metric_workers(
                metrics_cfg.get("target_workers", metrics_workers),
                metrics_cfg,
            ),
            "target_chunk_size": int(metrics_cfg.get("target_chunk_size", metrics_chunk_size)),
        }

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        metrics_output: Path | None,
        metrics_workers: int = 1,
        metrics_chunk_size: int = 128,
        target_workers: int = 1,
        target_chunk_size: int = 128,
        target_cache_dir: str | Path | None = None,
    ) -> None:
        device = self.device or resolve_device(str(self.cfg.training.device))
        model = self.build_model().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        _validate_checkpoint_config(checkpoint.get("config"), self.cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        if metrics_output is not None:
            metrics_output.parent.mkdir(parents=True, exist_ok=True)
            val_data = self._build_checkpoint_eval_val_data()
            eval_preds = collect_predictions(
                model,
                val_data.loader,
                device=device,
                max_windows=len(val_data.dataset),
            )
            show_progress = self._friendly_output_enabled(self._resolve_show_progress())
            target_features = load_or_build_target_feature_cache(
                eval_preds,
                self.cfg,
                cache_dir=target_cache_dir,
                show_progress=show_progress,
                target_workers=int(target_workers),
                target_chunk_size=int(target_chunk_size),
            )
            evaluate_predictions_chunked(
                eval_preds,
                self.cfg,
                method=str(self.cfg.model.name),
                metrics_workers=int(metrics_workers),
                metrics_chunk_size=int(metrics_chunk_size),
                target_features=target_features,
                show_progress=show_progress,
            ).to_csv(
                metrics_output,
                index=False,
            )

    def _build_checkpoint_eval_val_data(self) -> WindowDataBundle:
        """checkpoint 复评只需要 val，避免预加载 train 窗口造成额外 I/O 和内存峰。"""

        return build_window_data(
            self.cfg,
            split=str(self.cfg.data.val_split),
            max_windows=self.cfg.data.get("max_val_windows"),
            sample_strategy=str(self.cfg.data.val_sample_strategy),
            sample_seed=int(self.cfg.data.val_sample_seed),
            shuffle=False,
        )


def evaluate_tho_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    metrics_output_path: str | Path | None,
    overrides: list[str] | None = None,
    metrics_workers: int = 1,
    metrics_chunk_size: int = 128,
    target_workers: int = 1,
    target_chunk_size: int = 128,
    target_cache_dir: str | Path | None = None,
) -> Path:
    resolved_checkpoint = Path(checkpoint_path)
    resolved_config = _resolve_config_path(config_path, resolved_checkpoint)
    cfg = load_config(resolved_config, overrides=overrides)
    experiment = ThoExperiment(cfg)
    experiment.evaluate_checkpoint(
        resolved_checkpoint,
        metrics_output=Path(metrics_output_path) if metrics_output_path else None,
        metrics_workers=int(metrics_workers),
        metrics_chunk_size=int(metrics_chunk_size),
        target_workers=int(target_workers),
        target_chunk_size=int(target_chunk_size),
        target_cache_dir=target_cache_dir,
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


def _format_metrics_summary(metrics: pd.DataFrame) -> str:
    """训练结束后输出当前两个主任务护栏。"""
    rr_column = "rr_peak_band_robust_abs_error"
    count_column = "breath_count_zero_cross_abs_error"
    if rr_column not in metrics or count_column not in metrics:
        return ""
    rr_values = pd.to_numeric(metrics[rr_column], errors="coerce").dropna()
    count_values = pd.to_numeric(metrics[count_column], errors="coerce").dropna()
    if rr_values.empty or count_values.empty:
        return f"metrics: n={len(metrics)} {rr_column}=nan {count_column}=nan"
    return (
        f"metrics: n={len(metrics)} {rr_column} "
        f"mean={float(rr_values.mean()):.6f} "
        f"median={float(rr_values.median()):.6f} "
        f"p95={float(rr_values.quantile(0.95)):.6f} "
        f"frac_gt_1={float(np.mean(rr_values.to_numpy() > 1.0)):.6f} "
        f"{count_column} mean={float(count_values.mean()):.6f}"
    )


EPOCH_SUMMARY_METRICS = [
    "rr_peak_band_abs_error",
    "rr_peak_band_robust_abs_error",
    "rr_spec_abs_error",
    "breath_count_zero_cross_abs_error",
    "relative_envelope_mae",
    "relative_envelope_corr",
    "relative_envelope_corr_lag4s",
    "relative_envelope_mae_lag4s",
    "spectrum_similarity",
    "band_limited_corr",
    "best_lag_corr",
    "best_lag_corr_4s",
    "local_rr_mae",
    "local_rr_corr",
    "local_rr_valid_frac",
]


def _summarize_epoch_metrics(metrics: pd.DataFrame) -> dict[str, float | int]:
    record: dict[str, float | int] = {"n_windows": int(len(metrics))}
    for metric in EPOCH_SUMMARY_METRICS:
        values = pd.to_numeric(metrics.get(metric, pd.Series(dtype=float)), errors="coerce")
        record[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
        record[f"{metric}_median"] = float(values.median()) if not values.empty else float("nan")
    peak = pd.to_numeric(metrics.get("rr_peak_band_abs_error", pd.Series(dtype=float)), errors="coerce").dropna()
    for threshold in (1.0, 2.0):
        record[f"frac_gt_{threshold:g}"] = float((peak > threshold).mean()) if not peak.empty else float("nan")
    return record


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _task_selection_key(record: dict[str, object]) -> tuple[float, ...] | None:
    values: list[float] = []
    for column in EPOCH_TASK_SELECTION_COLUMNS:
        value = _finite_float(record.get(column))
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _resolve_epoch_metric_workers(value: object, metrics_cfg: object) -> int:
    if str(value).strip().lower() != "auto":
        return max(1, int(value))
    max_workers = int(metrics_cfg.get("auto_max_workers", 32))
    max_parallel = max(1, int(os.environ.get("RESP_TRAIN_MAX_PARALLEL", "1") or 1))
    cpu_count = os.cpu_count() or max_workers
    return max(1, min(max_workers, int(cpu_count) // max_parallel))


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
        "data.min_hard_valid_ratio",
        "data.min_state_alignment_valid_ratio",
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
        "model.time_backbone",
        "model.mixer_layers",
        "model.patch_len",
        "model.patch_stride",
        "model.overlap_window",
        "model.output_smoothing_kernel",
        "model.branch_mode",
        "model.fusion_mode",
        "model.fuse_len",
        "model.fusion_decoder",
        "model.stft_inject_position",
        "model.stft_win",
        "model.stft_hop",
        "model.stft_low_hz",
        "model.stft_high_hz",
        "model.stft_out_channels",
        "model.stft_norm",
        "model.stft_encoder_type",
        "loss.envelope_window_sec",
        "loss.spectrum_low_hz",
        "loss.spectrum_high_hz",
        "loss.stft_win_length",
        "loss.stft_hop_length",
        "loss.stft_n_fft",
        "loss.stft_center",
        "loss.stft_dist_low_hz",
        "loss.stft_dist_high_hz",
        "evaluation.max_lag_sec",
        "evaluation.lag_bandpass_order",
        "evaluation.raw_peak_min_good_segment_sec",
        "evaluation.local_rr_window_sec",
        "evaluation.local_rr_step_sec",
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
