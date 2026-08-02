from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from omegaconf import DictConfig


REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "tqdm",
    "omegaconf",
]


def check_required_packages(packages: Iterable[str] = REQUIRED_PACKAGES) -> list[str]:
    missing: list[str] = []
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    return missing


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> "DictConfig":
    from omegaconf import OmegaConf

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Any) -> None:
    from omegaconf import OmegaConf

    required = [
        "data.dataset_root",
        "data.index_csv",
        "data.input_set",
        "data.train_sample_strategy",
        "data.val_sample_strategy",
        "data.test_sample_strategy",
        "data.train_sample_seed",
        "data.val_sample_seed",
        "data.test_sample_seed",
        "data.stratify_column",
        "window.target_fs",
        "window.duration_samples",
        "model.name",
        "loss.band_low_hz",
        "loss.band_high_hz",
        "loss.scale_eps",
        "loss.dynamic_eps",
        "loss.corr_eps",
        "loss.power_eps",
        "loss.envelope_eps",
        "loss.max_lag_sec",
        "loss.global_rhythm_window_sec",
        "loss.local_rhythm_window_sec",
        "loss.local_rhythm_hop_sec",
        "loss.envelope_window_sec",
        "loss.envelope_step_sec",
        "loss.sync_weight",
        "loss.rhythm_weight",
        "loss.effort_weight",
        "loss.pol_start_weight",
        "loss.pol_fraction",
        "loss.smooth_l1_beta",
        "evaluation.local_rr_window_sec",
        "evaluation.local_rr_step_sec",
        "evaluation.ibi_peak_distance_samples",
        "evaluation.ibi_match_tolerance_sec",
        "evaluation.ibi_coverage_threshold",
        "evaluation.ndtw_fs",
        "evaluation.ndtw_radius_sec",
        "training.epochs",
        "training.batch_size",
        "training.learning_rate",
        "training.seed",
        "training.device",
        "training.lr_scheduler",
        "training.use_amp",
        "outputs.run_root",
    ]
    for key in required:
        if OmegaConf.select(cfg, key) is None:
            raise ValueError(f"配置缺少必需字段: {key}")

    sample_strategies = {"head", "random", "stratified_random"}
    for key in ("data.train_sample_strategy", "data.val_sample_strategy", "data.test_sample_strategy"):
        value = OmegaConf.select(cfg, key)
        if value not in sample_strategies:
            raise ValueError(f"{key} 必须是 {sorted(sample_strategies)} 之一，当前为: {value}")

    lr_scheduler = OmegaConf.select(cfg, "training.lr_scheduler")
    if lr_scheduler != "none":
        raise ValueError("当前冻结 baseline 不使用 learning-rate scheduler")

    if int(OmegaConf.select(cfg, "window.target_fs")) != 100:
        raise ValueError("当前冻结协议要求 window.target_fs=100")
    if int(OmegaConf.select(cfg, "window.duration_samples")) != 18000:
        raise ValueError("当前冻结协议要求 window.duration_samples=18000")
    if float(OmegaConf.select(cfg, "loss.band_low_hz")) != 0.05 or float(
        OmegaConf.select(cfg, "loss.band_high_hz")
    ) != 0.70:
        raise ValueError("当前冻结协议要求统一频带 0.05–0.70 Hz")
    if float(OmegaConf.select(cfg, "loss.max_lag_sec")) != 0.30:
        raise ValueError("当前冻结协议要求 loss.max_lag_sec=0.30")
    frozen_values = {
        "loss.scale_eps": 1e-8,
        "loss.dynamic_eps": 1e-8,
        "loss.corr_eps": 1e-8,
        "loss.power_eps": 1e-8,
        "loss.envelope_eps": 1e-8,
        "loss.global_rhythm_window_sec": 180,
        "loss.local_rhythm_window_sec": 30,
        "loss.local_rhythm_hop_sec": 10,
        "loss.envelope_window_sec": 10,
        "loss.envelope_step_sec": 5,
        "loss.sync_weight": 1.0,
        "loss.rhythm_weight": 0.5,
        "loss.effort_weight": 0.25,
        "loss.pol_start_weight": 0.05,
        "loss.pol_fraction": 0.15,
        "loss.smooth_l1_beta": 0.5,
        "evaluation.local_rr_window_sec": 60,
        "evaluation.local_rr_step_sec": 15,
        "evaluation.ibi_peak_distance_samples": 142,
        "evaluation.ibi_match_tolerance_sec": 0.5,
        "evaluation.ibi_coverage_threshold": 0.8,
        "evaluation.ndtw_fs": 10,
        "evaluation.ndtw_radius_sec": 0.3,
    }
    for key, expected in frozen_values.items():
        value = OmegaConf.select(cfg, key)
        if float(value) != float(expected):
            raise ValueError(f"当前冻结协议要求 {key}={expected}，当前为 {value}")
    if OmegaConf.select(cfg, "training.grad_clip_norm") is not None:
        raise ValueError("当前冻结 baseline 不使用 gradient clipping")
    if bool(OmegaConf.select(cfg, "training.use_amp")):
        raise ValueError("当前冻结协议要求 training.use_amp=false")
