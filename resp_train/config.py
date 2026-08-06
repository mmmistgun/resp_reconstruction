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


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    *,
    allow_legacy_envelope_metric_migration: bool = False,
) -> "DictConfig":
    from omegaconf import OmegaConf

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if allow_legacy_envelope_metric_migration:
        _migrate_legacy_envelope_metric_config(cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    _validate_config(cfg)
    return cfg


def _migrate_legacy_envelope_metric_config(cfg: Any) -> None:
    """只为旧 checkpoint sidecar 补入本次重评所需、不会影响训练 forward/loss 的冻结字段。"""

    from omegaconf import OmegaConf

    defaults = {
        "evaluation.envelope_quantile_method": "linear",
        "evaluation.envelope_strata_low": 0.30875308839006915,
        "evaluation.envelope_strata_high": 0.7031542121234101,
    }
    for key, value in defaults.items():
        if OmegaConf.select(cfg, key) is None:
            OmegaConf.update(cfg, key, value, merge=False)


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
        "loss.envelope_eps",
        "loss.max_lag_sec",
        "loss.envelope_window_sec",
        "loss.envelope_step_sec",
        "loss.sync_weight",
        "loss.effort_weight",
        "evaluation.local_rr_window_sec",
        "evaluation.local_rr_step_sec",
        "evaluation.ibi_peak_distance_samples",
        "evaluation.ibi_match_tolerance_sec",
        "evaluation.ibi_coverage_threshold",
        "evaluation.ndtw_fs",
        "evaluation.ndtw_radius_sec",
        "evaluation.envelope_quantile_method",
        "evaluation.envelope_strata_low",
        "evaluation.envelope_strata_high",
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

    model_name = str(OmegaConf.select(cfg, "model.name"))
    allowed_models = {
        "patch_mixer1d",
        "multiscale_patch_mixer1d",
        "time_stft_fusion1d",
        "time_stft_dual1d",
    }
    if model_name not in allowed_models:
        raise ValueError(f"当前实验只允许模型 {sorted(allowed_models)}，当前为: {model_name}")
    if model_name == "multiscale_patch_mixer1d":
        expected_multiscale = {
            "model.base_channels": 1,
            "model.patch_lengths": [256, 512, 1024, 2048],
            "model.patch_stride_ratio": 0.5,
            "model.mixer_layers": 2,
        }
        for key, expected in expected_multiscale.items():
            value = OmegaConf.select(cfg, key)
            normalized = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
            if normalized != expected:
                raise ValueError(f"M1 冻结要求 {key}={expected}，当前为 {value}")
    if model_name == "time_stft_fusion1d":
        expected_t1 = {
            "model.in_channels": 1,
            "model.out_channels": 1,
            "model.base_channels": 16,
            "model.patch_len": 256,
            "model.patch_stride": 128,
            "model.mixer_layers": 2,
            "model.overlap_window": "hann",
            "model.output_smoothing_kernel": 1,
            "model.stft_window_sec": 30,
            "model.stft_step_sec": 10,
            "model.stft_channels": 16,
            "model.stft_feature_eps": 1e-8,
        }
        for key, expected in expected_t1.items():
            value = OmegaConf.select(cfg, key)
            if value != expected:
                raise ValueError(f"T1 冻结要求 {key}={expected}，当前为 {value}")
    if model_name == "time_stft_dual1d":
        variant = str(OmegaConf.select(cfg, "model.experiment_variant"))
        common = {
            "model.name": "time_stft_dual1d",
            "model.experiment_variant": variant,
            "model.in_channels": 1,
            "model.out_channels": 1,
            "model.base_channels": 16,
            "model.time_backbone": "patch_mixer1d",
            "model.patch_len": 256,
            "model.patch_stride": 128,
            "model.mixer_layers": 2,
            "model.overlap_window": "hann",
            "model.output_smoothing_kernel": 1,
            "model.branch_mode": "dual",
            "model.stft_low_hz": 0.05,
            "model.stft_high_hz": 8.0,
            "model.stft_out_channels": 16,
            "model.stft_norm": "n0",
        }
        variants = {
            "t2_g3c_wide_native": {
                "model.stft_win": 2000,
                "model.stft_hop": 250,
                "model.stft_encoder_type": "conv2d",
                "model.fusion_mode": "native_inject",
                "model.stft_inject_position": "pre_mixer",
            },
            "t3_e3a0_wide_concat": {
                "model.stft_win": 3000,
                "model.stft_hop": 500,
                "model.stft_encoder_type": "conv2d",
                "model.fusion_mode": "concat_generic",
                "model.fuse_len": 600,
                "model.fusion_decoder": "deep",
            },
            "t4_g3c_bandenergy_native": {
                "model.stft_win": 2000,
                "model.stft_hop": 250,
                "model.stft_encoder_type": "bandenergy",
                "model.fusion_mode": "native_inject",
                "model.stft_inject_position": "pre_mixer",
            },
        }
        if variant not in variants:
            raise ValueError(f"当前时频实验不允许 variant={variant!r}")
        expected_variant = {**common, **variants[variant]}
        for key, expected in expected_variant.items():
            value = OmegaConf.select(cfg, key)
            if value != expected:
                raise ValueError(f"{variant} 冻结要求 {key}={expected}，当前为 {value}")
        expected_model_keys = {key.removeprefix("model.") for key in expected_variant}
        extra_model_keys = set(cfg.model.keys()) - expected_model_keys
        if extra_model_keys:
            raise ValueError(
                f"{variant} 不接受未冻结模型字段: {sorted(extra_model_keys)}"
            )

    lr_scheduler = OmegaConf.select(cfg, "training.lr_scheduler")
    if lr_scheduler != "none":
        raise ValueError("当前冻结协议不使用 learning-rate scheduler")

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
        "loss.envelope_eps": 1e-8,
        "loss.envelope_window_sec": 10,
        "loss.envelope_step_sec": 5,
        "loss.sync_weight": 1.0,
        "loss.effort_weight": 0.25,
        "evaluation.local_rr_window_sec": 60,
        "evaluation.local_rr_step_sec": 15,
        "evaluation.ibi_peak_distance_samples": 142,
        "evaluation.ibi_match_tolerance_sec": 0.5,
        "evaluation.ibi_coverage_threshold": 0.8,
        "evaluation.ndtw_fs": 10,
        "evaluation.ndtw_radius_sec": 0.3,
        "evaluation.envelope_strata_low": 0.30875308839006915,
        "evaluation.envelope_strata_high": 0.7031542121234101,
    }
    for key, expected in frozen_values.items():
        value = OmegaConf.select(cfg, key)
        if float(value) != float(expected):
            raise ValueError(f"当前冻结协议要求 {key}={expected}，当前为 {value}")
    if str(OmegaConf.select(cfg, "evaluation.envelope_quantile_method")) != "linear":
        raise ValueError("当前冻结协议要求 evaluation.envelope_quantile_method=linear")

    if OmegaConf.select(cfg, "training.grad_clip_norm") is not None:
        raise ValueError("当前冻结协议不使用 gradient clipping")
    if bool(OmegaConf.select(cfg, "training.use_amp")):
        raise ValueError("当前冻结协议要求 training.use_amp=false")
