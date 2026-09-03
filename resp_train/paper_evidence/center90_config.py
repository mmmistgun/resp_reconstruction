from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


CENTER90_PROTOCOL_ID = "paper-center90-context-v1-20260904"
CENTER90_MODEL_VARIANTS = ("c201_center90", "w_reduced_center90")
CENTER90_INPUT_SAMPLES = (9000, 13500, 18000)
CENTER90_OUTPUT_SAMPLES = 9000
CENTER90_FORMAL_SEEDS = (20260811, 20260812, 20260813)
CENTER90_FORMAL_OUTPUT_ROOT = "runs/paper_evidence_v1/center90_context/formal"
CENTER90_FORMAL_ENABLED = False
_REPO_ROOT = Path(__file__).resolve().parents[2]
CENTER90_W_CACHE_PATHS = {
    9000: str(
        _REPO_ROOT
        / "runs/paper_evidence_v1/center_context_w_cache/"
        "90s_eb33cf00339545a75c459ca864bb97b333e605aa4d6c8f3d78f36b0df162587c"
    ),
    18000: str(
        _REPO_ROOT
        / "runs/paper_evidence_v1/center_context_w_cache/"
        "180s_8414c1a1dd1acc27640bc22074805aad2f327ff7d56eca2ee4ea8193ab71b827"
    ),
}
CENTER90_W_CACHE_MANIFEST_SHA256 = {
    9000: "9442a33ea2c633b15642d033efa3d3e09278f30c4692a707abc9de460c784757",
    18000: "72402d8543adf3cda504967b87fc4f9528cacca58b9aa080f0dcfe06707037fe",
}


def load_center90_config(path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"center90 配置不存在: {config_path}")
    cfg = OmegaConf.load(config_path)
    base_reference = cfg.pop("_base_", None)
    if base_reference is not None:
        base_path = (config_path.parent / str(base_reference)).resolve()
        if base_path.parent != config_path.resolve().parent or not base_path.is_file():
            raise ValueError("center90 _base_ 只允许引用同目录现有配置")
        base_cfg = OmegaConf.load(base_path)
        if "_base_" in base_cfg:
            raise ValueError("center90 配置只允许一层 _base_")
        cfg = OmegaConf.merge(base_cfg, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    validate_center90_config(cfg)
    return cfg


def validate_center90_config(cfg: DictConfig) -> None:
    required = (
        "protocol.name",
        "protocol.stage",
        "protocol.run_role",
        "protocol.execution_gate",
        "data.dataset_root",
        "data.index_csv",
        "data.access_splits",
        "window.parent_samples",
        "window.input_samples",
        "window.output_samples",
        "window.target_fs",
        "model.name",
        "model.variant",
        "model.initialization_seed",
        "training.seed",
        "training.device",
        "outputs.run_root",
    )
    for key in required:
        if OmegaConf.select(cfg, key) is None:
            raise ValueError(f"center90 配置缺少必需字段: {key}")

    frozen: dict[str, Any] = {
        "protocol.name": CENTER90_PROTOCOL_ID,
        "data.format": "research_v2",
        "data.input_set": "research_v2_waveform",
        "data.train_split": "train",
        "data.val_split": "val",
        "data.test_split": "test",
        "data.access_splits": ["train", "val"],
        "data.target_task": "waveform",
        "data.bcg_input_key": "bcg_rawish_segment_soft_z_key",
        "data.target_key": "target_waveform_segment_soft_z_key",
        "data.filter_unusable": True,
        "data.drop_nonfinite_windows": False,
        "data.train_sample_strategy": "stratified_random",
        "data.val_sample_strategy": "stratified_random",
        "data.train_sample_seed": 20260610,
        "data.val_sample_seed": 20260611,
        "data.stratify_column": "allowed_losses",
        "data.min_hard_valid_ratio": 0.80,
        "data.min_state_alignment_valid_ratio": 0.80,
        "window.target_fs": 100,
        "window.parent_samples": 18000,
        "window.output_samples": CENTER90_OUTPUT_SAMPLES,
        "window.output_sec": 90,
        "model.name": "paper_center90_context_v1",
        "loss.band_low_hz": 0.05,
        "loss.band_high_hz": 0.70,
        "loss.scale_eps": 1e-8,
        "loss.dynamic_eps": 1e-8,
        "loss.corr_eps": 1e-8,
        "loss.envelope_eps": 1e-8,
        "loss.max_lag_sec": 0.30,
        "loss.envelope_window_sec": 10,
        "loss.envelope_step_sec": 5,
        "loss.sync_weight": 1.0,
        "loss.effort_weight": 0.25,
        "evaluation.ibi_peak_distance_samples": 142,
        "evaluation.ibi_match_tolerance_sec": 0.5,
        "evaluation.ibi_coverage_threshold": 0.80,
        "evaluation.envelope_quantile_method": "linear",
        "training.optimizer": "adamw",
        "training.max_learning_rate": 3e-4,
        "training.min_learning_rate": 3e-5,
        "training.adam_betas": [0.9, 0.999],
        "training.adam_eps": 1e-8,
        "training.weight_decay": 1e-4,
        "training.warmup_fraction": 0.05,
        "training.lr_schedule": "step_exact_warmup_cosine",
        "training.grad_clip_norm": 1.0,
        "training.use_amp": True,
        "training.amp_dtype": "bfloat16",
        "training.allow_tf32": False,
        "training.cudnn_benchmark": False,
        "training.drop_last": False,
        "training.resume": False,
        "training.early_stopping_enabled": False,
    }
    for key, expected in frozen.items():
        actual = OmegaConf.select(cfg, key)
        normalized = OmegaConf.to_container(actual, resolve=True) if OmegaConf.is_config(actual) else actual
        if normalized != expected:
            raise ValueError(f"center90 冻结要求 {key}={expected!r}，当前为 {normalized!r}")

    input_samples = int(cfg.window.input_samples)
    if input_samples not in CENTER90_INPUT_SAMPLES:
        raise ValueError(f"center90 input_samples 只允许 {list(CENTER90_INPUT_SAMPLES)}")
    if int(cfg.window.input_sec) * int(cfg.window.target_fs) != input_samples:
        raise ValueError("center90 input_sec 与 input_samples 不一致")
    variant = str(cfg.model.variant)
    if variant not in CENTER90_MODEL_VARIANTS:
        raise ValueError(f"center90 model.variant 只允许 {list(CENTER90_MODEL_VARIANTS)}")
    if int(cfg.model.initialization_seed) != int(cfg.training.seed):
        raise ValueError("center90 initialization_seed 必须与 training.seed 相同")
    if cfg.data.get("max_test_windows") is not None:
        raise ValueError("center90 不允许配置 test windows")

    role = str(cfg.protocol.run_role)
    stage = str(cfg.protocol.stage)
    gate = str(cfg.protocol.execution_gate)
    device = str(cfg.training.device)
    cache_path = cfg.data.get("center_w_cache_path")
    cache_hash = cfg.data.get("center_w_cache_manifest_sha256")
    maxima = (cfg.data.get("max_train_windows"), cfg.data.get("max_val_windows"))
    if role == "implementation":
        if stage != "p0_implementation" or gate != "p0_implementation_only" or device != "cpu":
            raise ValueError("center90 implementation 固定 stage/gate/CPU")
        if variant != "c201_center90" or cache_path not in (None, "") or cache_hash not in (None, ""):
            raise ValueError("center90 implementation config 固定使用 C201-center90")
        if (
            int(cfg.training.epochs),
            int(cfg.training.batch_size),
            int(cfg.training.gradient_accumulation_steps),
        ) != (1, 2, 1):
            raise ValueError("center90 implementation 固定 1 epoch / batch 2 / accumulation 1")
        if maxima != (2, 2):
            raise ValueError("center90 implementation 只允许 2/2 synthetic contract")
    elif role == "formal":
        if not CENTER90_FORMAL_ENABLED:
            raise ValueError("center90 formal 将在 cache 与 GPU 工程回执冻结后开放")
        if stage != "formal" or gate != "formal_full_matrix" or not device.startswith("cuda:"):
            raise ValueError("center90 formal 固定 formal/full-matrix/cuda 合同")
        if int(cfg.training.seed) not in CENTER90_FORMAL_SEEDS:
            raise ValueError(f"center90 formal 只开放 seeds={list(CENTER90_FORMAL_SEEDS)}")
        if maxima != (None, None) or str(cfg.outputs.run_root) != CENTER90_FORMAL_OUTPUT_ROOT:
            raise ValueError("center90 formal 必须完整 train/validation 并使用固定输出根")
        if (
            int(cfg.training.epochs),
            int(cfg.training.batch_size),
            int(cfg.training.gradient_accumulation_steps),
        ) != (80, 128, 1):
            raise ValueError("center90 formal 固定 80 epochs / physical batch 128 / accumulation 1")
        if variant == "c201_center90":
            if cache_path not in (None, "") or cache_hash not in (None, ""):
                raise ValueError("C201-center90 formal 不读取 W cache")
        else:
            if input_samples not in CENTER90_W_CACHE_PATHS:
                raise ValueError(f"W-reduced-center90 {input_samples // 100}s cache 尚未冻结")
            if Path(str(cache_path)).resolve() != Path(CENTER90_W_CACHE_PATHS[input_samples]).resolve():
                raise ValueError("W-reduced-center90 formal cache path 漂移")
            if str(cache_hash) != CENTER90_W_CACHE_MANIFEST_SHA256[input_samples]:
                raise ValueError("W-reduced-center90 formal cache manifest SHA-256 漂移")
    else:
        raise ValueError("center90 run_role 未注册")


__all__ = [
    "CENTER90_FORMAL_OUTPUT_ROOT",
    "CENTER90_FORMAL_ENABLED",
    "CENTER90_FORMAL_SEEDS",
    "CENTER90_INPUT_SAMPLES",
    "CENTER90_MODEL_VARIANTS",
    "CENTER90_OUTPUT_SAMPLES",
    "CENTER90_PROTOCOL_ID",
    "CENTER90_W_CACHE_MANIFEST_SHA256",
    "CENTER90_W_CACHE_PATHS",
    "load_center90_config",
    "validate_center90_config",
]
