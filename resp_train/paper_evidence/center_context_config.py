from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


CENTER_CONTEXT_PROTOCOL_ID = "paper-center-context-v1-20260901"
CENTER_CONTEXT_MODEL_VARIANTS = ("c201_center60", "w_reduced_center60")
CENTER_CONTEXT_INPUT_SAMPLES = (6000, 9000, 18000)
CENTER_CONTEXT_FORMAL_SEEDS = (20260811, 20260812, 20260813)
_REPO_ROOT = Path(__file__).resolve().parents[2]
CENTER_CONTEXT_P3_OUTPUT_ROOT = "runs/paper_evidence_v1/center_context/p3_single_seed"
CENTER_CONTEXT_P3_W_CACHE_PATHS = {
    6000: str(
        _REPO_ROOT
        / "runs/paper_evidence_v1/center_context_w_cache/"
        "60s_215c24b05b2e438f311edf13d20903d617131378a952b893234582c670848c6f"
    ),
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
CENTER_CONTEXT_P3_W_CACHE_MANIFEST_SHA256 = {
    6000: "e9d270c930d6862f9b5a9cbdb3c26765fe3b57d2573946640cba99e597389793",
    9000: "9442a33ea2c633b15642d033efa3d3e09278f30c4692a707abc9de460c784757",
    18000: "72402d8543adf3cda504967b87fc4f9528cacca58b9aa080f0dcfe06707037fe",
}


def load_center_context_config(path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"中心上下文配置不存在: {config_path}")
    cfg = OmegaConf.load(config_path)
    base_reference = cfg.pop("_base_", None)
    if base_reference is not None:
        base_path = (config_path.parent / str(base_reference)).resolve()
        if base_path.parent != config_path.resolve().parent:
            raise ValueError("中心上下文 _base_ 只允许引用同目录配置")
        if not base_path.is_file():
            raise FileNotFoundError(f"中心上下文 base 配置不存在: {base_path}")
        base_cfg = OmegaConf.load(base_path)
        if "_base_" in base_cfg:
            raise ValueError("中心上下文配置只允许一层 _base_，禁止递归继承")
        cfg = OmegaConf.merge(base_cfg, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    validate_center_context_config(cfg)
    return cfg


def validate_center_context_config(cfg: DictConfig) -> None:
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
            raise ValueError(f"中心上下文配置缺少必需字段: {key}")

    frozen: dict[str, Any] = {
        "protocol.name": CENTER_CONTEXT_PROTOCOL_ID,
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
        "window.output_samples": 6000,
        "window.output_sec": 60,
        "model.name": "paper_center_context_v1",
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
        "training.drop_last": False,
        "training.resume": False,
        "training.early_stopping_enabled": False,
    }
    for key, expected in frozen.items():
        actual = OmegaConf.select(cfg, key)
        normalized = OmegaConf.to_container(actual, resolve=True) if OmegaConf.is_config(actual) else actual
        if normalized != expected:
            raise ValueError(f"中心上下文冻结要求 {key}={expected!r}，当前为 {normalized!r}")

    input_samples = int(cfg.window.input_samples)
    if input_samples not in CENTER_CONTEXT_INPUT_SAMPLES:
        raise ValueError(f"window.input_samples 只允许 {list(CENTER_CONTEXT_INPUT_SAMPLES)}")
    if int(cfg.window.input_sec) * int(cfg.window.target_fs) != input_samples:
        raise ValueError("window.input_sec 与 input_samples 不一致")
    variant = str(cfg.model.variant)
    if variant not in CENTER_CONTEXT_MODEL_VARIANTS:
        raise ValueError(f"model.variant 只允许 {list(CENTER_CONTEXT_MODEL_VARIANTS)}")
    if int(cfg.model.initialization_seed) != int(cfg.training.seed):
        raise ValueError("model.initialization_seed 必须与 training.seed 相同")

    role = str(cfg.protocol.run_role)
    if bool(cfg.protocol.get("template_only", False)):
        raise ValueError("中心上下文 template_only 配置不得直接执行")
    cache_path = cfg.data.get("center_w_cache_path")
    cache_manifest_sha256 = cfg.data.get("center_w_cache_manifest_sha256")
    if variant == "w_reduced_center60" and not cache_path:
        raise ValueError("W-reduced-center60 必须显式提供 length-specific center_w_cache_path")
    if variant == "c201_center60" and (
        cache_path not in (None, "") or cache_manifest_sha256 not in (None, "")
    ):
        raise ValueError("C201-center60 不得读取 W cache")
    if role == "formal" and variant == "w_reduced_center60":
        expected_path = Path(CENTER_CONTEXT_P3_W_CACHE_PATHS[input_samples]).resolve()
        if Path(str(cache_path)).resolve() != expected_path:
            raise ValueError("P3 W-reduced 必须使用冻结的 length-specific W cache path")
        expected_sha256 = CENTER_CONTEXT_P3_W_CACHE_MANIFEST_SHA256[input_samples]
        if str(cache_manifest_sha256) != expected_sha256:
            raise ValueError("P3 W-reduced cache manifest SHA-256 漂移")
    if cfg.data.get("max_test_windows") is not None:
        raise ValueError("中心上下文任务不允许配置 test windows")

    gate = str(cfg.protocol.execution_gate)
    device = str(cfg.training.device)
    maxima = (cfg.data.get("max_train_windows"), cfg.data.get("max_val_windows"))
    if role == "implementation":
        if str(cfg.protocol.stage) != "p1_center_context":
            raise ValueError("P1 implementation stage 漂移")
        if gate != "p1_implementation_only" or device != "cpu":
            raise ValueError("P1 implementation 配置固定为 p1_implementation_only + CPU")
        if (int(cfg.training.epochs), int(cfg.training.batch_size), int(cfg.training.gradient_accumulation_steps)) != (1, 2, 1):
            raise ValueError("P1 implementation 配置固定 1 epoch / batch 2 / accumulation 1")
        if maxima != (2, 2):
            raise ValueError("P1 implementation 配置只允许 2/2 合成契约占位，不授权真实 lifecycle")
    elif role == "formal":
        if str(cfg.protocol.stage) != "p3_single_seed":
            raise ValueError("P3 formal stage 漂移")
        if gate != "p3_formal" or not device.startswith("cuda:"):
            raise ValueError("formal 必须使用 p3_formal + 显式 cuda:<index>")
        if maxima != (None, None):
            raise ValueError("formal 必须使用完整 train/validation")
        if (int(cfg.training.epochs), int(cfg.training.batch_size)) != (80, 128):
            raise ValueError("formal 固定 80 epochs / physical batch 128")
        if int(cfg.training.gradient_accumulation_steps) != 1:
            raise ValueError("当前冻结 formal 固定 accumulation=1；batch fallback 尚未触发")
        if cfg.training.get("allow_tf32") is not False:
            raise ValueError("P3 formal 固定 allow_tf32=false")
        if cfg.training.get("cudnn_benchmark") is not False:
            raise ValueError("P3 formal 固定 cudnn_benchmark=false")
        if int(cfg.training.seed) != CENTER_CONTEXT_FORMAL_SEEDS[0]:
            raise ValueError("P3 formal 只开放 seed=20260811")
        if str(cfg.outputs.run_root) != CENTER_CONTEXT_P3_OUTPUT_ROOT:
            raise ValueError(f"P3 formal outputs.run_root 必须为 {CENTER_CONTEXT_P3_OUTPUT_ROOT}")
    else:
        raise ValueError("protocol.run_role 未注册")


__all__ = [
    "CENTER_CONTEXT_FORMAL_SEEDS",
    "CENTER_CONTEXT_INPUT_SAMPLES",
    "CENTER_CONTEXT_MODEL_VARIANTS",
    "CENTER_CONTEXT_PROTOCOL_ID",
    "CENTER_CONTEXT_P3_OUTPUT_ROOT",
    "CENTER_CONTEXT_P3_W_CACHE_MANIFEST_SHA256",
    "CENTER_CONTEXT_P3_W_CACHE_PATHS",
    "load_center_context_config",
    "validate_center_context_config",
]
