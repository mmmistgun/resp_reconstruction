from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from resp_train.temporal.model import RTM_VARIANT_SPECS


RTM_PROTOCOL_VERSION = "resp-temporal-v1-validation-20260820"
FORMAL_SEEDS = (20260811, 20260812, 20260813)
SIGNAL_LOCK_SHA256 = "11bfcad00f4532d4bdfe1413a375b5f06f46eb8ac67dfcd475701872322fee69"
CANDIDATE_LOCK_SHA256 = "b4a2c83310fa2ce9519e3ca25814aea0b179458ab52d6380a932545c99c25f9b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resp_temporal_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"RTM-v1 配置不存在: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    base_reference = cfg.pop("_base_", None)
    if base_reference is not None:
        base_path = (cfg_path.parent / str(base_reference)).resolve()
        if base_path.parent != cfg_path.resolve().parent:
            raise ValueError("RTM-v1 _base_ 只允许引用同目录配置")
        if not base_path.is_file():
            raise FileNotFoundError(f"RTM-v1 base 配置不存在: {base_path}")
        base_cfg = OmegaConf.load(base_path)
        if "_base_" in base_cfg:
            raise ValueError("RTM-v1 配置只允许一层 _base_，禁止递归继承")
        cfg = OmegaConf.merge(base_cfg, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    _validate_resp_temporal_config(cfg)
    return cfg


def _assert_equal(cfg: DictConfig, key: str, expected: Any) -> None:
    value = OmegaConf.select(cfg, key)
    normalized = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if normalized != expected:
        raise ValueError(f"RTM-v1 冻结要求 {key}={expected!r}，当前为 {normalized!r}")


def _validate_lock_files(cfg: DictConfig) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for prefix, expected_sha in (
        ("locks.signal_substrate", SIGNAL_LOCK_SHA256),
        ("locks.candidate", CANDIDATE_LOCK_SHA256),
    ):
        relative = Path(str(OmegaConf.select(cfg, f"{prefix}.path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{prefix}.path 必须是仓库内相对路径")
        lock_path = repo_root / relative
        if not lock_path.is_file():
            raise FileNotFoundError(f"RTM-v1 lock 不存在: {lock_path}")
        actual_sha = _sha256(lock_path)
        if actual_sha != expected_sha:
            raise ValueError(f"RTM-v1 lock hash 漂移: {relative}: {actual_sha} != {expected_sha}")


def _validate_resp_temporal_config(cfg: DictConfig) -> None:
    variant = str(OmegaConf.select(cfg, "model.variant", default="")).strip().lower()
    if variant not in RTM_VARIANT_SPECS:
        raise ValueError(f"未知 RTM-v1 variant={variant!r}；可选 {list(RTM_VARIANT_SPECS)}")
    spec = RTM_VARIANT_SPECS[variant]
    expected = {
        "protocol.name": RTM_PROTOCOL_VERSION,
        "protocol.stage": "exact_cpu_implementation",
        "protocol.run_role": "implementation_only",
        "protocol.exact_implementation_enabled": True,
        "protocol.gpu_engineering_enabled": False,
        "protocol.formal_training_enabled": False,
        "protocol.validation_evaluation_enabled": False,
        "protocol.research_test_enabled": False,
        "locks.signal_substrate.path": "docs/experiments/resp_temporal_v1_signal_substrate_lock_20260820.json",
        "locks.signal_substrate.sha256": SIGNAL_LOCK_SHA256,
        "locks.candidate.path": "docs/experiments/resp_temporal_v1_candidate_lock_20260820.json",
        "locks.candidate.sha256": CANDIDATE_LOCK_SHA256,
        "data.format": "research_v2",
        "data.dataset_root": (
            "/mnt/disk_code/marques/resp_prepare/dataset/"
            "20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf"
        ),
        "data.index_csv": "training/dataset_index.csv",
        "data.input_set": "research_v2_waveform",
        "data.train_split": "train",
        "data.val_split": "val",
        "data.target_task": "waveform",
        "data.bcg_input_key": "bcg_rawish_segment_soft_z_key",
        "data.target_key": "target_waveform_segment_soft_z_key",
        "data.max_train_windows": None,
        "data.max_val_windows": None,
        "data.filter_unusable": True,
        "data.drop_nonfinite_windows": False,
        "data.preload_windows": True,
        "data.train_sample_strategy": "stratified_random",
        "data.val_sample_strategy": "stratified_random",
        "data.train_sample_seed": 20260610,
        "data.val_sample_seed": 20260611,
        "data.stratify_column": "allowed_losses",
        "data.min_hard_valid_ratio": 0.80,
        "data.min_state_alignment_valid_ratio": 0.80,
        "window.input_fs_hz": 100,
        "window.duration_samples": 18000,
        "window.duration_sec": 180,
        "substrate.order": [
            "fixed_anti_alias_100_to_20",
            "shared_learned_filtering_and_nonlinearity_at_20",
            "fixed_anti_alias_20_to_10",
        ],
        "substrate.anti_alias.window": "kaiser",
        "substrate.anti_alias.beta": 8.6,
        "substrate.anti_alias.padtype": "line",
        "substrate.anti_alias.stage_100_to_20": {
            "input_fs_hz": 100.0, "up": 1, "down": 5, "numtaps": 255, "cutoff_hz": 9.0,
        },
        "substrate.anti_alias.stage_20_to_10": {
            "input_fs_hz": 20.0, "up": 1, "down": 2, "numtaps": 127, "cutoff_hz": 4.5,
        },
        "substrate.audit_equivalence.float64_atol": 5e-12,
        "substrate.audit_equivalence.float64_rtol": 5e-12,
        "substrate.explicit_analytic_envelope_input_branch": False,
        "substrate.fixed_equal_rms_proxy_input": False,
        "multiscale.grids_hz": [10.0, 2.0, 1.0],
        "multiscale.downsampling": "reflect_fir_block_center_decimation",
        "multiscale.average_pooling_allowed": False,
        "multiscale.branch_waveform_heads_allowed": False,
        "multiscale.stage_10_to_2": {
            "input_fs_hz": 10.0, "up": 1, "down": 5, "numtaps": 255, "cutoff_hz": 0.85,
        },
        "multiscale.stage_10_to_1": {
            "input_fs_hz": 10.0, "up": 1, "down": 10, "numtaps": 255, "cutoff_hz": 0.45,
        },
        "model.name": "resp_temporal_v1",
        "model.variant": variant,
        "model.family": spec.family,
        "model.role": spec.role,
        "model.expected_trainable_parameters": spec.expected_trainable_parameters,
        "model.latent_hz": 10,
        "model.latent_channels": 96,
        "model.latent_length": 1800,
        "model.causal": False,
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
        "evaluation.local_rr_window_sec": 60,
        "evaluation.local_rr_step_sec": 15,
        "evaluation.ibi_peak_distance_samples": 142,
        "evaluation.ibi_match_tolerance_sec": 0.5,
        "evaluation.ibi_coverage_threshold": 0.80,
        "evaluation.ndtw_fs": 10,
        "evaluation.ndtw_radius_sec": 0.30,
        "evaluation.envelope_quantile_method": "linear",
        "evaluation.envelope_strata_low": 0.30875308839006915,
        "evaluation.envelope_strata_high": 0.7031542121234101,
        "training.optimizer": "adamw",
        "training.epochs": 80,
        "training.optimizer_updates": 6400,
        "training.batch_size": 128,
        "training.gradient_accumulation_steps": 1,
        "training.effective_batch_size": 128,
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
        "training.early_stopping_enabled": False,
        "training.resume": False,
        "resources.physical_batch_candidates": [128, 64, 32],
        "resources.resource_lock_required": True,
        "resources.resource_metrics_select_scientific_candidate": False,
        "outputs.run_root": f"runs/resp_temporal_v1/{variant}",
    }
    for key, value in expected.items():
        _assert_equal(cfg, key, value)
    if int(cfg.model.initialization_seed) != int(cfg.training.seed):
        raise ValueError("RTM-v1 model.initialization_seed 必须与 training.seed 一致")
    if int(cfg.training.seed) not in FORMAL_SEEDS:
        raise ValueError(f"RTM-v1 candidate seed 只允许 {list(FORMAL_SEEDS)}")

    allowed_keys = {
        "protocol": {
            "name", "stage", "run_role", "exact_implementation_enabled", "gpu_engineering_enabled",
            "formal_training_enabled", "validation_evaluation_enabled", "research_test_enabled",
        },
        "locks": {"signal_substrate", "candidate"},
        "data": {
            "format", "dataset_root", "index_csv", "input_set", "train_split", "val_split",
            "target_task", "bcg_input_key", "target_key", "max_train_windows", "max_val_windows",
            "filter_unusable", "drop_nonfinite_windows", "preload_windows", "train_sample_strategy",
            "val_sample_strategy", "train_sample_seed", "val_sample_seed", "stratify_column",
            "min_hard_valid_ratio", "min_state_alignment_valid_ratio",
        },
        "window": {"input_fs_hz", "duration_samples", "duration_sec"},
        "substrate": {
            "order", "anti_alias", "audit_equivalence", "explicit_analytic_envelope_input_branch",
            "fixed_equal_rms_proxy_input",
        },
        "multiscale": {
            "grids_hz", "downsampling", "average_pooling_allowed", "branch_waveform_heads_allowed",
            "stage_10_to_2", "stage_10_to_1",
        },
        "model": {
            "name", "variant", "family", "role", "expected_trainable_parameters",
            "initialization_seed", "latent_hz", "latent_channels", "latent_length", "causal",
        },
        "loss": {
            "band_low_hz", "band_high_hz", "scale_eps", "dynamic_eps", "corr_eps", "envelope_eps",
            "max_lag_sec", "envelope_window_sec", "envelope_step_sec", "sync_weight", "effort_weight",
        },
        "evaluation": {
            "local_rr_window_sec", "local_rr_step_sec", "ibi_peak_distance_samples",
            "ibi_match_tolerance_sec", "ibi_coverage_threshold", "ndtw_fs", "ndtw_radius_sec",
            "envelope_quantile_method", "envelope_strata_low", "envelope_strata_high",
        },
        "training": {
            "optimizer", "epochs", "optimizer_updates", "batch_size", "gradient_accumulation_steps",
            "effective_batch_size", "max_learning_rate", "min_learning_rate", "adam_betas", "adam_eps",
            "weight_decay", "warmup_fraction", "lr_schedule", "grad_clip_norm", "use_amp", "amp_dtype",
            "drop_last", "early_stopping_enabled", "resume", "num_workers", "persistent_workers",
            "prefetch_factor", "seed", "device", "show_progress",
        },
        "resources": {
            "physical_batch_candidates", "resource_lock_required",
            "resource_metrics_select_scientific_candidate",
        },
        "outputs": {"run_root"},
    }
    if set(cfg.keys()) != set(allowed_keys):
        raise ValueError(f"RTM-v1 顶层 section 必须是 {sorted(allowed_keys)}")
    for section, keys in allowed_keys.items():
        actual = set(cfg[section].keys())
        if actual != keys:
            raise ValueError(f"RTM-v1 {section} 字段必须严格为 {sorted(keys)}，当前为 {sorted(actual)}")
    if set(cfg.locks.signal_substrate.keys()) != {"path", "sha256"}:
        raise ValueError("RTM-v1 signal_substrate lock 字段必须严格为 path/sha256")
    if set(cfg.locks.candidate.keys()) != {"path", "sha256"}:
        raise ValueError("RTM-v1 candidate lock 字段必须严格为 path/sha256")
    if set(cfg.substrate.anti_alias.keys()) != {
        "window", "beta", "padtype", "stage_100_to_20", "stage_20_to_10",
    }:
        raise ValueError("RTM-v1 anti_alias 字段漂移")
    if set(cfg.substrate.audit_equivalence.keys()) != {"float64_atol", "float64_rtol"}:
        raise ValueError("RTM-v1 audit_equivalence 字段漂移")
    _validate_lock_files(cfg)
