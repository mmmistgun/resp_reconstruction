from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from resp_train.crd.model import (
    CRD_C1_VARIANTS,
    CRD_C2_VARIANTS,
    CRD_TF_P6_VARIANTS,
    CRD_TF_VARIANTS,
    CRD_VARIANTS,
)


CRD_PROTOCOL_VERSION = "crd-v1.1-s0-s1-20260808"
CRD_DIAGNOSTIC_PROTOCOL_VERSION = "crd-v1.1-s1d-20260808"
CRD_EXPLORATORY_PROTOCOL_VERSION = "crd-v1.1-s1e-20260809"
CRD_S1F_PROTOCOL_VERSION = "crd-v1.1-s1f-research-test-informed-20260809"
CRD_S2A_PROTOCOL_VERSION = "crd-v1.1-s2a-research-test-informed-20260809"
CRD_S2BR_PROTOCOL_VERSION = "crd-v1.1-s2br-result-informed-20260810"
CRD_102_FAILURE_DIAGNOSTIC_PROTOCOL_VERSION = "crd-v1.1-crd102-failure-diagnostic-20260811"
CRD_102_FAILURE_METADATA_PROTOCOL_VERSION = "crd-v1.1-crd102-failure-metadata-20260811"
CRD_102_MATCHED_OBSERVABILITY_PROTOCOL_VERSION = "crd-v1.1-crd102-matched-observability-20260811"
CRD_CONTROLS_PROTOCOL_VERSION = "crd-v1.1-controls-research-informed-20260811"
CRD_TF_PROTOCOL_VERSION = "crd-tf-v1-research-informed-20260812"
CRD_TF_P6A_PROTOCOL_VERSION = "crd-tf-v1-p6a-validation-development-20260815"
CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION = "crd-tf-v1-research-test-development-20260816"
CRD_TF_CACHE_PATH = (
    "/mnt/disk_code/marques/resp_reconstruction/runs/crd_tf_v1/cache/"
    "bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0"
)
CRD_DIAGNOSTIC_VARIANTS = {
    "crd_103_direct_local_mamba",
    "crd_105_direct_coarse",
}
CRD_EXPLORATORY_VARIANTS = {
    "crd_102_b0_local_mamba",
    "crd_104_direct_hier_mamba",
}
CRD_S1F_VARIANTS = {"crd_106_b0_hier_mamba"}
CRD_S2A_VARIANTS = {
    "crd_202_base_legacy_energy",
    "crd_203_base_analytic_am",
    "crd_204_base_morphology",
}
CRD_S2BR_VARIANTS = {
    "crd_205_base_em_static",
    "crd_206_base_am_static",
    "crd_207_base_cap_em",
    "crd_208_base_cap_am",
}
PINNED_DEPENDENCIES = {
    "mamba-ssm": "2.3.2.post1",
    "causal-conv1d": "1.6.2.post1",
}
FORMAL_SEEDS = (20260811, 20260812, 20260813)


def check_crd_dependencies() -> list[str]:
    """返回缺失或版本不一致的 CRD 原生依赖；绝不提供静默 fallback。"""

    problems: list[str] = []
    imports = {"mamba-ssm": "mamba_ssm", "causal-conv1d": "causal_conv1d"}
    for distribution, expected in PINNED_DEPENDENCIES.items():
        try:
            importlib.import_module(imports[distribution])
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{distribution}=={expected} 未安装")
            continue
        except (ImportError, OSError, RuntimeError) as exc:
            problems.append(f"{distribution}=={expected} 导入失败: {type(exc).__name__}: {exc}")
            continue
        if actual != expected:
            problems.append(f"{distribution} 要求 {expected}，当前 {actual}")
    return problems


def crd_dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in PINNED_DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def load_crd_config(path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    base_reference = cfg.pop("_base_", None)
    if base_reference is not None:
        base_path = (cfg_path.parent / str(base_reference)).resolve()
        if base_path.parent != cfg_path.resolve().parent:
            raise ValueError("CRD _base_ 只允许引用同目录配置")
        if not base_path.is_file():
            raise FileNotFoundError(f"CRD base 配置不存在: {base_path}")
        base_cfg = OmegaConf.load(base_path)
        if "_base_" in base_cfg:
            raise ValueError("CRD 配置只允许一层 _base_，禁止递归继承")
        cfg = OmegaConf.merge(base_cfg, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    _validate_crd_config(cfg)
    return cfg


def _require(cfg: DictConfig, keys: Iterable[str]) -> None:
    for key in keys:
        if OmegaConf.select(cfg, key) is None:
            raise ValueError(f"CRD 配置缺少必需字段: {key}")


def _assert_equal(cfg: DictConfig, key: str, expected: Any) -> None:
    value = OmegaConf.select(cfg, key)
    normalized = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if normalized != expected:
        raise ValueError(f"CRD-v1.1 冻结要求 {key}={expected!r}，当前为 {normalized!r}")


def _validate_crd_config(cfg: DictConfig) -> None:
    _require(
        cfg,
        (
            "protocol.name",
            "protocol.stage",
            "protocol.run_role",
            "data.dataset_root",
            "data.index_csv",
            "data.input_set",
            "window.target_fs",
            "window.duration_samples",
            "model.name",
            "model.variant",
            "model.initialization_seed",
            "loss.band_low_hz",
            "loss.band_high_hz",
            "training.optimizer",
            "training.epochs",
            "training.batch_size",
            "training.gradient_accumulation_steps",
            "training.max_learning_rate",
            "training.min_learning_rate",
            "training.seed",
            "training.device",
            "outputs.run_root",
        ),
    )

    variant = str(cfg.model.variant).lower()
    if variant not in CRD_VARIANTS:
        raise ValueError(f"未知 CRD variant={variant!r}；可选 {list(CRD_VARIANTS)}")
    if variant in CRD_TF_P6_VARIANTS:
        expected_stage = "tf_p6a"
    elif variant in CRD_TF_VARIANTS:
        expected_stage = "tf"
    elif variant in CRD_C1_VARIANTS:
        expected_stage = "c1"
    elif variant in CRD_C2_VARIANTS:
        expected_stage = "c2"
    elif variant in CRD_S2A_VARIANTS:
        expected_stage = "s2"
    elif variant in CRD_S2BR_VARIANTS:
        expected_stage = "s2br"
    else:
        expected_stage = "s0" if variant.startswith("crd_00") else "s1"
    if variant in CRD_TF_P6_VARIANTS:
        expected_protocol = CRD_TF_P6A_PROTOCOL_VERSION
    elif variant in CRD_TF_VARIANTS:
        expected_protocol = CRD_TF_PROTOCOL_VERSION
    elif variant in CRD_C1_VARIANTS | CRD_C2_VARIANTS:
        expected_protocol = CRD_CONTROLS_PROTOCOL_VERSION
    elif variant in CRD_DIAGNOSTIC_VARIANTS:
        expected_protocol = CRD_DIAGNOSTIC_PROTOCOL_VERSION
    elif variant in CRD_EXPLORATORY_VARIANTS:
        expected_protocol = CRD_EXPLORATORY_PROTOCOL_VERSION
    elif variant in CRD_S1F_VARIANTS:
        expected_protocol = CRD_S1F_PROTOCOL_VERSION
    elif variant in CRD_S2A_VARIANTS:
        expected_protocol = CRD_S2A_PROTOCOL_VERSION
    elif variant in CRD_S2BR_VARIANTS:
        expected_protocol = CRD_S2BR_PROTOCOL_VERSION
    else:
        expected_protocol = CRD_PROTOCOL_VERSION
    frozen = {
        "protocol.name": expected_protocol,
        "protocol.stage": expected_stage,
        "data.format": "research_v2",
        "data.dataset_root": (
            "/mnt/disk_code/marques/resp_prepare/dataset/"
            "20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf"
        ),
        "data.index_csv": "training/dataset_index.csv",
        "data.input_set": "research_v2_waveform",
        "data.train_split": "train",
        "data.val_split": "val",
        "data.test_split": "test",
        "data.target_task": "waveform",
        "data.bcg_input_key": "bcg_rawish_segment_soft_z_key",
        "data.target_key": "target_waveform_segment_soft_z_key",
        "data.filter_unusable": True,
        "data.drop_nonfinite_windows": False,
        "data.preload_windows": True,
        "data.train_sample_strategy": "stratified_random",
        "data.val_sample_strategy": "stratified_random",
        "data.test_sample_strategy": "stratified_random",
        "data.train_sample_seed": 20260610,
        "data.val_sample_seed": 20260611,
        "data.test_sample_seed": 20260612,
        "data.stratify_column": "allowed_losses",
        "data.min_hard_valid_ratio": 0.80,
        "data.min_state_alignment_valid_ratio": 0.80,
        "window.target_fs": 100,
        "window.duration_samples": 18000,
        "window.duration_sec": 180,
        "model.name": "crd_v1",
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
    }
    if variant in (*CRD_TF_VARIANTS, *CRD_TF_P6_VARIANTS):
        from resp_train.crd.tf_v1_model import TF_ALL_VARIANT_REPRESENTATIONS

        frozen.update(
            {
                "data.tf_cache_path": CRD_TF_CACHE_PATH,
                "model.tf_representations": list(TF_ALL_VARIANT_REPRESENTATIONS[variant]),
            }
        )
        if variant in CRD_TF_P6_VARIANTS:
            frozen.update(
                {
                    "training.early_stopping_enabled": True,
                    "training.early_stopping_patience": 30,
                    "training.early_stopping_min_delta": 0.0,
                }
            )
        else:
            frozen["training.early_stopping_enabled"] = False
    for key, expected in frozen.items():
        _assert_equal(cfg, key, expected)

    allowed_keys = {
        "protocol": {"name", "stage", "run_role"},
        "data": {
            "format", "dataset_root", "index_csv", "input_set", "train_split", "val_split", "test_split",
            "target_task", "bcg_input_key", "target_key", "max_train_windows", "max_val_windows",
            "max_test_windows", "filter_unusable", "drop_nonfinite_windows", "preload_windows",
            "train_sample_strategy", "val_sample_strategy", "test_sample_strategy", "train_sample_seed",
            "val_sample_seed", "test_sample_seed", "stratify_column", "min_hard_valid_ratio",
            "min_state_alignment_valid_ratio",
        },
        "window": {"target_fs", "duration_samples", "duration_sec"},
        "model": {"name", "variant", "initialization_seed"},
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
            "optimizer", "epochs", "batch_size", "gradient_accumulation_steps", "max_learning_rate",
            "min_learning_rate", "adam_betas", "adam_eps", "weight_decay", "warmup_fraction",
            "lr_schedule", "grad_clip_norm", "use_amp", "amp_dtype", "drop_last", "resume",
            "num_workers", "persistent_workers", "prefetch_factor", "seed", "device", "show_progress",
        },
        "outputs": {"run_root"},
    }
    if variant in (*CRD_TF_VARIANTS, *CRD_TF_P6_VARIANTS):
        allowed_keys["protocol"].add("execution_gate")
        allowed_keys["data"].add("tf_cache_path")
        allowed_keys["model"].add("tf_representations")
        allowed_keys["training"].add("early_stopping_enabled")
    if variant in CRD_TF_P6_VARIANTS:
        allowed_keys["training"].update({"early_stopping_patience", "early_stopping_min_delta"})
    top_level = set(cfg.keys())
    if top_level != set(allowed_keys):
        raise ValueError(f"CRD 配置顶层 section 必须是 {sorted(allowed_keys)}，当前为 {sorted(top_level)}")
    for section, expected_keys in allowed_keys.items():
        actual_keys = set(cfg[section].keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"CRD {section} 配置字段必须严格为 {sorted(expected_keys)}，当前为 {sorted(actual_keys)}"
            )
    if int(cfg.model.initialization_seed) != int(cfg.training.seed):
        raise ValueError("model.initialization_seed 必须与 training.seed 一致，以保证配对初始化")

    role = str(cfg.protocol.run_role).lower()
    if role not in {"formal", "acceptance", "smoke"}:
        raise ValueError("protocol.run_role 必须是 formal、acceptance 或 smoke")
    epochs = int(cfg.training.epochs)
    batch_size = int(cfg.training.batch_size)
    accumulation = int(cfg.training.gradient_accumulation_steps)
    if variant in (*CRD_TF_VARIANTS, *CRD_TF_P6_VARIANTS):
        execution_gate = str(cfg.protocol.execution_gate)
        device = str(cfg.training.device)
        if variant in CRD_TF_P6_VARIANTS and execution_gate == "p6a_cpu_only":
            if role != "smoke" or device != "cpu":
                raise ValueError("CRD-TF P6a p6a_cpu_only 只允许 smoke + CPU")
        elif variant in CRD_TF_P6_VARIANTS and execution_gate == "p6a_cuda_acceptance":
            if role != "acceptance" or not device.startswith("cuda:"):
                raise ValueError("CRD-TF P6a p6a_cuda_acceptance 只允许 acceptance + 显式 cuda:<index>")
        elif variant in CRD_TF_P6_VARIANTS and execution_gate == "p6a_formal":
            if role != "formal" or not device.startswith("cuda:"):
                raise ValueError("CRD-TF P6a p6a_formal 只允许 formal + 显式 cuda:<index>")
        elif variant in CRD_TF_P6_VARIANTS:
            raise ValueError("CRD-TF P6a execution_gate 必须与 P6a CPU/acceptance/formal 阶段严格匹配")
        elif execution_gate == "p2_cpu_only":
            if role != "smoke" or device != "cpu":
                raise ValueError("CRD-TF p2_cpu_only 只允许 smoke + CPU")
        elif execution_gate == "p3_cuda_acceptance":
            if role != "acceptance" or not device.startswith("cuda:"):
                raise ValueError("CRD-TF p3_cuda_acceptance 只允许 acceptance + 显式 cuda:<index>")
        elif execution_gate == "p4_formal":
            if role != "formal" or not device.startswith("cuda:"):
                raise ValueError("CRD-TF p4_formal 只允许 formal + 显式 cuda:<index>")
        else:
            raise ValueError("CRD-TF execution_gate 必须与已开放的 P2/P3/P4 阶段严格匹配")
    maxima = [cfg.data.get(name) for name in ("max_train_windows", "max_val_windows", "max_test_windows")]
    if role == "formal":
        if (epochs, batch_size, accumulation) != (80, 128, 1):
            raise ValueError("formal 固定 epochs=80、physical batch=128、accumulation=1")
        if any(value is not None for value in maxima):
            raise ValueError("formal 必须使用完整 train/val/test 索引，max_*_windows 均为 null")
        if int(cfg.training.seed) not in FORMAL_SEEDS:
            raise ValueError(f"formal seed 只允许 {list(FORMAL_SEEDS)}")
    elif role == "acceptance":
        if (epochs, batch_size, accumulation) != (1, 128, 1):
            raise ValueError("acceptance 固定 epochs=1、physical batch=128、accumulation=1")
        if maxima != [128, 32, None]:
            raise ValueError("acceptance 固定 max_train_windows=128、max_val_windows=32、max_test_windows=null")
    else:
        if not (1 <= epochs <= 2 and 1 <= batch_size <= 32 and 1 <= accumulation <= 4):
            raise ValueError("smoke 只允许 epochs<=2、batch<=32、accumulation<=4 的轻量覆盖")
        max_train, max_val, max_test = maxima
        if (
            max_train is None
            or max_val is None
            or not (1 <= int(max_train) <= 32)
            or not (1 <= int(max_val) <= 32)
            or max_test is not None
        ):
            raise ValueError("smoke 要求 train/val max windows 位于 1..32，且 max_test_windows=null")

    for key in ("data.max_train_windows", "data.max_val_windows", "data.max_test_windows"):
        value = OmegaConf.select(cfg, key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{key} 非 null 时必须为正整数")
    if int(cfg.training.num_workers) < 0:
        raise ValueError("training.num_workers 不能为负")
