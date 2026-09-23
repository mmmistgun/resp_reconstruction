from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable
import hashlib

from omegaconf import DictConfig, OmegaConf

from .model import ModelConfig, ARMS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/adv_fusion_factorial_v1/experiment.yaml"
TEMPLATE_SHA256 = "a73a5f7896f50bfe1c9c6a350707c284b87d74858b187ca68f32ef3afa4c4279"
PROTOCOL = "adv-fusion-factorial-v1-es30p15-train-val-20260922"
SEEDS = (20260811, 20260812, 20260813)
ALL_ARMS = tuple(ARMS)
INDEX_SHA256 = "f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f"
FORMAL_COUNTS = {"train": 10141, "val": 2675}


def _defaults():
    if hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("冻结配置模板被修改；改变协议字段需要显式修订")
    return OmegaConf.load(DEFAULT_CONFIG)


def load_experiment_config(path: str | Path = DEFAULT_CONFIG, overrides: Iterable[str] = ()) -> DictConfig:
    # 默认文件同时提供封闭 schema；未知字段（包括 test、旧 cache 或其他 loss）直接失败。
    cfg = _defaults()
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    validate_experiment_config(cfg)
    return cfg


def model_config(cfg: DictConfig) -> ModelConfig:
    return ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))


def validate_experiment_config(cfg: DictConfig) -> None:
    defaults = _defaults()
    observed = OmegaConf.to_container(cfg, resolve=True)
    baseline = OmegaConf.to_container(defaults, resolve=True)

    def compare(actual, expected, prefix=""):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"ADV 配置字段不匹配: {prefix}")
        for key, value in expected.items():
            name = f"{prefix}.{key}".strip(".")
            if isinstance(value, dict):
                compare(actual[key], value, name)
            elif name not in mutable and actual[key] != value:
                raise ValueError(f"ADV 冻结要求 {name}={value!r}")

    mutable = {
        "protocol.run_role", "data.dataset_root", "data.index_csv",
        "data.max_train_windows", "data.max_val_windows", "model.arm",
        "model.initialization_seed", "training.seed", "training.device", "training.show_progress",
        "training.epochs", "training.batch_size", "training.gradient_accumulation_steps", "training.use_amp",
    }
    compare(observed, baseline)
    if cfg.protocol.run_role not in {"smoke", "formal"}:
        raise ValueError("run_role 仅允许 smoke/formal")
    if cfg.model.arm not in ALL_ARMS:
        raise ValueError("融合候选未注册")
    model_config(cfg)
    if cfg.model.initialization_seed != cfg.training.seed:
        raise ValueError("模型 seed 必须与训练 seed 相同")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "seed"):
        value = cfg.training[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"training.{key} 必须为正整数")
    if type(cfg.training.use_amp) is not bool or type(cfg.training.show_progress) is not bool:
        raise ValueError("use_amp/show_progress 必须是布尔值")
    if cfg.protocol.run_role == "formal":
        if cfg.training.seed not in SEEDS or cfg.training.epochs != 80:
            raise ValueError("formal 固定三 seed、最大80 epochs及统一early stopping合同")
        if (cfg.training.batch_size, cfg.training.gradient_accumulation_steps) != (32, 4):
            raise ValueError("formal 必须 physical batch=32、accumulation=4")
        if not str(cfg.training.device).startswith("cuda") or not cfg.training.use_amp:
            raise ValueError("formal 要求 CUDA 和 bf16 AMP")
        if cfg.data.dataset_root != defaults.data.dataset_root or cfg.data.index_csv != defaults.data.index_csv:
            raise ValueError("formal 数据源必须为冻结 research-v2")
        if cfg.data.max_train_windows is not None or cfg.data.max_val_windows is not None:
            raise ValueError("formal 必须使用完整 train/val")
    else:
        if cfg.training.epochs > 2:
            raise ValueError("smoke 最多两个 epochs")
        for split in ("train", "val"):
            maximum = cfg.data[f"max_{split}_windows"]
            if type(maximum) is not int or not 1 <= maximum <= 64:
                raise ValueError("smoke 必须显式限制每个 split 为 1..64 窗")


def data_contract(cfg: DictConfig) -> dict:
    """缓存/训练共享的数据选择定义；与模型、训练 seed 和物理 batch 无关。"""
    return {
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "window": OmegaConf.to_container(cfg.window, resolve=True),
        "run_role": str(cfg.protocol.run_role),
    }
