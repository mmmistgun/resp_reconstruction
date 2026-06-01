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
        "window.duration_samples",
        "model.name",
        "training.epochs",
        "outputs.run_root",
    ]
    for key in required:
        if OmegaConf.select(cfg, key) is None:
            raise ValueError(f"配置缺少必需字段: {key}")
