from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """解析训练设备；未指定时优先使用 CUDA。"""
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境不可用")
    return resolved


def create_run_dir(run_root: str | Path) -> Path:
    """创建一次训练运行目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(run_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config(cfg: DictConfig, run_dir: str | Path) -> Path:
    """将 OmegaConf 配置保存到运行目录。"""
    path = Path(run_dir) / "config.yaml"
    OmegaConf.save(config=cfg, f=path)
    return path


def setup_logger(run_dir: str | Path) -> logging.Logger:
    """创建简洁的控制台/文件 logger。"""
    logger = logging.getLogger("resp_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免重复调用时重复添加 handler。
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(run_dir) / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
