from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
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


def save_execution_manifest(path: str | Path, **context: object) -> Path:
    """保存最小复现信息；Git 不可用时显式记录错误而不是中断实验。"""

    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    payload: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": list(sys.argv),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "git_error": None,
        **context,
    }
    errors = [result.stderr.strip() for result in (commit, status) if result.returncode != 0]
    if errors:
        payload["git_error"] = "; ".join(error for error in errors if error) or "git command failed"
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


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
