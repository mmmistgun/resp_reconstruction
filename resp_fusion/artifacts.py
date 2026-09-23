from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf

from .config import PROTOCOL, ROOT
from resp_train.aligned_dual_view.features import spec_digest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value, dtype="<f4")
    if not np.isfinite(array).all():
        raise FloatingPointError("待记录数组包含 NaN/Inf")
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """只创建新产物；历史文件存在时拒绝写入。"""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def code_identity() -> dict:
    from resp_train.aligned_dual_view.artifacts import code_identity as adv_identity
    files = dict(adv_identity()["files"])
    paths = [*sorted((ROOT / "resp_fusion").glob("*.py")),
             *sorted((ROOT / "configs/adv_fusion_factorial_v1").glob("*.*")),
             ROOT / "docs/experiments/adv_fusion_factorial_v1_protocol_20260922.md",
             ROOT / "scripts/run_adv_fusion_factorial_v1.py"]
    files.update({str(path.relative_to(ROOT)): sha256_file(path) for path in paths})
    return {"files": files, "sha256": spec_digest(files)}


@contextmanager
def artifact_directory(output: str | Path, *, kind: str, cfg):
    """独占目录保存本次 lifecycle，失败和中断产物均保留。"""
    output = Path(output).resolve()
    protected = [ROOT / name for name in ("resp_train", "resp_fusion", "resp_eval", "configs", "scripts", "tests", "docs", ".git")]
    protected.append(Path(cfg.data.dataset_root).resolve())
    if any(output.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("输出目录不能位于数据源或执行源码目录中")
    for parent in output.parents:
        if any((parent / name).exists() for name in
               ("started.json", "completed.json", "run_manifest.json", "cache_manifest.json")):
            raise ValueError("不能向已有实验/cache 目录补写子运行")
    output.mkdir(parents=True, exist_ok=False)
    try:
        code = code_identity()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout
        started = {
            "protocol": PROTOCOL, "kind": kind, "run_id": f"{kind}-{code['sha256'][:12]}-{uuid4().hex}",
            "created_at": datetime.now(timezone.utc).isoformat(), "command": list(sys.argv),
            "git_commit": commit, "git_status": status, "code": code,
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "torch", "ssqueezepy", "mamba-ssm", "causal-conv1d")},
        }
        write_json(output / "started.json", started)
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
        # 保存未提交文件的实际内容，避免仅记录 HEAD 无法复现新模型。
        for relative in code["files"]:
            destination = output / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
            if sha256_file(destination) != code["files"][relative]:
                raise RuntimeError(f"代码在快照期间发生变化: {relative}")
        yield output, started
        if code_identity() != code:
            raise RuntimeError("运行期间执行代码发生变化；保留产物并标记失败")
        manifest_sha = sha256_file(output / "manifest.json")
        write_json(output / "completed.json", {
            "status": "complete", "run_id": started["run_id"], "kind": kind,
            "manifest_sha256": manifest_sha,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
    except BaseException as exc:
        write_json(output / "failed.json", {
            "status": "failed", "kind": kind, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


def verified_manifest(root: str | Path, kind: str) -> dict:
    root = Path(root)
    if (root / "failed.json").exists():
        raise ValueError("产物 lifecycle 为失败")
    receipt = read_json(root / "completed.json")
    if receipt.get("status") != "complete" or receipt.get("kind") != kind:
        raise ValueError("产物未完成或 kind 不匹配")
    if receipt.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ValueError("产物 manifest 哈希不匹配")
    manifest = read_json(root / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("kind") != kind:
        raise ValueError("产物协议不匹配")
    started = read_json(root / "started.json")
    if (started.get("protocol") != PROTOCOL or started.get("kind") != kind
            or started.get("run_id") != receipt.get("run_id")):
        raise ValueError("开始与完成回执身份不匹配")
    return manifest


def require_finite_tree(value, name="checkpoint") -> None:
    """模型和优化器状态都检查，不能靠载入后某次预测有限来判定有效。"""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} 包含 NaN/Inf")
    elif isinstance(value, dict):
        for key, child in value.items():
            require_finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite_tree(child, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{name} 包含 NaN/Inf")
