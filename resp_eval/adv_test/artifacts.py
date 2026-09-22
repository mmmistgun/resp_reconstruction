from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf

from resp_train.aligned_dual_view.artifacts import read_json, sha256_file, write_json
from resp_train.aligned_dual_view.features import spec_digest

from .contract import EVIDENCE_ROLE, PROTOCOL, PROTOCOL_PATH, ROOT, repo_path


def execution_code(lock):
    files = dict(lock["training_code"]["files"])
    for relative, expected in files.items():
        if sha256_file(repo_path(relative)) != expected:
            raise ValueError(f"冻结训练源文件改变: {relative}")
    paths = [ROOT / "resp_eval/__init__.py", *sorted((ROOT / "resp_eval/adv_test").glob("*.py")),
             ROOT / "scripts/eval_aligned_dual_view_v1_research_test.py", PROTOCOL_PATH]
    files.update({str(path.relative_to(ROOT)): sha256_file(path) for path in paths})
    return {"files": files, "sha256": spec_digest(files)}


def check_output_location(output, cfg):
    output = Path(output).resolve()
    protected = [ROOT / name for name in ("resp_train", "resp_eval", "configs", "scripts", "tests", "docs", ".git")]
    protected.append(Path(cfg.data.dataset_root).resolve())
    if any(output.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("输出不得位于源码或数据源目录")
    for parent in output.parents:
        if any((parent / name).exists() for name in
               ("started.json", "completed.json", "run_manifest.json", "cache_manifest.json")):
            raise ValueError("不得向已有实验/cache 目录补写")
    return output


@contextmanager
def output_directory(output, *, kind, cfg, lock):
    """专项 test 产物独立记账；完成文件不可覆盖，失败文件原地保留。"""
    output = check_output_location(output, cfg)
    output.mkdir(parents=True, exist_ok=False)
    try:
        versions = {name: importlib.metadata.version(name) for name in lock["dependencies"]}
        if versions != lock["dependencies"]:
            raise ValueError("执行依赖偏离冻结训练环境")
        code = execution_code(lock)
        started = {
            "protocol": PROTOCOL, "kind": kind, "lock_id": lock["lock_id"],
            "run_id": f"{kind}-{lock['lock_id'][:12]}-{uuid4().hex}",
            "evidence_role": EVIDENCE_ROLE, "code": code, "dependencies": versions,
            "created_at": datetime.now(timezone.utc).isoformat(), "command": list(sys.argv),
            "python": platform.python_version(), "platform": platform.platform(),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         check=True, capture_output=True, text=True).stdout.strip(),
            "git_status": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                         check=True, capture_output=True, text=True).stdout,
        }
        write_json(output / "started.json", started)
        write_json(output / "candidate_lock.json", lock)
        OmegaConf.save(cfg, output / "source_config.yaml", resolve=True)
        for relative, expected in code["files"].items():
            destination = output / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo_path(relative), destination)
            if sha256_file(destination) != expected:
                raise ValueError("源码在快照期间发生变化")
        yield output, started
        if execution_code(lock) != code:
            raise ValueError("执行期间源码变化")
        write_json(output / "completed.json", {
            "status": "complete", "kind": kind, "run_id": started["run_id"],
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
    except BaseException as exc:
        write_json(output / "failed.json", {
            "status": "failed", "kind": kind, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


def verified_artifact(root, *, kind, lock):
    root = Path(root)
    if (root / "failed.json").exists():
        raise ValueError("research-test 产物处于失败状态")
    receipt = read_json(root / "completed.json")
    if receipt.get("status") != "complete" or receipt.get("kind") != kind:
        raise ValueError("research-test 产物未完成或 kind 不匹配")
    if receipt["manifest_sha256"] != sha256_file(root / "manifest.json"):
        raise ValueError("research-test manifest 哈希漂移")
    manifest = read_json(root / "manifest.json")
    if (manifest.get("protocol") != PROTOCOL or manifest.get("kind") != kind
            or manifest.get("lock_id") != lock["lock_id"] or manifest.get("evidence_role") != EVIDENCE_ROLE):
        raise ValueError("research-test 协议/候选锁/证据属性不匹配")
    return manifest


def common_manifest(kind, lock, started):
    return {"protocol": PROTOCOL, "kind": kind, "lock_id": lock["lock_id"],
            "evidence_role": EVIDENCE_ROLE, "execution_code_sha256": started["code"]["sha256"]}
