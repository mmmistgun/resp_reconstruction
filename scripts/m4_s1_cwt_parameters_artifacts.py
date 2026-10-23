"""一次session身份、阶段回执和追加式失败记录；不访问真实数据。"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import traceback
from uuid import uuid4

from scripts import m4_s1_cwt_parameters_spec as spec

ROOT = spec.ROOT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    return {"sha256": sha(path), "size_bytes": Path(path).stat().st_size}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify(path, expected):
    if identity(path) != expected:
        raise ValueError(f"文件身份变化: {path}")


def artifact_path(root, relative):
    root = Path(root).resolve()
    value = Path(relative)
    result = (root / value).resolve()
    if value.is_absolute() or not result.is_relative_to(root) or result == root:
        raise ValueError("产物路径越界")
    return result


def dependencies():
    names = ("torch", "numpy", "pandas", "scipy", "ssqueezepy", "mamba-ssm", "causal-conv1d", "omegaconf")
    return {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in names}}


def code_identity():
    """从本轮入口收集静态本地import闭包，包含包初始化及配置继承文件。"""
    queue = list((ROOT / "scripts").glob("m4_s1_cwt_parameters_*.py"))
    queue += [ROOT / "scripts/run_m4_s1_cwt_parameters.py"]
    found = set()
    while queue:
        path = queue.pop().resolve()
        if path in found or not path.is_file():
            continue
        found.add(path)
        package = list(path.relative_to(ROOT).parts[:-1])
        for depth in range(1, len(package) + 1):
            init = ROOT.joinpath(*package[:depth], "__init__.py")
            if init.is_file() and init not in found:
                queue.append(init)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join(prefix + ([node.module] if node.module else []))
                modules = [module, *(module + "." + a.name for a in node.names if a.name != "*")]
            for module in modules:
                local = ROOT.joinpath(*module.split("."))
                for candidate in (local.with_suffix(".py"), local / "__init__.py"):
                    if candidate.is_file() and candidate not in found:
                        queue.append(candidate)
    from resp_train.patch_aligned_tf_training import P2_LOCK_PATH
    found.update([spec.SPEC, spec.PROTOCOL_PATH, ROOT / P2_LOCK_PATH])
    found.update((ROOT / "configs/crd_tf_v1").glob("*.yaml"))
    found.update((ROOT / "tests").glob("test_m4_s1_cwt_parameters*.py"))
    found.add(ROOT / "tests/m4_s1_fixtures.py")
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(found)}


@contextmanager
def mutex(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"阶段已被其他进程占用: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def create_session(session, representations, initialization, resolved_configs, *, scope):
    """prepare调用方持有互斥锁；输入均为元数据/合成初始化，不读取样本。"""
    session = Path(session).resolve()
    sources = code_identity()
    with tarfile.open(session / "source_snapshot.tar.gz", "x:gz") as archive:
        for name in sources:
            archive.add(ROOT / name, arcname=name)
    if sources != code_identity():
        raise RuntimeError("保存快照期间源码变化")
    write_json(session / "session.json", {
        "protocol": spec.PROTOCOL, "spec": spec.spec(), "cells": spec.plan(), "scope": scope,
        "representations": representations, "initialization": initialization,
        "resolved_configs": resolved_configs, "code_sha256": sources,
        "snapshot": identity(session / "source_snapshot.tar.gz"), "dependencies": dependencies(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "splits": ["train", "val"], "research_test": False, "historical_training_reused": False,
    })
    write_json(session / "session_receipt.json", identity(session / "session.json"))


def load_session(session):
    session = Path(session).resolve()
    verify(session / "session.json", read_json(session / "session_receipt.json"))
    value = read_json(session / "session.json")
    if (value["protocol"] != spec.PROTOCOL or value["spec"] != spec.spec() or value["cells"] != spec.plan()
            or value["code_sha256"] != code_identity() or value["dependencies"] != dependencies()
            or value["scope"] not in ("synthetic_only", "train_validation")
            or value["splits"] != ["train", "val"] or value["research_test"] is not False
            or value["historical_training_reused"] is not False):
        raise ValueError("session源码、依赖、矩阵或访问合同变化")
    verify(session / "source_snapshot.tar.gz", value["snapshot"])
    for name, expected in value["resolved_configs"].items():
        verify(artifact_path(session, name), expected)
    return value


def require_data_scope(session):
    value = load_session(session)
    if value["scope"] != "train_validation":
        raise PermissionError("合成session不允许真实数据准备或正式训练")
    return value


def binding(session, phase, **values):
    return {"session_sha256": sha(Path(session) / "session.json"), "phase": phase, **values}


def verify_stage(path, expected_binding=None):
    path = Path(path).resolve()
    verify(path / "manifest.json", read_json(path / "receipt.json"))
    manifest = read_json(path / "manifest.json")
    if manifest["protocol"] != spec.PROTOCOL or manifest["status"] != "completed":
        raise ValueError("阶段尚未成功")
    if expected_binding is not None and manifest["binding"] != expected_binding:
        raise ValueError("阶段绑定不符")
    for name, expected in manifest["files"].items():
        verify(artifact_path(path, name), expected)
    return manifest


def completed(parent, expected_binding):
    markers = sorted(Path(parent).glob("attempt_*/receipt.json"))
    if len(markers) > 1:
        raise ValueError("同阶段存在多个成功实例")
    if markers:
        verify_stage(markers[0].parent, expected_binding)
        return markers[0].parent
    return None


@contextmanager
def attempt(parent, expected_binding, *, retry_failed=False):
    parent = Path(parent)
    with mutex(parent / ".stage.lock"):
        if completed(parent, expected_binding):
            raise FileExistsError("成功阶段禁止重跑")
        if list(parent.glob("attempt_*")) and not retry_failed:
            raise FileExistsError("存在失败或中断记录；须显式--retry-failed创建新attempt")
        output = parent / ("attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12])
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "started.json", {"binding": expected_binding, "command": sys.argv})
        try:
            yield output
            files = {str(p.relative_to(output)): identity(p) for p in sorted(output.rglob("*")) if p.is_file()}
            write_json(output / "manifest.json", {"protocol": spec.PROTOCOL, "status": "completed",
                       "binding": expected_binding, "files": files})
            write_json(output / "receipt.json", identity(output / "manifest.json"))
        except BaseException as exc:
            write_json(output / "failed.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                               "binding": expected_binding})
            raise
