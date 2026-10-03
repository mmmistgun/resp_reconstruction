"""来源快照与不可覆盖阶段；失败产物和 traceback 始终保留。"""
from __future__ import annotations
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .spec import ROOT, PROTOCOL, SPEC_PATH, SOURCE_LOCK


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def verify(path, expected):
    actual = identity(path)
    if actual["sha256"] != expected["sha256"] or ("size_bytes" in expected and actual["size_bytes"] != expected["size_bytes"]):
        raise ValueError(f"文件身份漂移: {path}")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12]


def dependencies():
    names = ("torch", "numpy", "pandas", "scipy", "ssqueezepy", "mamba-ssm", "causal-conv1d", "omegaconf")
    return {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in names}}


def source_files():
    """固定入口的静态本地 import 闭包；新增无关 Python 文件不进入旧会话身份。"""
    queue = list(Path(__file__).parent.glob("*.py"))
    queue += [ROOT / "scripts/run_cwt_apor_v2.py", ROOT / "scripts/eval_cwt_apor_v2_research_test.py",
              ROOT / "scripts/run_cwt_apor_v2_engineering.py", ROOT / "scripts/run_cwt_apor_v2_formal.py"]
    found = set()
    while queue:
        path = queue.pop().resolve()
        if path in found or not path.is_file():
            continue
        found.add(path)
        relative = path.relative_to(ROOT)
        package = list(relative.parts[:-1])
        # Python 隐式导入各级 __init__，也纳入来源身份。
        for depth in range(1, len(package) + 1):
            parent_init = ROOT.joinpath(*package[:depth], "__init__.py")
            if parent_init.is_file() and parent_init not in found:
                queue.append(parent_init)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join(prefix + ([node.module] if node.module else []))
                modules = [module, *(module + "." + alias.name for alias in node.names if alias.name != "*")]
            for module in modules:
                local = ROOT.joinpath(*module.split("."))
                for candidate in (local.with_suffix(".py"), local / "__init__.py"):
                    if candidate.is_file() and candidate not in found:
                        queue.append(candidate)
    found.update([SPEC_PATH, SOURCE_LOCK, ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml",
                  ROOT / "configs/crd_tf_v1/crd_tf101_m_smoke.yaml",
                  ROOT / "docs/experiments/cwt_apor_v2_protocol_20261001.md",
                  ROOT / "docs/experiments/cwt_apor_v2_formal_execution_20261001.md",
                  ROOT / "tests/test_cwt_apor_v2.py"])
    return sorted(found)


def provenance(output):
    files = {}
    for path in source_files():
        relative = str(path.relative_to(ROOT))
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        files[relative] = identity(target)
        verify(path, files[relative])
    return {"files": files, "dependencies": dependencies(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)}


def verify_provenance(payload, snapshot_root=None):
    if payload["dependencies"] != dependencies():
        raise ValueError("运行依赖改变")
    for relative, expected in payload["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("源码路径越界")
        verify(path, expected)
        if snapshot_root is not None:
            verify(snapshot_root / "source_snapshot" / relative, expected)


@contextmanager
def mutex(path, shared=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"阶段已在运行: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_stage(path, binding=None):
    path = Path(path).resolve()
    receipt = read_json(path / "receipt.json")
    verify(path / "manifest.json", receipt)
    manifest = read_json(path / "manifest.json")
    if manifest["protocol"] != PROTOCOL or manifest["status"] != "completed":
        raise ValueError("阶段未成功完成")
    if binding is not None and manifest["binding"] != binding:
        raise ValueError("阶段来源不匹配")
    for relative, expected in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path):
            raise ValueError("产物路径越界")
        verify(file, expected)
    return manifest


def completed(parent, binding):
    paths = sorted(Path(parent).glob("attempt_*/receipt.json"))
    if len(paths) > 1:
        raise RuntimeError("同阶段存在多个成功实例")
    if paths:
        verify_stage(paths[0].parent, binding)
        return paths[0].parent
    return None


def load_session(session):
    session = Path(session).resolve()
    verify(session / "session.json", read_json(session / "session_receipt.json"))
    value = read_json(session / "session.json")
    if value["protocol"] != PROTOCOL:
        raise ValueError("实验 session 协议不符")
    verify_provenance(value["provenance"], session)
    return value


def require_data_scope(session):
    value = load_session(session)
    if value.get("execution_scope") != "train_validation":
        raise PermissionError("此 session 仅供合成校准/GPU工程验收，不允许真实数据、训练或 research-test")
    return value


def binding(session, phase, **fields):
    return {"session": identity(Path(session) / "session.json")["sha256"], "phase": phase, **fields}


@contextmanager
def attempt(parent, binding, retry=False):
    parent = Path(parent)
    with mutex(parent / ".mutex"):
        if completed(parent, binding):
            raise FileExistsError("成功阶段禁止重跑")
        if list(parent.glob("attempt_*")) and not retry:
            raise RuntimeError("存在失败或中断现场；检查后显式 --retry-failed 创建新 attempt")
        output = parent / ("attempt_" + stamp())
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "started.json", {"protocol": PROTOCOL, "binding": binding})
        try:
            yield output
            files = {str(p.relative_to(output)): identity(p) for p in sorted(output.rglob("*")) if p.is_file()}
            write_json(output / "manifest.json", {"protocol": PROTOCOL, "binding": binding, "status": "completed", "files": files})
            write_json(output / "receipt.json", identity(output / "manifest.json"))
        except BaseException:
            write_json(output / "failed.json", {"binding": binding, "traceback": traceback.format_exc()})
            raise
