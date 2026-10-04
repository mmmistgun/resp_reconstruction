"""新实验的不可覆盖产物与源码快照；复用通用哈希和互斥实现。"""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import traceback
from resp_train.paper_evidence.cwt_apor_v2.artifacts import (
    read_json, write_json, identity, verify, json_hash, stamp, mutex, dependencies,
    source_files as inherited_sources,
)
from .spec import ROOT, PROTOCOL, contract


def sources():
    files = set(inherited_sources())
    files.update(Path(__file__).parent.glob("*.py"))
    files.update(ROOT.glob("scripts/*h_only_ablation_v1*.py"))
    files.update(ROOT.glob("tests/test_h_only_ablation_v1*.py"))
    files.update(ROOT.glob("configs/h_only_ablation_v1/*"))
    files.add(ROOT / "docs/experiments/h_only_ablation_v1_protocol_20261004.md")
    return sorted(files)


def snapshot(output):
    files = {}
    for path in sources():
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


def load_session(session):
    session = Path(session)
    verify(session / "session.json", read_json(session / "session_receipt.json"))
    frozen = read_json(session / "session.json")
    if frozen["contract"] != contract() or frozen["provenance"]["dependencies"] != dependencies():
        raise ValueError("session 合同/环境漂移")
    for name, expected in frozen["provenance"]["files"].items():
        verify(ROOT / name, expected)
        verify(session / "source_snapshot" / name, expected)
    return frozen


def binding(session, phase, **fields):
    return {"session": identity(Path(session) / "session.json")["sha256"], "phase": phase, **fields}


def verify_stage(path, key):
    path = Path(path)
    verify(path / "manifest.json", read_json(path / "receipt.json"))
    payload = read_json(path / "manifest.json")
    if payload["protocol"] != PROTOCOL or payload["binding"] != key or payload["status"] != "completed":
        raise ValueError("阶段身份不匹配")
    for name, expected in payload["files"].items():
        child = (path / name).resolve()
        if not child.is_relative_to(path.resolve()):
            raise ValueError("产物路径越界")
        verify(child, expected)
    return payload


def completed(parent, key):
    paths = list(Path(parent).glob("attempt_*/receipt.json"))
    if len(paths) > 1:
        raise RuntimeError("同阶段存在多个成功实例")
    if paths:
        verify_stage(paths[0].parent, key)
        return paths[0].parent
    return None


@contextmanager
def attempt(parent, key, retry=False):
    parent = Path(parent)
    with mutex(parent / ".mutex"):
        if completed(parent, key):
            raise FileExistsError("成功阶段禁止重跑")
        if list(parent.glob("attempt_*")) and not retry:
            raise RuntimeError("已有失败或中断现场，检查后用 --retry-failed 创建新 attempt")
        output = parent / ("attempt_" + stamp())
        output.mkdir(parents=True)
        write_json(output / "started.json", {"protocol": PROTOCOL, "binding": key})
        try:
            yield output
            files = {str(p.relative_to(output)): identity(p) for p in sorted(output.rglob("*")) if p.is_file()}
            write_json(output / "manifest.json", {"protocol": PROTOCOL, "binding": key,
                       "status": "completed", "files": files})
            write_json(output / "receipt.json", identity(output / "manifest.json"))
        except BaseException:
            write_json(output / "failed.json", {"traceback": traceback.format_exc()})
            raise
