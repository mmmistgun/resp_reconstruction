"""只读复用W0系列的模型无关产物；不复用其训练或GPU验收。"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import traceback
from . import artifacts as io
from .spec import ROOT, PROTOCOL, OUTPUT_ROOT, ARMS, load_spec, plan
from resp_train.paper_evidence.cwt_time_frequency_v1 import artifacts as legacy_io

LEGACY_PROTOCOL = "cwt-time-frequency-v1-20260930"
DEFAULT_SOURCE = ROOT / "runs/cwt_time_frequency_v1/session_20260930T184031Z_812b424a1c92"


def legacy_arm(arm):
    return "B" if arm == "A0" else arm


def canonical_representation(rep):
    result = deepcopy(rep)
    if result["arm"]["name"] == "B":
        result["arm"]["name"] = "A0"
    return result


def verify_reference_stage(path):
    protocol = io.read_json(Path(path) / "manifest.json")["protocol"]
    if protocol == LEGACY_PROTOCOL:
        return legacy_io.verify_stage(path)
    if protocol == PROTOCOL:
        return io.verify_stage(path)
    raise ValueError("不支持的复用来源协议")


def reference(session, kind, arm=None):
    frozen = io.load_session(session)
    refs = frozen.get("preprocessing_references")
    if not refs:
        return None
    entry = refs["caches"][arm] if kind == "cache" else refs[kind]
    path = Path(entry["path"])
    io.verify(path / "manifest.json", entry)
    manifest = verify_reference_stage(path)
    expected_phase = {"cache": "cache", "data": "data", "signals": "signals"}[kind]
    if manifest["binding"]["phase"] != expected_phase or manifest["binding"]["session"] != refs["source_session"]["sha256"]:
        raise ValueError("复用阶段不属于登记的来源会话")
    if kind == "cache":
        metadata = io.read_json(path / "cache.json")
        if canonical_representation(metadata["representation"]) != frozen["representations"][arm]:
            raise ValueError("复用缓存的实际变换/网格不一致")
    return path


def signals_receipt(session):
    path = reference(session, "signals")
    return path or io.completed(session / "signals", io.binding(session, "signals"))


def prepare_reuse(source_session=DEFAULT_SOURCE, *, data_scope=False):
    source_session = Path(source_session).resolve()
    io.verify(source_session / "session.json", io.read_json(source_session / "session_receipt.json"))
    source = io.read_json(source_session / "session.json")
    if source["protocol"] != LEGACY_PROTOCOL or source["research_test_open"] is not False:
        raise ValueError("必须使用已登记的W0 train/validation来源")
    if source["provenance"]["dependencies"] != io.dependencies():
        raise ValueError("复用来源的依赖环境发生变化")
    # CWT实现文件在新包中逐字节保留；其他公共科学依赖对照来源快照。
    feature_identity = source["provenance"]["files"]["resp_train/paper_evidence/cwt_time_frequency_v1/features.py"]
    io.verify(Path(__file__).with_name("features.py"), feature_identity)
    for relative, identity in source["provenance"]["files"].items():
        if relative.startswith(("resp_train/data/", "resp_train/metrics/", "resp_train/protocols/")) or relative == "resp_train/crd/tf_v1_features.py":
            io.verify(ROOT / relative, identity)
    representations = {}
    for arm, spec in ARMS.items():
        rep = canonical_representation(source["representations"][legacy_arm(arm)])
        if rep["arm"] != asdict(spec):
            raise ValueError(f"{arm}表示定义与旧缓存不一致")
        representations[arm] = rep
    refs = {"source_session": {"path": str(source_session), **io.identity(source_session / "session.json")}, "caches": {}}
    for kind in ("data", "signals"):
        path = legacy_io.completed(source_session / kind, legacy_io.binding(source_session, kind))
        if path is None:
            raise ValueError(f"来源{kind}未完成")
        refs[kind] = {"path": str(path), **io.identity(path / "manifest.json")}
    for arm in ARMS:
        old_arm = legacy_arm(arm)
        key = legacy_io.binding(source_session, "cache", arm=old_arm, data=refs["data"]["sha256"])
        path = legacy_io.completed(source_session / "cache" / old_arm, key)
        if path is None:
            raise ValueError(f"来源缓存未完成: {old_arm}")
        if canonical_representation(io.read_json(path / "cache.json")["representation"]) != representations[arm]:
            raise ValueError("缓存表示身份漂移")
        refs["caches"][arm] = {"path": str(path), **io.identity(path / "manifest.json")}
    calibration = Path(source["calibration"]["path"])
    io.verify(calibration / "manifest.json", source["calibration"])
    verify_reference_stage(calibration)
    output = OUTPUT_ROOT / ("session_" + io.stamp())
    output.mkdir(parents=True, exist_ok=False)
    try:
        io.write_json(output / "session.json", {"protocol": PROTOCOL, "spec": load_spec(), "plan": plan(),
                      "representations": representations, "provenance": io.provenance(output), "calibration": source["calibration"],
                      "preprocessing_references": refs, "execution_scope": "train_validation" if data_scope else "synthetic_only",
                      "parameter_review": source["parameter_review"], "research_test_open": False,
                      "model_family": "APOR_A0", "historical_training_reused": False,
                      "gpu_acceptance_reused": False, "created_for": "APOR新模型验收和独立实验生命周期"})
        io.write_json(output / "session_receipt.json", io.identity(output / "session.json"))
    except BaseException:
        io.write_json(output / "prepare_failed.json", {"traceback": traceback.format_exc()})
        raise
    return output
