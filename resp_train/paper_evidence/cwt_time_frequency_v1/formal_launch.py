"""已验收来源到独立 train/validation 会话的显式交接及后台执行。"""
from __future__ import annotations
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfreqz
from . import artifacts as io
from .spec import ROOT, PROTOCOL, OUTPUT_ROOT, ARMS, COUNTS, SEEDS, config, load_spec, plan

ALLOWED_CHANGED = frozenset({
    "resp_train/paper_evidence/cwt_time_frequency_v1/artifacts.py",
    "resp_train/paper_evidence/cwt_time_frequency_v1/engineering.py",
    "tests/test_cwt_time_frequency_v1.py",
})
ALLOWED_ADDED = frozenset({
    "resp_train/paper_evidence/cwt_time_frequency_v1/formal_launch.py",
    "scripts/run_cwt_time_frequency_v1_formal.py",
    "docs/experiments/cwt_time_frequency_v1_formal_execution_20261001.md",
})


def audit_reuse(source, current):
    """仅允许范围交接和回执引用入口变化；模型/变换/trainer/data等必须逐字节相同。"""
    old = source["provenance"]["files"]
    if source["spec"] != load_spec() or source["provenance"]["dependencies"] != io.dependencies():
        raise ValueError("正式交接的矩阵或依赖环境不一致")
    changed = {p for p in old.keys() & current.keys() if old[p] != current[p]}
    added, removed = current.keys()-old.keys(), old.keys()-current.keys()
    if removed or changed-ALLOWED_CHANGED or added-ALLOWED_ADDED:
        raise ValueError(f"验收实际依赖改变: changed={sorted(changed-ALLOWED_CHANGED)}, added={sorted(added-ALLOWED_ADDED)}, removed={sorted(removed)}")
    return {"changed": {p: {"old": old[p], "new": current[p]} for p in sorted(changed)},
            "added": {p: current[p] for p in sorted(added)}}


def frozen_engineering_source(path):
    path = Path(path).resolve()
    io.verify(path / "session.json", io.read_json(path / "session_receipt.json"))
    source = io.read_json(path / "session.json")
    if source["protocol"] != PROTOCOL or source["execution_scope"] != "synthetic_only":
        raise ValueError("来源必须是本轮仅合成验收会话")
    for relative, identity in source["provenance"]["files"].items():
        snapshot = (path / "source_snapshot" / relative).resolve()
        if not snapshot.is_relative_to(path / "source_snapshot"):
            raise ValueError("来源快照路径越界")
        io.verify(snapshot, identity)
    return source


def engineering_reference(session, device):
    frozen = io.load_session(session)
    ref = frozen["engineering_reference"]
    source_path = Path(ref["session"]["path"])
    io.verify(source_path / "session.json", ref["session"])
    source = frozen_engineering_source(source_path)
    audit_reuse(source, frozen["provenance"]["files"])
    item = ref["devices"].get(device)
    if item is None:
        raise ValueError("设备不在正式训练的验收来源集合")
    path = Path(item["path"])
    io.verify(path / "manifest.json", item)
    io.verify_stage(path, io.binding(source_path, "gpu", device=device))
    return path


def preprocessing_review(cfg, output, source):
    from .data import audit_rows
    _, rows = audit_rows(cfg)
    index = Path(cfg.data.dataset_root) / cfg.data.index_csv
    expected = {"rawish_low_hz": .03, "rawish_high_hz": 20., "rawish_order": 4}
    files, subjects = {}, set()
    for frame in rows.values():
        for row in frame.drop_duplicates("samp_id").itertuples(index=False):
            path = (index.parent / str(row.target_source_npz)).resolve().with_suffix(".json")
            metadata = io.read_json(path)
            if metadata.get("target_fs") != 100 or any(metadata["signal_params"].get(k) != v for k, v in expected.items()):
                raise ValueError(f"预处理频带元数据与计划不符: {path}")
            files[str(path)] = io.identity(path)
            subjects.add(int(row.samp_id))
    if len(subjects) != 39:
        raise ValueError("预处理核查未覆盖全部train/val受试者")
    freq = np.asarray(source["representations"]["C_20"]["frequencies_hz"])
    sos = butter(4, [.03, 20.], btype="bandpass", fs=100., output="sos")
    _, response = sosfreqz(sos, worN=np.r_[freq, 20.], fs=100.)
    table = pd.DataFrame({"frequency_hz": np.r_[freq, 20.], "nominal_zero_phase_amplitude": np.abs(response)**2})
    table.to_csv(output / "nominal_preprocessing_response.csv", index=False)
    calibration = Path(source["calibration"]["path"])
    io.verify(calibration / "manifest.json", source["calibration"])
    io.verify_stage(calibration)
    geometry = []
    for name, rep in source["representations"].items():
        response_table = pd.read_csv(calibration / f"{name}_response.csv")
        geometry.append({"arm": name, "scales": rep["shape"][0], "frames": rep["shape"][1],
                         "frequency_min_hz": min(rep["frequencies_hz"]), "frequency_max_hz": max(rep["frequencies_hz"]),
                         "duplicate_centers": rep["duplicate_center_count"],
                         "maximum_99percent_energy_radius_seconds": float(response_table.energy_99_radius_seconds.max())})
    pd.DataFrame(geometry).to_csv(output / "representation_review.csv", index=False)
    review = {"performed_by": "Codex", "authorization": "用户明确要求执行已批准60-cell正式GPU训练及其必要准备",
              "scope": "train_validation_only", "subjects_checked": sorted(subjects), "metadata": files,
              "signal_params": expected, "matrix_decision": "保留预先定义的20配置及全部频带端点",
              "frequency_interpretation": "20Hz是原始宽带滤波截止点；表中为标称四阶零相位响应。之后的对齐/soft-z非线性会改变谱结构，不能当作最终输入的线性传递函数。",
              "boundary_interpretation": "有限180秒冲激响应及重复中心保留；按预定义支持区报告低频关联的未定义分母，不据校准结果裁剪矩阵",
              "research_test_accessed": False}
    io.write_json(output / "parameter_review.json", review)
    return review


def prepare_formal(source_session, devices):
    source_session = Path(source_session).resolve()
    source = frozen_engineering_source(source_session)
    current = {str(p.relative_to(ROOT)): io.identity(p) for p in io.source_files()}
    changes = audit_reuse(source, current)
    refs = {}
    for device in devices:
        parent = source_session / "engineering" / device.replace(":", "_")
        path = io.completed(parent, io.binding(source_session, "gpu", device=device))
        if path is None:
            raise ValueError(f"{device}未完成验收")
        refs[device] = {"path": str(path), **io.identity(path / "manifest.json")}
    required = sum(sum(COUNTS.values())*np.prod(rep["shape"])*4 for rep in source["representations"].values())
    required += 80*1024**3  # checkpoint、逐窗口输出、signals、失败现场及余量。
    if shutil.disk_usage(OUTPUT_ROOT).free < required:
        raise RuntimeError(f"正式准备至少需要 {required} bytes可用磁盘")
    output = OUTPUT_ROOT / ("session_" + io.stamp())
    output.mkdir(parents=True, exist_ok=False)
    try:
        review = preprocessing_review(config("B", SEEDS[0], output / "unused"), output, source)
        provenance = io.provenance(output)
        if provenance["files"] != current:
            raise RuntimeError("正式准备过程中源码变化")
        io.write_json(output / "session.json", {"protocol": PROTOCOL, "spec": load_spec(), "plan": plan(),
                      "representations": source["representations"], "calibration": source["calibration"],
                      "provenance": provenance, "execution_scope": "train_validation", "research_test_open": False,
                      "parameter_review": review, "parameter_review_file": io.identity(output / "parameter_review.json"),
                      "engineering_reference": {"session": {"path": str(source_session), **io.identity(source_session / "session.json")},
                                                "devices": refs, "source_changes": changes, "rerun": False},
                      "minimum_disk_bytes": int(required)})
        io.write_json(output / "session_receipt.json", io.identity(output / "session.json"))
        from .engineering import require_gpu
        for device in devices:
            require_gpu(output, device)
    except BaseException:
        import traceback
        io.write_json(output / "prepare_failed.json", {"traceback": traceback.format_exc()})
        raise
    return output


def execute(session, devices):
    """先满足正式运行已有门控，再派发GPU矩阵；失败不自动缩减或改参数。"""
    from .data import prepare_data, build_cache
    from .signals import run_signals
    from .runtime import run_parallel
    key = io.binding(session, "formal_pipeline", devices=devices)
    with io.attempt(session / "pipeline", key) as output:
        io.write_json(output / "authorization.json", {"formal_cells": 60, "epochs": 80, "devices": devices,
                      "real_data_splits": ["train", "val"], "research_test": False})
        def step(name, action):
            print(f"PHASE_START={name}", flush=True)
            start = time.perf_counter()
            result = action()
            io.write_json(output / f"{name}.json", {"output": str(result), "elapsed_seconds": time.perf_counter()-start})
            print(f"PHASE_COMPLETE={name} OUTPUT={result}", flush=True)
        step("prepare_data", lambda: prepare_data(session))
        step("cache_C_20", lambda: build_cache(session, "C_20"))
        step("signals", lambda: run_signals(session))
        for arm in ARMS:
            if arm != "C_20":
                step(f"cache_{arm}", lambda arm=arm: build_cache(session, arm))
        step("formal_train_validation", lambda: run_parallel(session, devices))
        io.write_json(output / "completion.json", {"formal_cells": 60, "completed": True, "research_test": False})
    return output
