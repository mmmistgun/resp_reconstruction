"""CWT 时频系列：准备、合成验收、train/validation 与机制分析。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "calibrate", "prepare", "prepare-engineering", "prepare-reuse", "prepare-data", "cache", "signals", "gpu-acceptance",
                                           "formal", "parallel", "run-shard", "recover-export", "summarize", "status",
                                           "mechanisms", "summarize-mechanisms", "prepare-allowlist"))
    parser.add_argument("--session", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--source-session", type=Path)
    parser.add_argument("--parameter-review", help="TF-S1 频率/边界及输入预处理有效带宽核查说明，随 session 保存")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--shard", type=int)
    parser.add_argument("--source-attempt", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--confirm-synthetic", action="store_true")
    parser.add_argument("--confirm-data", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--confirm-inference", action="store_true")
    args = parser.parse_args()
    from resp_train.paper_evidence.cwt_apor_v2 import spec, artifacts as io
    command = args.command
    if command == "plan":
        print(json.dumps({"spec": spec.load_spec(), "cells": spec.plan(), "comparisons": spec.comparisons()}, ensure_ascii=False, indent=2))
        return
    if command == "calibrate":
        if not args.confirm_synthetic:
            parser.error("calibrate 需要 --confirm-synthetic")
        from resp_train.paper_evidence.cwt_apor_v2.calibration import calibrate
        print(calibrate(args.retry_failed))
        return
    if command == "prepare-reuse":
        if args.source_session is None:
            parser.error("prepare-reuse需要--source-session；只复用模型无关产物")
        from resp_train.paper_evidence.cwt_apor_v2.reuse import prepare_reuse
        print(prepare_reuse(args.source_session, data_scope=False))
        return
    if command in ("prepare", "prepare-engineering"):
        if args.calibration is None or (command == "prepare" and not args.parameter_review):
            parser.error("prepare 需要 --calibration 和 --parameter-review")
        from resp_train.paper_evidence.cwt_apor_v2.calibration import prepare
        print(prepare(args.calibration.resolve(), args.parameter_review, engineering_only=command == "prepare-engineering"))
        return
    if args.session is None:
        parser.error("此阶段需要 --session")
    session = args.session.resolve()
    io.load_session(session)
    if command in ("prepare-data", "cache", "signals") and not args.confirm_data:
        parser.error("真实 train/validation 数据操作需要 --confirm-data")
    if command == "gpu-acceptance" and not args.confirm_synthetic:
        parser.error("GPU synthetic 验收需要 --confirm-synthetic")
    if command in ("formal", "parallel", "run-shard") and not args.confirm_training:
        parser.error("训练需要 --confirm-training")
    if command in ("recover-export", "mechanisms") and not args.confirm_inference:
        parser.error("模型评价需要 --confirm-inference")
    if command in ("cache", "formal", "recover-export") and args.arm not in spec.ARMS:
        parser.error("提供矩阵内的 --arm")
    if command in ("formal", "recover-export", "mechanisms") and args.seed not in spec.SEEDS:
        parser.error("提供矩阵内的 --seed")
    if command == "prepare-data":
        from resp_train.paper_evidence.cwt_apor_v2.data import prepare_data
        result = prepare_data(session, args.retry_failed)
    elif command == "cache":
        from resp_train.paper_evidence.cwt_apor_v2.data import build_cache
        result = build_cache(session, args.arm, args.retry_failed)
    elif command == "signals":
        from resp_train.paper_evidence.cwt_apor_v2.signals import run_signals
        result = run_signals(session, args.retry_failed)
    elif command == "gpu-acceptance":
        from resp_train.paper_evidence.cwt_apor_v2.engineering import run_gpu
        result = run_gpu(session, args.device, args.retry_failed)
    elif command in ("formal", "parallel", "run-shard", "recover-export"):
        from resp_train.paper_evidence.cwt_apor_v2 import runtime
        if command == "formal":
            result = runtime.run_formal(session, args.arm, args.seed, args.device, args.retry_failed)
        elif command == "parallel":
            result = runtime.run_parallel(session, args.devices, args.retry_failed)
        elif command == "run-shard":
            if args.shard is None:
                parser.error("需要 --shard")
            result = runtime.run_shard(session, args.device, args.shard, args.retry_failed)
        else:
            if args.source_attempt is None:
                parser.error("recover-export 需要 --source-attempt")
            result = runtime.recover_export(session, args.arm, args.seed, args.source_attempt, args.device, args.retry_failed)
    elif command == "summarize":
        from resp_train.paper_evidence.cwt_apor_v2.summary import summarize
        result = summarize(session, args.retry_failed)
    elif command == "mechanisms":
        from resp_train.paper_evidence.cwt_apor_v2.interventions import run_interventions
        result = run_interventions(session, args.seed, args.device, args.retry_failed)
    elif command == "summarize-mechanisms":
        from resp_train.paper_evidence.cwt_apor_v2.interventions import summarize_interventions
        result = summarize_interventions(session, args.retry_failed)
    elif command == "prepare-allowlist":
        from resp_train.paper_evidence.cwt_apor_v2.research_test import prepare_allowlist
        result = prepare_allowlist(session, args.retry_failed)
    else:
        from resp_train.paper_evidence.cwt_apor_v2.runtime import cell_parent
        records = []
        for cell in spec.plan():
            parent = cell_parent(session, **cell)
            recovery = session / "exports" / cell["arm"] / f"seed_{cell['seed']}"
            paths = sorted(parent.glob("attempt_*")) + sorted(recovery.glob("attempt_*"))
            complete = [p for p in paths if (p / "receipt.json").exists()]
            state = "completed_unverified_snapshot" if complete else "failed_or_interrupted" if paths else "pending"
            records.append({**cell, "status": state, "attempts": [str(p) for p in paths]})
        result = json.dumps(records, ensure_ascii=False, indent=2)
    print(result)


if __name__ == "__main__":
    main()
