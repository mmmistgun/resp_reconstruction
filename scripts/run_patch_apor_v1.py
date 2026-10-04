"""PATCH/APOR统一实验命令；run-all依次验收、训练39个cell并汇总。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence import patch_apor_v1 as experiment
from resp_train.paper_evidence.patch_apor_v1_model import ARMS, SEEDS, PatchAporModel, model_contract


def main():
    parser = argparse.ArgumentParser(description="PATCH/TM0/REF0模块消融与APOR：13臂×3 seed")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("plan", help="输出39-cell计划；不读取真实数据")
    commands.add_parser("describe", help="构造CPU模型并输出模块与参数数量")
    commands.add_parser("prepare", help="保存独立会话源码快照与配置；不读取真实数据")
    for phase in ("status", "summarize", "gpu-acceptance", "formal", "run-all", "parallel", "run-shard"):
        sub = commands.add_parser(phase)
        sub.add_argument("--session", type=Path, required=phase in {"status", "summarize", "formal", "run-shard"})
        if phase in {"gpu-acceptance", "formal", "run-all", "parallel", "run-shard"}:
            if phase == "parallel":
                sub.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
            else:
                sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--retry-failed", action="store_true", help="保留原失败/中断产物，使用新attempt重试")
            flag = "--confirm-synthetic-gpu" if phase == "gpu-acceptance" else "--confirm-training"
            sub.add_argument(flag, action="store_true", required=True)
        if phase == "formal":
            sub.add_argument("--arm", choices=ARMS, required=True)
            sub.add_argument("--seed", type=int, choices=SEEDS, required=True)
        if phase == "run-shard":
            sub.add_argument("--shard-index", type=int, required=True)
            sub.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.phase == "plan":
        experiment.load_spec()
        result = {"protocol": experiment.PROTOCOL, "cells": experiment.plan(), "count": len(experiment.plan())}
    elif args.phase == "describe":
        result = {}
        for arm in ARMS:
            model = PatchAporModel(arm, SEEDS[0])
            result[arm] = {**model_contract(arm), "parameters": sum(p.numel() for p in model.parameters())}
    elif args.phase == "prepare":
        result = {"session": str(experiment.prepare())}
    elif args.phase == "status":
        result = experiment.status(args.session.resolve())
    else:
        session = args.session.resolve() if args.session else experiment.prepare()
        print(f"SESSION={session}", flush=True)
        if args.phase == "parallel":
            result = experiment.run_parallel(session, args.devices, args.retry_failed)
        elif args.phase == "run-shard":
            result = experiment.execute_shard(session, args.device, args.shard_index, args.shard_count, args.retry_failed)
        else:
            result = experiment.execute(session, getattr(args, "device", "cuda:0"), phase=args.phase,
                                        arm=getattr(args, "arm", None), seed=getattr(args, "seed", None),
                                        retry_failed=getattr(args, "retry_failed", False))
        if isinstance(result, Path):
            result = {"output": str(result), "session": str(session)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
