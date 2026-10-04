"""APOR激活规范化：验收两卡后完成U1三seed训练及U0/U1配对汇总。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"

import torch
from scripts import apor_activation_v1_runtime as experiment
from scripts.apor_activation_v1_model import ARMS, TRAIN_ARMS, SEEDS, ActivationModel, model_contract


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description="APOR激活规范化：复用U0三seed，新训练U1三seed")
    commands = parser.add_subparsers(dest="phase", required=True)
    for name in ("plan", "describe", "prepare"):
        commands.add_parser(name)
    equivalence = commands.add_parser("equivalence", help="隔离进程中的synthetic GPU严格等价复核")
    equivalence.add_argument("--device", required=True)
    equivalence.add_argument("--output", type=Path, required=True)
    equivalence.add_argument("--confirm-synthetic-gpu", action="store_true", required=True)
    for phase in ("status", "summarize", "gpu-acceptance", "formal", "run-all", "parallel", "pipeline", "run-shard"):
        sub = commands.add_parser(phase)
        sub.add_argument("--session", type=Path, required=phase in {"status", "summarize", "formal", "run-shard"})
        if phase in {"gpu-acceptance", "formal", "run-all", "parallel", "pipeline", "run-shard"}:
            if phase in {"parallel", "pipeline"}:
                sub.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
            else:
                sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--retry-failed", action="store_true")
            sub.add_argument("--confirm-synthetic-gpu" if phase == "gpu-acceptance" else "--confirm-training", action="store_true", required=True)
            if phase == "pipeline":
                sub.add_argument("--confirm-research-test", action="store_true", required=True)
        if phase == "formal":
            sub.add_argument("--arm", choices=TRAIN_ARMS, required=True)
            sub.add_argument("--seed", type=int, choices=SEEDS, required=True)
        if phase == "run-shard":
            sub.add_argument("--shard-index", type=int, required=True)
            sub.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.phase == "plan":
        experiment.load_spec()
        result = {"protocol": experiment.PROTOCOL, "scientific_cells": experiment.scientific_plan(), "new_training": experiment.plan()}
    elif args.phase == "describe":
        result = {arm: {**model_contract(arm), "actual_parameters": sum(p.numel() for p in ActivationModel(arm, SEEDS[0]).parameters())} for arm in ARMS}
    elif args.phase == "prepare":
        result = {"session": str(experiment.prepare())}
    elif args.phase == "status":
        result = experiment.status(args.session.resolve())
    elif args.phase == "equivalence":
        experiment.run_equivalence(args.device, args.output)
        result = {"equivalence_output": str(args.output)}
    else:
        session = args.session.resolve() if args.session else experiment.prepare()
        print(f"SESSION={session}", flush=True)
        if args.phase == "pipeline":
            result = run_pipeline(session, args.devices, args.retry_failed)
        elif args.phase == "parallel":
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


def run_pipeline(session, devices, retry_failed=False):
    """先冻结完整validation结果，再准备test allowlist；失败时不进入后续阶段。"""
    from scripts import eval_apor_activation_v1_research_test as research_test
    validation = experiment.run_parallel(session, devices, retry_failed)
    allowlist = research_test.prepare_allowlist(session)
    summary = research_test.parallel(allowlist, devices, retry_failed)
    result = {"validation_summary": str(validation), "test_allowlist": str(allowlist), "test_summary": str(summary)}
    receipt = session / "pipeline_result.json"
    if receipt.exists():
        if experiment.read_json(receipt) != result:
            raise ValueError("pipeline完成结果漂移")
    else:
        experiment.write_json(receipt, result)
    return result


if __name__ == "__main__":
    main()
