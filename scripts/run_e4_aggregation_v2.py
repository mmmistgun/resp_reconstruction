from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence import e4_aggregation_v2 as experiment
from resp_train.paper_evidence import e4_aggregation_v2_engineering as engineering


def main():
    parser = argparse.ArgumentParser(description="E4-v2 四聚合候选×三 seed 固定矩阵")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="核验 W0 来源与频率元数据，创建实现锁")
    for phase in ("gpu-acceptance", "benchmark"):
        commands.add_parser(phase).add_argument("--device", default="cuda:0")
    formal = commands.add_parser("formal", help="完整训练一个 arm/seed")
    formal.add_argument("--arm", required=True, choices=experiment.ARMS)
    formal.add_argument("--seed", required=True, type=int, choices=experiment.SEEDS)
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--gpu-receipt", required=True, type=Path)
    summary = commands.add_parser("summarize", help="完整12个训练 attempt 的汇总").add_mutually_exclusive_group(required=True)
    summary.add_argument("--runs", nargs=12, type=Path)
    summary.add_argument("--completed", action="store_true", help="定位当前锁下12个唯一成功 attempt")
    worker = commands.add_parser("_benchmark-worker", help="内部测量子进程")
    worker.add_argument("--arm", required=True, choices=("W0_FULL", *experiment.ARMS))
    worker.add_argument("--mode", required=True, choices=("eval", "train"))
    worker.add_argument("--device", required=True)
    worker.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "prepare-lock":
        output = experiment.prepare_lock()
    elif args.phase == "gpu-acceptance":
        output = engineering.run_gpu_acceptance(args.device)
    elif args.phase == "benchmark":
        output = engineering.run_benchmark(args.device)
    elif args.phase == "formal":
        output = experiment.run_formal(args.arm, args.seed, gpu_receipt=args.gpu_receipt.resolve(), device=args.device)
    elif args.phase == "summarize":
        output = experiment.summarize(experiment.completed_runs() if args.completed else [p.resolve() for p in args.runs])
    else:
        output = engineering.benchmark_worker(args.arm, args.mode, args.device, args.output.resolve())
    print(output)


if __name__ == "__main__":
    main()
