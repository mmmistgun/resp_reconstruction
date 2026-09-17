from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e4_scale_aggregation import ARM, SEEDS, prepare_lock, run_formal, summarize
from resp_train.paper_evidence.e4_scale_aggregation_engineering import run_gpu_acceptance, run_benchmark, benchmark_worker


def main():
    parser = argparse.ArgumentParser(description="E4：W0 四区域尺度聚合三 seed 实验")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="核验冻结来源字节身份并创建独立实现锁")
    for phase in ("gpu-acceptance", "benchmark"):
        sub = commands.add_parser(phase, help="用户执行的 synthetic GPU 阶段")
        sub.add_argument("--device", default="cuda:0")
    formal = commands.add_parser("formal", help="完整训练一个固定 seed")
    formal.add_argument("--seed", type=int, choices=SEEDS, required=True)
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--gpu-receipt", type=Path, required=True)
    summary = commands.add_parser("summarize", help="三个完整 seed 的一次性 validation 配对汇总")
    summary.add_argument("--runs", type=Path, nargs=3, required=True)
    worker = commands.add_parser("_benchmark-worker", help="内部测量子进程，由 benchmark 自动调用")
    worker.add_argument("--arm", choices=("W0_FULL", ARM), required=True)
    worker.add_argument("--mode", choices=("eval", "train"), required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare-lock":
        output = prepare_lock()
    elif args.phase == "gpu-acceptance":
        output = run_gpu_acceptance(args.device)
    elif args.phase == "benchmark":
        output = run_benchmark(args.device)
    elif args.phase == "formal":
        output = run_formal(args.seed, gpu_receipt=args.gpu_receipt.resolve(), device=args.device)
    elif args.phase == "summarize":
        output = summarize([path.resolve() for path in args.runs])
    else:
        output = benchmark_worker(arm=args.arm, mode=args.mode, device=args.device, destination=args.output.resolve())
    print(output)


if __name__ == "__main__":
    main()
