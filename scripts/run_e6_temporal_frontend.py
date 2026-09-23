from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e6_temporal_frontend import (
    ARM,
    SEEDS,
    load_experiment_spec,
    prepare_lock,
    run_formal,
    summarize,
)
from resp_train.paper_evidence.e6_temporal_frontend_engineering import (
    benchmark_worker,
    run_benchmark,
    run_gpu_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E6：W0 时域前端替换三 seed 实验")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("check-config", help="只读核验独立 E6 spec")
    commands.add_parser("prepare-lock", help="核验冻结来源字节身份并创建实现锁")
    for phase in ("gpu-acceptance", "benchmark"):
        sub = commands.add_parser(phase, help="用户执行的 synthetic GPU 阶段")
        sub.add_argument("--device", default="cuda:0")
    formal = commands.add_parser("formal", help="完整训练一个固定 seed")
    formal.add_argument("--seed", type=int, choices=SEEDS, required=True)
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--gpu-receipt", type=Path, required=True)
    summary = commands.add_parser("summarize", help="三个完成 seed 的 validation 配对汇总")
    summary.add_argument("--runs", type=Path, nargs=3, required=True)
    worker = commands.add_parser("_benchmark-worker", help="内部 benchmark 子进程")
    worker.add_argument("--arm", choices=("W0_FULL", ARM), required=True)
    worker.add_argument("--mode", choices=("eval", "train"), required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "check-config":
        load_experiment_spec()
        output: str | Path = "E6 config OK"
    elif args.phase == "prepare-lock":
        output = prepare_lock()
    elif args.phase == "gpu-acceptance":
        output = run_gpu_acceptance(args.device)
    elif args.phase == "benchmark":
        output = run_benchmark(args.device)
    elif args.phase == "formal":
        output = run_formal(
            args.seed,
            gpu_receipt=args.gpu_receipt.resolve(),
            device=args.device,
        )
    elif args.phase == "summarize":
        output = summarize([path.resolve() for path in args.runs])
    else:
        output = benchmark_worker(
            arm=args.arm,
            mode=args.mode,
            device=args.device,
            destination=args.output.resolve(),
        )
    print(output)


if __name__ == "__main__":
    main()
