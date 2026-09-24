from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e7_scale_encoding_aggregation_engineering import (
    ARMS,
    benchmark_worker,
    load_p3_lock,
    prepare_p3_lock,
    run_benchmark,
    run_gpu_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E7 P3：GPU acceptance 与 benchmark")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="在干净提交上创建 P3 engineering lock")
    commands.add_parser("check-lock", help="只读核验 P3 engineering lock")
    for phase in ("gpu-acceptance", "benchmark"):
        sub = commands.add_parser(phase, help="用户执行的 synthetic GPU 阶段")
        sub.add_argument("--device", default="cuda:0")
    worker = commands.add_parser("_benchmark-worker", help="内部独立 benchmark 进程")
    worker.add_argument("--arm", choices=ARMS, required=True)
    worker.add_argument("--mode", choices=("eval", "train"), required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare-lock":
        output = prepare_p3_lock()
    elif args.phase == "check-lock":
        _lock, output = load_p3_lock()
    elif args.phase == "gpu-acceptance":
        output = run_gpu_acceptance(args.device)
    elif args.phase == "benchmark":
        output = run_benchmark(args.device)
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
