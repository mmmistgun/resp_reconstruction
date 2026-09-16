from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e2_effort_ablation import SEEDS, prepare_lock, run_formal, run_gpu_smoke, summarize


def main():
    parser = argparse.ArgumentParser(description="最终 W0 相对努力损失三 seed 消融")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="只读核验来源，保存 E2 配置与实现锁")
    gpu = commands.add_parser("gpu-smoke", help="synthetic GPU batch-1 forward/backward 验收")
    gpu.add_argument("--device", default="cuda:0")
    formal = commands.add_parser("formal", help="执行一个冻结 seed 的完整 80 epochs")
    formal.add_argument("--seed", type=int, choices=SEEDS, required=True)
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--gpu-receipt", type=Path, required=True)
    summary = commands.add_parser("summarize", help="三个完成 seed 的一次性配对汇总")
    summary.add_argument("--runs", type=Path, nargs=3, required=True)
    args = parser.parse_args()
    if args.phase == "prepare-lock":
        result = prepare_lock()
    elif args.phase == "gpu-smoke":
        result = run_gpu_smoke(args.device)
    elif args.phase == "formal":
        result = run_formal(args.seed, gpu_receipt=args.gpu_receipt.resolve(), device=args.device)
    else:
        result = summarize([p.resolve() for p in args.runs])
    print(result)


if __name__ == "__main__":
    main()
