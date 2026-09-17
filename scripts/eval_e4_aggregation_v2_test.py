from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence import e4_aggregation_v2_test as evaluation


def main():
    parser = argparse.ArgumentParser(description="E4-v2 四候选固定 checkpoint 的完整 test 矩阵")
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("prepare-lock").add_argument("--validation-summary", type=Path, required=True)
    evaluate = phases.add_parser("evaluate")
    evaluate.add_argument("--arm", choices=evaluation.ARMS, required=True)
    evaluate.add_argument("--seed", choices=evaluation.SEEDS, type=int, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    summary = phases.add_parser("summarize").add_mutually_exclusive_group(required=True)
    summary.add_argument("--runs", nargs=12, type=Path)
    summary.add_argument("--completed", action="store_true", help="定位当前锁下12个唯一成功评价")
    args = parser.parse_args()
    if args.phase == "prepare-lock":
        output = evaluation.prepare_lock(args.validation_summary.resolve())
    elif args.phase == "evaluate":
        output = evaluation.run_evaluation(args.arm, args.seed, args.device)
    else:
        output = evaluation.summarize(evaluation.completed_runs() if args.completed else [p.resolve() for p in args.runs])
    print(output)


if __name__ == "__main__":
    main()
