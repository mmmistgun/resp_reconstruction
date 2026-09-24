from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e6_temporal_frontend_test import (
    SEEDS,
    prepare_lock,
    run_evaluation,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E6 固定 validation-selected checkpoints 的 research-test 评价"
    )
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="准备 E6 test 专项锁；不读取 test 数组")
    evaluate = commands.add_parser("evaluate", help="完整 test 评价一个预锁定 seed")
    evaluate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    summary = commands.add_parser("summarize", help="三个完成 seed 与冻结 W0 test 配对汇总")
    summary.add_argument("--runs", type=Path, nargs=3, required=True)
    args = parser.parse_args()
    if args.phase == "prepare-lock":
        output = prepare_lock()
    elif args.phase == "evaluate":
        output = run_evaluation(args.seed, args.device)
    else:
        output = summarize([path.resolve() for path in args.runs])
    print(output)


if __name__ == "__main__":
    main()
