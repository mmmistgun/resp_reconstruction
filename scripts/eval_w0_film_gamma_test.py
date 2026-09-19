from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.w0_film_gamma_test import (
    SEEDS,
    prepare_lock,
    run_evaluation,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W0 FiLM GAMMA_040 固定 validation-selected checkpoints 的复用 test 评价"
    )
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="准备 test 专项身份锁；不读取 test cache 数组")
    evaluate = commands.add_parser("evaluate", help="完整评价一个固定 seed")
    evaluate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    summary = commands.add_parser("summarize", help="汇总三个完成的 test attempts")
    summary.add_argument("--runs", type=Path, nargs=3, required=True)
    args = parser.parse_args()

    if args.phase == "prepare-lock":
        result = prepare_lock()
    elif args.phase == "evaluate":
        result = run_evaluation(args.seed, args.device)
    else:
        result = summarize([path.resolve() for path in args.runs])
    print(result)


if __name__ == "__main__":
    main()

