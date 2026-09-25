from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence import e7_scale_encoding_aggregation_test as evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="E7 六臂×三 seed 固定 checkpoint research-test")
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("prepare-lock")
    phases.add_parser("check-lock")
    evaluate = phases.add_parser("evaluate")
    evaluate.add_argument("--arm", choices=evaluation.ARMS, required=True)
    evaluate.add_argument("--seed", choices=evaluation.e7.SEEDS, type=int, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    phases.add_parser("check-completed")
    summary = phases.add_parser("summarize")
    group = summary.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", nargs=len(evaluation.ARMS) * len(evaluation.e7.SEEDS), type=Path)
    group.add_argument("--completed", action="store_true")
    args = parser.parse_args()

    if args.phase == "prepare-lock":
        output = evaluation.prepare_lock()
    elif args.phase == "check-lock":
        _lock, output = evaluation.load_lock()
    elif args.phase == "evaluate":
        output = evaluation.run_evaluation(args.arm, args.seed, device=args.device)
    elif args.phase == "check-completed":
        output = "\n".join(map(str, evaluation.completed_runs()))
    else:
        runs = evaluation.completed_runs() if args.completed else [path.resolve() for path in args.runs]
        output = evaluation.summarize(runs)
    print(output)


if __name__ == "__main__":
    main()
