#!/usr/bin/env python3
"""ADV-v1 train/validation cache、训练、评价与完整矩阵汇总。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resp_train.aligned_dual_view.config import DEFAULT_CONFIG, load_experiment_config
from resp_train.aligned_dual_view.data import build_cache
from resp_train.aligned_dual_view.experiment import evaluate_validation, train
from resp_train.aligned_dual_view.summary import summarize_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("cache", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--set", action="append", default=[], dest="overrides")
        command.add_argument("--output", type=Path, required=True, help="尚不存在的独立输出目录")
        if name == "train":
            command.add_argument("--cache", type=Path, required=True)
    evaluate = commands.add_parser("validation")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default=None)
    summary = commands.add_parser("summary")
    summary.add_argument("--run", type=Path, action="append", required=True, dest="runs")
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--include-scale-mean", action="store_true")
    args = parser.parse_args()
    if args.action == "cache":
        result = build_cache(load_experiment_config(args.config, args.overrides), args.output)
    elif args.action == "train":
        result = train(load_experiment_config(args.config, args.overrides), cache_root=args.cache, output=args.output)
    elif args.action == "validation":
        result = evaluate_validation(run_root=args.run, cache_root=args.cache, output=args.output, device=args.device)
    else:
        result = summarize_runs(args.runs, output=args.output, include_scale_mean=args.include_scale_mean)
    print(result)


if __name__ == "__main__":
    main()
