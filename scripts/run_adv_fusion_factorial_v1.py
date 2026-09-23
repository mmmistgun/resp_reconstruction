#!/usr/bin/env python3
"""ADV 融合方式 × 位置：独立 train/validation 实验入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resp_fusion.config import DEFAULT_CONFIG, load_experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("describe", "gpu-synthetic", "cache", "train", "validation", "summary"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument("--confirm-gpu", action="store_true")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config, args.override)
    if args.action != "describe" and args.output is None:
        parser.error("此动作要求 --output（全新目录）")
    if args.action in {"train", "validation"} and args.cache is None:
        parser.error("此动作要求 --cache")
    if args.action == "validation" and args.run is None:
        parser.error("validation 要求 --run")
    if args.action == "summary" and args.runs is None:
        parser.error("summary 要求完整 --runs")
    if args.action == "describe":
        from resp_fusion.engineering import describe
        print(json.dumps(describe(cfg), ensure_ascii=False, indent=2))
    elif args.action == "gpu-synthetic":
        from resp_fusion.engineering import gpu_synthetic
        print(gpu_synthetic(cfg, args.output, confirm_gpu=args.confirm_gpu))
    elif args.action == "cache":
        from resp_fusion.data import build_cache
        print(build_cache(cfg, args.output))
    elif args.action == "train":
        from resp_fusion.experiment import train
        print(train(cfg, cache_root=args.cache, output=args.output))
    elif args.action == "validation":
        from resp_fusion.experiment import evaluate_validation
        print(evaluate_validation(run_root=args.run, cache_root=args.cache, output=args.output,
                                  device=str(cfg.training.device)))
    else:
        from resp_fusion.summary import summarize_runs
        print(summarize_runs(args.runs, output=args.output))


if __name__ == "__main__":
    main()
