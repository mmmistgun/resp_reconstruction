"""1/2/4 秒 patch × 三 seed 的训练、恢复和汇总入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resp_train import patch_aligned_tf_training as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="显示矩阵；不读取真实数据")
    plan.add_argument("--shard-index", type=int, default=0)
    plan.add_argument("--shard-count", type=int, default=1)
    for name in ("prepare", "status", "summarize", "smoke", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--session", type=Path, required=True)
        if name in {"run", "smoke"}:
            sub.add_argument("--device", default="cuda:0")
        if name == "smoke":
            sub.add_argument("--patch-seconds", type=int, choices=run.PATCH_SECONDS, nargs="+", default=list(run.PATCH_SECONDS))
        if name == "run":
            sub.add_argument("--shard-index", type=int, default=0)
            sub.add_argument("--shard-count", type=int, default=1)
            sub.add_argument("--patch-seconds", type=int, choices=run.PATCH_SECONDS)
            sub.add_argument("--seed", type=int, choices=run.SEEDS)
            sub.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        result = run.plan(args.shard_index, args.shard_count)
    elif args.command == "prepare":
        result = str(run.prepare(args.session))
    elif args.command == "status":
        result = run.status(args.session)
    elif args.command == "summarize":
        result = str(run.summarize(args.session))
    elif args.command == "smoke":
        result = [str(run.smoke(args.session, p, args.device)) for p in args.patch_seconds]
    else:
        if (args.patch_seconds is None) != (args.seed is None):
            parser.error("单组运行必须同时指定 --patch-seconds 和 --seed")
        if args.patch_seconds is not None and (args.shard_index != 0 or args.shard_count != 1):
            parser.error("单组选择与分片参数不能同时使用")
        cells = ([{"patch_seconds": args.patch_seconds, "seed": args.seed}] if args.patch_seconds is not None
                 else run.plan(args.shard_index, args.shard_count))
        result = []
        for cell in cells:
            result.append(run.run_cell(args.session, cell["patch_seconds"], cell["seed"], args.device, resume=args.resume))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
