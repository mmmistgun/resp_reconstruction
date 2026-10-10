"""原始 M4 组件消融的参照审计、合成验收、训练、恢复和汇总入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import m4_components_runtime as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--include-references", action="store_true")
    sub.add_parser("audit-references")
    for command in ("prepare", "status", "smoke", "run", "summarize"):
        p = sub.add_parser(command)
        p.add_argument("--session", type=Path, required=True)
        if command in {"smoke", "run"}:
            p.add_argument("--device", default="cuda:0")
        if command == "smoke":
            p.add_argument("--arms", choices=run.TRAIN_ARMS, nargs="+", default=list(run.TRAIN_ARMS))
        if command == "run":
            p.add_argument("--arm", choices=run.TRAIN_ARMS)
            p.add_argument("--seed", type=int, choices=run.SEEDS)
            p.add_argument("--shard-index", type=int, default=0)
            p.add_argument("--shard-count", type=int, default=1)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        result = run.plan(args.shard_index, args.shard_count, include_references=args.include_references)
    elif args.command == "audit-references":
        result = run.audit_references()
    elif args.command == "prepare":
        result = str(run.prepare(args.session))
    elif args.command == "status":
        result = run.status(args.session)
    elif args.command == "summarize":
        result = str(run.summarize(args.session))
    elif args.command == "smoke":
        result = [str(run.ensure_smoke(args.session, a, args.device)) for a in args.arms]
    else:
        if (args.arm is None) != (args.seed is None):
            parser.error("单组运行须同时指定 --arm 和 --seed")
        if args.arm is not None and (args.shard_index != 0 or args.shard_count != 1):
            parser.error("单组选择与分片参数不能同时使用")
        cells = [{"arm": args.arm, "seed": args.seed}] if args.arm is not None else run.plan(args.shard_index, args.shard_count)
        result = [run.run_cell(args.session, c["arm"], c["seed"], args.device, resume=args.resume) for c in cells]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
