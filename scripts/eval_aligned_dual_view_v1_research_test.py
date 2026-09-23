#!/usr/bin/env python3
"""ADV-v1 固定 12-checkpoint research-test 专项入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resp_eval.adv_test.contract import LOCK_PATH, load_lock, prepare_lock
from resp_eval.adv_test.data import build_test_cache
from resp_eval.adv_test.evaluate import evaluate_checkpoint
from resp_eval.adv_test.summary import summarize_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare-lock", help="只读 validation 来源并创建不可覆盖的候选锁")
    prepare.add_argument("--output", type=Path, default=LOCK_PATH)
    check = commands.add_parser("check-lock", help="检查固定候选锁与训练代码；不访问数据集")
    check.add_argument("--lock", type=Path, default=LOCK_PATH)
    for action in ("cache", "evaluate", "summary"):
        command = commands.add_parser(action)
        command.add_argument("--lock", type=Path, default=LOCK_PATH)
        command.add_argument("--confirm-research-test", action="store_true", required=True)
        if action in {"cache", "summary"}:
            command.add_argument("--output", type=Path, required=True)
        if action in {"evaluate", "summary"}:
            command.add_argument("--cache", type=Path, required=True)
        if action == "evaluate":
            command.add_argument("--entry", required=True, help="如 joint_seed20260811")
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--attempt", default="r1", help="失败后的新尝试标识；成功后拒绝再跑")
            command.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare-lock":
        result = prepare_lock(args.output)
    elif args.action == "check-lock":
        lock = load_lock(args.lock)
        result = f"lock_id={lock['lock_id']} checkpoints={len(lock['entries'])}; dataset not accessed"
    elif args.action == "cache":
        result = build_test_cache(args.output, lock_path=args.lock, confirm_research_test=args.confirm_research_test)
    elif args.action == "evaluate":
        result = evaluate_checkpoint(args.entry, cache_root=args.cache, lock_path=args.lock, device=args.device,
                                     attempt=args.attempt, confirm_research_test=args.confirm_research_test,
                                     show_progress=not args.quiet)
    else:
        result = summarize_test(args.cache, args.output, lock_path=args.lock,
                                confirm_research_test=args.confirm_research_test)
    print(result)


if __name__ == "__main__":
    main()
