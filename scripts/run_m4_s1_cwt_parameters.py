"""M4-S1：28配置×3seed的CWT参数与模块消融，数据/GPU/训练操作均显式开启。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import m4_s1_cwt_parameters_spec as spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="只列矩阵，不导入模型或读取数据")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--summary", action="store_true")
    sub.add_parser("audit-initialization", help="CPU构造，核验三seed全部臂的共有初始化")
    p = sub.add_parser("prepare", help="冻结源码、表示元数据和84个resolved config")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--scope", choices=("synthetic_only", "train_validation"), default="synthetic_only")
    p.add_argument("--confirm-data", action="store_true")
    for command in ("calibrate", "prepare-data", "cache", "import-cache", "smoke", "run", "status", "summarize"):
        p = sub.add_parser(command)
        p.add_argument("--session", type=Path, required=True)
        if command in ("calibrate", "prepare-data", "cache", "import-cache", "smoke", "summarize"):
            p.add_argument("--retry-failed", action="store_true")
        if command in ("prepare-data", "cache", "import-cache"):
            p.add_argument("--confirm-data", action="store_true")
        if command in ("cache", "import-cache"):
            select = p.add_mutually_exclusive_group(required=True)
            select.add_argument("--representation", choices=tuple(spec.CWT_SETTINGS))
            select.add_argument("--all", action="store_true")
        if command == "import-cache":
            p.add_argument("--source-session", type=Path, required=True)
        if command in ("smoke", "run"):
            p.add_argument("--device", default="cuda:0")
        if command == "smoke":
            p.add_argument("--arms", choices=tuple(spec.ARMS), nargs="+", default=list(spec.ARMS))
            p.add_argument("--confirm-synthetic", action="store_true")
        if command == "run":
            p.add_argument("--arm", choices=tuple(spec.ARMS))
            p.add_argument("--seed", type=int, choices=spec.SEEDS)
            p.add_argument("--shard-index", type=int, default=0)
            p.add_argument("--shard-count", type=int, default=1)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args(argv)
    # 先验证执行标志，再导入训练模块，错误调用不会触碰CUDA或样本。
    if args.command == "plan":
        cells = spec.plan(args.shard_index, args.shard_count)
        result = {"configurations": len(spec.ARMS), "seeds": list(spec.SEEDS), "total_cells": 84,
                  "selected_cells": len(cells), "historical_training_reused": False} if args.summary else cells
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if args.command == "prepare" and args.scope == "train_validation" and not args.confirm_data:
        parser.error("train_validation session须显式--confirm-data")
    if args.command in ("prepare-data", "cache", "import-cache") and not args.confirm_data:
        parser.error("真实数据操作须显式--confirm-data")
    if args.command == "smoke" and not args.confirm_synthetic:
        parser.error("GPU验收须显式--confirm-synthetic")
    if args.command == "run":
        if not args.confirm_training:
            parser.error("正式训练须显式--confirm-training")
        if (args.arm is None) != (args.seed is None):
            parser.error("单cell选择须同时指定--arm和--seed")
        if args.arm is not None and (args.shard_index != 0 or args.shard_count != 1):
            parser.error("单cell选择不能与分片同时使用")
    from scripts import m4_s1_cwt_parameters_runtime as run
    from scripts import m4_s1_cwt_parameters_data as data
    if args.command == "audit-initialization":
        result = run.audit_initialization()
    elif args.command == "prepare":
        result = str(run.prepare(args.session, scope=args.scope))
    elif args.command == "calibrate":
        result = str(run.calibrate(args.session, retry_failed=args.retry_failed))
    elif args.command == "prepare-data":
        result = str(data.prepare_data(args.session, confirmed=True, retry_failed=args.retry_failed))
    elif args.command in ("cache", "import-cache"):
        arms = tuple(spec.CWT_SETTINGS) if args.all else [args.representation]
        if args.command == "cache":
            result = [str(data.build_cache(args.session, a, confirmed=True, retry_failed=args.retry_failed)) for a in arms]
        else:
            result = [str(data.import_cache(args.session, a, args.source_session, confirmed=True,
                                           retry_failed=args.retry_failed)) for a in arms]
    elif args.command == "smoke":
        result = [str(run.smoke(args.session, a, args.device, confirmed=True, retry_failed=args.retry_failed)) for a in args.arms]
    elif args.command == "run":
        cells = [{"arm": args.arm, "seed": args.seed}] if args.arm else spec.plan(args.shard_index, args.shard_count)
        result = []
        for cell in cells:
            receipt = run.run_cell(args.session, cell["arm"], cell["seed"], args.device, confirmed=True, resume=args.resume)
            result.append({k: receipt[k] for k in ("arm", "seed", "best_epoch", "completed_epochs")})
    elif args.command == "status":
        result = run.status(args.session)
    else:
        result = str(run.summarize(args.session, retry_failed=args.retry_failed))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
