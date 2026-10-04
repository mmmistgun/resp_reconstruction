#!/usr/bin/env python3
"""H-only 消融实验入口；plan 不访问真实数据，执行阶段需要显式确认。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resp_train.paper_evidence.h_only_ablation_v1.spec import ARMS, SEEDS, contract, plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--session", type=Path, required=True)
    prepare.add_argument("--source-session", type=Path)
    prepare.add_argument("--source-data", type=Path)
    prepare.add_argument("--source-cache", type=Path)
    prepare.add_argument("--confirm-train-validation", action="store_true", required=True)
    for name in ("acceptance", "train", "test"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--session", type=Path, required=True)
        cmd.add_argument("--arms", nargs="+", choices=ARMS, required=True)
        cmd.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
        cmd.add_argument("--device", required=True)
        cmd.add_argument("--retry-failed", action="store_true")
        cmd.add_argument({"acceptance": "--confirm-gpu", "train": "--confirm-training", "test": "--confirm-research-test"}[name],
                         action="store_true", required=True)
    reuse = sub.add_parser("reuse-ha16")
    reuse.add_argument("--session", type=Path, required=True)
    reuse.add_argument("--seed", type=int, choices=SEEDS, required=True)
    reuse_source = reuse.add_mutually_exclusive_group(required=True)
    reuse_source.add_argument("--source-attempt", type=Path)
    reuse_source.add_argument("--source-session", type=Path)
    recover = sub.add_parser("recover-export")
    recover.add_argument("--session", type=Path, required=True)
    recover.add_argument("--arm", choices=ARMS, required=True)
    recover.add_argument("--seed", type=int, choices=SEEDS, required=True)
    recover.add_argument("--source-attempt", type=Path, required=True)
    recover.add_argument("--device", required=True)
    recover.add_argument("--retry-failed", action="store_true", required=True)
    recover.add_argument("--confirm-train-validation", action="store_true", required=True)
    for name in ("summary", "freeze", "test-prepare"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--session", type=Path, required=True)
        if name != "freeze":
            cmd.add_argument("--retry-failed", action="store_true")
        if name == "summary":
            cmd.add_argument("--split", choices=("val", "test"), default="val")
            cmd.add_argument("--confirm-research-test", action="store_true")
        if name == "test-prepare":
            cmd.add_argument("--confirm-research-test", action="store_true", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "plan":
        print(json.dumps({**contract(), "cells": plan()}, ensure_ascii=False, indent=2))
        return
    from resp_train.paper_evidence.h_only_ablation_v1 import runtime, artifacts as io
    session = args.session.resolve()
    if args.command == "prepare":
        if args.source_session is not None:
            if args.source_data is not None or args.source_cache is not None:
                raise ValueError("source-session 与直接指定路径不能同时使用")
            data, cache = runtime.preprocessing_paths(args.source_session)
        else:
            if args.source_data is None or args.source_cache is None:
                raise ValueError("需要 source-session，或同时指定 source-data/source-cache")
            data, cache = args.source_data, args.source_cache
        result = runtime.prepare(session, data, cache)
    elif args.command == "reuse-ha16":
        source_attempt = args.source_attempt
        if args.source_session is not None:
            from resp_train.paper_evidence.cwt_apor_v2.runtime import cell_result as source_cell
            item = source_cell(args.source_session.resolve(), "H", args.seed)
            if item is None:
                raise RuntimeError("历史 H-only cell 未完成")
            source_attempt = item[0]
        result = runtime.reuse_h65(session, args.seed, source_attempt)
    elif args.command == "recover-export":
        result = runtime.recover_export(session, args.arm, args.seed, args.source_attempt, args.device, args.retry_failed)
    elif args.command == "summary":
        from resp_train.paper_evidence.h_only_ablation_v1.report import summarize
        if args.split == "test" and not args.confirm_research_test:
            raise PermissionError("test汇总需要本轮显式 --confirm-research-test")
        result = summarize(session, args.split, args.retry_failed, args.confirm_research_test)
    elif args.command in ("freeze", "test-prepare"):
        from resp_train.paper_evidence.h_only_ablation_v1 import research_test
        result = (research_test.freeze(session) if args.command == "freeze" else
                  research_test.prepare_data(session, args.confirm_research_test, args.retry_failed))
    else:
        io.load_session(session)
        if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
            raise ValueError("arm/seed不能重复")
        results = []
        for arm in args.arms:
            for seed in args.seeds:
                if args.command == "acceptance":
                    from resp_train.paper_evidence.h_only_ablation_v1.engineering import acceptance
                    path = io.completed(session / "acceptance" / arm / f"seed_{seed}",
                                        io.binding(session, "acceptance", arm=arm, seed=seed))
                    result = path or acceptance(session, arm, seed, args.device, args.retry_failed)
                elif args.command == "train":
                    existing = runtime.cell_result(session, arm, seed)
                    result = existing[0] if existing else runtime.run_formal(session, arm, seed, args.device, args.retry_failed)
                else:
                    from resp_train.paper_evidence.h_only_ablation_v1 import research_test
                    existing = research_test.test_result(session, arm, seed)
                    result = existing[0] if existing else research_test.run_test(
                        session, arm, seed, args.device, args.confirm_research_test, args.retry_failed)
                print(result, flush=True)
                results.append(str(result))
        result = results
    print(result)


if __name__ == "__main__":
    main()
