"""CWT 固定 checkpoint research-test 入口；不包含训练动作。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-data", "cache", "evaluate", "summarize", "mechanisms", "summarize-mechanisms"))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-research-test", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.command in ("prepare-data", "cache", "evaluate", "mechanisms") and not args.confirm_research_test:
        parser.error("此命令需要当次 --confirm-research-test")
    from resp_train.paper_evidence.cwt_apor_v2 import research_test as test
    from resp_train.paper_evidence.cwt_apor_v2.spec import ARMS, SEEDS
    if args.command in ("cache", "evaluate") and args.arm not in ARMS:
        parser.error("需要矩阵内的 --arm")
    if args.command in ("evaluate", "mechanisms") and args.seed not in SEEDS:
        parser.error("需要矩阵内的 --seed")
    session = args.session.resolve()
    if args.command == "prepare-data":
        result = test.prepare_test_data(session, args.confirm_research_test, args.retry_failed)
    elif args.command == "cache":
        result = test.build_test_cache(session, args.arm, args.confirm_research_test, args.retry_failed)
    elif args.command == "evaluate":
        result = test.evaluate(session, args.arm, args.seed, args.device, args.confirm_research_test, args.retry_failed)
    elif args.command == "mechanisms":
        result = test.mechanisms(session, args.seed, args.device, args.confirm_research_test, args.retry_failed)
    elif args.command == "summarize-mechanisms":
        result = test.summarize_mechanisms(session, args.retry_failed)
    else:
        result = test.summarize_test(session, args.retry_failed)
    print(result)


if __name__ == "__main__":
    main()
