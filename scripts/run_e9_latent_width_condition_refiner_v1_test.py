from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import ARMS
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_test import (
    evaluation_status,
    load_allowlist,
    prepare_allowlist,
    run_evaluation,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E9 固定 18-checkpoint research-test"
    )
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-allowlist", help="从冻结 validation/P5 生成 allowlist")
    check = commands.add_parser("check-allowlist", help="复核 allowlist、来源与代码身份")
    check.add_argument("--allowlist", required=True, type=Path)
    status = commands.add_parser("status", help="读取 allowlist 对应 18-cell 状态")
    status.add_argument("--allowlist", required=True, type=Path)
    evaluate = commands.add_parser("evaluate", help="评价一个固定 arm×seed checkpoint")
    evaluate.add_argument("--allowlist", required=True, type=Path)
    evaluate.add_argument("--arm", required=True, choices=ARMS)
    evaluate.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=(20260811, 20260812, 20260813),
    )
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--confirm-research-test", action="store_true")
    summary = commands.add_parser("summary", help="生成一次性完整 research-test 汇总")
    summary.add_argument("--allowlist", required=True, type=Path)
    summary.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()

    if args.phase == "prepare-allowlist":
        result: str | dict = str(prepare_allowlist())
    elif args.phase == "check-allowlist":
        allowlist, digest = load_allowlist(args.allowlist)
        result = {
            "allowlist_sha256": digest,
            "entries": len(allowlist["entries"]),
            "status": "passed",
            "test_array_read": False,
        }
    elif args.phase == "status":
        result = evaluation_status(args.allowlist)
    elif args.phase == "evaluate":
        if not args.confirm_research_test:
            raise SystemExit("E9 research-test 评价要求显式传入 --confirm-research-test")
        result = str(
            run_evaluation(
                args.allowlist,
                arm=args.arm,
                seed=args.seed,
                device=args.device,
            )
        )
    else:
        if not args.confirm_research_test:
            raise SystemExit("E9 research-test 汇总要求显式传入 --confirm-research-test")
        result = str(summarize(args.allowlist))
    print(
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
