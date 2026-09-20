from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.w0_cwt_film_behavior import SEEDS
from resp_train.paper_evidence.w0_film_local_sensitivity import (
    prepare_lock,
    run_analysis,
    run_finalize,
    run_smoke,
    run_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 W0 FiLM gamma/beta 局部剂量敏感性")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="核验来源并创建不可覆盖 implementation lock")

    smoke = commands.add_parser("smoke", help="真实 batch-1 原生/显式 0.5 融合一致性")
    smoke.add_argument("--device", default="cuda:0")

    analyze = commands.add_parser("analyze", help="单 seed 四个局部干预的完整 validation")
    analyze.add_argument("--seed", type=int, choices=SEEDS, required=True)
    analyze.add_argument("--device", default="cuda:0")
    analyze.add_argument("--smoke-receipt", type=Path, required=True)

    summarize = commands.add_parser("summarize", help="汇总三个完成的 seed attempt")
    summarize.add_argument("--runs", type=Path, nargs=3, required=True)

    finalize = commands.add_parser("finalize", help="闭合汇总和中文结论")
    finalize.add_argument("--summary", type=Path, required=True)

    args = parser.parse_args()
    if args.phase == "prepare-lock":
        print(prepare_lock())
    elif args.phase == "smoke":
        print(run_smoke(device=args.device))
    elif args.phase == "analyze":
        print(
            run_analysis(
                seed=args.seed,
                device=args.device,
                smoke_receipt=args.smoke_receipt,
            )
        )
    elif args.phase == "summarize":
        print(run_summary(runs=args.runs))
    else:
        print(run_finalize(summary=args.summary))


if __name__ == "__main__":
    main()
