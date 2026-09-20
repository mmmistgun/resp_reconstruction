from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.w0_cwt_film_behavior import SEEDS
from resp_train.paper_evidence.w0_cwt_film_behavior_runtime import (
    prepare_lock,
    run_analysis,
    run_cases,
    run_finalize,
    run_smoke,
    run_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 W0 CWT-FiLM 调制行为分析")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="核验冻结来源并创建不可覆盖 implementation lock")

    smoke = commands.add_parser("smoke", help="真实 W0/validation batch-1 非侵入采集检查")
    smoke.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    smoke.add_argument("--device", default="cuda:0")

    analyze = commands.add_parser("analyze", help="单 seed 完整 validation 调制统计")
    analyze.add_argument("--seed", type=int, choices=SEEDS, required=True)
    analyze.add_argument("--device", default="cuda:0")
    analyze.add_argument("--smoke-receipt", type=Path, required=True)

    summarize = commands.add_parser("summarize", help="汇总三个完成的 seed attempt")
    summarize.add_argument("--runs", type=Path, nargs=3, required=True)

    cases = commands.add_parser("render-cases", help="只重推预定义典型窗口")
    cases.add_argument("--summary", type=Path, required=True)
    cases.add_argument("--device", default="cuda:0")

    finalize = commands.add_parser("finalize", help="闭合 summary、案例图与中文结论")
    finalize.add_argument("--summary", type=Path, required=True)
    finalize.add_argument("--cases", type=Path, required=True)

    args = parser.parse_args()
    if args.phase == "prepare-lock":
        print(prepare_lock())
    elif args.phase == "smoke":
        print(run_smoke(seed=args.seed, device=args.device))
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
    elif args.phase == "render-cases":
        print(run_cases(summary=args.summary, device=args.device))
    else:
        print(run_finalize(summary=args.summary, cases=args.cases))


if __name__ == "__main__":
    main()
