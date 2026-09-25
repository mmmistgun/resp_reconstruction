from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e7_scale_encoding_aggregation import SEEDS
from resp_train.paper_evidence.e7_scale_encoding_aggregation_model import ARMS
from resp_train.paper_evidence.e7_scale_encoding_aggregation_p5 import (
    completed_diagnostics,
    completed_summary,
    run_diagnostic,
    run_finalize,
    run_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E7 P5：validation 汇总与 selected-checkpoint 诊断")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("summarize", help="汇总完整 18-cell validation 指标")
    diagnostic = commands.add_parser("diagnose", help="诊断一个 selected checkpoint")
    diagnostic.add_argument("--arm", choices=ARMS, required=True)
    diagnostic.add_argument("--seed", type=int, choices=SEEDS, required=True)
    diagnostic.add_argument("--device", default="cuda:0")
    commands.add_parser("check-summary", help="只读核验唯一 summary")
    commands.add_parser("check-diagnostics", help="只读核验完整 18-cell diagnostics")
    commands.add_parser("finalize", help="冻结 summary 与完整 diagnostics")
    args = parser.parse_args()
    if args.phase == "summarize":
        output = run_summary()
    elif args.phase == "diagnose":
        output = run_diagnostic(args.arm, args.seed, device=args.device)
    elif args.phase == "check-summary":
        output = completed_summary()
    elif args.phase == "check-diagnostics":
        output = "\n".join(map(str, completed_diagnostics()))
    else:
        output = run_finalize()
    print(output)


if __name__ == "__main__":
    main()
