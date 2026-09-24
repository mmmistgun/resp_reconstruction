from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e7_scale_encoding_aggregation import (
    check_p1,
    load_implementation_lock,
    prepare_implementation_lock,
    prepare_source_audit,
    write_validation_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E7：聚合前尺度编码 × 尺度聚合析因实验")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("check-p1", help="只读核验六臂结构与 18 份 resolved config")
    commands.add_parser("prepare-source-audit", help="核验 train/validation 来源并创建 source audit")
    commands.add_parser("prepare-lock", help="在干净提交上创建 P1 implementation lock")
    commands.add_parser("check-lock", help="只读核验已生成的 P1 implementation lock")
    summary = commands.add_parser("summarize-seed-metrics", help="写出完整 18-cell validation 析因汇总")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "check-p1":
        print(json.dumps(check_p1(), ensure_ascii=False, indent=2))
    elif args.phase == "prepare-source-audit":
        print(prepare_source_audit())
    elif args.phase == "prepare-lock":
        print(prepare_implementation_lock())
    elif args.phase == "check-lock":
        _lock, digest = load_implementation_lock()
        print(digest)
    elif args.phase == "summarize-seed-metrics":
        print(write_validation_summary(pd.read_csv(args.input), args.output))
    else:
        raise ValueError(f"E7 未知阶段 {args.phase!r}")


if __name__ == "__main__":
    main()
