from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e7_scale_encoding_aggregation_formal import (
    ARMS,
    completed_runs,
    load_p4_lock,
    prepare_p4_lock,
    run_formal,
)
from resp_train.paper_evidence.e7_scale_encoding_aggregation import SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description="E7 P4：18-cell formal train/validation")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="在干净提交上创建 P4 execution lock")
    commands.add_parser("check-lock", help="只读核验 P4 execution lock 与 P3 closeout")
    formal = commands.add_parser("formal", help="训练一个固定 arm/seed cell")
    formal.add_argument("--arm", choices=ARMS, required=True)
    formal.add_argument("--seed", type=int, choices=SEEDS, required=True)
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--gpu-receipt", type=Path, required=True)
    commands.add_parser("check-completed", help="只读核验完整 18-cell 成功矩阵")
    args = parser.parse_args()
    if args.phase == "prepare-lock":
        output = prepare_p4_lock()
    elif args.phase == "check-lock":
        _lock, output = load_p4_lock()
    elif args.phase == "formal":
        output = run_formal(
            args.arm,
            args.seed,
            gpu_receipt=args.gpu_receipt.resolve(),
            device=args.device,
        )
    else:
        output = "\n".join(map(str, completed_runs()))
    print(output)


if __name__ == "__main__":
    main()
