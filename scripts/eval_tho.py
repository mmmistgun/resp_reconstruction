from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.experiments.tho import evaluate_tho_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="复评当前 THO checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--confirm-research-test",
        action="store_true",
        help="明确确认本次将读取可重复观察的 research-test；test 评价必须显式提供",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_research_test:
        raise SystemExit("test 评价需要显式添加 --confirm-research-test")
    output = evaluate_tho_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        metrics_output_path=args.metrics_output,
        overrides=args.overrides,
        split=args.split,
        confirm_research_test=args.confirm_research_test,
    )
    print(output)


if __name__ == "__main__":
    main()
