"""按最终五指标一次性评价原始 W0 三个固定 checkpoint。"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的新产物目录")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error("本专项 test 访问要求 --confirm-research-test")
    from resp_train.paper_evidence.w0_final_evaluation import evaluate
    print(evaluate(args.output, device=args.device, command=shlex.join([sys.executable, *sys.argv])))


if __name__ == "__main__":
    main()
