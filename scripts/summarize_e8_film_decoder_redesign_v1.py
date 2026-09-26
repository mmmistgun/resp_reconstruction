from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e8_film_decoder_redesign_v1_summary import run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E8 P5 完整 validation 汇总")
    parser.add_argument("--confirm-p5-summary", action="store_true")
    args = parser.parse_args()
    if not args.confirm_p5_summary:
        raise SystemExit("E8 P5 summary 要求显式传入 --confirm-p5-summary")
    print(run_summary())


if __name__ == "__main__":
    main()
