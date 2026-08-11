from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.crd.c1_selection import DEFAULT_OUTPUT_ROOT, DEFAULT_RUNS_ROOT, summarize_c1
from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD_102 vs parameter-matched full-context TCN C1 冻结汇总")
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    print(
        summarize_c1(
            candidate_lock_path=args.candidate_lock,
            runs_root=args.runs_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()

