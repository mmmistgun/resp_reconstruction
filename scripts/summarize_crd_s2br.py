from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK
from resp_train.crd.s2br_selection import (
    DEFAULT_S2A_SUMMARY_ROOT,
    DEFAULT_S2BR_OUTPUT_ROOT,
    summarize_crd_s2br,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 CRD S2B-R validation 并应用冻结交互门槛")
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--runs-root", default="runs/crd_v1")
    parser.add_argument("--s2a-summary-root", default=str(DEFAULT_S2A_SUMMARY_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_S2BR_OUTPUT_ROOT))
    args = parser.parse_args()
    print(
        summarize_crd_s2br(
            candidate_lock_path=args.candidate_lock,
            runs_root=args.runs_root,
            s2a_summary_root=args.s2a_summary_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
