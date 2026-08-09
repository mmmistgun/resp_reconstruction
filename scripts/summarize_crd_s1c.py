from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK
from resp_train.crd.selection import summarize_crd_s1c


def main() -> None:
    parser = argparse.ArgumentParser(description="审计完整 CRD S1C 结果并应用冻结 eligibility/Pareto 规则")
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    args = parser.parse_args()
    print(summarize_crd_s1c(lock_path=args.candidate_lock))


if __name__ == "__main__":
    main()
