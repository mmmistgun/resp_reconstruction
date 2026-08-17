from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.tf_w_v2_audit import (
    CANDIDATE_LOCK,
    run_p_minus_1_film_statistics_correction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD-TF-W v2 P−1 FiLM statistics correction")
    parser.add_argument("--candidate-lock", default=str(CANDIDATE_LOCK))
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    problems = check_crd_dependencies()
    if problems:
        raise SystemExit("CRD 环境不满足冻结要求: " + "; ".join(problems))
    output = run_p_minus_1_film_statistics_correction(
        candidate_lock_path=args.candidate_lock,
        split=args.split,
        device_name=args.device,
    )
    print(output)


if __name__ == "__main__":
    main()
