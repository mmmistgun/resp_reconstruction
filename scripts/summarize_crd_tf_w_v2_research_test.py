from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.tf_w_v2 import CANDIDATE_LOCK
from resp_train.crd.tf_w_v2_p5 import ALLOWLIST, generate_p5_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 CRD-TF-W v2 P5 reused research-test 汇总")
    parser.add_argument("--candidate-lock", default=str(CANDIDATE_LOCK))
    parser.add_argument("--checkpoint-allowlist", default=str(ALLOWLIST))
    args = parser.parse_args()
    print(
        generate_p5_summary(
            candidate_lock=args.candidate_lock,
            allowlist_path=args.checkpoint_allowlist,
        ).resolve()
    )


if __name__ == "__main__":
    main()
