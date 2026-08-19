from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.tf_w_v2 import CANDIDATE_LOCK
from resp_train.crd.tf_w_v2_p4 import generate_p4_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="生成一次性 CRD-TF-W v2 P4 validation summary")
    parser.add_argument("--candidate-lock", default=str(CANDIDATE_LOCK))
    args = parser.parse_args()
    print(generate_p4_summary(args.candidate_lock))


if __name__ == "__main__":
    main()
