from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK, verify_candidate_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 CRD candidate-lock checkpoint 与关联产物")
    parser.add_argument("--lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--print-checkpoints", action="store_true", help="验证通过后只按 lock 顺序输出 checkpoint 路径")
    args = parser.parse_args()
    verification = verify_candidate_lock(args.lock)
    checked = len(verification.records)
    if args.print_checkpoints:
        for record in verification.records:
            print(record["checkpoint_path"])
        return
    print(f"candidate lock 验证通过: {checked} checkpoints, {checked * 4} artifacts")


if __name__ == "__main__":
    main()
