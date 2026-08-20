from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.temporal.formal import DEFAULT_PLAN_PATH, EXPECTED_CANDIDATES, FORMAL_SEEDS, run_formal_training


def main() -> None:
    parser = argparse.ArgumentParser(description="RTM-v1 frozen single-run formal training")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--candidate-id", required=True, choices=list(EXPECTED_CANDIDATES))
    parser.add_argument("--seed", required=True, type=int, choices=list(FORMAL_SEEDS))
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_assignment = (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(visible)} " if visible is not None else ""
    )
    command = (
        "env -u LD_LIBRARY_PATH -u LD_PRELOAD "
        + visible_assignment
        + shlex.join([sys.executable, *sys.argv])
    )
    receipt = run_formal_training(
        plan_path=args.plan,
        candidate_id=args.candidate_id,
        seed=args.seed,
        command=command,
    )
    print(f"RTM-v1 formal receipt: {receipt}")


if __name__ == "__main__":
    main()
