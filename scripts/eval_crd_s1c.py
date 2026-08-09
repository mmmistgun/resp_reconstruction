from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK
from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.confirmation import evaluate_crd_s1c_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="受控评价 CRD S1C frozen checkpoint 的现有 research-test")
    parser.add_argument("--checkpoint", required=True, help="candidate lock 中的 checkpoint_best_local_rr.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()

    dependency_problems = check_crd_dependencies()
    if dependency_problems:
        raise SystemExit("CRD 依赖检查失败:\n" + "\n".join(f"- {problem}" for problem in dependency_problems))
    output = evaluate_crd_s1c_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        confirm_research_test=args.confirm_research_test,
        lock_path=args.candidate_lock,
    )
    print(output)


if __name__ == "__main__":
    main()
