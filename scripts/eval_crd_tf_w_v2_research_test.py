from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import FORMAL_SEEDS, check_crd_dependencies
from resp_train.crd.tf_w_v2 import CANDIDATE_LOCK
from resp_train.crd.tf_w_v2_p5 import ALLOWLIST, evaluate_p5_research_test_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="评价冻结 W3 × 3 checkpoint 的最小 reused research-test")
    parser.add_argument("--candidate-lock", default=str(CANDIDATE_LOCK))
    parser.add_argument("--checkpoint-allowlist", default=str(ALLOWLIST))
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_test:
        raise SystemExit("必须显式传入 --confirm-research-test")
    if not str(args.device).startswith("cuda:"):
        raise SystemExit("CRD-TF-W v2 P5 research-test 必须显式使用 cuda:<index>")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("CRD-TF-W v2 P5 research-test 环境不满足冻结要求: " + "; ".join(problems))
    seeds = (int(args.seed),) if args.seed is not None else FORMAL_SEEDS
    for seed in seeds:
        output = evaluate_p5_research_test_seed(
            candidate_lock=args.candidate_lock,
            allowlist_path=args.checkpoint_allowlist,
            seed=seed,
            device=args.device,
        )
        print(f"P5 research-test complete/verified seed={seed}")
        print(output.resolve())


if __name__ == "__main__":
    main()
