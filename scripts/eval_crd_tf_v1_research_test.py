from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.tf_v1_research_test import evaluate_tf_v1_research_test_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="评价冻结 C201/M/W/MS checkpoint 的 reused research-test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_test:
        raise SystemExit("必须显式传入 --confirm-research-test")
    if not str(args.device).startswith("cuda:"):
        raise SystemExit("CRD-TF research-test 必须显式使用 cuda:<index>")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("CRD-TF research-test 环境不满足冻结要求: " + "; ".join(problems))
    print(
        evaluate_tf_v1_research_test_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device,
        ).resolve()
    )


if __name__ == "__main__":
    main()

