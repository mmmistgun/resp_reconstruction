from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.paper_evidence.context_length_research_test import evaluate_research_test


def main() -> None:
    parser = argparse.ArgumentParser(description="评价 center30/center60 冻结上下文 checkpoint 的独立测试集")
    parser.add_argument("--task", choices=("center30", "center60"), required=True)
    parser.add_argument("--model", choices=("c201", "wr"), required=True)
    parser.add_argument("--input-sec", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_test:
        raise SystemExit("必须显式传入 --confirm-research-test")
    if str(args.device) != "cuda:0":
        raise SystemExit("上下文 research-test 固定使用逻辑 cuda:0；物理卡由 CUDA_VISIBLE_DEVICES 选择")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("上下文 research-test 环境不满足冻结要求: " + "; ".join(problems))
    command = shlex.join([sys.executable, *sys.argv])
    print(
        evaluate_research_test(
            task=args.task,
            model_key=args.model,
            input_sec=args.input_sec,
            seed=args.seed,
            device=args.device,
            command=command,
        )
    )


if __name__ == "__main__":
    main()
