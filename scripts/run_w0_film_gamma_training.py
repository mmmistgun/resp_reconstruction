from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import FORMAL_SEEDS, check_crd_dependencies
from resp_train.crd.w0_film_gamma_training import prepare_lock, run_summary, run_training


def _require_environment() -> None:
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("W0 gamma training 环境不满足冻结要求: " + "; ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description="W0 FiLM gamma 系数训练验证")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-lock", help="创建不可覆盖 implementation lock")
    commands.add_parser("acceptance", help="运行 GAMMA_030 单次 GPU acceptance")
    formal = commands.add_parser("formal", help="运行一个固定 condition×seed formal")
    formal.add_argument("--condition", choices=("GAMMA_030", "GAMMA_040"), required=True)
    formal.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    summary = commands.add_parser("summarize", help="汇总完整 2×3 formal 矩阵")
    summary.add_argument("--runs", type=Path, nargs=6, required=True)
    args = parser.parse_args()

    if args.phase == "prepare-lock":
        print(prepare_lock())
        return
    if args.phase == "summarize":
        print(run_summary(runs=args.runs))
        return
    _require_environment()
    if args.phase == "acceptance":
        print(run_training(role="acceptance", condition="GAMMA_030"))
    else:
        print(run_training(role="formal", condition=args.condition, seed=args.seed))


if __name__ == "__main__":
    main()
