from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.tf_v1_p4 import formal_plan, matrix_identity, preflight_p4, run_formal_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD-TF v1 完整 45-run P4 formal matrix")
    parser.add_argument("--confirm-45-run-matrix", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只验证并打印冻结计划，不启动训练")
    parser.add_argument("--retry-failed", action="store_true", help="从头重跑 state 中明确失败的当前 run")
    parser.add_argument("--retry-interrupted", action="store_true", help="从头重跑 state 中中断/残留 running 的当前 run")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_45_run_matrix:
        raise SystemExit("启动 P4 必须显式传 --confirm-45-run-matrix")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("CRD-TF 环境不满足冻结要求: " + "; ".join(problems))
    commit = preflight_p4()
    if args.dry_run:
        payload = matrix_identity(commit)
        payload["run_count"] = len(formal_plan())
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    state_path = run_formal_matrix(
        retry_failed=args.retry_failed,
        retry_interrupted=args.retry_interrupted,
    )
    print(state_path.resolve())


if __name__ == "__main__":
    main()
