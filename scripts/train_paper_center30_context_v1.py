from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.paper_evidence.center30_config import load_center30_config
from resp_train.paper_evidence.center30_experiment import Center30Experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="运行独立 center-30 短窗口上下文任务")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--confirm-formal-training", action="store_true")
    args = parser.parse_args()
    cfg = load_center30_config(args.config, overrides=args.overrides)
    if str(cfg.protocol.run_role) != "formal":
        raise SystemExit("当前入口只执行 center30 formal config")
    if not args.confirm_formal_training:
        raise SystemExit("center30 formal training 需要显式 --confirm-formal-training")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("center30 环境不满足冻结要求: " + "; ".join(problems))
    print(Center30Experiment(cfg).train())


if __name__ == "__main__":
    main()
