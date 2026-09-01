from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.paper_evidence.center_context_config import load_center_context_config
from resp_train.paper_evidence.center_context_experiment import CenterContextExperiment


def main() -> None:
    parser = argparse.ArgumentParser(description="运行独立的中心 60 s 变长上下文任务")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--confirm-formal-training", action="store_true")
    args = parser.parse_args()
    cfg = load_center_context_config(args.config, overrides=args.overrides)
    role = str(cfg.protocol.run_role)
    if role == "formal" and not args.confirm_formal_training:
        raise SystemExit("formal training 未授权：需要显式 --confirm-formal-training")
    if role != "formal":
        raise SystemExit(f"当前入口不执行 run_role={role!r}")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("中心上下文环境不满足冻结要求: " + "; ".join(problems))
    print(CenterContextExperiment(cfg).train())


if __name__ == "__main__":
    main()
