from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.experiment import evaluate_crd_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="复评 CRD-v1.1 validation checkpoint（S0/S1 不开放 research-test）")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    problems = check_crd_dependencies()
    if problems:
        raise SystemExit("CRD 环境不满足冻结要求: " + "; ".join(problems))
    output = evaluate_crd_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        metrics_output_path=args.metrics_output,
        overrides=args.overrides,
    )
    print(output)


if __name__ == "__main__":
    main()
