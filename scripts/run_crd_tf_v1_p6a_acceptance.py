from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies, load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.tf_v1_p3 import require_clean_git
from resp_train.crd.tf_v1_p6a import P6A_ACCEPTANCE_VARIANTS, p6a_acceptance_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD-TF v1 P6a gate arms batch-128 acceptance")
    parser.add_argument("--variant", required=True, choices=P6A_ACCEPTANCE_VARIANTS)
    parser.add_argument("--config", default="configs/crd_tf_v1/crd_tf401_mws_add_smoke.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="runs/crd_tf_v1/p6a_acceptance")
    args = parser.parse_args()
    if not str(args.device).startswith("cuda:"):
        raise SystemExit("P6a acceptance 必须显式使用 cuda:<index>")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("CRD-TF P6a 环境不满足冻结要求: " + "; ".join(problems))
    require_clean_git()
    cfg = load_crd_config(
        args.config,
        overrides=p6a_acceptance_overrides(
            args.variant, device=args.device, output_root=args.output_root
        ),
    )
    print(CRDExperiment(cfg).train())


if __name__ == "__main__":
    main()

