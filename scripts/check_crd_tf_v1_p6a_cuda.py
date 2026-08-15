from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.tf_v1_p6a import run_p6a_cuda_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD-TF v1 P6a 三个新 variant CUDA synthetic 验收")
    parser.add_argument("--config", default="configs/crd_tf_v1/crd_tf401_mws_add_smoke.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="runs/crd_tf_v1/p6a_cuda_synthetic")
    args = parser.parse_args()
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("CRD-TF P6a 环境不满足冻结要求: " + "; ".join(problems))
    run_p6a_cuda_synthetic(
        config_path=args.config,
        device_name=args.device,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()

