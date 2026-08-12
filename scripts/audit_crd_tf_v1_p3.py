from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.tf_v1_p3 import audit_p3_acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 CRD-TF v1 P3 CUDA/acceptance 工程审计")
    parser.add_argument("--synthetic-receipt", required=True)
    parser.add_argument("--tf102-run", required=True)
    parser.add_argument("--tf204-run", required=True)
    parser.add_argument("--tf302-run", required=True)
    parser.add_argument("--output-root", default="runs/crd_tf_v1/p3_acceptance_audit")
    args = parser.parse_args()
    output = audit_p3_acceptance(
        synthetic_receipt=args.synthetic_receipt,
        run_dirs={
            "crd_tf102_w": args.tf102_run,
            "crd_tf204_wl": args.tf204_run,
            "crd_tf302_wls": args.tf302_run,
        },
        output_root=args.output_root,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
