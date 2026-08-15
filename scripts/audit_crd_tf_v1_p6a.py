from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.tf_v1_p6a import audit_p6a_acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 CRD-TF v1 P6a synthetic 与 batch-128 acceptance")
    parser.add_argument("--synthetic-receipt", required=True)
    parser.add_argument("--mws-gate-run", required=True)
    parser.add_argument("--ctrl-gate-run", required=True)
    parser.add_argument("--output-root", default="runs/crd_tf_v1/p6a_acceptance_audit")
    args = parser.parse_args()
    output = audit_p6a_acceptance(
        synthetic_receipt=args.synthetic_receipt,
        run_dirs={
            "crd_tf402_mws_gate": args.mws_gate_run,
            "crd_tf403_ctrl_gate": args.ctrl_gate_run,
        },
        output_root=args.output_root,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()

