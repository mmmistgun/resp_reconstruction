from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.tf_v1_calibration import DEFAULT_OUTPUT_ROOT, run_tf_v1_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD-TF v1 synthetic/input-only calibration")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    print(run_tf_v1_calibration(output_root=args.output_root, require_clean=True))


if __name__ == "__main__":
    main()
