from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK
from resp_train.crd.failure_diagnostics import DEFAULT_OUTPUT_ROOT as DEFAULT_FAILURE_OUTPUT_ROOT
from resp_train.crd.failure_metadata import DEFAULT_OUTPUT_ROOT, summarize_crd102_failure_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="把 CRD_102 failure consensus 与冻结 validation index 元数据配对")
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--failure-root", default=str(DEFAULT_FAILURE_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    print(
        summarize_crd102_failure_metadata(
            candidate_lock_path=args.candidate_lock,
            failure_root=args.failure_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
