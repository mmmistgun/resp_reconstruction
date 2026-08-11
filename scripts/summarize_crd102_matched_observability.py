from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK
from resp_train.crd.failure_metadata import DEFAULT_OUTPUT_ROOT as DEFAULT_METADATA_ROOT
from resp_train.crd.matched_observability import DEFAULT_OUTPUT_ROOT, summarize_crd102_matched_observability


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD_102 high-modulation failure/control 配对可观测性诊断")
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--metadata-root", default=str(DEFAULT_METADATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    print(
        summarize_crd102_matched_observability(
            candidate_lock_path=args.candidate_lock,
            metadata_root=args.metadata_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
