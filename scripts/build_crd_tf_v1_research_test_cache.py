from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.tf_v1_research_test_cache import (
    DEFAULT_CALIBRATION,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    build_tf_v1_research_test_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 CRD-TF v1 input-only research-test M/W/S cache")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("CRD-TF research-test 环境不满足冻结要求: " + "; ".join(problems))
    output = build_tf_v1_research_test_cache(
        config_path=args.config,
        calibration_path=args.calibration,
        output_root=args.output_root,
        show_progress=not args.no_progress,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()

