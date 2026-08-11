from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.tf_v1_cache import (
    DEFAULT_CANDIDATE_LOCK,
    DEFAULT_OUTPUT_ROOT,
    build_tf_v1_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 CRD-TF v1 train/validation fixed representation cache")
    parser.add_argument(
        "--config",
        default="configs/crd_v1/crd_c201_decoder_10hz_cap.yaml",
        help="只用于冻结 data identity；不运行模型",
    )
    parser.add_argument("--calibration", required=True, help="完整通过的 calibration.json")
    parser.add_argument("--candidate-lock", default=str(DEFAULT_CANDIDATE_LOCK))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-windows-per-split", type=int, default=None)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    print(
        build_tf_v1_cache(
            config_path=args.config,
            calibration_path=args.calibration,
            output_root=args.output_root,
            candidate_lock_path=args.candidate_lock,
            max_windows_per_split=args.max_windows_per_split,
            allow_dirty_smoke=args.allow_dirty_smoke,
            show_progress=not args.no_progress,
        )
    )


if __name__ == "__main__":
    main()
