#!/usr/bin/env python3
"""独立缓存/分组推理入口；沿用 ε-v2 数据、预算、N 选择和后滤波合同。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resp_train.respdiff_bcg.sampling_accelerated import AccelerationSpec
from scripts.run_respdiff_bcg_epsilon_inference_v2 import DEFAULT_SOURCE, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("synthetic-smoke", "subset", "full-validation"))
    parser.add_argument("--trajectory-group-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--subset-run", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.output, source_run=args.source_run, subset_run=args.subset_run, device=args.device,
        acceleration=AccelerationSpec(args.trajectory_group_size))
