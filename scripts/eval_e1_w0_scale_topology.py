from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.e1_scale_topology_runtime import prepare_locks, run_gpu_smoke, run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 冻结 W0 尺度重排敏感性审计")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("prepare-locks", help="只读核验 W0/validation 来源身份并生成不可覆盖的索引与实现锁")
    smoke = commands.add_parser("gpu-smoke", help="synthetic batch-1 GPU 验收")
    smoke.add_argument("--device", default="cuda:0")
    validation = commands.add_parser("validation", help="完整三 seed validation，FULL 通过后执行干预并汇总")
    validation.add_argument("--device", default="cuda:0")
    validation.add_argument("--gpu-receipt", type=Path, required=True, help="已完成 gpu-smoke 的 attempt 目录")
    args = parser.parse_args()
    if args.phase == "prepare-locks":
        for path in prepare_locks():
            print(path)
    elif args.phase == "gpu-smoke":
        print(run_gpu_smoke(device=args.device))
    else:
        print(run_validation(device=args.device, gpu_receipt=args.gpu_receipt))


if __name__ == "__main__":
    main()
