from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.paper_evidence.p7_center60_efficiency import (
    run_center60_gpu_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P7 auxiliary 60s-input/60s-output desktop-GPU efficiency measurement"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-gpu-benchmark", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gpu_benchmark:
        raise SystemExit("必须显式传入 --confirm-gpu-benchmark")
    if args.device != "cuda:0":
        raise SystemExit("P7 center60 device 固定为 cuda:0；用 CUDA_VISIBLE_DEVICES 选择物理卡")
    problems = (
        check_required_packages()
        + check_required_packages(("ssqueezepy",))
        + check_crd_dependencies()
    )
    if problems:
        raise SystemExit("P7 center60 benchmark 环境不满足要求: " + "; ".join(problems))
    output = run_center60_gpu_benchmark(
        repo_root=REPO_ROOT,
        command=shlex.join([sys.executable, *sys.argv]),
        device=args.device,
        confirm=True,
    )
    print(output)


if __name__ == "__main__":
    main()
