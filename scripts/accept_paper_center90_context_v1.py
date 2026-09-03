from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.center90_acceptance import run_center90_gpu_acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description="验收 center90 最大臂 batch-128 GPU 运行合同")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        default="runs/paper_evidence_v1/center90_context/gpu_acceptance",
    )
    parser.add_argument("--confirm-gpu-acceptance", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gpu_acceptance:
        raise SystemExit("center90 GPU acceptance 需要显式 --confirm-gpu-acceptance")
    print(
        run_center90_gpu_acceptance(
            config_path=args.config,
            output_root=args.output_root,
            device=args.device,
            command=list(sys.argv),
        )
    )


if __name__ == "__main__":
    main()
