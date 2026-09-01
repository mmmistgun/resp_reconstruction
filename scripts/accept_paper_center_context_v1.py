from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.center_context_acceptance import run_p2_gpu_acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description="中心上下文任务 P2 synthetic GPU 工程验收")
    parser.add_argument(
        "--config",
        default="configs/paper_evidence_v1/center_context_v1.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        default="runs/paper_evidence_v1/center_context/p2_gpu_acceptance",
    )
    parser.add_argument("--confirm-gpu-acceptance", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gpu_acceptance:
        raise SystemExit("P2 GPU acceptance 未授权：需要显式 --confirm-gpu-acceptance")
    print(
        run_p2_gpu_acceptance(
            config_path=args.config,
            output_root=args.output_root,
            device=args.device,
            command=list(sys.argv),
        )
    )


if __name__ == "__main__":
    main()
