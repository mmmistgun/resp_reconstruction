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
from resp_train.paper_evidence.p6_multi_attribute import export_p6_validation_waveforms


def main() -> None:
    parser = argparse.ArgumentParser(description="导出五 rows × 三 W0 checkpoints 的 P6 validation 波形")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-gpu-export", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gpu_export:
        raise SystemExit("必须显式传入 --confirm-gpu-export")
    if not str(args.device).startswith("cuda:"):
        raise SystemExit("P6 waveform export 必须显式使用 cuda:<index>")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("P6 waveform export 环境不满足要求: " + "; ".join(problems))
    output = export_p6_validation_waveforms(
        repo_root=REPO_ROOT,
        command=shlex.join([sys.executable, *sys.argv]),
        device=args.device,
    )
    print(output)


if __name__ == "__main__":
    main()
