from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.blocks import BidirectionalMamba2Block
from resp_train.crd.config import check_crd_dependencies, crd_dependency_versions
from resp_train.utils.run import resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD 官方 Mamba2 fast-path 短验收（不训练数据集）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--length", type=int, default=1800)
    args = parser.parse_args()

    problems = check_crd_dependencies()
    if problems:
        raise SystemExit("; ".join(problems))
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("Mamba fast-path 验收必须使用 CUDA")
    if args.batch_size <= 0 or args.length <= 0:
        raise SystemExit("batch-size/length 必须为正")

    set_seed(20260808)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    block = BidirectionalMamba2Block(96).to(device).train()
    signal = torch.randn(args.batch_size, args.length, 96, device=device, requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = block(signal)
        objective = output.float().square().mean()
    objective.backward()
    gradients = [parameter.grad for parameter in block.parameters() if parameter.requires_grad]
    if not torch.isfinite(output).all() or not torch.isfinite(signal.grad).all():
        raise FloatingPointError("Mamba2 output/input gradient 包含 NaN/Inf")
    if not gradients or any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("Mamba2 parameter gradient 缺失或包含 NaN/Inf")

    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                "dependency_versions": crd_dependency_versions(),
                "use_mem_eff_path": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
