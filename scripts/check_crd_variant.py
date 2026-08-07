from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.config import check_crd_dependencies, load_crd_config
from resp_train.crd.model import build_crd_model
from resp_train.losses.task import RespirationTaskLoss
from resp_train.utils.run import resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD 单个 synthetic microbatch 的模型/loss/backward 短验收")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch-size 必须为正")
    problems = check_crd_dependencies()
    if problems:
        raise SystemExit("; ".join(problems))

    cfg = load_crd_config(args.config)
    device = resolve_device(args.device)
    set_seed(20260808)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_crd_model(cfg).to(device).train()
    loss_fn = RespirationTaskLoss(cfg).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    time = torch.arange(18000, device=device, dtype=torch.float32) / 100.0
    envelope = 1.0 + 0.3 * torch.sin(2.0 * torch.pi * 0.01 * time)
    target = (envelope * torch.sin(2.0 * torch.pi * 0.2 * time))[None, None, :]
    target = target.repeat(args.batch_size, 1, 1)
    sensor = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    amp_enabled = device.type == "cuda"
    with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        output = model(sensor)
    with torch.amp.autocast(device.type, enabled=False):
        components = loss_fn.differentiable_component_sums(output, target.float())
        sync = components["loss_sync_sum"] / components["loss_sync_count"].clamp_min(1)
        effort = components["loss_effort_sum"] / components["loss_effort_count"].clamp_min(1)
        loss = loss_fn.sync_weight * sync + loss_fn.effort_weight * effort
    loss.backward()

    waveform = output["waveform"]
    parameter_gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    finite = (
        bool(torch.isfinite(waveform).all())
        and sensor.grad is not None
        and bool(torch.isfinite(sensor.grad).all())
        and bool(parameter_gradients)
        and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in parameter_gradients)
    )
    payload = {
        "status": "passed" if finite else "failed",
        "variant": str(cfg.model.variant),
        "device": str(device),
        "waveform_shape": list(waveform.shape),
        "loss": float(loss.detach().cpu()),
        "sync_eligible": int(components["loss_sync_count"].item()),
        "effort_eligible": int(components["loss_effort_count"].item()),
        "all_output_and_gradients_finite": finite,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
