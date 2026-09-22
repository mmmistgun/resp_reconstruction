#!/usr/bin/env python3
"""ADV-v1 结构统计及用户执行的原生 GPU 合成验证入口。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from resp_train.aligned_dual_view import (
    AlignedDualViewV1, load_model_config, prepare_cwt_batch, representation_spec,
)
from resp_train.crd.config import check_crd_dependencies
from resp_train.losses.task import RespirationTaskLoss


def check(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("; ".join(problems))
    cfg = load_model_config(args.config)
    model = AlignedDualViewV1(cfg).cpu()
    report = {
        "schema": "aligned-dual-view-v1-engineering-v1", "action": args.action,
        "model": asdict(cfg), "architecture_id": cfg.architecture_id,
        "representation": representation_spec(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_groups": {
            name: sum(p.numel() for p in module.parameters())
            for name, module in model.named_children()
        },
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "implementation_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([
                *(ROOT / "resp_train/aligned_dual_view").glob("*.py"),
                ROOT / "resp_train/aligned_dual_view/w0_grid.json",
                ROOT / "resp_train/crd/blocks.py", ROOT / "resp_train/crd/spectral_ops.py",
                ROOT / "resp_train/crd/initialization.py", ROOT / "resp_train/losses/task.py",
                ROOT / "resp_train/protocols/respiration.py", Path(__file__).resolve(),
            ])
        },
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("torch", "numpy", "scipy", "ssqueezepy", "mamba-ssm", "causal-conv1d")},
        "native_forward_checked": False, "native_backward_checked": False,
        "synthetic_only": True, "real_data_accessed": False,
    }
    if args.action == "describe":
        return report
    if not args.device.startswith("cuda"):
        raise ValueError("gpu-smoke 要求显式 CUDA device；不替换原生 Mamba")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    # 固定 FIR 需要完整 float32 乘法；AMP 仍可作用于可学习卷积和 Mamba。
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    time = torch.arange(18000, dtype=torch.float32) / 100.0
    waveform = ((1 + 0.3 * torch.sin(2 * torch.pi * 0.2 * time))
                * torch.sin(2 * torch.pi * 4 * time))[None, None]
    target = ((1 + 0.3 * torch.sin(2 * torch.pi * 0.015 * time))
              * torch.sin(2 * torch.pi * 0.2 * time))[None, None]
    features = prepare_cwt_batch(waveform).to(args.device) if model.uses_cwt else None
    model = model.to(args.device).train()
    loss_cfg = OmegaConf.create({
        "window": {"target_fs": 100, "duration_samples": 18000},
        "loss": {"band_low_hz": 0.05, "band_high_hz": 0.70,
                 "scale_eps": 1e-8, "dynamic_eps": 1e-8, "corr_eps": 1e-8,
                 "envelope_eps": 1e-8, "max_lag_sec": 0.30,
                 "envelope_window_sec": 10, "envelope_step_sec": 5,
                 "sync_weight": 1, "effort_weight": 0.25},
    })
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        prediction = model(waveform.to(args.device), cwt=features)
        loss, _ = RespirationTaskLoss(loss_cfg)(prediction, target.to(args.device))
    if not torch.isfinite(loss):
        raise FloatingPointError("合成 loss 非有限")
    loss.backward()
    missing, nonfinite = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    if missing or nonfinite:
        raise RuntimeError(f"梯度检查失败: missing={missing}, nonfinite={nonfinite}")
    report.update({
        "native_forward_checked": True, "native_backward_checked": True,
        "precision": args.precision, "device": args.device, "synthetic_loss": float(loss.detach()),
        "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
        "output_shapes": {key: list(value.shape) for key, value in prediction.items()},
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("describe", "gpu-smoke"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/aligned_dual_view_v1/joint.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--output", type=Path, help="独占创建报告文件；保留失败记录，拒绝覆盖")
    args = parser.parse_args()
    # 开始计算前独占创建输出；发生错误时保留失败生命周期。
    handle = args.output.open("x", encoding="utf-8") if args.output else None
    try:
        report = {"status": "passed", **check(args)}
    except Exception as exc:
        report = {"status": "failed", "action": args.action,
                  "error": f"{type(exc).__name__}: {exc}"}
        if handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        raise
    else:
        if handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    main()
