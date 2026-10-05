"""合成输入定向检查；CUDA 模式由用户执行以验证官方 Mamba2。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from resp_train.models.patch_aligned_tf_mamba import PatchAlignedTFMamba, PatchTFConfig, h_cwt_features


class SyntheticMamba(nn.Module):
    """CPU 检查显式替身；报告中标记，避免混同官方内核验证。"""

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, value):
        return self.projection(value)


def check(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 BF16")
    torch.set_num_threads(1)
    torch.manual_seed(20261004)
    t = np.arange(18000, dtype=np.float32) / 100
    signal = ((1 + .2 * np.sin(2 * np.pi * .25 * t)) * np.sin(2 * np.pi * 2 * t)).astype(np.float32)
    condition, frequencies = h_cwt_features(signal)
    factory = SyntheticMamba if device.type == "cpu" else None
    cfg = PatchTFConfig(patch_samples=args.patch_samples, patch_chunk_size=args.patch_chunk_size)
    model = PatchAlignedTFMamba(frequencies, config=cfg, mamba_factory=factory).to(device).train()
    x = torch.from_numpy(signal)[None, None].repeat(args.batch_size, 1, 1).to(device)
    w = torch.from_numpy(condition)[None].repeat(args.batch_size, 1, 1).to(device).requires_grad_()
    target = torch.from_numpy(np.sin(2 * np.pi * .25 * t).astype(np.float32))[None, None].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        w.grad = None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.dtype == "bfloat16"):
            output = model(x, tf={"w": w})["waveform"]
            loss = (output.float() - target).square().mean()
        if output.shape != (args.batch_size, 1, cfg.window_samples) or not torch.isfinite(loss):
            raise RuntimeError("输出/loss 检查失败")
        loss.backward()
        if w.grad is None or not torch.isfinite(w.grad).all():
            raise RuntimeError("条件梯度非有限或缺失")
        # 第一轮由零初始化 FiLM 投影开始学习，第二轮起检查条件上游梯度。
        if step > 0 and w.grad.abs().sum() == 0:
            raise RuntimeError("FiLM 更新后条件梯度仍为零")
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"参数梯度检查失败: {name}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("优化后参数非有限")
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"status": "passed", "device": str(device),
        "mamba": "synthetic_cpu_fixture" if factory else "official_mamba2",
        "cwt": "native_morlet_actual_frequencies", "output_shape": list(output.shape),
        "frequency_count": len(frequencies), "finite_gradients": True,
        "batch_size": args.batch_size, "dtype": args.dtype, "steps": args.steps,
        "model_config": asdict(cfg), "initialization_seed": 20261004,
        "parameters": sum(p.numel() for p in model.parameters()), "losses": losses,
        "elapsed_seconds": time.perf_counter() - start, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--patch-samples", type=int, default=200)
    parser.add_argument("--patch-chunk-size", type=int, default=16)
    parser.add_argument("--report", type=Path, help="可选 JSON 报告；路径必须尚不存在")
    args = parser.parse_args()
    if torch.device(args.device).type not in {"cpu", "cuda"}:
        parser.error("device 只支持 cpu/cuda")
    if args.batch_size < 1 or args.steps < 2:
        parser.error("batch-size 必须为正，steps 至少为 2")
    # 提前独占创建报告，失败也保留状态；同名报告拒绝覆盖。
    with (args.report.open("x", encoding="utf-8") if args.report else nullcontext(None)) as handle:
        try:
            report = check(args)
        except Exception as exc:
            if handle:
                json.dump({"status": "failed", "error": str(exc), "device": args.device,
                           "batch_size": args.batch_size, "dtype": args.dtype}, handle, ensure_ascii=False, indent=2)
            raise
        if handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
