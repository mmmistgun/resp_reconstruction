"""合成输入定向检查；CUDA 模式由用户执行以验证官方 Mamba2。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from resp_train.models.patch_aligned_tf_mamba import PatchAlignedTFMamba, h_cwt_features


class SyntheticMamba(nn.Module):
    """CPU 检查显式替身；报告中标记，避免混同官方内核验证。"""

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, value):
        return self.projection(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("device 只支持 cpu/cuda")
    torch.set_num_threads(1)
    torch.manual_seed(20261004)
    t = np.arange(18000, dtype=np.float32) / 100
    signal = ((1 + .2 * np.sin(2 * np.pi * .25 * t)) * np.sin(2 * np.pi * 2 * t)).astype(np.float32)
    condition, frequencies = h_cwt_features(signal)
    factory = SyntheticMamba if device.type == "cpu" else None
    model = PatchAlignedTFMamba(frequencies, mamba_factory=factory).to(device).eval()
    x = torch.from_numpy(signal)[None, None].to(device)
    w = torch.from_numpy(condition)[None].to(device).requires_grad_()
    # 打开 FiLM 投影，才能检查条件编码器和交叉注意力的梯度。
    with torch.no_grad():
        model.film[-1].weight.normal_(std=.01)
    output = model(x, tf={"w": w})["waveform"]
    output.square().mean().backward()
    if output.shape != (1, 1, 18000) or not torch.isfinite(output).all():
        raise RuntimeError("输出检查失败")
    if w.grad is None or not torch.isfinite(w.grad).all() or w.grad.abs().sum() == 0:
        raise RuntimeError("条件梯度检查失败")
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"参数梯度检查失败: {name}")
    print(json.dumps({"status": "passed", "device": str(device),
        "mamba": "synthetic_cpu_fixture" if factory else "official_mamba2",
        "cwt": "native_morlet_actual_frequencies", "output_shape": list(output.shape),
        "frequency_count": len(frequencies), "finite_gradients": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
