"""默认长度合成 BCG/CWT 的前后向、任务损失和优化器更新检查。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from omegaconf import OmegaConf

from resp_train.crd.training import build_crd_optimizer
from resp_train.losses.task import RespirationTaskLoss
from resp_train.models.bcg_cwt_patch_mamba import build_bcg_cwt_patch_mamba, environment_versions

ROOT = Path(__file__).resolve().parents[1]


class SyntheticMamba(nn.Module):
    """显式注入的可微接口替身；参数量和结果仅用于外围测试。"""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.projection(x)


def synthetic_config():
    cfg = OmegaConf.load(ROOT / "configs/bcg_cwt_patch_mamba_v1/model.yaml")
    task = OmegaConf.load(ROOT / "configs/crd_tf_v1/crd_tf101_m_smoke.yaml")
    cfg.window, cfg.loss, cfg.training = task.window, task.loss, task.training
    # 仅声明人工特征行身份，不将理想频率网格冒充缓存实际频率。
    cfg.model.frequencies_hz = torch.logspace(-1.5, .9, 97, dtype=torch.float64).tolist()
    return cfg


def synthetic_batch(batch_size, device):
    generator = torch.Generator(device=device).manual_seed(20261014)
    x = torch.randn(batch_size, 1, 18000, generator=generator, device=device, requires_grad=True)
    w = torch.rand(batch_size, 97, 360, generator=generator, device=device, requires_grad=True)
    t = torch.arange(18000, device=device) / 100
    # 呼吸目标包含缓慢幅值变化，使 sync 和 effort 均具有效目标资格。
    target = ((1 + .3 * torch.sin(2 * torch.pi * .015 * t)) * torch.sin(2 * torch.pi * .25 * t))
    return x, w, target[None, None].repeat(batch_size, 1, 1)


def smoke(*, batch_size=1, device="cpu", precision="float32", synthetic_mamba=False):
    cfg = synthetic_config()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return {"status": "skipped", "reason": "torch.cuda.is_available() 为 False"}
    if not synthetic_mamba and device.type != "cuda":
        return {"status": "skipped", "reason": "官方 Mamba2 fused 内核要求 CUDA；CPU 接口测试请显式使用 --synthetic-mamba"}
    if precision == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return {"status": "skipped", "reason": "当前 CUDA 设备不支持 bfloat16"}
    if not synthetic_mamba:
        try:
            from mamba_ssm import Mamba2  # noqa: F401
        except (ImportError, OSError) as exc:
            return {"status": "skipped", "reason": f"官方 mamba_ssm 导入失败: {exc}"}
    model = build_bcg_cwt_patch_mamba(cfg, mamba_factory=SyntheticMamba if synthetic_mamba else None).to(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, partition = build_crd_optimizer(model, cfg)
    x, w, target = synthetic_batch(batch_size, device)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    context = torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
    with context:
        prediction = model(x, tf={"w": w})
    loss, parts = loss_fn(prediction, target)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"参数梯度无效: {name}")
        branch = name.split(".")[0]
        gradients[branch] = gradients.get(branch, 0.) + float(parameter.grad.float().square().sum())
    if not all(value > 0 for value in gradients.values()):
        raise RuntimeError(f"分支梯度为零: {gradients}")
    for name, value in (("x", x), ("tf.w", w)):
        if value.grad is None or not torch.isfinite(value.grad).all() or value.grad.abs().sum() == 0:
            raise RuntimeError(f"输入梯度无效: {name}")
    optimizer.step()
    updated = {}
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"更新后参数非有限: {name}")
        branch = name.split(".")[0]
        updated[branch] = updated.get(branch, False) or not torch.equal(before[name], parameter)
    if not all(updated.values()):
        raise RuntimeError(f"分支参数未更新: {updated}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"status": "passed", "official_mamba": not synthetic_mamba,
            "environment": environment_versions(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "precision": precision, "batch_size": batch_size, "token_count": 180,
            "input_shape": list(x.shape), "cwt_shape": list(w.shape),
            "output_shape": list(prediction["waveform"].shape), "output_dtype": str(prediction["waveform"].dtype),
            "parameters": sum(p.numel() for p in model.parameters()),
            "patch_embedding_parameters": sum(p.numel() for p in model.patch_embedding.parameters()),
            "loss": float(loss.detach()), "loss_parts": {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in parts.items()},
            "gradient_squared_norms": gradients, "updated_branches": updated,
            "no_decay_parameter_tensors": len(partition.no_decay),
            "seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--synthetic-mamba", action="store_true")
    parser.add_argument("--output", type=Path, help="新建 JSON 记录，路径已存在则拒绝执行")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("--output 已存在，请使用新路径")
    torch.set_num_threads(1)
    try:
        report = smoke(batch_size=args.batch_size, device=args.device,
                       precision=args.precision, synthetic_mamba=args.synthetic_mamba)
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "environment": environment_versions()}
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        raise
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
