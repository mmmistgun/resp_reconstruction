from __future__ import annotations

import time

import numpy as np
import torch

from resp_train.aligned_dual_view.features import prepare_cwt_batch
from resp_train.losses.task import RespirationTaskLoss
from resp_train.crd.training import build_crd_optimizer
from .artifacts import artifact_directory, require_finite_tree, write_json
from .config import ALL_ARMS, PROTOCOL, model_config
from .model import FusionModel, module_report


def describe(cfg):
    reports = []
    for arm in ALL_ARMS:
        cfg.model.arm = arm
        reports.append(module_report(FusionModel(model_config(cfg))))
    return reports


def gpu_synthetic(cfg, output, *, confirm_gpu=False):
    """用户执行：原生 Mamba、实际合成 CWT、bf16、任务 loss 和 AdamW 更新。"""
    if not confirm_gpu or not str(cfg.training.device).startswith("cuda"):
        raise ValueError("GPU 合成验收需要 --confirm-gpu 和 CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    with artifact_directory(output, kind="gpu_synthetic", cfg=cfg) as (root, _):
        device = torch.device(cfg.training.device)
        torch.cuda.set_device(device)
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.manual_seed(int(cfg.training.seed))
        t = torch.arange(18000, dtype=torch.float32) / 100
        x = ((1 + 0.3*torch.sin(2*torch.pi*0.2*t))*torch.sin(2*torch.pi*4*t))[None, None]
        target = ((1+0.3*torch.sin(2*torch.pi*0.015*t))*torch.sin(2*torch.pi*0.2*t+0.1))[None, None]
        cwt = prepare_cwt_batch(x).values
        batch = int(cfg.training.batch_size)
        x = x.repeat(batch, 1, 1).to(device)
        target = target.repeat(batch, 1, 1).to(device)
        features = cwt.repeat(batch, 1, 1).to(device)
        model = FusionModel(model_config(cfg)).to(device).train()
        optimizer, _ = build_crd_optimizer(model, cfg)
        loss_fn = RespirationTaskLoss(cfg).to(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed, losses, gradients = [], [], []
        # 两步预热及解冻，后三步计时；不是吞吐 benchmark。
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_values = model(x, tf={"adv_cwt": features})
                loss, _ = loss_fn(output_values, target)
            if output_values["waveform"].shape != (batch, 1, 18000) or output_values["waveform_10hz"].shape != (batch, 1, 1800):
                raise ValueError("GPU 合成输出 shape 错误")
            require_finite_tree(loss, "synthetic loss")
            loss.backward()
            norms = {}
            for name, module in model.named_children():
                parameters = list(module.parameters())
                for param in parameters:
                    if param.grad is None:
                        raise RuntimeError(f"缺失梯度: {name}")
                    require_finite_tree(param.grad, f"gradient:{name}")
                if parameters:
                    norms[name] = sum(float(p.grad.detach().float().square().sum()) for p in parameters)**0.5
            if step >= 1:
                for name in ("waveform_encoder", "scale_projection", "wave_adapter", "cwt_adapter", "fusion"):
                    if norms[name] <= 0:
                        raise RuntimeError(f"更新后模块梯度仍为零: {name}")
                if model.method == "attention":
                    for name in ("query", "key", "value"):
                        if not getattr(model.fusion, name).weight.grad.abs().sum() > 0:
                            raise RuntimeError(f"更新后 attention {name} 梯度为零")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            require_finite_tree(model.state_dict())
            torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter()-start)
            losses.append(float(loss.detach()))
            gradients.append(norms)
        report = {**module_report(model), "batch_size": batch, "precision": "bfloat16",
                  "device": str(device), "device_name": torch.cuda.get_device_name(device),
                  "torch_version": torch.__version__, "cudnn_conv_fp32_precision": "ieee",
                  "step_seconds": elapsed, "measured_step_median_seconds": float(np.median(elapsed[2:])),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "losses": losses, "gradient_norms": gradients,
                  "output_shapes": {key: list(value.shape) for key, value in output_values.items()},
                  "real_data_accessed": False, "native_forward_backward": True,
                  "timing_scope": "full_model_task_loss_backward_optimizer; excludes offline CWT and data IO"}
        write_json(root / "report.json", report)
        from .artifacts import sha256_file
        write_json(root / "manifest.json", {"protocol": PROTOCOL, "kind": "gpu_synthetic",
                   "arm": cfg.model.arm, "report_sha256": sha256_file(root / "report.json"),
                   "real_data_accessed": False})
    return root
