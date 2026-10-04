"""固定阈值的增益分布与裁剪前梯度记录。"""
import json
import torch
from .spec import QUANTILES, GAIN_THRESHOLDS


def gain_statistics(value):
    value = value.detach().float().reshape(-1)
    if not len(value) or not torch.isfinite(value).all():
        raise FloatingPointError("增益统计输入为空或非有限")
    quantiles = torch.quantile(value, torch.tensor(QUANTILES, device=value.device)).cpu().tolist()
    return {"n": len(value), "min": quantiles[0], "max": quantiles[-1],
            "quantiles": dict(zip(map(str, QUANTILES), quantiles)),
            "negative_fraction": float((value < 0).float().mean()),
            "near_zero_fraction": float((value.abs() < GAIN_THRESHOLDS["near_zero_abs_lt"]).float().mean()),
            "extreme_fraction": float((value.abs() > GAIN_THRESHOLDS["extreme_abs_gt"]).float().mean())}


class TrainingMonitor:
    def __init__(self, model, directory, capture_gain):
        self.update = 0
        self.stream = (directory / "diagnostics.jsonl").open("x", encoding="utf-8")
        self.model = model
        self.handle = torch.autograd.graph.register_multi_grad_hook(tuple(model.parameters()), self.gradients)
        if capture_gain:
            model.gain_observer = self.gains

    def write(self, payload):
        self.stream.write(json.dumps(payload, allow_nan=False) + "\n")
        self.stream.flush()

    def gains(self, value, training):
        # 训练记录每批分布；完整 validation/test 的逐窗口统计由评价阶段另存。
        if training:
            self.write({"kind": "train_gain", "update": self.update, **gain_statistics(value)})

    def gradients(self, gradients):
        norms = [g.detach().float().norm() for g in gradients if g is not None]
        if not norms:
            raise RuntimeError("反向传播未产生梯度")
        norm = torch.stack(norms).norm()
        if not torch.isfinite(norm):
            self.write({"kind": "gradient_failure", "update": self.update, "finite": False})
            raise FloatingPointError("裁剪前梯度非有限")
        self.write({"kind": "gradient", "update": self.update, "pre_clip_norm": float(norm)})
        self.update += 1

    def close(self):
        self.handle.remove()
        self.model.gain_observer = None
        self.stream.close()
