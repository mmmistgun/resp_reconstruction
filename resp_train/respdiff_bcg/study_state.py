"""诊断无扰动上下文与检查点的有限性检查。"""

from contextlib import contextmanager
import copy
import random

import numpy as np
import torch

from resp_train.respdiff.model import finite
from .inference_v2 import state_digest


def finite_tree(value, name="checkpoint"):
    if isinstance(value, torch.Tensor):
        finite(name, value)
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{name} 非有限")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{name}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError(f"{name} 非有限")


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None}


def restore_rng(state, device):
    if (state["cuda"] is None) != (device.type != "cuda"):
        raise ValueError("恢复 RNG 的 CPU/CUDA 执行类型不一致")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


@contextmanager
def preserved_evaluation(model):
    """诊断前后恢复 RNG、模式、诊断缓存；参数或 buffer 变化显式失败。"""
    device = next(model.parameters()).device
    rng = rng_state(device)
    modes = [(module, module.training) for module in model.modules()]
    previous = copy.deepcopy(getattr(model, "last_diagnostics", None))
    before = state_digest(model)
    gradients = [(parameter, None if parameter.grad is None else parameter.grad.detach().clone())
                 for parameter in model.parameters()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        try:
            changed = state_digest(model) != before
            gradient_changed = any((parameter.grad is None) != (grad is None)
                                   or (grad is not None and not torch.equal(parameter.grad, grad))
                                   for parameter, grad in gradients)
        finally:
            for parameter, grad in gradients:
                parameter.grad = grad
            for module, mode in modes:
                module.training = mode
            if previous is not None:
                model.last_diagnostics = previous
            restore_rng(rng, device)
        if changed or gradient_changed:
            raise RuntimeError("诊断改变了模型参数、buffer 或已有梯度")
