from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf

from resp_train.experiments.tho import ThoExperiment
from resp_train.models.registry import build_model


class ThoE5A2Experiment(ThoExperiment):
    """E5-A2 专用训练语义：time-only warm-start + 分组学习率。

    该类只服务 cross-attention warm-start 探针，避免把一次性科研变量扩散到通用 THO 训练入口。
    """

    task_name = "tho_e5_a2"

    def build_model(self):
        model = build_model(self.cfg)
        checkpoint_path = OmegaConf.select(self.cfg, "training.warm_start_checkpoint")
        if checkpoint_path:
            _load_prefix_warm_start(
                model,
                Path(str(checkpoint_path)),
                prefixes=_warm_start_prefixes(self.cfg),
            )
        return model

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        base_lr = float(self.cfg.training.learning_rate)
        time_lr = OmegaConf.select(self.cfg, "training.time_backbone_learning_rate")
        if time_lr is None or not hasattr(model, "time_backbone") or model.time_backbone is None:
            return torch.optim.Adam(model.parameters(), lr=base_lr)

        time_params = [p for p in model.time_backbone.parameters() if p.requires_grad]
        time_ids = {id(p) for p in time_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in time_ids]
        groups = []
        if time_params:
            groups.append({"params": time_params, "lr": float(time_lr)})
        if other_params:
            groups.append({"params": other_params, "lr": base_lr})
        if not groups:
            raise ValueError("没有可训练参数，无法构建 optimizer")
        return torch.optim.Adam(groups)


def _warm_start_prefixes(cfg) -> tuple[str, ...]:
    raw = OmegaConf.select(cfg, "training.warm_start_prefixes")
    if raw is None:
        return ("time_backbone.",)
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)


def _load_prefix_warm_start(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    prefixes: Iterable[str],
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"warm-start checkpoint 不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    normalized_prefixes = tuple(str(prefix) for prefix in prefixes)
    selected = {
        key: value
        for key, value in source_state.items()
        if key.startswith(normalized_prefixes)
        and key in target_state
        and tuple(value.shape) == tuple(target_state[key].shape)
    }
    if not selected:
        raise ValueError(
            "warm-start checkpoint 中没有匹配当前模型的参数；"
            f"checkpoint={checkpoint_path} prefixes={normalized_prefixes}"
        )
    model.load_state_dict(selected, strict=False)
