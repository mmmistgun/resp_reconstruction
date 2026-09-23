from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class StoppingState:
    """每次完整 validation 后更新；最短训练期内同样累计未改善次数。"""
    max_epochs: int = 80
    min_epochs: int = 30
    patience: int = 15
    enabled: bool = True
    epoch: int = 0
    best_epoch: int = 0
    best_value: float = math.inf
    reason: str | None = None

    def step(self, value):
        if self.reason is not None:
            raise ValueError("停止条件已满足，不能追加 epoch")
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError("early stopping 的 Local RR 必须有限")
        self.epoch += 1
        improved = value < self.best_value  # min_delta=0；并列不更新最早选点或耐心计数。
        if improved:
            self.best_value, self.best_epoch = value, self.epoch
        if self.epoch >= self.max_epochs:
            self.reason = "max_epochs"
        elif self.enabled and self.epoch >= self.min_epochs and self.epoch-self.best_epoch >= self.patience:
            self.reason = "patience_exhausted"
        return improved

    def receipt(self):
        if self.reason is None:
            raise ValueError("训练未满足预算或 early stopping 终止条件")
        return {"enabled": self.enabled, "monitor": "full_validation_local_rr_mae", "mode": "min",
                "min_epochs": self.min_epochs, "patience": self.patience, "min_delta": 0.0,
                "max_epochs": self.max_epochs, "stopped_epoch": self.epoch, "best_epoch": self.best_epoch,
                "best_value": self.best_value, "bad_epochs": self.epoch-self.best_epoch,
                "reason": self.reason, "stopped_early": self.epoch < self.max_epochs}


def stopping_state(cfg):
    # Smoke 的两 epoch 预算只验证管线；formal 才使用统一的提前停止合同。
    return StoppingState(max_epochs=int(cfg.training.epochs),
                         min_epochs=int(cfg.training.early_stopping_min_epochs),
                         patience=int(cfg.training.early_stopping_patience),
                         enabled=bool(cfg.training.early_stopping_enabled and cfg.protocol.run_role == "formal"))
