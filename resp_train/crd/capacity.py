from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


S2_BASE_PARAMETER_COUNT = 1_068_745
S2BR_COMBINATION_PARAMETER_COUNTS = {
    "crd_205_base_em_static": 1_109_561,
    "crd_206_base_am_static": 1_235_897,
}


@dataclass(frozen=True)
class CapacityMatch:
    target_variant: str
    target_parameter_count: int
    block_count: int
    hidden_channels: int
    control_parameter_count: int
    parameter_difference: int
    relative_parameter_difference: float
    estimated_macs: int


class CapacityResidualBlock(nn.Module):
    """S2B-R capacity control 的 zero-init 1x1 residual block。"""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        if hidden_channels < 32 or hidden_channels > 2048 or hidden_channels % 8 != 0:
            raise ValueError("capacity hidden_channels 必须是 [32,2048] 内的 8 倍数")
        self.norm = nn.GroupNorm(12, 96, eps=1e-5, affine=True)
        self.expand = nn.Conv1d(96, hidden_channels, kernel_size=1, bias=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.10)
        self.project = nn.Conv1d(hidden_channels, 96, kernel_size=1, bias=True)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.kaiming_normal_(self.expand.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.expand(self.norm(x)))
        return x + self.project(self.dropout(residual))


class CapacityResidualStack(nn.Module):
    def __init__(self, *, block_count: int, hidden_channels: int) -> None:
        super().__init__()
        block_count = int(block_count)
        if not (1 <= block_count <= 8):
            raise ValueError("capacity block_count 必须位于 [1,8]")
        self.blocks = nn.ModuleList(
            [CapacityResidualBlock(hidden_channels) for _ in range(block_count)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def capacity_block_parameter_count(hidden_channels: int) -> int:
    # GN affine=192；expand=96H；project=96H+96。
    return 192 * int(hidden_channels) + 288


def select_capacity_match(target_variant: str) -> CapacityMatch:
    if target_variant not in S2BR_COMBINATION_PARAMETER_COUNTS:
        raise ValueError(f"未知 S2B-R target variant={target_variant!r}")
    target = int(S2BR_COMBINATION_PARAMETER_COUNTS[target_variant])
    candidates: list[CapacityMatch] = []
    for block_count in range(1, 9):
        for hidden_channels in range(32, 2049, 8):
            control = S2_BASE_PARAMETER_COUNT + block_count * capacity_block_parameter_count(hidden_channels)
            difference = control - target
            candidates.append(
                CapacityMatch(
                    target_variant=target_variant,
                    target_parameter_count=target,
                    block_count=block_count,
                    hidden_channels=hidden_channels,
                    control_parameter_count=control,
                    parameter_difference=difference,
                    relative_parameter_difference=abs(difference) / float(target),
                    estimated_macs=block_count * 2 * 96 * hidden_channels * 1800,
                )
            )
    selected = min(
        candidates,
        key=lambda item: (
            item.relative_parameter_difference,
            item.estimated_macs,
            item.block_count,
            item.hidden_channels,
        ),
    )
    if selected.relative_parameter_difference > 0.02:
        raise RuntimeError(
            f"capacity control 与 {target_variant} 参数差超过 2%: "
            f"{selected.relative_parameter_difference:.6%}"
        )
    return selected


__all__ = [
    "CapacityMatch",
    "CapacityResidualBlock",
    "CapacityResidualStack",
    "S2BR_COMBINATION_PARAMETER_COUNTS",
    "S2_BASE_PARAMETER_COUNT",
    "capacity_block_parameter_count",
    "select_capacity_match",
]
