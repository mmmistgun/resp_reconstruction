from __future__ import annotations

import torch

from resp_train.crd.capacity import (
    CapacityResidualBlock,
    capacity_block_parameter_count,
    select_capacity_match,
)


def test_capacity_block_parameter_formula_and_zero_identity() -> None:
    block = CapacityResidualBlock(104).eval()
    assert sum(parameter.numel() for parameter in block.parameters()) == capacity_block_parameter_count(104)
    signal = torch.randn(2, 96, 47)
    assert torch.equal(block(signal), signal)
    assert torch.count_nonzero(block.project.weight) == 0
    assert torch.count_nonzero(block.project.bias) == 0


def test_capacity_match_uses_frozen_lexicographic_rule() -> None:
    em = select_capacity_match("crd_205_base_em_static")
    am = select_capacity_match("crd_206_base_am_static")

    assert (em.block_count, em.hidden_channels, em.parameter_difference) == (2, 104, -304)
    assert (am.block_count, am.hidden_channels, am.parameter_difference) == (4, 216, -112)
    assert em.relative_parameter_difference < 0.001
    assert am.relative_parameter_difference < 0.001
