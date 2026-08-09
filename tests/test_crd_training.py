from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from resp_train.crd.blocks import CustomRMSNorm
from resp_train.crd.config import load_crd_config
from resp_train.crd.training import (
    crd_learning_rate,
    partition_weight_decay_parameters,
    structural_regularizer_ramp,
    train_crd_one_epoch,
)
from resp_train.losses.task import RespirationTaskLoss


def test_step_exact_learning_rate_hits_frozen_endpoints() -> None:
    kwargs = {
        "total_updates": 100,
        "max_learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "warmup_fraction": 0.05,
    }
    assert crd_learning_rate(0, **kwargs) == pytest.approx(6e-5)
    assert crd_learning_rate(4, **kwargs) == pytest.approx(3e-4)
    assert crd_learning_rate(5, **kwargs) == pytest.approx(3e-4)
    assert crd_learning_rate(99, **kwargs) == pytest.approx(3e-5)


def test_structural_regularizer_ramp_uses_optimizer_updates_per_epoch() -> None:
    assert structural_regularizer_ramp(400, updates_per_epoch=80) == 0.0
    assert structural_regularizer_ramp(800, updates_per_epoch=80) == pytest.approx(0.5)
    assert structural_regularizer_ramp(1200, updates_per_epoch=80) == 1.0


class _PartitionProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 2, 1, bias=True)
        self.norm = CustomRMSNorm(2)
        self.center_logits = nn.Parameter(torch.zeros(2))
        self.A_log = nn.Parameter(torch.zeros(2))
        self.P = nn.Parameter(torch.zeros(2))


class RMSNorm(nn.Module):
    """模拟 mamba_ssm 内部、不是本项目 CustomRMSNorm 子类的 RMSNorm。"""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))


def test_adamw_partition_has_explicit_no_decay_parameters() -> None:
    model = _PartitionProbe()
    model.mamba_norm = RMSNorm()
    partition = partition_weight_decay_parameters(model)
    assert set(partition.decay_names) == {"conv.weight"}
    assert set(partition.no_decay_names) == {
        "conv.bias",
        "norm.weight",
        "center_logits",
        "A_log",
        "P",
        "mamba_norm.weight",
    }


class _ThreeSampleDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        targets = (1.0, 0.0, 2.0)
        return {
            "x": torch.ones(1, 1),
            "target": torch.full((1, 1), targets[index]),
        }


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return x * self.weight


class _RegularizedScalarModel(_ScalarModel):
    def __init__(self) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.tensor(2.0))

    def regularization_terms(self):
        return {"loss_proto": self.P.square()}


class _EligibleProbeLoss(nn.Module):
    sync_weight = 1.0
    effort_weight = 0.25

    @torch.no_grad()
    def target_component_counts(self, target: torch.Tensor):
        flat = target[:, 0, 0]
        return {
            "loss_sync_count": (flat > 0).sum(),
            "loss_effort_count": (flat >= 2).sum(),
        }

    def differentiable_component_sums(self, prediction: torch.Tensor, target: torch.Tensor):
        pred = prediction[:, 0, 0]
        ref = target[:, 0, 0]
        sync = ref > 0
        effort = ref >= 2
        error = (pred - ref).square()
        return {
            "loss_sync_sum": error[sync].sum() if bool(sync.any()) else pred.sum() * 0.0,
            "loss_sync_count": sync.sum(),
            "loss_effort_sum": 2.0 * error[effort].sum() if bool(effort.any()) else pred.sum() * 0.0,
            "loss_effort_count": effort.sum(),
        }


def test_eligible_aware_accumulation_uses_whole_group_denominators() -> None:
    loader = DataLoader(_ThreeSampleDataset(), batch_size=2, shuffle=False)
    model = _ScalarModel()
    loss = _EligibleProbeLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    summary, next_update = train_crd_one_epoch(
        model,
        loader,
        loss,  # type: ignore[arg-type]
        optimizer,
        device="cpu",
        accumulation_steps=2,
        update_index=0,
        total_updates=1,
        max_learning_rate=0.1,
        min_learning_rate=0.01,
        warmup_fraction=0.05,
        grad_clip_norm=100.0,
        use_amp=True,
        show_progress=False,
    )

    # group gradient = sync(-3) + 0.25 * effort(-8) = -5；SGD(0.1) 后 weight=0.5。
    assert model.weight.item() == pytest.approx(0.5)
    assert next_update == 1
    assert summary["eligible_sync"] == 2
    assert summary["eligible_effort"] == 1


def test_prototype_regularizer_is_added_once_per_accumulation_group() -> None:
    loader = DataLoader(_ThreeSampleDataset(), batch_size=2, shuffle=False)
    model = _RegularizedScalarModel()
    loss = _EligibleProbeLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    summary, next_update = train_crd_one_epoch(
        model,
        loader,
        loss,  # type: ignore[arg-type]
        optimizer,
        device="cpu",
        accumulation_steps=2,
        update_index=10,
        total_updates=20,
        max_learning_rate=0.1,
        min_learning_rate=0.1,
        warmup_fraction=0.05,
        grad_clip_norm=100.0,
        use_amp=False,
        show_progress=False,
    )

    assert next_update == 11
    assert model.P.item() == pytest.approx(1.9998)
    assert summary["loss_proto"] == pytest.approx(4.0)
    assert summary["loss_proto_weighted"] == pytest.approx(0.002)
    assert summary["regularizer_ramp_first"] == pytest.approx(0.5)
    assert summary["regularizer_ramp_last"] == pytest.approx(0.5)
    assert summary["loss"] == pytest.approx(4.502)


def test_core_loss_target_only_counts_match_differentiable_sums() -> None:
    cfg = load_crd_config("configs/crd_v1/crd_101_b0_coarse.yaml")
    loss_fn = RespirationTaskLoss(cfg)
    time = torch.arange(18000, dtype=torch.float32) / 100.0
    dynamic = (1.0 + 0.3 * torch.sin(2.0 * torch.pi * 0.01 * time)) * torch.sin(
        2.0 * torch.pi * 0.2 * time
    )
    target = torch.stack((dynamic, torch.zeros_like(dynamic)), dim=0)[:, None, :]
    prediction = target.clone().requires_grad_(True)

    expected = loss_fn.target_component_counts(target)
    components = loss_fn.differentiable_component_sums(prediction, target)
    objective = components["loss_sync_sum"] + components["loss_effort_sum"]
    objective.backward()

    assert int(expected["loss_sync_count"]) == 1
    assert int(expected["loss_effort_count"]) == 1
    assert torch.equal(expected["loss_sync_count"], components["loss_sync_count"])
    assert torch.equal(expected["loss_effort_count"], components["loss_effort_count"])
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
