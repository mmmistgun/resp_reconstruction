from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from resp_train.paper_evidence.center_context_config import load_center_context_config
from resp_train.paper_evidence.center_context_experiment import center_experiment_id
from resp_train.paper_evidence.center_context_model import CenterContextModel, shared_state_identity


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper_evidence_v1/center_context_v1.yaml"


class _IdentityMamba(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def test_center_config_is_strict_and_implementation_role_cannot_become_test() -> None:
    cfg = load_center_context_config(CONFIG)
    assert cfg.protocol.name == "paper-center-context-v1-20260901"
    assert cfg.data.access_splits == ["train", "val"]
    with pytest.raises(ValueError, match="test windows"):
        load_center_context_config(CONFIG, overrides=["data.max_test_windows=1"])
    with pytest.raises(ValueError, match="W-reduced-center60"):
        load_center_context_config(CONFIG, overrides=["model.variant=w_reduced_center60"])


@pytest.mark.parametrize("length,context", [(6000, 120), (9000, 180), (18000, 360)])
def test_c201_w_reduced_forward_shapes_and_length_independent_parameter_count(length: int, context: int) -> None:
    seed = 20260811
    c201 = CenterContextModel("c201_center60", seed, mamba_factory=_IdentityMamba).eval()
    w_reduced = CenterContextModel("w_reduced_center60", seed, mamba_factory=_IdentityMamba).eval()
    x = torch.randn(1, 1, length)
    with torch.no_grad():
        c201_output = c201(x)
        w_reduced_output = w_reduced(x, tf={"w": torch.randn(1, 49, context)})
    assert c201_output["waveform"].shape == (1, 1, 6000)
    assert c201_output["waveform_10hz"].shape == (1, 1, 600)
    assert w_reduced_output["waveform"].shape == (1, 1, 6000)
    assert w_reduced_output["waveform_10hz"].shape == (1, 1, 600)
    assert c201.trainable_parameter_count() > 0
    assert w_reduced.trainable_parameter_count() > c201.trainable_parameter_count()


def test_same_seed_shared_state_is_tensor_identical_and_arm_ids_are_frozen() -> None:
    c201 = CenterContextModel("c201_center60", 20260811, mamba_factory=_IdentityMamba)
    w_reduced = CenterContextModel("w_reduced_center60", 20260811, mamba_factory=_IdentityMamba)
    audit = shared_state_identity(c201, w_reduced)
    assert audit["shared_tensors_identical"] is True
    assert center_experiment_id("c201_center60", 6000) == "CCV1_C201_60"
    assert center_experiment_id("w_reduced_center60", 18000) == "CCV1_WR_180"


def test_w_reduced_requires_only_length_specific_w_and_c201_rejects_tf() -> None:
    c201 = CenterContextModel("c201_center60", 1, mamba_factory=_IdentityMamba).eval()
    w_reduced = CenterContextModel("w_reduced_center60", 1, mamba_factory=_IdentityMamba).eval()
    x = torch.zeros(1, 1, 6000)
    with pytest.raises(ValueError, match="不得读取"):
        c201(x, tf={"w": torch.zeros(1, 49, 120)})
    with pytest.raises(ValueError, match="缺少"):
        w_reduced(x)
    with pytest.raises(ValueError, match="期望"):
        w_reduced(x, tf={"w": torch.zeros(1, 49, 360)})
