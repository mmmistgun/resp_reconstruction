from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import _validate_crd_config, load_crd_config
from resp_train.crd.model import CRDCoarseModel
from resp_train.crd.tf_v1_model import (
    BRANCH_TYPES,
    P_BRANCH_TARGET,
    P_BRANCH_TOLERANCE,
    CRDTfV1Model,
    TF_CONTROL_COUNT,
    TF_VARIANT_REPRESENTATIONS,
    TemporalCapacityBranch,
    trainable_parameter_count,
)


def _features(name: str, batch_size: int = 1) -> dict[str, torch.Tensor]:
    if name == "m":
        return {
            "m_slow": torch.randn(batch_size, 36, 101),
            "m_fast": torch.randn(batch_size, 44, 349),
        }
    if name == "w":
        return {"w": torch.randn(batch_size, 97, 360)}
    if name == "l":
        return {"l_spectrum": torch.randn(batch_size, 9001, dtype=torch.complex64)}
    if name == "s":
        return {"s": torch.randn(batch_size, 12, 360)}
    raise AssertionError(name)


@pytest.mark.parametrize("name", ("m", "w", "l", "s"))
def test_representation_branch_parameter_budget_shape_and_zero_init(name: str) -> None:
    branch = BRANCH_TYPES[name]().eval()
    count = trainable_parameter_count(branch)

    gamma, beta = branch(_features(name))

    assert abs(count - P_BRANCH_TARGET) / P_BRANCH_TARGET <= P_BRANCH_TOLERANCE
    assert gamma.shape == beta.shape == (1, 96, 1800)
    torch.testing.assert_close(gamma, torch.zeros_like(gamma), rtol=0.0, atol=0.0)
    torch.testing.assert_close(beta, torch.zeros_like(beta), rtol=0.0, atol=0.0)


def test_temporal_control_is_active_and_parameter_matched() -> None:
    branch = TemporalCapacityBranch()
    with torch.no_grad():
        branch.final_projection.weight.fill_(1e-4)
    latent = torch.randn(1, 96, 1800, requires_grad=True)

    gamma, beta = branch(latent)
    (gamma.square().mean() + beta.square().mean()).backward()

    assert abs(trainable_parameter_count(branch) - P_BRANCH_TARGET) / P_BRANCH_TARGET <= P_BRANCH_TOLERANCE
    assert branch.input_projection.weight.grad is not None
    assert torch.isfinite(branch.input_projection.weight.grad).all()
    assert torch.count_nonzero(branch.input_projection.weight.grad) > 0


def test_all_fifteen_variants_have_frozen_representation_or_control_contract() -> None:
    assert len(TF_VARIANT_REPRESENTATIONS) == 15
    assert set(TF_CONTROL_COUNT) == {"crd_tf_ctrl1", "crd_tf_ctrl2", "crd_tf_ctrl3"}
    assert TF_CONTROL_COUNT["crd_tf_ctrl3"] == 3
    assert TF_VARIANT_REPRESENTATIONS["crd_tf301_mls"] == ("m", "l", "s")
    assert TF_VARIANT_REPRESENTATIONS["crd_tf302_wls"] == ("w", "l", "s")


def test_same_seed_tf_base_state_is_tensor_identical_to_c201() -> None:
    seed = 20260811
    tf_model = CRDTfV1Model("crd_tf_ctrl1", seed)
    c201 = CRDCoarseModel("crd_c201_decoder_10hz_cap", seed)

    tf_state = tf_model.base.state_dict()
    c201_state = c201.state_dict()
    assert tf_state.keys() == c201_state.keys()
    for name in tf_state:
        torch.testing.assert_close(tf_state[name], c201_state[name], rtol=0.0, atol=0.0)


def test_zero_init_tf_film_preserves_exact_c201_waveform() -> None:
    seed = 20260811
    tf_model = CRDTfV1Model("crd_tf104_s", seed).eval()
    c201 = CRDCoarseModel("crd_c201_decoder_10hz_cap", seed).eval()
    # 原生 Mamba 的 CUDA acceptance 属于 P3；这里隔离验证 P2 FiLM 插入点与 decoder identity。
    tf_model.base.local_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in tf_model.base.local_blocks])
    c201.local_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in c201.local_blocks])
    signal = torch.randn(1, 1, 18000)
    features = _features("s")

    with torch.no_grad():
        candidate = tf_model(signal, tf=features)
        reference = c201(signal)

    assert candidate.keys() == reference.keys()
    for name in candidate:
        torch.testing.assert_close(candidate[name], reference[name], rtol=0.0, atol=0.0)


def test_strict_config_accepts_every_variant_and_rejects_gate_drift() -> None:
    base = load_crd_config("configs/crd_tf_v1/crd_tf101_m_smoke.yaml")
    for variant, representations in TF_VARIANT_REPRESENTATIONS.items():
        cfg = deepcopy(base)
        cfg.model.variant = variant
        cfg.model.tf_representations = list(representations)
        _validate_crd_config(cfg)

    bad = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    bad.protocol.execution_gate = "p3_cuda_acceptance"
    with pytest.raises(ValueError, match="execution_gate"):
        _validate_crd_config(bad)

    bad = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    bad.model.tf_representations = ["w"]
    with pytest.raises(ValueError, match="tf_representations"):
        _validate_crd_config(bad)
