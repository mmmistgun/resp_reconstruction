from __future__ import annotations

import pytest
import torch

import resp_train.crd.representations as representation_module
from resp_train.crd.representations import (
    AnalyticAMRepresentation,
    LegacyEnergyRepresentation,
    MorphologyRepresentation,
)
from resp_train.crd.spectral_ops import AM_CENTER_INITIAL, AM_RHO_INITIAL


def test_legacy_energy_uses_exact_stft_contract_and_zero_injection() -> None:
    branch = LegacyEnergyRepresentation(20260811).eval()
    signal = torch.randn(1, 1, 18000)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        energy = branch.band_energy(signal)
        output = branch(signal)

    assert energy.shape == (1, 5, 73)
    assert energy.dtype == torch.float32
    assert torch.isfinite(energy).all()
    assert output.shape == (1, 96, 1800)
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(branch.projection.weight) == 0
    assert torch.count_nonzero(branch.projection.bias) == 0


def test_analytic_am_has_frozen_bounded_initialization_and_float32_spectral_path() -> None:
    branch = AnalyticAMRepresentation(20260811).eval()
    signal = torch.randn(1, 1, 18000)
    assert branch.filterbank.centers().detach().tolist() == pytest.approx(AM_CENTER_INITIAL)
    assert branch.filterbank.rhos().detach().tolist() == pytest.approx([AM_RHO_INITIAL] * 8)
    assert torch.all(branch.filterbank.centers() >= branch.filterbank.center_bounds[:, 0])
    assert torch.all(branch.filterbank.centers() <= branch.filterbank.center_bounds[:, 1])

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        analytic = branch.filterbank(signal)
        modulation = branch.amplitude_modulation(signal)
        output = branch(signal)

    assert analytic.shape == (1, 8, 18000)
    assert analytic.dtype == torch.complex64
    assert modulation.shape == (1, 8, 1800)
    assert modulation.dtype == torch.float32
    assert torch.isfinite(modulation).all()
    assert torch.count_nonzero(output) == 0


def test_morphology_window_normalization_chunking_and_prototypes_are_frozen(monkeypatch) -> None:
    branch = MorphologyRepresentation(20260811).eval()
    signal = torch.randn(1, 1, 18000)
    windows = branch.normalized_windows(signal)

    assert windows.shape == (1, 1, 1800, 151)
    assert windows.dtype == torch.float32
    assert windows.mean(dim=-1).abs().max().item() < 1e-5
    assert branch.token_chunk_size == 128
    assert branch.prototype_count == 16
    assert branch.prototype_temperature == pytest.approx(0.10)
    assert branch.prototype_loss().item() < 1e-12

    zero_output = branch(signal)
    assert zero_output.shape == (1, 96, 1800)
    assert torch.count_nonzero(zero_output) == 0

    # 用非零 projection 验证 checkpoint 训练路径与普通前向数学等价，且输入本身无需梯度。
    with torch.no_grad():
        branch.projection.weight.normal_(mean=0.0, std=0.01)
    reference = branch(signal).detach()
    checkpoint_calls = 0
    real_checkpoint = representation_module.checkpoint

    def counted_checkpoint(function, *args, **kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(representation_module, "checkpoint", counted_checkpoint)
    branch.train()
    output = branch(signal)
    assert checkpoint_calls == 15
    assert torch.allclose(output, reference, atol=1e-6, rtol=1e-5)
    objective = output.square().mean() + 1e-3 * branch.prototype_loss()
    objective.backward()
    gradients = [parameter.grad for parameter in branch.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
