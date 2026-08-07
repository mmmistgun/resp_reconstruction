from __future__ import annotations

import torch

from resp_train.crd.spectral_ops import (
    DIRECT_BANDWIDTH_INITIAL,
    DIRECT_CENTER_INITIAL,
    GaussianAnalyticFilterbank,
    fft_hard_lowpass,
    fourier_interpolate,
)


def test_direct_filterbank_has_frozen_initial_parameters_and_gradients() -> None:
    bank = GaussianAnalyticFilterbank()
    assert torch.allclose(bank.centers(), torch.tensor(DIRECT_CENTER_INITIAL), atol=1e-7)
    assert torch.allclose(bank.bandwidths(), torch.tensor(DIRECT_BANDWIDTH_INITIAL), atol=1e-7)

    time = torch.arange(18000, dtype=torch.float32) / 100.0
    signal = torch.sin(2.0 * torch.pi * 0.18 * time)[None, None, :]
    output = bank(signal)
    output.square().mean().backward()

    assert output.shape == (1, 12, 18000)
    assert torch.isfinite(output).all()
    assert bank.center_logits.grad is not None
    assert bank.bandwidth_logits.grad is not None
    assert torch.isfinite(bank.center_logits.grad).all()
    assert torch.isfinite(bank.bandwidth_logits.grad).all()


def test_fourier_interpolation_is_periodic_and_preserves_source_samples() -> None:
    source_length = 180
    factor = 10
    time = torch.arange(source_length, dtype=torch.float32)
    source = torch.sin(2.0 * torch.pi * 7.0 * time / source_length)[None, None, :]

    interpolated = fourier_interpolate(source, target_length=source_length * factor)

    assert interpolated.dtype == torch.float32
    assert interpolated.shape == (1, 1, 1800)
    assert torch.allclose(interpolated[..., ::factor], source, atol=2e-5, rtol=2e-5)


def test_hard_lowpass_removes_above_cutoff_component() -> None:
    length = 1800
    time = torch.arange(length, dtype=torch.float32) / 10.0
    low = torch.sin(2.0 * torch.pi * 0.2 * time)
    high = 0.5 * torch.sin(2.0 * torch.pi * 1.0 * time)
    filtered = fft_hard_lowpass((low + high)[None, None, :], sample_rate=10.0, cutoff_hz=0.45)

    assert torch.mean((filtered[0, 0] - low).square()) < 1e-10
    assert filtered.dtype == torch.float32
