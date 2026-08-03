from __future__ import annotations

import numpy as np
import pytest
import torch

from resp_train.config import load_config
from resp_train.engine.train import _LossMeter
from resp_train.losses.task import RespirationTaskLoss
from resp_train.protocols.respiration import canonicalize_numpy, canonicalize_torch, lag_priority


def _config(overrides: list[str] | None = None):
    return load_config("configs/tho_research_v2.yaml", overrides=overrides)


def _waveform() -> torch.Tensor:
    time = torch.arange(18000, dtype=torch.float32) / 100.0
    amplitude = 1.0 + 0.3 * torch.sin(2.0 * torch.pi * 0.01 * time)
    return (amplitude * torch.sin(2.0 * torch.pi * (0.18 * time + 0.00005 * time.square())))[None, None, :]


def test_canonical_projection_is_finite_centered_and_unit_scale() -> None:
    signal = _waveform()
    band, canonical = canonicalize_torch(signal, fs=100, low_hz=0.05, high_hz=0.70)
    assert band.shape == signal.shape
    assert torch.isfinite(canonical).all()
    assert float(canonical.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(canonical.square().mean()) == pytest.approx(1.0, abs=1e-6)

    band_np, canonical_np = canonicalize_numpy(signal.numpy(), fs=100, low_hz=0.05, high_hz=0.70)
    np.testing.assert_allclose(band.detach().numpy(), band_np, atol=4e-4)
    np.testing.assert_allclose(canonical.detach().numpy(), canonical_np, atol=4e-4)


def test_lag_priority_matches_frozen_tie_rule() -> None:
    assert lag_priority(3) == [0, -1, 1, -2, 2, -3, 3]


def test_loss_identity_is_zero_and_gradient_is_finite() -> None:
    loss_fn = RespirationTaskLoss(_config())
    target = _waveform()
    prediction = target.clone().requires_grad_()
    loss, parts = loss_fn(prediction, target)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)
    for key in ("loss_sync", "loss_effort"):
        assert float(parts[key].detach()) == pytest.approx(0.0, abs=1e-6)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_inverse_signal_is_penalized_by_signed_sync() -> None:
    loss_fn = RespirationTaskLoss(_config())
    target = _waveform()
    prediction = (-target).clone().requires_grad_()
    loss, parts = loss_fn(prediction, target)
    loss.backward()
    assert float(parts["loss_sync"].detach()) > 1.9
    assert float(loss.detach()) > 1.9
    assert torch.isfinite(prediction.grad).all()


def test_final_loss_total_contains_only_sync_and_effort() -> None:
    loss_fn = RespirationTaskLoss(_config())
    target = _waveform()
    prediction = torch.roll(target, shifts=20, dims=-1) + 0.05 * torch.sin(
        2.0 * torch.pi * 0.6 * torch.arange(18000, dtype=torch.float32) / 100.0
    )
    total, parts = loss_fn(prediction, target)
    expected = loss_fn.sync_weight * parts["loss_sync"] + loss_fn.effort_weight * parts["loss_effort"]
    torch.testing.assert_close(total, expected)
    assert set(key for key in parts if not key.startswith("__")) == {"loss_sync", "loss_effort"}
    assert torch.isfinite(total)


@pytest.mark.parametrize("override", ["loss.rhythm_weight=0", "loss.pol_start_weight=0"])
def test_removed_loss_terms_cannot_be_reenabled_for_training(override: str) -> None:
    with pytest.raises(ValueError, match="训练配置不能再包含"):
        RespirationTaskLoss(_config([override]))


def test_loss_rejects_nonfinite_prediction() -> None:
    loss_fn = RespirationTaskLoss(_config())
    target = _waveform()
    prediction = target.clone()
    prediction[..., 0] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        loss_fn(prediction, target)


def test_task_output_and_loss_ignore_raw_offset_and_positive_scale() -> None:
    target = _waveform()
    transformed = 4.0 * target + 17.0
    _, canonical_target = canonicalize_torch(target, fs=100, low_hz=0.05, high_hz=0.70)
    _, canonical_transformed = canonicalize_torch(transformed, fs=100, low_hz=0.05, high_hz=0.70)
    torch.testing.assert_close(canonical_transformed, canonical_target, atol=2e-5, rtol=2e-5)

    loss_fn = RespirationTaskLoss(_config())
    loss, _ = loss_fn(transformed, target)
    assert float(loss) == pytest.approx(0.0, abs=2e-5)


def test_zero_prediction_has_finite_loss_and_gradient() -> None:
    loss_fn = RespirationTaskLoss(_config())
    target = _waveform()
    prediction = torch.zeros_like(target, requires_grad=True)
    loss, parts = loss_fn(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parts[name]) for name in ("loss_sync", "loss_effort"))
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_epoch_meter_uses_eligible_sum_and_count_instead_of_batch_mean() -> None:
    meter = _LossMeter()
    meter.update(
        torch.tensor(2.0),
        {
            "loss_sync": torch.tensor(2.0),
            "__sum_loss_sync": torch.tensor(4.0),
            "__count_loss_sync": torch.tensor(2.0),
        },
        batch_size=4,
    )
    meter.update(
        torch.tensor(1.0),
        {
            "loss_sync": torch.tensor(1.0),
            "__sum_loss_sync": torch.tensor(1.0),
            "__count_loss_sync": torch.tensor(1.0),
        },
        batch_size=1,
    )
    summary = meter.summary()
    assert summary["loss"] == pytest.approx(1.8)
    assert summary["loss_sync"] == pytest.approx(5.0 / 3.0)
