import torch
import pytest
from omegaconf import OmegaConf

from resp_train.losses.weak import WeakSyncLoss


def _cfg():
    return OmegaConf.create(
        {
            "window": {"target_fs": 100},
            "loss": {
                "envelope_weight": 1.0,
                "spectrum_weight": 0.2,
                "smooth_weight": 0.01,
                "envelope_window_sec": 2.0,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 0.7,
            },
        }
    )


def test_weak_sync_loss_returns_components_and_scalar():
    cfg = _cfg()
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800, requires_grad=True)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)

    assert total.ndim == 0
    assert set(parts) == {"envelope", "spectrum", "smooth"}
    total.backward()
    assert pred.grad is not None


def test_weak_sync_loss_rejects_mismatched_or_multichannel_shapes():
    loss_fn = WeakSyncLoss(_cfg())

    with pytest.raises(ValueError, match="shape 必须一致"):
        loss_fn(torch.randn(2, 1, 1800), torch.randn(1, 1, 1800))

    with pytest.raises(ValueError, match="必须为 \\[B, 1, T\\]"):
        loss_fn(torch.randn(2, 2, 1800), torch.randn(2, 2, 1800))


def test_weak_sync_loss_rejects_empty_spectrum_band():
    cfg = _cfg()
    cfg.loss.spectrum_low_hz = 40.0
    cfg.loss.spectrum_high_hz = 45.0
    loss_fn = WeakSyncLoss(cfg)

    with pytest.raises(ValueError, match="频谱损失频带为空"):
        loss_fn(torch.randn(2, 1, 8), torch.randn(2, 1, 8))
