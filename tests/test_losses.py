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
                "high_freq_weight": 0.0,
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
    assert set(parts) == {"envelope", "spectrum", "smooth", "high_freq", "relative_envelope", "band_waveform"}
    total.backward()
    assert pred.grad is not None


def test_optional_loss_weights_zero_keep_total_loss_unchanged():
    cfg = _cfg()
    cfg.loss.relative_envelope_weight = 0.0
    cfg.loss.band_waveform_weight = 0.0
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)
    expected = (
        cfg.loss.envelope_weight * parts["envelope"]
        + cfg.loss.spectrum_weight * parts["spectrum"]
        + cfg.loss.smooth_weight * parts["smooth"]
        + cfg.loss.high_freq_weight * parts["high_freq"]
        + cfg.loss.relative_envelope_weight * parts["relative_envelope"]
        + cfg.loss.band_waveform_weight * parts["band_waveform"]
    )

    assert torch.allclose(total, expected)


def test_relative_envelope_loss_penalizes_missing_relative_boost():
    cfg = _cfg()
    cfg.loss.relative_envelope_weight = 0.01
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    carrier = torch.sin(2 * torch.pi * 0.25 * time)
    boost = torch.ones_like(time)
    boost[(time >= 20) & (time <= 40)] = 1.8
    target = (boost * carrier).reshape(1, 1, -1)
    good = (3.0 * target).clone()
    bad = carrier.reshape(1, 1, -1)

    _, good_parts = loss_fn(good, target)
    _, bad_parts = loss_fn(bad, target)

    assert bad_parts["relative_envelope"] > good_parts["relative_envelope"] + 0.05


def test_band_waveform_loss_penalizes_phase_shift_in_respiratory_band():
    cfg = _cfg()
    cfg.loss.band_waveform_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    same_phase = (3.0 * target).clone()
    shifted = torch.sin(2 * torch.pi * 0.25 * time + torch.pi / 2).reshape(1, 1, -1)

    _, same_parts = loss_fn(same_phase, target)
    _, shifted_parts = loss_fn(shifted, target)

    assert same_parts["band_waveform"] < 1e-4
    assert shifted_parts["band_waveform"] > same_parts["band_waveform"] + 0.5


def test_band_waveform_loss_is_differentiable_when_enabled():
    cfg = _cfg()
    cfg.loss.band_waveform_weight = 0.5
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800, requires_grad=True)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)
    total.backward()

    assert parts["band_waveform"] >= 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_weak_sync_loss_penalizes_prediction_high_frequency_energy():
    cfg = _cfg()
    cfg.loss.high_freq_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 10, 1 / fs)
    low = torch.sin(2 * torch.pi * 0.2 * time).reshape(1, 1, -1)
    high = 0.5 * torch.sin(2 * torch.pi * 5.0 * time).reshape(1, 1, -1)

    _, low_parts = loss_fn(low, low)
    _, mixed_parts = loss_fn(low + high, low)

    assert "high_freq" in mixed_parts
    assert mixed_parts["high_freq"] > low_parts["high_freq"] * 100


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
