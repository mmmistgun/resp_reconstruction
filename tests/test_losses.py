import pytest
import torch
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
                "relative_envelope_weight": 0.0,
                "phase_alignment_weight": 0.0,
                "band_waveform_weight": 0.0,
                "curvature_weight": 0.0,
                "phase_alignment_zero_weight": 0.5,
                "phase_alignment_lag_weight": 0.5,
                "phase_alignment_lag_penalty_weight": 0.1,
                "phase_alignment_max_sec": 0.5,
                "phase_alignment_step_sec": 0.1,
                "phase_alignment_temperature": 0.05,
                "envelope_window_sec": 2.0,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 0.7,
            },
        }
    )


def test_weak_sync_loss_returns_current_components_and_scalar():
    cfg = _cfg()
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800, requires_grad=True)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)

    assert total.ndim == 0
    assert set(parts) == {
        "envelope",
        "spectrum",
        "smooth",
        "high_freq",
        "relative_envelope",
        "phase_alignment",
        "band_waveform",
        "curvature",
    }
    total.backward()
    assert pred.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 复现 AMP + cuFFT 行为")
def test_weak_sync_loss_supports_amp_with_non_power_of_two_window():
    cfg = _cfg()
    cfg.loss.high_freq_weight = 0.2
    cfg.loss.phase_alignment_weight = 0.005
    loss_fn = WeakSyncLoss(cfg).cuda()
    pred = torch.randn(2, 1, 18000, device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.randn(2, 1, 18000, device="cuda")

    with torch.amp.autocast("cuda", enabled=True):
        total, parts = loss_fn(pred, target)
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(pred.grad).all()
    assert parts["spectrum"].dtype == torch.float32
    assert parts["phase_alignment"].dtype == torch.float32


def test_optional_loss_weights_zero_keep_total_loss_unchanged():
    cfg = _cfg()
    cfg.loss.relative_envelope_weight = 0.0
    cfg.loss.phase_alignment_weight = 0.0
    cfg.loss.band_waveform_weight = 0.0
    cfg.loss.curvature_weight = 0.0
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
        + cfg.loss.phase_alignment_weight * parts["phase_alignment"]
        + cfg.loss.band_waveform_weight * parts["band_waveform"]
        + cfg.loss.curvature_weight * parts["curvature"]
    )

    assert torch.allclose(total, expected)


def test_relative_envelope_loss_penalizes_missing_relative_boost():
    cfg = _cfg()
    cfg.loss.relative_envelope_weight = 0.03
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


def test_phase_alignment_loss_penalizes_inverted_low_frequency_signal():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.high_freq_weight = 0.0
    cfg.loss.phase_alignment_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    same = target.clone()
    inverted = -target

    _, same_parts = loss_fn(same, target)
    _, inverted_parts = loss_fn(inverted, target)

    assert same_parts["phase_alignment"] < 0.05
    assert inverted_parts["phase_alignment"] > same_parts["phase_alignment"] + 1.0


def test_phase_alignment_loss_penalizes_large_lag_even_when_correlation_can_recover():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.high_freq_weight = 0.0
    cfg.loss.phase_alignment_weight = 1.0
    cfg.loss.phase_alignment_zero_weight = 0.5
    cfg.loss.phase_alignment_lag_weight = 0.5
    cfg.loss.phase_alignment_lag_penalty_weight = 0.5
    cfg.loss.phase_alignment_max_sec = 0.5
    cfg.loss.phase_alignment_step_sec = 0.1
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    shifted = torch.roll(target, shifts=int(round(0.5 * fs)), dims=-1)

    _, same_parts = loss_fn(target.clone(), target)
    _, shifted_parts = loss_fn(shifted, target)

    assert shifted_parts["phase_alignment"] > same_parts["phase_alignment"] + 0.2


def test_phase_alignment_loss_is_differentiable_when_enabled():
    cfg = _cfg()
    cfg.loss.phase_alignment_weight = 0.01
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800, requires_grad=True)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)
    total.backward()

    assert parts["phase_alignment"] >= 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_band_waveform_loss_penalizes_wrong_low_frequency_shape():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.band_waveform_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    same = target.clone()
    distorted = target + 0.4 * torch.sin(2 * torch.pi * 0.45 * time).reshape(1, 1, -1)

    _, same_parts = loss_fn(same, target)
    _, distorted_parts = loss_fn(distorted, target)

    assert same_parts["band_waveform"] < 0.01
    assert distorted_parts["band_waveform"] > same_parts["band_waveform"] + 0.05


def test_curvature_loss_penalizes_local_spike():
    cfg = _cfg()
    cfg.loss.curvature_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 20, 1 / fs)
    smooth = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    spiky = smooth.clone()
    spiky[..., 500] += 5.0

    _, smooth_parts = loss_fn(smooth, smooth)
    _, spiky_parts = loss_fn(spiky, smooth)

    assert spiky_parts["curvature"] > smooth_parts["curvature"] * 10


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


def test_weak_sync_loss_rejects_negative_phase_alignment_weight():
    cfg = _cfg()
    cfg.loss.phase_alignment_weight = -0.1

    with pytest.raises(ValueError, match="phase_alignment_weight 必须非负"):
        WeakSyncLoss(cfg)


def test_weak_sync_loss_rejects_negative_waveform_regularizer_weights():
    cfg = _cfg()
    cfg.loss.band_waveform_weight = -0.1

    with pytest.raises(ValueError, match="band_waveform_weight 必须非负"):
        WeakSyncLoss(cfg)

    cfg = _cfg()
    cfg.loss.curvature_weight = -0.1

    with pytest.raises(ValueError, match="curvature_weight 必须非负"):
        WeakSyncLoss(cfg)


def test_weak_sync_loss_rejects_empty_spectrum_band():
    cfg = _cfg()
    cfg.loss.spectrum_low_hz = 40.0
    cfg.loss.spectrum_high_hz = 45.0
    loss_fn = WeakSyncLoss(cfg)

    with pytest.raises(ValueError, match="频谱损失频带为空"):
        loss_fn(torch.randn(2, 1, 8), torch.randn(2, 1, 8))
