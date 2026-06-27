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
                "rhythm_weight": 0.0,
                "signed_cosine_weight": 0.0,
                "signed_corr_weight": 0.0,
                "signed_rms_envelope_weight": 0.0,
                "signed_mean_weight": 0.0,
                "si_sdr_weight": 0.0,
                "phase_alignment_zero_weight": 0.5,
                "phase_alignment_lag_weight": 0.5,
                "phase_alignment_lag_penalty_weight": 0.1,
                "phase_alignment_max_sec": 0.5,
                "phase_alignment_step_sec": 0.1,
                "phase_alignment_temperature": 0.05,
                "signed_env_temperature": 0.25,
                "signed_mean_window_sec": 2.0,
                "rhythm_min_period_sec": 2.0,
                "rhythm_max_period_sec": 20.0,
                "rhythm_step_sec": 0.25,
                "rhythm_temperature": 0.08,
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
        "rhythm",
        "signed_cosine",
        "signed_corr",
        "signed_rms_envelope",
        "signed_mean",
        "si_sdr",
        "stft_dist",
        "stft_band_energy",
        "stft_peak_anchor",
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
    cfg.loss.rhythm_weight = 0.0
    cfg.loss.signed_cosine_weight = 0.0
    cfg.loss.signed_corr_weight = 0.0
    cfg.loss.signed_rms_envelope_weight = 0.0
    cfg.loss.signed_mean_weight = 0.0
    cfg.loss.si_sdr_weight = 0.0
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
        + cfg.loss.rhythm_weight * parts["rhythm"]
        + cfg.loss.signed_cosine_weight * parts["signed_cosine"]
        + cfg.loss.signed_corr_weight * parts["signed_corr"]
        + cfg.loss.signed_rms_envelope_weight * parts["signed_rms_envelope"]
        + cfg.loss.signed_mean_weight * parts["signed_mean"]
        + cfg.loss.si_sdr_weight * parts["si_sdr"]
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


def _disable_base_losses(cfg):
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.high_freq_weight = 0.0
    cfg.loss.relative_envelope_weight = 0.0
    cfg.loss.phase_alignment_weight = 0.0
    cfg.loss.band_waveform_weight = 0.0
    cfg.loss.curvature_weight = 0.0
    cfg.loss.rhythm_weight = 0.0


def _resp_sine(cfg, seconds: float = 60.0, freq_hz: float = 0.25) -> torch.Tensor:
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, seconds, 1 / fs)
    return torch.sin(2 * torch.pi * freq_hz * time).reshape(1, 1, -1)


def test_stft_sample_weights_can_use_inverse_waveform_confidence_score():
    cfg = _cfg()
    cfg.loss.stft_sample_weight_mode = "waveform_confidence_score_inverse"
    cfg.loss.stft_sample_weight_min = 0.05
    loss_fn = WeakSyncLoss(cfg)

    weights = loss_fn.sample_weights_from_meta(
        {"waveform_confidence_score": torch.tensor([0.95, 0.2, 0.0])},
        device=torch.device("cpu"),
    )

    assert torch.allclose(weights, torch.tensor([0.05, 0.8, 1.0]))


def test_stft_sample_weights_can_use_waveform_confidence_level_medlow():
    cfg = _cfg()
    cfg.loss.stft_sample_weight_mode = "waveform_confidence_level_medlow"
    loss_fn = WeakSyncLoss(cfg)

    weights = loss_fn.sample_weights_from_meta(
        {"waveform_confidence_level": ["high", "medium", "low"]},
        device=torch.device("cpu"),
    )

    assert torch.allclose(weights, torch.tensor([0.0, 1.0, 1.0]))


def test_weak_sync_stft_loss_sample_weights_only_gate_stft_components():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.stft_dist_weight = 1.0
    cfg.loss.stft_band_energy_weight = 0.0
    cfg.loss.stft_win_length = 1000
    cfg.loss.stft_hop_length = 500
    cfg.loss.stft_n_fft = 1000
    cfg.loss.stft_sample_weight_mode = "waveform_confidence_score_inverse"
    loss_fn = WeakSyncLoss(cfg)
    good_target = _resp_sine(cfg, freq_hz=0.2)
    good_pred = good_target.clone()
    bad_target = _resp_sine(cfg, freq_hz=0.2)
    bad_pred = _resp_sine(cfg, freq_hz=0.4)
    target = torch.cat([good_target, bad_target], dim=0)
    pred = torch.cat([good_pred, bad_pred], dim=0)
    sample_weight = loss_fn.sample_weights_from_meta(
        {"waveform_confidence_score": torch.tensor([0.0, 1.0])},
        device=torch.device("cpu"),
    )

    weighted_total, weighted_parts = loss_fn(pred, target, sample_weight=sample_weight)
    unweighted_total, unweighted_parts = loss_fn(pred, target)

    assert weighted_parts["stft_dist"] < 1e-4
    assert unweighted_parts["stft_dist"] > weighted_parts["stft_dist"] + 0.02
    assert unweighted_total > weighted_total + 0.02


def test_signed_cosine_anchor_penalizes_inverted_band_signal_and_ignores_scale():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.signed_cosine_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    target = _resp_sine(cfg)
    same_scaled = 3.0 * target
    inverted = -target

    _, same_parts = loss_fn(same_scaled, target)
    _, inverted_parts = loss_fn(inverted, target)

    assert same_parts["signed_cosine"] < 0.05
    assert inverted_parts["signed_cosine"] > same_parts["signed_cosine"] + 1.5


def test_signed_cosine_weight_schedule_linearly_changes_effective_weight():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.signed_cosine_weight = 0.1
    cfg.loss.signed_cosine_schedule = {
        "mode": "linear",
        "start_epoch": 1,
        "end_epoch": 5,
        "start_weight": 0.1,
        "end_weight": 0.05,
    }
    loss_fn = WeakSyncLoss(cfg)
    target = _resp_sine(cfg)
    inverted = -target

    loss_fn.set_epoch(1)
    early_total, early_parts = loss_fn(inverted, target)
    loss_fn.set_epoch(5)
    late_total, late_parts = loss_fn(inverted, target)

    assert early_parts["signed_cosine"] > 1.5
    assert torch.allclose(early_parts["signed_cosine"], late_parts["signed_cosine"], atol=1e-6)
    assert torch.isclose(late_total, early_total * 0.5, rtol=1e-4, atol=1e-6)
    assert loss_fn.current_weights()["signed_cosine"] == pytest.approx(0.05)


def test_signed_corr_weight_schedule_linearly_changes_effective_weight():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.signed_corr_weight = 0.2
    cfg.loss.signed_corr_schedule = {
        "mode": "linear",
        "start_epoch": 1,
        "end_epoch": 6,
        "start_weight": 0.6,
        "end_weight": 0.2,
    }
    loss_fn = WeakSyncLoss(cfg)
    target = _resp_sine(cfg)
    inverted = -target

    loss_fn.set_epoch(1)
    early_total, early_parts = loss_fn(inverted, target)
    loss_fn.set_epoch(6)
    late_total, late_parts = loss_fn(inverted, target)

    assert early_parts["signed_corr"] > 1.5
    assert torch.allclose(early_parts["signed_corr"], late_parts["signed_corr"], atol=1e-6)
    assert torch.isclose(early_total, late_total * 3.0, rtol=1e-4, atol=1e-6)
    assert loss_fn.current_weights()["signed_corr"] == pytest.approx(0.2)


def test_signed_corr_anchor_penalizes_inverted_band_signal_and_is_differentiable():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.signed_corr_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    target = _resp_sine(cfg)
    pred = (-target).clone().requires_grad_(True)

    total, parts = loss_fn(pred, target)
    total.backward()

    assert parts["signed_corr"] > 1.5
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_signed_rms_envelope_anchor_combines_effort_and_low_frequency_sign():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.signed_rms_envelope_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    carrier = torch.sin(2 * torch.pi * 0.25 * time)
    effort = torch.ones_like(time)
    effort[(time >= 20) & (time <= 40)] = 1.8
    target = (effort * carrier).reshape(1, 1, -1)
    same_scaled = 2.0 * target
    inverted = -target

    _, same_parts = loss_fn(same_scaled, target)
    _, inverted_parts = loss_fn(inverted, target)

    assert same_parts["signed_rms_envelope"] < 0.05
    assert inverted_parts["signed_rms_envelope"] > same_parts["signed_rms_envelope"] + 1.0


def test_signed_mean_anchor_penalizes_inverted_windowed_effort():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.signed_mean_weight = 1.0
    cfg.loss.signed_mean_window_sec = 2.0
    loss_fn = WeakSyncLoss(cfg)
    target = _resp_sine(cfg)
    same = target.clone()
    inverted = -target

    _, same_parts = loss_fn(same, target)
    _, inverted_parts = loss_fn(inverted, target)

    assert same_parts["signed_mean"] < 0.05
    assert inverted_parts["signed_mean"] > same_parts["signed_mean"] + 1.0


def test_si_sdr_loss_is_scale_invariant_but_not_a_signed_anchor():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.si_sdr_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    target = _resp_sine(cfg)
    same_scaled = 3.0 * target
    inverted = -target
    distorted = target + 0.5 * _resp_sine(cfg, freq_hz=0.45)

    _, same_parts = loss_fn(same_scaled, target)
    _, inverted_parts = loss_fn(inverted, target)
    _, distorted_parts = loss_fn(distorted, target)

    assert same_parts["si_sdr"] < 0.01
    assert inverted_parts["si_sdr"] < 0.01
    assert distorted_parts["si_sdr"] > same_parts["si_sdr"] + 0.1


def test_si_sdr_with_signed_corr_can_penalize_inversion():
    cfg = _cfg()
    _disable_base_losses(cfg)
    cfg.loss.si_sdr_weight = 1.0
    cfg.loss.signed_corr_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    target = _resp_sine(cfg)
    inverted = -target

    total, parts = loss_fn(inverted, target)

    assert parts["si_sdr"] < 0.01
    assert parts["signed_corr"] > 1.5
    assert total > 1.5


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


def test_rhythm_loss_penalizes_wrong_period_but_tolerates_phase_shift():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.rhythm_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 80, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    shifted_same_period = torch.sin(2 * torch.pi * 0.25 * time + torch.pi / 2).reshape(1, 1, -1)
    wrong_period = torch.sin(2 * torch.pi * 0.42 * time).reshape(1, 1, -1)

    _, shifted_parts = loss_fn(shifted_same_period, target)
    _, wrong_parts = loss_fn(wrong_period, target)

    assert shifted_parts["rhythm"] < 0.05
    assert wrong_parts["rhythm"] > shifted_parts["rhythm"] + 0.5


def test_rhythm_loss_is_differentiable_when_enabled():
    cfg = _cfg()
    cfg.loss.rhythm_weight = 0.02
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800, requires_grad=True)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)
    total.backward()

    assert parts["rhythm"] >= 0
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

    for name in (
        "signed_cosine_weight",
        "signed_corr_weight",
        "signed_rms_envelope_weight",
        "signed_mean_weight",
        "si_sdr_weight",
    ):
        cfg = _cfg()
        cfg.loss[name] = -0.1

        with pytest.raises(ValueError, match=f"{name} 必须非负"):
            WeakSyncLoss(cfg)


def test_weak_sync_loss_rejects_invalid_rhythm_config():
    cfg = _cfg()
    cfg.loss.rhythm_weight = -0.1

    with pytest.raises(ValueError, match="rhythm_weight 必须非负"):
        WeakSyncLoss(cfg)

    cfg = _cfg()
    cfg.loss.rhythm_min_period_sec = 20.0
    cfg.loss.rhythm_max_period_sec = 2.0

    with pytest.raises(ValueError, match="rhythm_min_period_sec 必须小于 rhythm_max_period_sec"):
        WeakSyncLoss(cfg)

    cfg = _cfg()
    cfg.loss.rhythm_step_sec = 0.0

    with pytest.raises(ValueError, match="rhythm_step_sec 必须为正数"):
        WeakSyncLoss(cfg)

    cfg = _cfg()
    cfg.loss.rhythm_temperature = 0.0

    with pytest.raises(ValueError, match="rhythm_temperature 必须为正数"):
        WeakSyncLoss(cfg)


def test_weak_sync_loss_rejects_empty_spectrum_band():
    cfg = _cfg()
    cfg.loss.spectrum_low_hz = 40.0
    cfg.loss.spectrum_high_hz = 45.0
    loss_fn = WeakSyncLoss(cfg)

    with pytest.raises(ValueError, match="频谱损失频带为空"):
        loss_fn(torch.randn(2, 1, 8), torch.randn(2, 1, 8))
