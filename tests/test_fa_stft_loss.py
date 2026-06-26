import pytest
import torch
from omegaconf import OmegaConf

from resp_train.losses.stft import TargetStftLoss
from resp_train.losses.weak import WeakSyncLoss


def _weak_cfg():
    return OmegaConf.create(
        {
            "window": {"target_fs": 100},
            "loss": {
                "envelope_weight": 0.0,
                "spectrum_weight": 0.0,
                "smooth_weight": 0.0,
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
                "stft_dist_weight": 1.0,
                "stft_band_energy_weight": 0.0,
                "stft_win_length": 1000,
                "stft_hop_length": 500,
                "stft_n_fft": 1000,
                "stft_dist_low_hz": 0.067,
                "stft_dist_high_hz": 1.2,
                "stft_dist_beta": 3.0,
                "stft_frame_weight_mode": "target_band_energy",
            },
        }
    )


def _sine(freq_hz: float, seconds: float = 60.0, fs: float = 100.0, amp: torch.Tensor | float = 1.0):
    time = torch.arange(0, seconds, 1.0 / fs)
    return (amp * torch.sin(2 * torch.pi * freq_hz * time)).reshape(1, 1, -1)


def test_target_stft_dist_penalizes_harmonic_peak_shift():
    loss_fn = TargetStftLoss(
        sample_rate=100,
        win_length=1000,
        hop_length=500,
        n_fft=1000,
        dist_low_hz=0.067,
        dist_high_hz=1.2,
        dist_beta=3.0,
    )
    target = _sine(0.2)
    same = target.clone()
    harmonic = _sine(0.4)

    same_parts = loss_fn(same, target)
    harmonic_parts = loss_fn(harmonic, target)

    assert same_parts["stft_dist"] < 1e-4
    assert harmonic_parts["stft_dist"] > same_parts["stft_dist"] + 0.05


def test_target_stft_band_energy_penalizes_missing_energy_trajectory():
    loss_fn = TargetStftLoss(
        sample_rate=100,
        win_length=1000,
        hop_length=500,
        n_fft=1000,
        band_energy_bands=[(0.1, 0.7), (0.3, 1.2)],
        band_energy_weights=[1.0, 0.5],
    )
    fs = 100.0
    time = torch.arange(0, 60, 1.0 / fs)
    amp = torch.ones_like(time)
    amp[time >= 30] = 2.0
    target = _sine(0.25, amp=amp)
    same = target.clone()
    flat = _sine(0.25)

    same_parts = loss_fn(same, target)
    flat_parts = loss_fn(flat, target)

    assert same_parts["stft_band_energy"] < 1e-5
    assert flat_parts["stft_band_energy"] > same_parts["stft_band_energy"] + 0.05


def test_weak_sync_loss_includes_target_stft_loss_and_backpropagates():
    cfg = _weak_cfg()
    loss_fn = WeakSyncLoss(cfg)
    target = _sine(0.2)
    pred = _sine(0.4).requires_grad_(True)

    total, parts = loss_fn(pred, target)
    total.backward()

    assert total > 0
    assert parts["stft_dist"] > 0
    assert parts["stft_band_energy"] == 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_weak_sync_loss_can_report_weighted_component_grad_norms():
    cfg = _weak_cfg()
    cfg.loss.stft_band_energy_weight = 0.2
    loss_fn = WeakSyncLoss(cfg)
    target = _sine(0.2)
    pred = _sine(0.4).requires_grad_(True)

    norms = loss_fn.component_gradient_norms(pred, target)

    assert norms["grad_norm_base_total"] == pytest.approx(0.0)
    assert norms["grad_norm_stft_total"] > 0
    assert norms["grad_norm_stft_dist"] > 0


def test_target_stft_loss_rejects_empty_frequency_band():
    loss_fn = TargetStftLoss(
        sample_rate=100,
        win_length=1000,
        hop_length=500,
        n_fft=1000,
        dist_low_hz=60.0,
        dist_high_hz=70.0,
    )

    with pytest.raises(ValueError, match="STFT 频带为空"):
        loss_fn(_sine(0.2), _sine(0.2))
