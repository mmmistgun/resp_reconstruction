import pytest
import torch

from resp_train.models.stft_branch import FusionHead, STFTEncoder, TimeStftDual1D, align_to_time


def _encoder(high_hz: float, encoder_type: str = "conv1d") -> STFTEncoder:
    return STFTEncoder(
        sample_rate=100.0,
        stft_win=3000,
        stft_hop=500,
        low_hz=0.05,
        high_hz=high_hz,
        out_channels=16,
        norm="n0",
        encoder_type=encoder_type,
    )


@pytest.mark.parametrize("encoder_type", ["conv1d", "conv2d"])
def test_stft_encoder_output_is_3d_time_series(encoder_type):
    enc = _encoder(3.0, encoder_type)
    x = torch.randn(2, 1, 18000)

    feats = enc(x)

    assert feats.dim() == 3
    assert feats.shape[0] == 2
    assert feats.shape[1] == 16
    assert feats.shape[2] == 37


def test_stft_encoder_preserves_double_dtype_with_double_model():
    enc = _encoder(3.0).double()
    x = torch.randn(1, 1, 18000, dtype=torch.float64)

    feats = enc(x)

    assert feats.dtype == torch.float64


def test_stft_encoder_freq_bins_increase_with_high_hz():
    bins_3 = _encoder(3.0).band_bin_count()
    bins_8 = _encoder(8.0).band_bin_count()
    bins_12 = _encoder(12.0).band_bin_count()

    assert bins_3 < bins_8 < bins_12


def test_stft_encoder_persists_band_scale_for_reproducibility():
    enc = _encoder(8.0)

    assert "band_scale" in enc.state_dict()
    assert enc.state_dict()["band_scale"].shape == (enc.band_bin_count(),)


def test_stft_encoder_rejects_reversed_frequency_band():
    with pytest.raises(ValueError, match="low_hz.*high_hz"):
        STFTEncoder(sample_rate=100.0, stft_win=3000, low_hz=8.0, high_hz=3.0)


def test_stft_encoder_rejects_frequency_above_nyquist():
    with pytest.raises(ValueError, match="high_hz"):
        STFTEncoder(sample_rate=100.0, stft_win=3000, low_hz=0.05, high_hz=60.0)


def test_stft_encoder_rejects_band_without_fft_bins():
    with pytest.raises(ValueError, match="频带"):
        STFTEncoder(sample_rate=100.0, stft_win=100, low_hz=0.05, high_hz=0.1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate": float("nan")}, "sample_rate|有限"),
        ({"high_hz": float("inf")}, "high_hz|有限"),
    ],
)
def test_stft_encoder_rejects_non_finite_frequency_config(kwargs, match):
    params = dict(sample_rate=100.0, stft_win=3000, low_hz=0.05, high_hz=3.0)
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        STFTEncoder(**params)


def test_stft_encoder_rejects_n1_norm_until_task6():
    with pytest.raises(ValueError, match="norm"):
        STFTEncoder(norm="n1")


@pytest.mark.parametrize("encoder_type", ["conv1d", "conv2d"])
def test_stft_encoder_handles_zero_and_nan_input_without_crash(encoder_type):
    enc = _encoder(8.0, encoder_type)
    zero = enc(torch.zeros(1, 1, 18000))
    assert torch.isfinite(zero).all()

    x = torch.randn(1, 1, 18000)
    x[0, 0, :100] = float("nan")
    out = enc(x)
    assert out.shape[0] == 1
    assert torch.isnan(out).any()


def test_stft_encoder_unknown_encoder_type_raises():
    with pytest.raises(ValueError, match="encoder_type"):
        _encoder(3.0, "mixer")


def test_align_to_time_resamples_to_target_length():
    feats = torch.randn(2, 8, 31)

    aligned = align_to_time(feats, target_len=600)

    assert aligned.shape == (2, 8, 600)


def test_fusion_head_outputs_waveform_shape():
    head = FusionHead(in_channels=24, out_length=18000, hidden=16)
    fused = torch.randn(2, 24, 600)

    out = head(fused)

    assert out.shape == (2, 1, 18000)


def _dual(branch_mode: str) -> TimeStftDual1D:
    return TimeStftDual1D(
        time_backbone_name="patch_mixer1d",
        time_backbone_kwargs=dict(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64),
        time_feat_channels=8,
        branch_mode=branch_mode,
        out_length=18000,
        fuse_len=600,
        stft_kwargs=dict(
            sample_rate=100.0,
            stft_win=3000,
            stft_hop=500,
            low_hz=0.05,
            high_hz=3.0,
            out_channels=16,
            norm="n0",
        ),
    )


@pytest.mark.parametrize("branch_mode", ["time_only", "stft_only", "dual"])
def test_dual_outputs_waveform_shape(branch_mode):
    model = _dual(branch_mode)
    x = torch.randn(2, 1, 18000)

    out = model(x)

    assert out.shape == (2, 1, 18000)
    assert torch.isfinite(out).all()


def test_dual_mode_uses_both_branches_gradients():
    model = _dual("dual")
    x = torch.randn(2, 1, 18000)

    model(x).square().mean().backward()

    time_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.time_backbone.parameters())
    stft_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_encoder.parameters())
    assert time_grad and stft_grad


def test_stft_only_mode_has_no_time_backbone():
    model = _dual("stft_only")
    assert model.time_backbone is None


def test_time_only_mode_has_no_stft_encoder():
    model = _dual("time_only")
    assert model.stft_encoder is None


def test_dual_uses_stft_encoder_default_out_channels_when_missing():
    model = TimeStftDual1D(
        time_backbone_name="patch_mixer1d",
        time_backbone_kwargs=dict(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64),
        time_feat_channels=8,
        branch_mode="dual",
        out_length=18000,
        fuse_len=600,
        stft_kwargs=dict(sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=3.0, norm="n0"),
    )

    out = model(torch.randn(1, 1, 18000))

    assert out.shape == (1, 1, 18000)
