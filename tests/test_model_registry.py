import pytest
from omegaconf import OmegaConf
import torch
from torch import nn

from resp_train.models import build_model, list_models
from resp_train.models.timeseries import PatchMixer1D, PeriodicUNet1DTiny


def test_unet1d_tiny_is_registered():
    assert "unet1d_tiny" in list_models()


def test_unet1d_tiny_noskip1_is_registered():
    assert "unet1d_tiny_noskip1" in list_models()


def test_unet1d_tiny_noskip_all_is_registered():
    assert "unet1d_tiny_noskip_all" in list_models()


@pytest.mark.parametrize(
    "model_name",
    [
        "periodic_unet1d_tiny",
        "patch_mixer1d",
        "dlinear_waveform",
    ],
)
def test_time_series_library_inspired_models_are_registered(model_name):
    assert model_name in list_models()


def test_fir_frontend_patch_mixer_is_registered():
    assert "patch_mixer1d_fir_frontend" in list_models()


@pytest.mark.parametrize(
    "model_name",
    [
        "basis_decoder1d",
        "multiscale_decomp_mixer1d",
        "timesnet_lite1d",
        "frequency_bottleneck1d",
        "downsampled_ssm1d",
    ],
)
def test_lowfreq_structure_models_are_registered(model_name):
    assert model_name in list_models()


def test_build_unet1d_tiny_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
            }
        }
    )
    model = build_model(cfg)

    x = torch.randn(2, 1, 18000)
    y = model(x)

    assert y.shape == x.shape


def test_build_unet1d_tiny_uses_configured_channels_and_handles_odd_length():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny",
                "in_channels": 2,
                "out_channels": 1,
                "base_channels": 4,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 2, 18001))

    assert y.shape == (2, 1, 18001)


def test_build_unet1d_tiny_noskip1_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny_noskip1",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 18001))

    assert y.shape == (2, 1, 18001)


def test_build_unet1d_tiny_noskip_all_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny_noskip_all",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 18001))

    assert y.shape == (2, 1, 18001)


@pytest.mark.parametrize(
    "model_name",
    [
        "periodic_unet1d_tiny",
        "patch_mixer1d",
        "dlinear_waveform",
    ],
)
def test_build_time_series_library_inspired_models_preserve_waveform_shape(model_name):
    cfg = OmegaConf.create(
        {
            "model": {
                "name": model_name,
                "in_channels": 2,
                "out_channels": 1,
                "base_channels": 4,
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 2,
                "moving_avg": 101,
                "output_smoothing_kernel": 5,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 2, 1025))

    assert y.shape == (2, 1, 1025)
    if model_name == "patch_mixer1d":
        assert isinstance(model.output_smoother, nn.Module)
        assert not isinstance(model.output_smoother, nn.Identity)


def test_build_fir_frontend_patch_mixer_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "patch_mixer1d_fir_frontend",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 1,
                "overlap_window": "hann",
                "fir_kernel_size": 401,
                "fir_low_hz": 0.05,
                "fir_high_hz": 0.7,
                "fir_sample_rate": 100.0,
                "fir_trainable": True,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 1025))

    assert y.shape == (2, 1, 1025)
    assert model.fir.weight.requires_grad


@pytest.mark.parametrize(
    "model_name,extra",
    [
        ("basis_decoder1d", {"basis_count": 64, "encoder_stride": 20}),
        ("multiscale_decomp_mixer1d", {"downsample_factors": [1, 4, 16], "mixer_layers": 2}),
        ("timesnet_lite1d", {"period_top_k": 3, "period_min_sec": 2.0, "period_max_sec": 20.0}),
        ("frequency_bottleneck1d", {"max_freq_hz": 0.7, "freq_bins": 128}),
        ("downsampled_ssm1d", {"latent_stride": 20, "state_layers": 2}),
    ],
)
def test_lowfreq_structure_models_preserve_waveform_shape(model_name, extra):
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 1800},
            "model": {
                "name": model_name,
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                **extra,
            },
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 1800))

    assert y.shape == (2, 1, 1800)
    assert torch.isfinite(y).all()


def test_patch_mixer_hann_overlap_add_reduces_patch_boundary_step():
    uniform = PatchMixer1D(in_channels=1, out_channels=1, patch_len=8, patch_stride=4, overlap_window="uniform")
    hann = PatchMixer1D(in_channels=1, out_channels=1, patch_len=8, patch_stride=4, overlap_window="hann")
    patches = torch.zeros(1, 2, 1, 8)
    patches[:, 1] = 1.0

    uniform_out = uniform._overlap_add(patches, length=12, padded_length=12)
    hann_out = hann._overlap_add(patches, length=12, padded_length=12)

    uniform_step = torch.abs(uniform_out[..., 4] - uniform_out[..., 3])
    hann_step = torch.abs(hann_out[..., 4] - hann_out[..., 3])
    assert hann_step.item() < uniform_step.item()


def test_patch_mixer_hann_overlap_add_preserves_constant_signal():
    hann = PatchMixer1D(in_channels=1, out_channels=1, patch_len=8, patch_stride=4, overlap_window="hann")
    patches = torch.ones(1, 2, 1, 8)

    y = hann._overlap_add(patches, length=12, padded_length=12)

    assert torch.allclose(y, torch.ones_like(y))


def test_patch_mixer_output_smoothing_reduces_local_jitter():
    raw = PatchMixer1D(
        in_channels=1,
        out_channels=1,
        base_channels=1,
        patch_len=8,
        patch_stride=8,
        mixer_layers=0,
        overlap_window="uniform",
        output_smoothing_kernel=1,
    )
    smooth = PatchMixer1D(
        in_channels=1,
        out_channels=1,
        base_channels=1,
        patch_len=8,
        patch_stride=8,
        mixer_layers=0,
        overlap_window="uniform",
        output_smoothing_kernel=5,
    )
    smooth.load_state_dict(raw.state_dict(), strict=False)
    for model in (raw, smooth):
        model.patch_embed.weight.data.zero_()
        model.patch_embed.bias.data.zero_()
        model.patch_head.weight.data.zero_()
        model.patch_head.bias.data.copy_(torch.tensor([1.0, -1.0] * 4))

    x = torch.zeros(1, 1, 64)
    raw_y = raw(x)
    smooth_y = smooth(x)

    assert smooth_y.shape == raw_y.shape
    assert torch.mean(torch.abs(smooth_y[..., 1:] - smooth_y[..., :-1])).item() < torch.mean(
        torch.abs(raw_y[..., 1:] - raw_y[..., :-1])
    ).item()


def test_periodic_unet_output_smoothing_reduces_local_jitter():
    class ZeroResidual(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

    model = PeriodicUNet1DTiny(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        lowpass_kernel=3,
        output_smoothing_kernel=5,
    )
    model.smoother = nn.Identity()
    model.periodic_frontend = ZeroResidual()
    model.backbone = nn.Identity()

    x = torch.tensor([[[1.0, -1.0] * 16]])
    y = model(x)

    assert y.shape == x.shape
    assert torch.mean(torch.abs(y[..., 1:] - y[..., :-1])).item() < torch.mean(torch.abs(x[..., 1:] - x[..., :-1])).item()


def test_build_model_unknown_name_lists_available_models():
    cfg = OmegaConf.create({"model": {"name": "unknown"}})

    with pytest.raises(KeyError, match="未知模型.*unet1d_tiny"):
        build_model(cfg)
