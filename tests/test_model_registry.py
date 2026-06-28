import pytest
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from resp_train.models import build_model, list_models
from resp_train.models.lowfreq import MultiScaleDecompMixer1D, TimesNetLite1D
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
        "patch_hann_control_point_decoder1d",
        "patch_hann_basis_residual_decoder1d",
        "patch_hann_bandlimited_output1d",
        "multiscale_patch_hann_bandlimited1d",
        "period_aware_patch_hann_bandlimited1d",
        "polyphase_patch_hann_bandlimited1d",
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
        (
            "patch_hann_control_point_decoder1d",
            {"patch_len": 128, "patch_stride": 64, "mixer_layers": 1, "control_points": 96},
        ),
        (
            "patch_hann_basis_residual_decoder1d",
            {"patch_len": 128, "patch_stride": 64, "mixer_layers": 1, "basis_count": 64, "residual_scale": 0.05},
        ),
        (
            "patch_hann_bandlimited_output1d",
            {"patch_len": 128, "patch_stride": 64, "mixer_layers": 1, "max_freq_hz": 0.7},
        ),
        (
            "multiscale_patch_hann_bandlimited1d",
            {"patch_lengths": [128, 256], "patch_stride_ratio": 0.5, "mixer_layers": 1, "max_freq_hz": 0.7},
        ),
        (
            "period_aware_patch_hann_bandlimited1d",
            {"period_secs": [2.5, 4.0], "patch_stride_ratio": 0.5, "mixer_layers": 1, "max_freq_hz": 0.7},
        ),
        (
            "polyphase_patch_hann_bandlimited1d",
            {"offsets": [1, 2, 4], "patch_len": 128, "patch_stride": 64, "mixer_layers": 1, "max_freq_hz": 0.7},
        ),
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


def test_timesnet_lite_uses_lowpass_signal_for_period_selection():
    class RecordingTimesNetLite1D(TimesNetLite1D):
        def __init__(self) -> None:
            super().__init__(
                in_channels=1,
                out_channels=1,
                base_channels=2,
                period_top_k=1,
                period_min_sec=2.0,
                period_max_sec=20.0,
                sample_rate=100.0,
                lowpass_kernel=401,
            )
            self.period_input: torch.Tensor | None = None

        def _periods(self, x: torch.Tensor) -> list[int]:
            self.period_input = x.detach().clone()
            return [self.min_period]

    sample_rate = 100.0
    t = torch.arange(2000, dtype=torch.float32) / sample_rate
    slow = torch.sin(2.0 * torch.pi * 0.1 * t)
    strong_fast = 8.0 * torch.sin(2.0 * torch.pi * 0.5 * t)
    raw = (slow + strong_fast).view(1, 1, -1)

    reference = TimesNetLite1D(
        in_channels=1,
        out_channels=1,
        base_channels=2,
        period_top_k=1,
        period_min_sec=2.0,
        period_max_sec=20.0,
        sample_rate=sample_rate,
        lowpass_kernel=401,
    )
    raw_period = reference._periods(raw)[0]
    low = reference.lowpass(raw)
    lowpass_period = reference._periods(low)[0]

    assert raw_period == 200
    assert lowpass_period == 1000

    model = RecordingTimesNetLite1D()
    model(raw)

    assert model.period_input is not None
    assert torch.allclose(model.period_input, model.lowpass(raw))


@pytest.mark.parametrize(
    "model_name,extra",
    [
        ("basis_decoder1d", {"basis_count": 32, "encoder_stride": 40}),
        ("multiscale_decomp_mixer1d", {"downsample_factors": [1, 8, 32], "mixer_layers": 1}),
        (
            "timesnet_lite1d",
            {"period_top_k": 1, "period_min_sec": 2.0, "period_max_sec": 20.0, "lowpass_kernel": 401},
        ),
        ("frequency_bottleneck1d", {"max_freq_hz": 0.7, "freq_bins": 64}),
        ("downsampled_ssm1d", {"latent_stride": 40, "state_layers": 1}),
        (
            "patch_hann_control_point_decoder1d",
            {"patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "control_points": 180},
        ),
        (
            "patch_hann_basis_residual_decoder1d",
            {"patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "basis_count": 96, "residual_scale": 0.05},
        ),
        (
            "patch_hann_bandlimited_output1d",
            {"patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "max_freq_hz": 0.7},
        ),
        (
            "multiscale_patch_hann_bandlimited1d",
            {"patch_lengths": [256, 512], "patch_stride_ratio": 0.5, "mixer_layers": 1, "max_freq_hz": 0.7},
        ),
        (
            "period_aware_patch_hann_bandlimited1d",
            {"period_secs": [2.5, 4.0], "patch_stride_ratio": 0.5, "mixer_layers": 1, "max_freq_hz": 0.7},
        ),
        (
            "polyphase_patch_hann_bandlimited1d",
            {"offsets": [1, 2, 4], "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "max_freq_hz": 0.7},
        ),
    ],
)
def test_lowfreq_structure_models_support_full_window_backward(model_name, extra):
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": model_name,
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
                **extra,
            },
        }
    )
    model = build_model(cfg)
    x = torch.randn(1, 1, 18000)

    y = model(x)
    loss = y.square().mean()
    loss.backward()

    grad_norm = sum(
        float(param.grad.detach().abs().sum())
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    )
    assert y.shape == (1, 1, 18000)
    assert torch.isfinite(y).all()
    assert grad_norm > 0.0


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


def test_patch_mixer_return_features_is_backward_compatible():
    model = PatchMixer1D(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64)
    model.eval()
    x = torch.randn(2, 1, 4096)

    with torch.no_grad():
        default_out = model(x)
        explicit_out = model(x, return_features=False)

    assert torch.equal(default_out, explicit_out)


def test_patch_mixer_return_features_shape_contract():
    model = PatchMixer1D(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64)
    x = torch.randn(2, 1, 4096)

    features, length = model(x, return_features=True)

    assert features.dim() == 3
    assert features.shape == (2, 8, 63)
    assert length == 4096


def test_multiscale_decomp_return_features_is_backward_compatible():
    model = MultiScaleDecompMixer1D(in_channels=1, out_channels=1, base_channels=8, downsample_factors=[1, 4, 16])
    model.eval()
    x = torch.randn(2, 1, 4096)

    with torch.no_grad():
        default_out = model(x)
        explicit_out = model(x, return_features=False)

    assert torch.equal(default_out, explicit_out)


def test_multiscale_decomp_return_features_shape_contract():
    model = MultiScaleDecompMixer1D(in_channels=1, out_channels=1, base_channels=8, downsample_factors=[1, 4, 16])
    x = torch.randn(2, 1, 4096)

    features, length = model(x, return_features=True)

    assert features.shape == (2, 8 * 3, 4096)
    assert length == 4096


def test_time_stft_dual1d_is_registered():
    assert "time_stft_dual1d" in list_models()


def test_time_stft_low_complex_output1d_is_registered():
    assert "time_stft_low_complex_output1d" in list_models()


@pytest.mark.parametrize("branch_mode", ["time_only", "stft_only", "dual"])
def test_build_time_stft_dual1d_preserves_waveform_shape(branch_mode):
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": branch_mode,
                "time_backbone": "patch_mixer1d",
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 1,
                "stft_win": 3000,
                "stft_hop": 500,
                "stft_low_hz": 0.05,
                "stft_high_hz": 3.0,
                "stft_out_channels": 16,
                "stft_norm": "n0",
                "fuse_len": 600,
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()


def test_build_time_stft_dual1d_with_multiscale_backbone_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "dual",
                "time_backbone": "multiscale_decomp_mixer1d",
                "downsample_factors": [1, 4, 16],
                "mixer_layers": 1,
                "stft_win": 512,
                "stft_hop": 128,
                "stft_low_hz": 0.05,
                "stft_high_hz": 3.0,
                "stft_out_channels": 16,
                "stft_norm": "n0",
                "fuse_len": 128,
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 4096))
    assert y.shape == (2, 1, 4096)
    assert torch.isfinite(y).all()


def test_build_time_stft_dual1d_injects_n1_band_scale(tmp_path):
    stft_win = 512
    sample_rate = 100.0
    freqs = torch.fft.rfftfreq(stft_win, d=1.0 / sample_rate)
    start = int(torch.searchsorted(freqs, torch.tensor(0.05), right=False).item())
    end = int(torch.searchsorted(freqs, torch.tensor(3.0), right=True).item())
    scale = np.linspace(1.0, 2.0, end - start, dtype=np.float32)
    scale_path = tmp_path / "band_scale.npy"
    np.save(scale_path, scale)

    cfg = OmegaConf.create(
        {
            "window": {"target_fs": sample_rate, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "stft_only",
                "time_backbone": "patch_mixer1d",
                "stft_win": stft_win,
                "stft_hop": 128,
                "stft_low_hz": 0.05,
                "stft_high_hz": 3.0,
                "stft_out_channels": 16,
                "stft_norm": "n1",
                "stft_band_scale_path": str(scale_path),
                "fuse_len": 128,
            },
        }
    )

    model = build_model(cfg)

    assert torch.allclose(model.stft_encoder.band_scale, torch.from_numpy(scale))


def test_build_time_stft_dual1d_uses_stft_encoder_type_field():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "stft_only",
                "time_backbone": "patch_mixer1d",
                "stft_win": 512,
                "stft_hop": 128,
                "stft_low_hz": 0.05,
                "stft_high_hz": 3.0,
                "stft_out_channels": 16,
                "stft_norm": "n0",
                "stft_encoder_type": "conv2d",
                "fuse_len": 128,
            },
        }
    )

    model = build_model(cfg)

    assert model.stft_encoder.encoder_type == "conv2d"


@pytest.mark.parametrize("encoder_type", ["freq_mlp", "soft_band"])
def test_build_time_stft_dual1d_frequency_frontend_encoder_types(encoder_type):
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "dual",
                "time_backbone": "patch_mixer1d",
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 1,
                "stft_win": 512,
                "stft_hop": 128,
                "stft_low_hz": 0.05,
                "stft_high_hz": 3.0,
                "stft_out_channels": 16,
                "stft_norm": "n0",
                "stft_encoder_type": encoder_type,
                "fuse_len": 128,
                "fusion_mode": "concat_generic",
            },
        }
    )

    model = build_model(cfg)
    y = model(torch.randn(1, 1, 4096))

    assert model.stft_encoder.encoder_type == encoder_type
    assert y.shape == (1, 1, 4096)
    assert torch.isfinite(y).all()


def test_build_time_stft_dual1d_uses_fusion_decoder_field():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "time_only",
                "time_backbone": "multiscale_decomp_mixer1d",
                "downsample_factors": [1, 4, 16],
                "mixer_layers": 1,
                "fuse_len": 4096,
                "fusion_decoder": "lite",
            },
        }
    )

    model = build_model(cfg)

    assert model.fusion_head.decoder_style == "lite"


def test_build_time_stft_dual1d_stft_only_does_not_require_time_backbone_fields():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d",
                "branch_mode": "stft_only",
                "stft_win": 512,
                "stft_hop": 128,
                "stft_low_hz": 0.05,
                "stft_high_hz": 3.0,
                "stft_out_channels": 16,
                "stft_norm": "n0",
                "fuse_len": 128,
            },
        }
    )

    model = build_model(cfg)
    y = model(torch.randn(1, 1, 4096))

    assert y.shape == (1, 1, 4096)


def test_build_time_stft_dual1d_time_only_ignores_unused_stft_fields():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "time_only",
                "time_backbone": "patch_mixer1d",
                "patch_len": 128,
                "patch_stride": 64,
                "stft_high_hz": "unused-invalid",
                "fuse_len": 128,
            },
        }
    )

    model = build_model(cfg)
    y = model(torch.randn(1, 1, 4096))

    assert y.shape == (1, 1, 4096)


def test_patch_mixer_decode_from_features_matches_native_forward():
    torch.manual_seed(0)
    model = PatchMixer1D(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64)
    model.eval()
    x = torch.randn(2, 1, 4096)

    with torch.no_grad():
        native = model(x)
        tokens, length = model(x, return_features=True)
        recomposed = model.decode_from_features(tokens, length)

    assert torch.equal(native, recomposed)


def test_patch_mixer_decode_from_features_shape():
    model = PatchMixer1D(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64)
    x = torch.randn(2, 1, 4096)
    tokens, length = model(x, return_features=True)

    out = model.decode_from_features(tokens, length)

    assert out.shape == (2, 1, 4096)


def test_build_time_stft_dual1d_native_inject_patch_preserves_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.fusion_head is None  # native_inject 不建通用融合头


def test_build_time_stft_low_complex_output1d_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_low_complex_output1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "dual",
                "time_backbone": "patch_mixer1d",
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 1,
                "overlap_window": "hann",
                "stft_win": 512,
                "stft_hop": 128,
                "stft_low_hz": 0.05,
                "stft_high_hz": 8.0,
                "stft_out_channels": 8,
                "stft_norm": "n0",
                "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject",
                "stft_inject_position": "pre_mixer",
                "output_stft_win_length": 512,
                "output_stft_hop_length": 128,
                "output_stft_n_fft": 512,
                "output_stft_center": True,
                "output_stft_low_hz": 0.0,
                "output_stft_high_hz": 3.0,
                "output_stft_hidden_channels": 8,
            },
        }
    )
    model = build_model(cfg)

    out = model(torch.randn(2, 1, 4096))

    assert set(out) == {"waveform", "output_stft_realimag"}
    assert out["waveform"].shape == (2, 1, 4096)
    assert out["output_stft_realimag"].shape[:2] == (2, 2)
    assert torch.isfinite(out["waveform"]).all()
    assert model.output_stft_head.high_hz == pytest.approx(3.0)


def test_build_time_stft_dual1d_passes_fb_residual_head_config():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject", "stft_inject_position": "pre_mixer",
                "fb_residual_head": "low_complex_residual", "fb_residual_scale": 0.03,
                "fb_residual_stft_win_length": 3000, "fb_residual_stft_hop_length": 500,
                "fb_residual_stft_n_fft": 3000, "fb_residual_stft_center": True,
                "fb_residual_stft_low_hz": 0.067, "fb_residual_stft_high_hz": 1.2,
            },
        }
    )
    model = build_model(cfg)

    out = model(torch.randn(2, 1, 18000))

    assert set(out) == {"waveform", "base_waveform", "residual_waveform"}
    assert out["waveform"].shape == (2, 1, 18000)
    assert model.fb_residual_head.residual_scale == pytest.approx(0.03)


def test_build_time_stft_dual1d_passes_enc3_residual_head_and_cap_config():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject", "stft_inject_position": "pre_mixer",
                "fb_residual_head": "enc3_tfgrid_residual", "fb_residual_scale": 0.03,
                "fb_residual_energy_cap": 0.05,
                "fb_residual_stft_win_length": 3000, "fb_residual_stft_hop_length": 500,
                "fb_residual_stft_n_fft": 3000, "fb_residual_stft_center": True,
                "fb_residual_stft_low_hz": 0.067, "fb_residual_stft_high_hz": 1.2,
            },
        }
    )
    model = build_model(cfg)

    out = model(torch.randn(2, 1, 18000))

    assert set(out) == {"waveform", "base_waveform", "residual_waveform"}
    assert out["waveform"].shape == (2, 1, 18000)
    assert model.fb_residual_head_name == "enc3_tfgrid_residual"
    assert model.fb_residual_energy_cap == pytest.approx(0.05)


def test_build_time_stft_dual1d_passes_band_aware_aux_head_config():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject", "stft_inject_position": "pre_mixer",
                "fb_aux_head": "enc2_band_aware_aux",
                "fb_aux_stft_win_length": 3000, "fb_aux_stft_hop_length": 500,
                "fb_aux_stft_n_fft": 3000, "fb_aux_stft_center": False,
                "fb_aux_stft_low_hz": 0.033, "fb_aux_stft_high_hz": 3.0,
            },
        }
    )
    model = build_model(cfg)

    out = model(torch.randn(2, 1, 18000))

    assert set(out) == {"waveform", "aux_target_stft_logmag"}
    assert out["aux_target_stft_logmag"].shape == (2, 90, 31)
    assert model.fb_aux_head_name == "enc2_band_aware_aux"


@pytest.mark.parametrize("position", ["pre_mixer", "mid_mixer", "post_mixer"])
def test_build_time_stft_dual1d_passes_stft_inject_position(position):
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject", "stft_inject_position": position,
            },
        }
    )

    model = build_model(cfg)

    assert model.stft_inject_position == position


def test_build_time_stft_dual1d_token_context_inject_patch_preserves_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "token_context_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.fusion_head is None
    assert model.stft_adapter is not None


def test_build_time_stft_dual1d_gated_native_inject_patch_preserves_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "gated_native_inject", "stft_inject_position": "pre_mixer",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.fusion_head is None
    assert model.stft_gate is not None


def test_build_time_stft_dual1d_cross_attention_inject_patch_preserves_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "cross_attention_inject", "stft_inject_position": "pre_mixer",
                "cross_attention_heads": 2,
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.fusion_head is None
    assert model.cross_attention_adapter is not None
    assert model.cross_attention_adapter.attn.num_heads == 2


def test_build_time_stft_dual1d_native_inject_multiscale_raises():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "multiscale_decomp_mixer1d",
                "downsample_factors": [1, 4, 16], "mixer_layers": 1,
                "stft_win": 512, "stft_hop": 128, "stft_low_hz": 0.05, "stft_high_hz": 3.0,
                "stft_out_channels": 16, "stft_norm": "n0",
                "fusion_mode": "native_inject",
            },
        }
    )
    with pytest.raises(ValueError, match="native_inject"):
        build_model(cfg)


def test_build_time_stft_dual1d_defaults_to_concat_generic():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1,
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0",
            },
        }
    )
    model = build_model(cfg)
    assert model.fusion_mode == "concat_generic"
    assert model.fusion_head is not None


def test_build_time_stft_dual1d_bandenergy_native_inject_preserves_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "bandenergy",
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.stft_encoder.energy_band_count() == 5


def test_build_time_stft_dual1d_sst_cached_native_inject():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_out_channels": 16, "stft_encoder_type": "sst_cached", "stft_sst_in_freq": 159,
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000), sst=torch.randn(2, 159, 37))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.sst_cached is True


def test_build_time_stft_dual1d_cached_sequence_native_inject():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_out_channels": 16, "stft_encoder_type": "cached_sequence", "stft_cached_in_freq": 8,
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000), sst=torch.randn(2, 8, 180))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.cached_context is True


def test_build_time_stft_dual1d_cached_tf_tcn_native_inject():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_out_channels": 16, "stft_encoder_type": "cached_tf_tcn", "stft_cached_in_freq": 36,
                "stft_cached_hidden_channels": 32, "stft_cached_pooled_freq": 6,
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000), sst=torch.randn(2, 36, 180))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.cached_context is True


def test_build_time_stft_dual1d_cached_sequence_res_tcn_native_inject():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_out_channels": 16, "stft_encoder_type": "cached_sequence_res_tcn",
                "stft_cached_in_freq": 8, "stft_cached_hidden_channels": 32,
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000), sst=torch.randn(2, 8, 180))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.cached_context is True


def test_build_time_stft_dual1d_bandgroup_native_inject_preserves_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "bandgroup",
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.stft_encoder.band_group_count() == 5
