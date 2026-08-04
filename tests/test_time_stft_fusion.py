from __future__ import annotations

import torch

from resp_train.models.time_stft_fusion import RespiratoryBandStftFeatures, TimeStftFusion1D
from resp_train.models.timeseries import PatchMixer1D


def _feature_frontend() -> RespiratoryBandStftFeatures:
    return RespiratoryBandStftFeatures(
        sample_rate=100,
        window_samples=3000,
        hop_samples=1000,
        low_hz=0.05,
        high_hz=0.70,
        eps=1e-8,
    )


def test_t1_stft_features_have_frozen_bins_frames_and_finite_zero_semantics() -> None:
    frontend = _feature_frontend()
    features = frontend(torch.zeros(2, 1, 18000))

    assert frontend.band_indices.tolist() == list(range(2, 22))
    assert frontend.band_bin_count == 20
    assert frontend.frame_count(18000) == 16
    assert features.shape == (2, 21, 16)
    assert torch.isfinite(features).all()
    assert torch.equal(features, torch.zeros_like(features))


def test_t1_stft_features_retain_frequency_shape_and_relative_effort() -> None:
    frontend = _feature_frontend()
    time = torch.arange(18000, dtype=torch.float32) / 100.0
    amplitude = torch.where(time < 90.0, 1.0, 2.0)
    signal = amplitude * torch.sin(2.0 * torch.pi * 0.2 * time)

    features = frontend(signal[None, None, :])
    # 0.2 Hz 对应整窗 rFFT k=6，在裁剪后的 k=2…21 中索引为 4。
    assert int(features[0, :20, 0].argmax()) == 4
    assert float(features[0, 20, -1]) > float(features[0, 20, 0])
    assert abs(float(features[0, 20].mean())) < 1e-5


def test_t1_zero_initialized_fusion_starts_from_identical_patchmixer_output() -> None:
    common = dict(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        patch_len=128,
        patch_stride=64,
        mixer_layers=1,
        overlap_window="hann",
        output_smoothing_kernel=1,
    )
    torch.manual_seed(17)
    baseline = PatchMixer1D(**common)
    torch.manual_seed(17)
    fusion = TimeStftFusion1D(
        **common,
        sample_rate=100,
        stft_window_samples=1000,
        stft_hop_samples=500,
        stft_low_hz=0.10,
        stft_high_hz=0.70,
        stft_feature_eps=1e-8,
        stft_channels=4,
    )
    x = torch.randn(2, 1, 4000)

    with torch.no_grad():
        expected = baseline(x)
        actual = fusion(x)

    assert torch.equal(actual, expected)


def test_t1_fusion_branch_receives_gradient_after_projection_warm_start() -> None:
    torch.manual_seed(23)
    model = TimeStftFusion1D(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        patch_len=128,
        patch_stride=64,
        mixer_layers=1,
        overlap_window="hann",
        sample_rate=100,
        stft_window_samples=1000,
        stft_hop_samples=500,
        stft_low_hz=0.10,
        stft_high_hz=0.70,
        stft_feature_eps=1e-8,
        stft_channels=4,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    x = torch.randn(2, 1, 4000)

    first_loss = model(x).square().mean()
    first_loss.backward()
    projection_grad = model.stft_projection.weight.grad
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert torch.count_nonzero(projection_grad) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = model(x).square().mean()
    second_loss.backward()
    encoder_grad = model.stft_encoder[0].weight.grad
    assert encoder_grad is not None
    assert torch.isfinite(encoder_grad).all()
    assert torch.count_nonzero(encoder_grad) > 0
