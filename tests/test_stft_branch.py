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


@pytest.mark.parametrize("encoder_type", ["conv1d", "conv2d", "freq_mlp", "soft_band"])
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


def test_stft_encoder_rejects_unknown_norm():
    with pytest.raises(ValueError, match="norm"):
        STFTEncoder(norm="unknown")


def test_stft_encoder_n1_divides_by_band_scale(tmp_path):
    import numpy as np

    enc = _encoder(3.0)
    n_bins = enc.band_bin_count()
    scale = np.full(n_bins, 2.0, dtype=np.float32)
    scale_path = tmp_path / "band_scale_3hz.npy"
    np.save(scale_path, scale)

    enc_n1 = STFTEncoder(
        sample_rate=100.0,
        stft_win=3000,
        stft_hop=500,
        low_hz=0.05,
        high_hz=3.0,
        out_channels=16,
        norm="n1",
        band_scale_path=str(scale_path),
    )

    assert torch.allclose(enc_n1.band_scale, torch.full((n_bins,), 2.0))


def test_stft_encoder_rejects_invalid_band_scale_values(tmp_path):
    import numpy as np

    enc = _encoder(3.0)
    scale = np.ones(enc.band_bin_count(), dtype=np.float32)
    scale[0] = float("nan")
    scale_path = tmp_path / "bad_band_scale.npy"
    np.save(scale_path, scale)

    with pytest.raises(ValueError, match="band_scale"):
        STFTEncoder(
            sample_rate=100.0,
            stft_win=3000,
            stft_hop=500,
            low_hz=0.05,
            high_hz=3.0,
            out_channels=16,
            norm="n1",
            band_scale_path=str(scale_path),
        )


def test_stft_encoder_n1_forward_divides_by_band_scale(tmp_path):
    import numpy as np

    n0 = STFTEncoder(
        sample_rate=100.0,
        stft_win=512,
        stft_hop=128,
        low_hz=0.5,
        high_hz=3.0,
        out_channels=4,
        norm="n0",
        encoder_type="conv1d",
    )
    scale = np.full(n0.band_bin_count(), 2.0, dtype=np.float32)
    scale_path = tmp_path / "band_scale.npy"
    np.save(scale_path, scale)
    n1 = STFTEncoder(
        sample_rate=100.0,
        stft_win=512,
        stft_hop=128,
        low_hz=0.5,
        high_hz=3.0,
        out_channels=4,
        norm="n1",
        band_scale_path=str(scale_path),
        encoder_type="conv1d",
    )
    n0.encoder = torch.nn.Identity()
    n1.encoder = torch.nn.Identity()
    x = torch.randn(1, 1, 4096)

    assert torch.allclose(n1(x), n0(x) / 2.0, atol=1e-6, rtol=1e-5)


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


def test_fusion_head_lite_decoder_uses_only_pointwise_convs():
    head = FusionHead(in_channels=24, out_length=18000, hidden=16, decoder_style="lite")
    convs = [module for module in head.decoder if isinstance(module, torch.nn.Conv1d)]

    assert head.decoder_style == "lite"
    assert len(convs) == 2
    assert [conv.kernel_size for conv in convs] == [(1,), (1,)]
    assert not any(isinstance(module, torch.nn.GroupNorm) for module in head.decoder)


def test_fusion_head_k3_no_norm_keeps_deep_kernels_without_group_norm():
    head = FusionHead(in_channels=24, out_length=18000, hidden=16, decoder_style="k3_no_norm")
    convs = [module for module in head.decoder if isinstance(module, torch.nn.Conv1d)]

    assert head.decoder_style == "k3_no_norm"
    assert [conv.kernel_size for conv in convs] == [(3,), (3,), (1,)]
    assert not any(isinstance(module, torch.nn.GroupNorm) for module in head.decoder)


def test_fusion_head_k1_norm_adds_group_norm_to_pointwise_decoder():
    head = FusionHead(in_channels=24, out_length=18000, hidden=16, decoder_style="k1_norm")
    convs = [module for module in head.decoder if isinstance(module, torch.nn.Conv1d)]

    assert head.decoder_style == "k1_norm"
    assert [conv.kernel_size for conv in convs] == [(1,), (1,)]
    assert any(isinstance(module, torch.nn.GroupNorm) for module in head.decoder)


def test_fusion_head_rejects_unknown_decoder_style():
    with pytest.raises(ValueError, match="decoder_style"):
        FusionHead(in_channels=24, out_length=18000, hidden=16, decoder_style="wide")


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


def test_dual_passes_lite_decoder_style_to_fusion_head():
    model = TimeStftDual1D(
        time_backbone_name="patch_mixer1d",
        time_backbone_kwargs=dict(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64),
        time_feat_channels=8,
        branch_mode="time_only",
        out_length=18000,
        fuse_len=600,
        stft_kwargs={},
        fusion_decoder="lite",
    )

    assert model.fusion_head.decoder_style == "lite"


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


def _native_dual(
    branch_mode: str,
    backbone: str = "patch_mixer1d",
    fusion_mode: str = "native_inject",
) -> TimeStftDual1D:
    kwargs = (
        dict(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64, overlap_window="hann")
        if backbone == "patch_mixer1d"
        else dict(in_channels=1, out_channels=1, base_channels=8, downsample_factors=[1, 4, 16])
    )
    feat_ch = 8 if backbone == "patch_mixer1d" else 8 * 3
    return TimeStftDual1D(
        time_backbone_name=backbone,
        time_backbone_kwargs=kwargs,
        time_feat_channels=feat_ch,
        branch_mode=branch_mode,
        out_length=18000,
        fuse_len=600,
        stft_kwargs=dict(
            sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=8.0,
            out_channels=16, norm="n0", encoder_type="conv2d",
        ),
        fusion_mode=fusion_mode,
    )


def test_native_inject_time_only_equals_native_backbone_forward():
    model = _native_dual("time_only")
    model.eval()
    x = torch.randn(2, 1, 18000)

    with torch.no_grad():
        wrapped = model(x)
        native = model.time_backbone(x)  # 原生非特征前向

    assert torch.equal(wrapped, native)


def test_native_inject_dual_equals_time_only_at_init_due_to_zero_proj():
    model = _native_dual("dual")
    model.eval()
    x = torch.randn(2, 1, 18000)

    with torch.no_grad():
        dual_out = model(x)
        tokens, length = model.time_backbone(x, return_features=True)
        native = model.time_backbone.decode_from_features(tokens, length)

    assert torch.allclose(dual_out, native, atol=1e-6)


@pytest.mark.parametrize("position", ["pre_mixer", "mid_mixer", "post_mixer"])
def test_native_inject_position_dual_preserves_native_output_at_init(position):
    model = _native_dual("dual")
    model.stft_inject_position = position
    model.eval()
    x = torch.randn(2, 1, 18000)

    with torch.no_grad():
        dual_out = model(x)
        tokens, length = model.time_backbone(x, return_features=True)
        native = model.time_backbone.decode_from_features(tokens, length)

    assert torch.allclose(dual_out, native, atol=1e-6)


@pytest.mark.parametrize("position", ["pre_mixer", "mid_mixer", "post_mixer"])
def test_native_inject_position_proj_gets_first_step_gradient(position):
    model = _native_dual("dual")
    model.stft_inject_position = position
    x = torch.randn(2, 1, 18000)

    model(x).square().mean().backward()

    proj_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_proj.parameters())
    assert proj_grad


def test_native_inject_rejects_unknown_position():
    model = _native_dual("dual")
    model.stft_inject_position = "unknown"
    with pytest.raises(ValueError, match="stft_inject_position"):
        model(torch.randn(2, 1, 18000))


def test_native_inject_dual_connects_both_branches_to_graph():
    """两分支都接进计算图。

    注意：stft_proj 零初始化（标准 ReZero）下，首步 `∂out/∂stft_encoder = decode'·proj.weight = 0`，
    故 stft_encoder 首步梯度本就为 0（正确行为，非 bug）。此处验证真正该成立的事：
    time_backbone 与 stft_proj 首步即有梯度——即 STFT 分支已接入图、proj.weight 会离开零点，
    使 stft_encoder 从第二步起获得梯度。
    """
    model = _native_dual("dual")
    x = torch.randn(2, 1, 18000)

    model(x).square().mean().backward()

    time_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.time_backbone.parameters())
    proj_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_proj.parameters())
    assert time_grad and proj_grad


def test_native_inject_stft_encoder_learns_after_one_optimizer_step():
    """一步优化让 stft_proj 离开零点后，stft_encoder 即获非零梯度（链路打通）。"""
    model = _native_dual("dual")
    x = torch.randn(2, 1, 18000)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # 第一步：proj.weight 离开零点
    opt.zero_grad()
    model(x).square().mean().backward()
    opt.step()

    # 第二步：此时 proj.weight 非零，stft_encoder 应获得真实梯度
    opt.zero_grad()
    model(x).square().mean().backward()
    stft_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_encoder.parameters())
    assert stft_grad


def test_token_context_inject_dual_preserves_native_output_at_init():
    model = _native_dual("dual", fusion_mode="token_context_inject")
    model.eval()
    x = torch.randn(2, 1, 18000)

    with torch.no_grad():
        dual_out = model(x)
        tokens, length = model.time_backbone(x, return_features=True)
        native = model.time_backbone.decode_from_features(tokens, length)

    assert torch.allclose(dual_out, native, atol=1e-6)
    assert model.stft_adapter is not None
    assert model.stft_proj is None


def test_token_context_inject_context_adapter_gets_first_step_gradient():
    model = _native_dual("dual", fusion_mode="token_context_inject")
    x = torch.randn(2, 1, 18000)

    model(x).square().mean().backward()

    adapter_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_adapter.parameters())
    time_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.time_backbone.parameters())
    assert adapter_grad and time_grad


def test_gated_native_inject_dual_preserves_native_output_at_init():
    model = _native_dual("dual", fusion_mode="gated_native_inject")
    model.eval()
    x = torch.randn(2, 1, 18000)

    with torch.no_grad():
        dual_out = model(x)
        tokens, length = model.time_backbone(x, return_features=True)
        native = model.time_backbone.decode_from_features(tokens, length)

    assert torch.allclose(dual_out, native, atol=1e-6)
    assert model.stft_gate is not None
    assert model.stft_proj is not None


def test_gated_native_inject_gate_learns_after_projection_moves():
    model = _native_dual("dual", fusion_mode="gated_native_inject")
    x = torch.randn(2, 1, 18000)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    opt.zero_grad()
    model(x).square().mean().backward()
    opt.step()

    opt.zero_grad()
    model(x).square().mean().backward()
    gate_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_gate.parameters())
    assert gate_grad


def test_cross_attention_inject_dual_preserves_native_output_at_init():
    model = _native_dual("dual", fusion_mode="cross_attention_inject")
    model.eval()
    x = torch.randn(2, 1, 18000)

    with torch.no_grad():
        dual_out = model(x)
        tokens, length = model.time_backbone(x, return_features=True)
        native = model.time_backbone.decode_from_features(tokens, length)

    assert torch.allclose(dual_out, native, atol=1e-6)
    assert model.cross_attention_adapter is not None
    assert model.stft_proj is None


def test_cross_attention_inject_attention_learns_after_projection_moves():
    model = _native_dual("dual", fusion_mode="cross_attention_inject")
    x = torch.randn(2, 1, 18000)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    opt.zero_grad()
    model(x).square().mean().backward()
    opt.step()

    opt.zero_grad()
    model(x).square().mean().backward()
    attn_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.cross_attention_adapter.named_parameters()
        if not name.startswith("out_proj")
    )
    assert attn_grad


def test_native_inject_rejects_backbone_without_decode_from_features():
    with pytest.raises((ValueError, TypeError, AttributeError)):
        _native_dual("dual", backbone="multiscale_decomp_mixer1d")


def _band_energy_encoder(out_channels: int = 16, energy_bands=None) -> STFTEncoder:
    return STFTEncoder(
        sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=8.0,
        out_channels=out_channels, norm="n0", encoder_type="bandenergy",
        energy_bands=energy_bands,
    )


def test_bandenergy_output_is_3d_time_series_with_out_channels():
    enc = _band_energy_encoder(out_channels=16)
    feats = enc(torch.randn(2, 1, 18000))
    assert feats.dim() == 3
    assert feats.shape[0] == 2
    assert feats.shape[1] == 16  # out_channels，与 conv1d/conv2d 契约一致


def test_bandenergy_default_uses_five_bands():
    enc = _band_energy_encoder()
    assert enc.energy_band_count() == 5


def test_bandenergy_custom_band_count():
    enc = _band_energy_encoder(energy_bands=[(0.05, 0.7), (0.7, 8.0)])
    assert enc.energy_band_count() == 2
    feats = enc(torch.randn(1, 1, 18000))
    assert feats.shape[1] == 16


def test_bandenergy_rejects_band_outside_range():
    with pytest.raises(ValueError):
        _band_energy_encoder(energy_bands=[(0.05, 0.3), (3.0, 50.0)])  # 50Hz 越界


def test_bandenergy_handles_zero_input_without_crash():
    enc = _band_energy_encoder()
    out = enc(torch.zeros(1, 1, 18000))
    assert torch.isfinite(out).all()


def _band_group_encoder(out_channels: int = 16, band_group_f: int = 32, energy_bands=None) -> STFTEncoder:
    return STFTEncoder(
        sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=8.0,
        out_channels=out_channels, norm="n0", encoder_type="bandgroup",
        energy_bands=energy_bands, band_group_f=band_group_f,
    )


def test_bandgroup_output_is_3d_with_out_channels():
    enc = _band_group_encoder(out_channels=16)
    feats = enc(torch.randn(2, 1, 18000))
    assert feats.dim() == 3
    assert feats.shape[0] == 2
    assert feats.shape[1] == 16  # = out_channels，与其它 encoder 契约一致


def test_bandgroup_default_uses_five_bands():
    enc = _band_group_encoder()
    assert enc.band_group_count() == 5


def test_bandgroup_custom_f_and_bands():
    enc = _band_group_encoder(band_group_f=16, energy_bands=[(0.05, 0.7), (0.7, 8.0)])
    assert enc.band_group_count() == 2
    feats = enc(torch.randn(1, 1, 18000))
    assert feats.shape[1] == 16


def test_bandgroup_rejects_band_outside_range():
    with pytest.raises(ValueError):
        _band_group_encoder(energy_bands=[(0.05, 0.3), (3.0, 50.0)])


def test_bandgroup_handles_zero_input_without_crash():
    enc = _band_group_encoder()
    out = enc(torch.zeros(1, 1, 18000))
    assert torch.isfinite(out).all()


def test_freq_mlp_mixes_frequency_bins_before_time_encoding():
    enc = STFTEncoder(
        sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=8.0,
        out_channels=16, norm="n0", encoder_type="freq_mlp",
    )

    assert enc.encoder_type == "freq_mlp"
    assert enc.freq_mlp_bin_count() == enc.band_bin_count()
    assert enc.freq_mlp is not None
    assert enc.freq_mlp[0].in_features == enc.band_bin_count()


def test_soft_band_defaults_to_five_learnable_bands_initialized_from_overlaps():
    enc = STFTEncoder(
        sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=8.0,
        out_channels=16, norm="n0", encoder_type="soft_band",
    )

    weights = enc.soft_band_weights()
    assert enc.soft_band_count() == 5
    assert weights.shape == (5, enc.band_bin_count())
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-5)
    assert torch.isfinite(weights).all()


def _native_dual_sst(branch_mode: str, in_freq: int = 159) -> TimeStftDual1D:
    return TimeStftDual1D(
        time_backbone_name="patch_mixer1d",
        time_backbone_kwargs=dict(
            in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64, overlap_window="hann"
        ),
        time_feat_channels=8,
        branch_mode=branch_mode,
        out_length=18000,
        fuse_len=600,
        stft_kwargs=dict(encoder_type="sst_cached", in_freq=in_freq, out_channels=16),
        fusion_mode="native_inject",
    )


def test_sst_cached_encoder_output_contract():
    from resp_train.models.stft_branch import SSTCachedEncoder

    enc = SSTCachedEncoder(in_freq=159, out_channels=16)
    feats = enc(torch.randn(2, 159, 37))
    assert feats.shape == (2, 16, 37)  # (B,out_ch,T')，与 conv2d 契约一致


def test_native_inject_sst_time_only_equals_native():
    model = _native_dual_sst("time_only")
    model.eval()
    x = torch.randn(2, 1, 18000)
    with torch.no_grad():
        assert torch.equal(model(x), model.time_backbone(x))


def test_native_inject_sst_dual_uses_cached_sst():
    model = _native_dual_sst("dual")
    x = torch.randn(2, 1, 18000)
    sst = torch.randn(2, 159, 37)
    out = model(x, sst=sst)
    assert out.shape == (2, 1, 18000)
    assert torch.isfinite(out).all()


def test_native_inject_sst_dual_requires_sst():
    model = _native_dual_sst("dual")
    with pytest.raises((ValueError, TypeError)):
        model(torch.randn(2, 1, 18000))  # 缺 sst 应报错


def test_sst_cached_rejects_concat_generic():
    with pytest.raises(ValueError, match="native_inject"):
        TimeStftDual1D(
            time_backbone_name="patch_mixer1d",
            time_backbone_kwargs=dict(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64),
            time_feat_channels=8,
            branch_mode="dual",
            out_length=18000,
            fuse_len=600,
            stft_kwargs=dict(encoder_type="sst_cached", in_freq=159, out_channels=16),
            fusion_mode="concat_generic",
        )
