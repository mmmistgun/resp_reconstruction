"""合成输入与显式 Mamba 替身；验证几何、条件路径和梯度。"""
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from resp_train.models import patch_aligned_tf_mamba as mod


class SyntheticMamba(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.projection(x)


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def frequencies():
    # 此网格只用于接口测试；实际模型使用原生映射。
    return np.geomspace(.81, 8, 41)


def model():
    return mod.PatchAlignedTFMamba(frequencies(),
        config=mod.PatchTFConfig(dimension=32, waveform_channels=16, condition_channels=8, layers=1),
        mamba_factory=SyntheticMamba)


def test_local_readout_matches_physical_intervals():
    grid = torch.arange(41 * 360).reshape(1, 1, 41, 360)
    cells = mod.LocalTFAttention.local_cells(grid)
    assert cells.shape == (1, 357, 41, 4, 1)
    for j in (0, 17, 356):
        torch.testing.assert_close(cells[0, j, :, :, 0], grid[0, 0, :, j:j + 4])
    bins = (torch.arange(360, dtype=torch.float64) * 50 + 24.5) / 100
    centers = (torch.arange(357, dtype=torch.float64) * 50 + 99.5) / 100
    torch.testing.assert_close(bins.unfold(0, 4, 1) - centers[:, None],
        torch.tensor([-.75, -.25, .25, .75], dtype=torch.float64).expand(357, 4))


def test_overlap_reconstructs_input_and_gradients():
    signal = torch.randn(2, 18000, requires_grad=True)
    overlap = mod.PositiveOverlapAdd(1e-3)
    output = overlap(signal.unfold(-1, 200, 50))[:, 0]
    torch.testing.assert_close(output, signal)
    output.sum().backward()
    torch.testing.assert_close(signal.grad, torch.ones_like(signal))
    assert overlap.denominator.min() > 0


def test_continuous_stem_keeps_local_receptive_field():
    encoder = mod.ContinuousPatchEncoder(16, 32).eval()
    x = torch.randn(1, 1, 18000)
    altered = x.clone()
    altered[..., 10000] += 10
    with torch.no_grad():
        before, after = encoder.stem(x), encoder.stem(altered)
    torch.testing.assert_close(before[..., :9900], after[..., :9900], rtol=0, atol=0)
    assert encoder(x).shape == (1, 357, 32)
    assert encoder.relative_seconds[0].item() == pytest.approx(-.995)
    assert encoder.relative_seconds[-1].item() == pytest.approx(.995)


def test_zero_init_and_condition_gradients():
    net = model().eval()
    x = torch.randn(1, 1, 18000)
    w = torch.randn(1, 41, 360, requires_grad=True)
    with torch.no_grad():
        expected = net.overlap(net.decoder(net.trunk(net.waveform_encoder(x))))
        torch.testing.assert_close(net(x, tf={"w": w})["waveform"], expected, rtol=0, atol=0)
    net(x, tf={"w": w})["waveform"].square().mean().backward()
    assert net.film[-1].weight.grad.abs().sum() > 0
    net.zero_grad(set_to_none=True)
    w.grad = None
    # 零初始化投影打开后，条件上游应获得有效梯度。
    with torch.no_grad():
        net.film[-1].weight.normal_(std=.02)
    out = net(x, tf={"w": w})["waveform"]
    assert out.shape == (1, 1, 18000) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0
    for name, param in net.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    changed = net(x, tf={"w": w.detach().roll(12, -1)})["waveform"]
    assert (out.detach() - changed).abs().max() > 0


def test_attention_agrees_with_explicit_local_reference():
    attention = model().condition.eval()
    z, w = torch.randn(1, 357, 32), torch.randn(1, 41, 360)
    with torch.no_grad():
        actual = attention(z, w)
        features = attention.encoder(w[:, None])
        for j in (0, 100, 356):
            cells = features[0, :, :, j:j + 4].permute(1, 2, 0).reshape(164, 8)
            k = attention.key(cells).reshape(164, 4, 8)
            v = attention.value(cells).reshape(164, 4, 8)
            q = attention.query(z[0, j]).reshape(4, 8)
            fb = attention.frequency_bias(attention.frequencies_hz.log().float()[:, None])
            tb = attention.time_bias(attention.relative_seconds[:, None])
            bias = (fb[:, None] + tb[None]).reshape(164, 4).T
            weights = (torch.einsum("hd,khd->hk", q, k) / 8 ** .5 + bias).softmax(-1)
            expected = attention.output(torch.einsum("hk,khd->hd", weights, v).flatten())
            torch.testing.assert_close(actual[0, j], expected)


@pytest.mark.parametrize("bad", ["shape", "nan_x", "inf_w", "inf_film"])
def test_invalid_inputs_fail(bad):
    net = model().eval()
    x, w = torch.zeros(1, 1, 18000), torch.zeros(1, 41, 360)
    if bad == "shape":
        w = w[:, :, :-1]
    elif bad == "nan_x":
        x[..., 0] = float("nan")
    elif bad == "inf_w":
        w[..., 0] = float("inf")
    else:
        with torch.no_grad():
            net.film[-1].bias.fill_(float("inf"))
    with pytest.raises((ValueError, FloatingPointError)):
        net(x, tf={"w": w})


def test_frequency_validation_and_checkpoint_roundtrip():
    with pytest.raises(ValueError):
        mod.PatchAlignedTFMamba(frequencies()[::-1].copy(), mamba_factory=SyntheticMamba)
    a, b = model(), model()
    b.load_state_dict(a.state_dict())
    assert torch.equal(b.condition.frequencies_hz, a.condition.frequencies_hz)
    x, w = torch.randn(1, 1, 18000), torch.randn(1, 41, 360)
    with torch.no_grad():
        torch.testing.assert_close(a.eval()(x, tf={"w": w})["waveform"],
                                   b.eval()(x, tf={"w": w})["waveform"], rtol=0, atol=0)


def test_h_selection_and_builder_use_actual_metadata(monkeypatch):
    from resp_train.crd import tf_v1_features
    from resp_train.models.registry import build_model
    from omegaconf import OmegaConf

    actual = np.concatenate((np.linspace(.03, .8, 56), frequencies()))
    features = np.arange(97 * 360, dtype=np.float32).reshape(97, 360)
    monkeypatch.setattr(tf_v1_features, "morlet_scales_and_frequencies", lambda *a: (np.arange(97), actual))
    monkeypatch.setattr(tf_v1_features, "cwt_magnitude_features", lambda x: (features, actual))
    selected, f = mod.h_cwt_features(np.zeros(18000))
    np.testing.assert_array_equal(selected, features[56:])
    np.testing.assert_array_equal(f, actual[56:])
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/patch_aligned_tf_mamba/model.yaml")
    original = mod.build_patch_aligned_tf_mamba
    monkeypatch.setattr(mod, "build_patch_aligned_tf_mamba", lambda cfg: original(cfg, mamba_factory=SyntheticMamba))
    net = build_model(cfg)
    assert len(net.trunk) == 6 and net.config.dimension == 96
    np.testing.assert_array_equal(net.condition.frequencies_hz.numpy(), actual[56:])
