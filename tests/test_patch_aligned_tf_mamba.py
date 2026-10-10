"""合成输入与显式 Mamba 替身；验证几何、条件路径和梯度。"""
from pathlib import Path
from dataclasses import replace
import hashlib
import json

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


@pytest.mark.parametrize("patch_samples", [100, 200, 400])
def test_overlap_reconstructs_input_and_gradients(patch_samples):
    signal = torch.randn(2, 18000, requires_grad=True)
    cfg = mod.PatchTFConfig(patch_samples=patch_samples)
    overlap = mod.PositiveOverlapAdd(cfg)
    output = overlap(signal.unfold(-1, patch_samples, cfg.patch_hop_samples))[:, 0]
    torch.testing.assert_close(output, signal)
    output.sum().backward()
    torch.testing.assert_close(signal.grad, torch.ones_like(signal))
    assert overlap.denominator.min() > 0


def test_continuous_stem_keeps_local_receptive_field():
    cfg = mod.PatchTFConfig(waveform_channels=16, dimension=32)
    encoder = mod.ContinuousPatchEncoder(cfg).eval()
    assert cfg.stem_receptive_field == 79
    x = torch.randn(1, 1, 18000)
    altered = x.clone()
    altered[..., 10000] += 10
    with torch.no_grad():
        before, after = encoder.stem(x), encoder.stem(altered)
    torch.testing.assert_close(before[..., :9961], after[..., :9961], rtol=0, atol=0)
    torch.testing.assert_close(before[..., 10040:], after[..., 10040:], rtol=0, atol=0)
    assert (before[..., 9961:9980] - after[..., 9961:9980]).abs().sum() > 0
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


def test_h_selection_and_builder_use_actual_metadata(monkeypatch, frozen_cache):
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
    cfg.data.tf_cache_path = str(frozen_cache)
    original = mod.build_patch_aligned_tf_mamba
    monkeypatch.setattr(mod, "build_patch_aligned_tf_mamba", lambda cfg: original(cfg, mamba_factory=SyntheticMamba))
    net = build_model(cfg)
    assert len(net.trunk) == 6 and net.config.dimension == 96
    np.testing.assert_array_equal(net.condition.frequencies_hz.numpy(), actual[56:])


@pytest.fixture
def frozen_cache(tmp_path, monkeypatch):
    """临时 cache 使用生产 reader；仅将冻结 manifest 身份指向 fixture。"""
    from resp_train.crd import tf_v1_data
    from resp_train.crd.tf_v1_features import fixed_transform_spec, PROTOCOL

    root = tmp_path / "cache"
    root.mkdir()
    ids = np.array([11, 29], dtype=np.int64)
    np.save(root / "train_row_ids.npy", ids)
    np.save(root / "train_w.npy", np.random.default_rng(3).random((2, 97, 360), dtype=np.float32))
    np.save(root / "w_frequencies_hz.npy", np.concatenate((np.linspace(.03, .8, 56), frequencies())))
    inventory = {}
    for path in root.iterdir():
        value = np.load(path)
        inventory[path.name] = {"size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(value.shape), "dtype": str(value.dtype), "finite": True}
    manifest = {"protocol": PROTOCOL, "fixed_transform_spec": fixed_transform_spec(),
        "transform_sha256": tf_v1_data.FROZEN_CACHE_TRANSFORM_SHA256,
        "complete": True, "research_test_used": False, "test_cache_created": False, "target_read": False,
        "files": inventory, "splits": {"train": {"split": "train", "count": 2,
            "row_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}}}
    path = root / "cache_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(tf_v1_data, "FROZEN_CACHE_MANIFEST_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return root


@pytest.mark.parametrize("patch_samples", [100, 200, 400])
def test_parameterized_geometry_and_model_output(patch_samples):
    cfg = replace(model().config, patch_samples=patch_samples)
    net = mod.PatchAlignedTFMamba(frequencies(), config=cfg, mamba_factory=SyntheticMamba).eval()
    bins = cfg.cwt_bins_per_patch
    assert cfg.patch_count == (18000 - patch_samples) // 50 + 1
    expected = (torch.arange(bins) * 50 + 24.5 - (patch_samples - 1) / 2) / 100
    torch.testing.assert_close(net.condition.relative_seconds, expected)
    with torch.no_grad():
        out = net(torch.randn(1, 1, 18000), tf={"w": torch.rand(1, 41, 360)})["waveform"]
    assert out.shape == (1, 1, 18000)


@pytest.mark.parametrize("options", [{"patch_samples": 175}, {"patch_hop_samples": 100},
    {"patch_samples": 18100}, {"cwt_pool_samples": 100}, {"patch_chunk_size": 0}])
def test_invalid_geometry_fails(options):
    with pytest.raises(ValueError):
        mod.PatchTFConfig(**options)


def test_cwt_encoder_responds_to_local_amplitude_and_boundary_padding():
    net = model().condition.eval()
    assert any(isinstance(m, nn.GroupNorm) for m in net.encoder.modules())
    w = torch.rand(1, 1, 41, 360)
    altered = w.clone()
    altered[..., 100:140] *= 1.5
    with torch.no_grad():
        baseline, changed = net.encoder(w), net.encoder(altered)
    relative = (changed[..., 100:140] - baseline[..., 100:140]).norm() / baseline[..., 100:140].norm()
    assert relative > .05
    grid = torch.arange(12).reshape(1, 1, 3, 4).float()
    padded = mod.ScaleTimePad()(grid)
    torch.testing.assert_close(padded[..., 1:-1, 0], grid[..., 1])
    torch.testing.assert_close(padded[..., 0, 1:-1], grid[..., 0, :])


def test_chunked_pooling_matches_positioned_reference_and_gradients():
    cfg = replace(model().config, patch_chunk_size=7)
    net = mod.ContinuousPatchEncoder(cfg)
    reference = mod.ContinuousPatchEncoder(cfg)
    reference.load_state_dict(net.state_dict())
    x = torch.randn(1, 1, 18000)
    actual = net(x)
    features = reference.stem(x)
    positioned = features.unfold(-1, 200, 50).permute(0, 2, 3, 1) + reference.position(reference.relative_seconds[:, None])
    weights = reference.score(positioned).softmax(2)
    expected = reference.project((weights * positioned).sum(2))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, p), (_, q) in zip(net.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=1e-6, msg=name)


def test_attention_checkpoint_gradients_and_saved_tensor_bound():
    cfg = replace(model().config, patch_chunk_size=7)
    a = mod.LocalTFAttention(cfg, frequencies())
    b = mod.LocalTFAttention(replace(cfg, checkpoint_local=False, patch_chunk_size=357), frequencies())
    b.load_state_dict(a.state_dict())
    z, w = torch.randn(1, 357, 32), torch.rand(1, 41, 360)
    saved = []
    def pack(t):
        saved.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        actual = a(z, w)
    expected = b(z, w)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=1e-6, msg=name)
    # checkpoint 只保存原生 map/切片，不能保存整窗重叠 K/V 或 attention 概率。
    assert not any(357 in shape and (164 in shape or 41 in shape) for shape in saved)
    assert all(len(shape) <= 5 for shape in saved)


def test_bfloat16_forward_backward():
    net = model()
    with torch.no_grad():
        net.film[-1].weight.normal_(std=.01)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(torch.randn(1, 1, 18000), tf={"w": torch.rand(1, 41, 360)})["waveform"]
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_frozen_adapter_dataset_engine_training_step(frozen_cache, tmp_path, monkeypatch):
    import pandas as pd
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from resp_train.data.research_v2 import ResearchV2WindowDataset
    from resp_train.models.registry import build_model
    from resp_train.engine.train import train_one_epoch
    from resp_train.crd import tf_v1_features

    def no_transform(*args, **kwargs):
        raise AssertionError("训练数据链不得计算 CWT")
    monkeypatch.setattr(tf_v1_features, "cwt_magnitude_features", no_transform)
    monkeypatch.setattr(tf_v1_features, "morlet_scales_and_frequencies", no_transform)
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/patch_aligned_tf_mamba/model.yaml")
    cfg.data.tf_cache_path = str(frozen_cache)
    cfg.model.patch_aligned_tf_mamba.dimension = 32
    cfg.model.patch_aligned_tf_mamba.waveform_channels = 16
    cfg.model.patch_aligned_tf_mamba.condition_channels = 8
    cfg.model.patch_aligned_tf_mamba.layers = 1
    cfg.window = {"duration_samples": 18000, "target_fs": 100}
    np.savez(tmp_path / "signals.npz", bcg=np.random.default_rng(7).normal(size=36000).astype(np.float32),
             tho=np.sin(np.arange(36000) / 50).astype(np.float32))
    rows = pd.DataFrame([{"dataset_row_id": row_id, "split": "train", "samp_id": 1,
        "coupling_state_id": i, "window_start_sample": i * 18000, "window_end_sample": (i + 1) * 18000,
        "bcg_signal_key": "bcg", "target_signal_key": "tho", "source_npz": "signals.npz",
        "target_source_npz": "signals.npz", "allowed_losses": "waveform"} for i, row_id in enumerate([11, 29])])
    dataset = ResearchV2WindowDataset(tmp_path / "index.csv", rows, cfg)
    np.testing.assert_array_equal(dataset[0]["tf"]["w"].numpy(), np.load(frozen_cache / "train_w.npy")[0, 56:])
    original = mod.build_patch_aligned_tf_mamba
    monkeypatch.setattr(mod, "build_patch_aligned_tf_mamba", lambda cfg: original(cfg, mamba_factory=SyntheticMamba))
    net = build_model(cfg)
    class FixtureLoss(nn.Module):
        def forward(self, pred, target):
            loss = (pred["waveform"] - target).square().mean()
            return loss, {"mse": loss.detach()}
    stats = train_one_epoch(net, DataLoader(dataset, batch_size=1), FixtureLoss(),
        torch.optim.AdamW(net.parameters(), lr=1e-3), "cpu", show_progress=False)
    assert np.isfinite(stats["loss"])
    assert net.condition.value.weight.grad.abs().sum() > 0
    bad_rows = rows.copy()
    bad_rows["split"] = "test"
    with pytest.raises(ValueError, match="train/val"):
        ResearchV2WindowDataset(tmp_path / "index.csv", bad_rows, cfg)


def test_frozen_cache_tampering_and_missing_rows_fail(frozen_cache):
    from resp_train.data.patch_aligned_tf import FrozenHCWTReader, frozen_h_metadata
    reader = FrozenHCWTReader(frozen_cache, split="train")
    with pytest.raises(KeyError):
        reader.get(999)
    values = np.load(frozen_cache / "train_w.npy", mmap_mode="r+")
    values[0, 0, 0] = np.nan
    values.flush()
    with pytest.raises(FloatingPointError):
        reader.get(11)
    frequencies_path = frozen_cache / "w_frequencies_hz.npy"
    values = np.load(frequencies_path)
    values[-1] -= .001
    np.save(frequencies_path, values)
    with pytest.raises(RuntimeError, match="频率"):
        frozen_h_metadata(frozen_cache)
