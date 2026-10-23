"""人工张量验证 BCG-CWT 几何、算子、接口及序列化。"""
from copy import deepcopy
import io
import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from resp_train.models import bcg_cwt_patch_mamba as mod
from scripts.check_bcg_cwt_patch_mamba import SyntheticMamba, synthetic_config, smoke


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model():
    return mod.build_bcg_cwt_patch_mamba(synthetic_config(), mamba_factory=SyntheticMamba)


def test_default_yaml_and_geometry():
    cfg = synthetic_config()
    assert OmegaConf.to_container(cfg.model.architecture) == mod.DEFAULT_ARCHITECTURE
    config = mod.BCGCWTConfig(cfg.model.architecture)
    assert (config.patch_count, config.bins_per_patch, config.step_tf) == (180, 2, 2)
    align = config.alignment()
    assert align["sample_starts"][[0, -1]].tolist() == [0, 17900]
    assert align["cwt_columns"][[0, -1]].tolist() == [[0, 1], [358, 359]]
    torch.testing.assert_close(align["relative_seconds"], torch.tensor([-.25, .25]))
    assert align["patch_centers_seconds"][0] == pytest.approx(.495)
    torch.testing.assert_close(align["cwt_centers_seconds"][0], torch.tensor([.245, .745]))
    grid = torch.arange(97 * 360).reshape(1, 1, 97, 360)
    cells = mod.local_cells(grid).flatten(2, 3)
    assert cells.shape == (1, 180, 194, 1)
    for j in range(180):
        for f in (0, 48, 96):
            for r in range(2):
                assert cells[0, j, f * 2 + r, 0] == grid[0, 0, f, 2 * j + r]
    numbered = torch.arange(18000).reshape(1, 180, 100)
    assert torch.equal(mod.concatenate_patches(numbered), torch.arange(18000).reshape(1, 1, -1).float())


def test_conv_linear_equivalence_forward_and_all_gradients():
    projection = model().patch_embedding.double()
    x = torch.randn(1, 64, 300, dtype=torch.float64, requires_grad=True)
    weight = projection.weight.detach().clone().reshape(96, -1).requires_grad_()
    bias = projection.bias.detach().clone().requires_grad_()
    other_x = x.detach().clone().requires_grad_()
    actual = projection(x).transpose(1, 2)
    patches = other_x.unfold(-1, 100, 100).permute(0, 2, 1, 3).flatten(2)
    expected = F.linear(patches, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for a, b in ((x.grad, other_x.grad), (projection.weight.grad.flatten(1), weight.grad), (projection.bias.grad, bias.grad)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    assert sum(p.numel() for p in projection.parameters()) == 614496


def test_module_tree_position_sharing_and_continuous_shapes():
    net = model().eval()
    assert len(net.temporal) == 6
    directions = [getattr(block, direction) for block in net.temporal for direction in ("forward_mamba", "backward_mamba")]
    assert len({id(p) for m in directions for p in m.parameters()}) == 24
    assert net.fusion.frequency_embedding.shape == (97, 32)
    assert net.fusion.local_time_embedding.shape == (2, 32)
    assert (net.fusion.mha.embed_dim, net.fusion.mha.kdim, net.fusion.mha.vdim, net.fusion.mha.num_heads) == (96, 32, 32, 4)
    assert net.fusion.ffn[0].out_features == 384
    assert [m.out_features for m in net.patch_decoder if isinstance(m, nn.Linear)] == [96, 100]
    with torch.no_grad():
        assert net.waveform_encoder(torch.randn(1, 1, 18000)).shape == (1, 64, 18000)
        assert net.tf_encoder(torch.randn(1, 1, 97, 360)).shape == (1, 32, 97, 360)
    grid = torch.zeros(2, 32, 97, 6)
    positioned = net.fusion.positioned_cells(grid)
    expected = net.fusion.frequency_embedding[:, None] + net.fusion.local_time_embedding[None]
    torch.testing.assert_close(positioned, expected[None, None].expand(2, 3, 97, 2, 32))


@pytest.mark.parametrize("dtype,rtol,atol", [(torch.float64, 1e-8, 1e-9), (torch.float32, 2e-4, 2e-5)])
def test_chunked_attention_output_input_and_parameter_gradients(dtype, rtol, atol):
    small = mod.AlignedCrossAttention(mod.BCGCWTConfig()).to(dtype=dtype).eval()
    full = deepcopy(small)
    full.chunk_size = 180
    z = torch.randn(2, 19, 96, dtype=dtype, requires_grad=True)
    grid = torch.randn(2, 32, 97, 38, dtype=dtype, requires_grad=True)
    z2, grid2 = z.detach().clone().requires_grad_(), grid.detach().clone().requires_grad_()
    a, _ = small(z, grid)
    b, _ = full(z2, grid2)
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    a.square().sum().backward()
    b.square().sum().backward()
    for x, y in [(z.grad, z2.grad), (grid.grad, grid2.grad)] + [(p.grad, q.grad) for p, q in zip(small.parameters(), full.parameters())]:
        torch.testing.assert_close(x, y, rtol=rtol, atol=atol)
    with torch.no_grad():
        debug, weights = small(z, grid, return_attention=True)
    torch.testing.assert_close(a, debug, rtol=rtol, atol=atol)
    assert weights.shape == (2, 19, 4, 97, 2)
    torch.testing.assert_close(weights.flatten(-2).sum(-1), torch.ones(2, 19, 4, dtype=dtype))


def test_all_twelve_mamba_constructors_receive_specification():
    calls = []
    def recording_factory(**kwargs):
        calls.append(kwargs)
        return SyntheticMamba(**kwargs)
    net = mod.build_bcg_cwt_patch_mamba(synthetic_config(), mamba_factory=recording_factory)
    expected = {"d_model": 96, **mod.DEFAULT_ARCHITECTURE["temporal"]["mamba2"]}
    assert calls == [expected] * 12
    assert all(block.norm.eps == 1e-5 and block.dropout.p == .1 for block in net.temporal)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_default_length_backward_task_loss_and_optimizer(batch_size):
    report = smoke(batch_size=batch_size, synthetic_mamba=True)
    assert report["status"] == "passed"
    assert report["output_shape"] == [batch_size, 1, 18000]
    assert report["loss_parts"]["__count_loss_sync"] == batch_size
    assert report["loss_parts"]["__count_loss_effort"] == batch_size


@pytest.mark.parametrize("bad", ["shape_x", "shape_w", "batch", "int", "dtype", "device", "nan_x", "inf_w", "output", "missing"])
def test_input_and_output_failures(bad):
    net = model().eval()
    x, w = torch.zeros(1, 1, 18000), torch.zeros(1, 97, 360)
    if bad == "shape_x": x = x[..., :-1]
    elif bad == "shape_w": w = w[:, :-1]
    elif bad == "batch": w = w.repeat(2, 1, 1)
    elif bad == "int": x = x.long()
    elif bad == "dtype": w = w.double()
    elif bad == "device": w = w.to("meta")
    elif bad == "nan_x": x[..., 0] = float("nan")
    elif bad == "inf_w": w[..., 0] = float("inf")
    elif bad == "output":
        with torch.no_grad(): net.patch_decoder[-1].bias.fill_(float("inf"))
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        net(x, tf={} if bad == "missing" else {"w": w})


@pytest.mark.parametrize("patch", [
    {"patch_hop_samples": 50}, {"cwt_frequency_count": 41}, {"dimension": 95},
    {"fusion": {"key_input_dimension": 64}}, {"fusion": {"patch_chunk_size": 0}},
    {"tf_encoder": {"norm_groups": 3}}, {"decoder": {"activation": "silu"}},
    {"waveform_encoder": {"norm_eps": .01}}, {"temporal": {"mamba2": {"ngroups": 4}}},
    {"typo": 1}, {"dimension": True}, {"fusion": {"residual_dropout": float("nan")}},
])
def test_config_rejects_incompatible_fields(patch):
    with pytest.raises((ValueError, TypeError)):
        mod.BCGCWTConfig(patch)


@pytest.mark.parametrize("bad", ["shape", "order", "zero", "nan"])
def test_invalid_frequency_metadata(bad):
    freq = torch.linspace(.03, 8, 97, dtype=torch.float64)
    if bad == "shape": freq = freq[:-1]
    elif bad == "order": freq = freq.flip(0)
    elif bad == "zero": freq[0] = 0
    else: freq[0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        mod.BCGCWTPatchMamba(freq, mamba_factory=SyntheticMamba)


def test_serialization_identity_checks_before_buffer_or_parameter_copy():
    a, b = model().eval(), model().eval()
    with torch.no_grad(): a.patch_decoder[-1].bias.add_(.1)
    buffer = io.BytesIO()
    torch.save(a.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    b.load_state_dict(state)
    x, w = torch.randn(1, 1, 18000), torch.rand(1, 97, 360)
    with torch.no_grad():
        torch.testing.assert_close(a(x, tf={"w": w})["waveform"], b(x, tf={"w": w})["waveform"], rtol=0, atol=0)
    assert state["_extra_state"]["initialization_seed"] == 20261013
    for bad in ("frequency", "geometry", "relative", "nonfinite"):
        corrupted = deepcopy(state)
        if bad == "frequency": corrupted["frequencies_hz"][0] += .0001
        elif bad == "geometry": corrupted["_extra_state"]["architecture"]["cwt_pool_samples"] = 100
        elif bad == "relative": corrupted["relative_seconds"][0] += 1
        else: corrupted["patch_decoder.2.bias"][0] = float("nan")
        before = deepcopy(b.state_dict())
        with pytest.raises((ValueError, FloatingPointError)):
            b.load_state_dict(corrupted, strict=False)
        for key, value in b.state_dict().items():
            if isinstance(value, torch.Tensor): assert torch.equal(before[key], value), key


def test_factory_and_constructor_seed(monkeypatch):
    from resp_train.models.registry import build_model, list_models
    original = mod.build_bcg_cwt_patch_mamba
    monkeypatch.setattr(mod, "build_bcg_cwt_patch_mamba", lambda cfg: original(cfg, mamba_factory=SyntheticMamba))
    assert mod.MODEL_NAME in list_models()
    before = torch.random.get_rng_state().clone()
    a = build_model(synthetic_config())
    assert torch.equal(before, torch.random.get_rng_state())
    b = build_model(synthetic_config())
    for p, q in zip(a.parameters(), b.parameters()): assert torch.equal(p, q)
    cfg = synthetic_config()
    cfg.window.duration_samples = 100
    with pytest.raises(ValueError, match="时间几何"):
        build_model(cfg)


@pytest.mark.parametrize("precision,batch", [("float32", 1), ("bfloat16", 2)])
def test_official_mamba2(precision, batch):
    if os.environ.get("BCG_CWT_RUN_CUDA") != "1":
        pytest.skip("官方 CUDA 合成检查需显式 BCG_CWT_RUN_CUDA=1")
    report = smoke(batch_size=batch, device="cuda:0", precision=precision)
    if report["status"] == "skipped": pytest.skip(report["reason"])
    assert report["official_mamba"] and report["status"] == "passed"


def test_official_checkpoint_roundtrip_and_optimizer_markers(tmp_path):
    if os.environ.get("BCG_CWT_RUN_CUDA") != "1" or not torch.cuda.is_available():
        pytest.skip("官方 checkpoint 前向比较要求 BCG_CWT_RUN_CUDA=1 且 CUDA 可用")
    try:
        from mamba_ssm import Mamba2  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"官方 mamba_ssm 导入失败: {exc}")
    from resp_train.crd.training import partition_weight_decay_parameters
    a = mod.build_bcg_cwt_patch_mamba(synthetic_config()).cuda().eval()
    torch.save(a.state_dict(), tmp_path / "official.pt")
    b = mod.build_bcg_cwt_patch_mamba(synthetic_config()).cuda().eval()
    b.load_state_dict(torch.load(tmp_path / "official.pt", map_location="cpu", weights_only=True))
    partition = partition_weight_decay_parameters(b)
    marked = [p for p in b.parameters() if getattr(p, "_no_weight_decay", False)]
    assert marked and all(id(p) in {id(q) for q in partition.no_decay} for p in marked)
    x, w = torch.randn(1, 1, 18000, device="cuda"), torch.rand(1, 97, 360, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(a(x, tf={"w": w})["waveform"], b(x, tf={"w": w})["waveform"], rtol=1e-5, atol=1e-6)
