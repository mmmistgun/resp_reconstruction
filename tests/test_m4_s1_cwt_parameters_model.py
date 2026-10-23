"""合成CPU模型验收：实际网格、原生B等价、所有模块及非零条件梯度。"""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from resp_train.crd.training import build_crd_optimizer

from scripts import m4_s1_cwt_parameters_spec as spec
from scripts import m4_s1_cwt_parameters_model as mod
from scripts import m4_s1_cwt_parameters_runtime as run
from scripts.m4_linear_fusion_model import LinearFusionModel
from m4_s1_fixtures import FixtureMamba, cpu_only


@pytest.fixture(scope="module")
def reps():
    return spec.representations()


def small():
    return replace(mod.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)


def net(arm, reps, seed=spec.SEEDS[0]):
    key = spec.ARMS[arm].representation
    return mod.M4S1CWTModel(arm, seed, reps[key] if key else None,
        cfg=small(), mamba_factory=FixtureMamba, canonical_frequencies=reps["B"]["frequencies_hz"])


def inputs(arm, reps):
    key = spec.ARMS[arm].representation
    x = torch.randn(1, 1, 18000)
    return x, {"w": torch.rand(1, *reps[key]["shape"])} if key else None


def open_condition(model):
    if model.condition is not None:
        with torch.no_grad():
            model.condition.output.weight.normal_(std=.03)
            model.condition.output.bias.normal_(std=.01)


@pytest.mark.parametrize("seed", spec.SEEDS)
@pytest.mark.parametrize("bf16", [False, True])
def test_baseline_nonzero_forward_and_gradients_exact(reps, seed, bf16):
    original = LinearFusionModel("L1", seed, reps["B"]["frequencies_hz"], cfg=small(), mamba_factory=FixtureMamba).eval()
    adapted = net("B", reps, seed).eval()
    assert original.state_dict().keys() == adapted.state_dict().keys()
    assert list(original.state_dict()) == list(adapted.state_dict())
    assert [n for n, _ in original.named_parameters()] == [n for n, _ in adapted.named_parameters()]
    for name, value in original.state_dict().items():
        torch.testing.assert_close(adapted.state_dict()[name], value, atol=0, rtol=0)
    open_condition(original)
    adapted.load_state_dict(original.state_dict(), strict=True)
    x, tf = inputs("B", reps)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        left = original(x, tf=tf)["waveform"]
        right = adapted(x, tf=tf)["waveform"]
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    left.float().square().mean().backward()
    right.float().square().mean().backward()
    for name, value in original.named_parameters():
        torch.testing.assert_close(dict(adapted.named_parameters())[name].grad, value.grad, atol=0, rtol=0)


def test_baseline_optimizer_partition_and_step_match(reps, tmp_path):
    original = LinearFusionModel("L1", spec.SEEDS[0], reps["B"]["frequencies_hz"],
                                 cfg=small(), mamba_factory=FixtureMamba).eval()
    adapted = net("B", reps).eval()
    cfg = spec.config("B", spec.SEEDS[0], tmp_path)
    left_opt, left_partition = build_crd_optimizer(original, cfg)
    right_opt, right_partition = build_crd_optimizer(adapted, cfg)
    assert left_partition.decay_names == right_partition.decay_names
    assert left_partition.no_decay_names == right_partition.no_decay_names
    x, tf = inputs("B", reps)
    for _ in range(2):
        for model, optimizer in ((original, left_opt), (adapted, right_opt)):
            optimizer.zero_grad(set_to_none=True)
            model(x, tf=tf)["waveform"].square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(adapted.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("arm", tuple(spec.ARMS))
def test_all_arms_nonzero_condition_shapes_and_active_gradients(reps, arm):
    before = torch.get_rng_state().clone()
    model = net(arm, reps).eval()
    assert torch.equal(before, torch.get_rng_state())
    baseline = net("B", reps)
    original = dict(baseline.named_parameters())
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    open_condition(model)
    x, tf = inputs(arm, reps)
    out = model(x, tf=tf)["waveform"]
    assert out.shape == (1, 1, 18000) and torch.isfinite(out).all()
    out.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    if model.condition is not None:
        assert model.condition.encoder[1].weight.grad.abs().sum() > 0
        assert model.condition.value.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", ("P_025", "P_100", "Q_mu6_v24", "A_UniformTFReadout", "A_NoPoolPosition",
                                "A_ScorePositionOnly", "A_ValuePositionOnly"))
def test_bf16_adapter_gradients(reps, arm):
    model = net(arm, reps).eval()
    open_condition(model)
    x, tf = inputs(arm, reps)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, tf=tf)["waveform"]
    out.float().square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("arm,p", [("B", 50), ("P_025", 25), ("P_100", 100)])
def test_sampling_on_physical_time_ramp_and_boundary(reps, arm, p):
    condition = net(arm, reps).condition
    condition.encoder = nn.Identity()
    times = torch.tensor(reps[arm]["time_seconds"], dtype=torch.float32)
    values = times[None, None].expand(1, 97, -1).clone().requires_grad_()
    actual = condition.encoded_grid(values)
    targets = (torch.arange(360) * 50 + 24.5) / 100
    expected = targets.clamp(float(times[0]), float(times[-1]))
    torch.testing.assert_close(actual[0, 0, 0], expected, atol=2e-5, rtol=1e-6)
    assert actual.shape == (1, 1, 97, 360)
    actual.sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.sum() == 97 * 360
    if p == 50:
        torch.testing.assert_close(actual[:, 0], values, atol=0, rtol=0)


def test_uniform_readout_preserves_value_parameterization_and_gradients(reps):
    uniform = net("A_UniformTFReadout", reps).condition
    original = net("B", reps).condition
    for name, parameter in uniform.named_parameters():
        torch.testing.assert_close(parameter, dict(original.named_parameters())[name], atol=0, rtol=0)
    assert not any(name.startswith(("query.", "key.", "frequency_bias.", "time_bias.")) for name, _ in uniform.named_parameters())
    with torch.no_grad():
        uniform.output.weight.normal_(std=.03)
        original.output.load_state_dict(uniform.output.state_dict())
    z = torch.randn(1, 359, 32)
    w = torch.rand(1, 97, 360)
    actual = uniform(z, w)
    features = original.encoder(w[:, None]).movedim(1, -1)
    values = original.value(features).unflatten(-1, (4, 8))
    # 显式展开F×2单元作为独立参考，uniform路径仍保留原V和输出Linear。
    local = values.unfold(2, 2, 1).permute(0, 2, 3, 4, 1, 5)
    reference = original.output(local.float().mean((-2, -1)).flatten(-2))
    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
    actual.square().mean().backward()
    reference.square().mean().backward()
    for name, parameter in uniform.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(original.named_parameters())[name].grad, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("arm,score_position,value_position", [
    ("B", True, True), ("A_NoPoolPosition", False, False),
    ("A_ScorePositionOnly", True, False), ("A_ValuePositionOnly", False, True)])
def test_position_factorial_matches_explicit_weighted_formula(reps, arm, score_position, value_position):
    encoder = net(arm, reps).waveform_encoder
    x = torch.randn(1, 1, 18000)
    h = encoder.stem(x)
    p = encoder.position(encoder.relative_seconds[:, None]) if hasattr(encoder, "position") else None
    logits = encoder.score(h.transpose(1, 2)).squeeze(-1).unfold(-1, 100, 50)
    if score_position:
        logits = logits + F.linear(p, encoder.score.weight).squeeze(-1)
    weights = logits.softmax(-1)
    pooled = torch.einsum("bnp,bcnp->bnc", weights, h.unfold(-1, 100, 50))
    if value_position:
        pooled = pooled + torch.einsum("bnp,pc->bnc", weights, p)
    expected = encoder.project(pooled)
    actual = encoder(x)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    actual.square().mean().backward()
    if p is not None:
        assert encoder.position[0].weight.grad.abs().sum() > 0
    else:
        assert not any(name.startswith("position.") for name, _ in encoder.named_parameters())


def test_zero_projection_gates_then_opens_condition(reps):
    model = net("B", reps).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    x, tf = inputs("B", reps)
    captured = []
    hook = model.fusion.register_forward_hook(lambda _, args, out: captured.append((args[0].detach().clone(), out.detach().clone())))
    try:
        model(x, tf=tf)["waveform"].square().mean().backward()
        torch.testing.assert_close(*captured[0], atol=0, rtol=0)
        assert model.condition.output.weight.grad.abs().sum() > 0
        assert model.condition.encoder[1].weight.grad.count_nonzero() == 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(x, tf=tf)["waveform"].square().mean().backward()
        assert model.condition.encoder[1].weight.grad.abs().sum() > 0
    finally:
        hook.remove()


def test_input_and_representation_failures(reps):
    model = net("B", reps)
    x, tf = inputs("B", reps)
    tf["w"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        model(x, tf=tf)
    with pytest.raises(ValueError):
        net("A_NoTF", reps)(x, tf={"w": torch.zeros(1, 97, 360)})
    rep = deepcopy(reps["P_100"])
    rep["time_seconds"][0] += .1
    with pytest.raises(ValueError, match="时间"):
        mod.M4S1CWTModel("P_100", spec.SEEDS[0], rep, cfg=small(), mamba_factory=FixtureMamba)
    rep = deepcopy(reps["B"])
    rep["frequencies_hz"][0] = float("inf")
    with pytest.raises(ValueError):
        mod.validate_representation("B", rep)


def test_official_cpu_initialization_all_three_seeds(reps):
    # 官方Mamba只在CPU构造，不调用其CUDA forward/backward。
    audit = run.audit_initialization(reps)
    assert set(audit) == set(map(str, spec.SEEDS))
    for records in audit.values():
        assert len(records) == 28
        assert all(records[a]["parameters"] == 1046945 for a in spec.CWT_SETTINGS)
        assert records["A_ScorePositionOnly"]["parameters"] == records["B"]["parameters"]
        assert records["A_ValuePositionOnly"]["parameters"] == records["B"]["parameters"]
        assert records["A_NoPoolPosition"]["parameters"] < records["B"]["parameters"]
