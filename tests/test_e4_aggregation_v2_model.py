"""E4-v2 的纯 synthetic CPU 合同；原生 Mamba 仅构造，不执行 forward。"""

from __future__ import annotations

import io
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import CRDTfV1Model, CwtBranch, trainable_parameter_count
from resp_train.crd.training import partition_weight_decay_parameters
from resp_train.crd import tf_v1_model
from resp_train.paper_evidence.e4_aggregation_v2_model import (
    ADDED_PARAMETERS, ARMS, REGIONS, AggregationCwtBranch, AggregationModel,
    ScaleAggregator, aggregation_contract, centered_weights,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def frequencies():
    # 仅读取协议附带的频率坐标，不访问波形、cache 或独立测试集数组。
    path = Path(__file__).resolve().parents[1] / "docs/experiments/e4_w0_scale_aggregation_source_audit_20260917.json"
    return np.asarray(json.loads(path.read_text())["frequency"]["values_hz"], dtype=np.float64)


def synthetic_value(seed=8):
    return torch.randn(1, 96, 97, 360, generator=torch.Generator().manual_seed(seed))


def observed_weights(module, value):
    """通过生产 forward 捕获 score，再独立归一化以检查输入依赖。"""
    if hasattr(module, "logits"):
        scores = module.logits.detach()
    else:
        observed = []
        hook = module.score.register_forward_hook(lambda _, args, output: observed.append(output.detach()))
        try:
            with torch.no_grad():
                module(value)
        finally:
            hook.remove()
        scores = observed[0]
    return torch.softmax(scores.double() + module.prior_counts.double().log(), dim=2)


@pytest.mark.parametrize("seed", [20260811, 20260812, 20260813])
def test_all_arms_preserve_w0_shared_tensors_and_cpu_rng(seed, frequencies):
    torch.manual_seed(71)
    start = torch.get_rng_state()
    reference = CRDTfV1Model("crd_tf102_w", seed)
    expected_rng = torch.get_rng_state()
    assert trainable_parameter_count(reference) == 1_219_850
    attention_state = None
    for arm in ARMS:
        torch.set_rng_state(start)
        candidate = AggregationModel(arm, seed, frequencies)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        state = candidate.state_dict()
        for name, value in reference.state_dict().items():
            assert torch.equal(value, state[name]), (arm, seed, name)
        extras = set(state) - set(reference.state_dict())
        assert extras and all(name.startswith("branches.w.aggregation.") for name in extras)
        assert trainable_parameter_count(candidate) == 1_219_850 + ADDED_PARAMETERS[arm]
        assert trainable_parameter_count(candidate.branches["w"]) == 150_048 + ADDED_PARAMETERS[arm]
        assert candidate.branches["w"].parameter_fill.expand.out_channels == 65
        assert len(candidate.base.local_blocks) == 6
        assert all(parameter.device.type == "cpu" for parameter in candidate.parameters())
        ref_partition = partition_weight_decay_parameters(reference)
        partition = partition_weight_decay_parameters(candidate)
        for names, expected_names in ((partition.decay_names, ref_partition.decay_names),
                                      (partition.no_decay_names, ref_partition.no_decay_names)):
            assert tuple(name for name in names if not name.startswith("branches.w.aggregation.")) == expected_names
        if "attention" in arm:
            shared = {name: value.clone() for name, value in candidate.branches["w"].aggregation.state_dict().items()
                      if name.startswith(("content.", "score."))}
            if attention_state is not None:
                assert shared.keys() == attention_state.keys()
                assert all(torch.equal(value, attention_state[name]) for name, value in shared.items())
            attention_state = shared


def test_new_aggregation_initialization_does_not_add_cuda_seed_calls(monkeypatch, frequencies):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    CRDTfV1Model("crd_tf102_w", 20260811)
    expected = calls.copy()
    assert expected
    for arm in ARMS:
        calls.clear()
        AggregationModel(arm, 20260811, frequencies)
        assert calls == expected, arm


@pytest.mark.parametrize("arm", ARMS)
def test_chunked_nonzero_aggregation_matches_forward_and_gradients(arm, frequencies, monkeypatch):
    monkeypatch.setattr(tf_v1_model, "TF_BRANCH_CHECKPOINT_BATCH_CHUNK", 1)
    with module_seed(20260917, "tf_branch_w"):
        direct = AggregationCwtBranch(arm, 20260917, frequencies)
    with torch.no_grad():
        direct.final_projection.weight.normal_(0, .005)
        if hasattr(direct.aggregation, "logits"):
            direct.aggregation.logits.normal_(0, .1)
        else:
            direct.aggregation.score.weight.normal_(0, .1)
    chunked = copy.deepcopy(direct)
    value = torch.randn(2, 97, 360, generator=torch.Generator().manual_seed(53))
    left, right = value.clone().requires_grad_(), value.clone().requires_grad_()
    a = direct({"w": left})
    b = tf_v1_model._checkpointed_mapping_branch(chunked, {"w": right})
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
    sum(x.square().mean() for x in a).backward()
    sum(y.square().mean() for y in b).backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6)
    for x, y in zip(direct.parameters(), chunked.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_initialization_is_bitwise_original_mean(arm, dtype, frequencies):
    module = ScaleAggregator(arm, 7, frequencies).to(dtype=dtype)
    value = synthetic_value().to(dtype)
    with torch.no_grad():
        output = module(value)
    assert output.dtype == dtype
    assert torch.equal(output, value.mean(dim=2))


@pytest.mark.parametrize("counts", [[1] * 97, [25, 24, 24, 24]])
def test_centered_delta_matches_independent_softmax_and_is_shift_invariant(counts):
    prior = torch.tensor(counts).reshape(1, 1, -1, 1)
    scores = torch.linspace(-3, 4, len(counts)).reshape(1, 1, -1, 1)
    actual = centered_weights(scores, prior)
    expected = torch.softmax(scores.double() + prior.double().log(), dim=2) - prior.double() / sum(counts)
    torch.testing.assert_close(actual.double(), expected, atol=4e-8, rtol=2e-6)
    torch.testing.assert_close(actual.sum(dim=2), torch.zeros(1, 1, 1), atol=1e-7, rtol=0)
    torch.testing.assert_close(centered_weights(scores + 100, prior), actual, atol=5e-7, rtol=2e-5)
    assert torch.equal(centered_weights(torch.zeros_like(scores), prior), torch.zeros_like(scores))
    assert torch.equal(centered_weights(torch.full_like(scores, 1e30), prior), torch.zeros_like(scores))
    extremes = torch.full_like(scores, -torch.finfo(torch.float32).max)
    extremes[:, :, 0] = torch.finfo(torch.float32).max
    weight = centered_weights(extremes, prior) + prior / sum(counts)
    assert torch.isfinite(weight).all()
    torch.testing.assert_close(weight[:, :, 0], torch.ones(1, 1, 1))
    torch.testing.assert_close(weight[:, :, 1:], torch.zeros_like(weight[:, :, 1:]), atol=1e-7, rtol=0)


@pytest.mark.parametrize("arm", ["static_scale", "channel_region"])
def test_static_weights_are_input_independent_and_forward_uses_prior(arm, frequencies):
    module = ScaleAggregator(arm, 7, frequencies)
    with torch.no_grad():
        module.logits.copy_(torch.linspace(-1, 1, module.logits.numel()).reshape_as(module.logits))
    value = synthetic_value()
    weights = observed_weights(module, value)
    assert torch.equal(weights, observed_weights(module, value * 2 + 3))
    if arm == "channel_region":
        assert module.prior_counts.flatten().tolist() == [25, 24, 24, 24]
        features = torch.stack([value[:, :, start:stop].double().mean(dim=2) for start, stop in REGIONS], dim=2)
    else:
        features = value.double()
    expected = (features * weights).sum(dim=2)
    torch.testing.assert_close(module(value).double(), expected, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("arm", ["scale_attention", "frequency_attention"])
def test_dynamic_attention_changes_with_content_and_normalizes(arm, frequencies):
    module = ScaleAggregator(arm, 7, frequencies)
    with torch.no_grad():
        module.score.weight.fill_(.2)
    value = synthetic_value()
    first, second = observed_weights(module, value), observed_weights(module, -2 * value)
    assert not torch.allclose(first, second)
    assert (first > 0).all()
    torch.testing.assert_close(first.sum(dim=2), torch.ones(1, 1, 360, dtype=torch.float64))
    expected = (value.double() * first).sum(dim=2)
    torch.testing.assert_close(module(value).double(), expected, atol=3e-7, rtol=3e-5)


def test_frequency_coordinate_enters_nonlinearity_before_score(frequencies):
    module = ScaleAggregator("frequency_attention", 7, frequencies)
    with torch.no_grad():
        module.content.weight.zero_()
        module.content.bias.fill_(.7)
        module.frequency_weight.fill_(1.3)
        module.score.weight.fill_(.2)
    value = synthetic_value()
    weights = observed_weights(module, value)
    coordinate = 2 * (np.log(frequencies) - np.log(frequencies[0])) / (np.log(frequencies[-1]) - np.log(frequencies[0])) - 1
    torch.testing.assert_close(module.frequency_coordinate.flatten(), torch.tensor(coordinate, dtype=torch.float32))
    assert module.frequency_coordinate.flatten()[1] == module.frequency_coordinate.flatten()[2]
    correct_scores = 1.6 * F.silu(.7 + 1.3 * module.frequency_coordinate.double())
    correct = torch.softmax(correct_scores, dim=2).expand_as(weights)
    torch.testing.assert_close(weights, correct, atol=1e-8, rtol=2e-6)
    linear_position = torch.softmax(1.6 * (F.silu(torch.tensor(.7)) + 1.3 * module.frequency_coordinate.double()), dim=2)
    assert not torch.allclose(weights, linear_position.expand_as(weights), atol=1e-5, rtol=1e-3)


def test_region_weights_have_no_cross_channel_coupling(frequencies):
    module = ScaleAggregator("channel_region", 7, frequencies)
    value = synthetic_value()
    with torch.no_grad():
        module.logits[0, 17, :, 0] = torch.tensor([-1., 2., 0., .5])
        first = module(value)
        changed = value.clone()
        changed[:, 42] += 17
        second = module(changed)
    assert torch.equal(first[:, :42], second[:, :42])
    assert torch.equal(first[:, 43:], second[:, 43:])
    assert not torch.equal(first[:, 42], second[:, 42])
    module.zero_grad(set_to_none=True)
    module(value)[:, 17].square().mean().backward()
    assert module.logits.grad[:, 17].ne(0).all()
    assert torch.count_nonzero(module.logits.grad[:, :17]) == 0
    assert torch.count_nonzero(module.logits.grad[:, 18:]) == 0


@pytest.mark.parametrize("arm", ARMS)
def test_original_branch_initial_context_and_multistep_new_parameter_gradients(arm, frequencies):
    with module_seed(20260811, "tf_branch_w"):
        reference = CwtBranch()
    with module_seed(20260811, "tf_branch_w"):
        candidate = AggregationCwtBranch(arm, 20260811, frequencies)
    context = {}
    hooks = [reference.temporal.register_forward_pre_hook(lambda _, args: context.update(reference=args[0].clone())),
             candidate.temporal.register_forward_pre_hook(lambda _, args: context.update(candidate=args[0].clone()))]
    generator = torch.Generator().manual_seed(42)
    features = {"w": torch.rand(1, 97, 360, generator=generator)}
    with torch.no_grad():
        reference(features)
        candidate(features)
    for hook in hooks:
        hook.remove()
    assert torch.equal(context["reference"], context["candidate"])
    initial = {name: value.detach().clone() for name, value in candidate.aggregation.named_parameters()}
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4, weight_decay=0)
    target = torch.randn(1, 96, 1800, generator=generator)
    # 原 FiLM 与 attention score 均为零；保留真实分支，让梯度逐层打开。
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        gamma, beta = candidate(features)
        ((gamma - target).square().mean() + (beta + target).square().mean()).backward()
        for name, parameter in candidate.aggregation.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (arm, step, name)
            if step == 0:
                assert torch.count_nonzero(parameter.grad) == 0
            if step == 4:
                assert torch.count_nonzero(parameter.grad) == parameter.numel(), (arm, name)
        optimizer.step()
    for name, parameter in candidate.aggregation.named_parameters():
        assert torch.isfinite(parameter).all()
        assert not torch.equal(parameter, initial[name]), (arm, name)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("bad", ["shape", "nan", "inf", "parameter_inf"])
def test_bad_inputs_and_nonfinite_parameters_fail(arm, bad, frequencies):
    module = ScaleAggregator(arm, 7, frequencies)
    value = torch.zeros(1, 96, 97, 360)
    if bad == "shape":
        value = value[:, :, :96]
    elif bad == "parameter_inf":
        with torch.no_grad():
            next(module.parameters()).flatten()[0] = float("inf")
    else:
        value[0, 0, 0, 0] = float(bad)
    with pytest.raises((ValueError, FloatingPointError)):
        module(value)


@pytest.mark.parametrize("arm", ARMS)
def test_frequency_identity_errors_are_rejected(arm, frequencies):
    for bad in (frequencies.astype(np.float32), frequencies[:96], frequencies[::-1].copy(), frequencies + 1e-8):
        with pytest.raises(ValueError):
            ScaleAggregator(arm, 7, bad)
    for invalid in (np.nan, np.inf, 0.):
        bad = frequencies.copy()
        bad[0] = invalid
        with pytest.raises(ValueError):
            ScaleAggregator(arm, 7, bad)


@pytest.mark.parametrize("arm", ARMS)
def test_strict_checkpoint_roundtrip_and_w0_rejection(arm, frequencies):
    model = AggregationModel(arm, 7, frequencies)
    stream = io.BytesIO()
    torch.save({"model_state_dict": model.state_dict(), "aggregation_contract": aggregation_contract(arm)}, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    restored = AggregationModel(arm, 9, frequencies)
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["aggregation_contract"] == aggregation_contract(arm)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    reference = CRDTfV1Model("crd_tf102_w", 7)
    with pytest.raises(RuntimeError):
        restored.load_state_dict(reference.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        reference.load_state_dict(model.state_dict(), strict=True)
