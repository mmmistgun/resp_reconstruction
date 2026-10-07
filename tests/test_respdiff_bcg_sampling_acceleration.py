"""CPU 合成验收：原 B condition 统计、逐 step 误差、分组尾部和真实 prefix 计时。"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.acceleration_checks import TOLERANCES, error_record, require_passing, verify_grouped
from resp_train.respdiff_bcg.baseband import RespDiffBCGBaseband
from resp_train.respdiff_bcg.inference_v2 import predict_prefixes, state_digest
from resp_train.respdiff_bcg.runtime import environment, seed_all, sha256
from resp_train.respdiff_bcg.sampling import (
    SamplerSpec, batch_trajectory_noise, ensemble_prefixes, reconstruct_ensemble_parent,
)
from resp_train.respdiff_bcg.sampling_accelerated import (
    AccelerationSpec, accelerated_groups, accelerated_prefixes, encode_condition, execution_budget,
    forward_cached, grouped_noise,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old_threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        raise AssertionError("本次测试禁止 CUDA 调用")
    for name in ("_lazy_init", "is_available", "manual_seed_all", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(old_threads)


def model():
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(71).get_state())
    return RespDiffBCGBaseband(RespDiffSpec(8, 1, 4), objective="source_equivalent").eval()


def condition(batch):
    generator = torch.Generator(device="cpu").manual_seed(73)
    return torch.randn(batch, 1, 600, generator=generator)


def keys(batch):
    return [("synthetic", 20000 + index // 13, index % 13) for index in range(batch)]


@pytest.mark.parametrize("batch", [3, 64])
def test_cached_forward_exact_original_batch_and_cache_scope(batch):
    network = model().diffusion_model
    original = condition(batch)
    noisy = condition(batch) * .3
    step = torch.arange(batch, dtype=torch.long) % 50
    before = list(network.state_dict())
    expected = network(original, noisy, step)
    cache = encode_condition(network, original)
    actual = forward_cached(network, original, cache, noisy, step)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert list(network.state_dict()) == before
    with pytest.raises(ValueError, match="当前 batch"):
        forward_cached(network, original.clone(), cache, noisy, step)
    original.add_(.1)
    with pytest.raises(ValueError, match="当前 batch"):
        forward_cached(network, original, cache, noisy, step)


@pytest.mark.parametrize("sampler,nfe,count", [("ddim", 6, 9), ("ddpm", 50, 3), ("ddpm", 50, 4)])
def test_all_groups_per_step_trajectory_and_prefix_errors(tmp_path, sampler, nfe, count):
    network, original = model(), condition(2)
    spec = SamplerSpec(sampler=sampler, nfe=nfe, n_trajectories=count)
    counts = tuple(sorted({1, 2, count} | ({4, 8} if count == 9 else set())))
    checkpoint = tmp_path / "fixture_checkpoint.pt"
    torch.save(network.state_dict(), checkpoint)
    before = sha256(checkpoint), state_digest(network)
    reports = []
    for group in (1, 2, 4, 8):
        records = verify_grouped(network, original, keys(2), spec, AccelerationSpec(group), counts=counts)
        assert all(record["finite"] and record["passed"] for record in records)
        assert {record["stage"] for record in records} >= {"epsilon", "reverse", "trajectory", "prefix", "initial_noise"}
        if sampler == "ddpm":
            assert any(record["stage"] == "reverse_noise" for record in records)
        if group == 1:
            assert all(record["max_abs_error"] == 0 for record in records)
        if group <= 2:
            assert all(record["passed"] for record in records)
        reports.extend(records)
    assert (sha256(checkpoint), state_digest(network)) == before
    (tmp_path / "cpu_errors.json").write_text(json.dumps(reports, indent=2, allow_nan=False))


def test_grouped_forward_at_same_state_in_sensitive_timestep():
    from resp_train.respdiff_bcg.sampling_accelerated import sample_group
    network, original = model(), condition(2)
    spec = SamplerSpec(n_trajectories=9)
    observed = []
    @torch.inference_mode()
    def compare(event):
        if event["timestep"] == 19:
            # 使用分组路径的实际 x_t，排除先前 reverse 舍入带来的输入差异。
            expected = network.diffusion_model(original, event["current"][1],
                                               torch.full((2,), 19, dtype=torch.long))
            observed.append(error_record(event["epsilon"][1], expected, "epsilon"))
    sample_group(network, original, keys(2), spec, (4, 5, 6, 7),
                 encode_condition(network.diffusion_model, original), observer=compare)
    assert len(observed) == 1 and observed[0]["passed"]


def test_cache_called_once_group_shapes_and_partial_group():
    network, original = model(), condition(3)
    spec = SamplerSpec(n_trajectories=5)
    observed = {"fine": [], "coarse": [], "noisy": []}
    handles = [module.register_forward_pre_hook(lambda module, inputs, key=key:
        observed[key].append(len(inputs[0]))) for key, module in (
            ("fine", network.diffusion_model.ppg_encoder2), ("coarse", network.diffusion_model.ppg_encoder1),
            ("noisy", network.diffusion_model.noise_encoder))]
    try:
        result = dict(accelerated_prefixes(network, original, keys(3), spec, AccelerationSpec(4),
                                          checkpoints=(1, 2, 4, 5)))
    finally:
        for handle in handles:
            handle.remove()
    assert observed["fine"] == observed["coarse"] == [3]
    assert observed["noisy"] == [12] * 6 + [3] * 6
    assert set(result) == {1, 2, 4, 5}
    assert execution_budget(spec, AccelerationSpec(4))["actual_denoiser_forward_calls"] == 12


def test_noise_bitwise_and_nested_first_four_for_all_groups():
    original_keys = keys(3)
    for step in (None, 1, 49):
        spec = SamplerSpec(sampler="ddpm", nfe=50, n_trajectories=4)
        expected = torch.stack([batch_trajectory_noise(original_keys, trajectory_id=index,
            noise_seed=spec.noise_seed, device="cpu", reverse_step=step) for index in range(4)])
        for size in (1, 2, 4, 8):
            pieces = [grouped_noise(original_keys, tuple(range(first, min(first + size, 4))), spec, "cpu",
                                  reverse_step=step) for first in range(0, 4, size)]
            torch.testing.assert_close(torch.cat(pieces), expected, rtol=0, atol=0)
    network, original = model(), condition(2)
    for size in (1, 2, 4, 8):
        smaller = dict(accelerated_prefixes(network, original, keys(2), SamplerSpec(n_trajectories=4),
                                            AccelerationSpec(size)))
        larger = dict(accelerated_prefixes(network, original, keys(2), SamplerSpec(n_trajectories=8),
                                           AccelerationSpec(size), checkpoints=(4, 8)))
        report = error_record(smaller[4], larger[4], "prefix")
        assert report["passed"]
        if size <= 4:
            torch.testing.assert_close(smaller[4], larger[4], rtol=0, atol=0)


def test_parent_reconstruction_and_filter_match_reference(tmp_path):
    network, original = model(), condition(13)
    spec = SamplerSpec(n_trajectories=3)
    reference = dict(ensemble_prefixes(network, original, keys(13), spec))[3].numpy()
    expected = reconstruct_ensemble_parent(reference)
    reports = []
    for size in (1, 2, 4, 8):
        value = dict(accelerated_prefixes(network, original, keys(13), spec, AccelerationSpec(size)))[3].numpy()
        actual = reconstruct_ensemble_parent(value)
        for stage, observed, target in zip(("prefix", "raw_waveform", "postfiltered_waveform"), actual, expected):
            record = error_record(observed, target, stage, G=size)
            reports.append(record)
            assert record["finite"] and record["passed"]
    (tmp_path / "cpu_waveform_errors.json").write_text(json.dumps(reports, indent=2, allow_nan=False))


def test_input_tail_batch_and_shared_group_readiness(tmp_path):
    class Dataset:
        rows = pd.DataFrame({"dataset_row_id": [20000], "split": ["synthetic"]})
        def __len__(self):
            return 13
        def __getitem__(self, index):
            return {"x": condition(13)[index], "index": index}
    network = model()
    batch_sizes = []
    handle = network.diffusion_model.ppg_encoder1.register_forward_pre_hook(
        lambda module, inputs: batch_sizes.append(len(inputs[0])))
    try:
        predictions, profile = predict_prefixes(network, Dataset(), SamplerSpec(n_trajectories=4),
            (1, 2, 4), tmp_path, batch_size=5, acceleration=AccelerationSpec(4))
    finally:
        handle.remove()
    assert batch_sizes == [5, 5, 3]
    assert predictions[4].shape == (1, 13, 1, 600)
    assert [row["denoiser_calls_per_chunk"] for row in profile] == [6, 12, 24]
    assert all(row["batched_denoiser_forward_calls"] == 18 for row in profile)
    assert all(row["completed_trajectories_when_prefix_ready"] == 4 for row in profile)
    assert len({row["sampling_seconds"] for row in profile}) == 1
    assert all("not independent N runtime" in row["runtime_definition"] for row in profile)


def test_nonfinite_mode_group_and_cpu_only_runtime_guards():
    network = model()
    for size in (0, 3, 16):
        with pytest.raises(ValueError):
            AccelerationSpec(size)
    with pytest.raises(ValueError, match="非有限"):
        encode_condition(network.diffusion_model, torch.full((1, 1, 600), float("nan")))
    network.train()
    with pytest.raises(ValueError, match="eval"):
        encode_condition(network.diffusion_model, condition(1))
    # CPU synthetic CLI 不查询或 seed CUDA；fixture 已将这些调用设为失败。
    seed_all(71, cpu_only=True)
    assert environment(cpu_only=True)["cudnn"] is None
    rejected = error_record(np.array([1.]), np.array([0.]), "epsilon")
    assert not rejected["passed"] and rejected["n_outside_tolerance"] == 1
    assert TOLERANCES["initial_noise"] == {"atol": 0., "rtol": 0.}


def test_tolerance_revision_preserves_reverse_and_noise_guards_and_failure_details():
    # 合成一个与 T630 上报幅度相近的 ε 差异；不冒充实际 GPU 输出重验。
    reference = np.array([.319])
    actual = reference + .000477
    assert np.any(abs(actual - reference) > 1e-4 + 1e-3 * abs(reference))
    record = error_record(actual, reference, "epsilon", G=2, trajectory_id=6, timestep=0)
    assert record["passed"]
    require_passing([record], "synthetic")
    failed = error_record(reference + .002, reference, "epsilon", G=2, trajectory_id=6, timestep=0)
    with pytest.raises(FloatingPointError) as error:
        require_passing([record, failed], "condition_cache_G2")
    message = str(error.value)
    assert '1/2' in message and '"trajectory_id": 6' in message and '"timestep": 0' in message
    assert '"n_elements": 1' in message and '"rms_error"' in message
    assert not error_record(np.array([.021]), np.zeros(1), "reverse")["passed"]
    assert not error_record(np.array([1e-12]), np.zeros(1), "initial_noise")["passed"]


def test_inference_tensor_cache_rejects_inplace_condition_change():
    network = model().diffusion_model
    with torch.inference_mode():
        original = condition(1)
        cache = encode_condition(network, original)
        step = torch.zeros(1, dtype=torch.long)
        original.add_(.1)
        with pytest.raises(ValueError, match="缓存无效"):
            forward_cached(network, original, cache, condition(1), step)


def test_previous_group_released_before_next_group_compute(monkeypatch):
    import weakref
    import resp_train.respdiff_bcg.sampling_accelerated as accelerated
    original_sampler = accelerated.sample_group
    references = []
    def tracked(*args, **kwargs):
        if references:
            assert references[-1]() is None
        result = original_sampler(*args, **kwargs)
        references.append(weakref.ref(result))
        return result
    monkeypatch.setattr(accelerated, "sample_group", tracked)
    dict(accelerated.accelerated_prefixes(model(), condition(1), keys(1), SamplerSpec(n_trajectories=5),
                                         AccelerationSpec(2), checkpoints=(1, 2, 4, 5)))
    assert len(references) == 3 and all(reference() is None for reference in references)
