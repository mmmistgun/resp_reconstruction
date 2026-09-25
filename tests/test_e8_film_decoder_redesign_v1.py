from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from resp_train.crd.blocks import DecoderResidual
from resp_train.crd.training import build_crd_optimizer
from resp_train.paper_evidence import e8_film_decoder_redesign_v1 as e8
from resp_train.paper_evidence import e8_film_decoder_redesign_v1_engineering as engineering
from resp_train.paper_evidence import e8_film_decoder_redesign_v1_formal as formal
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_model import (
    ARMS,
    ARM_SPECS,
    CONDITION_REFINERS,
    DECODERS,
    ChannelResidualRefiner,
    E8FilmDecoderRedesignModel,
    SingleReadoutResidual,
    TemporalDecoderResidual,
    build_e8_film_decoder_redesign_model,
    model_contract,
    trainable_parameter_count,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def baseline(seed: int = e8.SEEDS[0]):
    return e8.load_w0_baseline(seed)


def test_spec_configs_and_formal_plan_are_complete():
    spec = e8.load_experiment_spec()
    assert tuple(spec.arms) == ARMS
    assert len(ARMS) == 12
    assert {(item.condition_refiner, item.decoder) for item in ARM_SPECS.values()} == {
        (condition, decoder)
        for condition in CONDITION_REFINERS
        for decoder in DECODERS
    }
    assert len(e8.formal_plan()) == 36
    assert {item["status"] for item in e8.formal_plan()} == {
        "blocked_until_engineering_acceptance"
    }
    for seed in e8.SEEDS:
        source = baseline(seed)
        frozen = OmegaConf.to_container(source, resolve=True)
        for arm in ARMS:
            output = Path("/tmp/e8") / arm / f"seed_{seed}"
            cfg = e8.derived_config(source, arm=arm, output_root=output, device="cpu")
            e8.validate_config(
                cfg,
                source,
                arm=arm,
                output_root=output,
                device="cpu",
            )
            assert OmegaConf.to_container(
                cfg.model.e8_film_decoder_redesign_v1, resolve=True
            ) == model_contract(arm)
        assert OmegaConf.to_container(source, resolve=True) == frozen


@pytest.mark.parametrize("arm", ARMS)
def test_twelve_models_have_exact_structure_parameters_and_optimizer(arm):
    cfg = e8.derived_config(
        baseline(),
        arm=arm,
        output_root=Path("/tmp/e8") / arm,
        device="cpu",
    )
    model = build_e8_film_decoder_redesign_model(cfg)
    spec = ARM_SPECS[arm]
    assert trainable_parameter_count(model) == spec.trainable_parameters

    refiner = model.branches["w"].parameter_fill
    if spec.condition_refiner == "fill65":
        assert refiner.__class__.__name__ == "_ActiveParameterFill"
    elif spec.condition_refiner == "direct":
        assert isinstance(refiner, nn.Identity)
        assert not tuple(refiner.parameters())
    else:
        assert isinstance(refiner, ChannelResidualRefiner)
        assert refiner.hidden_channels == (96 if spec.condition_refiner == "res96" else 192)

    decoder = model.base.decoder_residual
    expected_decoder = {
        "pointwise": DecoderResidual,
        "single": SingleReadoutResidual,
        "temporal": TemporalDecoderResidual,
    }[spec.decoder]
    assert isinstance(decoder, expected_decoder)
    if spec.decoder == "single":
        assert not tuple(decoder.parameters())

    optimizer, partition = build_crd_optimizer(model, cfg)
    named = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(partition.decay_names) | set(partition.no_decay_names) == named
    assert set(partition.decay_names).isdisjoint(partition.no_decay_names)
    optimizer_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_parameters == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_temporal_decoder_has_exact_local_time_contract():
    module = TemporalDecoderResidual()
    assert module.depthwise.kernel_size == (5,)
    assert module.depthwise.dilation == (2,)
    assert module.depthwise.padding == (4,)
    assert module.depthwise.groups == 32
    assert module.depthwise.bias is None
    assert module.mix.kernel_size == (1,)
    assert module.mix.bias is None
    assert trainable_parameter_count(module) == 1_217
    features = torch.randn(2, 32, 1800, generator=torch.Generator().manual_seed(7))
    output = module(features)
    assert output.shape == (2, 1, 1800)
    assert torch.equal(output, torch.zeros_like(output))


def test_film_and_decoder_final_layers_are_zero_initialized():
    for arm in ARMS:
        model = E8FilmDecoderRedesignModel(arm, e8.SEEDS[0])
        projection = model.branches["w"].final_projection
        assert projection.weight.count_nonzero() == 0
        assert projection.bias.count_nonzero() == 0
        if ARM_SPECS[arm].decoder != "single":
            output = model.base.decoder_residual.output
            assert output.weight.count_nonzero() == 0
            assert output.bias.count_nonzero() == 0
        del model
    gc.collect()


def test_same_seed_all_twelve_arms_have_identical_initial_function():
    seed = e8.SEEDS[0]
    x = torch.randn(1, 1, 18_000, generator=torch.Generator().manual_seed(10))
    w = torch.randn(1, 97, 360, generator=torch.Generator().manual_seed(11))
    expected = None
    for arm in ARMS:
        model = E8FilmDecoderRedesignModel(arm, seed).eval()
        # 初始函数测试不需要重复执行昂贵的 Mamba；其公共 state 另行逐 tensor 核验。
        model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
        with torch.no_grad():
            output = model(x, tf={"w": w})["waveform"]
        if expected is None:
            expected = output.clone()
        else:
            assert torch.equal(expected, output), arm
        del model, output
    gc.collect()


def test_shared_initialization_and_named_factor_initialization_are_paired():
    seed = e8.SEEDS[1]
    reference = E8FilmDecoderRedesignModel("e8_fill65_pointwise", seed).state_dict()
    shared_prefixes = (
        "base.frontend.",
        "base.local_blocks.",
        "base.refinement.",
        "base.head.",
        "branches.w.conv_in.",
        "branches.w.norm.",
        "branches.w.depthwise.",
        "branches.w.conv_out.",
        "branches.w.temporal.",
        "branches.w.final_projection.",
    )
    for arm in ARMS[1:]:
        state = E8FilmDecoderRedesignModel(arm, seed).state_dict()
        for name, value in reference.items():
            if name.startswith(shared_prefixes):
                assert name in state and torch.equal(value, state[name]), (arm, name)

    res96 = [
        E8FilmDecoderRedesignModel(f"e8_res96_{decoder}", seed).state_dict()
        for decoder in DECODERS
    ]
    for name, value in res96[0].items():
        if name.startswith("branches.w.parameter_fill."):
            assert all(torch.equal(value, state[name]) for state in res96[1:])
    temporal = [
        E8FilmDecoderRedesignModel(f"e8_{condition}_temporal", seed).state_dict()
        for condition in CONDITION_REFINERS
    ]
    for name, value in temporal[0].items():
        if name.startswith("base.decoder_residual."):
            assert all(torch.equal(value, state[name]) for state in temporal[1:])


def _has_nonzero_grad(module: nn.Module, name: str) -> bool:
    parameter = dict(module.named_parameters())[name]
    return parameter.grad is not None and bool(
        torch.isfinite(parameter.grad).all() and parameter.grad.ne(0).any()
    )


def test_zero_init_paths_open_over_multiple_updates():
    generator = torch.Generator().manual_seed(17)
    context = torch.randn(2, 96, 1800, generator=generator)
    target = torch.randn(2, 192, 1800, generator=generator)
    refiner = ChannelResidualRefiner(96)
    final = nn.Conv1d(96, 192, kernel_size=1, bias=True)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    optimizer = torch.optim.AdamW([*refiner.parameters(), *final.parameters()], lr=1e-3)
    observed = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (final(refiner(context)) - target).square().mean()
        loss.backward()
        observed.append(
            (
                _has_nonzero_grad(final, "weight"),
                _has_nonzero_grad(refiner, "project.weight"),
                _has_nonzero_grad(refiner, "expand.weight"),
            )
        )
        optimizer.step()
    assert observed[0] == (True, False, False)
    assert observed[1][1]
    assert observed[2][2]

    features = torch.randn(2, 32, 1800, generator=generator)
    waveform_target = torch.randn(2, 1, 1800, generator=generator)
    decoder = TemporalDecoderResidual()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    observed = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (decoder(features) - waveform_target).square().mean()
        loss.backward()
        observed.append(
            (
                _has_nonzero_grad(decoder, "output.weight"),
                _has_nonzero_grad(decoder, "mix.weight"),
                _has_nonzero_grad(decoder, "depthwise.weight"),
            )
        )
        optimizer.step()
    assert observed[0] == (True, False, False)
    assert observed[1][1] and observed[1][2]


def test_finite_and_shape_failures_are_explicit():
    refiner = ChannelResidualRefiner(96)
    with pytest.raises(ValueError, match="期望"):
        refiner(torch.zeros(1, 95, 1800))
    decoder = TemporalDecoderResidual()
    with pytest.raises(ValueError, match="期望"):
        decoder(torch.zeros(1, 31, 1800))
    model = E8FilmDecoderRedesignModel("e8_direct_temporal", e8.SEEDS[0])
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        model.base.decode_local(torch.full((1, 96, 1800), float("nan")))


def test_p2_synthetic_batch_is_deterministic_finite_and_shape_safe():
    first = engineering.synthetic_batch(2, 91)
    second = engineering.synthetic_batch(2, 91)
    assert first["x"].shape == (2, 1, 18_000)
    assert first["target"].shape == (2, 1, 18_000)
    assert first["tf"]["w"].shape == (2, 97, 360)
    assert torch.equal(first["x"], second["x"])
    assert torch.equal(first["target"], second["target"])
    assert torch.equal(first["tf"]["w"], second["tf"]["w"])
    engineering.finite_tree(first)
    with pytest.raises(ValueError, match="必须为正"):
        engineering.synthetic_batch(0, 91)


def test_p2_factor_tracking_matches_each_structural_level():
    assert engineering._factor_prefixes("e8_direct_single") == {
        "film_projection": "branches.w.final_projection."
    }
    assert set(engineering._factor_prefixes("e8_fill65_pointwise")) == {
        "film_projection",
        "condition_refiner",
        "decoder_residual",
    }
    assert set(engineering._factor_prefixes("e8_res192_temporal")) == {
        "film_projection",
        "condition_refiner",
        "decoder_residual",
    }
    assert engineering.MAX_RESOURCE_ARM == "e8_res192_temporal"
    assert engineering.BATCH1_UPDATES >= 3
    assert engineering.PHYSICAL_BATCH_UPDATES >= 3
    assert engineering.MEMORY_LIMIT_FRACTION == 0.80


def test_p2_exclusive_lifecycle_freezes_success_and_preserves_failure(tmp_path):
    identity_hash = "a" * 64
    with engineering.exclusive_attempt(
        tmp_path / "success",
        phase="synthetic",
        identity_hash=identity_hash,
    ) as output:
        (output / "result.txt").write_text("ok", encoding="utf-8")
    completed = next((tmp_path / "success").glob("*/freeze_receipt.json")).parent
    manifest = json.loads((completed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert "result.txt" in manifest["files"]
    with pytest.raises(FileExistsError, match="已完成"):
        with engineering.exclusive_attempt(
            tmp_path / "success",
            phase="synthetic",
            identity_hash=identity_hash,
        ):
            pass

    with pytest.raises(RuntimeError, match="fixture"):
        with engineering.exclusive_attempt(
            tmp_path / "failure",
            phase="synthetic",
            identity_hash=identity_hash,
        ) as output:
            (output / "partial.txt").write_text("keep", encoding="utf-8")
            raise RuntimeError("fixture failure")
    failed = next((tmp_path / "failure").glob("*/lifecycle_failed.json")).parent
    assert (failed / "partial.txt").read_text(encoding="utf-8") == "keep"
    assert not (failed / "freeze_receipt.json").exists()


def test_p2_frozen_evidence_and_formal_plan_are_complete():
    evidence = formal.verify_p2_evidence()
    assert evidence["gpu_acceptance"]["file_count"] == 54
    assert evidence["benchmark"]["file_count"] == 53
    assert evidence["max_acceptance_reserved_fraction"] < 0.8
    assert evidence["max_benchmark_reserved_fraction"] < 0.8
    plan = formal.formal_plan()
    assert len(plan) == 36
    assert {(item["arm"], item["seed"]) for item in plan} == {
        (arm, seed) for arm in ARMS for seed in e8.SEEDS
    }


def test_formal_runtime_compatibility_requires_same_stack_and_enough_memory():
    accepted = {
        "python": "3.12",
        "torch": "2.12",
        "cuda_runtime": "13.0",
        "cudnn": 92000,
        "dependencies": {"mamba-ssm": "2.3.2"},
        "device_name": "GPU",
        "amp_dtype": "bfloat16",
        "device_total_bytes": 100,
    }
    current = {**accepted, "device_total_bytes": 120}
    receipt = formal._runtime_compatibility(current, accepted)
    assert receipt["capacity_policy"] == "current_device_total_bytes_gte_p2"
    with pytest.raises(ValueError, match="总显存"):
        formal._runtime_compatibility({**current, "device_total_bytes": 99}, accepted)
    with pytest.raises(ValueError, match="torch"):
        formal._runtime_compatibility({**current, "torch": "other"}, accepted)


def test_formal_lifecycle_rejects_duplicate_completed_cell(tmp_path):
    lock_hash = "b" * 64
    parent = tmp_path / "formal" / "e8_direct_single" / "seed_20260811"
    with formal._formal_attempt(
        parent,
        lock_hash=lock_hash,
        arm="e8_direct_single",
        seed=20260811,
    ) as output:
        engineering._write_json(
            output / "formal_receipt.json",
            {
                "implementation_lock_sha256": lock_hash,
                "arm": "e8_direct_single",
                "seed": 20260811,
                "validation_rows": 2675,
            },
        )
    with pytest.raises(FileExistsError, match="已完成"):
        with formal._formal_attempt(
            parent,
            lock_hash=lock_hash,
            arm="e8_direct_single",
            seed=20260811,
        ):
            pass
