from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from resp_train.crd.blocks import DecoderResidual
from resp_train.crd.training import build_crd_optimizer
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1 as e9
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_engineering as engineering
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_formal as formal
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import (
    ARMS,
    ARM_SPECS,
    E9A_ARMS,
    E9B_ARMS,
    ChannelResidualRefiner,
    E9LatentWidthConditionRefinerModel,
    build_e9_latent_width_condition_refiner_model,
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


def _baseline(seed: int = e9.SEEDS[0]):
    return e9.load_w0_baseline(seed)


def _config(arm: str, seed: int = e9.SEEDS[0]):
    baseline = _baseline(seed)
    return e9.derived_config(
        baseline,
        arm=arm,
        output_root=Path("/tmp/e9") / arm / f"seed_{seed}",
        device="cpu",
    )


def test_spec_and_formal_plan_are_complete_and_closed():
    spec = e9.load_experiment_spec()
    assert tuple(spec.arms) == ARMS
    assert len(E9A_ARMS) == len(E9B_ARMS) == 3
    assert len(e9.formal_plan()) == 18
    assert {item["status"] for item in e9.formal_plan()} == {
        "blocked_until_gpu_acceptance_and_implementation_lock"
    }
    assert len({(item["arm"], item["seed"]) for item in e9.formal_plan()}) == 18
    for seed in e9.SEEDS:
        source = _baseline(seed)
        frozen = OmegaConf.to_container(source, resolve=True)
        for arm in ARMS:
            cfg = e9.derived_config(
                source,
                arm=arm,
                output_root=Path("/tmp/e9") / arm / f"seed_{seed}",
                device="cpu",
            )
            e9.validate_config(
                cfg,
                source,
                arm=arm,
                output_root=Path("/tmp/e9") / arm / f"seed_{seed}",
                device="cpu",
            )
            assert OmegaConf.to_container(
                cfg.model.e9_latent_width_condition_refiner_v1, resolve=True
            ) == model_contract(arm)
        assert OmegaConf.to_container(source, resolve=True) == frozen


@pytest.mark.parametrize("arm", ARMS)
def test_six_models_have_exact_structure_parameters_and_optimizer(arm):
    model = build_e9_latent_width_condition_refiner_model(_config(arm))
    spec = ARM_SPECS[arm]
    assert trainable_parameter_count(model) == spec.trainable_parameters
    assert isinstance(model.base.decoder_residual, DecoderResidual)
    assert model.base.decoder_residual.output.weight.count_nonzero() == 0
    assert model.base.decoder_residual.output.bias.count_nonzero() == 0
    projection = model.branches["w"].final_projection
    assert (projection.in_channels, projection.out_channels) == (
        spec.latent_channels,
        2 * spec.latent_channels,
    )
    assert projection.weight.count_nonzero() == 0
    assert projection.bias.count_nonzero() == 0

    if spec.family == "e9a":
        refiner = model.branches["w"].parameter_fill
    else:
        refiner = model.branches["w"].condition_refiner
    if spec.hidden_channels is None:
        assert isinstance(refiner, nn.Identity)
        assert not tuple(refiner.parameters())
    elif arm == "e9a_d96_h65":
        assert refiner.__class__.__name__ == "_ActiveParameterFill"
    else:
        assert isinstance(refiner, ChannelResidualRefiner)
        assert refiner.latent_channels == spec.latent_channels
        assert refiner.hidden_channels == spec.hidden_channels

    optimizer, partition = build_crd_optimizer(model, _config(arm))
    named = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(partition.decay_names) | set(partition.no_decay_names) == named
    assert set(partition.decay_names).isdisjoint(partition.no_decay_names)
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_d64_tensor_contracts_are_explicit():
    model = E9LatentWidthConditionRefinerModel("e9b_d64_h64", e9.SEEDS[0])
    assert model.base.frontend.adapter.in_channels == 16
    assert model.base.frontend.adapter.out_channels == 64
    assert (model.base.frontend.norm.num_groups, model.base.frontend.norm.num_channels) == (
        8,
        64,
    )
    assert len(model.base.local_blocks) == 6
    for block in model.base.local_blocks:
        assert block.merge.in_features == 128
        assert block.merge.out_features == 64
    branch = model.branches["w"]
    assert branch.conv_out.in_channels == 48
    assert branch.conv_out.out_channels == 64
    assert [block.depthwise.dilation for block in branch.temporal] == [(1,), (2,), (4,)]
    assert all(block.norm.num_groups == 8 for block in branch.temporal)
    assert [block.depthwise.dilation for block in model.base.refinement] == [(1,), (2,)]
    assert model.base.head.conv.in_channels == model.base.head.conv.out_channels == 64
    assert model.base.head.depthwise.groups == 64
    assert model.base.head.reduce.out_channels == 32
    assert 64 % 32 == 0


def test_shared_initialization_is_tensor_identical_within_each_width_family():
    seed = e9.SEEDS[1]
    for family_arms in (E9A_ARMS, E9B_ARMS):
        reference_model = E9LatentWidthConditionRefinerModel(family_arms[0], seed)
        reference = reference_model.state_dict()
        prefixes = reference_model.shared_state_prefixes
        for arm in family_arms[1:]:
            state = E9LatentWidthConditionRefinerModel(arm, seed).state_dict()
            for name, value in reference.items():
                if name.startswith(prefixes):
                    assert name in state and torch.equal(value, state[name]), (arm, name)


def test_same_seed_has_identical_initial_waveform_within_each_width_family():
    seed = e9.SEEDS[0]
    x = torch.randn(1, 1, 18_000, generator=torch.Generator().manual_seed(10))
    w = torch.randn(1, 97, 360, generator=torch.Generator().manual_seed(11))
    for family_arms in (E9A_ARMS, E9B_ARMS):
        expected = None
        for arm in family_arms:
            model = E9LatentWidthConditionRefinerModel(arm, seed).eval()
            model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
            with torch.no_grad():
                output = model(x, tf={"w": w})["waveform"]
            assert output.shape == (1, 1, 18_000)
            assert bool(torch.isfinite(output).all())
            if expected is None:
                expected = output.clone()
            else:
                assert torch.equal(expected, output), arm
            del model, output
        gc.collect()


def _has_nonzero_finite_grad(module: nn.Module, name: str) -> bool:
    parameter = dict(module.named_parameters())[name]
    return parameter.grad is not None and bool(
        torch.isfinite(parameter.grad).all() and parameter.grad.ne(0).any()
    )


@pytest.mark.parametrize("latent,hidden", [(96, 64), (96, 48), (64, 64), (64, 48)])
def test_zero_init_film_opens_refiner_gradient_over_three_real_updates(latent, hidden):
    generator = torch.Generator().manual_seed(latent + hidden)
    context = torch.randn(2, latent, 64, generator=generator)
    target = torch.randn(2, 2 * latent, 64, generator=generator)
    refiner = ChannelResidualRefiner(latent, hidden)
    projection = nn.Conv1d(latent, 2 * latent, kernel_size=1, bias=True)
    nn.init.zeros_(projection.weight)
    nn.init.zeros_(projection.bias)
    optimizer = torch.optim.AdamW([*refiner.parameters(), *projection.parameters()], lr=1e-3)
    observed = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        # Refiner 的生产合同固定 T=1800；梯度测试保留生产算子但缩短 disposable T。
        value = context + refiner.project(torch.nn.functional.silu(refiner.expand(context)))
        loss = (projection(value) - target).square().mean()
        loss.backward()
        observed.append(
            (
                _has_nonzero_finite_grad(projection, "weight"),
                _has_nonzero_finite_grad(refiner, "project.weight"),
                _has_nonzero_finite_grad(refiner, "expand.weight"),
            )
        )
        optimizer.step()
    assert observed[0] == (True, False, False)
    assert observed[1][1]
    assert observed[2][2]


def test_shape_nonfinite_and_state_roundtrip_fail_explicitly():
    refiner = ChannelResidualRefiner(64, 48)
    with pytest.raises(ValueError, match="期望"):
        refiner(torch.zeros(1, 63, 1800))
    model = E9LatentWidthConditionRefinerModel("e9b_d64_direct", e9.SEEDS[0])
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        model.base.decode_local(torch.full((1, 64, 1800), float("nan")))
    clone = E9LatentWidthConditionRefinerModel("e9b_d64_direct", e9.SEEDS[0])
    clone.load_state_dict(model.state_dict(), strict=True)
    for name, value in clone.state_dict().items():
        assert torch.isfinite(value).all(), name


def test_parameter_and_declared_mac_order_reflects_capacity_change():
    assert ARM_SPECS["e9a_d96_h48"].trainable_parameters < ARM_SPECS[
        "e9a_d96_h64"
    ].trainable_parameters < ARM_SPECS["e9a_d96_h65"].trainable_parameters
    assert ARM_SPECS["e9b_d64_direct"].trainable_parameters < ARM_SPECS[
        "e9b_d64_h48"
    ].trainable_parameters < ARM_SPECS["e9b_d64_h64"].trainable_parameters
    assert max(ARM_SPECS[arm].declared_covered_macs for arm in E9B_ARMS) < min(
        ARM_SPECS[arm].declared_covered_macs for arm in E9A_ARMS
    )


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


def test_p2_factor_tracking_and_resource_contracts():
    assert set(engineering._factor_prefixes("e9a_d96_h65")) == {
        "film_projection",
        "condition_refiner",
        "decoder_residual",
    }
    assert set(engineering._factor_prefixes("e9b_d64_direct")) == {
        "film_projection",
        "decoder_residual",
    }
    assert set(engineering._factor_prefixes("e9b_d64_h48")) == {
        "film_projection",
        "condition_refiner",
        "decoder_residual",
    }
    assert engineering.MAX_RESOURCE_ARM == "e9a_d96_h65"
    assert engineering.BATCH1_UPDATES >= 3
    assert engineering.PHYSICAL_BATCH_UPDATES >= 3
    assert engineering.PHYSICAL_BATCH_SIZE == 128
    assert engineering.MEMORY_LIMIT_FRACTION == 0.85


def test_p2_exclusive_lifecycle_freezes_success_and_preserves_failure(tmp_path):
    identity_hash = "a" * 64
    with engineering.exclusive_attempt(
        tmp_path / "success",
        phase="synthetic",
        identity_hash=identity_hash,
    ) as output:
        (output / "result.txt").write_text("ok", encoding="utf-8")
    completed = next((tmp_path / "success").glob("*/freeze_receipt.json")).parent
    assert (completed / "manifest.json").is_file()
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


def test_p2_frozen_evidence_is_complete():
    evidence = formal.verify_p2_evidence()
    assert evidence["file_count"] == 24
    assert evidence["engineering_identity_sha256"] == formal.P2_ENGINEERING_IDENTITY
    assert evidence["max_resource"]["arm"] == engineering.MAX_RESOURCE_ARM
    assert evidence["max_resource"]["batch_size"] == 128
    assert evidence["max_resource"]["updates"] == 3
    assert evidence["max_resource"]["peak_reserved_fraction"] < 0.85


def test_formal_runtime_compatibility_requires_same_stack_and_memory():
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
    receipt = formal._runtime_compatibility(
        {**accepted, "device_total_bytes": 120}, accepted
    )
    assert receipt["capacity_policy"] == "current_device_total_bytes_gte_p2"
    with pytest.raises(ValueError, match="总显存"):
        formal._runtime_compatibility(
            {**accepted, "device_total_bytes": 99}, accepted
        )
    with pytest.raises(ValueError, match="torch"):
        formal._runtime_compatibility({**accepted, "torch": "other"}, accepted)


def test_formal_lifecycle_rejects_duplicate_completed_cell(tmp_path):
    lock_hash = "b" * 64
    parent = tmp_path / "formal" / "e9a_d96_h65" / "seed_20260811"
    with formal._formal_attempt(
        parent,
        lock_hash=lock_hash,
        arm="e9a_d96_h65",
        seed=20260811,
    ) as output:
        engineering._write_json(
            output / "formal_receipt.json",
            {
                "implementation_lock_sha256": lock_hash,
                "arm": "e9a_d96_h65",
                "seed": 20260811,
                "validation_rows": 2675,
            },
        )
    with pytest.raises(FileExistsError, match="已完成"):
        with formal._formal_attempt(
            parent,
            lock_hash=lock_hash,
            arm="e9a_d96_h65",
            seed=20260811,
        ):
            pass
