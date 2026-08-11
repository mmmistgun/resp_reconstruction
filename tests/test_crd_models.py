from __future__ import annotations

import torch
from torch import nn

import resp_train.crd.model as crd_model_module
from resp_train.crd.blocks import BidirectionalMamba2Block, ResidualDWBlock
from resp_train.crd.config import load_crd_config
from resp_train.crd.model import GlobalContextStage, build_crd_model
from resp_train.models import list_models


class _FakeMamba(nn.Module):
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        super().__init__()
        type(self).calls.append(dict(kwargs))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _ZeroFrontend(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 96, 1800, device=x.device, dtype=x.dtype)


def test_residual_dw_block_is_exact_identity_at_initialization() -> None:
    block = ResidualDWBlock(96, dilation=2).eval()
    x = torch.randn(2, 96, 41)
    assert torch.equal(block(x), x)
    assert torch.count_nonzero(block.project.weight) == 0
    assert torch.count_nonzero(block.project.bias) == 0


def test_bidirectional_mamba_uses_two_independent_pinned_instances() -> None:
    _FakeMamba.calls.clear()
    block = BidirectionalMamba2Block(96, mamba_factory=_FakeMamba)
    x = torch.randn(2, 17, 96, requires_grad=True)
    output = block(x)
    output.sum().backward()

    assert output.shape == x.shape
    assert block.forward_mamba is not block.backward_mamba
    assert len(_FakeMamba.calls) == 2
    for kwargs in _FakeMamba.calls:
        assert kwargs == {
            "d_model": 96,
            "d_state": 64,
            "d_conv": 4,
            "expand": 2,
            "headdim": 32,
            "ngroups": 1,
            "chunk_size": 256,
            "rmsnorm": True,
            "bias": False,
            "conv_bias": True,
            "use_mem_eff_path": True,
        }


def test_b0_t4_and_bridge_share_exact_patch_initialization() -> None:
    b0 = build_crd_model(load_crd_config("configs/crd_v1/crd_001_b0_retrain.yaml"))
    t4 = build_crd_model(load_crd_config("configs/crd_v1/crd_002_t4_retrain.yaml"))
    bridge = build_crd_model(load_crd_config("configs/crd_v1/crd_101_b0_coarse.yaml"))

    reference_modules = (b0.patch_backbone.patch_embed, b0.patch_backbone.blocks)
    for candidate_modules in (
        (t4.t4.time_backbone.patch_embed, t4.t4.time_backbone.blocks),
        (bridge.frontend.encoder.patch_embed, bridge.frontend.encoder.blocks),
    ):
        for reference, candidate in zip(reference_modules, candidate_modules):
            assert candidate.state_dict().keys() == reference.state_dict().keys()
            assert all(
                torch.equal(reference.state_dict()[name], candidate.state_dict()[name])
                for name in reference.state_dict()
            )
    signal = torch.randn(1, 1, 18000)
    b0_tokens, _ = b0.patch_backbone.encode_tokens(signal)
    bridge_tokens = bridge.frontend.encoder(signal)
    assert torch.equal(b0_tokens, bridge_tokens)
    b0.eval()
    t4.eval()
    assert torch.equal(b0(signal)["waveform"], t4(signal)["waveform"])


def test_crd_101_forward_contract_and_registry() -> None:
    model = build_crd_model(load_crd_config("configs/crd_v1/crd_101_b0_coarse.yaml")).eval()
    output = model(torch.randn(1, 1, 18000))

    assert "crd_v1" in list_models()
    assert set(output) == {"waveform", "waveform_10hz"}
    assert output["waveform"].shape == (1, 1, 18000)
    assert output["waveform_10hz"].shape == (1, 1, 1800)
    assert output["waveform"].dtype == torch.float32
    assert torch.isfinite(output["waveform"]).all()


def test_crd_002_t4_spectral_path_survives_bfloat16_autocast() -> None:
    model = build_crd_model(load_crd_config("configs/crd_v1/crd_002_t4_retrain.yaml")).eval()
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        output = model(torch.randn(1, 1, 18000))
    assert output["waveform"].shape == (1, 1, 18000)
    assert torch.isfinite(output["waveform"]).all()


def test_crd_variant_trainable_parameter_counts_are_frozen() -> None:
    expected = {
        "crd_001_b0_retrain": 11408,
        "crd_002_t4_retrain": 12752,
        "crd_101_b0_coarse": 117361,
        "crd_102_b0_local_mamba": 1068745,
        "crd_103_direct_local_mamba": 1144165,
        "crd_104_direct_hier_mamba": 2256613,
        "crd_105_direct_coarse": 192781,
        "crd_106_b0_hier_mamba": 2181193,
        "crd_202_base_legacy_energy": 1071449,
        "crd_203_base_analytic_am": 1197785,
        "crd_204_base_morphology": 1106857,
        "crd_205_base_em_static": 1109561,
        "crd_206_base_am_static": 1235897,
        "crd_207_base_cap_em": 1109257,
        "crd_208_base_cap_am": 1235785,
        "crd_c101_b0_local_tcn": 1062001,
    }
    shared_decoder_state: dict[str, torch.Tensor] | None = None
    shared_local_state: dict[str, torch.Tensor] | None = None
    direct_frontend_state: dict[str, torch.Tensor] | None = None
    for variant, parameter_count in expected.items():
        cfg = load_crd_config(f"configs/crd_v1/{variant}.yaml")
        model = build_crd_model(cfg)
        assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == parameter_count
        if variant in {
            "crd_101_b0_coarse",
            "crd_102_b0_local_mamba",
            "crd_103_direct_local_mamba",
            "crd_104_direct_hier_mamba",
            "crd_105_direct_coarse",
            "crd_106_b0_hier_mamba",
            "crd_202_base_legacy_energy",
            "crd_203_base_analytic_am",
            "crd_204_base_morphology",
            "crd_205_base_em_static",
            "crd_206_base_am_static",
            "crd_207_base_cap_em",
            "crd_208_base_cap_am",
            "crd_c101_b0_local_tcn",
        }:
            current = {
                **{f"refine.{key}": value for key, value in model.refinement.state_dict().items()},
                **{f"head.{key}": value for key, value in model.head.state_dict().items()},
            }
            if shared_decoder_state is None:
                shared_decoder_state = {key: value.clone() for key, value in current.items()}
            else:
                assert current.keys() == shared_decoder_state.keys()
                assert all(torch.equal(current[key], shared_decoder_state[key]) for key in current)
        if variant in {
            "crd_102_b0_local_mamba",
            "crd_103_direct_local_mamba",
            "crd_104_direct_hier_mamba",
            "crd_106_b0_hier_mamba",
            "crd_202_base_legacy_energy",
            "crd_203_base_analytic_am",
            "crd_204_base_morphology",
            "crd_205_base_em_static",
            "crd_206_base_am_static",
            "crd_207_base_cap_em",
            "crd_208_base_cap_am",
        }:
            current_local = model.local_blocks.state_dict()
            if shared_local_state is None:
                shared_local_state = {key: value.clone() for key, value in current_local.items()}
            else:
                assert current_local.keys() == shared_local_state.keys()
                assert all(torch.equal(current_local[key], shared_local_state[key]) for key in current_local)
        if variant in {
            "crd_103_direct_local_mamba",
            "crd_104_direct_hier_mamba",
            "crd_105_direct_coarse",
        }:
            current_direct = model.frontend.state_dict()
            if direct_frontend_state is None:
                direct_frontend_state = {key: value.clone() for key, value in current_direct.items()}
            else:
                assert current_direct.keys() == direct_frontend_state.keys()
                assert all(torch.equal(current_direct[key], direct_frontend_state[key]) for key in current_direct)
        if variant in {"crd_104_direct_hier_mamba", "crd_106_b0_hier_mamba"}:
            assert torch.count_nonzero(model.global_stage.film.weight) == 0
            assert torch.count_nonzero(model.global_stage.film.bias) == 0


def test_crd_105_is_direct_coarse_without_mamba() -> None:
    model = build_crd_model(load_crd_config("configs/crd_v1/crd_105_direct_coarse.yaml")).eval()

    assert len(model.local_blocks) == 0
    assert model.global_stage is None
    output = model(torch.randn(1, 1, 18000))
    assert output["waveform"].shape == (1, 1, 18000)
    assert output["waveform_10hz"].shape == (1, 1, 1800)
    assert torch.isfinite(output["waveform"]).all()


def test_crd_106_is_exact_102_plus_104_global_stage() -> None:
    model_102 = build_crd_model(load_crd_config("configs/crd_v1/crd_102_b0_local_mamba.yaml"))
    model_104 = build_crd_model(load_crd_config("configs/crd_v1/crd_104_direct_hier_mamba.yaml"))
    model_106 = build_crd_model(load_crd_config("configs/crd_v1/crd_106_b0_hier_mamba.yaml"))

    for module_102, module_106 in (
        (model_102.frontend, model_106.frontend),
        (model_102.local_blocks, model_106.local_blocks),
        (model_102.refinement, model_106.refinement),
        (model_102.head, model_106.head),
    ):
        assert module_102.state_dict().keys() == module_106.state_dict().keys()
        assert all(
            torch.equal(module_102.state_dict()[name], module_106.state_dict()[name])
            for name in module_102.state_dict()
        )
    assert model_104.global_stage.state_dict().keys() == model_106.global_stage.state_dict().keys()
    assert all(
        torch.equal(model_104.global_stage.state_dict()[name], model_106.global_stage.state_dict()[name])
        for name in model_104.global_stage.state_dict()
    )


def test_c1_tcn_is_parameter_matched_full_context_and_preserves_shared_state() -> None:
    base = build_crd_model(load_crd_config("configs/crd_v1/crd_102_b0_local_mamba.yaml"))
    control = build_crd_model(load_crd_config("configs/crd_v1/crd_c101_b0_local_tcn.yaml"))

    for base_module, control_module in (
        (base.frontend, control.frontend),
        (base.refinement, control.refinement),
        (base.head, control.head),
    ):
        assert base_module.state_dict().keys() == control_module.state_dict().keys()
        assert all(
            torch.equal(base_module.state_dict()[name], control_module.state_dict()[name])
            for name in base_module.state_dict()
        )

    assert len(control.local_blocks) == 0
    assert [block.dilation for block in control.local_tcn_blocks] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    receptive_field = 1 + 4 * sum(block.dilation for block in control.local_tcn_blocks)
    assert receptive_field == 4093
    tcn_parameters = sum(parameter.numel() for parameter in control.local_tcn_blocks.parameters())
    mamba_parameters = sum(parameter.numel() for parameter in base.local_blocks.parameters())
    assert tcn_parameters == 944640
    assert mamba_parameters == 951384
    assert abs(tcn_parameters - mamba_parameters) / mamba_parameters <= 0.02

    latent = torch.randn(2, 96, 1800)
    output = latent
    for block in control.local_tcn_blocks:
        output = block(output)
    assert torch.equal(output, latent)


def test_global_context_stage_is_exact_identity_at_initialization() -> None:
    stage = GlobalContextStage().eval()
    stage.blocks = nn.ModuleList([nn.Identity() for _ in stage.blocks])
    local = torch.randn(1, 96, 1800)

    assert torch.equal(stage(local), local)


def test_s2a_variants_preserve_exact_102_trunk_and_use_zero_static_injection() -> None:
    base = build_crd_model(load_crd_config("configs/crd_v1/crd_102_b0_local_mamba.yaml"))
    for variant in (
        "crd_202_base_legacy_energy",
        "crd_203_base_analytic_am",
        "crd_204_base_morphology",
    ):
        candidate = build_crd_model(load_crd_config(f"configs/crd_v1/{variant}.yaml"))
        for base_module, candidate_module in (
            (base.frontend, candidate.frontend),
            (base.local_blocks, candidate.local_blocks),
            (base.refinement, candidate.refinement),
            (base.head, candidate.head),
        ):
            assert base_module.state_dict().keys() == candidate_module.state_dict().keys()
            assert all(
                torch.equal(base_module.state_dict()[name], candidate_module.state_dict()[name])
                for name in base_module.state_dict()
            )
        assert candidate.global_stage is None
        assert candidate.representation is not None
        assert torch.count_nonzero(candidate.representation.projection.weight) == 0
        assert torch.count_nonzero(candidate.representation.projection.bias) == 0
        if variant == "crd_204_base_morphology":
            assert set(candidate.regularization_terms()) == {"loss_proto"}
        else:
            assert candidate.regularization_terms() == {}


def test_s2br_combinations_reuse_single_branch_states_and_controls_are_identity() -> None:
    base = build_crd_model(load_crd_config("configs/crd_v1/crd_102_b0_local_mamba.yaml"))
    energy = build_crd_model(load_crd_config("configs/crd_v1/crd_202_base_legacy_energy.yaml"))
    analytic = build_crd_model(load_crd_config("configs/crd_v1/crd_203_base_analytic_am.yaml"))
    morphology = build_crd_model(load_crd_config("configs/crd_v1/crd_204_base_morphology.yaml"))
    em = build_crd_model(load_crd_config("configs/crd_v1/crd_205_base_em_static.yaml"))
    am = build_crd_model(load_crd_config("configs/crd_v1/crd_206_base_am_static.yaml"))

    for reference, candidate in (
        (energy.representation, em.representation),
        (morphology.representation, em.additional_representation),
        (analytic.representation, am.representation),
        (morphology.representation, am.additional_representation),
    ):
        assert reference.state_dict().keys() == candidate.state_dict().keys()
        assert all(
            torch.equal(reference.state_dict()[name], candidate.state_dict()[name])
            for name in reference.state_dict()
        )
    assert set(em.regularization_terms()) == {"loss_proto"}
    assert set(am.regularization_terms()) == {"loss_proto"}
    assert em.checkpoint_primary_representation is False
    assert am.checkpoint_primary_representation is True
    for combination in (em, am):
        assert combination.frontend.state_dict().keys() == base.frontend.state_dict().keys()
        assert all(
            torch.equal(combination.frontend.state_dict()[name], base.frontend.state_dict()[name])
            for name in base.frontend.state_dict()
        )
        assert torch.count_nonzero(combination.representation.projection.weight) == 0
        assert torch.count_nonzero(combination.additional_representation.projection.weight) == 0

    expected_matches = {
        "crd_207_base_cap_em": ("crd_205_base_em_static", 2, 104, 1109257, -304),
        "crd_208_base_cap_am": ("crd_206_base_am_static", 4, 216, 1235785, -112),
    }
    latent = torch.randn(2, 96, 31)
    for variant, expected in expected_matches.items():
        control = build_crd_model(load_crd_config(f"configs/crd_v1/{variant}.yaml")).eval()
        for base_module, control_module in (
            (base.frontend, control.frontend),
            (base.local_blocks, control.local_blocks),
            (base.refinement, control.refinement),
            (base.head, control.head),
        ):
            assert base_module.state_dict().keys() == control_module.state_dict().keys()
            assert all(
                torch.equal(base_module.state_dict()[name], control_module.state_dict()[name])
                for name in base_module.state_dict()
            )
        match = control.capacity_match
        assert (
            match.target_variant,
            match.block_count,
            match.hidden_channels,
            match.control_parameter_count,
            match.parameter_difference,
        ) == expected
        assert match.relative_parameter_difference <= 0.02
        assert torch.equal(control.capacity_control(latent), latent)
        assert control.regularization_terms() == {}


def test_s2br_am_checkpoint_path_preserves_parameter_gradients(monkeypatch) -> None:
    model = build_crd_model(load_crd_config("configs/crd_v1/crd_206_base_am_static.yaml"))
    model.frontend = _ZeroFrontend()
    model.local_blocks = nn.ModuleList()
    model.refinement = nn.Identity()
    model.head = nn.Conv1d(96, 1, kernel_size=1, bias=False)
    nn.init.ones_(model.head.weight)
    calls = 0
    real_checkpoint = crd_model_module.checkpoint

    def counted_checkpoint(function, *args, **kwargs):
        nonlocal calls
        calls += 1
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(crd_model_module, "checkpoint", counted_checkpoint)
    model.train()
    output = model(torch.randn(1, 1, 18000))
    objective = output["waveform_10hz"].sum() + 1e-3 * model.regularization_terms()["loss_proto"]
    objective.backward()

    assert calls == 1
    parameters = [
        *model.representation.parameters(),
        *model.additional_representation.parameters(),
    ]
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
