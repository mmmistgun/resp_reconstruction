from __future__ import annotations

import torch
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block, ResidualDWBlock
from resp_train.crd.config import load_crd_config
from resp_train.crd.model import build_crd_model
from resp_train.models import list_models


class _FakeMamba(nn.Module):
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        super().__init__()
        type(self).calls.append(dict(kwargs))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


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
    }
    shared_trunk_state: dict[str, torch.Tensor] | None = None
    direct_frontend_state: dict[str, torch.Tensor] | None = None
    for variant, parameter_count in expected.items():
        cfg = load_crd_config(f"configs/crd_v1/{variant}.yaml")
        model = build_crd_model(cfg)
        assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == parameter_count
        if variant in {
            "crd_102_b0_local_mamba",
            "crd_103_direct_local_mamba",
            "crd_104_direct_hier_mamba",
        }:
            current = {
                **{f"local.{key}": value for key, value in model.local_blocks.state_dict().items()},
                **{f"refine.{key}": value for key, value in model.refinement.state_dict().items()},
                **{f"head.{key}": value for key, value in model.head.state_dict().items()},
            }
            if shared_trunk_state is None:
                shared_trunk_state = {key: value.clone() for key, value in current.items()}
            else:
                assert current.keys() == shared_trunk_state.keys()
                assert all(torch.equal(current[key], shared_trunk_state[key]) for key in current)
        if variant in {"crd_103_direct_local_mamba", "crd_104_direct_hier_mamba"}:
            current_direct = model.frontend.state_dict()
            if direct_frontend_state is None:
                direct_frontend_state = {key: value.clone() for key, value in current_direct.items()}
            else:
                assert current_direct.keys() == direct_frontend_state.keys()
                assert all(torch.equal(current_direct[key], direct_frontend_state[key]) for key in current_direct)
        if variant == "crd_104_direct_hier_mamba":
            assert torch.count_nonzero(model.global_stage.film.weight) == 0
            assert torch.count_nonzero(model.global_stage.film.bias) == 0
