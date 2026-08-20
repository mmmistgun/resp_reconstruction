from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from resp_train.crd.config import check_crd_dependencies
from resp_train.temporal.blocks import (
    BidirectionalLSTMTrunk,
    BidirectionalMambaTrunk,
    FixedBlockCenterDecimator1D,
    FixedPolyphaseDecimator1D,
    MultiscaleTemporalPyramid,
    TemporalStem,
    tcn_receptive_field_tokens,
)
from resp_train.temporal.config import load_resp_temporal_config
from resp_train.temporal.model import (
    MULTISCALE_COARSE_DILATIONS,
    RTM_VARIANT_SPECS,
    TCN_DILATIONS,
    RespTemporalModel,
    build_resp_temporal_model,
    trainable_parameter_count,
    validate_variant_structure,
)
from resp_train.temporal_signal_audit import (
    ResampleSpec,
    fixed_resample,
    lowpass_block_center_decimate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs/resp_temporal_v1"
FLOAT64_ATOL = 5e-12
FLOAT64_RTOL = 5e-12


class _FakeMamba(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        dimension = int(kwargs["d_model"])
        self.projection = nn.Linear(dimension, dimension, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def _model(variant: str, *, fake_mamba: bool = False) -> RespTemporalModel:
    return RespTemporalModel(
        variant,
        20260811,
        mamba_factory=_FakeMamba if fake_mamba else None,
    )


@pytest.mark.parametrize(
    ("input_fs", "down", "numtaps", "cutoff_hz", "length"),
    [
        (100.0, 5, 255, 9.0, 1800),
        (20.0, 2, 127, 4.5, 1800),
    ],
)
def test_fixed_polyphase_matches_train_audit_operator(
    input_fs: float,
    down: int,
    numtaps: int,
    cutoff_hz: float,
    length: int,
) -> None:
    rng = np.random.default_rng(20260820 + down)
    signals = rng.normal(size=(2, length)) + np.linspace(-0.4, 0.7, length)
    module = FixedPolyphaseDecimator1D(
        input_fs=input_fs,
        down=down,
        numtaps=numtaps,
        cutoff_hz=cutoff_hz,
    )
    actual = module(torch.from_numpy(signals)[:, None, :]).detach().numpy()[:, 0]
    expected = np.stack(
        [
            fixed_resample(
                signal,
                input_fs=input_fs,
                spec=ResampleSpec(up=1, down=down, numtaps=numtaps, cutoff_hz=cutoff_hz),
                beta=8.6,
                padtype="line",
            )
            for signal in signals
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=FLOAT64_ATOL, rtol=FLOAT64_RTOL)
    assert not tuple(module.parameters())
    assert "taps" in dict(module.named_buffers())


@pytest.mark.parametrize(("down", "cutoff_hz"), [(5, 0.85), (10, 0.45)])
def test_fixed_multiscale_decimation_matches_train_audit_operator(down: int, cutoff_hz: float) -> None:
    rng = np.random.default_rng(20260900 + down)
    signals = rng.normal(size=(2, 1800))
    module = FixedBlockCenterDecimator1D(
        input_fs=10.0,
        down=down,
        numtaps=255,
        cutoff_hz=cutoff_hz,
    )
    actual = module(torch.from_numpy(signals)[:, None, :]).detach().numpy()[:, 0]
    expected = np.stack(
        [
            lowpass_block_center_decimate(
                signal,
                input_fs=10.0,
                spec=ResampleSpec(up=1, down=down, numtaps=255, cutoff_hz=cutoff_hz),
                beta=8.6,
            )
            for signal in signals
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=FLOAT64_ATOL, rtol=FLOAT64_RTOL)
    assert not tuple(module.parameters())
    assert "taps" in dict(module.named_buffers())


def test_fixed_decimators_remain_differentiable_and_finite() -> None:
    x = torch.randn(1, 2, 1800, requires_grad=True)
    module = FixedBlockCenterDecimator1D(input_fs=10.0, down=5, numtaps=255, cutoff_hz=0.85)
    loss = module(x).square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_locked_stem_order_shape_and_tcn_context_contracts() -> None:
    assert tcn_receptive_field_tokens(TCN_DILATIONS) == 2045
    assert 2045 / 10.0 == 204.5
    assert tcn_receptive_field_tokens(MULTISCALE_COARSE_DILATIONS) == 253
    assert 253 / 1.0 == 253.0
    stem = TemporalStem().eval()
    assert stem.carrier_filter_20.stride == (1,)
    assert stem.project_20.stride == (1,)
    with torch.no_grad():
        latent = stem(torch.zeros(1, 1, 18000))
    assert latent.shape == (1, 96, 1800)
    assert torch.isfinite(latent).all()


def test_control_end_to_end_output_is_raw_100hz_and_finite() -> None:
    model = _model("rtm_v1_t0_locked_stem_head").eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 18000))
    assert set(output) == {"waveform", "waveform_10hz"}
    assert output["waveform"].shape == (1, 1, 18000)
    assert output["waveform_10hz"].shape == (1, 1, 1800)
    assert torch.isfinite(output["waveform"]).all()
    assert torch.isfinite(output["waveform_10hz"]).all()


@pytest.mark.parametrize(
    "module",
    [
        BidirectionalLSTMTrunk(layers=2),
        MultiscaleTemporalPyramid(hidden_channels=384),
        BidirectionalMambaTrunk(layers=6, mamba_factory=_FakeMamba),
    ],
)
def test_locked_temporal_trunks_preserve_10hz_grid(module: nn.Module) -> None:
    module.eval()
    with torch.no_grad():
        output = module(torch.randn(1, 96, 1800))
    assert output.shape == (1, 96, 1800)
    assert torch.isfinite(output).all()


def test_same_seed_common_stem_and_decoder_are_tensor_identical_across_all_candidates() -> None:
    control = _model("rtm_v1_t0_locked_stem_head")
    candidates = [
        _model("rtm_v1_tcn_d9_h384"),
        _model("rtm_v1_bimamba2_d96_l6", fake_mamba=True),
        _model("rtm_v1_bilstm_h96_l2"),
        _model("rtm_v1_multiscale_10_2_1_h384"),
    ]
    for candidate in candidates:
        for key, value in control.stem.state_dict().items():
            torch.testing.assert_close(value, candidate.stem.state_dict()[key], rtol=0.0, atol=0.0)
        for key, value in control.decoder.state_dict().items():
            torch.testing.assert_close(value, candidate.decoder.state_dict()[key], rtol=0.0, atol=0.0)


def test_non_mamba_parameter_and_structure_contracts() -> None:
    for variant, spec in RTM_VARIANT_SPECS.items():
        if spec.family == "bimamba2":
            continue
        model = _model(variant)
        assert trainable_parameter_count(model) == spec.expected_trainable_parameters
        validate_variant_structure(model)


def test_mamba_parameter_contract_with_pinned_dependency() -> None:
    problems = check_crd_dependencies()
    if problems:
        pytest.skip("; ".join(problems))
    variant = "rtm_v1_bimamba2_d96_l6"
    model = _model(variant)
    assert trainable_parameter_count(model) == RTM_VARIANT_SPECS[variant].expected_trainable_parameters
    validate_variant_structure(model)


def test_builder_uses_strict_candidate_identity() -> None:
    cfg = load_resp_temporal_config(CONFIG_ROOT / "rtm_v1_tcn_d9_h384.yaml")
    model = build_resp_temporal_model(cfg)
    assert model.variant == "rtm_v1_tcn_d9_h384"
    assert trainable_parameter_count(model) == int(cfg.model.expected_trainable_parameters)


def test_model_rejects_wrong_shape_and_nonfinite_input() -> None:
    model = _model("rtm_v1_t0_locked_stem_head")
    with pytest.raises(ValueError, match="期望输入"):
        model(torch.zeros(1, 1, 17999))
    bad = torch.zeros(1, 1, 18000)
    bad[..., 0] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        model(bad)
