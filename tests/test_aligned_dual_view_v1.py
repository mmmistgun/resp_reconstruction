from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.signal import convolve
from torch import nn
from torch.nn import functional as F

from resp_train.aligned_dual_view import (
    AlignedDualViewV1, CWTBatch, ModelConfig, extract_cwt_10hz,
    load_model_config, prepare_cwt_batch, representation_spec,
)
from resp_train.aligned_dual_view.features import load_w0_grid
from resp_train.aligned_dual_view.signal import CenteredFIRDecimate10, fir_coefficients
from resp_train.losses.task import RespirationTaskLoss

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("joint", "waveform", "cwt", "joint_scale_mean")


class ContractMamba(nn.Module):
    """仅验证接线和梯度；不模拟原生 Mamba 的时序计算或性能。"""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.projection = nn.Linear(kwargs["d_model"], kwargs["d_model"], bias=False)

    def forward(self, value):
        return self.projection(value).tanh()


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(previous)


def synthetic_input():
    time = torch.arange(18000) / 100.0
    return ((1 + 0.3 * torch.sin(2 * torch.pi * 0.2 * time))
            * torch.sin(2 * torch.pi * 4 * time))[None, None]


def synthetic_features(batch=1, length=1800, *, requires_grad=False):
    values = torch.randn(batch, 97, length, requires_grad=requires_grad)
    return CWTBatch(values, representation_spec()["representation_id"])


def test_float32_fir_meets_frequency_spec():
    kernel = CenteredFIRDecimate10().kernel.numpy().ravel()
    np.testing.assert_array_equal(kernel, kernel[::-1])
    assert float(kernel.astype(np.float64).sum()) == pytest.approx(1, abs=1e-7)
    frequency = np.fft.rfftfreq(262144, d=0.01)
    response = np.abs(np.fft.rfft(kernel, n=262144))
    response_db = 20 * np.log10(np.maximum(response, 1e-30))
    assert np.max(np.abs(response_db[frequency <= 0.7])) < 0.1
    assert np.max(response_db[frequency >= 5]) < -60
    np.testing.assert_allclose(kernel, fir_coefficients(), atol=1e-8, rtol=1e-7)


def test_decimation_matches_independent_convolution_at_boundaries_and_has_gradient():
    layer = CenteredFIRDecimate10()
    values = torch.randn(2, 3, 811, requires_grad=True)
    actual = layer(values)
    reference = np.stack([
        convolve(np.pad(row, (250, 250), mode="reflect"),
                 fir_coefficients(), mode="valid", method="direct")[::10]
        for row in values.detach().numpy().reshape(-1, 811)
    ]).reshape(2, 3, -1)
    np.testing.assert_allclose(actual.detach().numpy(), reference, atol=5e-7, rtol=2e-5)
    actual.square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0
    reference_input = values.detach().clone().requires_grad_()
    full = F.conv1d(F.pad(reference_input.reshape(6, 1, 811), (250, 250), mode="reflect"), layer.kernel)
    full[..., ::10].square().sum().backward()
    torch.testing.assert_close(values.grad, reference_input.grad, atol=1e-6, rtol=1e-5)


def test_decimation_origin_and_amp_precision():
    layer = CenteredFIRDecimate10()
    impulse = torch.zeros(1, 1, 18000)
    impulse[..., 9000] = 1
    assert layer(impulse).shape == (1, 1, 1800)
    assert layer(impulse).argmax().item() == 900
    ones = torch.ones(1, 2, 600)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = layer(ones)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.ones_like(actual), atol=5e-7, rtol=0)
    with pytest.raises(TypeError, match="float32"):
        layer.bfloat16()(ones)


def test_waveform_frontend_can_preserve_signed_low_frequency_input():
    model = AlignedDualViewV1(mamba_factory=ContractMamba)
    conv = model.waveform_encoder[0]
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[0, 0, 25] = 1
        conv.weight[1, 0, 25] = -1
    value = torch.randn(1, 1, 18000)
    encoded = model.downsample(model.waveform_encoder(value))
    torch.testing.assert_close(encoded[:, :1] - encoded[:, 1:2], model.downsample(value),
                               atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("view", VIEWS)
def test_model_shapes_native_contract_and_branch_gradients(view):
    model = AlignedDualViewV1(ModelConfig(input_view=view), mamba_factory=ContractMamba)
    value = synthetic_input().requires_grad_()
    features = synthetic_features(requires_grad=True) if model.uses_cwt else None
    output = model(value, cwt=features)
    assert output["waveform"].shape == (1, 1, 18000)
    assert output["waveform_10hz"].shape == (1, 1, 1800)
    output["waveform"].square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    if model.uses_waveform:
        assert value.grad is not None and value.grad.abs().sum() > 0
        assert model.waveform_encoder[0].weight.grad.abs().sum() > 0
    if model.uses_cwt:
        assert features.values.grad is not None and features.values.grad.abs().sum() > 0
        assert model.scale_projection[0].weight.grad.abs().sum() > 0
    assert len(model.blocks) == 6
    for block in model.blocks:
        assert block.dropout.p == 0
        assert block.forward_mamba is not block.backward_mamba
        assert block.forward_mamba.kwargs == {
            "d_model": 64, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32,
            "ngroups": 1, "chunk_size": 256, "rmsnorm": True, "bias": False,
            "conv_bias": True, "use_mem_eff_path": True,
        }


def test_shared_initialization_is_paired_and_rng_is_restored():
    before = torch.get_rng_state()
    models = [AlignedDualViewV1(ModelConfig(input_view=view), mamba_factory=ContractMamba)
              for view in VIEWS]
    assert torch.equal(before, torch.get_rng_state())
    for model in models[1:]:
        for name in ("blocks", "readout"):
            for key, value in getattr(models[0], name).state_dict().items():
                assert torch.equal(value, getattr(model, name).state_dict()[key])
    assert models[1].waveform_encoder[0].out_channels == 64
    assert models[2].scale_projection[0].out_channels == 64
    assert models[3].scale_projection[0].in_channels == 1


def test_scale_mean_control_is_permutation_invariant_but_joint_is_not():
    value, features = synthetic_input(), synthetic_features()
    permutation = CWTBatch(features.values.flip(1), features.representation_id)
    mean_model = AlignedDualViewV1(ModelConfig(input_view="joint_scale_mean"), mamba_factory=ContractMamba)
    joint = AlignedDualViewV1(mamba_factory=ContractMamba)
    with torch.no_grad():
        torch.testing.assert_close(mean_model(value, cwt=features)["waveform"],
                                   mean_model(value, cwt=permutation)["waveform"], atol=1e-6, rtol=1e-5)
        assert not torch.allclose(joint(value, cwt=features)["waveform"],
                                  joint(value, cwt=permutation)["waveform"])


def test_invalid_features_inputs_and_prediction_fail_explicitly():
    model = AlignedDualViewV1(mamba_factory=ContractMamba)
    value, features = synthetic_input(), synthetic_features()
    with pytest.raises(ValueError, match="身份"):
        model(value)
    with pytest.raises(ValueError, match="身份"):
        model(value, cwt=CWTBatch(features.values, "old-cache"))
    with pytest.raises(ValueError, match="2 Hz"):
        model(value, cwt=synthetic_features(length=360))
    with pytest.raises(ValueError, match="float32"):
        model(value, cwt=CWTBatch(features.values.double(), features.representation_id))
    with pytest.raises(ValueError, match="输入期望"):
        model(value[..., :-1], cwt=features)
    value[..., 1] = float("nan")
    with pytest.raises(FloatingPointError, match="输入"):
        model(value, cwt=features)
    value = synthetic_input()
    features.values[..., 0] = float("inf")
    with pytest.raises(FloatingPointError, match="CWT 输入"):
        model(value, cwt=features)
    with torch.no_grad():
        model.readout.bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="prediction"):
        model(value, cwt=synthetic_features())


def test_fixed_grid_and_real_cwt_on_synthetic_signal():
    scales, mapped, order = load_w0_grid()
    assert scales.shape == mapped.shape == order.shape == (97,)
    assert mapped[order][0] == pytest.approx(0.03662109375)
    assert mapped[order][-1] == pytest.approx(7.99560546875)
    original = synthetic_input()
    batch = prepare_cwt_batch(original)
    negative = extract_cwt_10hz(-original[0])
    assert batch.values.shape == (1, 97, 1800)
    assert batch.representation_id == representation_spec()["representation_id"]
    assert torch.isfinite(batch.values).all()
    np.testing.assert_allclose(batch.values[0].numpy(), negative, atol=1e-6, rtol=1e-5)
    # 4 Hz 载波幅度按 0.2 Hz 调制，去掉边界后的幅度轨迹应保持调制时相。
    row = int(np.argmin(np.abs(mapped[order] - 4)))
    envelope = batch.values[0, row, 200:-200].numpy()
    expected = np.sin(2 * np.pi * 0.2 * np.arange(1800)[200:-200] / 10)
    assert np.corrcoef(envelope, expected)[0, 1] > 0.98
    model = AlignedDualViewV1(mamba_factory=ContractMamba)
    assert torch.isfinite(model(original, cwt=batch)["waveform"]).all()


def test_cwt_nonfinite_fails_before_transform():
    value = np.zeros(18000, dtype=np.float32)
    value[0] = np.nan
    with pytest.raises(FloatingPointError, match="CWT 输入"):
        extract_cwt_10hz(value)


def test_model_config_identity_and_strict_fields():
    for view in VIEWS:
        cfg = load_model_config(ROOT / f"configs/aligned_dual_view_v1/{view}.json")
        assert cfg.input_view == view and cfg.dimension == 64 and cfg.depth == 6
    cfg = ModelConfig()
    assert cfg.architecture_id == dataclasses.replace(cfg, initialization_seed=3).architecture_id
    assert cfg.architecture_id != dataclasses.replace(cfg, depth=4).architecture_id
    with pytest.raises(ValueError):
        ModelConfig(dimension=48)
    with pytest.raises(TypeError):
        ModelConfig(depth=2.5)
    with pytest.raises(ValueError):
        ModelConfig(waveform_kernel=50)


def test_existing_loss_accepts_output_and_backpropagates():
    cfg = OmegaConf.create({
        "window": {"target_fs": 100, "duration_samples": 18000},
        "loss": {"band_low_hz": 0.05, "band_high_hz": 0.70,
                 "scale_eps": 1e-8, "dynamic_eps": 1e-8, "corr_eps": 1e-8,
                 "envelope_eps": 1e-8, "max_lag_sec": 0.30,
                 "envelope_window_sec": 10, "envelope_step_sec": 5,
                 "sync_weight": 1, "effort_weight": 0.25},
    })
    model = AlignedDualViewV1(mamba_factory=ContractMamba)
    time = torch.arange(18000) / 100
    target = ((1 + 0.3 * torch.sin(2 * torch.pi * 0.015 * time))
              * torch.sin(2 * torch.pi * 0.2 * time))[None, None]
    loss, parts = RespirationTaskLoss(cfg)(model(synthetic_input(), cwt=synthetic_features()), target)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, parts["loss_sync"] + 0.25 * parts["loss_effort"])
    loss.backward()
    assert model.readout.weight.grad is not None and torch.isfinite(model.readout.weight.grad).all()
