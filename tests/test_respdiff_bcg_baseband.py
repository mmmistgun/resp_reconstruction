"""呼吸基带合同：滤波顺序、数学等价、probe 无扰动和三臂合成全链路。"""

import json

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest
from scipy.signal import sosfreqz
import torch

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import (
    OBJECTIVES, ROOT, SCHEMA, SOS, LowpassParents, RespDiffBCGBaseband,
    load_baseband_config, lowpass_parent, resp_spectral_per_sample,
)
from resp_train.respdiff_bcg.baseband_diagnostics import TrainingProbe, waveform_diagnostics
from resp_train.respdiff_bcg.data import ChunkDataset
from resp_train.respdiff_bcg.runtime import train_updates
from resp_train.respdiff_bcg.signal import chunks_from_parent, prepare_parent
from scripts.run_respdiff_bcg_baseband_v1 import fixture, run


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(objective="snr_resp_spectral"):
    return load_baseband_config(ROOT / f"configs/respdiff_bcg_baseband_v1/{objective}.yaml")


def test_full_parent_filter_gain_phase_and_pipeline(tmp_path):
    time = np.arange(18000) / 100
    low = np.sin(2 * np.pi * .2 * time)
    high = np.sin(2 * np.pi * 3 * time)
    filtered = lowpass_parent(low + high)
    np.testing.assert_allclose(filtered[2000:-2000], low[2000:-2000], atol=3e-5)
    _, response = sosfreqz(SOS, worN=[.2, 1., 3.], fs=100)
    assert abs(response[0]) ** 2 > .999
    assert abs(response[1]) ** 2 == pytest.approx(.5)
    assert abs(response[2]) ** 2 < 1e-7
    dataset = fixture(tmp_path, config(), "train")
    raw = dataset.parents.raw[0]
    for key in ("x", "target"):
        expected_parent = lowpass_parent(raw[key].numpy().reshape(-1))
        np.testing.assert_array_equal(dataset.parents[0][key].numpy().reshape(-1), expected_parent)
        expected = chunks_from_parent(prepare_parent(expected_parent))
        actual = torch.stack([dataset[i][key] for i in range(13)]).numpy()
        np.testing.assert_array_equal(actual, expected)
    # 两路相同波形采用完全相同处理，且不能把 600 点 chunk 传给 parent LPF。
    class Identical:
        def __getitem__(self, index):
            value = torch.from_numpy((low + high).astype(np.float32))[None]
            return {"x": value, "target": value, "meta": {}}
    item = LowpassParents(Identical())[0]
    torch.testing.assert_close(item["x"], item["target"], rtol=0, atol=0)
    with pytest.raises(ValueError):
        lowpass_parent(np.zeros(600))
    with pytest.raises(ValueError):
        lowpass_parent(np.full(18000, np.nan))


def test_snr_hann_band_loss_and_gradient_match_direct_all_timesteps():
    generator = torch.Generator().manual_seed(10)
    target = torch.randn(50, 1, 600, generator=generator, dtype=torch.float64)
    error = torch.randn(50, 1, 600, generator=generator, dtype=torch.float64, requires_grad=True)
    alpha = torch.from_numpy(np.cumprod(1 - np.linspace(.0001, .5, 50)))[:, None, None]
    snr = alpha / (1 - alpha)
    window = torch.hann_window(600, periodic=True, dtype=torch.float64)
    bins = torch.arange(301, dtype=torch.float64) * (20 / 600)
    mask = (bins >= .05) & (bins <= .70)
    assert torch.where(mask)[0].tolist() == list(range(2, 22))
    x0 = target + error / snr.sqrt()
    raw = (torch.fft.rfft(x0 * window, norm="ortho").abs()[..., mask]
           - torch.fft.rfft(target * window, norm="ortho").abs()[..., mask]).square().mean((1, 2))
    expected = snr.flatten() * raw
    actual = resp_spectral_per_sample(error, target, alpha)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    left = torch.autograd.grad(actual.mean(), error, retain_graph=True)[0]
    right = torch.autograd.grad(expected.mean(), error)[0]
    torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)
    assert not torch.isclose(actual.mean(), snr.mean() * raw.mean())


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_objective_matches_explicit_formula_and_identical_network(objective):
    torch.manual_seed(12)
    model = RespDiffBCGBaseband(RespDiffSpec(8, 1, 4), objective=objective)
    torch.manual_seed(12)
    reference = RespDiffBCGBaseband(RespDiffSpec(8, 1, 4), objective="epsilon_only")
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    condition, target, noise = (torch.randn(3, 1, 600) for _ in range(3))
    step = torch.tensor([0, 29, 49])
    alpha = model.alpha_torch[step]
    noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    predicted = model.diffusion_model(condition, noisy, step)
    error = noise - predicted
    expected_noise = error.square().mean()
    losses = model.training_loss(condition, target, step=step, noise=noise)
    torch.testing.assert_close(losses["loss_noise"], expected_noise)
    if objective == "source_equivalent":
        x0 = (noisy - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
        raw = (torch.fft.fft(x0, norm="ortho").abs()
               - torch.fft.fft(target, norm="ortho").abs()).square().mean()
        torch.testing.assert_close(losses["loss_spec"], raw)
        torch.testing.assert_close(losses["loss_spec_weighted"], raw * (.01 / 128))
        # 作者 B=128 的总梯度整体除 128 后的相对 balance；不除实际 B=3。
        torch.testing.assert_close(losses["loss"], (128 * expected_noise + .01 * raw) / 128)
    elif objective == "epsilon_only":
        assert float(losses["loss_spec"]) == 0
        torch.testing.assert_close(losses["loss"], expected_noise)
    else:
        torch.testing.assert_close(losses["loss_spec_weighted"], .01 * losses["loss_spec"])
    assert model.last_diagnostics["timestep"] == step.tolist()
    assert model.last_diagnostics["epsilon_residual_max"] >= model.last_diagnostics["epsilon_residual_p999"]


def test_probe_preserves_optimizer_trajectory_rng_and_grad(tmp_path):
    cfg = config()
    dataset = fixture(tmp_path, cfg, "train")
    states = []
    for enabled in (False, True):
        torch.manual_seed(40)
        model = RespDiffBCGBaseband(RespDiffSpec(8, 1, 4))
        output = tmp_path / str(enabled)
        output.mkdir()
        probe = TrainingProbe(model, dataset, cfg, output, tiny=True) if enabled else None
        train_updates(model, dataset, cfg, output, updates=2, batch_size=2, diagnostic_callback=probe)
        states.append(({key: value.clone() for key, value in model.state_dict().items()},
                       [p.grad.clone() for p in model.parameters()], torch.get_rng_state().clone()))
    for key, value in states[0][0].items():
        torch.testing.assert_close(value, states[1][0][key], rtol=0, atol=0)
    for left, right in zip(states[0][1], states[1][1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(states[0][2], states[1][2])


def test_output_ratios_energy_and_zero_denominators():
    time = np.arange(18000) / 100
    wave = np.sin(2 * np.pi * .2 * time) + 2 * np.sin(2 * np.pi * 2 * time)
    stats = waveform_diagnostics(wave)
    assert stats["resp_energy_ratio"] == pytest.approx(.2)
    assert stats["above_1hz_energy_ratio"] == pytest.approx(.8)
    assert stats["top_1pct_sample_energy_ratio"] == pytest.approx(np.sort(wave ** 2)[-180:].sum() / (wave ** 2).sum())
    zero = waveform_diagnostics(np.zeros(18000))
    assert zero["resp_energy_ratio"] is None and zero["zero_centered_energy"]
    assert zero["top_1pct_sample_energy_ratio"] is None and zero["zero_raw_energy"]
    with pytest.raises(ValueError):
        waveform_diagnostics(np.full(18000, np.inf))


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_three_arm_synthetic_run_artifacts(tmp_path, objective):
    cfg = config(objective)
    output = tmp_path / objective
    run("synthetic-smoke", output, cfg)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete" and not receipt["formal_evidence"]
    checkpoint = torch.load(output / "final.pt", weights_only=False)
    assert checkpoint["schema"] == SCHEMA and checkpoint["update"] == 2
    assert checkpoint["experiment"]["objective"]["name"] == objective
    records = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert len(record["timestep"]) == 2
        assert record["loss"] == pytest.approx(record["loss_noise"] + record["loss_spec_weighted"])
    probes = [json.loads(line) for line in (output / "training_probes.jsonl").read_text().splitlines()]
    assert len(probes) == 12
    assert {row["timestep"] for row in probes} == {0, 9, 19, 29, 39, 49}
    assert {row["update"] for row in probes} == {0, 2}
    for row in probes:
        assert row["r_t"] == pytest.approx(row["grad_weighted_spec_norm"] / row["grad_noise_norm"])
        assert row["first_step_x0_max"] >= row["first_step_x0_p999"]
    if objective == "epsilon_only":
        assert all(row["r_t"] == 0 for row in probes)
    summary = json.loads((output / "validation_output_summary.json").read_text())
    assert summary["n_parents"] == 2 and not summary["row_10382"]["present"]
    diag = pd.read_csv(output / "validation_output_diagnostics.csv")
    assert summary["prediction_parent_max_p99"] == pytest.approx(diag.prediction_max_abs.quantile(.99))
    with np.load(output / "validation_waveforms.npz") as payload:
        for index in range(2):
            with np.load(output / f"val_{index}_tho.npz") as raw:
                np.testing.assert_allclose(payload["tho_ref"][index], lowpass_parent(raw["tho"]), atol=1e-6)
    with pytest.raises(FileExistsError):
        run("synthetic-smoke", output, cfg)
    with pytest.raises(ValueError, match="confirm-training"):
        run("train", tmp_path / "forbidden", cfg)


@pytest.mark.parametrize("key,value", [("training.seed", 1), ("objective.spectral_weight", .1),
    ("data.val_split", "test"), ("preprocessing.cutoff_hz", 2), ("diagnostics.timesteps", [49]),
    ("training.max_updates", 100), ("inference.timesteps", [49, 0])])
def test_config_rejects_contract_changes(tmp_path, key, value):
    cfg = config()
    OmegaConf.update(cfg, key, value)
    path = tmp_path / "invalid.yaml"
    OmegaConf.save(cfg, path)
    with pytest.raises(ValueError, match="合同"):
        load_baseband_config(path)
