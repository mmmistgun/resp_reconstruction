"""SNR 修复的数学等价性、混合 timestep 梯度尺度及独立实验身份。"""

import importlib.util
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest
import torch

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.config import ROOT
from resp_train.respdiff_bcg.model import RespDiffBCG
from resp_train.respdiff_bcg.repair import (
    RespDiffBCGLossRepair, SCHEMA, load_repair_config, snr_fft_per_sample,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(objective):
    return load_repair_config(ROOT / f"configs/respdiff_bcg_loss_repair_v1/{objective}.yaml")


def test_stable_fft_matches_explicit_per_sample_snr_and_gradient():
    generator = torch.Generator().manual_seed(1)
    target = torch.randn(50, 1, 600, generator=generator, dtype=torch.float64)
    error = torch.randn(50, 1, 600, generator=generator, dtype=torch.float64, requires_grad=True)
    alpha = torch.from_numpy(np.cumprod(1 - np.linspace(.0001, .5, 50)))[:, None, None]
    snr = alpha / (1 - alpha)
    direct_x0 = target + error / snr.sqrt()
    raw = (torch.fft.fft(direct_x0, norm="ortho").abs()
           - torch.fft.fft(target, norm="ortho").abs()).square().mean((1, 2))
    expected = snr.flatten() * raw
    actual = snr_fft_per_sample(error, target, alpha)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    left = torch.autograd.grad(actual.mean(), error, retain_graph=True)[0]
    right = torch.autograd.grad(expected.mean(), error)[0]
    torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-10)
    assert not torch.isclose(actual.mean(), snr.mean() * raw.mean())


def test_noise_output_gradient_bound_for_all_timesteps():
    target = torch.randn(50, 1, 600, dtype=torch.float64)
    error = torch.randn_like(target, requires_grad=True)
    alpha = torch.from_numpy(np.cumprod(1 - np.linspace(.0001, .5, 50)))[:, None, None]
    spectral = .01 * snr_fft_per_sample(error, target, alpha).mean()
    g_fft = torch.autograd.grad(spectral, error, retain_graph=True)[0]
    g_noise = torch.autograd.grad(error.square().mean(), error)[0]
    ratio = g_fft.flatten(1).norm(dim=1) / g_noise.flatten(1).norm(dim=1)
    assert torch.all(ratio <= .01 + 1e-10)
    torch.testing.assert_close(snr_fft_per_sample(torch.zeros_like(error), target, alpha), torch.zeros(50, dtype=torch.float64))


def test_epsilon_only_matches_manual_mse_and_preserves_network():
    torch.manual_seed(2)
    baseline = RespDiffBCG(RespDiffSpec(8, 1, 4))
    torch.manual_seed(2)
    model = RespDiffBCGLossRepair(RespDiffSpec(8, 1, 4), objective="epsilon_only")
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    condition, target, noise = (torch.randn(2, 1, 600) for _ in range(3))
    step = torch.tensor([0, 49])
    alpha = model.alpha_torch[step]
    predicted = model.diffusion_model(condition, alpha.sqrt() * target + (1 - alpha).sqrt() * noise, step)
    expected = (noise - predicted).square().mean()
    loss = model.training_loss(condition, target, step=step, noise=noise)
    torch.testing.assert_close(loss["loss"], expected)
    assert set(loss) == {"loss", "loss_noise"}


@pytest.mark.parametrize("objective", ["snr_weighted_fft", "epsilon_only"])
def test_repair_synthetic_cli_checkpoint_and_method_identity(tmp_path, objective):
    spec = importlib.util.spec_from_file_location("repair_cli", ROOT / "scripts/run_respdiff_bcg_loss_repair_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = config(objective)
    output = tmp_path / objective
    module.run("synthetic-smoke", output, cfg)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["objective"] == objective
    assert not receipt["formal_evidence"]
    checkpoint = torch.load(output / "final.pt", weights_only=False)
    assert checkpoint["schema"] == SCHEMA and checkpoint["update"] == 2
    assert checkpoint["experiment"]["objective"]["name"] == objective
    frame = pd.read_csv(output / "validation_per_parent.csv")
    assert set(frame.method) == {f"RespDiff-BCG-loss-repair-v1/{objective}"}
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    for row in history:
        weighted = row.get("loss_fft_weighted", 0)
        assert row["loss"] == pytest.approx(row["loss_noise"] + weighted)
        if objective == "snr_weighted_fft":
            assert weighted == pytest.approx(.01 * row["loss_fft_snr"])
    with pytest.raises(FileExistsError):
        module.run("synthetic-smoke", output, cfg)
    with pytest.raises(ValueError, match="confirm-training"):
        module.run("train", tmp_path / "forbidden", cfg)


@pytest.mark.parametrize("key,value", [("training.max_updates", 100), ("objective.fft_weight", .1),
                                      ("data.val_split", "test"), ("inference.timesteps", [49, 0])])
def test_repair_contract_rejects_unrelated_changes(tmp_path, key, value):
    cfg = config("snr_weighted_fft")
    OmegaConf.update(cfg, key, value)
    path = tmp_path / "changed.yaml"
    OmegaConf.save(cfg, path)
    with pytest.raises(ValueError, match="合同"):
        load_repair_config(path)
