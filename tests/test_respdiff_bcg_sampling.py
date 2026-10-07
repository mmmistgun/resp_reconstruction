"""ε 推理协议：旧轨迹兼容、nested 在线均值、DDPM 公式和保幅边界。"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import ROOT, lowpass_parent
from resp_train.respdiff_bcg.inference_v2 import choose_ensemble, select_diagnostic_indices, state_digest
from resp_train.respdiff_bcg.model import RespDiffBCG
from resp_train.respdiff_bcg.runtime import sha256
from resp_train.respdiff_bcg.sampling import (
    SamplerSpec, batch_trajectory_noise, ensemble_prefixes, reconstruct_ensemble_parent,
    sample_trajectory, trajectory_parent_noise,
)
from resp_train.respdiff_bcg.signal import chunks_from_parent, parent_noise
from scripts.run_respdiff_bcg_epsilon_inference_v2 import run


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    return RespDiffBCG(RespDiffSpec(8, 1, 4)).eval()


class ZeroDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = []

    def forward(self, condition, current, steps):
        assert (steps == steps[0]).all()
        self.steps.append(int(steps[0]))
        return torch.zeros_like(current)


def zero_model():
    model = tiny_model()
    model.diffusion_model = ZeroDenoiser()
    return model.eval()


def keys(count=2):
    return [("val", 10382, chunk) for chunk in range(count)]


def test_trajectory0_ddim_N1_matches_history_exactly():
    torch.manual_seed(3)
    model = tiny_model()
    condition = torch.randn(2, 1, 600)
    noise = parent_noise(split="val", row_id=10382, seed=20261003)[:2]
    expected = model.sample_chunks(condition, noise)
    spec = SamplerSpec(n_trajectories=1, postfilter=False)
    actual = sample_trajectory(model, condition, keys(), spec, 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_noise_nested_independence_overlap_and_batch_partition():
    legacy = parent_noise(split="val", row_id=10382, seed=20261003)
    rows = [trajectory_parent_noise(split="val", row_id=10382, trajectory_id=i,
                                    noise_seed=20261003) for i in range(8)]
    torch.testing.assert_close(rows[0], legacy, rtol=0, atol=0)
    for index, left in enumerate(rows):
        torch.testing.assert_close(left[:-1, :, 300:], left[1:, :, :300], rtol=0, atol=0)
        for right in rows[index + 1:]:
            assert not torch.equal(left, right)
    for step in (None, 1, 49):
        batch = batch_trajectory_noise(keys(13), trajectory_id=2, noise_seed=20261003,
                                      device="cpu", reverse_step=step)
        pieces = [batch_trajectory_noise(part, trajectory_id=2, noise_seed=20261003,
                                        device="cpu", reverse_step=step) for part in (keys(13)[:5], keys(13)[5:])]
        torch.testing.assert_close(batch, torch.cat(pieces), rtol=0, atol=0)
    assert not torch.equal(batch_trajectory_noise(keys(), trajectory_id=0, noise_seed=20261003,
        device="cpu", reverse_step=1), batch_trajectory_noise(keys(), trajectory_id=0, noise_seed=20261003,
        device="cpu", reverse_step=2))


def test_online_prefix_means_match_stack_and_N4_is_prefix_of_N8():
    model = zero_model()
    condition = torch.zeros(2, 1, 600)
    spec = SamplerSpec(n_trajectories=8)
    stacked = torch.stack([sample_trajectory(model, condition, keys(), spec, i) for i in range(8)])
    means = dict(ensemble_prefixes(model, condition, keys(), spec, checkpoints=(1, 2, 4, 8)))
    for count, value in means.items():
        torch.testing.assert_close(value, stacked[:count].double().mean(0).float(), rtol=0, atol=0)
        torch.testing.assert_close(value, stacked[:count].mean(0), rtol=2e-6, atol=1e-4)
    smaller = dict(ensemble_prefixes(model, condition, keys(), SamplerSpec(n_trajectories=4)))
    torch.testing.assert_close(smaller[4], means[4], rtol=0, atol=0)


@pytest.mark.parametrize("sampler,nfe", [("ddim", 6), ("ddpm", 50)])
def test_step_sequence_finite_unbounded_and_checkpoint_unchanged(tmp_path, sampler, nfe):
    model = zero_model()
    path = tmp_path / "source.pt"
    torch.save(model.state_dict(), path)
    before = sha256(path), state_digest(model)
    spec = SamplerSpec(sampler=sampler, nfe=nfe, n_trajectories=1)
    result = sample_trajectory(model, torch.zeros(2, 1, 600), keys(), spec, 0)
    assert model.diffusion_model.steps == list(spec.timesteps)
    assert torch.isfinite(result).all() and result.max() > 1 and result.min() < -1
    assert (sha256(path), state_digest(model)) == before
    if sampler == "ddpm":
        current = batch_trajectory_noise(keys(), trajectory_id=0, noise_seed=spec.noise_seed, device="cpu")
        for step in range(49, -1, -1):
            # FP32 在每步舍入；使用公式中的逆系数乘法，以免 50 步放大
            # 除法与乘法求值顺序的舍入差异。
            current = (1 / model.alpha_hat[step].sqrt()) * current
            if step:
                sigma = ((1 - model.alpha[step - 1]) / (1 - model.alpha[step]) * model.beta[step]).sqrt()
                current += sigma * batch_trajectory_noise(keys(), trajectory_id=0, noise_seed=spec.noise_seed,
                                                         device="cpu", reverse_step=step)
        torch.testing.assert_close(result, current, rtol=2e-6, atol=1e-4)


def test_postfilter_after_parent_reconstruction_only():
    time = np.arange(3600) / 20
    parent = np.sin(2 * np.pi * .2 * time) + 2 * np.sin(2 * np.pi * 3 * time)
    chunks = chunks_from_parent(parent)
    low, raw, filtered = reconstruct_ensemble_parent(chunks)
    np.testing.assert_allclose(low, parent, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(filtered, lowpass_parent(raw))
    assert raw.max() > 1 and filtered.max() > .9
    _, unfiltered, identical = reconstruct_ensemble_parent(chunks, postfilter=False)
    np.testing.assert_array_equal(unfiltered, identical)
    with pytest.raises(ValueError):
        lowpass_parent(chunks[0, 0])


def test_fixed_subset_does_not_use_metrics_and_includes_forced_rows():
    rows = pd.DataFrame({"dataset_row_id": list(range(10000, 12675)), "split": "val"})
    chosen = select_diagnostic_indices(rows)
    uniform = set(np.linspace(0, 2674, 64, dtype=int))
    assert set(chosen) == uniform | {382, 2226}
    assert chosen == sorted(chosen)
    with pytest.raises(ValueError, match="缺失"):
        select_diagnostic_indices(rows, forced=(99999,))


def curve():
    return pd.DataFrame({"sampler": "ddim", "N": [1, 2, 4, 8, 16],
        "postfiltered_parent_max_p99": 100., "postfiltered_top_1pct_sample_energy_ratio": .2,
        "postfiltered_whole_rr_abs_error_bpm": 1., "postfiltered_lag_aware_signed_pcc": .5})


def test_plateau_minimum_N2_strict_threshold_and_nonconvergence():
    table = curve()
    assert choose_ensemble(table)["selected_N"] == 2
    table.loc[table.N >= 4, "postfiltered_parent_max_p99"] = 105
    selected = choose_ensemble(table)
    assert not selected["comparisons"][0]["plateau"] and selected["selected_N"] == 4
    table["postfiltered_parent_max_p99"] = [100, 200, 400, 800, 1600]
    selected = choose_ensemble(table)
    assert selected["selected_N"] == 16 and selected["status"] == "ensemble not converged by N=16"
    table.loc[table.N == 2, "postfiltered_parent_max_p99"] = 0
    assert choose_ensemble(table)["comparisons"][0]["relative_changes"]["postfiltered_parent_max_p99"] is None


def test_sampler_rejects_budget_or_semantic_changes():
    for kwargs in ({"sampler": "ddpm", "nfe": 50, "n_trajectories": 8}, {"n_trajectories": 100},
                   {"nfe": 50}, {"eta": 1}, {"noise_seed": -1}):
        with pytest.raises(ValueError):
            SamplerSpec(**kwargs)


def test_synthetic_workflow_outputs_receipt_and_finite_curves(tmp_path):
    output = tmp_path / "smoke"
    run("synthetic-smoke", output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete" and not receipt["formal_evidence"]
    assert len(pd.read_csv(output / "ensemble_curve.csv")) == 5
    assert (output / "metrics_vs_N.png").exists()
    for sampler, count in [("ddim", 4), ("ddpm", 2)]:
        folder = output / f"{sampler}_N{count}"
        for name in ("validation_raw_waveforms.npz", "validation_postfiltered_waveforms.npz",
                     "validation_raw_diagnostics.csv", "validation_postfiltered_diagnostics.csv",
                     "validation_summary.csv"):
            assert (folder / name).exists()
    with pytest.raises(FileExistsError):
        run("synthetic-smoke", output)
    with pytest.raises(ValueError, match="subset-run"):
        run("full-validation", tmp_path / "invalid")
