"""验证诊断不改变生产六步公式、分项损失及已训练参数。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.model import RespDiffBCG, TIMESTEPS


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_respdiff_bcg_checkpoint.py"
    spec = importlib.util.spec_from_file_location("respdiff_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module
    torch.set_num_threads(previous)


def test_six_step_trace_matches_production(diagnostic):
    model = RespDiffBCG(RespDiffSpec(8, 1, 4)).eval()
    condition, initial = torch.randn(2, 1, 600), torch.randn(2, 1, 600)
    expected = model.sample_chunks(condition, initial).numpy()
    actual, trace, records = diagnostic.trace_ddim(model, condition, initial, TIMESTEPS)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(trace["states"][0], initial.numpy())
    np.testing.assert_array_equal(trace["states"][-1], actual)
    assert len(records) == 6 and records[-1]["next_t"] == -1


def test_probe_matches_training_loss_and_preserves_weights(diagnostic):
    class KnownNoise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.1))

        def forward(self, condition, current, steps):
            return condition * self.scale

    model = RespDiffBCG(RespDiffSpec(8, 1, 4))
    model.diffusion_model = KnownNoise()
    condition, target, noise = [torch.randn(2, 1, 600) for _ in range(3)]
    before = diagnostic.state_digest(model)
    summary, rows, details = diagnostic.probe_timestep(model, condition, target, noise, 39)
    expected = model.training_loss(condition, target, step=torch.full((2,), 39), noise=noise)
    assert summary["loss_noise"] == pytest.approx(float(expected["loss_noise"].detach()), rel=1e-6)
    assert summary["weighted_fft_loss"] == pytest.approx(float(.01 * expected["loss_fft"].detach()), rel=1e-6)
    assert len(rows) == 2 and details["predicted_noise"].shape == (2, 1, 600)
    assert np.isfinite(summary["gradient_norm_ratio"])
    assert diagnostic.state_digest(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_batch_boundaries_and_explicit_access_gate(diagnostic, tmp_path):
    indices = {i: None for batch in diagnostic.BATCHES for i in range(batch * 64, (batch + 1) * 64)}
    assert len(indices) == 192
    assert diagnostic.completed_parents(indices) == list(range(4)) + list(range(872, 881))
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="confirm-validation-diagnostic"):
        diagnostic.run(tmp_path / "missing", output, torch.device("cuda:0"))
    assert not output.exists()
