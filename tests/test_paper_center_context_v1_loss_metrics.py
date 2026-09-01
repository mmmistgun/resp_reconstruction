from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from resp_train.paper_evidence.center_context_config import load_center_context_config
from resp_train.paper_evidence.center_context_loss import CenterContextLoss, pi60_numpy, pi60_torch
from resp_train.paper_evidence.center_context_metrics import (
    center_rr_strictly_improved,
    evaluate_center_predictions,
    summarize_center_metrics,
    validation_center_rr_mean,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper_evidence_v1/center_context_v1.yaml"


def _cfg():
    return load_center_context_config(CONFIG)


def _signal(*, frequency_hz: float = 0.2, modulation: float = 0.25) -> np.ndarray:
    time = np.arange(6000, dtype=np.float64) / 100.0
    envelope = 1.0 + modulation * np.sin(2.0 * np.pi * time / 50.0)
    return envelope * np.sin(2.0 * np.pi * frequency_hz * time)


def _predictions(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "r_tho_hat": pred[None, None].astype(np.float32),
        "tho_ref": target[None, None].astype(np.float32),
        "dataset_row_id": np.asarray([1]),
        "split": np.asarray(["val"]),
        "samp_id": np.asarray([2]),
        "coupling_state_id": np.asarray([3]),
    }


def test_pi60_torch_numpy_equivalence_frequency_scale_finite_and_gradient() -> None:
    signal = _signal().astype(np.float32)
    tensor = torch.tensor(signal).view(1, -1).requires_grad_()
    torch_value = pi60_torch(tensor)
    numpy_value = pi60_numpy(signal[None])
    np.testing.assert_allclose(torch_value.detach().numpy(), numpy_value, rtol=0.0, atol=1e-3)
    assert np.isclose(float(torch_value.square().mean().detach()), 1.0, atol=1e-5)
    torch_value.square().mean().backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert torch.isfinite(torch_value).all()
    bad = signal.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        pi60_numpy(bad)


def test_center_loss_identity_delay_antiphase_effort_and_no_eligible() -> None:
    cfg = _cfg()
    loss_fn = CenterContextLoss(cfg)
    target = torch.from_numpy(_signal().astype(np.float32))[None, None]
    identity, parts = loss_fn(target.clone(), target)
    assert float(identity) < 1e-5
    assert float(parts["loss_sync"]) < 1e-5
    delayed = torch.roll(target, shifts=20, dims=-1)
    delayed_loss, _ = loss_fn(delayed, target)
    assert float(delayed_loss) < 1e-4
    anti_loss, anti_parts = loss_fn(-target, target)
    assert float(anti_parts["loss_sync"]) > 1.9
    assert float(anti_loss) > 1.9
    changed_effort = target * torch.linspace(0.3, 1.7, 6000)[None, None]
    _, effort_parts = loss_fn(changed_effort, target)
    assert float(effort_parts["loss_effort"]) > float(parts["loss_effort"]) + 1e-3
    zero = torch.zeros_like(target)
    empty_loss, empty_parts = loss_fn(zero, zero)
    assert float(empty_loss) == 0.0
    assert int(empty_parts["__count_loss_sync"]) == 0
    assert int(empty_parts["__count_loss_effort"]) == 0


def test_center_metrics_identity_error_ordering_coverage_degradation_and_selector_tie() -> None:
    cfg = _cfg()
    target = _signal()
    identity = evaluate_center_predictions(_predictions(target, target), cfg)
    wrong_rr = evaluate_center_predictions(_predictions(_signal(frequency_hz=0.35), target), cfg)
    degenerate = evaluate_center_predictions(_predictions(np.zeros_like(target), target), cfg)
    assert identity.loc[0, "center_rr_mae_bpm"] < wrong_rr.loc[0, "center_rr_mae_bpm"]
    assert identity.loc[0, "center_lag_aware_signed_pcc"] > wrong_rr.loc[0, "center_lag_aware_signed_pcc"]
    assert degenerate.loc[0, "center_rr_mae_bpm"] == 39.0
    assert degenerate.loc[0, "center_lag_aware_signed_pcc"] == -1.0
    assert degenerate.loc[0, "center_ibi_coverage"] == 0.0
    assert bool(identity.loc[0, "center_ibi_target_eligible"])
    summary = summarize_center_metrics(identity)
    assert summary.loc[0, "n_samples"] == 1
    assert "center_ibi_interpretable_fraction" in summary
    selector = validation_center_rr_mean(_predictions(target, target), cfg)
    assert selector == pytest.approx(identity.loc[0, "center_rr_mae_bpm"])
    assert center_rr_strictly_improved(0.9, 1.0) is True
    assert center_rr_strictly_improved(1.0, 1.0) is False


def test_nonfinite_prediction_fails_entire_center_checkpoint_evaluation() -> None:
    cfg = _cfg()
    target = _signal()
    pred = target.copy()
    pred[10] = np.nan
    with pytest.raises(FloatingPointError, match="整个 checkpoint"):
        evaluate_center_predictions(_predictions(pred, target), cfg)
