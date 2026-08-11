from __future__ import annotations

import numpy as np

from resp_train.crd.decoder_roundtrip import (
    apply_roundtrip_decision,
    numerical_error_rows,
    roundtrip_target_batch,
    summarize_numerical_errors,
)


def _bandlimited_targets() -> np.ndarray:
    time = np.arange(18000, dtype=np.float64) / 100.0
    signals = []
    for phase in (0.0, 0.37, 1.11):
        signal = (
            np.sin(2.0 * np.pi * 0.20 * time + phase)
            + 0.35 * np.sin(2.0 * np.pi * 0.55 * time - 0.5 * phase)
            + 0.10 * np.cos(2.0 * np.pi * 0.10 * time + 0.2)
        )
        signals.append(signal)
    return np.stack(signals).astype(np.float32)


def test_roundtrip_target_batch_is_negligible_for_formal_bandlimited_signal() -> None:
    reconstructed, target_canonical, reconstructed_canonical = roundtrip_target_batch(_bandlimited_targets())
    assert reconstructed.shape == target_canonical.shape == reconstructed_canonical.shape == (3, 18000)
    rows = numerical_error_rows(target_canonical, reconstructed_canonical)
    audit = summarize_numerical_errors(rows)
    assert audit["max_abs_error"] <= 1e-4
    assert audit["rmse"] <= 1e-5
    assert audit["start_boundary_rmse"] <= 1e-5
    assert audit["end_boundary_rmse"] <= 1e-5


def test_roundtrip_target_batch_rejects_nonfinite_and_wrong_shape() -> None:
    targets = _bandlimited_targets()
    try:
        roundtrip_target_batch(targets[:, :-1])
    except ValueError as exc:
        assert "(B,18000)" in str(exc)
    else:
        raise AssertionError("wrong shape 应失败")
    targets[0, 0] = np.nan
    try:
        roundtrip_target_batch(targets)
    except FloatingPointError as exc:
        assert "NaN/Inf" in str(exc)
    else:
        raise AssertionError("nonfinite target 应失败")


def test_roundtrip_decision_requires_all_frozen_checks() -> None:
    numerical = {
        "n_samples": 2675,
        "max_abs_error": 5e-5,
        "rmse": 5e-6,
        "start_boundary_rmse": 5e-6,
        "end_boundary_rmse": 5e-6,
    }
    summary = {
        "whole_rr_abs_error_bpm_mean": 5e-4,
        "local_rr_mae_bpm_mean": 5e-4,
        "envelope_trajectory_mae_mean": 5e-5,
        "global_envelope_modulation_error_mean": 5e-5,
        "lag_aware_signed_pcc_mean": 0.99995,
        "joint_prediction_degenerate_fraction": 0.0,
    }
    passed = apply_roundtrip_decision(numerical, summary)
    assert passed["roundtrip_negligible"] is True
    assert passed["c2_capacity_placement_framing_required"] is True
    assert passed["c2_resolution_recovery_claim_allowed"] is False

    summary["local_rr_mae_bpm_mean"] = 0.0011
    failed = apply_roundtrip_decision(numerical, summary)
    assert failed["roundtrip_negligible"] is False
    assert failed["checks"]["local_rr"] is False

