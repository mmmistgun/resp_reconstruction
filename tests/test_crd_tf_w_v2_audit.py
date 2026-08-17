from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from resp_train.crd.tf_v1_model import CRDTfV1Model
from resp_train.crd.tf_w_v2_audit import (
    CANDIDATE_LOCK,
    CANDIDATE_LOCK_SHA256,
    AuditedW0Model,
    _load_candidate_lock,
    _sha256_file,
    _validate_audit_split,
    apply_w_intervention,
    decide_p2_fusion,
)


class _FakeBase(nn.Module):
    def encode_local(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode_local(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"waveform": latent}


class _FakeWBranch(nn.Module):
    def forward(self, tf: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        value = tf["w"]
        pooled = value.mean(dim=1)
        gamma = pooled[:, :3].unsqueeze(1).expand(-1, 2, -1)
        beta = (2.0 * pooled[:, :3]).unsqueeze(1).expand(-1, 2, -1)
        return gamma, beta


class _FakeW0(nn.Module):
    tf_variant = "crd_tf102_w"

    def __init__(self) -> None:
        super().__init__()
        self.base = _FakeBase()
        self.branches = nn.ModuleDict({"w": _FakeWBranch()})
        self.controls = nn.ModuleList()
        self.fusion_gate = None

    def forward(self, x: torch.Tensor, *, tf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gamma, beta = self.branches["w"](tf)
        effective_gamma = 0.5 * torch.tanh(gamma)
        effective_beta = 0.5 * torch.tanh(beta)
        return self.base.decode_local(x * (1.0 + effective_gamma) + effective_beta)


def test_candidate_lock_identity_and_p0_boundary() -> None:
    path, lock = _load_candidate_lock(CANDIDATE_LOCK)

    assert _sha256_file(path) == CANDIDATE_LOCK_SHA256
    assert lock["status"] == "p0_complete_downstream_implementation_not_activated"
    assert lock["forbidden"]["samp_id_analysis"] is True
    assert lock["p0_outcome"]["research_test_opened"] is False


def test_w_interventions_are_exact_and_do_not_mutate_source() -> None:
    value = torch.arange(2 * 97 * 360, dtype=torch.float32).reshape(2, 97, 360)
    source = value.clone()

    full = apply_w_intervention(value, "FULL")
    resp = apply_w_intervention(value, "RESP")
    carrier = apply_w_intervention(value, "CARRIER")
    carrier_l = apply_w_intervention(value, "CARRIER_L")
    carrier_h = apply_w_intervention(value, "CARRIER_H")
    time_mean = apply_w_intervention(value, "TIME_MEAN")
    time_shift = apply_w_intervention(value, "TIME_SHIFT_30S")

    assert full.data_ptr() == value.data_ptr()
    torch.testing.assert_close(resp[:, :56], value[:, :56])
    torch.testing.assert_close(resp[:, 56:], torch.zeros_like(resp[:, 56:]))
    torch.testing.assert_close(carrier[:, :56], torch.zeros_like(carrier[:, :56]))
    torch.testing.assert_close(carrier[:, 56:], value[:, 56:])
    torch.testing.assert_close(carrier_l[:, 56:72], value[:, 56:72])
    assert torch.count_nonzero(carrier_l[:, :56]) == 0
    assert torch.count_nonzero(carrier_l[:, 72:]) == 0
    torch.testing.assert_close(carrier_h[:, 72:], value[:, 72:])
    assert torch.count_nonzero(carrier_h[:, :72]) == 0
    torch.testing.assert_close(time_mean, value.mean(dim=-1, keepdim=True).expand_as(value))
    torch.testing.assert_close(time_shift, torch.roll(value, shifts=60, dims=-1))
    torch.testing.assert_close(value, source, rtol=0.0, atol=0.0)


def test_w_intervention_rejects_unknown_name_and_shape() -> None:
    with pytest.raises(ValueError, match="未知 P−1"):
        apply_w_intervention(torch.zeros(1, 97, 360), "FULL_6V")
    with pytest.raises(ValueError, match="期望"):
        apply_w_intervention(torch.zeros(1, 49, 360), "FULL")


def test_full_wrapper_is_exact_and_film_modes_match_formula() -> None:
    model = _FakeW0().eval()
    x = torch.linspace(-1.0, 1.0, 12).reshape(2, 2, 3)
    w = torch.linspace(-0.5, 0.5, 2 * 97 * 360).reshape(2, 97, 360)
    gamma_raw, beta_raw = model.branches["w"]({"w": w})
    gamma = 0.5 * torch.tanh(gamma_raw)
    beta = 0.5 * torch.tanh(beta_raw)

    expected_full = model(x, tf={"w": w})["waveform"]
    observed_full = AuditedW0Model(model, intervention="FULL")(x, tf={"w": w})["waveform"]
    beta_only = AuditedW0Model(model, intervention="BETA_ONLY")(x, tf={"w": w})["waveform"]
    gamma_only = AuditedW0Model(model, intervention="GAMMA_ONLY")(x, tf={"w": w})["waveform"]
    off = AuditedW0Model(model, intervention="CONDITION_OFF")(x, tf={"w": w})["waveform"]

    torch.testing.assert_close(observed_full, expected_full, rtol=0.0, atol=0.0)
    torch.testing.assert_close(beta_only, x + beta, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma_only, x * (1.0 + gamma), rtol=0.0, atol=0.0)
    torch.testing.assert_close(off, x, rtol=0.0, atol=0.0)


def test_full_wrapper_matches_real_w0_forward_with_active_film() -> None:
    model = CRDTfV1Model("crd_tf102_w", 20260811).eval()
    model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in model.base.local_blocks])
    with torch.no_grad():
        model.branches["w"].final_projection.weight.normal_(std=1e-4)
        model.branches["w"].final_projection.bias.normal_(std=1e-4)
    x = torch.randn(1, 1, 18000)
    w = torch.randn(1, 97, 360)

    with torch.no_grad():
        expected = model(x, tf={"w": w})
        observed = AuditedW0Model(model, intervention="FULL")(x, tf={"w": w})

    assert observed.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(observed[key], expected[key], rtol=0.0, atol=0.0)


def test_full_wrapper_records_compact_per_sample_statistics_once() -> None:
    model = _FakeW0().eval()
    wrapper = AuditedW0Model(model, intervention="FULL", record_film_statistics=True)
    x = torch.ones(2, 2, 3)
    w = torch.randn(2, 97, 360)

    wrapper(x, tf={"w": w})
    statistics = wrapper.take_film_statistics()

    assert set(statistics) == {
        "mean_abs_gamma",
        "median_abs_gamma",
        "mean_abs_beta",
        "median_abs_beta",
        "gamma_time_mean_abs_difference",
        "beta_time_mean_abs_difference",
        "gamma_saturation_fraction",
        "beta_saturation_fraction",
    }
    assert all(value.shape == (2,) for value in statistics.values())
    assert all(np.isfinite(value).all() for value in statistics.values())
    assert wrapper.take_film_statistics() == {}


def _seed_summaries(*, beta: tuple[float, float], gamma: tuple[float, float]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in (20260811, 20260812, 20260813):
        rows.append(_summary_row(seed, "FULL", error=1.0, pcc=0.800))
        rows.append(_summary_row(seed, "BETA_ONLY", error=beta[0], pcc=beta[1]))
        rows.append(_summary_row(seed, "GAMMA_ONLY", error=gamma[0], pcc=gamma[1]))
    return pd.DataFrame(rows)


def _summary_row(seed: int, intervention: str, *, error: float, pcc: float) -> dict[str, float | int | str]:
    return {
        "seed": seed,
        "intervention": intervention,
        "whole_rr_abs_error_bpm_mean": error,
        "local_rr_mae_bpm_mean": error,
        "envelope_trajectory_mae_mean": error,
        "global_envelope_modulation_error_mean": error,
        "lag_aware_signed_pcc_mean": pcc,
        "joint_prediction_degenerate_fraction": 0.0,
    }


@pytest.mark.parametrize(
    ("beta", "gamma", "decision", "variant", "both_near"),
    (
        ((1.005, 0.798), (1.03, 0.790), "train_add", "crd_tfw_v2_f1_add_full_12v_d6", False),
        ((1.03, 0.790), (1.005, 0.798), "train_scale", "crd_tfw_v2_f2_scale_full_12v_d6", False),
        ((1.005, 0.798), (1.004, 0.799), "train_add", "crd_tfw_v2_f1_add_full_12v_d6", True),
        ((1.03, 0.790), (1.03, 0.790), "retain_film_no_p2_training", None, False),
    ),
)
def test_p2_decision_is_frozen_and_never_selects_two_arms(
    beta: tuple[float, float],
    gamma: tuple[float, float],
    decision: str,
    variant: str | None,
    both_near: bool,
) -> None:
    output = decide_p2_fusion(_seed_summaries(beta=beta, gamma=gamma))

    assert output["decision"] == decision
    assert output["selected_p2_variant"] == variant
    assert output["both_near_prefers_add"] is both_near
    assert output["samp_id_analysis_used"] is False
    assert output["research_test_used"] is False


def test_p2_decision_requires_all_fixed_seeds() -> None:
    frame = _seed_summaries(beta=(1.0, 0.8), gamma=(1.0, 0.8))
    frame = frame.loc[~((frame["seed"] == 20260813) & (frame["intervention"] == "BETA_ONLY"))]

    with pytest.raises(ValueError, match="三个固定 seed"):
        decide_p2_fusion(frame)


def test_p_minus_1_rejects_any_non_validation_split() -> None:
    _validate_audit_split("val")
    with pytest.raises(ValueError, match="只允许 split=val"):
        _validate_audit_split("test")
