from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from resp_train.paper_evidence import w0_film_local_sensitivity as local


class _Branch(nn.Module):
    def forward(self, tf):
        value = tf["w"].mean(dim=1, keepdim=True)
        value = F.interpolate(value, size=1800, mode="linear", align_corners=False)
        return value.repeat(1, 96, 1) * 0.4, value.repeat(1, 96, 1) * -0.3


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
        self.local_tcn_blocks = nn.ModuleList()
        self.global_stage = None
        self.representation = None
        self.additional_representation = None
        self.capacity_control = None
        self.refinement = nn.Identity()

    def encode_local(self, x):
        value = F.interpolate(x, size=1800, mode="linear", align_corners=False)
        tokens = value.repeat(1, 96, 1).transpose(1, 2)
        for block in self.local_blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2)

    def decode_local(self, latent):
        waveform_10hz = self.refinement(latent).mean(dim=1, keepdim=True)
        return {
            "waveform_10hz": waveform_10hz,
            "waveform": F.interpolate(
                waveform_10hz, size=18000, mode="linear", align_corners=False
            ),
        }


class _W0(nn.Module):
    tf_variant = "crd_tf102_w"

    def __init__(self):
        super().__init__()
        self.base = _Base()
        self.branches = nn.ModuleDict({"w": _Branch()})
        self.controls = nn.ModuleList()
        self.fusion_gate = None

    def forward(self, x, *, tf):
        z = self.base.encode_local(x)
        gamma, beta = self.branches["w"](tf)
        gamma_sum = torch.zeros_like(z)
        beta_sum = torch.zeros_like(z)
        gamma_sum = gamma_sum + 0.5 * torch.tanh(gamma)
        beta_sum = beta_sum + 0.5 * torch.tanh(beta)
        return self.base.decode_local(z * (1 + gamma_sum) + beta_sum)


def test_explicit_full_is_identical_to_native():
    torch.manual_seed(3)
    model = _W0().eval()
    x = torch.randn(2, 1, 18000)
    w = torch.randn(2, 97, 360)
    with torch.inference_mode():
        native = model(x, tf={"w": w})
        explicit = local.LocalSensitivityModel(
            model, "FULL", force_explicit_full=True
        )(x, tf={"w": w})
    for key in native:
        assert torch.equal(native[key], explicit[key])


def test_single_forward_fusion_anchor_is_exact():
    torch.manual_seed(4)
    model = _W0().eval()
    x = torch.randn(1, 1, 18000)
    w = torch.randn(1, 97, 360)
    with torch.inference_mode():
        _, captured = local.forward_with_capture(model, x, tf={"w": w})
        reconstructed = local._fuse_w0(
            captured.z,
            captured.gamma_raw,
            captured.beta_raw,
            gamma_coefficient=0.5,
            beta_coefficient=0.5,
        )
    assert torch.equal(reconstructed, captured.z_prime)


def test_gamma_and_beta_coefficients_change_only_the_intended_path():
    model = _W0().eval()
    x = torch.ones(1, 1, 18000)
    w = torch.ones(1, 97, 360)
    with torch.inference_mode():
        outputs = {
            condition: local.LocalSensitivityModel(model, condition)(x, tf={"w": w})[
                "waveform_10hz"
            ]
            for condition in local.CONDITIONS
        }
    z = torch.ones(1, 1, 1800)
    gamma = torch.full_like(z, 0.4)
    beta = torch.full_like(z, -0.3)
    for condition, (gamma_coefficient, beta_coefficient) in local.COEFFICIENTS.items():
        expected = z * (1 + gamma_coefficient * torch.tanh(gamma)) + beta_coefficient * torch.tanh(beta)
        assert torch.allclose(outputs[condition], expected)
    assert float(1 - 0.6 * math.tanh(0.4)) > 0.0


def _condition_frame(seed=1):
    count = local.WINDOW_COUNT
    ids = np.arange(count, dtype=np.int64)
    frames = []
    offsets = {
        "FULL": 0.0,
        "GAMMA_040": -0.1,
        "GAMMA_060": 0.1,
        "BETA_040": -0.2,
        "BETA_060": 0.2,
    }
    for condition in local.CONDITIONS:
        value = 1.0 + ids / count + offsets[condition]
        frame = pd.DataFrame(
            {
                "seed": seed,
                "condition": condition,
                "dataset_row_id": ids,
                "samp_id": ids % 7,
                "split": "val",
                "whole_rr_abs_error_bpm": value,
                "local_rr_mae_bpm": value,
                "envelope_trajectory_mae": value,
                "global_envelope_modulation_error": value,
                "one_minus_lag_aware_signed_pcc": value,
                "lag_aware_signed_pcc": 1.0 - value,
                "envelope_target_stratum": np.resize(["low", "medium", "high"], count),
                "waveform_confidence_level": np.where(ids % 2, "high", "medium"),
                "waveform_confidence_score": 0.7,
                "transient_motion_ratio": 0.4,
                "coupling_state_id": ids % 3,
                "window_start_s": ids * 30.0,
                "window_end_s": ids * 30.0 + 180.0,
                "whole_rr_target_eligible": True,
                "local_rr_target_eligible": True,
                "local_rr_target_eligible_windows": 9,
                "joint_target_eligible": True,
                "envelope_spearman_target_eligible": True,
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_paired_deltas_strata_and_local_response():
    matrix = _condition_frame()
    local.validate_condition_matrix(matrix, seeds=(1,))
    paired = local.build_paired_deltas(matrix)
    assert len(paired) == len(local.NEW_CONDITIONS) * local.WINDOW_COUNT
    summary = local.summarize_paired_deltas(paired)
    response = local.local_response(summary)
    gamma = response.loc[
        response["path"].eq("gamma")
        & response["scope"].eq("all")
        & response["metric"].eq("local_rr_mae_bpm")
    ].iloc[0]
    beta = response.loc[
        response["path"].eq("beta")
        & response["scope"].eq("all")
        & response["metric"].eq("local_rr_mae_bpm")
    ].iloc[0]
    assert gamma["central_slope"] == pytest.approx(1.0)
    assert gamma["local_curvature"] == pytest.approx(0.0)
    assert beta["central_slope"] == pytest.approx(2.0)
    assert beta["local_curvature"] == pytest.approx(0.0)
    assert {"all", "target_stratum", "quality_level", "target_quality", "samp_id"} == set(
        summary["scope"]
    )


def test_matrix_missing_condition_and_eligibility_drift_fail():
    matrix = _condition_frame()
    missing = matrix.loc[~matrix["condition"].eq("BETA_060")]
    with pytest.raises(ValueError, match="矩阵不完整"):
        local.validate_condition_matrix(missing, seeds=(1,))
    matrix.loc[
        (matrix["condition"].eq("GAMMA_040")) & (matrix["dataset_row_id"].eq(0)),
        "joint_target_eligible",
    ] = False
    with pytest.raises(ValueError, match="eligibility"):
        local.build_paired_deltas(matrix)


def test_error_aligned_pcc_and_nonfinite_rejection():
    frame = pd.DataFrame(
        {
            "whole_rr_abs_error_bpm": [1.0],
            "local_rr_mae_bpm": [2.0],
            "envelope_trajectory_mae": [3.0],
            "global_envelope_modulation_error": [4.0],
            "lag_aware_signed_pcc": [0.75],
        }
    )
    aligned = local.error_aligned_metrics(frame)
    assert aligned["one_minus_lag_aware_signed_pcc"].iloc[0] == pytest.approx(0.25)
    frame.loc[0, "local_rr_mae_bpm"] = np.inf
    with pytest.raises(FloatingPointError):
        local.error_aligned_metrics(frame)


def test_attempt_lifecycle_is_hashed(tmp_path):
    lock_hash = "b" * 64
    with local.attempt("summary", lock_hash, output_root=tmp_path) as output:
        local.write_json(output / "matrix_receipt.json", {"status": "ok"})
    manifest = local.verify_attempt(output, phase="summary", lock_hash=lock_hash)
    assert manifest["status"] == "completed"
    (output / "matrix_receipt.json").write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="身份漂移"):
        local.verify_attempt(output, phase="summary", lock_hash=lock_hash)


def test_three_seed_summary_plots_and_conclusion(tmp_path):
    matrix = pd.concat([_condition_frame(seed) for seed in local.SEEDS], ignore_index=True)
    paired = local.build_paired_deltas(matrix)
    strata = local.summarize_paired_deltas(paired)
    response = local.local_response(strata)
    seeds = local.seed_summary(strata)
    local._plot_summary(tmp_path, response, seeds)
    assert {
        "coefficient_response.png",
        "high_stratum_response.png",
        "quality_response.png",
    } == {path.name for path in (tmp_path / "figures").iterdir()}
    response.to_csv(tmp_path / "local_response.csv", index=False)
    seeds.to_csv(tmp_path / "seed_summary.csv", index=False)
    conclusion = local._conclusion(tmp_path)
    assert "## 观察结果" in conclusion
    assert "high target-modulation" in conclusion
