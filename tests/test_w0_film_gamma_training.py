from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from resp_train.crd.model import build_crd_model
from resp_train.crd.tf_v1_model import CRDTfV1Model, trainable_parameter_count
from resp_train.crd.w0_film_gamma_training import (
    BETA_COEFFICIENT,
    CONDITIONS,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORMAL_UPDATES,
    PROTOCOL,
    TRAINABLE_PARAMETERS,
    _candidate_decisions,
    _paired_window_rows,
    _primary_values_finite,
    load_resolved_training_config,
    load_training_config,
    model_contract,
    validate_training_config,
)


class _ConstantBranch(nn.Module):
    def forward(self, tf):
        batch = int(tf["w"].shape[0])
        gamma = torch.full((batch, 96, 1800), 0.4, dtype=tf["w"].dtype)
        beta = torch.full((batch, 96, 1800), -0.3, dtype=tf["w"].dtype)
        return gamma, beta


class _FastBase(nn.Module):
    def encode_local(self, x):
        return torch.ones(x.shape[0], 96, 1800, dtype=x.dtype)

    def decode_local(self, latent):
        return {"waveform": latent, "waveform_10hz": latent[:, :1]}


def _fast_model(gamma: float) -> CRDTfV1Model:
    model = CRDTfV1Model(
        "crd_tf102_w",
        20260811,
        gamma_coefficient=gamma,
        beta_coefficient=0.5,
    )
    model.base = _FastBase()
    model.branches["w"] = _ConstantBranch()
    return model


def test_gamma_coefficients_change_only_scale_path() -> None:
    x = torch.zeros(1, 1, 18000)
    w = torch.zeros(1, 97, 360)
    outputs = {
        gamma: _fast_model(gamma)(x, tf={"w": w})["waveform"]
        for gamma in (0.3, 0.4, 0.5)
    }
    beta = 0.5 * torch.tanh(torch.tensor(-0.3))
    for gamma, output in outputs.items():
        expected = 1.0 + gamma * torch.tanh(torch.tensor(0.4)) + beta
        torch.testing.assert_close(output, torch.full_like(output, expected))
    assert float(outputs[0.3].mean()) < float(outputs[0.4].mean()) < float(outputs[0.5].mean())


def test_default_w0_remains_exactly_the_05_model() -> None:
    default = CRDTfV1Model("crd_tf102_w", 20260811)
    explicit = CRDTfV1Model(
        "crd_tf102_w",
        20260811,
        gamma_coefficient=0.5,
        beta_coefficient=0.5,
    )
    assert trainable_parameter_count(default) == trainable_parameter_count(explicit) == TRAINABLE_PARAMETERS
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0.0, atol=0.0)
    assert default.gamma_coefficient == explicit.gamma_coefficient == 0.5
    assert default.beta_coefficient == explicit.beta_coefficient == 0.5


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_model_rejects_invalid_coefficients(value: float) -> None:
    with pytest.raises(ValueError, match="coefficient"):
        CRDTfV1Model("crd_tf102_w", 20260811, gamma_coefficient=value)
    with pytest.raises(ValueError, match="coefficient"):
        CRDTfV1Model("crd_tf102_w", 20260811, beta_coefficient=value)


def test_acceptance_and_formal_configs_are_fixed() -> None:
    acceptance = load_training_config(role="acceptance", condition="GAMMA_030")
    assert str(acceptance.protocol.name) == PROTOCOL
    assert str(acceptance.protocol.execution_gate) == "film_gamma_cuda_acceptance"
    assert (acceptance.training.epochs, acceptance.training.batch_size) == (1, 128)
    assert (acceptance.data.max_train_windows, acceptance.data.max_val_windows) == (128, 32)
    assert float(acceptance.model.film_gamma_coefficient) == CONDITIONS["GAMMA_030"]
    assert float(acceptance.model.film_beta_coefficient) == BETA_COEFFICIENT

    for condition, coefficient in CONDITIONS.items():
        for seed in FORMAL_SEEDS:
            cfg = load_training_config(role="formal", condition=condition, seed=seed)
            assert (cfg.training.epochs, cfg.training.batch_size) == (FORMAL_EPOCHS, 128)
            assert cfg.training.early_stopping_enabled is False
            assert int(cfg.training.seed) == int(cfg.model.initialization_seed) == seed
            assert float(cfg.model.film_gamma_coefficient) == coefficient
            assert float(cfg.model.film_beta_coefficient) == BETA_COEFFICIENT
            token = condition.lower().replace("_", "")
            assert str(cfg.outputs.run_root).endswith(f"/{token}/seed_{seed}")


def test_config_rejects_coefficient_and_identity_drift() -> None:
    cfg = load_training_config(role="formal", condition="GAMMA_030", seed=20260811)
    bad = deepcopy(cfg)
    bad.model.film_gamma_coefficient = 0.31
    with pytest.raises(ValueError, match="coefficient"):
        validate_training_config(bad)
    bad = deepcopy(cfg)
    bad.model.film_beta_coefficient = 0.4
    with pytest.raises(ValueError, match="coefficient"):
        validate_training_config(bad)
    bad = deepcopy(cfg)
    bad.outputs.run_root = "/tmp/wrong"
    with pytest.raises(ValueError, match="output root"):
        validate_training_config(bad)


def test_model_contract_records_training_identity() -> None:
    cfg = load_training_config(role="formal", condition="GAMMA_040", seed=20260813)
    contract = model_contract(cfg)
    assert contract == {
        "protocol": PROTOCOL,
        "condition": "GAMMA_040",
        "variant": "crd_tf102_w",
        "gamma_coefficient": 0.4,
        "beta_coefficient": 0.5,
        "seed": 20260813,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "research_test_used": False,
    }
    assert FORMAL_UPDATES == 6400


def test_resolved_config_replays_coefficients_through_model_factory(tmp_path) -> None:
    cfg = load_training_config(role="formal", condition="GAMMA_030", seed=20260812)
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path)
    replay = load_resolved_training_config(path)
    model = build_crd_model(replay)
    assert isinstance(model, CRDTfV1Model)
    assert model.gamma_coefficient == 0.3
    assert model.beta_coefficient == 0.5
    assert trainable_parameter_count(model) == TRAINABLE_PARAMETERS


def test_model_factory_rejects_partial_coefficient_config() -> None:
    cfg = load_training_config(role="formal", condition="GAMMA_030", seed=20260811)
    del cfg.model.film_beta_coefficient
    with pytest.raises(ValueError, match="同时提供"):
        build_crd_model(cfg)


def test_nondefault_coefficients_are_w0_only() -> None:
    with pytest.raises(ValueError, match="只为 W0"):
        CRDTfV1Model("crd_tf101_m", 20260811, gamma_coefficient=0.4)


def test_primary_finite_respects_target_eligibility() -> None:
    frame = pd.DataFrame(
        {
            "whole_rr_abs_error_bpm": [1.0, np.nan],
            "whole_rr_target_eligible": [True, False],
            "local_rr_mae_bpm": [2.0, np.nan],
            "local_rr_target_eligible": [True, False],
            "lag_aware_signed_pcc": [0.5, np.nan],
            "joint_target_eligible": [True, False],
            "envelope_trajectory_mae": [0.1, 0.2],
            "global_envelope_modulation_error": [0.3, 0.4],
        }
    )
    assert _primary_values_finite(frame)
    frame.loc[0, "local_rr_mae_bpm"] = np.inf
    assert not _primary_values_finite(frame)


def test_paired_rows_require_identity_and_preserve_strata() -> None:
    base = pd.DataFrame(
        {
            "dataset_row_id": [1, 2],
            "samp_id": [10, 11],
            "split": ["val", "val"],
            "whole_rr_abs_error_bpm": [1.0, np.nan],
            "whole_rr_target_eligible": [True, False],
            "local_rr_mae_bpm": [2.0, 3.0],
            "local_rr_target_eligible": [True, True],
            "envelope_trajectory_mae": [0.1, 0.2],
            "global_envelope_modulation_error": [0.3, 0.4],
            "lag_aware_signed_pcc": [0.5, 0.6],
            "joint_target_eligible": [True, True],
            "envelope_target_stratum": ["low", "high"],
        }
    )
    candidate = base.copy()
    candidate["local_rr_mae_bpm"] -= 0.1
    quality = base[["dataset_row_id", "samp_id"]].copy()
    quality["waveform_confidence_level"] = ["high", "medium"]
    paired = _paired_window_rows(
        condition="GAMMA_030",
        seed=20260811,
        candidate=candidate,
        baseline=base,
        quality=quality,
    )
    local = paired.loc[paired["metric"].eq("local_rr_mae_bpm")]
    assert len(paired) == 9
    assert local["delta"].tolist() == pytest.approx([-0.1, -0.1])
    assert local["target_stratum"].tolist() == ["low", "high"]
    assert local["quality_level"].tolist() == ["high", "medium"]
    bad = candidate.copy()
    bad.loc[0, "dataset_row_id"] = 99
    with pytest.raises(RuntimeError, match="identity"):
        _paired_window_rows(
            condition="GAMMA_030",
            seed=20260811,
            candidate=bad,
            baseline=base,
            quality=quality,
        )


def test_candidate_decision_applies_preregistered_guardrails() -> None:
    rows = []
    for condition in CONDITIONS:
        for metric in (
            "whole_rr_abs_error_bpm",
            "local_rr_mae_bpm",
            "envelope_trajectory_mae",
            "global_envelope_modulation_error",
            "lag_aware_signed_pcc",
        ):
            relative = -1.0 if metric == "local_rr_mae_bpm" else 0.0
            delta = 0.003 if metric == "lag_aware_signed_pcc" else -0.01
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "baseline_seed_mean": 1.0,
                    "candidate_seed_mean": 1.0 + delta,
                    "delta_seed_mean": delta,
                    "delta_seed_sd": 0.0,
                    "relative_delta_percent": relative if metric != "lag_aware_signed_pcc" else np.nan,
                    "improve_seed_count": 3,
                    "worse_seed_count": 0,
                }
            )
    decisions = _candidate_decisions(pd.DataFrame(rows))
    assert all(row["quality_candidate"] for row in decisions)
    assert all(row["tolerance_aware_pareto"] for row in decisions)

    bad = pd.DataFrame(rows)
    bad.loc[
        bad["condition"].eq("GAMMA_030")
        & bad["metric"].eq("global_envelope_modulation_error"),
        "relative_delta_percent",
    ] = 4.0
    decision = next(row for row in _candidate_decisions(bad) if row["condition"] == "GAMMA_030")
    assert decision["catastrophic_failure"] is True
    assert decision["quality_candidate"] is False
