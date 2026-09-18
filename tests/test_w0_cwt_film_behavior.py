from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from resp_train.paper_evidence import w0_cwt_film_behavior as film
from resp_train.paper_evidence import w0_cwt_film_behavior_runtime as runtime


class _FakeBranch(nn.Module):
    def forward(self, tf):
        value = tf["w"].mean(dim=1, keepdim=True)
        value = F.interpolate(value, size=film.LATENT_FRAMES, mode="linear", align_corners=False)
        gamma = value.repeat(1, film.CHANNELS, 1) * 0.2
        beta = value.repeat(1, film.CHANNELS, 1) * -0.1
        return gamma, beta


class _FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Conv1d(1, film.CHANNELS, 1, bias=False)
        nn.init.constant_(self.frontend.weight, 0.1)
        self.local_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
        self.local_tcn_blocks = nn.ModuleList()
        self.global_stage = None
        self.representation = None
        self.additional_representation = None
        self.capacity_control = None
        self.refinement = nn.Identity()

    def encode_local(self, x):
        value = F.interpolate(x, size=film.LATENT_FRAMES, mode="linear", align_corners=False)
        tokens = self.frontend(value).transpose(1, 2)
        for block in self.local_blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2)

    def decode_local(self, latent):
        refined = self.refinement(latent)
        waveform_10hz = refined.mean(dim=1, keepdim=True)
        return {
            "waveform_10hz": waveform_10hz,
            "waveform": F.interpolate(
                waveform_10hz, size=18000, mode="linear", align_corners=False
            ),
        }


class _FakeW0(nn.Module):
    tf_variant = "crd_tf102_w"

    def __init__(self):
        super().__init__()
        self.base = _FakeBase()
        self.branches = nn.ModuleDict({"w": _FakeBranch()})
        self.controls = nn.ModuleList()
        self.fusion_gate = None

    def forward(self, x, *, tf):
        z = self.base.encode_local(x)
        gamma_raw, beta_raw = self.branches["w"](tf)
        conditioned = z * (1.0 + 0.5 * torch.tanh(gamma_raw)) + 0.5 * torch.tanh(beta_raw)
        return self.base.decode_local(conditioned)


def _captured(
    *,
    z: float = 1.0,
    gamma_raw: float = math.atanh(0.5),
    beta_raw: float = math.atanh(-0.5),
) -> film.CapturedFilmBatch:
    shape = (1, film.CHANNELS, film.LATENT_FRAMES)
    z_tensor = torch.full(shape, z)
    gamma = torch.full(shape, gamma_raw)
    beta = torch.full(shape, beta_raw)
    g = 0.5 * torch.tanh(gamma)
    b = 0.5 * torch.tanh(beta)
    return film.CapturedFilmBatch(
        z=z_tensor,
        gamma_raw=gamma,
        beta_raw=beta,
        z_prime=z_tensor * (1 + g) + b,
    )


def test_forward_capture_is_tensor_identical_and_hooks_are_removed():
    torch.manual_seed(7)
    model = _FakeW0().eval()
    x = torch.randn(2, 1, 18000)
    w = torch.randn(2, 97, 360)
    with torch.inference_mode():
        expected = model(x, tf={"w": w})
        observed, captured = film.forward_with_capture(model, x, tf={"w": w})
        second = model(x, tf={"w": w})
    for key in expected:
        assert torch.equal(expected[key], observed[key])
        assert torch.equal(expected[key], second[key])
    assert captured.z.shape == (2, film.CHANNELS, film.LATENT_FRAMES)
    assert captured.gamma_raw.shape == captured.beta_raw.shape == captured.z_prime.shape
    assert not model.base.local_blocks[-1]._forward_hooks
    assert not model.branches["w"]._forward_hooks
    assert not model.base.refinement._forward_pre_hooks


def test_forward_capture_rejects_training_or_grad_mode():
    model = _FakeW0()
    x = torch.zeros(1, 1, 18000)
    w = torch.zeros(1, 97, 360)
    with pytest.raises(ValueError, match="model.eval"):
        film.forward_with_capture(model, x, tf={"w": w})
    model.eval()
    with pytest.raises(ValueError, match="inference_mode"):
        film.forward_with_capture(model, x, tf={"w": w})


def test_strength_keeps_scale_and_shift_when_they_cancel():
    stats = film.compute_batch_statistics(_captured())
    assert stats.window["r_scale"][0] == pytest.approx(0.25, abs=1e-6)
    assert stats.window["r_shift"][0] == pytest.approx(0.25, abs=1e-6)
    assert stats.window["r_total"][0] == pytest.approx(0.0, abs=1e-6)
    assert stats.window["closure_max_abs"][0] == pytest.approx(0.0, abs=1e-7)
    assert stats.channel["r_scale"].shape == (1, film.CHANNELS)
    assert stats.time_bin["r_scale"].shape == (1, film.TIME_BIN_COUNT)
    expected_elements = film.CHANNELS * film.LATENT_FRAMES
    assert int(stats.histograms["g_counts"].sum()) == expected_elements
    assert int(stats.histograms["gamma_raw_counts"].sum()) == expected_elements


def test_low_energy_and_positive_negative_edge_statistics():
    shape = (1, film.CHANNELS, film.LATENT_FRAMES)
    gamma = torch.full(shape, 3.0)
    beta = torch.full(shape, -3.0)
    z = torch.zeros(shape)
    g = 0.5 * torch.tanh(gamma)
    b = 0.5 * torch.tanh(beta)
    stats = film.compute_batch_statistics(
        film.CapturedFilmBatch(z=z, gamma_raw=gamma, beta_raw=beta, z_prime=z * (1 + g) + b)
    )
    assert bool(stats.window["denom_small"][0])
    for label in ("090", "095", "099"):
        assert stats.window[f"g_edge_pos_tau_{label}"][0] == pytest.approx(1.0)
        assert stats.window[f"g_edge_neg_tau_{label}"][0] == pytest.approx(0.0)
        assert stats.window[f"b_edge_pos_tau_{label}"][0] == pytest.approx(0.0)
        assert stats.window[f"b_edge_neg_tau_{label}"][0] == pytest.approx(1.0)


def test_nonfinite_capture_fails_closed():
    captured = _captured()
    captured.z[0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="Z"):
        film.compute_batch_statistics(captured)


def _window_frame(count=film.WINDOW_COUNT, seed=film.SEEDS[0]):
    ids = np.arange(count, dtype=np.int64)
    return pd.DataFrame(
        {
            "seed": seed,
            "dataset_row_id": ids,
            "samp_id": ids % 7,
            "split": "val",
            "g_mean_abs": np.full(count, 0.2),
            "g_median_abs": np.full(count, 0.21),
            "b_mean_abs": np.full(count, 0.1),
            "b_median_abs": np.full(count, 0.11),
            "g_time_mean_abs_difference": np.full(count, 0.02),
            "b_time_mean_abs_difference": np.full(count, 0.03),
            "g_legacy_abs_ge_0p49": np.full(count, 0.04),
            "b_legacy_abs_ge_0p49": np.full(count, 0.05),
        }
    )


def test_historical_corrected_film_match_and_drift():
    observed = _window_frame()
    historical = pd.DataFrame(
        {
            "seed": film.SEEDS[0],
            "intervention": "FULL",
            "dataset_row_id": observed.dataset_row_id,
            "samp_id": observed.samp_id,
            "mean_abs_gamma": observed.g_mean_abs,
            "median_abs_gamma": observed.g_median_abs,
            "mean_abs_beta": observed.b_mean_abs,
            "median_abs_beta": observed.b_median_abs,
            "gamma_time_mean_abs_difference": observed.g_time_mean_abs_difference,
            "beta_time_mean_abs_difference": observed.b_time_mean_abs_difference,
            "gamma_saturation_fraction": observed.g_legacy_abs_ge_0p49,
            "beta_saturation_fraction": observed.b_legacy_abs_ge_0p49,
        }
    )
    receipt = film.validate_historical_film(observed, historical, seed=film.SEEDS[0])
    assert receipt["passed"]
    observed.loc[0, "g_mean_abs"] += 2e-6
    with pytest.raises(RuntimeError, match="未复现"):
        film.validate_historical_film(observed, historical, seed=film.SEEDS[0])


def test_join_frozen_metrics_and_independent_quality_metadata():
    count = film.WINDOW_COUNT
    ids = np.arange(count, dtype=np.int64)
    statistics = pd.DataFrame(
        {
            "seed": film.SEEDS[0],
            "dataset_row_id": ids,
            "samp_id": ids % 7,
            "split": "val",
            "r_total": np.linspace(0.0, 1.0, count),
        }
    )
    metrics = pd.DataFrame(
        {
            "dataset_row_id": ids,
            "samp_id": ids % 7,
            "whole_rr_abs_error_bpm": 0.1,
            "local_rr_mae_bpm": 0.2,
            "envelope_trajectory_mae": 0.3,
            "global_envelope_modulation_error": 0.4,
            "lag_aware_signed_pcc": 0.8,
            "target_envelope_modulation": 0.5,
            "envelope_target_stratum": "medium",
            "split": "val",
            "whole_rr_target_eligible": True,
            "local_rr_target_eligible": True,
            "joint_target_eligible": True,
            "envelope_spearman_target_eligible": True,
            "joint_prediction_degenerate": False,
        }
    )
    rows = pd.DataFrame(
        {
            "dataset_row_id": ids,
            "samp_id": ids % 7,
            "waveform_confidence_level": np.where(ids % 2, "high", "medium"),
            "waveform_confidence_score": 0.7,
            "transient_motion_ratio": 0.6,
            "coupling_state_id": 1,
            "window_start_s": ids * 30.0,
            "window_end_s": ids * 30.0 + 180.0,
            "split": "val",
            "usable": True,
        }
    )
    joined = film.join_frozen_sources(statistics, metrics, rows)
    assert len(joined) == count
    assert np.allclose(joined["one_minus_lag_aware_signed_pcc"], 0.2)
    assert set(joined["waveform_confidence_level"]) == {"medium", "high"}
    shuffled = metrics.sample(frac=1.0, random_state=3).reset_index(drop=True)
    joined_shuffled = film.join_frozen_sources(statistics, shuffled, rows)
    pd.testing.assert_frame_equal(joined, joined_shuffled)


def _joined_seed(seed: int, count: int = 12) -> pd.DataFrame:
    ids = np.arange(count, dtype=np.int64)
    base = np.linspace(0.01, 0.99, count)
    frame = pd.DataFrame(
        {
            "seed": seed,
            "dataset_row_id": ids,
            "samp_id": ids % 4,
            "split": "val",
            "r_scale": base + 0.01,
            "r_shift": base[::-1] + 0.01,
            "r_total": base,
            "g_mean_abs": base * 0.4,
            "b_mean_abs": base * 0.3,
            "g_edge_abs_tau_095": base * 0.1,
            "b_edge_abs_tau_095": base * 0.05,
            "z_rms": base + 1,
            "whole_rr_abs_error_bpm": base,
            "local_rr_mae_bpm": base,
            "envelope_trajectory_mae": base,
            "global_envelope_modulation_error": base,
            "one_minus_lag_aware_signed_pcc": base,
            "waveform_confidence_score": 1 - base,
            "transient_motion_ratio": base,
            "waveform_confidence_level": np.where(ids % 2, "high", "medium"),
            "target_envelope_modulation": base,
            "envelope_target_stratum": np.where(ids < count // 2, "low", "high"),
        }
    )
    for column in film.SATURATION_COLUMNS:
        if column not in frame:
            frame[column] = base * 0.01
    return frame


def test_associations_subject_quality_and_seed_summary():
    frame = pd.concat([_joined_seed(seed) for seed in film.SEEDS], ignore_index=True)
    associations = film.association_table(frame)
    match = associations.loc[
        associations["view"].eq("pooled_windows")
        & associations["modulation_variable"].eq("r_total")
        & associations["target_variable"].eq("local_rr_mae_bpm")
    ]
    assert len(match) == 3
    assert np.allclose(match["spearman"], 1.0)
    subjects = film.subject_summary(frame)
    assert len(subjects) == len(film.SEEDS) * 4
    quality = film.quality_group_summary(frame)
    assert set(quality.waveform_confidence_level) == {"medium", "high"}
    seed_summary = film.summarize_seed_variation(
        match,
        value_column="spearman",
        group_columns=["view", "modulation_variable", "target_variable"],
    )
    assert seed_summary.iloc[0]["defined_seed_count"] == 3
    assert seed_summary.iloc[0]["spearman_seed_mean"] == pytest.approx(1.0)


def test_typical_window_selection_is_deterministic_and_deduplicated():
    frame = pd.concat([_joined_seed(seed, count=20) for seed in film.SEEDS], ignore_index=True)
    first = film.select_typical_windows(frame)
    second = film.select_typical_windows(frame.sample(frac=1.0, random_state=4))
    pd.testing.assert_frame_equal(first, second)
    assert set(first["selection_kind"]) == {"modulation_error_quadrant", "quality_level"}
    assert int(first["selected_for_render"].sum()) == first.loc[
        first["selected_for_render"], "dataset_row_id"
    ].nunique()
    empty = first.loc[first["selection_status"].eq("empty_cell")]
    assert set(empty["candidate_count"]) == {0}


def test_attempt_lifecycle_and_manifest_verification(tmp_path):
    lock_hash = "a" * 64
    with runtime.attempt("summary", lock_hash, output_root=tmp_path) as output:
        runtime.write_json(output / "summary_receipt.json", {"status": "ok"})
    manifest = runtime.verify_attempt(output, phase="summary", lock_hash=lock_hash)
    assert manifest["status"] == "completed"
    assert "summary_receipt.json" in manifest["files"]
    (output / "summary_receipt.json").write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="身份漂移"):
        runtime.verify_attempt(output, phase="summary", lock_hash=lock_hash)


def test_summary_figures_and_chinese_conclusion_are_generated(tmp_path):
    frame = pd.concat([_joined_seed(seed, count=20) for seed in film.SEEDS], ignore_index=True)
    frame["denom_small"] = False
    saturation = film.saturation_summary(frame)
    quality = film.quality_group_summary(frame)
    associations = film.association_table(frame)
    histograms = {}
    for name, low, high, bins in (
        ("g", -0.5, 0.5, 200),
        ("b", -0.5, 0.5, 200),
        ("gamma_raw", -8.0, 8.0, 160),
        ("beta_raw", -8.0, 8.0, 160),
    ):
        histograms[f"{name}_edges"] = np.linspace(low, high, bins + 1)
        histograms[f"{name}_counts"] = np.ones(bins, dtype=np.int64)
    runtime._plot_summary(tmp_path, frame, saturation, quality, associations, histograms)
    assert {
        "parameter_distributions.png",
        "saturation_by_sign_threshold.png",
        "modulation_strength_error_association.png",
        "quality_group_comparison.png",
    } == {path.name for path in (tmp_path / "figures").iterdir()}
    frame.to_csv(tmp_path / "window_statistics.csv", index=False)
    saturation.to_csv(tmp_path / "saturation_threshold_summary.csv", index=False)
    associations.to_csv(tmp_path / "modulation_error_associations.csv", index=False)
    quality.to_csv(tmp_path / "quality_group_summary.csv", index=False)
    conclusion = runtime._conclusions_markdown(tmp_path)
    assert "## 观察结果" in conclusion
    assert "## 机制假设" in conclusion
    assert "## 待验证问题" in conclusion


def test_selected_cases_preserve_original_batch_boundaries():
    count = film.WINDOW_COUNT
    data = SimpleNamespace(
        rows=pd.DataFrame({"dataset_row_id": np.arange(count, dtype=np.int64)}),
        dataset=list(range(count)),
        loader=SimpleNamespace(
            collate_fn=lambda samples: {
                "meta": {"dataset_row_id": np.asarray(samples, dtype=np.int64)}
            }
        ),
    )
    batches = runtime._selected_original_batches(data, [5, 130, count - 1])
    assert [len(batch["meta"]["dataset_row_id"]) for batch, _, _ in batches] == [128, 128, 115]
    assert [indices for _, indices, _ in batches] == [[5], [2], [114]]
