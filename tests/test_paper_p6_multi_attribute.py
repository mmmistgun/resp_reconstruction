import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from resp_train.metrics.task import (
    compute_log_rms_envelopes,
    compute_target_whole_rr_bpm,
    compute_target_waveform_attributes,
    evaluate_task_predictions,
)
from resp_train.paper_evidence import p6_multi_attribute as p6


def metric_config():
    return OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "loss": {
                "band_low_hz": 0.05,
                "band_high_hz": 0.7,
                "scale_eps": 1e-8,
                "dynamic_eps": 1e-8,
                "corr_eps": 1e-8,
                "envelope_eps": 1e-8,
                "max_lag_sec": 0.3,
                "envelope_window_sec": 10,
                "envelope_step_sec": 5,
            },
            "evaluation": {
                "local_rr_window_sec": 60,
                "local_rr_step_sec": 15,
                "ibi_peak_distance_samples": 142,
                "ibi_match_tolerance_sec": 0.5,
                "ibi_coverage_threshold": 0.8,
                "ndtw_fs": 10,
                "ndtw_radius_sec": 0.3,
                "envelope_strata_low": 0.3,
                "envelope_strata_high": 0.7,
                "envelope_quantile_method": "linear",
            },
        }
    )


def synthetic_signal(freq_hz=0.2):
    time = np.arange(18000, dtype=np.float64) / 100.0
    return np.sin(2 * np.pi * freq_hz * time).astype(np.float32)


def test_target_only_rr_matches_self_prediction_and_joint_attributes():
    cfg = metric_config()
    signal = synthetic_signal()
    rr, eligible = compute_target_whole_rr_bpm(signal[None], cfg)
    joint_rr, joint_eligible, modulation = compute_target_waveform_attributes(signal[None], cfg)
    metrics = evaluate_task_predictions({"r_tho_hat": signal[None], "tho_ref": signal[None]}, cfg)
    assert eligible.tolist() == [True] and joint_eligible.tolist() == [True]
    assert rr[0] == pytest.approx(12.0, abs=1e-5)
    assert joint_rr[0] == pytest.approx(rr[0], abs=1e-12)
    assert metrics.iloc[0]["whole_rr_abs_error_bpm"] == pytest.approx(0.0, abs=1e-12)
    assert modulation.shape == (1,) and np.isfinite(modulation).all()


def test_log_rms_envelope_shape_and_nonfinite_fail_closed():
    cfg = metric_config()
    envelope = compute_log_rms_envelopes(synthetic_signal()[None], cfg)
    assert envelope.shape == (1, 35)
    bad = synthetic_signal()
    bad[0] = np.nan
    with pytest.raises(FloatingPointError):
        compute_log_rms_envelopes(bad[None], cfg)


def test_linear_rr_cutpoints_and_boundary_assignment():
    cutpoints = p6.rr_cutpoints([6.0, 12.0, 18.0, 24.0])
    assert cutpoints == pytest.approx((12.0, 18.0))
    assert p6.assign_rr_strata([6.0, 12.0, 12.1, 18.0, 18.1], cutpoints).tolist() == [
        "low", "low", "medium", "medium", "high"
    ]
    with pytest.raises(ValueError, match="严格递增"):
        p6.rr_cutpoints([12.0, 12.0, 12.0])


def validation_frames(n=8):
    frames = {}
    ids = np.arange(100, 100 + n)
    for seed_index, seed in enumerate(p6.SEEDS):
        frame = pd.DataFrame(
            {
                "evaluation_split": "validation",
                "method": "crd_tf102_w",
                "dataset_row_id": ids,
                "split": "val",
                "samp_id": ids % 3,
                "target_envelope_modulation": np.linspace(0.1, 0.8, n),
                "whole_rr_abs_error_bpm": np.linspace(0.1, 0.8, n) + seed_index * 0.01,
                "local_rr_mae_bpm": np.linspace(0.8, 0.1, n) + seed_index * 0.01,
                "envelope_trajectory_mae": np.linspace(0.2, 0.9, n) + seed_index * 0.01,
                "global_envelope_modulation_error": np.linspace(0.9, 0.2, n) + seed_index * 0.01,
                "lag_aware_signed_pcc": np.linspace(0.7, 0.9, n) - seed_index * 0.001,
            }
        )
        frames[seed] = frame
    return frames


def validation_attributes(n=8):
    ids = np.arange(100, 100 + n)
    return pd.DataFrame(
        {
            "dataset_row_id": ids,
            "split": "val",
            "samp_id": ids % 3,
            "target_sha256": [hashlib.sha256(str(value).encode()).hexdigest() for value in ids],
            "target_rr_bpm": np.linspace(8, 24, n),
            "target_rr_eligible": True,
            "target_envelope_modulation": np.linspace(0.1, 0.8, n),
            "rr_stratum": np.resize(np.asarray(["low", "medium", "high"]), n),
        }
    )


def test_three_seed_aggregation_and_deterministic_distinct_selection():
    aggregate = p6.aggregate_w0_validation_metrics(validation_frames(), expected_count=8)
    contract = p6.load_contract()["waveform_selection"]
    candidates, selected, rule_hash = p6.build_waveform_selection(
        aggregate, validation_attributes(), contract
    )
    assert len(aggregate) == 8
    assert len(candidates) == 8 * 5
    assert selected["category"].tolist() == list(p6.CATEGORIES)
    assert selected["dataset_row_id"].nunique() == 5
    assert "samp_id" not in candidates and "samp_id" not in selected
    assert selected["selection_rule_sha256"].eq(rule_hash).all()
    json.dumps(
        selected[["category", "dataset_row_id", "target_rr_bpm", "rr_stratum"]].to_dict("records"),
        allow_nan=False,
    )
    repeated = p6.build_waveform_selection(aggregate, validation_attributes(), contract)[1]
    assert repeated["dataset_row_id"].tolist() == selected["dataset_row_id"].tolist()


def test_validation_identity_and_target_join_fail_closed():
    frames = validation_frames()
    frames[p6.SEEDS[1]].loc[0, "dataset_row_id"] = 999
    with pytest.raises(ValueError, match="跨 seed"):
        p6.aggregate_w0_validation_metrics(frames, expected_count=8)
    aggregate = p6.aggregate_w0_validation_metrics(validation_frames(), expected_count=8)
    attrs = validation_attributes().iloc[:-1]
    with pytest.raises(ValueError, match="row 集合"):
        p6.build_waveform_selection(aggregate, attrs, p6.load_contract()["waveform_selection"])


def test_target_attribute_hash_and_metadata():
    cfg = metric_config()
    rows = pd.DataFrame({"dataset_row_id": [1, 2], "samp_id": [8, 8]})
    targets = [synthetic_signal(0.2), synthetic_signal(0.25)]
    frame = p6.build_target_attribute_frame(rows, targets, cfg, split="train")
    assert frame["target_rr_bpm"].tolist() == pytest.approx([12.0, 15.0], abs=1e-5)
    assert frame["target_rr_eligible"].all()
    assert frame.loc[0, "target_sha256"] == hashlib.sha256(targets[0].tobytes()).hexdigest()


def test_waveform_panel_cpu_smoke(tmp_path):
    selected = validation_attributes(5).assign(category=list(p6.CATEGORIES))
    signal = np.stack([synthetic_signal(0.15 + i * 0.01) for i in range(5)])
    predictions = np.stack([signal, signal * 0.9, signal * 1.1])
    envelope = compute_log_rms_envelopes(signal, metric_config())
    path = tmp_path / "panel.png"
    p6.render_waveform_panel(
        selected,
        signal,
        signal,
        predictions,
        envelope,
        envelope,
        np.stack([envelope, envelope, envelope]),
        path,
        fs=100,
        center_samples=(6000, 12000),
    )
    assert path.stat().st_size > 0


def test_historical_batch_shape_replay_plan_and_extraction():
    selected_positions = {
        "typical": 1701,
        "rr_difficult": 721,
        "effort_difficult": 2565,
        "rr_effort_inconsistent": 2399,
        "joint_failure": 1257,
    }
    selected = pd.DataFrame(
        {
            "category": list(p6.CATEGORIES),
            "dataset_row_id": [selected_positions[category] for category in p6.CATEGORIES],
        }
    )
    replay = p6.load_contract()["waveform_export"]["numerical_replay"]
    plan, extraction = p6.build_numerical_replay_plan(selected, list(range(2675)), replay)
    assert [group["batch_size"] for group in plan] == [128, 115]
    assert extraction == [37, 81, 133, 95, 105]
    predictions = {
        "dataset_row_id": np.arange(243, dtype=np.int64),
        "r_tho_hat": np.arange(243 * 2, dtype=np.float32).reshape(243, 1, 2),
    }
    sliced = p6._slice_prediction_rows(predictions, extraction)
    assert sliced["dataset_row_id"].tolist() == extraction
    assert sliced["r_tho_hat"].shape == (5, 1, 2)


def test_replay_plan_rejects_historical_slot_drift():
    selected = pd.DataFrame(
        {"category": list(p6.CATEGORIES), "dataset_row_id": [1700, 721, 2565, 2399, 1257]}
    )
    replay = p6.load_contract()["waveform_export"]["numerical_replay"]
    with pytest.raises(RuntimeError, match="shape/slot"):
        p6.build_numerical_replay_plan(selected, list(range(2675)), replay)


def test_anchor_failure_reports_observed_expected_and_delta():
    observed = pd.DataFrame({"dataset_row_id": [1], **{metric: [1.0] for metric in p6.PRIMARY_METRICS}})
    frozen = observed.copy()
    frozen.loc[0, "local_rr_mae_bpm"] = 2.0
    with pytest.raises(RuntimeError, match=r"observed=1.*expected=2.*abs_delta=1"):
        p6._anchor_export_metrics(observed, frozen, atol=1e-6)


def test_existing_output_rejected_before_git_or_source_access(tmp_path, monkeypatch):
    output = tmp_path / p6.TARGET_OUTPUT
    output.mkdir(parents=True)
    sentinel = output / "user.txt"
    sentinel.write_text("保留", encoding="utf-8")
    monkeypatch.setattr(p6, "_require_clean_git", lambda _: pytest.fail("不应检查 Git"))
    with pytest.raises(FileExistsError):
        p6.build_p6_target_attributes(repo_root=tmp_path, command="test")
    assert sentinel.read_text(encoding="utf-8") == "保留"


def test_frozen_contract_and_static_sources_are_validation_only():
    root = Path(__file__).resolve().parents[1]
    contract = p6.load_contract()
    records = p6._audit_static_sources(
        root,
        contract,
        include_metrics=True,
        include_checkpoints=False,
    )
    assert contract["method_allowlist"][5] == "crd_c201_decoder_10hz_cap"
    assert len(contract["method_allowlist"]) == 10
    assert len(records) == 11
    assert not any(record["path"].endswith("research_test_metrics.csv") for record in records)
    assert contract["waveform_export"]["numerical_replay"]["processed_batch_elements_total"] == 729
    if (root / p6.SELECTION_OUTPUT / "artifact_manifest.json").is_file():
        attrs, _ = p6._load_target_attributes(root)
        selected, _ = p6._load_selected_rows(root)
        assert len(attrs) == 2675 and len(selected) == 5


def test_cli_confirmation_and_no_matrix_override():
    root = Path(__file__).resolve().parents[1]
    target = (root / "scripts/build_paper_p6_target_attributes_v1.py").read_text(encoding="utf-8")
    select = (root / "scripts/select_paper_p6_validation_waveforms_v1.py").read_text(encoding="utf-8")
    export = (root / "scripts/export_paper_p6_validation_waveforms_v1.py").read_text(encoding="utf-8")
    assert "--confirm-target-attribute-build" in target
    assert "--confirm-gpu-export" in export and "cuda:<index>" in export
    assert "--split" not in target + select + export
    assert "--checkpoint" not in target + select + export
    assert "--output" not in target + select + export
