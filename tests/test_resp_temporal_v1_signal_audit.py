from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest

import resp_train.temporal_signal_audit as signal_audit
from resp_train.metrics.task import TaskMetricConfig
from resp_train.temporal_signal_audit import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    PROTOCOL_ID,
    RECEIPT_SCHEMA_VERSION,
    SignalAuditConfig,
    audit_signal_window,
    above_nyquist_power_fraction,
    boxcar_decimate,
    build_carrier_sampling_paths,
    fft_band_extract,
    load_signal_audit_config,
    lowpass_block_center_decimate,
    reconstruct_block_center_grid,
    sample_direct_mean,
    target_timescale_metrics,
    validate_access_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _metric_config() -> TaskMetricConfig:
    return TaskMetricConfig.from_config(OmegaConf.load(REPO_ROOT / "configs/tho_research_v2.yaml"))


def _valid_receipt() -> dict:
    column_counts = {"n_total": 1, "n_finite": 1, "n_null": 0, "n_infinite": 0}
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_utc": "2026-08-20T00:00:00+00:00",
        "execution": {
            "command": "python scripts/audit_resp_temporal_v1_signal.py",
            "cwd": str(REPO_ROOT),
            "git_commit": "0" * 40,
            "git_dirty": False,
            "python_version": "3.12.0",
            "platform": "test",
            "dependencies": {"numpy": "test"},
            "audit_config_path": str(DEFAULT_CONFIG_PATH),
            "audit_config_sha256": "a" * 64,
            "source_config_path": str(REPO_ROOT / "configs/tho_research_v2.yaml"),
            "source_config_sha256": "b" * 64,
        },
        "data": {
            "dataset_root": "/dataset",
            "shared_index_path": "/dataset/training/dataset_index.csv",
            "shared_index_sha256": "c" * 64,
            "selected_dataset_row_ids_sha256": "d" * 64,
            "selected_input_content_sha256": "e" * 64,
            "selected_target_content_sha256": "f" * 64,
            "split": "train",
            "input_key": "bcg_rawish_segment_soft_z_key",
            "target_key": "target_waveform_segment_soft_z_key",
        },
        "access": {
            "shared_index_metadata_read": True,
            "signal_splits_accessed": ["train"],
            "validation_accessed": False,
            "validation_target_accessed": False,
            "validation_prediction_accessed": False,
            "research_test_accessed": False,
            "checkpoint_accessed": False,
            "model_training_used": False,
            "model_inference_used": False,
            "gpu_used": False,
        },
        "counts": {
            "expected_windows": 10141,
            "actual_windows": 10141,
            "expected_samp_ids": 32,
            "actual_samp_ids": 32,
            "exclusions": {},
            "output_rows": {"x.csv": 1},
        },
        "finite_audit": {"x.csv": {"n_rows": 1, "numeric_columns": {"x": column_counts}}},
        "artifacts": [{"filename": "x.csv", "sha256": "1" * 64, "size_bytes": 1, "rows": 1}],
        "manifest_sha256": "2" * 64,
    }


def test_signal_audit_config_is_byte_and_semantically_locked(tmp_path: Path) -> None:
    config = load_signal_audit_config()
    assert config.raw["schema_version"] == CONFIG_SCHEMA_VERSION
    assert config.raw["access"]["split"] == "train"
    assert config.raw["bands"]["displacement"]["high_hz"] == 0.8
    assert config.raw["bands"]["carrier_low"]["low_inclusive"] is False
    assert config.raw["bands"]["carrier_high"]["high_hz"] == 8.0
    assert config.raw["bands"]["target"]["high_hz"] == 0.7
    assert config.raw["metrics"]["grids_hz"] == [10.0, 2.0, 1.0, 0.5]
    assert config.raw["output"]["allow_overwrite"] is False

    changed = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("split: train", "split: val")
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="配置哈希不匹配"):
        load_signal_audit_config(mutated)


def test_fixed_band_endpoints_do_not_overlap() -> None:
    config = load_signal_audit_config()
    fs = 100.0
    time = np.arange(18000, dtype=np.float64) / fs
    tones = {
        0.8: np.sin(2.0 * np.pi * 0.8 * time),
        3.0: np.sin(2.0 * np.pi * 3.0 * time),
        8.0: np.sin(2.0 * np.pi * 8.0 * time),
    }
    displacement = fft_band_extract(sum(tones.values()), fs=fs, band=config.band("displacement"))
    carrier_low = fft_band_extract(sum(tones.values()), fs=fs, band=config.band("carrier_low"))
    carrier_high = fft_band_extract(sum(tones.values()), fs=fs, band=config.band("carrier_high"))

    assert abs(np.dot(displacement, tones[0.8]) / np.dot(tones[0.8], tones[0.8]) - 1.0) < 1e-10
    assert abs(np.dot(carrier_low, tones[0.8])) < 1e-8
    assert abs(np.dot(carrier_low, tones[3.0]) / np.dot(tones[3.0], tones[3.0]) - 1.0) < 1e-10
    assert abs(np.dot(carrier_high, tones[3.0])) < 1e-8
    assert abs(np.dot(carrier_high, tones[8.0]) / np.dot(tones[8.0], tones[8.0]) - 1.0) < 1e-10


def test_carrier_demodulation_paths_are_fixed_finite_and_shape_safe() -> None:
    config = load_signal_audit_config()
    time = np.arange(18000, dtype=np.float64) / 100.0
    modulation = 1.0 + 0.35 * np.sin(2.0 * np.pi * 0.2 * time)
    raw = modulation * np.sin(2.0 * np.pi * 6.0 * time)
    paths = build_carrier_sampling_paths(raw, carrier_band=config.band("carrier_high"), config=config)
    assert set(paths) == {
        "p1_demod_100_then_10",
        "p2_100_to_20_demod_then_10",
        "p3_100_to_10_then_demod",
    }
    assert all(values.shape == (1800,) and np.isfinite(values).all() for values in paths.values())
    reference = paths["p1_demod_100_then_10"]
    path_2 = paths["p2_100_to_20_demod_then_10"]
    assert np.corrcoef(reference, path_2)[0, 1] > 0.95


def test_explicit_grid_decimation_differs_from_boxcar_when_alias_risk_exists() -> None:
    config = load_signal_audit_config()
    time = np.arange(1800, dtype=np.float64) / 10.0
    signal = np.sin(2.0 * np.pi * 0.2 * time) + 0.8 * np.sin(2.0 * np.pi * 0.8 * time)
    assert above_nyquist_power_fraction(signal, fs=10.0, destination_fs=1.0) > 0.25
    lowpass = lowpass_block_center_decimate(signal, input_fs=10.0, spec=config.resample("10_to_1"))
    boxcar = boxcar_decimate(signal, factor=10)
    assert lowpass.shape == boxcar.shape == (180,)
    assert np.sqrt(np.mean(np.square(lowpass - boxcar))) > 0.05

    low_frequency_only = np.sin(2.0 * np.pi * 0.2 * time)
    pooled = boxcar_decimate(low_frequency_only, factor=10)
    reconstructed = reconstruct_block_center_grid(pooled, factor=10, target_length=1800, target_fs=10.0)
    assert np.corrcoef(low_frequency_only, reconstructed)[0, 1] > 0.999


def test_single_window_audit_covers_all_predeclared_paths_and_grids() -> None:
    config = load_signal_audit_config()
    time = np.arange(18000, dtype=np.float64) / 100.0
    target = (1.0 + 0.15 * np.sin(2.0 * np.pi * 0.02 * time)) * np.sin(2.0 * np.pi * 0.22 * time)
    raw = (
        0.8 * target
        + (1.0 + 0.25 * target) * np.sin(2.0 * np.pi * 1.6 * time)
        + (1.0 + 0.20 * target) * np.sin(2.0 * np.pi * 5.5 * time)
    )
    target_row, target_sequence_rows, proxy_rows, sampling_rows, grid_rows = audit_signal_window(
        raw,
        target,
        identity={"dataset_row_id": 1, "samp_id": 2, "coupling_state_id": 3, "split": "train"},
        source_metric_cfg=_metric_config(),
        audit_cfg=config,
    )
    assert target_row["target_stratum"] in {"low", "medium", "high"}
    assert len(target_sequence_rows) == 44
    assert sum(row["sequence"] == "local_rr_bpm" for row in target_sequence_rows) == 9
    assert sum(row["sequence"] == "log_effort" for row in target_sequence_rows) == 35
    assert len(proxy_rows) == 4
    assert len(sampling_rows) == 6
    assert len(grid_rows) == 24
    assert {row["grid_hz"] for row in grid_rows} == {2.0, 1.0, 0.5}
    assert all(row["full_waveform_eligible"] is (row["grid_hz"] == 2.0) for row in grid_rows)
    for collection in (proxy_rows, sampling_rows, grid_rows):
        for row in collection:
            assert row["split"] == "train"
            numeric = [float(value) for value in row.values() if isinstance(value, (float, np.floating))]
            assert not any(np.isinf(numeric))


def test_low_dynamic_target_is_counted_not_silently_dropped() -> None:
    record, _, _, sequence = target_timescale_metrics(
        np.zeros(18000, dtype=np.float64),
        metric_cfg=_metric_config(),
    )
    assert record["target_waveform_dynamic"] is False
    assert np.isnan(record["target_whole_rr_bpm"])
    assert np.isnan(record["target_spectral_centroid_hz"])
    local = [row for row in sequence if row["sequence"] == "local_rr_bpm"]
    assert len(local) == 9
    assert all(row["eligible"] is False and np.isnan(row["value"]) for row in local)


def test_receipt_schema_fails_closed_on_access_or_unknown_field() -> None:
    receipt = _valid_receipt()
    validate_access_receipt(receipt)

    leaked = deepcopy(receipt)
    leaked["access"]["validation_target_accessed"] = True
    with pytest.raises(ValueError, match="越界访问"):
        validate_access_receipt(leaked)

    unknown = deepcopy(receipt)
    unknown["access"]["validation_predictions"] = False
    with pytest.raises(ValueError, match="字段不严格匹配"):
        validate_access_receipt(unknown)

    incomplete = deepcopy(receipt)
    incomplete["counts"]["actual_windows"] = 10140
    with pytest.raises(ValueError, match="window count 不完整"):
        validate_access_receipt(incomplete)


def test_runner_fails_before_git_or_data_access_when_output_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = load_signal_audit_config()
    raw = deepcopy(frozen.raw)
    output = tmp_path / "already_exists"
    output.mkdir()
    raw["output"]["directory"] = str(output)
    synthetic = SignalAuditConfig(path=frozen.path, raw=raw, sha256=frozen.sha256)
    monkeypatch.setattr(signal_audit, "load_signal_audit_config", lambda _: synthetic)
    monkeypatch.setattr(
        signal_audit,
        "_git_identity",
        lambda: (_ for _ in ()).throw(AssertionError("output preflight 后不应访问 Git")),
    )
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        signal_audit.run_train_signal_audit(config_path=frozen.path, command="test")


def test_sample_direct_mean_excludes_numeric_group_dimensions() -> None:
    frame = pd.DataFrame(
        {
            "dataset_row_id": [1, 2, 3, 4],
            "samp_id": [10, 10, 11, 11],
            "coupling_state_id": [1, 1, 2, 2],
            "target_stratum": ["low", "low", "high", "high"],
            "source": ["target"] * 4,
            "grid_hz": [2.0] * 4,
            "method": ["explicit_lowpass_decimation"] * 4,
            "roundtrip_nrmse": [0.1, 0.3, 0.2, 0.4],
        }
    )
    result = sample_direct_mean(frame, dimensions=["source", "grid_hz", "method"])
    assert "grid_hz" in result.columns
    assert not result.columns.duplicated().any()
    overall = result[result["target_stratum"].eq("all")].sort_values("samp_id")
    assert overall["grid_hz"].tolist() == [2.0, 2.0]
    assert np.allclose(overall["roundtrip_nrmse"], [0.2, 0.3])


def test_cli_exposes_no_split_output_or_override_arguments() -> None:
    script = (REPO_ROOT / "scripts/audit_resp_temporal_v1_signal.py").read_text(encoding="utf-8")
    assert 'add_argument("--config"' in script
    assert 'add_argument("--split"' not in script
    assert 'add_argument("--output' not in script
    assert 'add_argument("--set"' not in script
    assert "checkpoint" not in script.lower()
    assert "validation" not in script.lower()
    assert "research-test" not in script.lower()


def test_confirmed_signal_and_candidate_locks_are_strictly_linked() -> None:
    signal_path = REPO_ROOT / "docs/experiments/resp_temporal_v1_signal_substrate_lock_20260820.json"
    candidate_path = REPO_ROOT / "docs/experiments/resp_temporal_v1_candidate_lock_20260820.json"
    signal_bytes = signal_path.read_bytes()
    candidate_bytes = candidate_path.read_bytes()
    signal = json.loads(signal_bytes)
    candidate = json.loads(candidate_bytes)

    assert sha256(signal_bytes).hexdigest() == "11bfcad00f4532d4bdfe1413a375b5f06f46eb8ac67dfcd475701872322fee69"
    assert sha256(candidate_bytes).hexdigest() == "b4a2c83310fa2ce9519e3ca25814aea0b179458ab52d6380a932545c99c25f9b"
    assert candidate["source_lock"]["sha256"] == sha256(signal_bytes).hexdigest()
    assert signal["multiscale_grid_lock"]["grids_hz"] == [10.0, 2.0, 1.0]
    assert signal["multiscale_grid_lock"]["average_pooling_allowed"] is False
    assert signal["common_signal_substrate"]["current_probe_stem_accepted_as_is"] is False
    assert [record["candidate_id"] for record in candidate["candidates"]] == [
        "rtm_v1_t0_locked_stem_head",
        "rtm_v1_tcn_d9_h384",
        "rtm_v1_bimamba2_d96_l6",
        "rtm_v1_bilstm_h96_l2",
        "rtm_v1_multiscale_10_2_1_h384",
    ]
    multiscale = candidate["candidates"][-1]
    assert multiscale["coarse"]["dilations"] == [1, 2, 4, 8, 16, 32]
    assert multiscale["coarse"]["theoretical_receptive_field_seconds"] >= 180.0
    assert candidate["closed_candidates_and_search"]["global_token_mixer"]["included"] is False
    assert candidate["authorization"]["gpu_engineering_allowed"] is False
    assert candidate["authorization"]["formal_training_allowed"] is False
