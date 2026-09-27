"""定性导出只使用 synthetic/disposable fixture 的快速 CPU 验证。"""

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from resp_train.metrics.task import evaluate_task_predictions
from resp_train.paper_evidence.w0_test_qualitative import (
    METHODS, PRIMARY, SEED, anchor_deltas, render_window, save_arrays, waveform_details,
)
from resp_train.paper_evidence.w0_test_qualitative_runtime import check_rows, finish, render


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "window": {"target_fs": 100, "duration_samples": 18000},
        "loss": {"band_low_hz": .05, "band_high_hz": .7, "scale_eps": 1e-8,
                 "dynamic_eps": 1e-8, "corr_eps": 1e-8, "envelope_eps": 1e-8,
                 "max_lag_sec": .3, "envelope_window_sec": 10, "envelope_step_sec": 5},
        "evaluation": {"local_rr_window_sec": 60, "local_rr_step_sec": 15,
                       "ibi_peak_distance_samples": 142, "ibi_match_tolerance_sec": .5,
                       "ibi_coverage_threshold": .8, "ndtw_fs": 10, "ndtw_radius_sec": .3,
                       "envelope_strata_low": .3, "envelope_strata_high": .7},
    })


@pytest.fixture
def sample(cfg):
    t = np.arange(18000) / 100
    target = (1 + .2 * np.sin(2*np.pi*.01*t)) * np.sin(2*np.pi*.25*t)
    waves = np.stack([target, np.sin(2*np.pi*.23*t), np.sin(2*np.pi*.27*t), target + .05*np.sin(2*np.pi*.35*t)])
    frames = [evaluate_task_predictions({"r_tho_hat": waves[i:i+1], "tho_ref": waves[:1],
                                        "dataset_row_id": np.array([7])}, cfg, method=method)
              for i, method in enumerate(METHODS, 1)]
    arrays = {
        "dataset_row_id": np.array(7), "fs_hz": np.array(100.),
        "bcg": target + np.sin(2*np.pi*3*t), "waveforms": waves,
        "cwt_w": np.ones((97, 360)), "cwt_frequency_hz": np.geomspace(.03, 8, 97),
        "g": np.zeros((96, 1800)), "b": np.zeros((96, 1800)),
        "latent_time_s": (np.arange(1800) + .5) / 10,
        "r_scale_time": np.zeros(1800), "r_shift_time": np.zeros(1800), "r_total_time": np.zeros(1800),
        **waveform_details(waves, cfg),
    }
    return arrays, pd.concat(frames, ignore_index=True)


def test_trajectories_reconstruct_native_local_rr(sample):
    arrays, metrics = sample
    rr = arrays["local_rr_bpm"]
    eligible = arrays["local_rr_target_eligible"]
    assert arrays["local_rr_time_s"].tolist() == list(range(30, 151, 15))
    assert arrays["envelope_time_s"].tolist() == list(range(5, 176, 5))
    for i, method in enumerate(METHODS, 1):
        error = np.where(arrays["local_rr_valid"][i], abs(rr[i] - rr[0]), 39.)
        expected = metrics.loc[metrics.method == method, "local_rr_mae_bpm"].item()
        assert np.mean(error[eligible]) == pytest.approx(expected, abs=1e-10)
        envelopes = arrays["centered_log_rms_envelopes"]
        expected_env = metrics.loc[metrics.method == method, "envelope_trajectory_mae"].item()
        assert np.mean(abs(envelopes[i] - envelopes[0])) == pytest.approx(expected_env, abs=1e-10)


def test_degenerate_prediction_preserves_rows_and_flags(cfg, tmp_path):
    t = np.arange(18000) / 100
    waves = np.stack([np.sin(2*np.pi*.25*t), np.zeros(len(t))])
    result = waveform_details(waves, cfg)
    assert result["local_rr_valid"][0].all()
    assert not result["local_rr_valid"][1].any()
    assert np.isnan(result["local_rr_bpm"][1]).all()
    save_arrays(tmp_path / "valid_missing.npz", result)
    result["local_rr_valid"][1, 0] = True
    with pytest.raises(ValueError, match="有效性"):
        save_arrays(tmp_path / "bad_missing.npz", result)


def test_anchor_identity_and_numeric_failure(sample):
    _, metrics = sample
    observed = metrics.loc[metrics.method == "W0"].copy()
    assert anchor_deltas(observed, observed).within_atol.all()
    altered = observed.copy()
    altered.loc[:, PRIMARY[0]] += 1e-3
    assert not anchor_deltas(altered, observed).within_atol.all()
    altered.loc[:, "dataset_row_id"] = 8
    with pytest.raises(ValueError, match="集合"):
        anchor_deltas(altered, observed)
    with pytest.raises(ValueError, match="重复"):
        anchor_deltas(pd.concat([observed, observed]), observed)
    observed.loc[:, PRIMARY[0]] = np.nan
    with pytest.raises(FloatingPointError):
        anchor_deltas(observed, observed)


def test_safe_npz_roundtrip(tmp_path):
    path = tmp_path / "result.npz"
    save_arrays(path, {"g": np.array([.5], dtype=np.float32)})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_arrays(path, {"g": np.array([.1])})
    assert path.read_bytes() == original
    with pytest.raises(FloatingPointError):
        save_arrays(tmp_path / "nan.npz", {"g": np.array([np.nan])})
    assert not (tmp_path / "nan.npz").exists()
    with np.load(path, allow_pickle=False) as blob:
        np.testing.assert_array_equal(blob["g"], [.5])


def test_full_row_contract_is_not_silently_reduced():
    rows = pd.DataFrame({"dataset_row_id": np.arange(2310), "samp_id": np.arange(2310) % 8,
                         "coupling_state_id": np.zeros(2310, dtype=int), "split": "test"})
    digest = hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest()
    check_rows(rows, rows.copy(), digest)
    with pytest.raises(ValueError, match="数量"):
        check_rows(rows.iloc[:-1], rows, digest)
    with pytest.raises(ValueError, match="order"):
        check_rows(rows.iloc[::-1], rows, digest)
    with pytest.raises(ValueError, match="identity"):
        check_rows(rows, rows.assign(samp_id=0), digest)


def test_cli_requires_test_confirmation_before_access(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/export_w0_test_qualitative.py"
    output = tmp_path / "never_created"
    result = subprocess.run([sys.executable, str(script), "export", "--output", str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "--confirm-research-test-export" in result.stderr
    assert not output.exists()


def test_plot_rejects_mismatched_metrics(sample, tmp_path):
    arrays, metrics = sample
    with pytest.raises(ValueError, match="row"):
        render_window(arrays, metrics.assign(dataset_row_id=8), tmp_path,
                      row_id=7, subject=1, start_s=0)
    with pytest.raises(ValueError, match="zoom"):
        render_window(arrays, metrics, tmp_path, row_id=7, subject=1, start_s=0, zoom=(90, 200))


def test_offline_render_complete_export_and_tamper(sample, tmp_path):
    arrays, metrics = sample
    source = tmp_path / "export"
    (source / "windows").mkdir(parents=True)
    coordinates = {key: arrays.pop(key) for key in ("cwt_frequency_hz", "latent_time_s")}
    save_arrays(source / "coordinates.npz", coordinates)
    save_arrays(source / "windows/row_7.npz", arrays)
    metrics.to_csv(source / "metrics.csv", index=False)
    index = {"dataset_row_id": 7, "samp_id": 1, "window_start_s": 30,
             "file": "windows/row_7.npz", "W0_lag_aware_signed_pcc": .9, "W0_local_rr_mae_bpm": .1}
    pd.DataFrame([index]).to_csv(source / "window_index.csv", index=False)
    finish(source, {"phase": "export", "seed": SEED})
    output = render(source, tmp_path / "plots", row_ids=[7], zoom=(30, 60), command="synthetic")
    assert len(list((output / "figures").glob("*.png"))) == 3
    assert (output / "index.html").is_file()
    assert (output / "artifact_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        render(source, output, row_ids=[7], zoom=None, command="synthetic")
    with pytest.raises(ValueError, match="不存在"):
        render(source, tmp_path / "bad_rows", row_ids=[8], zoom=None, command="synthetic")
    (source / "windows/row_7.npz").write_bytes(b"corrupted disposable fixture")
    with pytest.raises(RuntimeError, match="身份"):
        render(source, tmp_path / "tampered", row_ids=[7], zoom=None, command="synthetic")
    assert (tmp_path / "tampered/failure.json").is_file()
