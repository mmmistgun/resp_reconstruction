from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from resp_train.data.factory import build_tho_data
from resp_train.data.research_v2 import (
    ResearchV2WindowDataset,
    _DEFAULT_RR_PEAK_BAD_MASK_KEYS,
    adapt_research_v2_index,
)


def _write_research_v2_npzs(root: Path, samp_id: int) -> tuple[str, str]:
    align_dir = root / "whole_night" / "alignment" / str(samp_id)
    bank_dir = root / "whole_night" / "signal_bank" / str(samp_id)
    align_dir.mkdir(parents=True, exist_ok=True)
    bank_dir.mkdir(parents=True, exist_ok=True)
    base = np.arange(80, dtype=np.float32)
    np.savez(
        align_dir / "research_v2_alignment.npz",
        bcg_resp_band_to_tho_timebase=base,
        bcg_resp_band_state_aligned=base + 100,
    )
    np.savez(
        bank_dir / "research_v2_signal_bank.npz",
        tho_waveform_ref=base * 2,
        tho_event_phase_ref=base * 3,
        tho_rate_ref=base * 4,
        tho_bad_sec=np.zeros(8, dtype=np.uint8),
        bcg_bad_sec=np.zeros(8, dtype=np.uint8),
        hard_invalid_sec=np.zeros(8, dtype=np.uint8),
    )
    return (
        f"../whole_night/alignment/{samp_id}/research_v2_alignment.npz",
        f"../whole_night/signal_bank/{samp_id}/research_v2_signal_bank.npz",
    )


def _research_v2_rows(root: Path) -> pd.DataFrame:
    rows = []
    row_id = 1
    for split, samp_id in [("train", 88), ("val", 220)]:
        source_npz, target_source_npz = _write_research_v2_npzs(root, samp_id)
        for allowed_losses in ["rate;waveform", "rate"]:
            rows.append(
                {
                    "dataset_row_id": row_id,
                    "split": split,
                    "samp_id": samp_id,
                    "coupling_state_id": 1,
                    "window_start_s": 0.1,
                    "window_end_s": 0.5,
                    "window_duration_s": 0.4,
                    "source_npz": source_npz,
                    "target_source_npz": target_source_npz,
                    "bcg_input_key": "bcg_resp_band_to_tho_timebase",
                    "bcg_input_aligned_key": "bcg_resp_band_state_aligned",
                    "target_event_phase_key": "tho_event_phase_ref",
                    "target_waveform_key": "tho_waveform_ref",
                    "target_rate_key": "tho_rate_ref",
                    "hard_valid_ratio": 1.0,
                    "state_alignment_valid_ratio": 1.0,
                    "transient_motion_ratio": 0.0,
                    "posture_transition_ratio": 0.0,
                    "amplitude_reliable_ratio": 1.0,
                    "normalization_reliable_ratio": 1.0,
                    "rate_confidence_score": 0.8,
                    "rate_confidence_level": "high",
                    "phase_confidence_score": 0.1,
                    "phase_confidence_level": "low",
                    "event_confidence_score": 0.1,
                    "event_confidence_level": "low",
                    "waveform_confidence_score": 0.8 if "waveform" in allowed_losses else 0.1,
                    "waveform_confidence_level": "high" if "waveform" in allowed_losses else "low",
                    "alignment_confidence_score": 0.8,
                    "alignment_confidence_level": "high",
                    "supervision_confidence_score": 0.8,
                    "supervision_confidence_level": "high",
                    "state_alignment_method": "constant_shift",
                    "state_alignment_lag_s": 0.2,
                    "state_alignment_drift_s_per_hour": 0.0,
                    "state_alignment_is_reference_assisted": 1,
                    "allowed_losses": allowed_losses,
                    "reason": "",
                }
            )
            row_id += 1
    return pd.DataFrame.from_records(rows)


def _prepare_research_v2_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "research_v2_dataset"
    training_dir = root / "training"
    training_dir.mkdir(parents=True)
    _research_v2_rows(root).to_csv(training_dir / "dataset_index.csv", index=False)
    return root


def _cfg(root: Path):
    return OmegaConf.create(
        {
            "data": {
                "format": "research_v2",
                "dataset_root": str(root),
                "index_csv": "training/dataset_index.csv",
                "input_set": "research_v2_waveform",
                "train_split": "train",
                "val_split": "val",
                "target_task": "waveform",
                "bcg_input_key": "bcg_input_aligned_key",
                "target_key": "target_waveform_key",
                "max_train_windows": None,
                "max_val_windows": None,
                "filter_unusable": True,
                "preload_windows": True,
                "train_sample_strategy": "stratified_random",
                "val_sample_strategy": "stratified_random",
                "train_sample_seed": 1,
                "val_sample_seed": 2,
                "stratify_column": "allowed_losses",
                "min_hard_valid_ratio": 0.8,
                "min_state_alignment_valid_ratio": 0.8,
            },
            "window": {"duration_samples": 40, "target_fs": 100},
            "training": {"batch_size": 2, "num_workers": 0},
        }
    )


def test_adapt_research_v2_index_keeps_only_waveform_rows(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    raw = pd.read_csv(root / "training" / "dataset_index.csv")

    adapted = adapt_research_v2_index(raw, cfg)

    assert adapted["input_set"].unique().tolist() == ["research_v2_waveform"]
    assert adapted["usable"].tolist() == [True, False, True, False]
    assert adapted["window_start_sample"].tolist() == [10, 10, 10, 10]
    assert adapted["window_end_sample"].tolist() == [50, 50, 50, 50]
    assert adapted["bcg_signal_key"].tolist() == ["bcg_resp_band_state_aligned"] * 4
    assert adapted["target_signal_key"].tolist() == ["tho_waveform_ref"] * 4


def test_adapt_research_v2_index_supports_renamed_soft_z_columns(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.data.bcg_input_key = "bcg_rawish_segment_soft_z_key"
    cfg.data.target_key = "target_waveform_segment_soft_z_key"
    raw = pd.read_csv(root / "training" / "dataset_index.csv")
    raw = raw.rename(
        columns={
            "bcg_input_key": "bcg_input_observed_key",
            "bcg_input_aligned_key": "bcg_input_segment_robust_z_key",
            "target_waveform_key": "target_waveform_observed_key",
        }
    )
    raw["bcg_rawish_segment_soft_z_key"] = "bcg_rawish_wideband_state_aligned_segment_soft_z"
    raw["target_waveform_segment_soft_z_key"] = "tho_waveform_segment_soft_z"

    adapted = adapt_research_v2_index(raw, cfg)

    assert adapted["bcg_signal_key"].tolist() == ["bcg_rawish_wideband_state_aligned_segment_soft_z"] * 4
    assert adapted["target_signal_key"].tolist() == ["tho_waveform_segment_soft_z"] * 4


def test_adapt_research_v2_index_keeps_finite_ratios_as_audit_only(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    raw = pd.read_csv(root / "training" / "dataset_index.csv")
    raw["input_finite_ratio"] = [0.5, 1.0, 1.0, 1.0]
    raw["target_finite_ratio"] = [1.0, 1.0, 0.5, 1.0]

    adapted = adapt_research_v2_index(raw, cfg)

    assert adapted["input_finite_ratio"].tolist() == [0.5, 1.0, 1.0, 1.0]
    assert adapted["target_finite_ratio"].tolist() == [1.0, 1.0, 0.5, 1.0]
    assert adapted["usable"].tolist() == [True, False, True, False]


def test_research_v2_dataset_slices_alignment_and_signal_bank(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    rows = adapt_research_v2_index(pd.read_csv(root / "training" / "dataset_index.csv"), cfg)
    rows = rows[(rows["split"] == "train") & rows["usable"]].reset_index(drop=True)

    dataset = ResearchV2WindowDataset(root / "training" / "dataset_index.csv", rows, cfg, preload_windows=True)
    sample = dataset[0]

    assert torch.equal(sample["x"], torch.arange(110, 150, dtype=torch.float32).view(1, -1))
    assert torch.equal(sample["target"], (torch.arange(10, 50, dtype=torch.float32) * 2).view(1, -1))
    assert sample["meta"]["samp_id"] == 88
    assert sample["meta"]["allowed_losses"] == "rate;waveform"
    assert sample["meta"]["coupling_state_id"] == 1
    assert sample["meta"]["waveform_confidence_score"] == pytest.approx(0.8)
    assert sample["meta"]["waveform_confidence_level"] == "high"
    assert torch.equal(sample["meta"]["rr_peak_valid_mask"], torch.ones(40, dtype=torch.bool))


def test_research_v2_dataset_expands_second_level_bad_masks_for_raw_peak_metrics(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.window.duration_samples = 400
    bank_npz = root / "whole_night" / "signal_bank" / "88" / "research_v2_signal_bank.npz"
    base = np.arange(800, dtype=np.float32)
    tho_bad_sec = np.zeros(8, dtype=np.uint8)
    bcg_bad_sec = np.zeros(8, dtype=np.uint8)
    hard_invalid_sec = np.zeros(8, dtype=np.uint8)
    tho_bad_sec[2] = 1
    bcg_bad_sec[4] = 1
    np.savez(
        root / "whole_night" / "alignment" / "88" / "research_v2_alignment.npz",
        bcg_resp_band_to_tho_timebase=base,
        bcg_resp_band_state_aligned=base + 100,
    )
    np.savez(
        bank_npz,
        tho_waveform_ref=base * 2,
        tho_event_phase_ref=base * 3,
        tho_rate_ref=base * 4,
        tho_bad_sec=tho_bad_sec,
        bcg_bad_sec=bcg_bad_sec,
        hard_invalid_sec=hard_invalid_sec,
    )
    rows = adapt_research_v2_index(pd.read_csv(root / "training" / "dataset_index.csv"), cfg)
    rows = rows[(rows["split"] == "train") & rows["usable"]].reset_index(drop=True)
    rows.loc[0, "window_start_s"] = 1.0
    rows.loc[0, "window_start_sample"] = 100
    rows.loc[0, "window_end_sample"] = 500

    dataset = ResearchV2WindowDataset(root / "training" / "dataset_index.csv", rows, cfg)
    mask = dataset[0]["meta"]["rr_peak_valid_mask"]

    assert mask.shape == (400,)
    assert mask[:100].all()
    assert not mask[100:200].any()
    assert mask[200:300].all()
    assert not mask[300:400].any()


def test_research_v2_dataset_reuses_whole_night_rr_peak_mask(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.data.drop_nonfinite_windows = False
    cfg.data.preload_windows = False
    rows = adapt_research_v2_index(pd.read_csv(root / "training" / "dataset_index.csv"), cfg)
    row = rows[(rows["split"] == "train") & rows["usable"]].iloc[0].copy()
    second = row.copy()
    second["dataset_row_id"] = 999
    second["window_start_sample"] = 20
    second["window_end_sample"] = 60
    dataset_rows = pd.DataFrame.from_records([row.to_dict(), second.to_dict()])
    dataset = ResearchV2WindowDataset(root / "training" / "dataset_index.csv", dataset_rows, cfg)
    mask_keys = set(_DEFAULT_RR_PEAK_BAD_MASK_KEYS)
    original_get_arrays = dataset.target_cache.get_arrays
    mask_requests: list[tuple[str, ...]] = []

    def counting_get_arrays(source_npz: str, keys: list[str]):
        if any(key in mask_keys for key in keys):
            mask_requests.append(tuple(keys))
        return original_get_arrays(source_npz, keys)

    dataset.target_cache.get_arrays = counting_get_arrays

    first = dataset[0]["meta"]["rr_peak_valid_mask"]
    first_request_count = len(mask_requests)
    second_mask = dataset[1]["meta"]["rr_peak_valid_mask"]

    assert first_request_count > 0
    assert len(mask_requests) == first_request_count
    assert torch.equal(first, torch.ones(40, dtype=torch.bool))
    assert torch.equal(second_mask, torch.ones(40, dtype=torch.bool))


def test_research_v2_dataset_drops_nonfinite_selected_windows(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.data.drop_nonfinite_windows = True
    source_npz = root / "whole_night" / "alignment" / "88" / "research_v2_alignment.npz"
    base = np.arange(80, dtype=np.float32)
    aligned = base + 100
    aligned[20] = np.nan
    np.savez(
        source_npz,
        bcg_resp_band_to_tho_timebase=base,
        bcg_resp_band_state_aligned=aligned,
    )
    rows = adapt_research_v2_index(pd.read_csv(root / "training" / "dataset_index.csv"), cfg)
    rows = rows[(rows["split"] == "train") & rows["usable"]].reset_index(drop=True)

    dataset = ResearchV2WindowDataset(root / "training" / "dataset_index.csv", rows, cfg)

    assert len(dataset) == 0


def test_research_v2_dataset_keeps_nonfinite_windows_by_default_until_item_load(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    source_npz = root / "whole_night" / "alignment" / "88" / "research_v2_alignment.npz"
    base = np.arange(80, dtype=np.float32)
    aligned = base + 100
    aligned[20] = np.nan
    np.savez(
        source_npz,
        bcg_resp_band_to_tho_timebase=base,
        bcg_resp_band_state_aligned=aligned,
    )
    rows = adapt_research_v2_index(pd.read_csv(root / "training" / "dataset_index.csv"), cfg)
    rows = rows[(rows["split"] == "train") & rows["usable"]].reset_index(drop=True)

    dataset = ResearchV2WindowDataset(root / "training" / "dataset_index.csv", rows, cfg)

    assert len(dataset) == 1
    with pytest.raises(ValueError, match="窗口包含非有限值"):
        dataset[0]


def test_build_tho_data_supports_research_v2_format(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)

    data = build_tho_data(cfg)

    assert len(data.train.dataset) == 1
    assert len(data.val.dataset) == 1
    batch = next(iter(data.train.loader))
    assert batch["x"].shape == (1, 1, 40)
    assert batch["target"].shape == (1, 1, 40)
    assert set(data.audit_summary["input_set"]) == {"research_v2_waveform"}


def test_adapt_research_v2_index_rejects_invalid_target_task(tmp_path: Path):
    root = _prepare_research_v2_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.data.target_task = "phase"

    with pytest.raises(ValueError, match="target_task"):
        adapt_research_v2_index(pd.read_csv(root / "training" / "dataset_index.csv"), cfg)
