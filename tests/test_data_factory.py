from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import resp_train.data.dataset as dataset_module
from resp_train.data.factory import build_tho_data, build_window_data


def _write_npz(root: Path, samp_id: int) -> str:
    npz_dir = root / "whole_night" / "mixed_bcg_to_tho" / str(samp_id)
    npz_dir.mkdir(parents=True, exist_ok=True)
    signal = np.linspace(0, 1, 64, dtype=np.float32)
    np.savez(npz_dir / "sample.npz", bcg=signal, tho=signal * 2)
    return f"../whole_night/mixed_bcg_to_tho/{samp_id}/sample.npz"


def _index_rows(root: Path) -> pd.DataFrame:
    qualities = [
        "near_zero_residual",
        "stable_nonzero_residual",
        "near_zero_residual",
        "high_residual",
    ]
    records = []
    row_id = 1
    for split in ("train", "val"):
        for offset, quality in enumerate(qualities):
            samp_id = 100 + offset
            records.append(
                {
                    "dataset_row_id": row_id,
                    "input_set": "mixed_zscore",
                    "split": split,
                    "samp_id": samp_id,
                    "segment_id": 1,
                    "window_id_in_segment": offset + 1,
                    "source_npz": _write_npz(root, samp_id),
                    "bcg_signal_key": "bcg",
                    "target_signal_key": "tho",
                    "valid_sec_key": "valid",
                    "segment_decision": "include_candidate",
                    "window_start_sample": 0,
                    "window_end_sample": 32,
                    "window_duration_samples": 32,
                    "target_fs": 100,
                    "valid_ratio": 1.0,
                    "input_finite_ratio": 1.0,
                    "target_finite_ratio": 1.0,
                    "residual_quality_class": quality,
                    "base_alignment_method": "keep_original",
                    "apply_decision": "approved",
                    "reason": "ok",
                }
            )
            row_id += 1
    return pd.DataFrame.from_records(records)


def _prepare_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    training_dir = root / "training"
    training_dir.mkdir(parents=True)
    _index_rows(root).to_csv(training_dir / "dataset_index.csv", index=False)
    return root


def _cfg(root: Path):
    return OmegaConf.create(
        {
            "data": {
                "dataset_root": str(root),
                "index_csv": "training/dataset_index.csv",
                "input_set": "mixed_zscore",
                "train_split": "train",
                "val_split": "val",
                "max_train_windows": 3,
                "max_val_windows": 2,
                "filter_unusable": True,
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": [],
                "preload_windows": True,
                "train_sample_strategy": "stratified_random",
                "val_sample_strategy": "head",
                "train_sample_seed": 1,
                "val_sample_seed": 2,
                "stratify_column": "residual_quality_class",
            },
            "window": {"duration_samples": 32, "target_fs": 100},
            "training": {"batch_size": 2, "num_workers": 0},
        }
    )


def test_build_window_data_returns_rows_dataset_loader_and_audit(tmp_path: Path):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)

    bundle = build_window_data(
        cfg,
        split="train",
        max_windows=cfg.data.max_train_windows,
        sample_strategy=cfg.data.train_sample_strategy,
        sample_seed=cfg.data.train_sample_seed,
        shuffle=False,
    )

    assert bundle.index_path == root / "training" / "dataset_index.csv"
    assert len(bundle.rows) == 3
    assert len(bundle.dataset) == 3
    assert len(bundle.audited) == 8
    assert bundle.audit_summary["n_windows"].sum() == 8
    batch = next(iter(bundle.loader))
    assert batch["x"].shape == (2, 1, 32)
    assert batch["target"].shape == (2, 1, 32)


def test_build_window_data_pins_memory_for_cuda_training(tmp_path: Path):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.training.device = "cuda:0"

    bundle = build_window_data(
        cfg,
        split="train",
        max_windows=cfg.data.max_train_windows,
        sample_strategy=cfg.data.train_sample_strategy,
        sample_seed=cfg.data.train_sample_seed,
        shuffle=False,
    )

    assert bundle.loader.pin_memory is True


def test_build_tho_data_uses_independent_train_and_val_sampling(tmp_path: Path):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)

    data = build_tho_data(cfg)

    assert len(data.train.rows) == 3
    assert len(data.val.rows) == 2
    assert data.train.rows["split"].unique().tolist() == ["train"]
    assert data.val.rows["split"].unique().tolist() == ["val"]
    assert data.val.rows["dataset_row_id"].tolist() == [5, 6]
    assert len(data.audited) == 8
    assert data.audit_summary["n_windows"].sum() == 8


def test_build_tho_data_preload_progress_follows_show_progress(tmp_path: Path, monkeypatch):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.training.show_progress = True
    seen_desc: list[str] = []

    def fake_tqdm(iterable, *, desc=None, disable=None, leave=None):
        seen_desc.append(str(desc))
        assert disable is False
        return iterable

    monkeypatch.setattr(dataset_module, "tqdm", fake_tqdm, raising=False)

    build_tho_data(cfg)

    assert seen_desc == ["preload train windows", "preload val windows"]


def test_build_tho_data_quiet_mode_skips_preload_progress(tmp_path: Path, monkeypatch):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.training.show_progress = False
    seen_desc: list[str] = []

    def fake_tqdm(iterable, *, desc=None, disable=None, leave=None):
        seen_desc.append(str(desc))
        return iterable

    monkeypatch.setattr(dataset_module, "tqdm", fake_tqdm, raising=False)

    build_tho_data(cfg)

    assert seen_desc == []


def test_build_tho_data_rejects_empty_train_split(tmp_path: Path):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.data.train_split = "missing"

    with pytest.raises(RuntimeError, match="train.*为空"):
        build_tho_data(cfg)
