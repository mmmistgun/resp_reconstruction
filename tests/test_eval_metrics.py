import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train.engine.train import collect_predictions, save_checkpoint
from resp_train.metrics.evaluate import evaluate_prediction_dict
from scripts.eval_tho_small import _resolve_config_path, _validate_checkpoint_config


def _cfg():
    return OmegaConf.create(
        {
            "window": {"target_fs": 100},
            "loss": {
                "envelope_window_sec": 2.0,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 0.7,
            },
        }
    )


def _modulated_breath_signal(fs: float, duration_sec: float) -> np.ndarray:
    t = np.arange(int(fs * duration_sec), dtype=np.float64) / fs
    envelope = 1.0 + 0.25 * np.sin(2 * np.pi * 0.03 * t)
    return envelope * np.sin(2 * np.pi * 0.23 * t + 0.15)


def test_evaluate_prediction_dict_returns_window_metrics():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t).astype(np.float32)
    preds = {
        "r_tho_hat": target.reshape(1, 1, -1),
        "tho_ref": target.reshape(1, 1, -1),
        "dataset_row_id": np.asarray([1]),
        "split": np.asarray(["val"]),
        "input_set": np.asarray(["mixed_zscore"]),
        "residual_quality_class": np.asarray(["near_zero_residual"]),
    }

    frame = evaluate_prediction_dict(preds, _cfg(), method="model")

    assert frame.loc[0, "method"] == "model"
    assert frame.loc[0, "dataset_row_id"] == 1
    assert frame.loc[0, "rr_spec_abs_error"] < 1.0
    assert frame.loc[0, "rr_peak_band_abs_error"] < 1.0
    assert frame.loc[0, "spectrum_similarity"] > 0.99
    assert frame.loc[0, "relative_envelope_corr"] > 0.99
    assert frame.loc[0, "relative_envelope_mae"] < 0.01
    assert frame.loc[0, "band_limited_corr"] > 0.99
    assert frame.loc[0, "best_lag_corr"] > 0.99
    assert abs(frame.loc[0, "best_lag_sec"]) < 1e-6


def test_evaluate_prediction_dict_reports_bandpassed_peak_rate_for_spiky_prediction():
    fs = 100
    t = np.arange(0, 80, 1 / fs)
    target = np.sin(2 * np.pi * 0.2 * t).astype(np.float32)
    pred = (target + 2.0 * np.sin(2 * np.pi * 2.0 * t)).astype(np.float32)
    preds = {
        "r_tho_hat": pred.reshape(1, 1, -1),
        "tho_ref": target.reshape(1, 1, -1),
    }

    frame = evaluate_prediction_dict(preds, _cfg(), method="model")

    assert frame.loc[0, "rr_peak_band_abs_error"] < 0.5
    assert np.isfinite(frame.loc[0, "pred_rr_peak_band_bpm"])
    assert np.isfinite(frame.loc[0, "target_rr_peak_band_bpm"])


def test_evaluate_prediction_dict_masks_bad_segments_for_raw_peak_rate():
    fs = 100
    t = np.arange(0, 80, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t).astype(np.float32)
    pred = target.copy()
    bad = (t >= 20.0) & (t < 60.0)
    pred[bad] = (8.0 * np.sin(2 * np.pi * 0.5 * t[bad])).astype(np.float32)
    valid_mask = (~bad).astype(np.bool_)
    preds = {
        "r_tho_hat": pred.reshape(1, 1, -1),
        "tho_ref": target.reshape(1, 1, -1),
        "rr_peak_valid_mask": valid_mask.reshape(1, -1),
    }

    frame = evaluate_prediction_dict(preds, _cfg(), method="model")

    assert frame.loc[0, "rr_peak_abs_error"] < 0.5
    assert frame.loc[0, "rr_peak_unmasked_abs_error"] > 5.0
    assert frame.loc[0, "rr_peak_valid_ratio"] == pytest.approx(0.5)
    assert frame.loc[0, "rr_peak_valid_segment_count"] == 2


def test_evaluate_prediction_dict_reports_best_lag_for_shifted_prediction():
    fs = 100
    target = _modulated_breath_signal(fs, 80.0).astype(np.float32)
    delay_samples = int(round(0.5 * fs))
    pred = np.zeros_like(target)
    pred[delay_samples:] = target[:-delay_samples]
    cfg = _cfg()
    cfg.evaluation = {"max_lag_sec": 1.0, "lag_bandpass_order": 4}
    preds = {
        "r_tho_hat": pred.reshape(1, 1, -1),
        "tho_ref": target.reshape(1, 1, -1),
    }

    frame = evaluate_prediction_dict(preds, cfg, method="model")

    assert frame.loc[0, "best_lag_corr"] > 0.99
    assert abs(frame.loc[0, "best_lag_sec"] - 0.5) < 1 / fs


def test_evaluate_prediction_dict_reports_bandpassed_zero_crossing_breath_counts():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t - np.pi / 2).astype(np.float32)
    pred = np.sin(2 * np.pi * 0.30 * t - np.pi / 2).astype(np.float32)
    preds = {
        "r_tho_hat": pred.reshape(1, 1, -1),
        "tho_ref": target.reshape(1, 1, -1),
    }

    frame = evaluate_prediction_dict(preds, _cfg(), method="model")

    assert frame.loc[0, "pred_breath_count_zero_cross"] == 18
    assert frame.loc[0, "target_breath_count_zero_cross"] == 15
    assert frame.loc[0, "breath_count_zero_cross_abs_error"] == 3


def test_evaluate_prediction_dict_rejects_empty_predictions():
    preds = {
        "r_tho_hat": np.empty((0, 1, 100), dtype=np.float32),
        "tho_ref": np.empty((0, 1, 100), dtype=np.float32),
        "dataset_row_id": np.empty((0,), dtype=np.int64),
    }

    with pytest.raises(ValueError, match="预测不能为空"):
        evaluate_prediction_dict(preds, _cfg(), method="model")


class _MetaDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> dict:
        value = torch.full((1, 8), float(idx), dtype=torch.float32)
        return {
            "x": value,
            "target": value + 1,
            "meta": {
                "dataset_row_id": idx + 10,
                "split": "val",
                "input_set": "mixed_zscore",
                "residual_quality_class": "ok",
                "rr_peak_valid_mask": torch.tensor([True, False, True, True, True, True, True, True]),
            },
        }


class _IdentityModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


def test_collect_predictions_extracts_default_collated_meta():
    loader = DataLoader(_MetaDataset(), batch_size=2, shuffle=False)

    preds = collect_predictions(_IdentityModel(), loader, device=torch.device("cpu"), max_windows=2)

    assert preds["r_tho_hat"].shape == (2, 1, 8)
    assert preds["tho_ref"].shape == (2, 1, 8)
    assert preds["dataset_row_id"].tolist() == [10, 11]
    assert preds["split"].tolist() == ["val", "val"]
    assert preds["input_set"].tolist() == ["mixed_zscore", "mixed_zscore"]
    assert preds["residual_quality_class"].tolist() == ["ok", "ok"]
    assert preds["rr_peak_valid_mask"].shape == (2, 8)
    assert preds["rr_peak_valid_mask"][0].tolist() == [True, False, True, True, True, True, True, True]


def test_collect_predictions_rejects_non_positive_max_windows():
    loader = DataLoader(_MetaDataset(), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="max_windows 必须大于 0"):
        collect_predictions(_IdentityModel(), loader, device=torch.device("cpu"), max_windows=0)


def test_save_checkpoint_stores_resolved_config(tmp_path):
    cfg = OmegaConf.create({"model": {"base_channels": 4}, "data": {"input_set": "mixed_zscore"}})
    model = torch.nn.Conv1d(1, 1, kernel_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model=model, optimizer=optimizer, epoch=2, metrics={"val_loss": 1.2}, cfg=cfg)

    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["config"]["model"]["base_channels"] == 4
    assert checkpoint["config"]["data"]["input_set"] == "mixed_zscore"


def test_eval_resolves_config_from_checkpoint_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.pt"
    sidecar_config = run_dir / "config.yaml"
    checkpoint.write_bytes(b"placeholder")
    sidecar_config.write_text("model:\n  name: unet1d_tiny\n", encoding="utf-8")

    assert _resolve_config_path("", checkpoint) == sidecar_config


def test_eval_checkpoint_config_validation_rejects_key_mismatch():
    checkpoint_cfg = {
        "model": {"name": "unet1d_tiny", "base_channels": 4},
        "data": {"input_set": "mixed_zscore"},
    }
    current_cfg = OmegaConf.create(
        {
            "model": {"name": "unet1d_tiny", "base_channels": 8},
            "data": {"input_set": "mixed_zscore"},
        }
    )

    with pytest.raises(ValueError, match="checkpoint 配置与当前配置不一致"):
        _validate_checkpoint_config(checkpoint_cfg, current_cfg, keys=["model.name", "model.base_channels"])


def test_eval_checkpoint_config_validation_checks_filter_fields():
    checkpoint_cfg = {
        "data": {
            "filter_unusable": True,
            "valid_ratio_min": 0.99,
            "input_finite_ratio_min": 0.99,
            "target_finite_ratio_min": 0.99,
            "unusable_residual_classes": ["bad"],
        },
        "model": {"name": "unet1d_tiny", "in_channels": 1, "out_channels": 1, "base_channels": 4},
        "window": {"target_fs": 100, "duration_samples": 18000},
        "loss": {"envelope_window_sec": 2.0, "spectrum_low_hz": 0.05, "spectrum_high_hz": 0.7},
    }
    current_cfg = OmegaConf.create(
        {
            "data": {
                "filter_unusable": False,
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": ["bad"],
            },
            "model": {"name": "unet1d_tiny", "in_channels": 1, "out_channels": 1, "base_channels": 4},
            "window": {"target_fs": 100, "duration_samples": 18000},
            "loss": {"envelope_window_sec": 2.0, "spectrum_low_hz": 0.05, "spectrum_high_hz": 0.7},
        }
    )

    with pytest.raises(ValueError, match="data.filter_unusable"):
        _validate_checkpoint_config(checkpoint_cfg, current_cfg)


def test_eval_checkpoint_config_validation_requires_checkpoint_config():
    with pytest.raises(ValueError, match="缺少训练配置"):
        _validate_checkpoint_config(None, OmegaConf.create({}))
