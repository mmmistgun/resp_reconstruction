import pandas as pd
import pytest
from omegaconf import OmegaConf

from resp_train.data.audit import add_usable_flag, summarize_audit
from resp_train.data.index import filter_index, validate_index_columns


def _rows():
    return pd.DataFrame(
        [
            {
                "dataset_row_id": 1,
                "input_set": "mixed_zscore",
                "split": "train",
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 1,
                "source_npz": "../whole_night/mixed_bcg_to_tho/88/mixed_bcg_to_tho.npz",
                "bcg_signal_key": "bcg_mixed_refined_to_tho_zscore",
                "target_signal_key": "tho_ref",
                "valid_sec_key": "mixed_train_valid_sec",
                "segment_decision": "include_candidate",
                "window_start_sample": 0,
                "window_end_sample": 18000,
                "window_duration_samples": 18000,
                "target_fs": 100,
                "valid_ratio": 1.0,
                "input_finite_ratio": 1.0,
                "target_finite_ratio": 1.0,
                "residual_quality_class": "near_zero_residual",
                "base_alignment_method": "keep_original",
                "apply_decision": "approved",
                "reason": "ok",
            },
            {
                "dataset_row_id": 2,
                "input_set": "legacy_v1",
                "split": "train",
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 2,
                "source_npz": "../whole_night/legacy_v1/88/stage_1_4_applied.npz",
                "bcg_signal_key": "bcg_refined_to_tho",
                "target_signal_key": "tho_ref",
                "valid_sec_key": "valid_after_refinement_sec",
                "segment_decision": "include_candidate",
                "window_start_sample": 0,
                "window_end_sample": 18000,
                "window_duration_samples": 18000,
                "target_fs": 100,
                "valid_ratio": 0.5,
                "input_finite_ratio": 1.0,
                "target_finite_ratio": 1.0,
                "residual_quality_class": "bad",
                "base_alignment_method": "keep_original",
                "apply_decision": "approved",
                "reason": "low valid",
            },
        ]
    )


def test_validate_index_columns_accepts_required_columns():
    validate_index_columns(_rows())


def test_validate_index_columns_raises_for_missing_required_column():
    rows = _rows().drop(columns=["reason"])
    with pytest.raises(ValueError, match="reason"):
        validate_index_columns(rows)


def test_filter_index_selects_input_set_and_split_and_limit():
    cfg = OmegaConf.create({"data": {"input_set": "mixed_zscore"}})
    rows = pd.concat([_rows(), _rows().assign(dataset_row_id=[0, 3])], ignore_index=True)
    filtered = filter_index(rows, cfg, split="train", max_windows=2)
    assert filtered["dataset_row_id"].tolist() == [0, 1]


def test_filter_index_filters_unusable_before_limit_when_configured():
    cfg = OmegaConf.create({"data": {"input_set": "mixed_zscore", "filter_unusable": True}})
    rows = pd.DataFrame(
        [
            {"dataset_row_id": 1, "input_set": "mixed_zscore", "split": "train", "usable": False},
            {"dataset_row_id": 2, "input_set": "mixed_zscore", "split": "train", "usable": True},
            {"dataset_row_id": 3, "input_set": "mixed_zscore", "split": "train", "usable": True},
        ]
    )

    filtered = filter_index(rows, cfg, split="train", max_windows=2)

    assert filtered["dataset_row_id"].tolist() == [2, 3]


def test_add_usable_flag_uses_thresholds_and_classes():
    cfg = OmegaConf.create(
        {
            "data": {
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": ["bad"],
            }
        }
    )
    audited = add_usable_flag(_rows(), cfg)
    assert bool(audited.loc[0, "usable"]) is True
    assert bool(audited.loc[1, "usable"]) is False


def test_summarize_audit_groups_by_split_input_set_quality():
    cfg = OmegaConf.create(
        {
            "data": {
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": [],
            }
        }
    )
    summary = summarize_audit(add_usable_flag(_rows(), cfg))
    assert summary.columns[:6].tolist() == [
        "split",
        "input_set",
        "residual_quality_class",
        "n_windows",
        "n_usable",
        "usable_ratio",
    ]
    assert set(summary.columns) >= {
        "split",
        "input_set",
        "residual_quality_class",
        "n_windows",
        "n_usable",
        "usable_ratio",
    }
    assert summary["n_windows"].sum() == 2
