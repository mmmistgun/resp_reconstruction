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


def _sampling_rows():
    base = _rows().iloc[0].to_dict()
    classes = [
        "near_zero_residual",
        "near_zero_residual",
        "near_zero_residual",
        "near_zero_residual",
        "near_zero_residual",
        "stable_nonzero_residual",
        "stable_nonzero_residual",
        "stable_nonzero_residual",
        "stable_nonzero_residual",
        "high_residual",
        "high_residual",
        "high_residual",
    ]
    rows = []
    for index, residual_class in enumerate(classes, start=1):
        row = base.copy()
        row.update(
            {
                "dataset_row_id": index,
                "split": "train",
                "window_id_in_segment": index,
                "window_start_sample": (index - 1) * 18000,
                "window_end_sample": index * 18000,
                "residual_quality_class": residual_class,
                "usable": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


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


def test_filter_index_defaults_to_stratified_random():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "stratified_random",
                "train_sample_seed": 42,
                "stratify_column": "residual_quality_class",
            }
        }
    )

    filtered = filter_index(_sampling_rows(), cfg, split="train", max_windows=5)

    assert len(filtered) == 5
    assert set(filtered["residual_quality_class"]) == {
        "near_zero_residual",
        "stable_nonzero_residual",
        "high_residual",
    }
    assert filtered["dataset_row_id"].tolist() == sorted(filtered["dataset_row_id"].tolist())


def test_filter_index_random_is_reproducible():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "random",
                "train_sample_seed": 123,
                "stratify_column": "residual_quality_class",
            }
        }
    )

    first = filter_index(_sampling_rows(), cfg, split="train", max_windows=4)
    second = filter_index(_sampling_rows(), cfg, split="train", max_windows=4)

    assert first["dataset_row_id"].tolist() == second["dataset_row_id"].tolist()
    assert first["dataset_row_id"].tolist() != [1, 2, 3, 4]


def test_filter_index_head_is_debug_prefix_only_when_explicit():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "head",
                "train_sample_seed": 42,
                "stratify_column": "residual_quality_class",
            }
        }
    )

    filtered = filter_index(_sampling_rows(), cfg, split="train", max_windows=4)

    assert filtered["dataset_row_id"].tolist() == [1, 2, 3, 4]


def test_filter_index_uses_independent_val_strategy_and_seed():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "random",
                "val_sample_strategy": "random",
                "train_sample_seed": 1,
                "val_sample_seed": 2,
                "stratify_column": "residual_quality_class",
            }
        }
    )
    train_rows = _sampling_rows().assign(split="train")
    val_rows = _sampling_rows().assign(split="val")
    rows = pd.concat([train_rows, val_rows], ignore_index=True)

    train = filter_index(rows, cfg, split="train", max_windows=4)
    val = filter_index(rows, cfg, split="val", max_windows=4)

    assert train["dataset_row_id"].tolist() != val["dataset_row_id"].tolist()


def test_filter_index_rejects_missing_stratify_column():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "stratified_random",
                "train_sample_seed": 42,
                "stratify_column": "missing_column",
            }
        }
    )

    with pytest.raises(ValueError, match="missing_column"):
        filter_index(_sampling_rows(), cfg, split="train", max_windows=5)


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
