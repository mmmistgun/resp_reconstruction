import pandas as pd

from resp_train.data.independence import audit_split_independence


def test_audit_split_independence_detects_samp_and_segment_overlap():
    train = pd.DataFrame(
        [
            {"dataset_row_id": 1, "samp_id": 88, "segment_id": "a", "allowed_losses": "waveform", "valid_ratio": 0.9},
            {"dataset_row_id": 2, "samp_id": 89, "segment_id": "b", "allowed_losses": "rate", "valid_ratio": 0.8},
        ]
    )
    val = pd.DataFrame(
        [
            {"dataset_row_id": 3, "samp_id": 88, "segment_id": "a", "allowed_losses": "waveform", "valid_ratio": 0.95},
            {"dataset_row_id": 4, "samp_id": 90, "segment_id": "c", "allowed_losses": "rate", "valid_ratio": 0.7},
        ]
    )

    report = audit_split_independence(train, val)

    summary = report["summary"]
    assert int(summary.loc[0, "train_windows"]) == 2
    assert int(summary.loc[0, "val_windows"]) == 2
    assert int(summary.loc[0, "overlap_samp_id_count"]) == 1
    assert int(summary.loc[0, "overlap_segment_count"]) == 1
    assert bool(summary.loc[0, "has_samp_id_leakage"]) is True
    assert bool(summary.loc[0, "has_segment_leakage"]) is True


def test_audit_split_independence_reports_distribution_shift():
    train = pd.DataFrame(
        [
            {"dataset_row_id": 1, "samp_id": 1, "segment_id": "a", "allowed_losses": "waveform", "valid_ratio": 0.9},
            {"dataset_row_id": 2, "samp_id": 2, "segment_id": "b", "allowed_losses": "waveform", "valid_ratio": 0.8},
        ]
    )
    val = pd.DataFrame(
        [
            {"dataset_row_id": 3, "samp_id": 3, "segment_id": "c", "allowed_losses": "rate", "valid_ratio": 0.4},
        ]
    )

    report = audit_split_independence(
        train,
        val,
        categorical_columns=("allowed_losses",),
        numeric_columns=("valid_ratio",),
    )

    distribution = report["categorical_distribution"]
    numeric = report["numeric_distribution"]
    assert set(distribution["column"]) == {"allowed_losses"}
    assert set(distribution["split"]) == {"train", "val"}
    assert numeric.loc[numeric["split"].eq("train"), "valid_ratio_mean"].iloc[0] == 0.85
    assert numeric.loc[numeric["split"].eq("val"), "valid_ratio_mean"].iloc[0] == 0.4


def test_audit_split_independence_handles_empty_and_all_nan_segments():
    columns = ["dataset_row_id", "samp_id", "segment_id"]
    train = pd.DataFrame(columns=columns)
    val = pd.DataFrame(
        [
            {"dataset_row_id": 1, "samp_id": None, "segment_id": None},
        ]
    )

    report = audit_split_independence(train, val)

    summary = report["summary"]
    assert int(summary.loc[0, "train_windows"]) == 0
    assert int(summary.loc[0, "val_windows"]) == 1
    assert int(summary.loc[0, "overlap_samp_id_count"]) == 0
    assert int(summary.loc[0, "overlap_segment_count"]) == 0
    assert list(report["overlap_samp_id"].columns) == ["samp_id"]
    assert list(report["overlap_segment"].columns) == ["segment_key"]
    assert list(report["categorical_distribution"].columns) == ["column", "split", "value", "count", "ratio"]
    assert list(report["per_samp_id"].columns) == ["split", "samp_id", "window_count"]


def test_audit_split_independence_requires_samp_and_segment_columns():
    train = pd.DataFrame([{"samp_id": 1, "segment_id": "a"}])
    val = pd.DataFrame([{"samp_id": 1}])

    try:
        audit_split_independence(train, val)
    except ValueError as exc:
        assert "segment_id" in str(exc)
    else:
        raise AssertionError("缺少 segment_id 时应报错")


def test_audit_split_independence_normalizes_mixed_numeric_id_types():
    train = pd.DataFrame(
        [
            {"dataset_row_id": 1, "samp_id": 88, "segment_id": 10},
            {"dataset_row_id": 2, "samp_id": "subject-a", "segment_id": "seg-1"},
        ]
    )
    val = pd.DataFrame(
        [
            {"dataset_row_id": 3, "samp_id": 88.0, "segment_id": 10.0},
            {"dataset_row_id": 4, "samp_id": "subject-a", "segment_id": "seg-2"},
        ]
    )

    report = audit_split_independence(train, val)

    summary = report["summary"]
    assert int(summary.loc[0, "overlap_samp_id_count"]) == 2
    assert int(summary.loc[0, "overlap_segment_count"]) == 1
    assert set(report["overlap_samp_id"]["samp_id"]) == {"88", "subject-a"}
    assert set(report["overlap_segment"]["segment_key"]) == {"88::10"}
