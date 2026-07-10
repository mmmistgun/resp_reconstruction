from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.export_g_series_comparison_cache import (
    REQUIRED_MODELS,
    atomic_save_npy,
    build_export_plan,
    load_comparison_specs,
    sha256_file,
    validate_signal_matrix,
)


def _write_spec(tmp_path: Path) -> Path:
    checkpoints: list[Path] = []
    for label in REQUIRED_MODELS:
        checkpoint = tmp_path / label / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        checkpoints.append(checkpoint)

    spec_path = tmp_path / "comparison.csv"
    spec_path.write_text(
        "label,seed,checkpoint,selection_source\n"
        f"g0_time_only,20260837,{checkpoints[0]},validation_topk_legacy_task_selection\n"
        f"g0_f0_native_stft_pre_mixer,20260700,{checkpoints[1]},validation_topk_legacy_task_selection\n"
        f"g3_c_wide_8p0,20260700,{checkpoints[2]},validation_topk_legacy_task_selection\n"
        f"g3_c_bandenergy,20260700,{checkpoints[3]},validation_topk_legacy_task_selection\n",
        encoding="utf-8",
    )
    return spec_path


def test_load_comparison_specs_requires_exact_frozen_models(tmp_path: Path) -> None:
    specs = load_comparison_specs(_write_spec(tmp_path))

    assert [item.label for item in specs] == list(REQUIRED_MODELS)
    assert [item.seed for item in specs] == [20260837, 20260700, 20260700, 20260700]
    assert all(item.selection_source == "validation_topk_legacy_task_selection" for item in specs)


def test_load_comparison_specs_rejects_unknown_label(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    content = spec_path.read_text(encoding="utf-8").replace("g3_c_bandenergy", "unknown_model")
    spec_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="label"):
        load_comparison_specs(spec_path)


def test_build_export_plan_rejects_existing_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "cache"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="输出目录已存在"):
        build_export_plan(
            load_comparison_specs(_write_spec(tmp_path)),
            output_dir=output_dir,
            devices=["cuda:0"],
        )


def test_validate_signal_matrix_normalizes_channel_dimension_and_rejects_invalid_values() -> None:
    values = np.ones((2, 1, 4), dtype=np.float64)

    normalized = validate_signal_matrix(values, name="prediction", expected_rows=2)

    assert normalized.shape == (2, 4)
    assert normalized.dtype == np.float32
    with pytest.raises(ValueError, match="非有限值"):
        validate_signal_matrix(np.asarray([[np.nan]]), name="prediction")
    with pytest.raises(ValueError, match="行数不一致"):
        validate_signal_matrix(np.ones((2, 4)), name="prediction", expected_rows=3)


def test_atomic_save_npy_writes_hashable_array_without_temp_files(tmp_path: Path) -> None:
    output = tmp_path / "array.npy"
    values = np.arange(8, dtype=np.float32).reshape(2, 4)

    atomic_save_npy(output, values)

    np.testing.assert_array_equal(np.load(output), values)
    assert len(sha256_file(output)) == 64
    assert not list(tmp_path.glob("*.tmp.npy"))
