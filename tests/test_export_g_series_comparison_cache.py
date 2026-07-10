from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts.export_g_series_comparison_cache import (
    REQUIRED_MODELS,
    atomic_save_npy,
    build_export_plan,
    build_prediction_tasks,
    load_comparison_specs,
    parse_args,
    sha256_file,
    spec_fingerprint,
    validate_prediction_batch,
    validate_config_consistency,
    validate_signal_matrix,
    write_cache_arrays,
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


def test_write_cache_arrays_are_memmap_readable(tmp_path: Path) -> None:
    ids = np.asarray([11, 12, 13], dtype=np.int64)
    bcg = np.arange(24, dtype=np.float32).reshape(3, 8)
    tho = -bcg

    paths = write_cache_arrays(
        tmp_path,
        dataset_row_id=ids,
        bcg_input=bcg,
        tho_ref=tho,
        predictions={"g0_time_only": tho + 0.25},
    )

    assert np.load(paths["bcg_input"], mmap_mode="r").shape == (3, 8)
    np.testing.assert_array_equal(np.load(paths["dataset_row_id"]), ids)
    np.testing.assert_array_equal(np.load(paths["prediction:g0_time_only"]), tho + 0.25)


def test_validate_prediction_batch_rejects_target_or_id_drift() -> None:
    ids = np.asarray([1, 2], dtype=np.int64)
    target = np.ones((2, 4), dtype=np.float32)

    validate_prediction_batch(ids, ids, target, target)

    with pytest.raises(ValueError, match="dataset_row_id 顺序"):
        validate_prediction_batch(ids[::-1], ids, target, target)
    with pytest.raises(ValueError, match="THO target"):
        validate_prediction_batch(ids, ids, target + 1, target)


def test_validate_config_consistency_rejects_changed_test_definition() -> None:
    base = OmegaConf.create(
        {
            "data": {
                "dataset_root": "/dataset",
                "index_csv": "index.csv",
                "format": "research_v2",
                "input_set": "research_v2_waveform",
                "test_split": "test",
                "max_test_windows": None,
                "test_sample_strategy": "stratified_random",
                "test_sample_seed": 1,
            },
            "window": {"target_fs": 100.0},
            "loss": {"spectrum_low_hz": 0.05, "spectrum_high_hz": 0.7},
        }
    )
    changed = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    changed.data.test_sample_seed = 2

    validate_config_consistency(base, [base])

    with pytest.raises(ValueError, match="data.test_sample_seed"):
        validate_config_consistency(base, [changed])


def test_build_prediction_tasks_assigns_devices_round_robin(tmp_path: Path) -> None:
    specs = load_comparison_specs(_write_spec(tmp_path))
    tasks = build_prediction_tasks(
        specs,
        output_dir=tmp_path / "cache",
        expected_row_ids=np.asarray([1, 2], dtype=np.int64),
        devices=["cuda:0", "cuda:1"],
    )

    assert [task.device for task in tasks] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert [task.output_path.name for task in tasks] == ["r_tho_hat.npy"] * 4


def test_parse_args_accepts_dry_run_without_creating_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_g_series_comparison_cache.py",
            "--spec",
            "spec.csv",
            "--output-dir",
            "cache",
            "--device",
            "cuda:0",
            "--dry-run",
        ],
    )

    args = parse_args()

    assert args.spec == Path("spec.csv")
    assert args.output_dir == Path("cache")
    assert args.devices == ["cuda:0"]
    assert args.dry_run is True


def test_build_export_plan_resume_requires_matching_complete_manifest(tmp_path: Path) -> None:
    specs = load_comparison_specs(_write_spec(tmp_path))
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps({"status": "complete", "spec_fingerprint": spec_fingerprint(specs)}),
        encoding="utf-8",
    )

    plan = build_export_plan(specs, output_dir=output_dir, devices=["cuda:0"], resume=True)

    assert plan.output_dir == output_dir
    (output_dir / "manifest.json").write_text(
        json.dumps({"status": "complete", "spec_fingerprint": "other"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="spec_fingerprint"):
        build_export_plan(specs, output_dir=output_dir, devices=["cuda:0"], resume=True)
