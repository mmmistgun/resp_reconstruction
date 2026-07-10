from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.export_harmonic_predictions import (
    build_export_tasks,
    load_positive_row_ids,
    save_prediction_payload,
)


def _write_spec(tmp_path: Path) -> Path:
    records = []
    labels = ["time", "f0", "wide", "bandenergy"]
    for label in labels:
        for seed in (101, 102, 103):
            run_dir = tmp_path / "runs" / f"{label}_{seed}"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"not-loaded-during-dry-run")
            (run_dir / "config.yaml").write_text("training:\n  device: auto\n", encoding="utf-8")
            records.append({"label": label, "seed": seed, "checkpoint": str(checkpoint)})
    spec_path = tmp_path / "spec.csv"
    pd.DataFrame.from_records(records).to_csv(spec_path, index=False)
    return spec_path


def _write_labels(tmp_path: Path) -> Path:
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "dataset_row_id": [1, 2, 3],
            "split": "test",
            "harmonic_positive": [True, False, True],
        }
    ).to_csv(path, index=False)
    return path


def test_load_positive_row_ids_rejects_duplicates_and_empty(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path)
    assert load_positive_row_ids(labels_path) == [1, 3]

    duplicate = pd.read_csv(labels_path)
    duplicate.loc[2, "dataset_row_id"] = 1
    duplicate.to_csv(labels_path, index=False)
    with pytest.raises(ValueError, match="重复 dataset_row_id"):
        load_positive_row_ids(labels_path)

    duplicate["dataset_row_id"] = [1, 2, 3]
    duplicate["harmonic_positive"] = False
    duplicate.to_csv(labels_path, index=False)
    with pytest.raises(ValueError, match="没有阳性"):
        load_positive_row_ids(labels_path)


def test_build_export_tasks_dry_run_has_twelve_unique_outputs_and_round_robin_devices(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    labels_path = _write_labels(tmp_path)

    tasks = build_export_tasks(
        spec_path=spec_path,
        labels_path=labels_path,
        output_dir=tmp_path / "predictions",
        devices=["cuda:0", "cuda:1"],
    )

    assert len(tasks) == 12
    assert [task.device for task in tasks[:4]] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert len({task.output_path for task in tasks}) == 12
    assert all(task.n_selected_rows == 2 for task in tasks)
    assert all(task.config_path.name == "config.yaml" for task in tasks)
    assert not any(task.output_path.exists() for task in tasks)


def test_save_prediction_payload_writes_exact_schema_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "prediction.npz"
    row_ids = np.asarray([7, 9], dtype=np.int64)
    predictions = np.arange(16, dtype=np.float32).reshape(2, 8)
    targets = -predictions

    manifest_path = save_prediction_payload(
        output,
        dataset_row_id=row_ids,
        r_tho_hat=predictions,
        tho_ref=targets,
        manifest={"label": "wide", "seed": 101, "labels_sha256": "abc"},
    )

    with np.load(output, allow_pickle=False) as blob:
        assert set(blob.files) == {"dataset_row_id", "r_tho_hat", "tho_ref"}
        np.testing.assert_array_equal(blob["dataset_row_id"], row_ids)
        np.testing.assert_array_equal(blob["r_tho_hat"], predictions)
        np.testing.assert_array_equal(blob["tho_ref"], targets)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["label"] == "wide"
    assert manifest["n_windows"] == 2
    assert manifest["array_schema"]["r_tho_hat"] == [2, 8]

    with pytest.raises(FileExistsError, match="已存在"):
        save_prediction_payload(
            output,
            dataset_row_id=row_ids,
            r_tho_hat=predictions,
            tho_ref=targets,
            manifest={"label": "wide", "seed": 101},
        )
