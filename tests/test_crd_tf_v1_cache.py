from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resp_train.crd.tf_v1_calibration import calibration_implementation_identity
from resp_train.crd.tf_v1_cache import (
    _validate_rows,
    _verify_calibration,
    array_audit,
    inventory_files,
    sha256_json,
)
from resp_train.crd.tf_v1_features import PROTOCOL, fixed_transform_spec


def test_calibration_identity_requires_all_four_passed_components(tmp_path: Path) -> None:
    spec = fixed_transform_spec()
    implementation = calibration_implementation_identity()
    payload = {
        "protocol": PROTOCOL,
        "spec": spec,
        "spec_sha256": sha256_json(spec),
        "calibration_identity_sha256": sha256_json(
            {"spec_sha256": sha256_json(spec), "implementation": implementation}
        ),
        "implementation": implementation,
        "complete": True,
        "passed": True,
        "research_test_used": False,
        "validation_target_used": False,
        "git": {"commit": "test", "dirty": False, "error": None},
        "results": {
            "m": {"passed": True},
            "w": {"passed": True},
            "l": {"passed": True},
            "s": {
                "passed": True,
                "selected": {"smoothness_penalty": 2.0, "suppression_radius_bins": 4},
            },
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _verify_calibration(path)["results"]["s"]["selected"]["suppression_radius_bins"] == 4

    payload["results"]["w"]["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="失败分支"):
        _verify_calibration(path)


def test_cache_rows_must_be_sorted_unique_and_disjoint() -> None:
    train = pd.DataFrame({"dataset_row_id": [1, 2], "split": ["train", "train"]})
    val = pd.DataFrame({"dataset_row_id": [3], "split": ["val"]})
    _validate_rows({"train": train, "val": val}, full_cache=False)

    duplicate_val = pd.DataFrame({"dataset_row_id": [2], "split": ["val"]})
    with pytest.raises(RuntimeError, match="重叠"):
        _validate_rows({"train": train, "val": duplicate_val}, full_cache=False)

    unsorted = pd.DataFrame({"dataset_row_id": [2, 1], "split": ["train", "train"]})
    with pytest.raises(RuntimeError, match="严格递增"):
        _validate_rows({"train": unsorted, "val": val}, full_cache=False)


def test_cache_inventory_records_hash_shape_dtype_and_complex_semantics(tmp_path: Path) -> None:
    real = np.arange(12, dtype=np.float32).reshape(3, 4)
    complex_values = (real + 1j * real).astype(np.complex64)
    np.save(tmp_path / "real.npy", real, allow_pickle=False)
    np.save(tmp_path / "complex.npy", complex_values, allow_pickle=False)

    inventory = inventory_files(tmp_path)

    assert inventory["real.npy"]["shape"] == [3, 4]
    assert inventory["real.npy"]["dtype"] == "float32"
    assert inventory["real.npy"]["summary_semantics"] == "value"
    assert inventory["complex.npy"]["dtype"] == "complex64"
    assert inventory["complex.npy"]["summary_semantics"] == "absolute_value"
    expected = hashlib.sha256((tmp_path / "real.npy").read_bytes()).hexdigest()
    assert inventory["real.npy"]["sha256"] == expected


def test_array_audit_rejects_nonfinite_values() -> None:
    with pytest.raises(FloatingPointError):
        array_audit(np.asarray([1.0, np.inf], dtype=np.float32))
    with pytest.raises(FloatingPointError):
        array_audit(np.asarray([1.0 + 0.0j, np.nan + 1.0j], dtype=np.complex64))
