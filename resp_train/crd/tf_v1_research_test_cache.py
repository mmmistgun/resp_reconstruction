from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from resp_train.crd.config import CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION, load_crd_config
from resp_train.crd.tf_v1_cache import (
    EXPECTED_CANDIDATE_LOCK_SHA256,
    _verify_calibration,
    _verify_candidate_lock,
    dependency_versions,
    git_identity,
    inventory_files,
    sha256_file,
    sha256_json,
)
from resp_train.crd.tf_v1_features import (
    RidgeParameters,
    cwt_magnitude_features,
    fixed_transform_spec,
    multires_stft_features,
    wsst_energy_map,
    wsst_ridge_features,
)
from resp_train.data.cache import WholeNightCache
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/crd_tf_v1/crd_tf203_ms_formal.yaml"
DEFAULT_CALIBRATION = REPO_ROOT / (
    "runs/crd_tf_v1/calibration/"
    "7e29795edc13fe8dc2e12ada8d619c22d0ae19fa8d13feb729d2f9b261fd5535/"
    "calibration.json"
)
DEFAULT_CANDIDATE_LOCK = REPO_ROOT / "docs/experiments/crd_tf_v1_candidate_lock_20260812.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_v1/research_test_cache"
P5_SUMMARY = REPO_ROOT / "runs/crd_tf_v1/p5_validation_summary/p5_summary.json"
P5_SUMMARY_SHA256 = "afb6feba1600c9c5e07d713db9cac7c886aa87a033037649d3e9ac32cb753b4e"
EXPECTED_TEST_COUNT = 2310
RESEARCH_TEST_REPRESENTATIONS = ("m", "w", "s")


def build_tf_v1_research_test_cache(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    show_progress: bool = True,
) -> Path:
    """只从冻结 research-test 窗口的 BCG 输入构建 M/W/S cache。"""

    _assert_clean_repository()
    cfg = load_crd_config(config_path)
    _validate_config(cfg)
    calibration_path = Path(calibration_path).resolve()
    calibration = _verify_calibration(calibration_path)
    candidate_lock_path = Path(candidate_lock_path).resolve()
    candidate_lock_sha256 = _verify_candidate_lock(candidate_lock_path)
    if candidate_lock_sha256 != EXPECTED_CANDIDATE_LOCK_SHA256:
        raise RuntimeError("research-test candidate lock SHA-256 不一致")
    if sha256_file(P5_SUMMARY) != P5_SUMMARY_SHA256:
        raise RuntimeError("research-test P5 candidate summary SHA-256 不一致")

    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = filter_index(
        audited,
        cfg,
        split=str(cfg.data.test_split),
        max_windows=None,
        sample_strategy=str(cfg.data.test_sample_strategy),
        sample_seed=int(cfg.data.test_sample_seed),
    )
    _validate_test_rows(rows, str(cfg.data.test_split))
    calibration_sha256 = sha256_file(calibration_path)
    identity = {
        "protocol": CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION,
        "fixed_transform_spec": fixed_transform_spec(),
        "calibration_sha256": calibration_sha256,
        "candidate_lock_sha256": candidate_lock_sha256,
        "p5_summary_sha256": P5_SUMMARY_SHA256,
        "candidate_pool": ["crd_tf101_m", "crd_tf102_w", "crd_tf203_ms"],
        "dataset_index_sha256": sha256_file(index_path),
        "dataset_root": str(Path(str(cfg.data.dataset_root)).resolve()),
        "index_csv": str(cfg.data.index_csv),
        "source_key": str(cfg.data.bcg_input_key),
        "split": str(cfg.data.test_split),
        "test_sample_seed": int(cfg.data.test_sample_seed),
        "representations": list(RESEARCH_TEST_REPRESENTATIONS),
        "complete": True,
    }
    cache_identity_sha256 = sha256_json(identity)
    output_dir = Path(output_root).resolve() / cache_identity_sha256
    if output_dir.exists():
        raise FileExistsError(f"research-test cache 禁止覆盖: {output_dir}")
    selected = calibration["results"]["s"]["selected"]
    ridge_parameters = RidgeParameters(
        float(selected["smoothness_penalty"]), int(selected["suppression_radius_bins"])
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        frequency_identity: dict[str, np.ndarray] = {}
        split_manifest = _write_test_cache(
            rows=rows,
            cfg=cfg,
            index_path=index_path,
            output_dir=temporary,
            ridge_parameters=ridge_parameters,
            frequency_identity=frequency_identity,
            show_progress=show_progress,
        )
        for name, values in frequency_identity.items():
            np.save(temporary / f"{name}.npy", np.asarray(values, dtype=np.float64), allow_pickle=False)
        manifest = {
            **identity,
            "cache_identity_sha256": cache_identity_sha256,
            "calibration_path": str(calibration_path),
            "candidate_lock_path": str(candidate_lock_path),
            "dataset_index": str(index_path),
            "research_test_used": True,
            "research_test_input_used": True,
            "test_target_array_read": False,
            "target_read": False,
            "test_cache_created": True,
            "model_inference_used": False,
            "ridge_parameters": {
                "smoothness_penalty": ridge_parameters.smoothness_penalty,
                "suppression_radius_bins": ridge_parameters.suppression_radius_bins,
            },
            "dependencies": dependency_versions(),
            "git": git_identity(),
            "splits": {str(cfg.data.test_split): split_manifest},
        }
        manifest["files"] = inventory_files(temporary)
        (temporary / "cache_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / "cache_manifest.json"


def _write_test_cache(
    *,
    rows: pd.DataFrame,
    cfg: DictConfig,
    index_path: Path,
    output_dir: Path,
    ridge_parameters: RidgeParameters,
    frequency_identity: dict[str, np.ndarray],
    show_progress: bool,
) -> dict[str, Any]:
    split = str(cfg.data.test_split)
    count = len(rows)
    arrays = {
        "row_ids": np.lib.format.open_memmap(
            output_dir / f"{split}_row_ids.npy", mode="w+", dtype=np.int64, shape=(count,)
        ),
        "m_slow": np.lib.format.open_memmap(
            output_dir / f"{split}_m_slow.npy", mode="w+", dtype=np.float32, shape=(count, 36, 101)
        ),
        "m_fast": np.lib.format.open_memmap(
            output_dir / f"{split}_m_fast.npy", mode="w+", dtype=np.float32, shape=(count, 44, 349)
        ),
        "w": np.lib.format.open_memmap(
            output_dir / f"{split}_w.npy", mode="w+", dtype=np.float32, shape=(count, 97, 360)
        ),
        "s": np.lib.format.open_memmap(
            output_dir / f"{split}_s.npy", mode="w+", dtype=np.float32, shape=(count, 12, 360)
        ),
    }
    source_cache = WholeNightCache(index_path)
    source_keys: set[str] = set()
    iterator = tqdm(
        rows.itertuples(index=False), total=count, desc="CRD-TF research-test cache", disable=not show_progress
    )
    for output_index, row in enumerate(iterator):
        row_id = int(row.dataset_row_id)
        source_key = str(row.bcg_signal_key)
        source_keys.add(source_key)
        start, end = int(row.window_start_sample), int(row.window_end_sample)
        if end - start != 18000:
            raise RuntimeError(f"research-test cache row={row_id} 窗口长度不是 18000")
        source = source_cache.get_arrays(str(row.source_npz), [source_key])[source_key]
        waveform = np.asarray(source[start:end], dtype=np.float32)
        if waveform.shape != (18000,) or not np.isfinite(waveform).all():
            raise FloatingPointError(f"research-test cache row={row_id} 输入 shape/finite 错误")
        m = multires_stft_features(waveform)
        w, w_frequencies = cwt_magnitude_features(waveform)
        s_energy, s_frequencies = wsst_energy_map(waveform)
        s = wsst_ridge_features(s_energy, s_frequencies, ridge_parameters)
        _record_frequency_identity(frequency_identity, "w_frequencies_hz", w_frequencies)
        _record_frequency_identity(frequency_identity, "s_frequencies_hz", s_frequencies)
        arrays["row_ids"][output_index] = row_id
        arrays["m_slow"][output_index] = m["m_slow"]
        arrays["m_fast"][output_index] = m["m_fast"]
        arrays["w"][output_index] = w
        arrays["s"][output_index] = s
    for array in arrays.values():
        array.flush()
    row_ids = np.asarray(arrays["row_ids"], dtype=np.int64)
    if np.unique(row_ids).size != row_ids.size or not np.all(np.diff(row_ids) > 0):
        raise RuntimeError("research-test cache row_ids 必须严格递增且无重复")
    return {
        "split": split,
        "count": count,
        "samp_id_count": int(rows["samp_id"].nunique()),
        "row_ids_sha256": hashlib.sha256(row_ids.tobytes(order="C")).hexdigest(),
        "source_keys": sorted(source_keys),
        "sample_seed": int(cfg.data.test_sample_seed),
    }


def _validate_config(cfg: DictConfig) -> None:
    if (
        str(cfg.data.test_split) != "test"
        or str(cfg.data.bcg_input_key) != "bcg_rawish_segment_soft_z_key"
        or int(cfg.window.target_fs) != 100
        or int(cfg.window.duration_samples) != 18000
        or cfg.data.get("max_test_windows") is not None
    ):
        raise RuntimeError("research-test cache config identity 不合格")


def _validate_test_rows(rows: pd.DataFrame, split: str) -> None:
    if len(rows) != EXPECTED_TEST_COUNT or not rows["split"].eq(split).all():
        raise RuntimeError(f"research-test 完整窗口数必须为 {EXPECTED_TEST_COUNT}，实际 {len(rows)}")
    row_ids = rows["dataset_row_id"].to_numpy(dtype=np.int64)
    if np.unique(row_ids).size != row_ids.size or not np.all(np.diff(row_ids) > 0):
        raise RuntimeError("research-test rows 必须严格递增且无重复")


def _record_frequency_identity(store: dict[str, np.ndarray], name: str, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64)
    if name not in store:
        store[name] = array.copy()
    elif not np.array_equal(store[name], array):
        raise RuntimeError(f"research-test {name} 在 sample 间漂移")


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if status.returncode != 0:
        raise RuntimeError(f"无法检查 Git 工作树: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("完整 research-test cache 要求干净 Git 工作树")

