from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
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

from resp_train.crd.config import load_crd_config
from resp_train.crd.tf_v1_calibration import CALIBRATION_FILENAME, calibration_implementation_identity
from resp_train.crd.tf_v1_features import (
    PROTOCOL,
    RidgeParameters,
    cwt_magnitude_features,
    fixed_transform_spec,
    multires_stft_features,
    one_sided_input_spectrum,
    wsst_energy_map,
    wsst_ridge_features,
)
from resp_train.data.cache import WholeNightCache
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_v1/cache"
DEFAULT_CANDIDATE_LOCK = REPO_ROOT / "docs/experiments/crd_tf_v1_candidate_lock_20260812.json"
EXPECTED_CANDIDATE_LOCK_SHA256 = "c8d4823500e6096fcacb8d2e8787f7b3422160813eabe31adf01b1f1f75cc139"
EXPECTED_SPLIT_COUNTS = {"train": 10141, "val": 2675}


def build_tf_v1_cache(
    *,
    config_path: str | Path,
    calibration_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    max_windows_per_split: int | None = None,
    allow_dirty_smoke: bool = False,
    show_progress: bool = True,
) -> Path:
    """构建不可覆盖的 M/W/S/L-spectrum cache；完整运行只允许 train/validation。"""

    full_cache = max_windows_per_split is None
    if full_cache:
        _assert_clean_repository()
    elif not allow_dirty_smoke:
        raise ValueError("partial cache 只允许显式 --allow-dirty-smoke")
    elif int(max_windows_per_split) <= 0:
        raise ValueError("max_windows_per_split 必须为正")

    cfg = load_crd_config(config_path)
    _validate_data_identity(cfg)
    calibration_path = Path(calibration_path).resolve()
    calibration = _verify_calibration(calibration_path)
    candidate_lock_path = Path(candidate_lock_path).resolve()
    candidate_lock_sha256 = _verify_candidate_lock(candidate_lock_path)

    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows_by_split = {
        "train": filter_index(
            audited,
            cfg,
            split=str(cfg.data.train_split),
            max_windows=max_windows_per_split,
            sample_strategy=str(cfg.data.train_sample_strategy),
            sample_seed=int(cfg.data.train_sample_seed),
        ),
        "val": filter_index(
            audited,
            cfg,
            split=str(cfg.data.val_split),
            max_windows=max_windows_per_split,
            sample_strategy=str(cfg.data.val_sample_strategy),
            sample_seed=int(cfg.data.val_sample_seed),
        ),
    }
    _validate_rows(rows_by_split, full_cache=full_cache)

    calibration_sha256 = sha256_file(calibration_path)
    cache_identity = {
        "protocol": PROTOCOL,
        "fixed_transform_spec": fixed_transform_spec(),
        "calibration_sha256": calibration_sha256,
        "candidate_lock_sha256": candidate_lock_sha256,
        "dataset_index_sha256": sha256_file(index_path),
        "dataset_root": str(Path(str(cfg.data.dataset_root)).resolve()),
        "index_csv": str(cfg.data.index_csv),
        "source_key": str(cfg.data.bcg_input_key),
        "train_sample_seed": int(cfg.data.train_sample_seed),
        "val_sample_seed": int(cfg.data.val_sample_seed),
        "complete": full_cache,
        "max_windows_per_split": max_windows_per_split,
    }
    transform_sha256 = sha256_json(cache_identity)
    suffix = transform_sha256 if full_cache else f"{transform_sha256}-smoke"
    output_dir = Path(output_root).resolve() / suffix
    if output_dir.exists():
        raise FileExistsError(f"CRD-TF cache 禁止覆盖: {output_dir}")

    selected = calibration["results"]["s"]["selected"]
    ridge_parameters = RidgeParameters(
        float(selected["smoothness_penalty"]),
        int(selected["suppression_radius_bins"]),
    )
    source_cache = WholeNightCache(index_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        frequency_identity: dict[str, np.ndarray] = {}
        split_manifests = {}
        for split, rows in rows_by_split.items():
            split_manifests[split] = _write_split_cache(
                split=split,
                rows=rows,
                cfg=cfg,
                source_cache=source_cache,
                output_dir=temporary,
                ridge_parameters=ridge_parameters,
                frequency_identity=frequency_identity,
                show_progress=show_progress,
            )
        for name, values in frequency_identity.items():
            np.save(temporary / f"{name}.npy", np.asarray(values, dtype=np.float64), allow_pickle=False)

        manifest = {
            **cache_identity,
            "transform_sha256": transform_sha256,
            "calibration_path": str(calibration_path),
            "candidate_lock_path": str(candidate_lock_path),
            "dataset_index": str(index_path),
            "research_test_used": False,
            "test_cache_created": False,
            "target_read": False,
            "model_inference_used": False,
            "ridge_parameters": {
                "smoothness_penalty": ridge_parameters.smoothness_penalty,
                "suppression_radius_bins": ridge_parameters.suppression_radius_bins,
            },
            "dependencies": dependency_versions(),
            "git": git_identity(),
            "splits": split_manifests,
        }
        # Manifest 最后写入；文件 inventory 不把 manifest 自身递归纳入 hash。
        manifest["files"] = inventory_files(temporary)
        (temporary / "cache_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / "cache_manifest.json"


def _write_split_cache(
    *,
    split: str,
    rows: pd.DataFrame,
    cfg: DictConfig,
    source_cache: WholeNightCache,
    output_dir: Path,
    ridge_parameters: RidgeParameters,
    frequency_identity: dict[str, np.ndarray],
    show_progress: bool,
) -> dict[str, Any]:
    count = int(len(rows))
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
        "l_spectrum": np.lib.format.open_memmap(
            output_dir / f"{split}_l_spectrum.npy", mode="w+", dtype=np.complex64, shape=(count, 9001)
        ),
    }
    iterator = tqdm(
        rows.itertuples(index=False),
        total=count,
        desc=f"CRD-TF cache {split}",
        disable=not show_progress,
    )
    source_keys: set[str] = set()
    for output_index, row in enumerate(iterator):
        row_id = int(row.dataset_row_id)
        source_key = str(row.bcg_signal_key)
        source_keys.add(source_key)
        start = int(row.window_start_sample)
        end = int(row.window_end_sample)
        if end - start != 18000:
            raise RuntimeError(f"CRD-TF cache row={row_id} 窗口长度不是 18000")
        source = source_cache.get_arrays(str(row.source_npz), [source_key])[source_key]
        waveform = np.asarray(source[start:end], dtype=np.float32)
        if waveform.shape != (18000,) or not np.isfinite(waveform).all():
            raise FloatingPointError(f"CRD-TF cache row={row_id} 输入 shape/finite 错误")

        m = multires_stft_features(waveform)
        w, w_frequencies = cwt_magnitude_features(waveform)
        s_energy, s_frequencies = wsst_energy_map(waveform)
        s = wsst_ridge_features(s_energy, s_frequencies, ridge_parameters)
        l_spectrum = one_sided_input_spectrum(waveform)
        _record_frequency_identity(frequency_identity, "w_frequencies_hz", w_frequencies)
        _record_frequency_identity(frequency_identity, "s_frequencies_hz", s_frequencies)

        arrays["row_ids"][output_index] = row_id
        arrays["m_slow"][output_index] = m["m_slow"]
        arrays["m_fast"][output_index] = m["m_fast"]
        arrays["w"][output_index] = w
        arrays["s"][output_index] = s
        arrays["l_spectrum"][output_index] = l_spectrum

    for array in arrays.values():
        array.flush()
    row_ids = np.asarray(arrays["row_ids"], dtype=np.int64)
    if np.unique(row_ids).size != row_ids.size or not np.all(np.diff(row_ids) > 0):
        raise RuntimeError(f"CRD-TF {split} row_ids 必须严格递增且无重复")
    return {
        "split": split,
        "count": count,
        "samp_id_count": int(rows["samp_id"].nunique()),
        "row_ids_sha256": hashlib.sha256(row_ids.tobytes(order="C")).hexdigest(),
        "source_keys": sorted(source_keys),
        "sample_seed": int(
            cfg.data.train_sample_seed if split == "train" else cfg.data.val_sample_seed
        ),
    }


def _verify_calibration(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.name != CALIBRATION_FILENAME:
        raise FileNotFoundError(f"CRD-TF calibration 文件不存在或名称错误: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("CRD-TF calibration protocol 不一致")
    if payload.get("complete") is not True or payload.get("passed") is not True:
        raise RuntimeError("CRD-TF calibration 未完整通过")
    if payload.get("research_test_used") is not False or payload.get("validation_target_used") is not False:
        raise RuntimeError("CRD-TF calibration 证据边界不合格")
    expected_spec = fixed_transform_spec()
    if payload.get("spec") != expected_spec or payload.get("spec_sha256") != sha256_json(expected_spec):
        raise RuntimeError("CRD-TF calibration transform spec 已漂移")
    implementation = calibration_implementation_identity()
    expected_identity = sha256_json(
        {"spec_sha256": sha256_json(expected_spec), "implementation": implementation}
    )
    if payload.get("implementation") != implementation:
        raise RuntimeError("CRD-TF calibration implementation hash 已漂移")
    if payload.get("calibration_identity_sha256") != expected_identity:
        raise RuntimeError("CRD-TF calibration identity hash 不一致")
    if payload.get("git", {}).get("dirty") is not False:
        raise RuntimeError("CRD-TF calibration 必须来自干净工作树")
    if set(payload.get("results", {})) != {"m", "w", "l", "s"}:
        raise RuntimeError("CRD-TF calibration 四项结果不完整")
    if not all(result.get("passed") is True for result in payload["results"].values()):
        raise RuntimeError("CRD-TF calibration 存在失败分支")
    return payload


def _verify_candidate_lock(path: Path) -> str:
    lock_sha256 = sha256_file(path)
    if lock_sha256 != EXPECTED_CANDIDATE_LOCK_SHA256:
        raise RuntimeError(f"CRD-TF candidate lock hash 不一致: {lock_sha256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("lock_name") != "TF000_C201_ANCHOR" or payload.get("protocol") != PROTOCOL:
        raise RuntimeError("CRD-TF candidate lock identity 不一致")
    for record in payload.get("seeds", []):
        run_dir = REPO_ROOT / str(record["run_dir"])
        for name, metadata in record["files"].items():
            file_path = run_dir / name
            if file_path.stat().st_size != int(metadata["size_bytes"]):
                raise RuntimeError(f"candidate lock 文件大小不一致: {file_path}")
            if sha256_file(file_path) != str(metadata["sha256"]):
                raise RuntimeError(f"candidate lock 文件 hash 不一致: {file_path}")
    if len(payload.get("seeds", [])) != 3:
        raise RuntimeError("CRD-TF candidate lock seed 数不完整")
    return lock_sha256


def _validate_data_identity(cfg: DictConfig) -> None:
    expected = {
        "data.format": "research_v2",
        "data.train_split": "train",
        "data.val_split": "val",
        "data.test_split": "test",
        "data.bcg_input_key": "bcg_rawish_segment_soft_z_key",
        "window.target_fs": 100,
        "window.duration_samples": 18000,
    }
    from omegaconf import OmegaConf

    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise RuntimeError(f"CRD-TF cache 数据 identity 漂移: {key}")
    if cfg.data.get("max_train_windows") is not None or cfg.data.get("max_val_windows") is not None:
        raise RuntimeError("CRD-TF cache 基础 config 必须指向完整 train/validation")


def _validate_rows(rows_by_split: dict[str, pd.DataFrame], *, full_cache: bool) -> None:
    if set(rows_by_split) != {"train", "val"}:
        raise RuntimeError("CRD-TF cache 只允许 train/val")
    all_ids = []
    for split, rows in rows_by_split.items():
        if rows.empty or not rows["split"].eq(split).all():
            raise RuntimeError(f"CRD-TF {split} rows 为空或 split 错误")
        ids = rows["dataset_row_id"].to_numpy(dtype=np.int64)
        if np.unique(ids).size != ids.size or not np.all(np.diff(ids) > 0):
            raise RuntimeError(f"CRD-TF {split} row_ids 必须严格递增且无重复")
        if full_cache and len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise RuntimeError(
                f"CRD-TF {split} 完整窗口数必须为 {EXPECTED_SPLIT_COUNTS[split]}，实际 {len(rows)}"
            )
        all_ids.append(ids)
    if np.intersect1d(all_ids[0], all_ids[1]).size:
        raise RuntimeError("CRD-TF train/validation dataset_row_id 重叠")


def _record_frequency_identity(store: dict[str, np.ndarray], name: str, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64)
    if name not in store:
        store[name] = array.copy()
        return
    if not np.array_equal(store[name], array):
        raise RuntimeError(f"CRD-TF {name} 在 sample 间漂移")


def inventory_files(directory: Path) -> dict[str, dict[str, Any]]:
    inventory = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        record: dict[str, Any] = {
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            record.update(array_audit(array))
        inventory[path.name] = record
    return inventory


def array_audit(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    is_complex = bool(np.iscomplexobj(array))
    semantics = "absolute_value" if is_complex else "value"
    flat = array.reshape(-1)
    count = int(flat.size)
    if count <= 0:
        raise ValueError("CRD-TF cache array 不得为空")
    minimum = float("inf")
    maximum = float("-inf")
    total = 0.0
    total_square = 0.0
    # 大型 memmap 必须分块审计，避免为数 GiB cache 复制完整 float64/finite 数组。
    for start in range(0, count, 1_000_000):
        block = np.asarray(flat[start : start + 1_000_000])
        if is_complex:
            finite = bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
            observed = np.abs(block).astype(np.float64)
        else:
            finite = bool(np.isfinite(block).all())
            observed = block.astype(np.float64)
        if not finite:
            raise FloatingPointError("CRD-TF cache array 包含 NaN/Inf")
        minimum = min(minimum, float(observed.min()))
        maximum = max(maximum, float(observed.max()))
        total += float(observed.sum(dtype=np.float64))
        total_square += float(np.square(observed).sum(dtype=np.float64))
    mean = total / count
    variance = max(0.0, total_square / count - mean * mean)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "finite": True,
        "summary_semantics": semantics,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": math.sqrt(variance),
    }


def dependency_versions() -> dict[str, str]:
    distributions = ("numpy", "torch", "scipy", "PyWavelets", "ssqueezepy")
    return {distribution: importlib.metadata.version(distribution) for distribution in distributions}


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "error": None if commit.returncode == 0 and status.returncode == 0 else "git identity failed",
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"无法检查 Git 工作树: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("完整 CRD-TF cache 要求干净 Git 工作树")
