"""W0 CWT-FiLM 行为分析的来源锁、GPU 推理、汇总和案例生命周期。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import check_crd_dependencies, load_crd_config
from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.crd.model import build_crd_model
from resp_train.crd.tf_v1_data import batch_tf_to_device
from resp_train.crd.tf_w_v2_audit import _w0_seed_entries
from resp_train.data.factory import build_window_data
from resp_train.paper_evidence.w0_cwt_film_behavior import (
    BOUNDED_HISTOGRAM_BINS,
    BOUNDED_HISTOGRAM_RANGE,
    CHANNELS,
    EFFECTIVE_HISTOGRAM_BINS,
    EFFECTIVE_HISTOGRAM_RANGE,
    ERROR_COLUMNS,
    LATENT_FRAMES,
    MAIN_TAU,
    PRIMARY_ASSOCIATION_COLUMNS,
    PROTOCOL,
    QUALITY_COLUMNS,
    QUALITY_GROUP_COLUMN,
    RAW_HISTOGRAM_BINS,
    RAW_HISTOGRAM_RANGE,
    SEEDS,
    TAUS,
    TIME_BIN_COUNT,
    TIME_BIN_FRAMES,
    WINDOW_COUNT,
    association_table,
    compute_batch_statistics,
    forward_with_capture,
    join_frozen_sources,
    quality_group_summary,
    saturation_summary,
    select_typical_windows,
    subject_summary,
    summarize_seed_variation,
    validate_historical_film,
    validate_window_identity,
)


CODE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get("RESP_RECONSTRUCTION_SOURCE_ROOT", "/mnt/disk_code/marques/resp_reconstruction")
).resolve()
OUTPUT_ROOT = SOURCE_ROOT / "runs/w0_cwt_film_behavior_v1"

DOCS = Path("docs/experiments")
SOURCE_LOCK = DOCS / "crd_tf_w_v2_candidate_lock_20260817.json"
SOURCE_LOCK_SHA256 = "6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6"
METRIC_LOCK = DOCS / "e3_w0_metric_association_lock_20260916.json"
METRIC_LOCK_SHA256 = "f9fadfc620ab07b6d58bb7508ed441f923def01dc1a948a029190fbdc9e094bc"
HISTORICAL_FILM = Path(
    "runs/crd_tf_w_v2/p_minus_1_film_statistics_correction/film_statistics_corrected.csv"
)
HISTORICAL_FILM_SHA256 = "2b7a4edef8e9828356a336c3c1ec7880235914207b00d164bf016ce1cd7a5203"
PREVIOUS_IMPLEMENTATION_LOCK = DOCS / "w0_cwt_film_behavior_implementation_lock_20260918.json"
IMPLEMENTATION_LOCK = DOCS / "w0_cwt_film_behavior_implementation_lock_20260918_r2.json"
PROTOCOL_PATH = DOCS / "w0_cwt_film_behavior_protocol_20260918.md"
SCRIPT_PATH = Path("scripts/analyze_w0_cwt_film_behavior.py")
CORE_PATH = Path("resp_train/paper_evidence/w0_cwt_film_behavior.py")
RUNTIME_PATH = Path("resp_train/paper_evidence/w0_cwt_film_behavior_runtime.py")
TEST_PATH = Path("tests/test_w0_cwt_film_behavior.py")
README_PATH = Path("scripts/README.md")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def verify_file(path: Path, expected: Mapping[str, Any]) -> None:
    expected_identity = {key: expected[key] for key in ("size_bytes", "sha256")}
    if not path.is_file() or identity(path) != expected_identity:
        raise RuntimeError(f"FiLM 分析文件身份漂移: {path}")


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git_state(*, require_clean: bool = False) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=CODE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    if require_clean and status:
        raise RuntimeError("FiLM 正式执行要求独立 worktree 为干净提交")
    return {"commit": commit, "status_porcelain": status}


def environment(device: str | None) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "torch", "omegaconf", "mamba-ssm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    output: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "device": device,
        "torch_num_threads": int(torch.get_num_threads()),
    }
    if device is not None and torch.device(device).type == "cuda":
        output.update(
            {
                "gpu_name": torch.cuda.get_device_name(torch.device(device)),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            }
        )
    return output


def require_gpu(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("真实 W0 FiLM 分析要求 CUDA；CPU 只允许 synthetic 测试与汇总")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("CRD 原生依赖检查失败: " + "; ".join(problems))
    return resolved


def _expected_metric_sources(metric_lock: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    output: dict[int, Mapping[str, Any]] = {}
    source_files = metric_lock["source_files"]
    for entry in metric_lock["entries"]:
        if entry["split"] != "validation":
            continue
        seed = int(entry["seed"])
        path = str(Path(entry["metrics"]).resolve())
        if path not in source_files:
            raise RuntimeError(f"E3 锁缺少 validation metrics identity: {path}")
        output[seed] = {
            "path": path,
            "identity": source_files[path],
            "selected_epoch": int(entry["selected_epoch"]),
            "checkpoint_sha256": entry["checkpoint_sha256"],
        }
    if set(output) != set(SEEDS):
        raise RuntimeError("E3 锁的 validation metrics seed 不完整")
    return output


def prepare_lock(*, code_root: Path = CODE_ROOT, source_root: Path = SOURCE_ROOT) -> Path:
    """核验冻结来源并创建不可覆盖实现锁；会读取大文件计算 SHA-256。"""

    destination = code_root / IMPLEMENTATION_LOCK
    if destination.exists():
        raise FileExistsError(f"FiLM implementation lock 已存在: {destination}")
    source_lock_path = code_root / SOURCE_LOCK
    metric_lock_path = code_root / METRIC_LOCK
    if sha256_file(source_lock_path) != SOURCE_LOCK_SHA256:
        raise RuntimeError("W0 candidate lock identity 漂移")
    if sha256_file(metric_lock_path) != METRIC_LOCK_SHA256:
        raise RuntimeError("E3 metric lock identity 漂移")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    metric_lock = json.loads(metric_lock_path.read_text(encoding="utf-8"))
    entries = _w0_seed_entries(source_lock)
    metric_sources = _expected_metric_sources(metric_lock)
    source_files: dict[str, dict[str, Any]] = {
        str(source_lock_path.resolve()): identity(source_lock_path),
        str(metric_lock_path.resolve()): identity(metric_lock_path),
    }

    for entry in entries:
        seed = int(entry["seed"])
        if metric_sources[seed]["checkpoint_sha256"] != entry["checkpoint"]["sha256"]:
            raise RuntimeError(f"W0/E3 checkpoint identity 不一致 seed={seed}")
        if metric_sources[seed]["selected_epoch"] != int(entry["selected_epoch"]):
            raise RuntimeError(f"W0/E3 selected epoch 不一致 seed={seed}")
        for key, filename in (
            ("checkpoint", "checkpoint_best_local_rr.pt"),
            ("config", "config.yaml"),
            ("manifest", "run_manifest.json"),
            ("validation_summary", "metrics_summary.csv"),
        ):
            path = source_root / str(entry["run_dir"]) / filename
            verify_file(path, entry[key])
            source_files[str(path.resolve())] = dict(entry[key])
        metric_path = Path(metric_sources[seed]["path"])
        verify_file(metric_path, metric_sources[seed]["identity"])
        source_files[str(metric_path)] = dict(metric_sources[seed]["identity"])

    cache = source_lock["cache_lock"]
    cache_entries = (
        cache["manifest"],
        cache["val_w"],
        {
            "path": cache["frequency_file"]["path"],
            "size_bytes": cache["frequency_file"]["size_bytes"],
            "sha256": cache["frequency_file"]["file_sha256"],
        },
    )
    for entry in cache_entries:
        path = source_root / str(entry["path"])
        verify_file(path, entry)
        source_files[str(path.resolve())] = {
            "size_bytes": int(entry["size_bytes"]),
            "sha256": str(entry["sha256"]),
        }
    row_path = source_root / str(cache["root"]) / "val_row_ids.npy"
    row_identity = identity(row_path)
    if row_identity["sha256"] != cache["row_identity"]["val_row_file_sha256"]:
        raise RuntimeError("validation row-id 文件 identity 漂移")
    source_files[str(row_path.resolve())] = row_identity

    cache_manifest_path = source_root / str(cache["manifest"]["path"])
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    dataset_index = Path(cache_manifest["dataset_index"]).resolve()
    dataset_identity = identity(dataset_index)
    if dataset_identity["sha256"] != cache_manifest["dataset_index_sha256"]:
        raise RuntimeError("dataset index identity 漂移")
    source_files[str(dataset_index)] = dataset_identity

    historical_path = source_root / HISTORICAL_FILM
    if sha256_file(historical_path) != HISTORICAL_FILM_SHA256:
        raise RuntimeError("历史 corrected FiLM identity 漂移")
    source_files[str(historical_path.resolve())] = identity(historical_path)

    code_paths = sorted((code_root / "resp_train").rglob("*.py"))
    code_paths.extend(
        code_root / path for path in (SCRIPT_PATH, TEST_PATH, PROTOCOL_PATH, README_PATH)
    )
    lock = {
        "protocol": PROTOCOL,
        "status": "implementation_locked_real_smoke_and_analysis_pending",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": git_state(),
        "code_root": str(code_root.resolve()),
        "source_root": str(source_root.resolve()),
        "output_root": str((source_root / "runs/w0_cwt_film_behavior_v1").resolve()),
        "split": "val",
        "seeds": list(SEEDS),
        "windows_per_seed": WINDOW_COUNT,
        "samp_ids": 7,
        "batch_size": 128,
        "amp_dtype": "bfloat16",
        "tau_values": list(TAUS),
        "main_tau": MAIN_TAU,
        "epsilon": 1e-12,
        "low_energy_rms": 1e-6,
        "time_bin_frames": TIME_BIN_FRAMES,
        "time_bin_count": TIME_BIN_COUNT,
        "quality_group": QUALITY_GROUP_COLUMN,
        "quality_continuous": list(QUALITY_COLUMNS),
        "w0_entries": entries,
        "cache_lock": cache,
        "dataset_index": {"path": str(dataset_index), **dataset_identity},
        "historical_corrected_film": {"path": str(historical_path.resolve()), **identity(historical_path)},
        "source_files": source_files,
        "code_files": {
            str(path.relative_to(code_root)): identity(path)
            for path in sorted(set(code_paths))
        },
        "source_verification": "SHA-256 and size; no checkpoint deserialization during prepare-lock",
        "research_test_used": False,
    }
    if (code_root / PREVIOUS_IMPLEMENTATION_LOCK).is_file():
        lock["supersedes"] = {
            "path": str(PREVIOUS_IMPLEMENTATION_LOCK),
            **identity(code_root / PREVIOUS_IMPLEMENTATION_LOCK),
        }
        lock["revision"] = (
            "同步实现完成状态、smoke receipt 参数和原始 batch 上下文案例复现；"
            "前版未用于任何真实 smoke 或 analysis"
        )
    write_json(destination, lock)
    return destination


def load_lock(*, code_root: Path = CODE_ROOT) -> tuple[dict[str, Any], str]:
    path = code_root / IMPLEMENTATION_LOCK
    if not path.is_file():
        raise FileNotFoundError(f"缺少 FiLM implementation lock: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or tuple(lock.get("seeds", ())) != SEEDS
        or int(lock.get("windows_per_seed", -1)) != WINDOW_COUNT
        or lock.get("split") != "val"
        or tuple(float(value) for value in lock.get("tau_values", ())) != TAUS
        or float(lock.get("main_tau", -1)) != MAIN_TAU
        or int(lock.get("time_bin_frames", -1)) != TIME_BIN_FRAMES
        or Path(lock.get("code_root", "")).resolve() != code_root.resolve()
        or Path(lock.get("source_root", "")).resolve() != SOURCE_ROOT
    ):
        raise ValueError("FiLM implementation lock 合同漂移")
    for relative, expected in lock["code_files"].items():
        verify_file(code_root / relative, expected)
    return lock, sha256_file(path)


@contextmanager
def attempt(
    phase: str,
    lock_hash: str,
    *,
    seed: int | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> Iterator[Path]:
    if phase not in {"smoke", "analysis", "summary", "cases", "final"}:
        raise ValueError(f"未知 FiLM phase={phase}")
    parent = output_root / phase
    if seed is not None:
        parent = parent / f"seed_{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    path.mkdir(exist_ok=False)
    context = {
        "protocol": PROTOCOL,
        "phase": phase,
        "seed": seed,
        "implementation_lock_sha256": lock_hash,
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(path / "lifecycle_started.json", {**context, "status": "running"})
    print(f"FiLM attempt: {path}", flush=True)
    try:
        yield path
        write_json(
            path / "lifecycle_completed.json",
            {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
        )
        manifest = {
            **context,
            "status": "completed",
            "files": {
                str(file.relative_to(path)): identity(file)
                for file in sorted(path.rglob("*"))
                if file.is_file()
            },
        }
        write_json(path / "artifact_manifest.json", manifest)
        write_json(
            path / "freeze_receipt.json",
            {"protocol": PROTOCOL, "phase": phase, "manifest": identity(path / "artifact_manifest.json")},
        )
    except BaseException as exc:
        write_json(
            path / "lifecycle_failed.json",
            {
                **context,
                "status": "failed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def verify_attempt(path: Path, *, phase: str, lock_hash: str) -> dict[str, Any]:
    path = path.resolve()
    if (path / "lifecycle_failed.json").exists():
        raise ValueError(f"FiLM attempt 已失败: {path}")
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    verify_file(path / "artifact_manifest.json", freeze["manifest"])
    manifest = json.loads((path / "artifact_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
    ):
        raise ValueError("FiLM attempt 协议、阶段或实现身份不一致")
    for relative, expected in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path):
            raise ValueError("FiLM manifest 路径越界")
        verify_file(file, expected)
    return manifest


def _entry(lock: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    matches = [entry for entry in lock["w0_entries"] if int(entry["seed"]) == int(seed)]
    if len(matches) != 1 or int(seed) not in SEEDS:
        raise ValueError(f"未知或重复 W0 seed={seed}")
    return matches[0]


def _load_model(lock: Mapping[str, Any], seed: int, device: str):
    entry = _entry(lock, seed)
    config_path = SOURCE_ROOT / str(entry["run_dir"]) / "config.yaml"
    cfg = load_crd_config(
        config_path,
        overrides=[f"training.device={device}", "training.show_progress=false"],
    )
    if (
        str(cfg.protocol.name) != "crd-tf-v1-research-informed-20260812"
        or str(cfg.protocol.run_role) != "formal"
        or str(cfg.model.variant) != "crd_tf102_w"
        or list(cfg.model.tf_representations) != ["w"]
        or int(cfg.training.seed) != int(seed)
        or int(cfg.training.batch_size) != 128
        or bool(cfg.training.drop_last)
        or not bool(cfg.training.use_amp)
        or str(cfg.training.amp_dtype) != "bfloat16"
        or cfg.data.max_val_windows is not None
    ):
        raise RuntimeError(f"W0 config 合同漂移 seed={seed}")
    checkpoint_path = SOURCE_ROOT / str(entry["run_dir"]) / "checkpoint_best_local_rr.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    for name, value in checkpoint["model_state_dict"].items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"W0 checkpoint 非有限: {name}")
    model = build_crd_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return cfg, model


def _build_validation_data(cfg: Any, lock: Mapping[str, Any]):
    data = build_window_data(
        cfg,
        split="val",
        max_windows=None,
        sample_strategy=str(cfg.data.val_sample_strategy),
        sample_seed=int(cfg.data.val_sample_seed),
        shuffle=False,
    )
    rows = data.rows
    if (
        len(rows) != WINDOW_COUNT
        or len(data.dataset) != WINDOW_COUNT
        or rows["dataset_row_id"].duplicated().any()
        or rows["samp_id"].nunique() != 7
        or set(rows["split"].astype(str)) != {"val"}
    ):
        raise ValueError("W0 validation rows 数量、唯一性、samp 或 split 漂移")
    sorted_ids = np.sort(rows["dataset_row_id"].to_numpy(dtype=np.int64))
    observed = hashlib.sha256(sorted_ids.tobytes(order="C")).hexdigest()
    expected = lock["cache_lock"]["row_identity"]["val_row_content_sha256"]
    if observed != expected:
        raise RuntimeError("W0 validation row content identity 漂移")
    return data


def _checked_batches(loader: Any, rows: pd.DataFrame) -> Iterator[tuple[int, Mapping[str, Any]]]:
    offset = 0
    for batch in loader:
        count = int(len(batch["x"]))
        expected = rows.iloc[offset : offset + count]
        if count == 0 or len(expected) != count:
            raise ValueError("FiLM loader 出现空或多余 batch")
        for key in ("dataset_row_id", "samp_id", "split"):
            actual = batch["meta"][key]
            if torch.is_tensor(actual):
                actual = actual.cpu().numpy()
            if not np.array_equal(np.asarray(actual), expected[key].to_numpy()):
                raise ValueError(f"FiLM batch identity/order 漂移: {key}")
        if set(batch.get("tf", {})) != {"w"} or tuple(batch["tf"]["w"].shape) != (
            count,
            97,
            360,
        ):
            raise ValueError("FiLM W batch shape/keys 漂移")
        for tensor in (batch["x"], batch["target"], batch["tf"]["w"]):
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError("FiLM batch input/target/W 含 NaN/Inf")
        yield offset, batch
        offset += count
    if offset != len(rows):
        raise ValueError("FiLM loader 未完整覆盖 validation rows")


def _batch_to_device(batch: Mapping[str, Any], device: torch.device):
    x = batch["x"].to(device, non_blocking=True)
    tf = batch_tf_to_device(batch, device, non_blocking=True)
    if tf is None:
        raise ValueError("FiLM batch 缺少 W feature")
    return x, tf


def _window_frame(
    statistics: Mapping[str, np.ndarray],
    identities: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(statistics)
    if len(frame) != len(identities):
        raise ValueError("窗口统计与 batch identity 行数不一致")
    frame.insert(0, "split", identities["split"].to_numpy())
    frame.insert(0, "samp_id", identities["samp_id"].to_numpy(dtype=np.int64))
    frame.insert(0, "dataset_row_id", identities["dataset_row_id"].to_numpy(dtype=np.int64))
    frame.insert(0, "seed", int(seed))
    return frame


def _axis_frame(
    statistics: Mapping[str, np.ndarray],
    identities: pd.DataFrame,
    *,
    seed: int,
    axis_name: str,
    axis_size: int,
) -> pd.DataFrame:
    arrays = {name: np.asarray(value) for name, value in statistics.items()}
    if any(value.shape != (len(identities), axis_size) for value in arrays.values()):
        raise ValueError(f"{axis_name} 统计 shape 不一致")
    frame = pd.DataFrame({name: value.reshape(-1) for name, value in arrays.items()})
    frame.insert(0, axis_name, np.tile(np.arange(axis_size, dtype=np.int64), len(identities)))
    frame.insert(0, "samp_id", np.repeat(identities["samp_id"].to_numpy(dtype=np.int64), axis_size))
    frame.insert(
        0,
        "dataset_row_id",
        np.repeat(identities["dataset_row_id"].to_numpy(dtype=np.int64), axis_size),
    )
    frame.insert(0, "seed", int(seed))
    return frame


def _aggregate_axis(frame: pd.DataFrame, axis_name: str) -> pd.DataFrame:
    identity = ["seed", "samp_id", axis_name]
    value_columns = [column for column in frame if column not in {"dataset_row_id", *identity}]
    grouped = frame.groupby(identity, sort=True)[value_columns].mean().reset_index()
    counts = (
        frame.groupby(identity, sort=True)["dataset_row_id"].nunique().rename("windows").reset_index()
    )
    return grouped.merge(counts, on=identity, how="left", validate="one_to_one")


def _add_histograms(target: dict[str, np.ndarray], batch: Mapping[str, np.ndarray]) -> None:
    for key, value in batch.items():
        current = np.asarray(value, dtype=np.int64)
        if key not in target:
            target[key] = current.copy()
        else:
            if target[key].shape != current.shape:
                raise ValueError(f"histogram shape 漂移: {key}")
            target[key] += current


def _histogram_edges() -> dict[str, np.ndarray]:
    return {
        "g_edges": np.linspace(
            EFFECTIVE_HISTOGRAM_RANGE[0],
            EFFECTIVE_HISTOGRAM_RANGE[1],
            EFFECTIVE_HISTOGRAM_BINS + 1,
        ),
        "b_edges": np.linspace(
            EFFECTIVE_HISTOGRAM_RANGE[0],
            EFFECTIVE_HISTOGRAM_RANGE[1],
            EFFECTIVE_HISTOGRAM_BINS + 1,
        ),
        "gamma_tanh_edges": np.linspace(
            BOUNDED_HISTOGRAM_RANGE[0],
            BOUNDED_HISTOGRAM_RANGE[1],
            BOUNDED_HISTOGRAM_BINS + 1,
        ),
        "beta_tanh_edges": np.linspace(
            BOUNDED_HISTOGRAM_RANGE[0],
            BOUNDED_HISTOGRAM_RANGE[1],
            BOUNDED_HISTOGRAM_BINS + 1,
        ),
        "gamma_raw_edges": np.linspace(
            RAW_HISTOGRAM_RANGE[0], RAW_HISTOGRAM_RANGE[1], RAW_HISTOGRAM_BINS + 1
        ),
        "beta_raw_edges": np.linspace(
            RAW_HISTOGRAM_RANGE[0], RAW_HISTOGRAM_RANGE[1], RAW_HISTOGRAM_BINS + 1
        ),
    }


def _verify_sources(lock: Mapping[str, Any]) -> None:
    for path, expected in lock["source_files"].items():
        verify_file(Path(path), expected)


def _write_provenance(output: Path, lock: Mapping[str, Any]) -> None:
    write_json(output / "implementation_lock.json", lock)
    pd.DataFrame(
        [
            {"path": path, "size_bytes": value["size_bytes"], "sha256": value["sha256"]}
            for path, value in sorted(lock["source_files"].items())
        ]
    ).to_csv(output / "source_audit.csv", index=False)
    with (output / "commands.txt").open("x", encoding="utf-8") as handle:
        handle.write(" ".join(sys.argv) + "\n")


def run_smoke(*, seed: int = SEEDS[0], device: str = "cuda:0") -> Path:
    lock, lock_hash = load_lock()
    resolved = require_gpu(device)
    with attempt("smoke", lock_hash, seed=int(seed)) as output:
        _write_provenance(output, lock)
        write_json(output / "environment.json", {**environment(device), "git": git_state(require_clean=True)})
        _verify_sources(lock)
        cfg, model = _load_model(lock, int(seed), device)
        data = _build_validation_data(cfg, lock)
        offset, batch = next(_checked_batches(data.loader, data.rows))
        if offset != 0:
            raise RuntimeError("smoke 必须从 validation 首 batch 开始")
        # 只取 batch 中第一项，避免 smoke 扩大真实数据访问。
        small = dict(batch)
        small["x"] = batch["x"][:1]
        small["target"] = batch["target"][:1]
        small["tf"] = {"w": batch["tf"]["w"][:1]}
        x, tf = _batch_to_device(small, resolved)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            native = model(x, tf=tf)
            observed, captured = forward_with_capture(model, x, tf=tf)
            # forward 已完整结束；保留同一 autocast 语义复现模型中的 tanh 有效量。
            stats = compute_batch_statistics(captured)
        maximums: dict[str, float] = {}
        exact: dict[str, bool] = {}
        for key in native:
            if key not in observed or native[key].shape != observed[key].shape:
                raise ValueError(f"smoke output key/shape 漂移: {key}")
            if not bool(torch.isfinite(observed[key]).all()):
                raise FloatingPointError(f"smoke output 非有限: {key}")
            maximums[key] = float((observed[key].float() - native[key].float()).abs().max())
            exact[key] = bool(torch.equal(observed[key], native[key]))
        passed = all(value <= 1e-6 for value in maximums.values())
        if not passed:
            raise RuntimeError(f"hook 改变原生输出: {maximums}")
        preview = {name: np.asarray(value).reshape(-1)[0].item() for name, value in stats.window.items()}
        write_json(
            output / "smoke_receipt.json",
            {
                "protocol": PROTOCOL,
                "passed": True,
                "seed": int(seed),
                "batch_size": 1,
                "dataset_row_id": int(data.rows.iloc[0]["dataset_row_id"]),
                "native_hooked_max_abs_deltas": maximums,
                "native_hooked_exact": exact,
                "captured_shapes": {
                    "z": list(captured.z.shape),
                    "gamma_raw": list(captured.gamma_raw.shape),
                    "beta_raw": list(captured.beta_raw.shape),
                    "z_prime": list(captured.z_prime.shape),
                },
                "window_statistics": preview,
                "research_test_used": False,
            },
        )
        write_json(
            output / "access_receipt.json",
            {
                "split": "val",
                "windows": 1,
                "checkpoint_seed": int(seed),
                "checkpoint_modified": False,
                "cache_modified": False,
                "research_test_used": False,
            },
        )
        del model
        torch.cuda.empty_cache()
    return output


def _validate_smoke(path: Path, lock_hash: str) -> dict[str, Any]:
    verify_attempt(path, phase="smoke", lock_hash=lock_hash)
    receipt = json.loads((path / "smoke_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("passed") is not True
        or int(receipt.get("seed", -1)) != SEEDS[0]
        or int(receipt.get("batch_size", -1)) != 1
        or max(receipt["native_hooked_max_abs_deltas"].values()) > 1e-6
    ):
        raise ValueError("FiLM smoke receipt 未通过固定合同")
    return {"path": str(path.resolve()), "receipt": receipt}


def run_analysis(
    *,
    seed: int,
    device: str,
    smoke_receipt: Path,
) -> Path:
    lock, lock_hash = load_lock()
    if int(seed) not in SEEDS:
        raise ValueError(f"seed 必须属于 {SEEDS}")
    resolved = require_gpu(device)
    smoke = _validate_smoke(smoke_receipt, lock_hash)
    with attempt("analysis", lock_hash, seed=int(seed)) as output:
        _write_provenance(output, lock)
        write_json(output / "environment.json", {**environment(device), "git": git_state(require_clean=True)})
        write_json(output / "smoke_source.json", smoke)
        write_json(
            output / "access_started.json",
            {
                "status": "planned_access",
                "split": "val",
                "windows": WINDOW_COUNT,
                "checkpoint_seed": int(seed),
                "waveform_scope": "frozen admitted validation rows",
                "cache_scope": "frozen validation W cache",
                "research_test_used": False,
            },
        )
        _verify_sources(lock)
        cfg, model = _load_model(lock, int(seed), device)
        OmegaConf.save(cfg, output / "resolved_config.yaml")
        data = _build_validation_data(cfg, lock)
        data.rows.to_csv(output / "validation_rows.csv", index=False)

        window_frames: list[pd.DataFrame] = []
        channel_frames: list[pd.DataFrame] = []
        time_frames: list[pd.DataFrame] = []
        histograms: dict[str, np.ndarray] = {}
        output_keys: set[str] | None = None
        for offset, batch in _checked_batches(data.loader, data.rows):
            count = int(len(batch["x"]))
            identities = data.rows.iloc[offset : offset + count]
            x, tf = _batch_to_device(batch, resolved)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction, captured = forward_with_capture(model, x, tf=tf)
                # 不在 decoder 前做任何归约；仅在原生输出完成后、同一 autocast 语义内统计。
                statistics = compute_batch_statistics(captured)
            if output_keys is None:
                output_keys = set(prediction)
            elif set(prediction) != output_keys:
                raise ValueError("W0 output keys 跨 batch 漂移")
            for key, value in prediction.items():
                if int(value.shape[0]) != count or not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"W0 prediction shape/finite 失败: {key}")
            window_frames.append(_window_frame(statistics.window, identities, seed=int(seed)))
            channel_frames.append(
                _axis_frame(
                    statistics.channel,
                    identities,
                    seed=int(seed),
                    axis_name="channel",
                    axis_size=CHANNELS,
                )
            )
            time_frames.append(
                _axis_frame(
                    statistics.time_bin,
                    identities,
                    seed=int(seed),
                    axis_name="time_bin",
                    axis_size=TIME_BIN_COUNT,
                )
            )
            _add_histograms(histograms, statistics.histograms)
            del prediction, captured, statistics, x, tf

        windows = pd.concat(window_frames, ignore_index=True)
        validate_window_identity(windows)
        entry = _entry(lock, int(seed))
        metrics = pd.read_csv(SOURCE_ROOT / str(entry["run_dir"]) / "metrics.csv")
        windows = join_frozen_sources(windows, metrics, data.rows)
        historical = pd.read_csv(Path(lock["historical_corrected_film"]["path"]))
        historical_receipt = validate_historical_film(windows, historical, seed=int(seed))
        channels = _aggregate_axis(pd.concat(channel_frames, ignore_index=True), "channel")
        times = _aggregate_axis(pd.concat(time_frames, ignore_index=True), "time_bin")
        times["time_start_s"] = times["time_bin"] * (TIME_BIN_FRAMES / 10.0)
        times["time_end_s"] = (times["time_bin"] + 1) * (TIME_BIN_FRAMES / 10.0)

        windows.to_csv(output / "window_statistics.csv", index=False)
        channels.to_csv(output / "channel_summary.csv", index=False)
        times.to_csv(output / "time_bin_summary.csv", index=False)
        np.savez_compressed(
            output / "distribution_histograms.npz",
            **histograms,
            **_histogram_edges(),
        )
        write_json(
            output / "analysis_receipt.json",
            {
                "protocol": PROTOCOL,
                "seed": int(seed),
                "selected_epoch": int(entry["selected_epoch"]),
                "checkpoint_sha256": entry["checkpoint"]["sha256"],
                "split": "val",
                "windows": int(len(windows)),
                "samp_ids": int(windows["samp_id"].nunique()),
                "output_keys": sorted(output_keys or ()),
                "denom_small_windows": int(windows["denom_small"].astype(bool).sum()),
                "closure_max_abs": float(windows["closure_max_abs"].max()),
                "historical_film_anchor": historical_receipt,
                "quality_level_counts": {
                    str(key): int(value)
                    for key, value in windows[QUALITY_GROUP_COLUMN].value_counts().sort_index().items()
                },
                "research_test_used": False,
                "training_used": False,
                "checkpoint_modified": False,
                "cache_modified": False,
            },
        )
        write_json(
            output / "access_receipt.json",
            {
                "split": "val",
                "windows": WINDOW_COUNT,
                "samp_ids": 7,
                "checkpoint_seed": int(seed),
                "source_files_verified": len(lock["source_files"]),
                "waveform_access": "admitted validation rows only",
                "cache_access": "frozen val W cache, read-only",
                "research_test_used": False,
            },
        )
        del model
        torch.cuda.empty_cache()
    return output


def _load_analysis_attempt(path: Path, lock_hash: str) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verify_attempt(path, phase="analysis", lock_hash=lock_hash)
    receipt = json.loads((path / "analysis_receipt.json").read_text(encoding="utf-8"))
    seed = int(receipt["seed"])
    windows = pd.read_csv(path / "window_statistics.csv")
    channels = pd.read_csv(path / "channel_summary.csv")
    times = pd.read_csv(path / "time_bin_summary.csv")
    validate_window_identity(windows)
    if int(windows["seed"].iloc[0]) != seed or set(windows["seed"]) != {seed}:
        raise ValueError("analysis receipt/window seed 不一致")
    return seed, windows, channels, times


def _plot_summary(
    output: Path,
    windows: pd.DataFrame,
    saturation: pd.DataFrame,
    quality: pd.DataFrame,
    associations: pd.DataFrame,
    histograms: Mapping[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(exist_ok=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, name, title in zip(
        axes.ravel(),
        ("gamma_raw", "beta_raw", "g", "b"),
        ("gamma_raw", "beta_raw", "g", "b"),
        strict=True,
    ):
        edges = histograms[f"{name}_edges"]
        counts = histograms[f"{name}_counts"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.plot(centers, counts / max(1, counts.sum()))
        below = int(np.asarray(histograms.get(f"{name}_below", [0])).sum())
        above = int(np.asarray(histograms.get(f"{name}_above", [0])).sum())
        axis.set_title(f"{title} (below={below}, above={above})")
        axis.set_xlabel("value")
        axis.set_ylabel("fraction")
    fig.tight_layout()
    fig.savefig(figures / "parameter_distributions.png", dpi=180)
    plt.close(fig)

    subset = saturation.loc[saturation["edge"].isin(["pos", "neg"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, parameter in zip(axes, ("g", "b"), strict=True):
        part = subset.loc[subset["parameter"].eq(parameter)]
        for edge, marker in (("pos", "o"), ("neg", "s")):
            values = part.loc[part["edge"].eq(edge)].groupby("tau")["mean_fraction"].mean()
            axis.plot(values.index, values.values, marker=marker, label=edge)
        axis.set_title(parameter)
        axis.set_xlabel("tau")
        axis.set_ylabel("element fraction")
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "saturation_by_sign_threshold.png", dpi=180)
    plt.close(fig)

    pooled = associations.loc[
        associations["view"].eq("pooled_windows")
        & associations["target_kind"].eq("error")
        & associations["status"].eq("ok")
    ]
    matrix = pooled.pivot_table(
        index="modulation_variable",
        columns="target_variable",
        values="spearman",
        aggfunc="mean",
    ).reindex(index=PRIMARY_ASSOCIATION_COLUMNS, columns=ERROR_COLUMNS)
    fig, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    fig.colorbar(image, ax=axis, label="Spearman")
    fig.tight_layout()
    fig.savefig(figures / "modulation_strength_error_association.png", dpi=180)
    plt.close(fig)

    quality_plot = quality.loc[
        quality["variable"].isin(("r_scale", "r_shift", "r_total"))
    ]
    levels = sorted(quality_plot[QUALITY_GROUP_COLUMN].unique())
    fig, axis = plt.subplots(figsize=(8, 4))
    width = 0.22
    positions = np.arange(len(levels))
    for index, variable in enumerate(("r_scale", "r_shift", "r_total")):
        values = (
            quality_plot.loc[quality_plot["variable"].eq(variable)]
            .groupby(QUALITY_GROUP_COLUMN)["mean"]
            .mean()
            .reindex(levels)
        )
        axis.bar(positions + (index - 1) * width, values, width=width, label=variable)
    axis.set_xticks(positions, levels)
    axis.set_ylabel("mean relative strength")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "quality_group_comparison.png", dpi=180)
    plt.close(fig)


def run_summary(*, runs: Sequence[Path]) -> Path:
    lock, lock_hash = load_lock()
    if len(runs) != len(SEEDS):
        raise ValueError("summary 必须提供三个 analysis attempt")
    with attempt("summary", lock_hash) as output:
        _write_provenance(output, lock)
        write_json(output / "environment.json", {**environment(None), "git": git_state(require_clean=True)})
        window_frames, channel_frames, time_frames = [], [], []
        histograms: dict[str, np.ndarray] = {}
        source_records = []
        observed_seeds = set()
        for path in runs:
            path = path.resolve()
            seed, windows, channels, times = _load_analysis_attempt(path, lock_hash)
            if seed in observed_seeds:
                raise ValueError(f"summary 重复 seed={seed}")
            observed_seeds.add(seed)
            window_frames.append(windows)
            channel_frames.append(channels)
            time_frames.append(times)
            with np.load(path / "distribution_histograms.npz", allow_pickle=False) as bundle:
                for key in bundle.files:
                    value = bundle[key]
                    if key.endswith("_edges"):
                        if key in histograms and not np.array_equal(histograms[key], value):
                            raise ValueError(f"histogram edges 跨 seed 漂移: {key}")
                        histograms[key] = value.copy()
                    else:
                        histograms[key] = histograms.get(key, np.zeros_like(value)) + value
            source_records.append(
                {
                    "seed": seed,
                    "path": str(path),
                    "manifest": identity(path / "artifact_manifest.json"),
                }
            )
        if observed_seeds != set(SEEDS):
            raise ValueError("summary seed 矩阵不完整")
        windows = pd.concat(window_frames, ignore_index=True)
        if len(windows) != len(SEEDS) * WINDOW_COUNT or windows[
            ["seed", "dataset_row_id"]
        ].duplicated().any():
            raise ValueError("summary 窗口矩阵不完整或重复")
        row_sets = {
            int(seed): tuple(group["dataset_row_id"].sort_values().to_numpy(dtype=np.int64))
            for seed, group in windows.groupby("seed", sort=True)
        }
        if len(row_sets) != len(SEEDS) or any(
            not np.array_equal(row_sets[SEEDS[0]], row_sets[seed]) for seed in SEEDS[1:]
        ):
            raise ValueError("summary 三 seed validation row identity 不一致")
        static_columns = (
            "samp_id",
            QUALITY_GROUP_COLUMN,
            *QUALITY_COLUMNS,
            "target_envelope_modulation",
            "envelope_target_stratum",
            "coupling_state_id",
            "window_start_s",
            "window_end_s",
        )
        static_uniques = windows.groupby("dataset_row_id", sort=True)[list(static_columns)].nunique(
            dropna=False
        )
        if (static_uniques > 1).any().any():
            raise ValueError("summary 三 seed 的 target/quality/static metadata 不一致")
        channels = pd.concat(channel_frames, ignore_index=True)
        times = pd.concat(time_frames, ignore_index=True)
        if (
            len(channels) != len(SEEDS) * 7 * CHANNELS
            or channels[["seed", "samp_id", "channel"]].duplicated().any()
        ):
            raise ValueError("summary channel 矩阵不完整或重复")
        if (
            len(times) != len(SEEDS) * 7 * TIME_BIN_COUNT
            or times[["seed", "samp_id", "time_bin"]].duplicated().any()
        ):
            raise ValueError("summary time-bin 矩阵不完整或重复")
        saturation = saturation_summary(windows)
        subjects = subject_summary(windows)
        quality = quality_group_summary(windows)
        associations = association_table(windows)
        association_seed_summary = summarize_seed_variation(
            associations,
            value_column="spearman",
            group_columns=[
                "view",
                "samp_id",
                "target_stratum",
                "modulation_variable",
                "target_variable",
                "target_kind",
            ],
        )
        selection = select_typical_windows(windows)

        windows.to_csv(output / "window_statistics.csv", index=False)
        subjects.to_csv(output / "subject_summary.csv", index=False)
        channels.to_csv(output / "channel_summary.csv", index=False)
        times.to_csv(output / "time_bin_summary.csv", index=False)
        saturation.to_csv(output / "saturation_threshold_summary.csv", index=False)
        quality.to_csv(output / "quality_group_summary.csv", index=False)
        associations.to_csv(output / "modulation_error_associations.csv", index=False)
        association_seed_summary.to_csv(output / "association_seed_summary.csv", index=False)
        selection.to_csv(output / "case_selection.csv", index=False)
        np.savez_compressed(output / "distribution_histograms.npz", **histograms)
        _plot_summary(output, windows, saturation, quality, associations, histograms)
        write_json(output / "analysis_sources.json", {"runs": source_records})
        write_json(
            output / "summary_receipt.json",
            {
                "protocol": PROTOCOL,
                "seeds": list(SEEDS),
                "windows": int(len(windows)),
                "samp_ids": int(windows["samp_id"].nunique()),
                "denom_small_windows": int(windows["denom_small"].astype(bool).sum()),
                "quality_level_counts": {
                    str(key): int(value)
                    for key, value in windows.drop_duplicates("dataset_row_id")[
                        QUALITY_GROUP_COLUMN
                    ].value_counts().sort_index().items()
                },
                "case_rows": int(len(selection)),
                "case_render_rows": int(selection["selected_for_render"].sum()),
                "research_test_used": False,
            },
        )
    return output


def _selected_original_batches(
    data: Any,
    row_ids: Sequence[int],
    *,
    batch_size: int = 128,
) -> list[tuple[Mapping[str, Any], list[int], list[int]]]:
    """按首遍 batch 边界重建案例，避免改变 batch shape 引入数值漂移。"""

    ordered_ids = data.rows["dataset_row_id"].to_numpy(dtype=np.int64)
    positions = {int(row_id): int(index) for index, row_id in enumerate(ordered_ids)}
    missing = [int(row_id) for row_id in row_ids if int(row_id) not in positions]
    if missing:
        raise KeyError(f"selected validation rows 缺失: {missing}")
    grouped: dict[int, list[tuple[int, int]]] = {}
    for row_id in row_ids:
        position = positions[int(row_id)]
        batch_start = (position // batch_size) * batch_size
        grouped.setdefault(batch_start, []).append((position, int(row_id)))
    output = []
    for batch_start, selected in sorted(grouped.items()):
        batch_stop = min(batch_start + batch_size, len(data.dataset))
        samples = [data.dataset[index] for index in range(batch_start, batch_stop)]
        batch = data.loader.collate_fn(samples)
        local_indices = [position - batch_start for position, _ in selected]
        selected_ids = [row_id for _, row_id in selected]
        actual_ids = np.asarray(batch["meta"]["dataset_row_id"], dtype=np.int64)
        if not np.array_equal(actual_ids[local_indices], np.asarray(selected_ids, dtype=np.int64)):
            raise ValueError("selected original-batch identity/order 漂移")
        output.append((batch, local_indices, selected_ids))
    return output


def _save_selected_arrays(
    output: Path,
    *,
    seed: int,
    row_ids: Sequence[int],
    batch: Mapping[str, Any],
    prediction: Mapping[str, torch.Tensor],
    captured: Any,
    sample_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    g = (0.5 * torch.tanh(captured.gamma_raw)).float()
    b = (0.5 * torch.tanh(captured.beta_raw)).float()
    z = captured.z.float()
    z_prime = captured.z_prime.float()
    scale = g * z
    total = z_prime - z
    records = []
    tensor_root = output / "selected_window_tensors"
    tensor_root.mkdir(exist_ok=True)
    indices = list(range(len(row_ids))) if sample_indices is None else [int(value) for value in sample_indices]
    if len(indices) != len(row_ids):
        raise ValueError("selected tensor row_ids/sample_indices 数量不一致")
    for index, row_id in zip(indices, row_ids, strict=True):
        path = tensor_root / f"row_{int(row_id)}__seed_{int(seed)}.npz"
        arrays = {
            "input_bcg": batch["x"][index].detach().cpu().float().numpy(),
            "target_respiration": batch["target"][index].detach().cpu().float().numpy(),
            "prediction_waveform": prediction["waveform"][index].detach().cpu().float().numpy(),
            "prediction_waveform_10hz": prediction["waveform_10hz"][index]
            .detach()
            .cpu()
            .float()
            .numpy(),
            "z": z[index].detach().cpu().numpy(),
            "g": g[index].detach().cpu().numpy(),
            "b": b[index].detach().cpu().numpy(),
            "z_prime": z_prime[index].detach().cpu().numpy(),
            "scale_delta": scale[index].detach().cpu().numpy(),
            "total_delta": total[index].detach().cpu().numpy(),
            "g_channel_mean": g[index].mean(dim=0).detach().cpu().numpy(),
            "b_channel_mean": b[index].mean(dim=0).detach().cpu().numpy(),
            "scale_channel_rms": scale[index].square().mean(dim=0).sqrt().detach().cpu().numpy(),
            "shift_channel_rms": b[index].square().mean(dim=0).sqrt().detach().cpu().numpy(),
            "total_channel_rms": total[index].square().mean(dim=0).sqrt().detach().cpu().numpy(),
            "g_positive_edge_fraction": (torch.tanh(captured.gamma_raw[index].float()) > MAIN_TAU)
            .float()
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy(),
            "g_negative_edge_fraction": (torch.tanh(captured.gamma_raw[index].float()) < -MAIN_TAU)
            .float()
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy(),
            "b_positive_edge_fraction": (torch.tanh(captured.beta_raw[index].float()) > MAIN_TAU)
            .float()
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy(),
            "b_negative_edge_fraction": (torch.tanh(captured.beta_raw[index].float()) < -MAIN_TAU)
            .float()
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy(),
        }
        np.savez_compressed(path, **arrays)
        records.append(
            {
                "seed": int(seed),
                "dataset_row_id": int(row_id),
                "path": str(path.relative_to(output)),
                **identity(path),
                "arrays": {name: list(np.asarray(value).shape) for name, value in arrays.items()},
            }
        )
    return records


def _plot_cases(output: Path, records: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(exist_ok=False)
    for row_id, group in records.groupby("dataset_row_id", sort=True):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
        for _, record in group.sort_values("seed").iterrows():
            with np.load(output / record["path"], allow_pickle=False) as bundle:
                target = bundle["target_respiration"].reshape(-1)
                prediction = bundle["prediction_waveform"].reshape(-1)
                input_bcg = bundle["input_bcg"].reshape(-1)
                time = np.arange(len(target)) / 100.0
                axes[0].plot(time, prediction, alpha=0.75, label=f"seed {record['seed']}")
                if len(axes[0].lines) == 1:
                    axes[0].plot(time, target, color="black", linewidth=1, label="reference")
                    axes[1].plot(time, input_bcg, color="0.35", linewidth=0.7)
                latent_time = np.arange(LATENT_FRAMES) / 10.0
                axes[2].plot(latent_time, bundle["scale_channel_rms"], label=f"scale {record['seed']}")
                axes[2].plot(
                    latent_time,
                    bundle["shift_channel_rms"],
                    linestyle="--",
                    label=f"shift {record['seed']}",
                )
        axes[0].set_title(f"dataset_row_id={int(row_id)}")
        axes[0].legend(ncol=2, fontsize=8)
        axes[1].set_ylabel("input BCG")
        axes[2].set_ylabel("feature delta RMS")
        axes[2].set_xlabel("seconds")
        axes[2].legend(ncol=3, fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / f"row_{int(row_id)}.png", dpi=160)
        plt.close(fig)


def run_cases(*, summary: Path, device: str) -> Path:
    lock, lock_hash = load_lock()
    verify_attempt(summary, phase="summary", lock_hash=lock_hash)
    resolved = require_gpu(device)
    selection = pd.read_csv(summary / "case_selection.csv")
    selected = selection.loc[selection["selected_for_render"].astype(bool), "dataset_row_id"]
    row_ids = sorted(selected.dropna().astype(np.int64).unique().tolist())
    if not row_ids:
        raise ValueError("case selection 没有可渲染窗口")
    reference_windows = pd.read_csv(summary / "window_statistics.csv")
    with attempt("cases", lock_hash) as output:
        _write_provenance(output, lock)
        write_json(output / "environment.json", {**environment(device), "git": git_state(require_clean=True)})
        _verify_sources(lock)
        write_json(
            output / "summary_source.json",
            {"path": str(summary.resolve()), "manifest": identity(summary / "artifact_manifest.json")},
        )
        artifact_records = []
        consistency_records = []
        accessed_windows: dict[int, int] = {}
        for seed in SEEDS:
            cfg, model = _load_model(lock, seed, device)
            data = _build_validation_data(cfg, lock)
            first_pass = (
                reference_windows.loc[
                    reference_windows["seed"].eq(seed)
                    & reference_windows["dataset_row_id"].isin(row_ids)
                ]
                .sort_values("dataset_row_id")
                .reset_index(drop=True)
            )
            processed: list[int] = []
            selected_batches = _selected_original_batches(data, row_ids)
            accessed_windows[seed] = int(sum(len(batch["x"]) for batch, _, _ in selected_batches))
            for batch, local_indices, selected_ids in selected_batches:
                for tensor in (batch["x"], batch["target"], batch["tf"]["w"]):
                    if not bool(torch.isfinite(tensor).all()):
                        raise FloatingPointError("selected case input/target/W 非有限")
                x, tf = _batch_to_device(batch, resolved)
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    prediction, captured = forward_with_capture(model, x, tf=tf)
                    statistics = compute_batch_statistics(captured)
                for name in ("r_scale", "r_shift", "r_total"):
                    observed = np.asarray(statistics.window[name], dtype=np.float64)[local_indices]
                    expected = (
                        first_pass.set_index("dataset_row_id")
                        .loc[selected_ids, name]
                        .to_numpy(dtype=np.float64)
                    )
                    maximum = float(np.abs(observed - expected).max(initial=0.0))
                    consistency_records.append(
                        {
                            "seed": seed,
                            "statistic": name,
                            "batch_start_dataset_row_id": int(
                                np.asarray(batch["meta"]["dataset_row_id"], dtype=np.int64)[0]
                            ),
                            "max_abs_delta": maximum,
                        }
                    )
                    if maximum > 1e-6:
                        raise RuntimeError(
                            f"selected case 二次推理统计漂移 seed={seed} {name}={maximum}"
                        )
                artifact_records.extend(
                    _save_selected_arrays(
                        output,
                        seed=seed,
                        row_ids=selected_ids,
                        batch=batch,
                        prediction=prediction,
                        captured=captured,
                        sample_indices=local_indices,
                    )
                )
                processed.extend(selected_ids)
                del prediction, captured, statistics, x, tf
            if sorted(processed) != row_ids:
                raise RuntimeError(f"selected case 未完整覆盖 seed={seed}")
            del model
            torch.cuda.empty_cache()
        records = pd.DataFrame.from_records(artifact_records)
        records.to_json(output / "selected_tensor_index.json", orient="records", indent=2)
        pd.DataFrame.from_records(consistency_records).to_csv(
            output / "first_second_pass_consistency.csv", index=False
        )
        _plot_cases(output, records)
        write_json(
            output / "cases_receipt.json",
            {
                "protocol": PROTOCOL,
                "seeds": list(SEEDS),
                "dataset_row_ids": row_ids,
                "tensor_files": int(len(records)),
                "accessed_original_batch_windows_per_seed": accessed_windows,
                "first_second_pass_max_abs_delta": float(
                    max(record["max_abs_delta"] for record in consistency_records)
                ),
                "research_test_used": False,
            },
        )
        write_json(
            output / "access_receipt.json",
            {
                "split": "val",
                "selected_windows_per_seed": len(row_ids),
                "accessed_original_batch_windows_per_seed": accessed_windows,
                "checkpoint_seeds": list(SEEDS),
                "research_test_used": False,
            },
        )
    return output


def _conclusions_markdown(summary: Path) -> str:
    windows = pd.read_csv(summary / "window_statistics.csv")
    saturation = pd.read_csv(summary / "saturation_threshold_summary.csv")
    associations = pd.read_csv(summary / "modulation_error_associations.csv")
    quality = pd.read_csv(summary / "quality_group_summary.csv")
    strength_rows = []
    for name in ("r_scale", "r_shift", "r_total"):
        per_seed = windows.groupby("seed")[name].median()
        strength_rows.append(
            f"- {name}：各 seed 窗口中位数为 "
            + " / ".join(f"{value:.6g}" for value in per_seed.to_numpy())
            + f"，三 seed 中位数均值为 {per_seed.mean():.6g}。"
        )
    main = saturation.loc[
        saturation["tau"].eq(MAIN_TAU) & saturation["edge"].eq("abs")
    ]
    edge_rows = []
    for parameter in ("g", "b"):
        values = main.loc[main["parameter"].eq(parameter)].groupby("seed")["mean_fraction"].first()
        edge_rows.append(f"- {parameter} 的 tau=0.95 mean edge fraction：{values.mean():.6g}。")
    pooled = associations.loc[
        associations["view"].eq("pooled_windows")
        & associations["target_kind"].eq("error")
        & associations["status"].eq("ok")
    ].copy()
    pooled_mean = (
        pooled.groupby(["modulation_variable", "target_variable"], sort=True)["spearman"]
        .mean()
        .reset_index()
    )
    pooled_mean["absolute"] = pooled_mean["spearman"].abs()
    strongest = pooled_mean.sort_values("absolute", ascending=False).head(5)
    association_rows = [
        f"- {row.modulation_variable} 与 {row.target_variable}：三 seed pooled Spearman 均值 "
        f"{row.spearman:.4f}。"
        for row in strongest.itertuples()
    ]
    quality_levels = sorted(quality[QUALITY_GROUP_COLUMN].astype(str).unique())
    return "\n".join(
        [
            "# W0 CWT-FiLM 调制行为：描述性结论",
            "",
            "## 观察结果",
            "",
            *strength_rows,
            *edge_rows,
            f"- 独立 waveform confidence 分组实际为：{', '.join(quality_levels)}。",
            f"- 低特征能量窗口数：{int(windows['denom_small'].astype(bool).sum())}。",
            "",
            "误差关联中绝对值最大的五项描述如下（窗口重叠，不作独立样本推断）：",
            "",
            *association_rows,
            "",
            "## 机制假设",
            "",
            "- scale、shift 与 total 的相对大小可提示两条路径的使用方式；它们本身不识别因果贡献。",
            "- 调制—质量或调制—误差相关可能同时受 samp_id、目标 modulation 和窗口重叠影响。",
            "- 接近 tanh 边缘只说明当前参数分布位置，不能单独证明限幅造成误差。",
            "",
            "## 待验证问题",
            "",
            "- 如 shift 长期弱于 scale，另立 scale-only 重训练消融。",
            "- 如边缘比例与误差稳定同向，另立限幅系数 validation 敏感性实验。",
            "- 如 total 较小，先做冻结输出的局部调制敏感性，不直接判定条件分支无效。",
            "- 所有结构选择仍限于 train/validation；test 访问需独立协议。",
            "",
        ]
    )


def run_finalize(*, summary: Path, cases: Path) -> Path:
    lock, lock_hash = load_lock()
    verify_attempt(summary, phase="summary", lock_hash=lock_hash)
    verify_attempt(cases, phase="cases", lock_hash=lock_hash)
    with attempt("final", lock_hash) as output:
        _write_provenance(output, lock)
        write_json(output / "environment.json", {**environment(None), "git": git_state(require_clean=True)})
        write_json(
            output / "bundle_index.json",
            {
                "protocol": PROTOCOL,
                "summary": {
                    "path": str(summary.resolve()),
                    "manifest": identity(summary / "artifact_manifest.json"),
                },
                "cases": {
                    "path": str(cases.resolve()),
                    "manifest": identity(cases / "artifact_manifest.json"),
                },
                "research_test_used": False,
            },
        )
        with (output / "conclusions_zh.md").open("x", encoding="utf-8") as handle:
            handle.write(_conclusions_markdown(summary))
        write_json(
            output / "final_receipt.json",
            {
                "protocol": PROTOCOL,
                "status": "complete",
                "scope": "three frozen W0 checkpoints, validation-only descriptive analysis",
                "causal_claim": False,
                "research_test_used": False,
            },
        )
    return output
