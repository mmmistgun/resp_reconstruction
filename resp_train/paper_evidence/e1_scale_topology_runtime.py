"""E1 独立生命周期：锁准备、synthetic GPU 验收、完整 validation 与汇总。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.crd.model import build_crd_model
from resp_train.crd.tf_w_v2_audit import _load_audit_config, _w0_seed_entries
from resp_train.data.factory import build_window_data
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.paper_evidence.e1_scale_topology import (
    CONDITIONS, FILM_COLUMNS, FULL_ATOL, FULL_RTOL, PRIMARY, PROTOCOL, SEEDS, WINDOW_COUNT, ScaleAuditModel,
    array_hash, check_full_anchor, index_self_checks, make_index_lock, raw_pair_mae,
    summarize_pairs, validate_index_lock, validate_metrics, validate_rows,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path("docs/experiments")
SOURCE_LOCK = DOCS / "crd_tf_w_v2_candidate_lock_20260817.json"
SOURCE_LOCK_SHA256 = "6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6"
INDEX_LOCK = DOCS / "e1_w0_scale_indices_20260915.json"
PREVIOUS_IMPLEMENTATION_LOCK = DOCS / "e1_w0_scale_implementation_lock_20260915.json"
IMPLEMENTATION_LOCK = DOCS / "e1_w0_scale_implementation_lock_20260915_r2.json"
PROTOCOL_PATH = DOCS / "e1_w0_scale_topology_protocol_20260915.md"
SCRIPT = Path("scripts/eval_e1_w0_scale_topology.py")
TEST = Path("tests/test_e1_scale_topology.py")
OUTPUT = Path("runs/e1_w0_scale_topology_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_file(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file() or identity(path) != {key: expected[key] for key in ("size_bytes", "sha256")}:
        raise RuntimeError(f"E1 文件身份漂移: {path}")


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git_state(root: Path = ROOT, *, require_clean: bool = False) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    if require_clean and status:
        raise RuntimeError("E1 正式执行要求干净提交；请先整理并提交待执行代码及协议")
    return {"commit": commit, "status_porcelain": status}


def prepare_locks(root: Path = ROOT) -> tuple[Path, Path]:
    """只读核验 W0/validation 来源字节身份，创建独立索引与代码锁。"""
    if (root / IMPLEMENTATION_LOCK).exists():
        raise FileExistsError(f"E1 锁已存在: {root / IMPLEMENTATION_LOCK}")
    source_path = root / SOURCE_LOCK
    if sha256_file(source_path) != SOURCE_LOCK_SHA256:
        raise RuntimeError("E1 历史 candidate lock identity 漂移")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    entries = _w0_seed_entries(source)
    if [entry["selected_epoch"] for entry in entries] != [13, 15, 14]:
        raise ValueError("E1 selected epoch identity 漂移")
    sources: dict[str, dict[str, Any]] = {str(SOURCE_LOCK): identity(source_path)}
    for entry in entries:
        for key, filename in (("checkpoint", "checkpoint_best_local_rr.pt"), ("config", "config.yaml"),
                              ("manifest", "run_manifest.json"), ("validation_summary", "metrics_summary.csv")):
            relative = str(Path(entry["run_dir"]) / filename)
            verify_file(root / relative, entry[key])
            sources[relative] = dict(entry[key])
    cache = source["cache_lock"]
    for key in ("manifest", "val_w", "frequency_file"):
        entry = cache[key]
        expected = {"size_bytes": entry["size_bytes"], "sha256": entry.get("sha256", entry.get("file_sha256"))}
        verify_file(root / entry["path"], expected)
        sources[entry["path"]] = expected
    row_path = Path(cache["root"]) / "val_row_ids.npy"
    row_identity = identity(root / row_path)
    if row_identity["sha256"] != cache["row_identity"]["val_row_file_sha256"]:
        raise ValueError("E1 validation row 文件身份漂移")
    sources[str(row_path)] = row_identity
    cache_manifest = json.loads((root / cache["manifest"]["path"]).read_text(encoding="utf-8"))
    lock = {
        "protocol": PROTOCOL, "status": "implementation_locked_runtime_acceptance_pending",
        "prepared_at": datetime.now(timezone.utc).isoformat(), "preparation_git": git_state(root),
        "conditions": list(CONDITIONS), "seeds": list(SEEDS), "split": "val", "windows": WINDOW_COUNT,
        "samp_ids": 7, "batch_size": 128, "amp_dtype": "bfloat16", "full_atol": FULL_ATOL,
        "full_rtol": FULL_RTOL,
        "w0_entries": entries, "cache_lock": cache, "source_files": sources,
        "dataset_index": {"path": cache_manifest["dataset_index"], "sha256": cache_manifest["dataset_index_sha256"]},
        "source_verification": "SHA-256 and size only; checkpoint/cache payloads not deserialized",
    }
    if (root / INDEX_LOCK).exists():
        index = json.loads((root / INDEX_LOCK).read_text(encoding="utf-8"))
        validate_index_lock(index)
        if index_self_checks(index) != index["self_checks"]:
            raise ValueError("E1 已有索引锁自检回执不一致")
    else:
        index = make_index_lock()
        write_json(root / INDEX_LOCK, index)
    if (root / PREVIOUS_IMPLEMENTATION_LOCK).exists():
        previous = json.loads((root / PREVIOUS_IMPLEMENTATION_LOCK).read_text(encoding="utf-8"))
        verify_file(root / INDEX_LOCK, previous["index_lock"])
        lock["supersedes"] = {"path": str(PREVIOUS_IMPLEMENTATION_LOCK), **identity(root / PREVIOUS_IMPLEMENTATION_LOCK)}
        lock["revision"] = "用户指定 FULL 容差为 1e-3 * abs(reference)，保留既有尺度索引"
    lock["index_lock"] = {"path": str(INDEX_LOCK), **identity(root / INDEX_LOCK)}
    # 覆盖原生推理、数据、指标及其本仓库依赖；第三方实现由环境回执记录。
    code_paths = sorted((root / "resp_train").rglob("*.py"))
    code_paths.extend(root / item for item in (SCRIPT, TEST, PROTOCOL_PATH))
    lock["code_files"] = {str(path.relative_to(root)): identity(path) for path in code_paths}
    write_json(root / IMPLEMENTATION_LOCK, lock)
    return root / INDEX_LOCK, root / IMPLEMENTATION_LOCK


def load_locks(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], str]:
    lock_path = root / IMPLEMENTATION_LOCK
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock["protocol"] != PROTOCOL or tuple(lock["conditions"]) != CONDITIONS or tuple(lock["seeds"]) != SEEDS
            or lock["split"] != "val" or lock["windows"] != WINDOW_COUNT or lock["batch_size"] != 128
            or lock["amp_dtype"] != "bfloat16" or lock["full_atol"] != FULL_ATOL
            or lock.get("full_rtol") != FULL_RTOL):
        raise ValueError("E1 implementation lock 合同不一致")
    for relative, expected in lock["code_files"].items():
        verify_file(root / relative, expected)
    if lock["index_lock"]["path"] != str(INDEX_LOCK):
        raise ValueError("E1 index lock 路径漂移")
    verify_file(root / INDEX_LOCK, lock["index_lock"])
    index = json.loads((root / INDEX_LOCK).read_text(encoding="utf-8"))
    validate_index_lock(index)
    if index_self_checks(index) != index["self_checks"]:
        raise ValueError("E1 index 自检回执不一致")
    return lock, index, sha256_file(lock_path)


@contextmanager
def attempt(phase: str, lock_hash: str, root: Path = ROOT) -> Iterator[Path]:
    if phase not in {"gpu_smoke", "validation"}:
        raise ValueError("未知 E1 phase")
    parent = root / OUTPUT / phase
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    started = {"protocol": PROTOCOL, "phase": phase, "implementation_lock_sha256": lock_hash,
               "started_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv}
    write_json(output / "lifecycle_started.json", {**started, "status": "running"})
    print(f"E1 attempt: {output}", flush=True)
    try:
        yield output
        write_json(output / "lifecycle_completed.json", {**started, "status": "completed",
                   "ended_at": datetime.now(timezone.utc).isoformat()})
        manifest = {"protocol": PROTOCOL, "phase": phase, "status": "completed",
                    "implementation_lock_sha256": lock_hash,
                    "files": {str(path.relative_to(output)): identity(path)
                              for path in sorted(output.rglob("*")) if path.is_file()}}
        write_json(output / "manifest.json", manifest)
        write_json(output / "freeze_receipt.json", {"protocol": PROTOCOL, "phase": phase,
                   "manifest": identity(output / "manifest.json")})
    except BaseException as exc:
        write_json(output / "lifecycle_failed.json", {**started, "status": "failed",
                   "ended_at": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc()})
        raise


def environment(device: str) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "torch", "omegaconf", "mamba-ssm", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "device": device, "gpu_name": torch.cuda.get_device_name(torch.device(device)),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads()}


def require_gpu(device: str) -> None:
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("E1 运行需要 CUDA；CPU 使用 synthetic 定向测试")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("E1 原生依赖检查失败: " + "; ".join(problems))


class FilmStore:
    """FULL raw 以 float32 无损保存 bf16 输出，后续按同一 row 顺序读取配对。"""

    def __init__(self, directory: Path, count: int, *, full: bool) -> None:
        self.count, self.offset, self.full = count, 0, full
        self.values: list[np.ndarray] = []
        self.dtypes: set[str] = set()
        self.arrays = []
        for name in ("gamma", "beta"):
            path = directory / f"full_{name}_raw.npy"
            if full:
                # 原始 FiLM 文件在本 attempt 内保留，以使配对来源可审计。
                with path.open("xb"):
                    pass
                array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(count, 96, 1800))
            else:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != (count, 96, 1800) or array.dtype != np.float32:
                    raise ValueError("E1 FULL FiLM 存储 identity 不一致")
            self.arrays.append(array)

    def __call__(self, raw: tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(raw) != 2 or raw[0].shape != raw[1].shape or tuple(raw[0].shape[1:]) != (96, 1800):
            raise ValueError("E1 原始 FiLM shape 不一致")
        stop = self.offset + raw[0].shape[0]
        if stop > self.count:
            raise ValueError("E1 FiLM row 数超出合同")
        paired = []
        for tensor, saved in zip(raw, self.arrays, strict=True):
            if tensor.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(f"E1 原始 FiLM dtype 不合格: {tensor.dtype}")
            self.dtypes.add(str(tensor.dtype))
            value = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError("E1 原始 FiLM 非有限")
            if self.full:
                saved[self.offset:stop] = value
            paired.append(raw_pair_mae(value, saved[self.offset:stop]))
        self.values.append(np.stack(paired, axis=1))
        self.offset = stop

    def finish(self) -> np.ndarray:
        if self.offset != self.count:
            raise ValueError("E1 FiLM row 数不完整")
        if self.full:
            for array in self.arrays:
                array.flush()
        result = np.concatenate(self.values)
        if self.full and np.any(result != 0):
            raise RuntimeError("E1 FULL FiLM 自配对必须为零")
        return result


def checked_batches(loader: Any, rows: pd.DataFrame) -> Iterator[Mapping[str, Any]]:
    """在送入 GPU 前核对实际 batch identity、输入与 target；尾 batch 完整保留。"""
    offset = 0
    for batch in loader:
        count = len(batch["x"])
        expected = rows.iloc[offset:offset + count]
        if len(expected) != count or count == 0:
            raise ValueError("E1 loader 多余或空 batch")
        for key in ("dataset_row_id", "samp_id", "split"):
            actual = batch["meta"][key]
            if torch.is_tensor(actual):
                actual = actual.cpu().numpy()
            if not np.array_equal(np.asarray(actual), expected[key].to_numpy()):
                raise ValueError(f"E1 batch identity/order 错误: {key}")
        if set(batch.get("tf", {})) != {"w"} or tuple(batch["tf"]["w"].shape) != (count, 97, 360):
            raise ValueError("E1 W batch shape/keys 错误")
        for tensor in (batch["x"], batch["target"], batch["tf"]["w"]):
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError("E1 batch input/target 非有限")
        offset += count
        yield batch
    if offset != len(rows):
        raise ValueError("E1 loader 未覆盖完整 rows")


def load_model(entry: Mapping[str, Any], device: str):
    cfg = _load_audit_config(entry, device_name=device)
    if (int(cfg.training.batch_size) != 128 or bool(cfg.training.drop_last)
            or not cfg.training.use_amp or str(cfg.training.amp_dtype) != "bfloat16"):
        raise ValueError("E1 历史推理 batch/AMP 合同漂移")
    checkpoint = torch.load(ROOT / entry["run_dir"] / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    for name, value in checkpoint["model_state_dict"].items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"E1 checkpoint 非有限: {name}")
    model = build_crd_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return cfg, model


def evaluate_condition(model: torch.nn.Module, cfg: Any, entry: Mapping[str, Any], condition: str,
                       index: Mapping[str, Any], data: Any, output: Path, seed_dir: Path, device: str) -> pd.DataFrame:
    seed = int(entry["seed"])
    directory = output / f"seed_{seed}" / condition
    directory.mkdir(parents=True, exist_ok=False)
    store = FilmStore(seed_dir, len(data.rows), full=condition == "FULL")
    wrapper = ScaleAuditModel(model, condition, index, store)
    predictions = collect_predictions(wrapper, checked_batches(data.loader, data.rows), device=device,
                                      max_windows=len(data.rows), use_amp=True)
    film_values = store.finish()
    for key in ("dataset_row_id", "samp_id", "split"):
        if not np.array_equal(predictions[key], data.rows[key].to_numpy()):
            raise ValueError(f"E1 prediction identity/order 错误: {key}")
    metrics = evaluate_task_predictions(predictions, cfg, include_test_only=False, method="crd_tf102_w")
    validate_metrics(metrics, data.rows)
    if condition != "FULL":
        full_metrics = pd.read_csv(output / f"seed_{seed}" / "FULL" / "metrics.csv")
        for key in ("whole_rr_target_eligible", "local_rr_target_eligible", "local_rr_target_eligible_windows",
                    "joint_target_eligible", "envelope_spearman_target_eligible"):
            if not np.array_equal(metrics[key].to_numpy(), full_metrics[key].to_numpy()):
                raise ValueError(f"E1 干预改变 target eligibility: {key}")
    summary = summarize_task_metrics(metrics)
    if not np.isfinite(summary[[f"{key}_mean" for key in PRIMARY]].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("E1 summary 主指标非有限")
    metrics.insert(0, "seed", seed)
    metrics.insert(0, "condition", condition)
    metrics.to_csv(directory / "metrics.csv", index=False)
    film = data.rows[["dataset_row_id", "split", "samp_id"]].copy()
    film.insert(0, "seed", seed)
    film.insert(0, "condition", condition)
    for position, key in enumerate(FILM_COLUMNS):
        film[key] = film_values[:, position]
        summary[key] = float(film[key].mean())
    film.to_csv(directory / "film_pair_metrics.csv", index=False)
    summary.insert(0, "selected_epoch", entry["selected_epoch"])
    summary.insert(0, "checkpoint_sha256", entry["checkpoint"]["sha256"])
    summary.insert(0, "seed", seed)
    summary.insert(0, "condition", condition)
    summary.to_csv(directory / "summary.csv", index=False)
    write_json(directory / "evaluation_receipt.json", {
        "seed": seed, "condition": condition, "split": "val", "rows": len(metrics),
        "row_order_sha256": array_hash(data.rows.dataset_row_id.to_numpy()), "film_raw_dtypes": sorted(store.dtypes),
        "prediction_shape": list(predictions["r_tho_hat"].shape), "finite_eligibility_passed": True,
        "prediction_degeneracy_count": 0,
    })
    print(f"E1 完成 seed={seed} condition={condition} rows={len(metrics)}", flush=True)
    return summary


def run_validation(*, device: str = "cuda:0", gpu_receipt: Path) -> Path:
    lock, index, lock_hash = load_locks()
    with attempt("validation", lock_hash) as output:
        git = git_state(require_clean=True)
        require_gpu(device)
        smoke = validate_gpu_receipt(gpu_receipt, lock_hash)
        write_json(output / "environment.json", {**environment(device), "git": git})
        write_json(output / "implementation_lock.json", lock)
        write_json(output / "scale_indices.json", index)
        write_json(output / "gpu_acceptance_source.json", smoke)
        write_json(output / "access_started.json", {
            "status": "planned_access", "split": "val", "source_files": list(lock["source_files"]),
            "dataset_index": lock["dataset_index"], "checkpoint_seeds": list(SEEDS),
            "waveform_scope": "2675 admitted validation windows", "cache_scope": "frozen validation W cache",
        })
        for relative, expected in lock["source_files"].items():
            verify_file(ROOT / relative, expected)
        index_identity = lock["dataset_index"]
        if sha256_file(Path(index_identity["path"])) != index_identity["sha256"]:
            raise ValueError("E1 dataset index identity 漂移")
        entries = lock["w0_entries"]
        cfg = _load_audit_config(entries[0], device_name=device)
        data = build_window_data(cfg, split="val", max_windows=None, sample_strategy=str(cfg.data.val_sample_strategy),
                                 sample_seed=int(cfg.data.val_sample_seed), shuffle=False)
        validate_rows(data.rows)
        if data.rows.samp_id.nunique() != 7 or len(data.dataset) != WINDOW_COUNT:
            raise ValueError("E1 validation 窗口或 samp_id 数量不匹配")
        if array_hash(np.sort(data.rows.dataset_row_id.to_numpy())) != lock["cache_lock"]["row_identity"]["val_row_content_sha256"]:
            raise ValueError("E1 validation row 集合 identity 漂移")
        data.rows.to_csv(output / "validation_rows.csv", index=False)
        frequencies = np.load(ROOT / lock["cache_lock"]["frequency_file"]["path"], allow_pickle=False)
        if frequencies.shape != (97,) or not np.isfinite(frequencies).all():
            raise ValueError("E1 frequency identity/finite 错误")
        write_json(output / "condition_frequencies.json", {
            name: frequencies[index["conditions"][name]["index"]].tolist() for name in CONDITIONS})
        write_json(output / "access_receipt.json", {
            "split": "val", "rows": WINDOW_COUNT, "samp_ids": 7,
            "dataset_index": index_identity, "source_files_verified": lock["source_files"],
            "row_order_sha256": array_hash(data.rows.dataset_row_id.to_numpy()),
            "waveform_access": "admitted validation rows through frozen ResearchV2WindowDataset",
            "cache_access": "val_row_ids.npy / val_w.npy / w_frequencies_hz.npy; read-only",
            "checkpoint_seeds": list(SEEDS), "metadata_scope": "existing shared dataset index and validation rows",
        })
        summaries, anchors = [], []
        # FULL 全部通过之后才进入任何干预，完整保留 raw FiLM 作为配对来源。
        for entry in entries:
            seed = int(entry["seed"])
            seed_dir = output / f"seed_{seed}"
            seed_dir.mkdir(exist_ok=False)
            cfg, model = load_model(entry, device)
            OmegaConf.save(cfg, seed_dir / "resolved_config.yaml")
            summary = evaluate_condition(model, cfg, entry, "FULL", index, data, output, seed_dir, device)
            reference = pd.read_csv(ROOT / entry["run_dir"] / "metrics_summary.csv")
            anchors.append(check_full_anchor(summary, reference, seed))
            summaries.append(summary)
            del model
            torch.cuda.empty_cache()
        write_json(output / "full_anchor_receipt.json", {"passed": True, "seeds": anchors})
        for entry in entries:
            cfg, model = load_model(entry, device)
            seed_dir = output / f"seed_{entry['seed']}"
            for condition in CONDITIONS[1:]:
                summaries.append(evaluate_condition(model, cfg, entry, condition, index, data, output, seed_dir, device))
            del model
            torch.cuda.empty_cache()
        summary = pd.concat(summaries, ignore_index=True)
        aggregate, paired, delta = summarize_pairs(summary)
        summary.to_csv(output / "seed_summary.csv", index=False)
        aggregate.to_csv(output / "three_seed_summary.csv", index=False)
        paired.to_csv(output / "paired_seed_delta.csv", index=False)
        delta.to_csv(output / "paired_delta_summary.csv", index=False)
        write_json(output / "completion_receipt.json", {"split": "val", "conditions": list(CONDITIONS),
                   "seeds": list(SEEDS), "evaluation_count": 12, "metric_rows": 12 * WINDOW_COUNT,
                   "film_rows": 12 * WINDOW_COUNT, "full_anchor_passed": True,
                   "interpretation": "continuous effects; interpretation after complete results"})
    return output


def validate_gpu_receipt(directory: Path, lock_hash: str) -> dict[str, Any]:
    directory = directory.resolve()
    if (directory / "lifecycle_failed.json").exists():
        raise ValueError("E1 GPU attempt 存在失败状态")
    manifest_path = directory / "manifest.json"
    freeze = json.loads((directory / "freeze_receipt.json").read_text(encoding="utf-8"))
    verify_file(manifest_path, freeze["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["protocol"] != PROTOCOL or manifest["phase"] != "gpu_smoke"
            or manifest["status"] != "completed" or manifest["implementation_lock_sha256"] != lock_hash):
        raise ValueError("E1 GPU 验收回执身份不匹配")
    if not {"gpu_acceptance.json", "lifecycle_completed.json"}.issubset(manifest["files"]):
        raise ValueError("E1 GPU manifest 缺少必需产物")
    for relative, expected in manifest["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("E1 GPU manifest 路径越界")
        verify_file(path, expected)
    receipt = json.loads((directory / "gpu_acceptance.json").read_text(encoding="utf-8"))
    if receipt["passed"] is not True or receipt["conditions"] != list(CONDITIONS) or receipt["batch_size"] != 1:
        raise ValueError("E1 GPU 验收未通过")
    return {"directory": str(directory), "manifest": identity(manifest_path), "receipt": receipt}


def run_gpu_smoke(*, device: str = "cuda:0") -> Path:
    lock, index, lock_hash = load_locks()
    with attempt("gpu_smoke", lock_hash) as output:
        git = git_state(require_clean=True)
        require_gpu(device)
        write_json(output / "environment.json", {**environment(device), "git": git})
        write_json(output / "implementation_lock.json", lock)
        write_json(output / "scale_indices.json", index)
        from resp_train.crd.tf_v1_model import CRDTfV1Model

        torch.manual_seed(20260915)
        model = CRDTfV1Model("crd_tf102_w", SEEDS[0]).to(device).eval()
        # 新模型 FiLM 投影原本为零；synthetic 验收激活该路径以覆盖实际数据流。
        with torch.no_grad():
            model.branches["w"].final_projection.weight.normal_(std=1e-4)
            model.branches["w"].final_projection.bias.normal_(std=1e-4)
        x = torch.randn(1, 1, 18000, device=device)
        w = torch.randn(1, 97, 360, device=device)
        original = w.clone()
        raw_full: list[np.ndarray] = []
        details = []
        full_deltas = {}
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            native = model(x, tf={"w": w})
            for condition in CONDITIONS:
                captured: list[tuple[torch.Tensor, torch.Tensor]] = []
                result = ScaleAuditModel(model, condition, index, captured.append).to(device)(x, tf={"w": w})
                if set(result) != set(native):
                    raise ValueError("E1 GPU output keys 漂移")
                for key in result:
                    if result[key].shape != native[key].shape or not bool(torch.isfinite(result[key]).all()):
                        raise FloatingPointError("E1 GPU output shape/finite 失败")
                    if condition == "FULL":
                        full_deltas[key] = float((result[key].float() - native[key].float()).abs().max())
                raw = [value.detach().float().cpu().numpy() for value in captured[0]]
                if any(value.shape != (1, 96, 1800) for value in raw):
                    raise ValueError("E1 GPU 原始 FiLM shape 错误")
                if condition == "FULL":
                    raw_full = raw
                pairs = [raw_pair_mae(value, anchor).tolist() for value, anchor in zip(raw, raw_full, strict=True)]
                details.append({"condition": condition, "raw_pair_mae": pairs,
                                "output_shapes": {key: list(value.shape) for key, value in result.items()}})
        if not torch.equal(w, original):
            raise RuntimeError("E1 synthetic 输入被修改")
        torch.cuda.synchronize(device)
        write_json(output / "gpu_acceptance.json", {"protocol": PROTOCOL, "passed": True,
                   "conditions": list(CONDITIONS), "batch_size": 1, "fixture": "synthetic random input, fresh W0 with active FiLM",
                   "seed": 20260915, "native_full_max_abs_deltas": full_deltas,
                   "native_full_exact": all(value == 0 for value in full_deltas.values()),
                   "source_unchanged": True, "details": details})
        write_json(output / "access_receipt.json", {"data": "synthetic", "checkpoint": "fresh initialization",
                   "historical_artifacts_read": False})
    return output
