from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from omegaconf import OmegaConf


REQUIRED_MODELS = (
    "g0_time_only",
    "g0_f0_native_stft_pre_mixer",
    "g3_c_wide_8p0",
    "g3_c_bandenergy",
)
SELECTION_SOURCE = "validation_topk_legacy_task_selection"


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    seed: int
    checkpoint: Path
    selection_source: str


@dataclass(frozen=True)
class ExportPlan:
    specs: tuple[ComparisonSpec, ...]
    output_dir: Path
    dataset_row_id_path: Path
    bcg_input_path: Path
    tho_ref_path: Path
    prediction_paths: dict[str, Path]
    devices: tuple[str, ...]


@dataclass(frozen=True)
class PredictionTask:
    label: str
    seed: int
    checkpoint_path: Path
    config_path: Path
    output_path: Path
    device: str
    expected_row_ids: np.ndarray


COMPARISON_CONFIG_KEYS = (
    "data.dataset_root",
    "data.index_csv",
    "data.format",
    "data.input_set",
    "data.test_split",
    "data.max_test_windows",
    "data.test_sample_strategy",
    "data.test_sample_seed",
    "window.target_fs",
    "loss.spectrum_low_hz",
    "loss.spectrum_high_hz",
)


def spec_fingerprint(specs: Iterable[ComparisonSpec]) -> str:
    payload = [
        {
            "label": spec.label,
            "seed": spec.seed,
            "checkpoint": str(spec.checkpoint),
            "selection_source": spec.selection_source,
        }
        for spec in specs
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_signal_matrix(
    values: np.ndarray,
    *,
    name: str,
    expected_rows: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} 必须为 [N, T]，当前 shape={array.shape}")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError(f"{name} 行数不一致: {array.shape[0]} != {expected_rows}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限值")
    return array.astype(np.float32, copy=False)


def atomic_save_npy(path: str | Path, values: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npy"
    )
    try:
        np.save(temporary_path, values)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def _validate_dataset_row_ids(values: np.ndarray, *, expected_rows: int | None = None) -> np.ndarray:
    row_ids = np.asarray(values, dtype=np.int64).reshape(-1)
    if row_ids.size == 0:
        raise ValueError("dataset_row_id 不能为空")
    if expected_rows is not None and row_ids.size != expected_rows:
        raise ValueError(f"dataset_row_id 行数不一致: {row_ids.size} != {expected_rows}")
    if np.unique(row_ids).size != row_ids.size:
        raise ValueError("dataset_row_id 存在重复值")
    return row_ids


def validate_prediction_batch(
    dataset_row_id: np.ndarray,
    expected_row_ids: np.ndarray,
    tho_ref: np.ndarray,
    expected_tho_ref: np.ndarray,
) -> None:
    actual_ids = _validate_dataset_row_ids(dataset_row_id)
    expected_ids = _validate_dataset_row_ids(expected_row_ids, expected_rows=actual_ids.size)
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError("dataset_row_id 顺序或集合与共享缓存不一致")
    actual_target = validate_signal_matrix(tho_ref, name="预测返回的 THO target", expected_rows=actual_ids.size)
    expected_target = validate_signal_matrix(
        expected_tho_ref,
        name="共享 THO target",
        expected_rows=actual_ids.size,
    )
    if actual_target.shape != expected_target.shape or not np.array_equal(actual_target, expected_target):
        raise ValueError("预测返回的 THO target 与共享缓存不一致")


def write_cache_arrays(
    output_dir: str | Path,
    *,
    dataset_row_id: np.ndarray,
    bcg_input: np.ndarray,
    tho_ref: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Path]:
    output_path = Path(output_dir)
    row_ids = _validate_dataset_row_ids(dataset_row_id)
    bcg = validate_signal_matrix(bcg_input, name="BCG input", expected_rows=row_ids.size)
    target = validate_signal_matrix(tho_ref, name="THO target", expected_rows=row_ids.size)
    if bcg.shape != target.shape:
        raise ValueError(f"BCG input 和 THO target shape 不一致: {bcg.shape} != {target.shape}")

    written = {
        "dataset_row_id": atomic_save_npy(output_path / "dataset_row_id.npy", row_ids),
        "bcg_input": atomic_save_npy(output_path / "bcg_input.npy", bcg),
        "tho_ref": atomic_save_npy(output_path / "tho_ref.npy", target),
    }
    for label, prediction in predictions.items():
        values = validate_signal_matrix(prediction, name=f"{label}.r_tho_hat", expected_rows=row_ids.size)
        if values.shape != target.shape:
            raise ValueError(f"{label}.r_tho_hat 和 THO target shape 不一致")
        written[f"prediction:{label}"] = atomic_save_npy(
            output_path / "predictions" / str(label) / "r_tho_hat.npy",
            values,
        )
    return written


def validate_config_consistency(reference: Any, candidates: Iterable[Any]) -> None:
    for candidate in candidates:
        for key in COMPARISON_CONFIG_KEYS:
            expected = OmegaConf.select(reference, key)
            actual = OmegaConf.select(candidate, key)
            if actual != expected:
                raise ValueError(f"配置字段不一致 {key}: {actual!r} != {expected!r}")


def build_prediction_tasks(
    specs: Iterable[ComparisonSpec],
    *,
    output_dir: str | Path,
    expected_row_ids: np.ndarray,
    devices: Iterable[str],
) -> tuple[PredictionTask, ...]:
    resolved_specs = tuple(specs)
    resolved_devices = tuple(str(device).strip() for device in devices)
    if not resolved_devices or any(not device for device in resolved_devices):
        raise ValueError("至少需要一个非空 device")
    row_ids = _validate_dataset_row_ids(expected_row_ids)
    root = Path(output_dir)
    tasks = tuple(
        PredictionTask(
            label=spec.label,
            seed=spec.seed,
            checkpoint_path=spec.checkpoint,
            config_path=spec.checkpoint.parent / "config.yaml",
            output_path=root / "predictions" / spec.label / "r_tho_hat.npy",
            device=resolved_devices[index % len(resolved_devices)],
            expected_row_ids=row_ids,
        )
        for index, spec in enumerate(resolved_specs)
    )
    if len({task.output_path for task in tasks}) != len(tasks):
        raise ValueError("预测任务输出路径冲突")
    return tasks


def _config_path_for_checkpoint(checkpoint_path: Path) -> Path:
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"checkpoint 同目录缺少 config.yaml: {config_path}")
    return config_path


def _test_bundle(cfg: Any):
    from resp_train.data.factory import build_window_data

    return build_window_data(
        cfg,
        split=str(cfg.data.get("test_split", "test")),
        max_windows=cfg.data.get("max_test_windows", None),
        sample_strategy=str(cfg.data.get("test_sample_strategy", "stratified_random")),
        sample_seed=int(cfg.data.get("test_sample_seed", cfg.training.get("seed", 0))),
        shuffle=False,
    )


def _write_shared_cache(plan: ExportPlan, cfg: Any) -> tuple[np.ndarray, Path]:
    from numpy.lib.format import open_memmap

    bundle = _test_bundle(cfg)
    dataset = bundle.dataset
    if len(dataset) == 0:
        raise RuntimeError("test 数据为空，无法导出比较缓存")
    n_rows = len(dataset)
    n_samples = int(cfg.window.duration_samples)
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    temporary_paths = {
        "dataset_row_id": plan.dataset_row_id_path.with_name(f".{plan.dataset_row_id_path.name}.{uuid.uuid4().hex}.tmp.npy"),
        "bcg_input": plan.bcg_input_path.with_name(f".{plan.bcg_input_path.name}.{uuid.uuid4().hex}.tmp.npy"),
        "tho_ref": plan.tho_ref_path.with_name(f".{plan.tho_ref_path.name}.{uuid.uuid4().hex}.tmp.npy"),
    }
    try:
        ids = open_memmap(temporary_paths["dataset_row_id"], mode="w+", dtype=np.int64, shape=(n_rows,))
        bcg = open_memmap(temporary_paths["bcg_input"], mode="w+", dtype=np.float32, shape=(n_rows, n_samples))
        tho = open_memmap(temporary_paths["tho_ref"], mode="w+", dtype=np.float32, shape=(n_rows, n_samples))
        for position in range(n_rows):
            sample = dataset[position]
            ids[position] = int(sample["meta"]["dataset_row_id"])
            x = np.asarray(sample["x"].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            target = np.asarray(sample["target"].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            if x.size != n_samples or target.size != n_samples:
                raise ValueError(f"test window 长度异常 row={ids[position]}")
            bcg[position] = x
            tho[position] = target
        ids.flush()
        bcg.flush()
        tho.flush()
        row_ids = _validate_dataset_row_ids(np.asarray(ids))
        validate_signal_matrix(np.asarray(bcg), name="BCG input", expected_rows=n_rows)
        validate_signal_matrix(np.asarray(tho), name="THO target", expected_rows=n_rows)
        for key, destination in (
            ("dataset_row_id", plan.dataset_row_id_path),
            ("bcg_input", plan.bcg_input_path),
            ("tho_ref", plan.tho_ref_path),
        ):
            os.replace(temporary_paths[key], destination)
    finally:
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()
    return row_ids, plan.tho_ref_path


def _export_prediction_task(task: PredictionTask, shared_target_path: str | Path) -> tuple[str, Path]:
    import torch

    from resp_train.config import load_config
    from resp_train.engine import collect_predictions
    from resp_train.experiments.tho import _validate_checkpoint_config
    from resp_train.models.registry import build_model

    cfg = load_config(task.config_path)
    bundle = _test_bundle(cfg)
    expected_ids = _validate_dataset_row_ids(task.expected_row_ids)
    actual_ids = np.asarray(
        [int(value) for value in bundle.dataset.rows["dataset_row_id"].tolist()],
        dtype=np.int64,
    )
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError(f"{task.label} 的 test dataset_row_id 与共享缓存不一致")
    device = torch.device(task.device)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(task.checkpoint_path, map_location=device)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = collect_predictions(model, bundle.loader, device=device, max_windows=len(bundle.dataset))
    expected_target = np.load(shared_target_path, mmap_mode="r")
    validate_prediction_batch(
        np.asarray(predictions["dataset_row_id"], dtype=np.int64),
        expected_ids,
        np.asarray(predictions["tho_ref"]),
        np.asarray(expected_target),
    )
    values = validate_signal_matrix(
        np.asarray(predictions["r_tho_hat"]),
        name=f"{task.label}.r_tho_hat",
        expected_rows=expected_ids.size,
    )
    atomic_save_npy(task.output_path, values)
    return task.label, task.output_path


def _code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_complete_manifest(plan: ExportPlan, *, config_paths: dict[str, Path]) -> Path:
    array_paths = {
        "dataset_row_id": plan.dataset_row_id_path,
        "bcg_input": plan.bcg_input_path,
        "tho_ref": plan.tho_ref_path,
        **{f"prediction:{label}": path for label, path in plan.prediction_paths.items()},
    }
    arrays = {}
    for key, path in array_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"缓存数组缺失，拒绝写 complete manifest: {path}")
        array = np.load(path, mmap_mode="r")
        arrays[key] = {"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype), "sha256": sha256_file(path)}
    payload = {
        "schema_version": 1,
        "status": "complete",
        "selection_source": SELECTION_SOURCE,
        "spec_fingerprint": spec_fingerprint(plan.specs),
        "specs": [
            {
                "label": spec.label,
                "seed": spec.seed,
                "checkpoint": str(spec.checkpoint),
                "checkpoint_sha256": sha256_file(spec.checkpoint),
                "config": str(config_paths[spec.label]),
                "config_sha256": sha256_file(config_paths[spec.label]),
            }
            for spec in plan.specs
        ],
        "dataset_row_id_sha256": sha256_file(plan.dataset_row_id_path),
        "dataset_index_sha256": sha256_file(Path(str(_load_config(config_paths[plan.specs[0].label]).data.dataset_root)) / str(_load_config(config_paths[plan.specs[0].label]).data.index_csv)),
        "arrays": arrays,
        "code_version": _code_version(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = plan.output_dir / "manifest.json"
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest_path


def _load_config(path: Path):
    from resp_train.config import load_config

    return load_config(path)


def export_comparison_cache(plan: ExportPlan, *, max_parallel: int) -> Path:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from resp_train.config import load_config

    if max_parallel < 1:
        raise ValueError("--max-parallel 必须 >= 1")
    config_paths = {spec.label: _config_path_for_checkpoint(spec.checkpoint) for spec in plan.specs}
    configs = {label: load_config(path) for label, path in config_paths.items()}
    reference_cfg = configs[plan.specs[0].label]
    validate_config_consistency(reference_cfg, configs.values())
    expected_ids, shared_target_path = _write_shared_cache(plan, reference_cfg)
    tasks = build_prediction_tasks(
        plan.specs,
        output_dir=plan.output_dir,
        expected_row_ids=expected_ids,
        devices=plan.devices,
    )
    context = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=min(max_parallel, len(tasks)), mp_context=context) as pool:
            futures = {pool.submit(_export_prediction_task, task, shared_target_path): task for task in tasks}
            for future in as_completed(futures):
                label, output_path = future.result()
                print(f"completed {label}: {output_path}", flush=True)
    except Exception:
        raise
    return _write_complete_manifest(plan, config_paths=config_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 G 系列四模型完整测试集预测缓存")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", action="append", dest="devices", required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")
    specs = load_comparison_specs(args.spec)
    plan = build_export_plan(
        specs,
        output_dir=args.output_dir,
        devices=args.devices,
        resume=bool(args.resume),
    )
    if args.resume:
        manifest_path = plan.output_dir / "manifest.json"
        print(f"resume accepted: {manifest_path}", flush=True)
        return
    for index, spec in enumerate(plan.specs):
        print(
            f"plan label={spec.label} seed={spec.seed} device={plan.devices[index % len(plan.devices)]} "
            f"checkpoint={spec.checkpoint} output={plan.prediction_paths[spec.label]}",
            flush=True,
        )
    if args.dry_run:
        print("dry-run complete: shared=3 arrays predictions=4 arrays estimated_storage≈1 GB", flush=True)
        return
    manifest_path = export_comparison_cache(plan, max_parallel=int(args.max_parallel))
    print(f"complete manifest: {manifest_path}", flush=True)


def load_comparison_specs(path: str | Path, *, require_paths: bool = True) -> tuple[ComparisonSpec, ...]:
    spec_path = Path(path)
    if not spec_path.exists():
        raise FileNotFoundError(f"checkpoint spec 不存在: {spec_path}")
    with spec_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"label", "seed", "checkpoint", "selection_source"}
        missing = sorted(required_columns - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"checkpoint spec 缺少列: {missing}")
        rows = list(reader)

    labels = tuple(str(row.get("label", "")).strip() for row in rows)
    if labels != REQUIRED_MODELS:
        raise ValueError(f"checkpoint spec 的 label 必须依次为: {list(REQUIRED_MODELS)}")

    specs: list[ComparisonSpec] = []
    checkpoints: set[Path] = set()
    for row in rows:
        checkpoint = Path(str(row["checkpoint"])).expanduser()
        if checkpoint in checkpoints:
            raise ValueError(f"checkpoint spec 存在重复 checkpoint: {checkpoint}")
        checkpoints.add(checkpoint)
        if require_paths and not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
        selection_source = str(row["selection_source"]).strip()
        if selection_source != SELECTION_SOURCE:
            raise ValueError(f"selection_source 必须为 {SELECTION_SOURCE!r}")
        specs.append(
            ComparisonSpec(
                label=str(row["label"]),
                seed=int(row["seed"]),
                checkpoint=checkpoint,
                selection_source=selection_source,
            )
        )
    return tuple(specs)


def build_export_plan(
    specs: Iterable[ComparisonSpec],
    *,
    output_dir: str | Path,
    devices: Iterable[str],
    resume: bool = False,
) -> ExportPlan:
    resolved_specs = tuple(specs)
    if tuple(spec.label for spec in resolved_specs) != REQUIRED_MODELS:
        raise ValueError("导出计划必须使用固定的四个模型")
    output_path = Path(output_dir)
    if output_path.exists() and not resume:
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_path}")
    if output_path.exists() and resume:
        manifest_path = output_path / "manifest.json"
        if not manifest_path.exists():
            raise FileExistsError(f"输出目录缺少 complete manifest，拒绝 resume: {output_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"resume manifest 不是有效 JSON: {manifest_path}") from exc
        if manifest.get("status") != "complete":
            raise ValueError(f"resume manifest status 必须为 complete: {manifest_path}")
        actual_fingerprint = manifest.get("spec_fingerprint")
        expected_fingerprint = spec_fingerprint(resolved_specs)
        if actual_fingerprint != expected_fingerprint:
            raise ValueError("resume manifest 的 spec_fingerprint 与当前 spec 不一致")
    resolved_devices = tuple(str(device).strip() for device in devices)
    if not resolved_devices or any(not device for device in resolved_devices):
        raise ValueError("至少需要一个非空 --device")
    return ExportPlan(
        specs=resolved_specs,
        output_dir=output_path,
        dataset_row_id_path=output_path / "dataset_row_id.npy",
        bcg_input_path=output_path / "bcg_input.npy",
        tho_ref_path=output_path / "tho_ref.npy",
        prediction_paths={spec.label: output_path / "predictions" / spec.label / "r_tho_hat.npy" for spec in resolved_specs},
        devices=resolved_devices,
    )


if __name__ == "__main__":
    main()
