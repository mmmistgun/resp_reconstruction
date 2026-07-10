from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import uuid
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.experiments.tho import _resolve_config_path, _validate_checkpoint_config
from resp_train.models.registry import build_model


@dataclass(frozen=True)
class ExportTask:
    label: str
    seed: int
    checkpoint_path: Path
    config_path: Path
    labels_path: Path
    labels_sha256: str
    output_path: Path
    manifest_path: Path
    device: str
    n_selected_rows: int


def load_positive_row_ids(labels_path: Path) -> list[int]:
    frame = pd.read_csv(labels_path)
    required = {"dataset_row_id", "split", "harmonic_positive"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"谐波标签缺少列: {sorted(missing)}")
    if frame["dataset_row_id"].duplicated().any():
        raise ValueError("谐波标签存在重复 dataset_row_id")
    if set(frame["split"].astype(str)) != {"test"}:
        raise ValueError("预测导出只允许 held-out test 标签")
    positive = frame.loc[frame["harmonic_positive"].astype(bool), "dataset_row_id"].astype(int).tolist()
    if not positive:
        raise ValueError("谐波标签中没有阳性窗口，无法导出预测")
    return positive


def build_export_tasks(
    *,
    spec_path: Path,
    labels_path: Path,
    output_dir: Path,
    devices: list[str],
) -> list[ExportTask]:
    if not devices or any(not str(device).strip() for device in devices):
        raise ValueError("至少需要一个非空 device")
    spec = pd.read_csv(spec_path)
    required = {"label", "seed", "checkpoint"}
    missing = required - set(spec.columns)
    if missing:
        raise ValueError(f"checkpoint spec 缺少列: {sorted(missing)}")
    if spec[["label", "seed"]].duplicated().any():
        raise ValueError("checkpoint spec 存在重复 label/seed")
    positive_ids = load_positive_row_ids(labels_path)
    labels_sha256 = _sha256_file(labels_path)
    output_dir = Path(output_dir)
    tasks: list[ExportTask] = []
    for index, row in spec.reset_index(drop=True).iterrows():
        label = str(row["label"])
        seed = int(row["seed"])
        checkpoint_path = Path(str(row["checkpoint"]))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
        config_path = _resolve_config_path(None, checkpoint_path)
        tag = f"{label}_{seed}"
        output_path = output_dir / f"{tag}_harmonic_predictions.npz"
        manifest_path = output_dir / f"{tag}_harmonic_predictions_manifest.json"
        if output_path.exists() or manifest_path.exists():
            raise FileExistsError(f"预测导出目标已存在，拒绝覆盖: {tag}")
        tasks.append(
            ExportTask(
                label=label,
                seed=seed,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                labels_path=Path(labels_path),
                labels_sha256=labels_sha256,
                output_path=output_path,
                manifest_path=manifest_path,
                device=str(devices[index % len(devices)]),
                n_selected_rows=len(positive_ids),
            )
        )
    if not tasks:
        raise ValueError("checkpoint spec 为空")
    if len({task.output_path for task in tasks}) != len(tasks):
        raise ValueError("checkpoint spec 产生了冲突的预测输出路径")
    return tasks


def save_prediction_payload(
    output_path: Path,
    *,
    dataset_row_id: np.ndarray,
    r_tho_hat: np.ndarray,
    tho_ref: np.ndarray,
    manifest: dict[str, Any],
    manifest_path: Path | None = None,
) -> Path:
    output_path = Path(output_path)
    resolved_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else output_path.with_name(f"{output_path.stem}_manifest.json")
    )
    if output_path.exists() or resolved_manifest.exists():
        raise FileExistsError(f"预测输出已存在，拒绝覆盖: {output_path}")
    ids = np.asarray(dataset_row_id, dtype=np.int64).reshape(-1)
    predictions = _as_prediction_matrix(r_tho_hat, name="r_tho_hat")
    targets = _as_prediction_matrix(tho_ref, name="tho_ref")
    if predictions.shape != targets.shape:
        raise ValueError(f"预测和目标 shape 不一致: {predictions.shape} != {targets.shape}")
    if ids.size != predictions.shape[0]:
        raise ValueError(f"dataset_row_id 数量与预测窗口数不一致: {ids.size} != {predictions.shape[0]}")
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("预测 payload 存在重复 dataset_row_id")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("预测 payload 包含非有限值")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    npz_tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    manifest_tmp = resolved_manifest.with_name(
        f".{resolved_manifest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    payload_manifest = {
        **manifest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "n_windows": int(ids.size),
        "array_schema": {
            "dataset_row_id": list(ids.shape),
            "r_tho_hat": list(predictions.shape),
            "tho_ref": list(targets.shape),
        },
    }
    try:
        np.savez(
            npz_tmp,
            dataset_row_id=ids,
            r_tho_hat=predictions.astype(np.float32, copy=False),
            tho_ref=targets.astype(np.float32, copy=False),
        )
        manifest_tmp.write_text(
            json.dumps(payload_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(npz_tmp, output_path)
        os.replace(manifest_tmp, resolved_manifest)
    finally:
        if npz_tmp.exists():
            npz_tmp.unlink()
        if manifest_tmp.exists():
            manifest_tmp.unlink()
    return resolved_manifest


def export_prediction_task(task: ExportTask) -> tuple[str, Path]:
    import torch
    from torch.utils.data import DataLoader

    selected_ids = load_positive_row_ids(task.labels_path)
    if len(selected_ids) != task.n_selected_rows:
        raise ValueError(f"任务选定 row 数发生变化: {len(selected_ids)} != {task.n_selected_rows}")
    cfg = load_config(task.config_path)
    dataset = _load_selected_dataset(cfg, selected_ids)
    device = torch.device(task.device)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(task.checkpoint_path, map_location=device)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(
        dataset,
        batch_size=min(128, max(1, int(cfg.training.batch_size))),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions = collect_predictions(
        model,
        loader,
        device=device,
        max_windows=len(dataset),
    )
    actual_ids = np.asarray(predictions["dataset_row_id"], dtype=np.int64).reshape(-1)
    if actual_ids.tolist() != selected_ids:
        raise ValueError(f"推理 row 顺序与固定阳性标签不一致: label={task.label} seed={task.seed}")
    save_prediction_payload(
        task.output_path,
        dataset_row_id=actual_ids,
        r_tho_hat=np.asarray(predictions["r_tho_hat"]),
        tho_ref=np.asarray(predictions["tho_ref"]),
        manifest={
            "label": task.label,
            "seed": task.seed,
            "checkpoint": str(task.checkpoint_path),
            "checkpoint_sha256": _sha256_file(task.checkpoint_path),
            "config": str(task.config_path),
            "labels_path": str(task.labels_path),
            "labels_sha256": task.labels_sha256,
            "split": "test",
            "device": task.device,
        },
        manifest_path=task.manifest_path,
    )
    return f"{task.label}_{task.seed}", task.output_path


def run_export_tasks(tasks: list[ExportTask], *, max_parallel: int) -> list[Path]:
    worker_count = min(max(1, int(max_parallel)), len(tasks))
    context = mp.get_context("spawn")
    outputs: list[Path] = []
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        futures = {pool.submit(export_prediction_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            tag, output = future.result()
            outputs.append(output)
            print(f"completed {tag} device={task.device} output={output}", flush=True)
    return sorted(outputs)


def _load_selected_dataset(cfg, row_ids: list[int]) -> ResearchV2WindowDataset:
    if str(cfg.data.get("format", "")) != "research_v2":
        raise ValueError("谐波预测导出只支持 research_v2 数据")
    rows = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    test_split = str(cfg.data.get("test_split", "test"))
    rows = rows[rows["split"].astype(str).eq(test_split)].copy()
    positions = {int(row_id): idx for idx, row_id in enumerate(row_ids)}
    selected = rows[rows["dataset_row_id"].astype(int).isin(positions)].copy()
    selected["_export_order"] = selected["dataset_row_id"].astype(int).map(positions)
    selected = selected.sort_values("_export_order").drop(columns="_export_order")
    actual_ids = selected["dataset_row_id"].astype(int).tolist()
    if actual_ids != row_ids:
        missing = sorted(set(row_ids) - set(actual_ids))
        raise ValueError(f"research_v2 test index 缺少阳性 row: {missing[:12]}")
    index_path = Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)
    dataset = ResearchV2WindowDataset(index_path, selected, cfg, preload_windows=False)
    if [int(value) for value in dataset.rows["dataset_row_id"].tolist()] != row_ids:
        raise ValueError("ResearchV2WindowDataset 过滤后 row 顺序或集合发生变化")
    return dataset


def _as_prediction_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} 必须是 [N, T] 或 [N, 1, T]，当前 shape={array.shape}")
    return array


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出二次谐波阳性测试窗口的模型预测")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", action="append", dest="devices", required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")
    tasks = build_export_tasks(
        spec_path=args.spec,
        labels_path=args.labels,
        output_dir=args.output_dir,
        devices=args.devices,
    )
    for task in tasks:
        print(
            f"task label={task.label} seed={task.seed} device={task.device} "
            f"rows={task.n_selected_rows} checkpoint={task.checkpoint_path} output={task.output_path}"
        )
    if args.dry_run:
        print(f"dry-run complete: tasks={len(tasks)} max_parallel={args.max_parallel}")
        return
    outputs = run_export_tasks(tasks, max_parallel=args.max_parallel)
    print(f"prediction export complete: {len(outputs)}/{len(tasks)}")


if __name__ == "__main__":
    main()
