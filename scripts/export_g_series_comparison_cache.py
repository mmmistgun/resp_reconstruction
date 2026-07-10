from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Iterable
import uuid

import numpy as np


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
