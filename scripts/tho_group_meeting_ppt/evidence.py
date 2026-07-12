from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


FORMAL_LABELS = (
    "g0_time_only",
    "g0_f0_native_stft_pre_mixer",
    "g3_c_wide_8p0",
    "g3_c_bandenergy",
)


@dataclass(frozen=True)
class RunConfigEvidence:
    path: Path
    patch_len: int
    patch_stride: int
    mixer_layers: int
    base_channels: int
    stft_win: int
    stft_hop: int
    stft_low_hz: float
    stft_high_hz: float
    stft_encoder_type: str
    stft_inject_position: str


@dataclass(frozen=True)
class EvidenceCatalog:
    dataset_root: Path
    dataset_index: Path
    general_signal_npz: Path
    general_sample_row_id: int
    run_configs: dict[str, RunConfigEvidence]
    result_root: Path
    harmonic_root: Path
    case_row_ids: tuple[int, int, int, int]


_MODEL_FIELDS: tuple[tuple[str, type], ...] = (
    ("patch_len", int),
    ("patch_stride", int),
    ("mixer_layers", int),
    ("base_channels", int),
    ("stft_win", int),
    ("stft_hop", int),
    ("stft_low_hz", float),
    ("stft_high_hz", float),
    ("stft_encoder_type", str),
    ("stft_inject_position", str),
)


def read_run_config(path: str | Path) -> RunConfigEvidence:
    """从正式 run 的 resolved config 中读取 PPT 所需的模型证据。"""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"run 配置不存在: {config_path}")

    cfg = OmegaConf.load(config_path)
    missing = [name for name, _ in _MODEL_FIELDS if OmegaConf.select(cfg, f"model.{name}") is None]
    if missing:
        fields = ", ".join(f"model.{name}" for name in missing)
        raise ValueError(f"缺少必要字段: {fields}; 配置路径: {config_path}")

    values = {
        name: cast(OmegaConf.select(cfg, f"model.{name}"))
        for name, cast in _MODEL_FIELDS
    }
    return RunConfigEvidence(path=config_path, **values)


def _find_formal_manifest(repo_root: Path) -> Path:
    candidates = sorted(
        path
        for path in (repo_root / "runs").glob("test_eval_g_series_*_canonical/g_series_test_eval_manifest.csv")
        if path.is_file()
    )
    if len(candidates) != 1:
        found = ", ".join(str(path) for path in candidates) or "无"
        raise FileNotFoundError(
            "无法唯一定位正式 G 系列 canonical manifest "
            f"(runs/test_eval_g_series_*_canonical/g_series_test_eval_manifest.csv); 找到: {found}"
        )
    return candidates[0].resolve()


def _read_manifest_configs(repo_root: Path, manifest: Path) -> dict[str, RunConfigEvidence]:
    with manifest.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows or not {"label", "checkpoint"} <= set(rows[0]):
        raise ValueError(f"G 系列 manifest 缺少 label/checkpoint 列: {manifest}")

    configs: dict[str, RunConfigEvidence] = {}
    for label in FORMAL_LABELS:
        label_rows = [row for row in rows if row.get("label") == label]
        if not label_rows:
            raise ValueError(f"G 系列 manifest 缺少正式模型 {label}: {manifest}")

        evidence_by_seed: list[RunConfigEvidence] = []
        for row in label_rows:
            checkpoint = Path(str(row["checkpoint"]))
            if not checkpoint.is_absolute():
                checkpoint = repo_root / checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(f"manifest 指向的 checkpoint 不存在: {checkpoint}; manifest: {manifest}")
            evidence_by_seed.append(read_run_config(checkpoint.parent / "config.yaml"))

        # 路径允许随 seed 不同，但同一对照的 PPT 结构参数必须一致。
        signatures = {evidence.__class__(path=Path("config.yaml"), **{
            field: getattr(evidence, field) for field, _ in _MODEL_FIELDS
        }) for evidence in evidence_by_seed}
        if len(signatures) != 1:
            paths = ", ".join(str(evidence.path) for evidence in evidence_by_seed)
            raise ValueError(f"正式对照 {label} 的多 seed 模型配置不一致: {paths}")
        configs[label] = evidence_by_seed[0]
    return configs


def _dataset_paths(configs: dict[str, RunConfigEvidence]) -> tuple[Path, Path]:
    paths: set[tuple[Path, Path]] = set()
    for evidence in configs.values():
        cfg = OmegaConf.load(evidence.path)
        dataset_root_value = OmegaConf.select(cfg, "data.dataset_root")
        index_value = OmegaConf.select(cfg, "data.index_csv")
        missing = [
            field
            for field, value in (("data.dataset_root", dataset_root_value), ("data.index_csv", index_value))
            if value is None
        ]
        if missing:
            raise ValueError(f"缺少必要字段: {', '.join(missing)}; 配置路径: {evidence.path}")
        root = Path(str(dataset_root_value)).resolve()
        index = (root / str(index_value)).resolve()
        paths.add((root, index))

    if len(paths) != 1:
        details = ", ".join(f"{root} -> {index}" for root, index in sorted(paths))
        raise ValueError(f"正式 G 系列配置指向不同数据集: {details}")
    dataset_root, dataset_index = paths.pop()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {dataset_root}")
    if not dataset_index.is_file():
        raise FileNotFoundError(f"数据集索引不存在: {dataset_index}")
    return dataset_root, dataset_index


def _find_general_signal(repo_root: Path) -> Path:
    candidates = sorted(
        path.resolve()
        for path in (repo_root / "docs" / "figure").glob("**/f0_visual_sample_signals.npz")
        if path.is_file()
    )
    if len(candidates) != 1:
        found = ", ".join(str(path) for path in candidates) or "无"
        raise FileNotFoundError(f"无法唯一定位 f0_visual_sample_signals.npz; 找到: {found}")
    return candidates[0]


def _find_harmonic_root(repo_root: Path) -> Path:
    candidates: list[Path] = []
    for case_manifest in (repo_root / "runs").glob("*/figures/model_case_manifest.csv"):
        root = case_manifest.parent.parent
        required = (
            root / "test_v2" / "test_harmonic_labels.csv",
            root / "model_metrics" / "model_stratified_metrics_summary.csv",
            root / "corrections" / "model_harmonic_correction_summary.csv",
        )
        if all(path.is_file() for path in required):
            candidates.append(root.resolve())
    if len(candidates) != 1:
        found = ", ".join(str(path) for path in candidates) or "无"
        raise FileNotFoundError(f"无法唯一定位正式 harmonic 结果目录; 找到: {found}")
    return candidates[0]


def build_evidence_catalog(repo_root: str | Path) -> EvidenceCatalog:
    root = Path(repo_root).resolve()
    manifest = _find_formal_manifest(root)
    run_configs = _read_manifest_configs(root, manifest)
    dataset_root, dataset_index = _dataset_paths(run_configs)
    return EvidenceCatalog(
        dataset_root=dataset_root,
        dataset_index=dataset_index,
        general_signal_npz=_find_general_signal(root),
        general_sample_row_id=8025,
        run_configs=run_configs,
        result_root=manifest.parent,
        harmonic_root=_find_harmonic_root(root),
        case_row_ids=(640, 873, 1353, 3584),
    )
