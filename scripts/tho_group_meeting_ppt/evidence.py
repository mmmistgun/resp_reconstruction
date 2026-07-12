from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


FORMAL_LABELS = (
    "g0_time_only",
    "g0_f0_native_stft_pre_mixer",
    "g3_c_wide_8p0",
    "g3_c_bandenergy",
)
FORMAL_SEEDS = (20260700, 20260837, 20260901)


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


@dataclass(frozen=True)
class _RunRecord:
    label: str
    seed: int
    config: RunConfigEvidence
    data_provenance: dict[str, Any]


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
_DATA_FIELDS = (
    "format",
    "input_set",
    "train_split",
    "val_split",
    "target_task",
    "bcg_input_key",
    "target_key",
)
_MANIFEST_FIELDS = {
    "label",
    "seed",
    "checkpoint",
    "metrics_output",
    "summary_output",
    "manifest_output",
}
_HARMONIC_FILES = (
    Path("figures/model_case_manifest.csv"),
    Path("test_v2/test_harmonic_labels.csv"),
    Path("test_v2/analysis_manifest.json"),
    Path("model_metrics/model_stratified_metrics_summary.csv"),
    Path("model_metrics/analysis_manifest.json"),
    Path("corrections/model_harmonic_correction_summary.csv"),
    Path("corrections/analysis_manifest.json"),
)
_EXPECTED_CASES = {
    640: "all_corrected",
    873: "all_not_corrected",
    1353: "model_disagreement",
    3584: "threshold_boundary",
}


def _invalid_model_field(path: Path, field: str, value: Any) -> ValueError:
    return ValueError(
        f"model.{field} 类型或取值无效: {value!r}; 配置路径: {path}"
    )


def _validated_model_value(path: Path, field: str, expected: type, value: Any) -> Any:
    if expected is int:
        if type(value) is not int:
            raise _invalid_model_field(path, field, value)
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _invalid_model_field(path, field, value)
        converted = float(value)
        if not math.isfinite(converted):
            raise _invalid_model_field(path, field, value)
        return converted
    if expected is str:
        if type(value) is not str or not value.strip():
            raise _invalid_model_field(path, field, value)
        return value.strip()
    raise AssertionError(f"未处理的模型字段类型: {expected}")


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
        name: _validated_model_value(
            config_path,
            name,
            expected,
            OmegaConf.select(cfg, f"model.{name}"),
        )
        for name, expected in _MODEL_FIELDS
    }
    return RunConfigEvidence(path=config_path, **values)


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _require_file(repo_root: Path, path: str | Path, description: str) -> Path:
    resolved = _resolve_path(repo_root, path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description}不存在: {resolved}")
    return resolved


def _require_harmonic_root(repo_root: Path, path: str | Path) -> Path:
    root = _resolve_path(repo_root, path)
    if not root.is_dir():
        raise FileNotFoundError(f"harmonic 结果目录不存在: {root}")
    missing = [str(relative) for relative in _HARMONIC_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"harmonic 结果结构不完整，缺少 {', '.join(missing)}; 目录: {root}"
        )
    return root


def _select_unique_candidate(
    description: str,
    candidates: list[Path],
    explicit_parameter: str,
) -> Path:
    unique = sorted({candidate.resolve() for candidate in candidates})
    if not unique:
        raise FileNotFoundError(
            f"未找到结构完整的 {description} 候选；"
            f"请使用 {explicit_parameter} 显式指定"
        )
    if len(unique) > 1:
        listed = ", ".join(str(candidate) for candidate in unique)
        raise ValueError(
            f"{description} 候选不唯一: {listed}；"
            f"请使用 {explicit_parameter} 显式指定以消歧"
        )
    return unique[0]


def _discover_manifest(repo_root: Path) -> Path:
    candidates: list[Path] = []
    pattern = "test_eval_g_series_*_canonical/g_series_test_eval_manifest.csv"
    for path in (repo_root / "runs").glob(pattern):
        if not path.is_file():
            continue
        try:
            records = _read_manifest_records(repo_root, path)
            _validate_records(records)
            dataset_root = records[0].data_provenance["data.dataset_root"]
            dataset_index = records[0].data_provenance["data.index_csv"]
            if not dataset_root.is_dir() or not dataset_index.is_file():
                continue
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        candidates.append(path)
    return _select_unique_candidate("正式 G 系列 manifest", candidates, "manifest_path")


def _discover_harmonic_root(repo_root: Path) -> Path:
    candidates = [
        path
        for path in (repo_root / "runs").glob("*")
        if path.is_dir() and all((path / relative).is_file() for relative in _HARMONIC_FILES)
    ]
    return _select_unique_candidate("harmonic 结果目录", candidates, "harmonic_root")


def _signal_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_name("f0_visual_sample_metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"通用信号 metadata 不存在: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"无法读取通用信号 metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict) or metadata.get("dataset_row_id") != 8025:
        raise ValueError(f"通用信号 metadata 必须标记 dataset_row_id=8025: {metadata_path}")
    files = metadata.get("files")
    if not isinstance(files, dict) or files.get("signal_arrays_npz") != path.name:
        raise ValueError(f"通用信号 metadata 未正确指向 {path.name}: {metadata_path}")
    return metadata


def _require_general_signal(repo_root: Path, path: str | Path) -> Path:
    signal = _require_file(repo_root, path, "通用信号 NPZ")
    _signal_metadata(signal)
    return signal


def _discover_general_signal(repo_root: Path) -> Path:
    candidates: list[Path] = []
    for path in (repo_root / "docs" / "figure").glob("**/f0_visual_sample_signals.npz"):
        if not path.is_file():
            continue
        try:
            _signal_metadata(path)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append(path)
    return _select_unique_candidate("通用信号 NPZ", candidates, "general_signal_npz")


def _read_data_provenance(config_path: Path, repo_root: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    required = ("dataset_root", "index_csv", *_DATA_FIELDS)
    missing = [field for field in required if OmegaConf.select(cfg, f"data.{field}") is None]
    if missing:
        fields = ", ".join(f"data.{field}" for field in missing)
        raise ValueError(f"缺少必要字段: {fields}; 配置路径: {config_path}")

    raw_root = OmegaConf.select(cfg, "data.dataset_root")
    raw_index = OmegaConf.select(cfg, "data.index_csv")
    if type(raw_root) is not str or not raw_root.strip():
        raise ValueError(f"data.dataset_root 必须是非空字符串; 配置路径: {config_path}")
    if type(raw_index) is not str or not raw_index.strip():
        raise ValueError(f"data.index_csv 必须是非空字符串; 配置路径: {config_path}")

    dataset_root = _resolve_path(repo_root, raw_root)
    index_candidate = Path(raw_index)
    dataset_index = (
        index_candidate.resolve()
        if index_candidate.is_absolute()
        else (dataset_root / index_candidate).resolve()
    )
    provenance: dict[str, Any] = {
        "data.dataset_root": dataset_root,
        "data.index_csv": dataset_index,
    }
    for field in _DATA_FIELDS:
        value = OmegaConf.select(cfg, f"data.{field}")
        if type(value) is not str or not value.strip():
            raise ValueError(
                f"data.{field} 必须是非空字符串; 配置路径: {config_path}"
            )
        provenance[f"data.{field}"] = value.strip()
    return provenance


def _manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fields = set(reader.fieldnames or ())
        missing = sorted(_MANIFEST_FIELDS - fields)
        if missing:
            raise ValueError(f"G 系列 manifest 缺少列 {missing}: {manifest}")
        rows = list(reader)
    return rows


def _validated_manifest_keys(rows: list[dict[str, str]], manifest: Path) -> list[tuple[str, int, dict[str, str]]]:
    parsed: list[tuple[str, int, dict[str, str]]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        label = str(row["label"]).strip()
        try:
            seed = int(str(row["seed"]).strip())
        except ValueError as exc:
            raise ValueError(f"manifest seed 不是整数: {row['seed']!r}; 路径: {manifest}") from exc
        key = (label, seed)
        if key in seen:
            raise ValueError(f"manifest 存在重复 label/seed: {label}/{seed}; 路径: {manifest}")
        seen.add(key)
        parsed.append((label, seed, row))

    if len(rows) != len(FORMAL_LABELS) * len(FORMAL_SEEDS):
        raise ValueError(
            f"G 系列 manifest 必须包含 12 行（4 labels × 3 seeds），"
            f"实际 {len(rows)} 行: {manifest}"
        )
    expected = {(label, seed) for label in FORMAL_LABELS for seed in FORMAL_SEEDS}
    actual = {(label, seed) for label, seed, _ in parsed}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"manifest label/seed 集不完整，缺少 {missing}，额外 {extra}; 路径: {manifest}")
    return parsed


def _require_manifest_output(repo_root: Path, row: dict[str, str], field: str, manifest: Path) -> None:
    value = str(row[field]).strip()
    if not value:
        raise ValueError(f"manifest {field} 为空: {manifest}")
    path = _resolve_path(repo_root, value)
    if not path.is_file():
        raise FileNotFoundError(f"manifest {field} 指向的结果不存在: {path}; manifest: {manifest}")


def _read_manifest_records(repo_root: Path, manifest: Path) -> list[_RunRecord]:
    rows = _manifest_rows(manifest)
    keyed_rows = _validated_manifest_keys(rows, manifest)
    records: list[_RunRecord] = []
    for label, seed, row in keyed_rows:
        checkpoint = _require_file(repo_root, row["checkpoint"], "manifest 指向的 checkpoint")
        for output_field in ("metrics_output", "summary_output", "manifest_output"):
            _require_manifest_output(repo_root, row, output_field, manifest)
        config = read_run_config(checkpoint.parent / "config.yaml")
        records.append(
            _RunRecord(
                label=label,
                seed=seed,
                config=config,
                data_provenance=_read_data_provenance(config.path, repo_root),
            )
        )
    return records


def _validate_records(records: list[_RunRecord]) -> None:
    baseline = records[0]
    for record in records[1:]:
        for field, expected in baseline.data_provenance.items():
            actual = record.data_provenance[field]
            if actual != expected:
                raise ValueError(
                    f"{record.label}/{record.seed} 的 {field} 与正式对照不一致: "
                    f"{actual!r} != {expected!r}; 配置路径: {record.config.path}"
                )

    for label in FORMAL_LABELS:
        label_records = [record for record in records if record.label == label]
        reference = label_records[0]
        for record in label_records[1:]:
            for field, _ in _MODEL_FIELDS:
                if getattr(record.config, field) != getattr(reference.config, field):
                    raise ValueError(
                        f"{label}/{record.seed} 的 model.{field} 与同组 seed 不一致; "
                        f"配置路径: {record.config.path}"
                    )


def _catalog_run_configs(records: list[_RunRecord]) -> dict[str, RunConfigEvidence]:
    return {
        label: next(
            record.config
            for record in records
            if record.label == label and record.seed == FORMAL_SEEDS[0]
        )
        for label in FORMAL_LABELS
    }


def _load_json_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"{path}: 无法解析 JSON manifest") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest 根节点必须是 object，实际 {type(payload).__name__}")
    return payload


def _provenance_error(path: Path, field: str, actual: Any, expected: Any) -> ValueError:
    return ValueError(f"{path}: {field} 实际 {actual!r}，期望 {expected!r}")


def _required_field(path: Path, payload: dict[str, Any], field: str) -> Any:
    current: Any = payload
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            raise _provenance_error(path, field, "<missing>", "必需字段")
        current = current[part]
    return current


def _resolved_manifest_path(repo_root: Path, path: Path, field: str, value: Any) -> Path:
    if type(value) is not str or not value.strip():
        raise _provenance_error(path, field, value, "非空路径字符串")
    return _resolve_path(repo_root, value)


def _require_equal(path: Path, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise _provenance_error(path, field, actual, expected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_harmonic_manifests(
    repo_root: Path,
    harmonic_root: Path,
    result_root: Path,
    dataset_root: Path,
    dataset_index: Path,
) -> None:
    model_path = harmonic_root / "model_metrics" / "analysis_manifest.json"
    test_path = harmonic_root / "test_v2" / "analysis_manifest.json"
    corrections_path = harmonic_root / "corrections" / "analysis_manifest.json"
    model = _load_json_manifest(model_path)
    test = _load_json_manifest(test_path)
    corrections = _load_json_manifest(corrections_path)

    _require_equal(
        model_path,
        "eval_root",
        _resolved_manifest_path(repo_root, model_path, "eval_root", _required_field(model_path, model, "eval_root")),
        result_root.resolve(),
    )
    _require_equal(
        model_path,
        "dataset_index",
        _resolved_manifest_path(
            repo_root,
            model_path,
            "dataset_index",
            _required_field(model_path, model, "dataset_index"),
        ),
        dataset_index.resolve(),
    )
    model_labels = _required_field(model_path, model, "model_labels")
    expected_labels = set(FORMAL_LABELS)
    if not isinstance(model_labels, list) or len(model_labels) != len(expected_labels) or set(model_labels) != expected_labels:
        raise _provenance_error(model_path, "model_labels", model_labels, sorted(expected_labels))
    model_seeds = _required_field(model_path, model, "seeds")
    expected_seeds = set(FORMAL_SEEDS)
    if not isinstance(model_seeds, list) or len(model_seeds) != len(expected_seeds) or set(model_seeds) != expected_seeds:
        raise _provenance_error(model_path, "seeds", model_seeds, sorted(expected_seeds))

    test_dataset_root = _resolved_manifest_path(
        repo_root,
        test_path,
        "dataset_root",
        _required_field(test_path, test, "dataset_root"),
    )
    _require_equal(test_path, "dataset_root", test_dataset_root, dataset_root.resolve())
    raw_index = _required_field(test_path, test, "index_csv")
    if type(raw_index) is not str or not raw_index.strip():
        raise _provenance_error(test_path, "index_csv", raw_index, "非空路径字符串")
    index_candidate = Path(raw_index)
    test_dataset_index = (
        index_candidate.resolve()
        if index_candidate.is_absolute()
        else (test_dataset_root / index_candidate).resolve()
    )
    _require_equal(test_path, "index_csv", test_dataset_index, dataset_index.resolve())
    _require_equal(test_path, "split", _required_field(test_path, test, "split"), "test")

    expected_labels_path = (harmonic_root / "test_v2" / "test_harmonic_labels.csv").resolve()
    for path, payload, field in (
        (model_path, model, "labels_path"),
        (test_path, test, "outputs.labels"),
        (corrections_path, corrections, "labels_path"),
    ):
        _require_equal(
            path,
            field,
            _resolved_manifest_path(repo_root, path, field, _required_field(path, payload, field)),
            expected_labels_path,
        )

    # model_metrics 与 corrections manifest 共享 labels_sha256，这是现有的标签文件 hash 链。
    model_labels_hash = _required_field(model_path, model, "labels_sha256")
    corrections_labels_hash = _required_field(corrections_path, corrections, "labels_sha256")
    if type(model_labels_hash) is not str or not model_labels_hash.strip():
        raise _provenance_error(model_path, "labels_sha256", model_labels_hash, "非空 SHA-256 字符串")
    _require_equal(
        corrections_path,
        "labels_sha256",
        corrections_labels_hash,
        model_labels_hash,
    )
    actual_labels_hash = _sha256_file(expected_labels_path)
    _require_equal(
        expected_labels_path,
        "labels_sha256",
        actual_labels_hash,
        model_labels_hash,
    )

    thresholds_value = _required_field(test_path, test, "thresholds_path")
    thresholds_path = _resolved_manifest_path(
        repo_root,
        test_path,
        "thresholds_path",
        thresholds_value,
    )
    if not thresholds_path.is_relative_to(harmonic_root.resolve()):
        raise _provenance_error(
            test_path,
            "thresholds_path",
            thresholds_path,
            f"位于 {harmonic_root.resolve()} 内",
        )
    if not thresholds_path.is_file():
        raise _provenance_error(test_path, "thresholds_path", thresholds_path, "存在的文件")
    expected_thresholds_hash = _required_field(test_path, test, "thresholds_sha256")
    if type(expected_thresholds_hash) is not str or not expected_thresholds_hash.strip():
        raise _provenance_error(
            test_path,
            "thresholds_sha256",
            expected_thresholds_hash,
            "非空 SHA-256 字符串",
        )
    actual_thresholds_hash = _sha256_file(thresholds_path)
    _require_equal(
        thresholds_path,
        "thresholds_sha256",
        actual_thresholds_hash,
        expected_thresholds_hash,
    )
    test_threshold_version = _required_field(test_path, test, "threshold_version")
    corrections_threshold_version = _required_field(
        corrections_path,
        corrections,
        "threshold_version",
    )
    _require_equal(
        corrections_path,
        "threshold_version",
        corrections_threshold_version,
        test_threshold_version,
    )

    # corrections.runs 是实际 manifest 记录模型/seed 组合的位置。
    runs = _required_field(corrections_path, corrections, "runs")
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        raise _provenance_error(corrections_path, "runs", runs, "label/seed object 列表")
    actual_pairs = [(run.get("label"), run.get("seed")) for run in runs]
    expected_pairs = {(label, seed) for label in FORMAL_LABELS for seed in FORMAL_SEEDS}
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise _provenance_error(corrections_path, "runs[label,seed]", actual_pairs, sorted(expected_pairs))


def _validate_case_manifest(harmonic_root: Path) -> None:
    path = harmonic_root / "figures" / "model_case_manifest.csv"
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fields = set(reader.fieldnames or ())
        missing = {"dataset_row_id", "case_category"} - fields
        if missing:
            raise _provenance_error(path, "columns", sorted(fields), sorted(missing))
        rows = list(reader)

    for row_id, expected_category in _EXPECTED_CASES.items():
        matches = [row for row in rows if str(row.get("dataset_row_id", "")).strip() == str(row_id)]
        if len(matches) != 1:
            raise _provenance_error(path, f"dataset_row_id={row_id}", len(matches), "恰好 1 行")
        actual_category = str(matches[0].get("case_category", "")).strip()
        if actual_category != expected_category:
            raise _provenance_error(
                path,
                f"dataset_row_id={row_id}.case_category",
                actual_category,
                expected_category,
            )


def _validate_harmonic_provenance(
    repo_root: Path,
    harmonic_root: Path,
    result_root: Path,
    dataset_root: Path,
    dataset_index: Path,
) -> None:
    _validate_harmonic_manifests(
        repo_root,
        harmonic_root,
        result_root,
        dataset_root,
        dataset_index,
    )
    _validate_case_manifest(harmonic_root)


def build_evidence_catalog(
    repo_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    harmonic_root: str | Path | None = None,
    general_signal_npz: str | Path | None = None,
) -> EvidenceCatalog:
    root = Path(repo_root).resolve()
    manifest = (
        _discover_manifest(root)
        if manifest_path is None
        else _require_file(root, manifest_path, "正式 G 系列 manifest")
    )
    signal = (
        _discover_general_signal(root)
        if general_signal_npz is None
        else _require_general_signal(root, general_signal_npz)
    )
    resolved_harmonic_root = (
        _discover_harmonic_root(root)
        if harmonic_root is None
        else _require_harmonic_root(root, harmonic_root)
    )
    records = _read_manifest_records(root, manifest)
    _validate_records(records)

    dataset_root = records[0].data_provenance["data.dataset_root"]
    dataset_index = records[0].data_provenance["data.index_csv"]
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {dataset_root}")
    if not dataset_index.is_file():
        raise FileNotFoundError(f"数据集索引不存在: {dataset_index}")
    _validate_harmonic_provenance(
        root,
        resolved_harmonic_root,
        manifest.parent,
        dataset_root,
        dataset_index,
    )
    return EvidenceCatalog(
        dataset_root=dataset_root,
        dataset_index=dataset_index,
        general_signal_npz=signal,
        general_sample_row_id=8025,
        run_configs=_catalog_run_configs(records),
        result_root=manifest.parent,
        harmonic_root=resolved_harmonic_root,
        case_row_ids=tuple(_EXPECTED_CASES),
    )
