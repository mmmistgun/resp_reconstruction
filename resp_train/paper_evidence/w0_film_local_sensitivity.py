"""冻结 W0 的 FiLM gamma/beta 局部系数敏感性实验。"""

from __future__ import annotations

import hashlib
import json
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
from torch import nn

from resp_train.crd.model import _validate_input
from resp_train.crd.tf_v1_model import _checkpointed_mapping_branch
from resp_train.crd.tf_w_v2_audit import _w0_seed_entries
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.paper_evidence.e1_scale_topology import (
    ERRORS,
    PCC,
    PRIMARY,
    validate_metrics,
)
from resp_train.paper_evidence.w0_cwt_film_behavior import (
    QUALITY_COLUMNS,
    QUALITY_GROUP_COLUMN,
    SEEDS,
    WINDOW_COUNT,
    forward_with_capture,
    validate_w0_contract,
)
from resp_train.paper_evidence.w0_cwt_film_behavior_runtime import (
    CODE_ROOT,
    SOURCE_ROOT,
    _build_validation_data,
    _checked_batches,
    _expected_metric_sources,
    _load_model,
    environment,
    git_state,
    identity,
    require_gpu,
    sha256_file,
    verify_file,
    write_json,
)


PROTOCOL = "w0-film-local-sensitivity-v1-20260918"
CONDITIONS = ("FULL", "GAMMA_040", "GAMMA_060", "BETA_040", "BETA_060")
NEW_CONDITIONS = CONDITIONS[1:]
COEFFICIENTS = {
    "FULL": (0.5, 0.5),
    "GAMMA_040": (0.4, 0.5),
    "GAMMA_060": (0.6, 0.5),
    "BETA_040": (0.5, 0.4),
    "BETA_060": (0.5, 0.6),
}
AXES = (*ERRORS, "one_minus_lag_aware_signed_pcc")
FULL_ANCHOR_ATOL = 1e-12

DOCS = Path("docs/experiments")
SOURCE_LOCK = DOCS / "crd_tf_w_v2_candidate_lock_20260817.json"
SOURCE_LOCK_SHA256 = "6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6"
METRIC_LOCK = DOCS / "e3_w0_metric_association_lock_20260916.json"
METRIC_LOCK_SHA256 = "f9fadfc620ab07b6d58bb7508ed441f923def01dc1a948a029190fbdc9e094bc"
PRIOR_RESULTS = DOCS / "w0_cwt_film_behavior_results_20260918.md"
PRIOR_SUMMARY = SOURCE_ROOT / (
    "runs/w0_cwt_film_behavior_v1/summary/"
    "summary_c4bfc1907266_20260918T055139Z_773b7065a629"
)
PRIOR_SUMMARY_MANIFEST_SHA256 = (
    "d6dee7a97d2fd1a5b7892ef9c8b4f7dca53739a43361d267af8b2bf6ca0ceff1"
)
PREVIOUS_IMPLEMENTATION_LOCK = DOCS / "w0_film_local_sensitivity_implementation_lock_20260918.json"
IMPLEMENTATION_LOCK = DOCS / "w0_film_local_sensitivity_implementation_lock_r2_20260918.json"
PROTOCOL_PATH = DOCS / "w0_film_local_sensitivity_protocol_20260918.md"
SCRIPT_PATH = Path("scripts/analyze_w0_film_local_sensitivity.py")
TEST_PATH = Path("tests/test_w0_film_local_sensitivity.py")
OUTPUT_ROOT = SOURCE_ROOT / "runs/w0_film_local_sensitivity_v1"


class LocalSensitivityModel(nn.Module):
    """复用原生模块，只替换 FiLM 融合中的两个标量系数。"""

    def __init__(
        self,
        model: nn.Module,
        condition: str,
        *,
        force_explicit_full: bool = False,
    ) -> None:
        super().__init__()
        validate_w0_contract(model)
        if condition not in CONDITIONS:
            raise ValueError(f"未知 FiLM 局部条件: {condition}")
        self.model = model
        self.condition = condition
        self.force_explicit_full = bool(force_explicit_full)

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> Mapping[str, torch.Tensor]:
        if tf is None or set(tf) != {"w"}:
            raise ValueError("FiLM 局部敏感性只接受 W feature")
        if self.condition == "FULL" and not self.force_explicit_full:
            return self.model(x, tf=tf)
        gamma_coefficient, beta_coefficient = COEFFICIENTS[self.condition]
        _validate_input(x)
        latent = self.model.base.encode_local(x)
        gamma_raw, beta_raw = _checkpointed_mapping_branch(
            self.model.branches["w"], {"w": tf["w"]}
        )
        conditioned = _fuse_w0(
            latent,
            gamma_raw,
            beta_raw,
            gamma_coefficient=gamma_coefficient,
            beta_coefficient=beta_coefficient,
        )
        return self.model.base.decode_local(conditioned)


def _fuse_w0(
    latent: torch.Tensor,
    gamma_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    *,
    gamma_coefficient: float,
    beta_coefficient: float,
) -> torch.Tensor:
    """逐算子复刻原生 W0 融合，仅允许替换两个系数。"""

    gamma_sum = torch.zeros_like(latent)
    beta_sum = torch.zeros_like(latent)
    gamma_sum = gamma_sum + gamma_coefficient * torch.tanh(gamma_raw)
    beta_sum = beta_sum + beta_coefficient * torch.tanh(beta_raw)
    return latent * (1.0 + gamma_sum) + beta_sum


def error_aligned_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*ERRORS, PCC}
    if missing := sorted(required - set(frame)):
        raise KeyError(f"FiLM 局部 metrics 缺少列: {missing}")
    output = frame.copy()
    output["one_minus_lag_aware_signed_pcc"] = 1.0 - output[PCC].to_numpy(dtype=np.float64)
    values = output[list(AXES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("FiLM 局部 error-aligned metrics 非有限")
    return output


def attach_metadata(metrics: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    if len(metrics) != len(rows):
        raise ValueError("FiLM 局部 metrics/rows 行数不一致")
    for key in ("dataset_row_id", "samp_id", "split"):
        if not np.array_equal(metrics[key].to_numpy(), rows[key].to_numpy()):
            raise ValueError(f"FiLM 局部 metrics/rows identity 不一致: {key}")
    required = {
        QUALITY_GROUP_COLUMN,
        *QUALITY_COLUMNS,
        "coupling_state_id",
        "window_start_s",
        "window_end_s",
    }
    if missing := sorted(required - set(rows)):
        raise KeyError(f"FiLM 局部 metadata 缺少列: {missing}")
    output = metrics.copy()
    for column in required:
        output[column] = rows[column].to_numpy()
    return output


def validate_condition_matrix(frame: pd.DataFrame, *, seeds: Sequence[int] = SEEDS) -> None:
    required = {"seed", "condition", "dataset_row_id", "samp_id", "split", *AXES}
    if missing := sorted(required - set(frame)):
        raise KeyError(f"FiLM 局部矩阵缺少列: {missing}")
    expected = {
        (int(seed), condition)
        for seed in seeds
        for condition in CONDITIONS
    }
    observed = set(frame[["seed", "condition"]].drop_duplicates().itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("FiLM 局部 condition×seed 矩阵不完整")
    if frame[["seed", "condition", "dataset_row_id"]].duplicated().any():
        raise ValueError("FiLM 局部 condition×seed×row identity 重复")
    counts = frame.groupby(["seed", "condition"]).size()
    if not (counts == WINDOW_COUNT).all():
        raise ValueError("FiLM 局部每 condition/seed 必须为完整 validation")
    row_sets = {
        (int(seed), condition): tuple(
            group["dataset_row_id"].sort_values().to_numpy(dtype=np.int64)
        )
        for (seed, condition), group in frame.groupby(["seed", "condition"], sort=True)
    }
    anchor = row_sets[(int(seeds[0]), "FULL")]
    if any(not np.array_equal(anchor, value) for value in row_sets.values()):
        raise ValueError("FiLM 局部 condition/seed row identity 不一致")
    if set(frame["split"].astype(str)) != {"val"}:
        raise ValueError("FiLM 局部矩阵只允许 validation")
    if not np.isfinite(frame[list(AXES)].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("FiLM 局部矩阵主轴非有限")


def build_paired_deltas(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    validate_condition_matrix(condition_metrics, seeds=tuple(sorted(condition_metrics["seed"].unique())))
    records: list[pd.DataFrame] = []
    identity = ["seed", "dataset_row_id", "samp_id", "split"]
    metadata = [
        "envelope_target_stratum",
        QUALITY_GROUP_COLUMN,
        *QUALITY_COLUMNS,
        "coupling_state_id",
        "window_start_s",
        "window_end_s",
    ]
    eligibility = [
        "whole_rr_target_eligible",
        "local_rr_target_eligible",
        "local_rr_target_eligible_windows",
        "joint_target_eligible",
        "envelope_spearman_target_eligible",
    ]
    for seed, seed_frame in condition_metrics.groupby("seed", sort=True):
        full = seed_frame.loc[seed_frame["condition"].eq("FULL")].sort_values(
            "dataset_row_id"
        )
        for condition in NEW_CONDITIONS:
            candidate = seed_frame.loc[seed_frame["condition"].eq(condition)].sort_values(
                "dataset_row_id"
            )
            for key in ("dataset_row_id", "samp_id", "split", *eligibility):
                if not np.array_equal(candidate[key].to_numpy(), full[key].to_numpy()):
                    raise ValueError(f"FiLM 局部干预改变 identity/eligibility: {seed}/{condition}/{key}")
            output = full[identity + metadata].copy()
            output.insert(1, "condition", condition)
            for axis in AXES:
                full_values = full[axis].to_numpy(dtype=np.float64)
                candidate_values = candidate[axis].to_numpy(dtype=np.float64)
                output[f"{axis}__full"] = full_values
                output[f"{axis}__candidate"] = candidate_values
                output[f"{axis}__delta"] = candidate_values - full_values
                if axis in ERRORS:
                    relative = np.full(len(full_values), np.nan, dtype=np.float64)
                    nonzero = full_values != 0.0
                    relative[nonzero] = (
                        100.0
                        * (candidate_values[nonzero] - full_values[nonzero])
                        / full_values[nonzero]
                    )
                    output[f"{axis}__relative_delta_percent"] = relative
            records.append(output)
    paired = pd.concat(records, ignore_index=True)
    if len(paired) != len(set(condition_metrics["seed"])) * len(NEW_CONDITIONS) * WINDOW_COUNT:
        raise ValueError("FiLM 局部 paired delta 行数错误")
    delta_columns = [column for column in paired if column.endswith("__delta")]
    if not np.isfinite(paired[delta_columns].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("FiLM 局部 paired delta 非有限")
    return paired


def _scope_groups(frame: pd.DataFrame) -> Iterator[tuple[str, str, pd.DataFrame]]:
    yield "all", "all", frame
    for value, group in frame.groupby("envelope_target_stratum", sort=True):
        yield "target_stratum", str(value), group
    for value, group in frame.groupby(QUALITY_GROUP_COLUMN, sort=True):
        yield "quality_level", str(value), group
    for (target, quality), group in frame.groupby(
        ["envelope_target_stratum", QUALITY_GROUP_COLUMN], sort=True
    ):
        yield "target_quality", f"{target}__{quality}", group
    for value, group in frame.groupby("samp_id", sort=True):
        yield "samp_id", str(int(value)), group


def summarize_paired_deltas(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (seed, condition), condition_frame in paired.groupby(["seed", "condition"], sort=True):
        for scope, level, group in _scope_groups(condition_frame):
            for axis in AXES:
                full = group[f"{axis}__full"].to_numpy(dtype=np.float64)
                candidate = group[f"{axis}__candidate"].to_numpy(dtype=np.float64)
                delta = group[f"{axis}__delta"].to_numpy(dtype=np.float64)
                record: dict[str, Any] = {
                    "seed": int(seed),
                    "condition": condition,
                    "scope": scope,
                    "level": level,
                    "metric": axis,
                    "windows": int(len(group)),
                    "samp_ids": int(group["samp_id"].nunique()),
                    "full_mean": float(full.mean()),
                    "candidate_mean": float(candidate.mean()),
                    "delta_mean": float(delta.mean()),
                    "delta_median": float(np.median(delta)),
                    "delta_p05": float(np.quantile(delta, 0.05, method="linear")),
                    "delta_p25": float(np.quantile(delta, 0.25, method="linear")),
                    "delta_p75": float(np.quantile(delta, 0.75, method="linear")),
                    "delta_p95": float(np.quantile(delta, 0.95, method="linear")),
                    "worse_count": int((delta > 0).sum()),
                    "equal_count": int((delta == 0).sum()),
                    "better_count": int((delta < 0).sum()),
                    "worse_fraction": float((delta > 0).mean()),
                }
                if axis in ERRORS:
                    relative = group[f"{axis}__relative_delta_percent"].dropna().to_numpy(
                        dtype=np.float64
                    )
                    record["relative_delta_percent_mean"] = (
                        float(relative.mean()) if len(relative) else None
                    )
                    record["relative_defined_windows"] = int(len(relative))
                else:
                    record["relative_delta_percent_mean"] = None
                    record["relative_defined_windows"] = 0
                records.append(record)
    return pd.DataFrame.from_records(records)


def local_response(stratum_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    pairs = {
        "gamma": ("GAMMA_040", "GAMMA_060"),
        "beta": ("BETA_040", "BETA_060"),
    }
    group_keys = ["seed", "scope", "level", "metric"]
    for keys, group in stratum_summary.groupby(group_keys, sort=True):
        indexed = group.set_index("condition")
        for path, (down_name, up_name) in pairs.items():
            if down_name not in indexed.index or up_name not in indexed.index:
                raise ValueError(f"FiLM 局部响应缺少 {path} down/up")
            down = indexed.loc[down_name]
            up = indexed.loc[up_name]
            if int(down["windows"]) != int(up["windows"]):
                raise ValueError("FiLM 局部响应 down/up 分母不一致")
            full_values = np.asarray([down["full_mean"], up["full_mean"]], dtype=np.float64)
            if not np.allclose(full_values, full_values[0], rtol=0.0, atol=FULL_ANCHOR_ATOL):
                raise RuntimeError("FiLM 局部响应 FULL anchor 漂移")
            full = float(full_values[0])
            down_value = float(down["candidate_mean"])
            up_value = float(up["candidate_mean"])
            record = dict(zip(group_keys, keys, strict=True))
            record.update(
                {
                    "path": path,
                    "windows": int(down["windows"]),
                    "samp_ids": int(down["samp_ids"]),
                    "full_coefficient": 0.5,
                    "down_coefficient": 0.4,
                    "up_coefficient": 0.6,
                    "full_mean": full,
                    "down_mean": down_value,
                    "up_mean": up_value,
                    "down_delta": down_value - full,
                    "up_delta": up_value - full,
                    "central_slope": (up_value - down_value) / 0.2,
                    "local_curvature": up_value - 2.0 * full + down_value,
                }
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def seed_summary(stratum_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "scope", "level", "metric"]
    records = []
    for values, group in stratum_summary.groupby(keys, sort=True):
        delta = group["delta_mean"].to_numpy(dtype=np.float64)
        relative = group["relative_delta_percent_mean"].dropna().to_numpy(dtype=np.float64)
        record = dict(zip(keys, values, strict=True))
        record.update(
            {
                "delta_seed_mean": float(delta.mean()),
                "delta_seed_sd": float(delta.std(ddof=1)),
                "worse_seed_count": int((delta > 0).sum()),
                "equal_seed_count": int((delta == 0).sum()),
                "better_seed_count": int((delta < 0).sum()),
                "relative_delta_percent_seed_mean": (
                    float(relative.mean()) if len(relative) else None
                ),
                "relative_delta_percent_seed_sd": (
                    float(relative.std(ddof=1)) if len(relative) >= 2 else None
                ),
                "defined_seed_count": int(len(delta)),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _check_full_source(frame: pd.DataFrame, reference: pd.DataFrame, *, seed: int) -> None:
    left = frame.sort_values("dataset_row_id").reset_index(drop=True)
    right = reference.sort_values("dataset_row_id").reset_index(drop=True)
    if len(left) != WINDOW_COUNT or len(right) != WINDOW_COUNT:
        raise ValueError("FiLM 局部 FULL 来源行数错误")
    for column in ("dataset_row_id", "samp_id", "split", *PRIMARY):
        if column in PRIMARY:
            delta = np.abs(
                left[column].to_numpy(dtype=np.float64)
                - right[column].to_numpy(dtype=np.float64)
            )
            if float(delta.max(initial=0.0)) > FULL_ANCHOR_ATOL:
                raise RuntimeError(f"FiLM 局部 FULL 数值来源漂移: {seed}/{column}")
        elif not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise RuntimeError(f"FiLM 局部 FULL identity 来源漂移: {seed}/{column}")


def _source_files_for_lock(
    *,
    code_root: Path,
    source_root: Path,
    source_lock: Mapping[str, Any],
    metric_lock: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    entries = _w0_seed_entries(source_lock)
    metric_sources = _expected_metric_sources(metric_lock)
    files: dict[str, dict[str, Any]] = {}
    for entry in entries:
        seed = int(entry["seed"])
        for key, filename in (
            ("checkpoint", "checkpoint_best_local_rr.pt"),
            ("config", "config.yaml"),
            ("manifest", "run_manifest.json"),
            ("validation_summary", "metrics_summary.csv"),
        ):
            path = source_root / str(entry["run_dir"]) / filename
            verify_file(path, entry[key])
            files[str(path.resolve())] = dict(entry[key])
        metric_path = Path(metric_sources[seed]["path"])
        verify_file(metric_path, metric_sources[seed]["identity"])
        files[str(metric_path)] = dict(metric_sources[seed]["identity"])
    cache = source_lock["cache_lock"]
    for raw in (
        cache["manifest"],
        cache["val_w"],
        {
            "path": cache["frequency_file"]["path"],
            "size_bytes": cache["frequency_file"]["size_bytes"],
            "sha256": cache["frequency_file"]["file_sha256"],
        },
    ):
        path = source_root / str(raw["path"])
        verify_file(path, raw)
        files[str(path.resolve())] = {
            "size_bytes": int(raw["size_bytes"]),
            "sha256": str(raw["sha256"]),
        }
    row_path = source_root / str(cache["root"]) / "val_row_ids.npy"
    row_identity = identity(row_path)
    if row_identity["sha256"] != cache["row_identity"]["val_row_file_sha256"]:
        raise RuntimeError("FiLM 局部 validation row file 漂移")
    files[str(row_path.resolve())] = row_identity
    cache_manifest = json.loads(
        (source_root / str(cache["manifest"]["path"])).read_text(encoding="utf-8")
    )
    dataset_path = Path(cache_manifest["dataset_index"]).resolve()
    dataset_identity = identity(dataset_path)
    if dataset_identity["sha256"] != cache_manifest["dataset_index_sha256"]:
        raise RuntimeError("FiLM 局部 dataset index 漂移")
    files[str(dataset_path)] = dataset_identity
    return entries, files, {"path": str(dataset_path), **dataset_identity}


def prepare_lock(*, code_root: Path = CODE_ROOT, source_root: Path = SOURCE_ROOT) -> Path:
    destination = code_root / IMPLEMENTATION_LOCK
    if destination.exists():
        raise FileExistsError(f"FiLM 局部 implementation lock 已存在: {destination}")
    previous_lock_path = code_root / PREVIOUS_IMPLEMENTATION_LOCK
    if not previous_lock_path.is_file():
        raise FileNotFoundError(f"缺少被替代的 FiLM 局部 implementation lock: {previous_lock_path}")
    source_path = code_root / SOURCE_LOCK
    metric_path = code_root / METRIC_LOCK
    if sha256_file(source_path) != SOURCE_LOCK_SHA256:
        raise RuntimeError("FiLM 局部 W0 source lock 漂移")
    if sha256_file(metric_path) != METRIC_LOCK_SHA256:
        raise RuntimeError("FiLM 局部 metric lock 漂移")
    source_lock = json.loads(source_path.read_text(encoding="utf-8"))
    metric_lock = json.loads(metric_path.read_text(encoding="utf-8"))
    entries, source_files, dataset_index = _source_files_for_lock(
        code_root=code_root,
        source_root=source_root,
        source_lock=source_lock,
        metric_lock=metric_lock,
    )
    for path in (source_path, metric_path):
        source_files[str(path.resolve())] = identity(path)

    prior_manifest = PRIOR_SUMMARY / "artifact_manifest.json"
    if sha256_file(prior_manifest) != PRIOR_SUMMARY_MANIFEST_SHA256:
        raise RuntimeError("前序 FiLM summary manifest 漂移")
    manifest = json.loads(prior_manifest.read_text(encoding="utf-8"))
    source_files[str(prior_manifest.resolve())] = identity(prior_manifest)
    for filename in (
        "window_statistics.csv",
        "modulation_error_associations.csv",
        "saturation_threshold_summary.csv",
        "summary_receipt.json",
    ):
        expected = manifest["files"][filename]
        path = PRIOR_SUMMARY / filename
        verify_file(path, expected)
        source_files[str(path.resolve())] = dict(expected)

    code_paths = sorted((code_root / "resp_train").rglob("*.py"))
    code_paths.extend(
        code_root / path
        for path in (SCRIPT_PATH, TEST_PATH, PROTOCOL_PATH, PRIOR_RESULTS)
    )
    lock = {
        "protocol": PROTOCOL,
        "status": "implementation_locked_real_smoke_and_analysis_pending",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": git_state(),
        "code_root": str(code_root.resolve()),
        "source_root": str(source_root.resolve()),
        "output_root": str(OUTPUT_ROOT.resolve()),
        "conditions": list(CONDITIONS),
        "coefficients": {key: list(value) for key, value in COEFFICIENTS.items()},
        "new_inference_conditions": list(NEW_CONDITIONS),
        "seeds": list(SEEDS),
        "split": "val",
        "windows_per_condition": WINDOW_COUNT,
        "batch_size": 128,
        "amp_dtype": "bfloat16",
        "w0_entries": entries,
        "cache_lock": source_lock["cache_lock"],
        "dataset_index": dataset_index,
        "prior_summary": {
            "path": str(PRIOR_SUMMARY.resolve()),
            "manifest_sha256": PRIOR_SUMMARY_MANIFEST_SHA256,
        },
        "supersedes": {
            "path": str(previous_lock_path.resolve()),
            **identity(previous_lock_path),
            "reason": (
                "真实 BF16/CUDA smoke 的原判据未测量原生跨 forward 重复性，"
                "无法区分 wrapper 算子路径差异与重复执行漂移；r2 改为同一次原生 "
                "forward 的融合张量硬锚点，并逐算子复刻原生融合。"
            ),
        },
        "source_files": source_files,
        "code_files": {
            str(path.relative_to(code_root)): identity(path)
            for path in sorted(set(code_paths))
        },
        "research_test_used": False,
    }
    write_json(destination, lock)
    return destination


def load_lock(*, code_root: Path = CODE_ROOT) -> tuple[dict[str, Any], str]:
    path = code_root / IMPLEMENTATION_LOCK
    if not path.is_file():
        raise FileNotFoundError(f"缺少 FiLM 局部 implementation lock: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or tuple(lock.get("conditions", ())) != CONDITIONS
        or tuple(lock.get("new_inference_conditions", ())) != NEW_CONDITIONS
        or tuple(lock.get("seeds", ())) != SEEDS
        or lock.get("coefficients") != {key: list(value) for key, value in COEFFICIENTS.items()}
        or int(lock.get("windows_per_condition", -1)) != WINDOW_COUNT
        or lock.get("split") != "val"
        or Path(lock.get("code_root", "")).resolve() != code_root.resolve()
        or Path(lock.get("source_root", "")).resolve() != SOURCE_ROOT
    ):
        raise ValueError("FiLM 局部 implementation lock 合同漂移")
    for relative, expected in lock["code_files"].items():
        verify_file(code_root / relative, expected)
    supersedes = lock.get("supersedes", {})
    if Path(supersedes.get("path", "")).resolve() != (
        code_root / PREVIOUS_IMPLEMENTATION_LOCK
    ).resolve():
        raise ValueError("FiLM 局部 implementation lock supersedes 漂移")
    verify_file(Path(supersedes["path"]), supersedes)
    return lock, sha256_file(path)


@contextmanager
def attempt(
    phase: str,
    lock_hash: str,
    *,
    seed: int | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> Iterator[Path]:
    if phase not in {"smoke", "analysis", "summary", "final"}:
        raise ValueError(f"未知 FiLM 局部 phase={phase}")
    parent = output_root / phase
    if seed is not None:
        parent = parent / f"seed_{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    context = {
        "protocol": PROTOCOL,
        "phase": phase,
        "seed": seed,
        "implementation_lock_sha256": lock_hash,
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "lifecycle_started.json", {**context, "status": "running"})
    print(f"FiLM local attempt: {output}", flush=True)
    try:
        yield output
        write_json(
            output / "lifecycle_completed.json",
            {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
        )
        manifest = {
            **context,
            "status": "completed",
            "files": {
                str(path.relative_to(output)): identity(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        }
        write_json(output / "artifact_manifest.json", manifest)
        write_json(
            output / "freeze_receipt.json",
            {"protocol": PROTOCOL, "phase": phase, "manifest": identity(output / "artifact_manifest.json")},
        )
    except BaseException as exc:
        write_json(
            output / "lifecycle_failed.json",
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
        raise ValueError(f"FiLM 局部 attempt 已失败: {path}")
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    verify_file(path / "artifact_manifest.json", freeze["manifest"])
    manifest = json.loads((path / "artifact_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
    ):
        raise ValueError("FiLM 局部 attempt 身份不一致")
    for relative, expected in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path):
            raise ValueError("FiLM 局部 manifest 路径越界")
        verify_file(file, expected)
    return manifest


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


def _verify_sources(lock: Mapping[str, Any]) -> None:
    for path, expected in lock["source_files"].items():
        verify_file(Path(path), expected)


def run_smoke(*, device: str = "cuda:0") -> Path:
    lock, lock_hash = load_lock()
    resolved = require_gpu(device)
    with attempt("smoke", lock_hash, seed=SEEDS[0]) as output:
        _write_provenance(output, lock)
        write_json(output / "environment.json", {**environment(device), "git": git_state(require_clean=True)})
        write_json(
            output / "access_started.json",
            {
                "status": "planned_access",
                "split": "val",
                "windows": 1,
                "checkpoint_seed": SEEDS[0],
                "research_test_used": False,
            },
        )
        _verify_sources(lock)
        cfg, model = _load_model(lock, SEEDS[0], device)
        data = _build_validation_data(cfg, lock)
        _, batch = next(_checked_batches(data.loader, data.rows))
        x = batch["x"][:1].to(resolved)
        w = batch["tf"]["w"][:1].to(resolved)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            native, captured = forward_with_capture(model, x, tf={"w": w})
            reconstructed = _fuse_w0(
                captured.z,
                captured.gamma_raw,
                captured.beta_raw,
                gamma_coefficient=0.5,
                beta_coefficient=0.5,
            )
            explicit = LocalSensitivityModel(
                model, "FULL", force_explicit_full=True
            )(x, tf={"w": w})
            native_repeat = model(x, tf={"w": w})
        fusion_maximum = float(
            (reconstructed.float() - captured.z_prime.float()).abs().max()
        )
        fusion_exact = bool(torch.equal(reconstructed, captured.z_prime))
        if fusion_maximum > 1e-6:
            raise RuntimeError(
                "FiLM 局部 0.5/0.5 融合改变同次原生 forward 的 Z': "
                f"{fusion_maximum}"
            )
        explicit_maximums = {
            key: float((explicit[key].float() - native[key].float()).abs().max())
            for key in native
        }
        explicit_exact = {
            key: bool(torch.equal(explicit[key], native[key])) for key in native
        }
        repeat_maximums = {
            key: float((native_repeat[key].float() - native[key].float()).abs().max())
            for key in native
        }
        repeat_exact = {
            key: bool(torch.equal(native_repeat[key], native[key])) for key in native
        }
        output_repeatable_at_tolerance = max(repeat_maximums.values()) <= 1e-6
        if output_repeatable_at_tolerance and max(explicit_maximums.values()) > 1e-6:
            raise RuntimeError(
                "FiLM 局部 explicit FULL 在原生重复 forward 可复现时改变输出: "
                f"explicit={explicit_maximums}, native_repeat={repeat_maximums}"
            )
        write_json(
            output / "smoke_receipt.json",
            {
                "protocol": PROTOCOL,
                "passed": True,
                "seed": SEEDS[0],
                "dataset_row_id": int(data.rows.iloc[0]["dataset_row_id"]),
                "fusion_anchor_max_abs_delta": fusion_maximum,
                "fusion_anchor_exact": fusion_exact,
                "native_explicit_max_abs_deltas": explicit_maximums,
                "native_explicit_exact": explicit_exact,
                "native_repeat_max_abs_deltas": repeat_maximums,
                "native_repeat_exact": repeat_exact,
                "output_repeatable_at_tolerance": output_repeatable_at_tolerance,
                "research_test_used": False,
            },
        )
        write_json(
            output / "access_receipt.json",
            {
                "split": "val",
                "windows": 1,
                "checkpoint_seed": SEEDS[0],
                "research_test_used": False,
            },
        )
        del model
        torch.cuda.empty_cache()
    return output


def _validate_smoke(path: Path, lock_hash: str) -> None:
    verify_attempt(path, phase="smoke", lock_hash=lock_hash)
    receipt = json.loads((path / "smoke_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("passed") is not True
        or int(receipt.get("seed", -1)) != SEEDS[0]
        or float(receipt.get("fusion_anchor_max_abs_delta", float("inf"))) > 1e-6
        or (
            receipt.get("output_repeatable_at_tolerance") is True
            and max(receipt["native_explicit_max_abs_deltas"].values()) > 1e-6
        )
    ):
        raise ValueError("FiLM 局部 smoke receipt 未通过")


def _condition_metrics(
    *,
    model: nn.Module,
    cfg: Any,
    data: Any,
    condition: str,
    seed: int,
    device: str,
    output: Path,
) -> pd.DataFrame:
    wrapper = LocalSensitivityModel(model, condition)
    batches = (batch for _, batch in _checked_batches(data.loader, data.rows))
    predictions = collect_predictions(
        wrapper,
        batches,
        device=device,
        max_windows=WINDOW_COUNT,
        use_amp=True,
    )
    metrics = evaluate_task_predictions(
        predictions,
        cfg,
        include_test_only=False,
        method=f"crd_tf102_w__local_sensitivity__{condition.lower()}",
    )
    validate_metrics(metrics, data.rows)
    metrics = attach_metadata(error_aligned_metrics(metrics), data.rows)
    metrics.insert(0, "condition", condition)
    metrics.insert(0, "seed", int(seed))
    directory = output / condition
    directory.mkdir(exist_ok=False)
    metrics.to_csv(directory / "metrics.csv", index=False)
    summary = summarize_task_metrics(metrics)
    summary.insert(0, "condition", condition)
    summary.insert(0, "seed", int(seed))
    summary.to_csv(directory / "summary.csv", index=False)
    write_json(
        directory / "condition_receipt.json",
        {
            "seed": int(seed),
            "condition": condition,
            "gamma_coefficient": COEFFICIENTS[condition][0],
            "beta_coefficient": COEFFICIENTS[condition][1],
            "windows": int(len(metrics)),
            "prediction_degeneracy_count": int(
                metrics["joint_prediction_degenerate"].astype(bool).sum()
            ),
        },
    )
    return metrics


def run_analysis(*, seed: int, device: str, smoke_receipt: Path) -> Path:
    lock, lock_hash = load_lock()
    if int(seed) not in SEEDS:
        raise ValueError(f"seed 必须属于 {SEEDS}")
    require_gpu(device)
    _validate_smoke(smoke_receipt, lock_hash)
    with attempt("analysis", lock_hash, seed=int(seed)) as output:
        _write_provenance(output, lock)
        write_json(output / "environment.json", {**environment(device), "git": git_state(require_clean=True)})
        write_json(
            output / "smoke_source.json",
            {"path": str(smoke_receipt.resolve()), "manifest": identity(smoke_receipt / "artifact_manifest.json")},
        )
        write_json(
            output / "access_started.json",
            {
                "status": "planned_access",
                "split": "val",
                "windows_per_new_condition": WINDOW_COUNT,
                "new_conditions": list(NEW_CONDITIONS),
                "checkpoint_seed": int(seed),
                "research_test_used": False,
            },
        )
        _verify_sources(lock)
        cfg, model = _load_model(lock, int(seed), device)
        OmegaConf.save(cfg, output / "resolved_config.yaml")
        data = _build_validation_data(cfg, lock)
        entry = next(item for item in lock["w0_entries"] if int(item["seed"]) == int(seed))
        frozen = pd.read_csv(SOURCE_ROOT / str(entry["run_dir"]) / "metrics.csv")
        validate_metrics(frozen, data.rows)
        full = attach_metadata(error_aligned_metrics(frozen), data.rows)
        full.insert(0, "condition", "FULL")
        full.insert(0, "seed", int(seed))
        _check_full_source(full, attach_metadata(error_aligned_metrics(frozen), data.rows), seed=int(seed))

        frames = [full]
        for condition in NEW_CONDITIONS:
            metrics = _condition_metrics(
                model=model,
                cfg=cfg,
                data=data,
                condition=condition,
                seed=int(seed),
                device=device,
                output=output,
            )
            for column in (
                "whole_rr_target_eligible",
                "local_rr_target_eligible",
                "local_rr_target_eligible_windows",
                "joint_target_eligible",
                "envelope_spearman_target_eligible",
            ):
                if not np.array_equal(metrics[column].to_numpy(), full[column].to_numpy()):
                    raise RuntimeError(f"FiLM 局部干预改变 target eligibility: {condition}/{column}")
            if metrics["joint_prediction_degenerate"].astype(bool).any():
                raise RuntimeError(f"FiLM 局部干预出现 prediction degeneracy: {condition}")
            frames.append(metrics)
            print(f"FiLM local complete seed={seed} condition={condition}", flush=True)
            torch.cuda.empty_cache()

        all_metrics = pd.concat(frames, ignore_index=True)
        validate_condition_matrix(all_metrics, seeds=(int(seed),))
        paired = build_paired_deltas(all_metrics)
        strata = summarize_paired_deltas(paired)
        response = local_response(strata)
        all_metrics.to_csv(output / "condition_metrics.csv", index=False)
        paired.to_csv(output / "paired_window_deltas.csv", index=False)
        strata.to_csv(output / "condition_summary.csv", index=False)
        response.to_csv(output / "local_response.csv", index=False)
        write_json(
            output / "analysis_receipt.json",
            {
                "protocol": PROTOCOL,
                "seed": int(seed),
                "conditions": list(CONDITIONS),
                "new_inference_conditions": list(NEW_CONDITIONS),
                "windows_per_condition": WINDOW_COUNT,
                "new_metric_rows": len(NEW_CONDITIONS) * WINDOW_COUNT,
                "paired_rows": int(len(paired)),
                "prediction_degeneracy_count": 0,
                "research_test_used": False,
                "training_used": False,
            },
        )
        write_json(
            output / "access_receipt.json",
            {
                "split": "val",
                "windows_per_new_condition": WINDOW_COUNT,
                "new_condition_count": len(NEW_CONDITIONS),
                "checkpoint_seed": int(seed),
                "research_test_used": False,
            },
        )
        del model
        torch.cuda.empty_cache()
    return output


def _plot_summary(output: Path, response: pd.DataFrame, seed_level: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(exist_ok=False)
    overall = response.loc[response["scope"].eq("all")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, metric in zip(axes.ravel(), AXES, strict=False):
        part = overall.loc[overall["metric"].eq(metric)]
        for path, marker in (("gamma", "o"), ("beta", "s")):
            group = part.loc[part["path"].eq(path)]
            for row in group.itertuples():
                axis.plot(
                    [0.4, 0.5, 0.6],
                    [row.down_mean, row.full_mean, row.up_mean],
                    marker=marker,
                    alpha=0.7,
                    label=f"{path}-{row.seed}",
                )
        axis.set_title(metric)
        axis.set_xlabel("coefficient")
        axis.set_ylabel("error-aligned mean")
    axes.ravel()[-1].axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "coefficient_response.png", dpi=180)
    plt.close(fig)

    high = response.loc[
        response["scope"].eq("target_stratum") & response["level"].eq("high")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, path in zip(axes, ("gamma", "beta"), strict=True):
        pivot = high.loc[high["path"].eq(path)].pivot_table(
            index="metric", columns="seed", values="central_slope"
        )
        image = axis.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
        axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
        axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45)
        axis.set_title(f"high stratum {path} slope")
        fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(figures / "high_stratum_response.png", dpi=180)
    plt.close(fig)

    quality = seed_level.loc[seed_level["scope"].eq("quality_level")]
    quality = quality.loc[quality["metric"].eq("local_rr_mae_bpm")]
    fig, axis = plt.subplots(figsize=(9, 4))
    labels = sorted(quality["condition"].unique())
    levels = sorted(quality["level"].unique())
    width = 0.8 / max(1, len(levels))
    positions = np.arange(len(labels))
    for index, level in enumerate(levels):
        values = (
            quality.loc[quality["level"].eq(level)]
            .set_index("condition")
            .reindex(labels)["delta_seed_mean"]
        )
        axis.bar(positions + index * width, values, width=width, label=level)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions + width * (len(levels) - 1) / 2, labels, rotation=20)
    axis.set_ylabel("Local RR paired delta")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "quality_response.png", dpi=180)
    plt.close(fig)


def run_summary(*, runs: Sequence[Path]) -> Path:
    lock, lock_hash = load_lock()
    if len(runs) != len(SEEDS):
        raise ValueError("FiLM 局部 summary 必须提供三个 analysis attempts")
    with attempt("summary", lock_hash) as output:
        _write_provenance(output, lock)
        write_json(output / "environment.json", {**environment(None), "git": git_state(require_clean=True)})
        metrics_frames, paired_frames = [], []
        sources, observed_seeds = [], set()
        for path in runs:
            path = path.resolve()
            verify_attempt(path, phase="analysis", lock_hash=lock_hash)
            receipt = json.loads((path / "analysis_receipt.json").read_text(encoding="utf-8"))
            seed = int(receipt["seed"])
            if seed in observed_seeds:
                raise ValueError(f"FiLM 局部 summary 重复 seed={seed}")
            observed_seeds.add(seed)
            metrics_frames.append(pd.read_csv(path / "condition_metrics.csv"))
            paired_frames.append(pd.read_csv(path / "paired_window_deltas.csv"))
            sources.append(
                {"seed": seed, "path": str(path), "manifest": identity(path / "artifact_manifest.json")}
            )
        if observed_seeds != set(SEEDS):
            raise ValueError("FiLM 局部 summary seed 不完整")
        metrics = pd.concat(metrics_frames, ignore_index=True)
        validate_condition_matrix(metrics)
        paired = pd.concat(paired_frames, ignore_index=True)
        expected_paired = len(SEEDS) * len(NEW_CONDITIONS) * WINDOW_COUNT
        if len(paired) != expected_paired or paired[
            ["seed", "condition", "dataset_row_id"]
        ].duplicated().any():
            raise ValueError("FiLM 局部 summary paired 矩阵不完整")
        strata = summarize_paired_deltas(paired)
        seeds = seed_summary(strata)
        response = local_response(strata)
        quality_subject = strata.loc[
            strata["scope"].isin(["quality_level", "target_quality", "samp_id"])
        ].copy()
        paired.to_csv(output / "paired_window_deltas.csv", index=False)
        strata.to_csv(output / "stratum_summary.csv", index=False)
        seeds.to_csv(output / "seed_summary.csv", index=False)
        response.to_csv(output / "local_response.csv", index=False)
        quality_subject.to_csv(output / "quality_subject_summary.csv", index=False)
        write_json(output / "analysis_sources.json", {"runs": sources})
        write_json(
            output / "matrix_receipt.json",
            {
                "protocol": PROTOCOL,
                "seeds": list(SEEDS),
                "conditions": list(CONDITIONS),
                "new_inference_evaluations": len(SEEDS) * len(NEW_CONDITIONS),
                "new_metric_rows": len(SEEDS) * len(NEW_CONDITIONS) * WINDOW_COUNT,
                "paired_rows": int(len(paired)),
                "prediction_degeneracy_count": 0,
                "research_test_used": False,
            },
        )
        _plot_summary(output, response, seeds)
    return output


def _conclusion(summary: Path) -> str:
    response = pd.read_csv(summary / "local_response.csv")
    seeds = pd.read_csv(summary / "seed_summary.csv")
    overall = seeds.loc[seeds["scope"].eq("all")]
    high = response.loc[
        response["scope"].eq("target_stratum") & response["level"].eq("high")
    ]
    lines = [
        "# W0 FiLM 局部剂量敏感性：描述性结论",
        "",
        "## 观察结果",
        "",
    ]
    for condition in NEW_CONDITIONS:
        part = overall.loc[
            overall["condition"].eq(condition)
            & overall["metric"].eq("local_rr_mae_bpm")
        ]
        if len(part) == 1:
            row = part.iloc[0]
            lines.append(
                f"- {condition}：整体 Local RR paired delta seed mean="
                f"{row['delta_seed_mean']:.6g}，worse seeds={int(row['worse_seed_count'])}/3。"
            )
    lines.extend(["", "high target-modulation 的中心斜率：", ""])
    for path in ("gamma", "beta"):
        part = high.loc[high["path"].eq(path)]
        for metric in AXES:
            values = part.loc[part["metric"].eq(metric), "central_slope"]
            if len(values):
                lines.append(f"- {path}/{metric}: {values.mean():.6g}。")
    lines.extend(
        [
            "",
            "## 机制假设",
            "",
            "- 局部 coefficient perturbation 只刻画冻结模型附近的输出敏感性，不等同于重新训练后的可达性能。",
            "- high stratum 与其他层方向不一致时，优先解释为任务层交互，不选择全局系数。",
            "",
            "## 待验证问题",
            "",
            "- 只有方向跨 seed、主体和 strata 足够稳定时，才另立训练系数实验。",
            "- test 未访问；任何 test 评价需独立附件。",
            "",
        ]
    )
    return "\n".join(lines)


def run_finalize(*, summary: Path) -> Path:
    lock, lock_hash = load_lock()
    verify_attempt(summary, phase="summary", lock_hash=lock_hash)
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
                "research_test_used": False,
            },
        )
        with (output / "conclusions_zh.md").open("x", encoding="utf-8") as handle:
            handle.write(_conclusion(summary))
        write_json(
            output / "final_receipt.json",
            {
                "protocol": PROTOCOL,
                "status": "complete",
                "scope": "three frozen W0 checkpoints, validation-only local coefficient sensitivity",
                "causal_claim": False,
                "research_test_used": False,
            },
        )
    return output
