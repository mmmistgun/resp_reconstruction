from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train.data.index import filter_index
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.paper_evidence.center30_data import INPUT_SLICES as CENTER30_INPUT_SLICES
from resp_train.paper_evidence.center30_data import TARGET_SLICE as CENTER30_TARGET_SLICE
from resp_train.paper_evidence.center30_metrics import (
    CENTER30_PRIMARY_METRICS,
    evaluate_center30_predictions,
    summarize_center30_metrics,
)
from resp_train.paper_evidence.center30_model import build_center30_model
from resp_train.paper_evidence.center_context_data import INPUT_SLICES as CENTER60_INPUT_SLICES
from resp_train.paper_evidence.center_context_data import TARGET_SLICE as CENTER60_TARGET_SLICE
from resp_train.paper_evidence.center_context_experiment import _configure_training_runtime
from resp_train.paper_evidence.center_context_metrics import (
    CENTER_PRIMARY_METRICS,
    evaluate_center_predictions,
    summarize_center_metrics,
)
from resp_train.paper_evidence.center_context_model import build_center_context_model
from resp_train.paper_evidence.context_length_research_test_cache import (
    ContextLengthResearchTestWCacheReader,
    EXPECTED_DATASET_INDEX_SHA256,
    EXPECTED_TEST_ROW_IDS_SHA256,
    EXPECTED_TEST_SAMP_IDS,
    EXPECTED_TEST_WINDOWS,
    INPUT_KEY,
    PROTOCOL_ID,
    TARGET_KEY,
    TEST_SAMPLE_SEED,
    TEST_SAMPLE_STRATEGY,
    _validate_test_rows,
    sha256_file,
)
from resp_train.utils.run import resolve_device, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_SCHEMA_VERSION = "paper-context-length-research-test-evaluation-v1"
FROZEN_CACHE_ROOT = REPO_ROOT / (
    "runs/paper_evidence_v1/context_length_research_test_w_cache/"
    "cf89c6e678bb243801c0ca577ec14d69d724603e51b3335c4e1f474e7f999d5c"
)
FROZEN_CACHE_MANIFEST_SHA256 = "9b475926258d129851fb7b9c10d2b342ac2e833b121842b18437585059bf1b98"
OUTPUT_ROOT = REPO_ROOT / "runs/paper_evidence_v1/context_length_research_test"
CENTER30_SUMMARY_ROOT = REPO_ROOT / "runs/paper_evidence_v1/center30_context/p4s3_validation_summary"
CENTER60_SUMMARY_ROOT = REPO_ROOT / "runs/paper_evidence_v1/center_context/p4_validation_summary"
CENTER30_SUMMARY_RECEIPT_SHA256 = "2c533117c735e5bedb65f31ca77fa9dd22663433c0a284b4ee7bd9dc0beee2ef"
CENTER30_SUMMARY_MANIFEST_SHA256 = "b18bbed8ab826b8a6e415866c4ad1b9d5171fd4eebc0d5c4090544428fc19bba"
CENTER60_SUMMARY_RECEIPT_SHA256 = "7a8e5a3118e5059fc46ff287f4553952f2fea44653501c7ddedfd09145da9cb9"
CENTER60_SUMMARY_MANIFEST_SHA256 = "c0806893d1b4350da46fb9bc02d2056a8b6c8e29d368205709be00aae9bacb2a"
FORMAL_SEEDS = (20260811, 20260812, 20260813)
TASK_MATRIX = {
    "center30": {
        "c201": ("c201_center30", (30, 45, 60, 90)),
        "wr": ("w_reduced_center30", (30, 45, 60, 90)),
    },
    "center60": {
        "c201": ("c201_center60", (60, 90, 180)),
        "wr": ("w_reduced_center60", (60, 90, 180)),
    },
}
EXPECTED_PARAMETER_COUNTS = {"c201": 1069802, "wr": 1219850}
EXPECTED_RUNTIME = {
    "device_type": "cuda",
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_benchmark": False,
    "amp_enabled": True,
    "amp_dtype": "bfloat16",
}


@dataclass(frozen=True)
class EvaluationSpec:
    task: str
    model_key: str
    variant: str
    experiment_id: str
    input_sec: int
    output_sec: int
    seed: int
    run_dir: str
    run_commit: str
    artifact_manifest_sha256: str
    resolved_config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    selected_epoch: int
    parameter_count: int

    @property
    def identity(self) -> tuple[str, str, int, int]:
        return (self.task, self.model_key, self.input_sec, self.seed)


def expected_research_test_evaluations() -> dict[tuple[str, str, int, int], EvaluationSpec]:
    records: dict[tuple[str, str, int, int], EvaluationSpec] = {}
    for task in ("center30", "center60"):
        receipt, receipt_hash = _load_frozen_summary(task)
        expected_inputs = 24 if task == "center30" else 18
        inputs = receipt.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != expected_inputs:
            raise RuntimeError(f"{task} validation summary input matrix 不完整")
        for model_key, (variant, lengths) in TASK_MATRIX[task].items():
            for input_sec in lengths:
                experiment_id = _experiment_id(task, model_key, input_sec)
                for seed in FORMAL_SEEDS:
                    matches = [
                        item
                        for item in inputs
                        if item.get("experiment_id") == experiment_id and int(item.get("seed", -1)) == seed
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} 未被 validation summary 唯一锁定")
                    summary_input = matches[0]
                    run_dir = _expected_run_dir(task, experiment_id, seed)
                    if Path(str(summary_input.get("run_dir"))).resolve() != run_dir:
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} run path 漂移")
                    lifecycle_path = run_dir / "lifecycle.json"
                    manifest_path = run_dir / "artifact_manifest.json"
                    if sha256_file(lifecycle_path) != summary_input.get("lifecycle_sha256"):
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} lifecycle hash 漂移")
                    lifecycle = _read_json(lifecycle_path)
                    if lifecycle != {"status": "complete"}:
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} lifecycle 未 complete")
                    actual_manifest_hash = sha256_file(manifest_path)
                    if actual_manifest_hash != summary_input.get("artifact_manifest_sha256"):
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} artifact manifest hash 漂移")
                    manifest = _read_json(manifest_path)
                    checkpoint_name = (
                        "checkpoint_best_center30_rr.pt" if task == "center30" else "checkpoint_best_center_rr.pt"
                    )
                    artifacts = manifest.get("artifacts", {})
                    checkpoint_record = artifacts.get(checkpoint_name)
                    config_record = artifacts.get("resolved_config.yaml")
                    if not isinstance(checkpoint_record, Mapping) or not isinstance(config_record, Mapping):
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} checkpoint/config 未登记")
                    config_path = run_dir / "resolved_config.yaml"
                    if (
                        config_path.stat().st_size != int(config_record.get("size_bytes", -1))
                        or sha256_file(config_path) != config_record.get("sha256")
                    ):
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} resolved config hash 漂移")
                    cfg = OmegaConf.load(config_path)
                    expected_commit = _expected_run_commit(task, seed)
                    expected_selector = (
                        "full_validation_center30_rr_mae_bpm_strict_lower_tie_earlier"
                        if task == "center30"
                        else "full_validation_center_rr_mae_bpm_strict_lower_tie_earlier"
                    )
                    if (
                        manifest.get("experiment_id") != experiment_id
                        or int(manifest.get("seed", -1)) != seed
                        or int(manifest.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNTS[model_key]
                        or manifest.get("selector") != expected_selector
                        or manifest.get("git") != {"commit": expected_commit, "dirty": False, "error": None}
                        or manifest.get("test_access") is not False
                        or str(cfg.model.variant) != variant
                        or int(cfg.window.input_sec) != input_sec
                        or int(cfg.training.seed) != seed
                        or int(cfg.model.initialization_seed) != seed
                    ):
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} formal identity 漂移")
                    checkpoint_path = run_dir / checkpoint_name
                    if checkpoint_path.stat().st_size != int(checkpoint_record.get("size_bytes", -1)):
                        raise RuntimeError(f"{task}/{experiment_id}/seed_{seed} checkpoint size 漂移")
                    spec = EvaluationSpec(
                        task=task,
                        model_key=model_key,
                        variant=variant,
                        experiment_id=experiment_id,
                        input_sec=input_sec,
                        output_sec=30 if task == "center30" else 60,
                        seed=seed,
                        run_dir=str(run_dir),
                        run_commit=expected_commit,
                        artifact_manifest_sha256=actual_manifest_hash,
                        resolved_config_sha256=str(config_record["sha256"]),
                        checkpoint_path=str(checkpoint_path),
                        checkpoint_sha256=str(checkpoint_record["sha256"]),
                        checkpoint_size_bytes=int(checkpoint_record["size_bytes"]),
                        selected_epoch=int(manifest["best_epoch"]),
                        parameter_count=EXPECTED_PARAMETER_COUNTS[model_key],
                    )
                    if spec.identity in records:
                        raise RuntimeError(f"重复 research-test identity: {spec.identity}")
                    records[spec.identity] = spec
        if receipt_hash not in {CENTER30_SUMMARY_RECEIPT_SHA256, CENTER60_SUMMARY_RECEIPT_SHA256}:
            raise AssertionError("unreachable summary hash")
    wanted = {
        (task, model_key, input_sec, seed)
        for task, models in TASK_MATRIX.items()
        for model_key, (_, lengths) in models.items()
        for input_sec in lengths
        for seed in FORMAL_SEEDS
    }
    if set(records) != wanted or len(records) != 42:
        raise RuntimeError("research-test allowlist 必须恰为 42 checkpoints")
    return records


class ContextLengthResearchTestDataset(Dataset):
    def __init__(
        self,
        parent: Dataset,
        *,
        task: str,
        input_samples: int,
        w_cache: ContextLengthResearchTestWCacheReader | None,
    ) -> None:
        self.parent = parent
        self.task = str(task)
        self.input_samples = int(input_samples)
        if self.task == "center30":
            self.input_slices = CENTER30_INPUT_SLICES
            self.target_slice = CENTER30_TARGET_SLICE
        elif self.task == "center60":
            self.input_slices = CENTER60_INPUT_SLICES
            self.target_slice = CENTER60_TARGET_SLICE
        else:
            raise ValueError("research-test task 只允许 center30/center60")
        if self.input_samples not in self.input_slices:
            raise ValueError(f"{self.task} 不允许 input_samples={self.input_samples}")
        self.w_cache = w_cache
        rows = getattr(parent, "rows", None)
        if not isinstance(rows, pd.DataFrame) or set(rows["split"].astype(str)) != {"test"}:
            raise ValueError("research-test parent dataset 必须且只能包含 test rows")
        if self.w_cache is not None:
            self.w_cache.verify_rows(rows["dataset_row_id"].astype(int).tolist())

    def __len__(self) -> int:
        return len(self.parent)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.parent[index]
        if not isinstance(item, Mapping) or not {"x", "target", "meta"}.issubset(item):
            raise TypeError("research-test parent item 必须包含 x/target/meta")
        x = item["x"]
        target = item["target"]
        if x.shape[-1] != 18000 or target.shape[-1] != 18000:
            raise ValueError("research-test parent waveform 必须为 18000 点")
        if not bool(torch.isfinite(x).all() and torch.isfinite(target).all()):
            raise FloatingPointError("research-test parent input/target 含 NaN/Inf")
        input_start, input_stop = self.input_slices[self.input_samples]
        target_start, target_stop = self.target_slice
        meta = dict(item["meta"])
        if str(meta.get("split")) != "test":
            raise ValueError("research-test item split 漂移")
        if "rr_peak_valid_mask" in meta:
            meta["rr_peak_valid_mask"] = meta["rr_peak_valid_mask"][target_start:target_stop].contiguous()
        meta.update(
            {
                "research_test_task": self.task,
                "parent_samples": 18000,
                "input_samples": self.input_samples,
                "input_slice_start": input_start,
                "input_slice_stop": input_stop,
                "target_slice_start": target_start,
                "target_slice_stop": target_stop,
            }
        )
        output: dict[str, Any] = {
            "x": x[..., input_start:input_stop].contiguous(),
            "target": target[..., target_start:target_stop].contiguous(),
            "meta": meta,
        }
        if self.w_cache is not None:
            output["tf"] = {"w": self.w_cache.get(int(meta["dataset_row_id"]))}
        return output


def evaluate_research_test(
    *,
    task: str,
    model_key: str,
    input_sec: int,
    seed: int,
    device: str,
    command: str,
) -> Path:
    evaluation_commit = _require_clean_git()
    if str(device) != "cuda:0":
        raise ValueError("上下文 research-test evaluation 固定使用逻辑 cuda:0")
    key = (str(task), str(model_key), int(input_sec), int(seed))
    spec = expected_research_test_evaluations().get(key)
    if spec is None:
        raise ValueError(f"evaluation identity 不在冻结 42-checkpoint 矩阵: {key}")
    cache_manifest_path = FROZEN_CACHE_ROOT / "cache_manifest.json"
    if sha256_file(cache_manifest_path) != FROZEN_CACHE_MANIFEST_SHA256:
        raise RuntimeError("冻结联合 W cache manifest SHA-256 漂移")
    checkpoint_path = Path(spec.checkpoint_path)
    if (
        checkpoint_path.stat().st_size != spec.checkpoint_size_bytes
        or sha256_file(checkpoint_path) != spec.checkpoint_sha256
    ):
        raise RuntimeError("冻结 validation-selected checkpoint SHA-256/size 漂移")
    output_dir = OUTPUT_ROOT / spec.task / spec.experiment_id / f"seed_{spec.seed}"
    if output_dir.exists():
        raise FileExistsError(f"research-test evaluation 输出禁止覆盖: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    lifecycle_path = output_dir / "lifecycle.json"
    _write_json(lifecycle_path, {"status": "running"})
    try:
        cfg = _load_evaluation_config(spec, device=device)
        OmegaConf.save(cfg, output_dir / "resolved_evaluation_config.yaml")
        _write_json(output_dir / "checkpoint_identity.json", asdict(spec))
        set_seed(spec.seed)
        resolved_device = resolve_device(device)
        runtime = _configure_training_runtime(cfg, resolved_device)
        if runtime != EXPECTED_RUNTIME:
            raise RuntimeError("research-test runtime identity 漂移")
        runtime_identity = {
            **runtime,
            "logical_device": str(resolved_device),
            "cuda_visible_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": str(torch.cuda.get_device_name(resolved_device)),
        }
        _write_json(output_dir / "runtime_identity.json", runtime_identity)
        model = _build_model(cfg, spec)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_checkpoint_payload(checkpoint, spec)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # 先闭合 checkpoint payload/state identity，再首次读取 test target。
        loader, data_identity = _build_test_loader(cfg, spec)
        _write_json(output_dir / "test_data_identity.json", data_identity)
        predictions = collect_predictions(
            model,
            loader,
            device=resolved_device,
            max_windows=EXPECTED_TEST_WINDOWS,
            use_amp=True,
        )
        metrics = _evaluate_predictions(predictions, cfg, spec)
        _validate_metrics(metrics, spec)
        metrics_path = output_dir / "research_test_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        summary = _summarize_metrics(metrics, spec)
        summary.to_csv(output_dir / "research_test_metrics_summary.csv", index=False)
        receipt = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": asdict(spec),
            "execution": {
                "command": command,
                "cwd": str(REPO_ROOT),
                "git_commit": evaluation_commit,
                "git_dirty": False,
                "runtime": runtime_identity,
                "dependencies": _dependency_versions(),
            },
            "data": data_identity,
            "counts": {"test_rows": len(metrics), "expected_test_rows": EXPECTED_TEST_WINDOWS},
            "access": {
                "train_signal_or_target_read": False,
                "validation_signal_target_or_prediction_read": False,
                "test_input_read": True,
                "test_target_array_read": True,
                "checkpoint_content_read": True,
                "checkpoint_final_read": False,
                "w_cache_read": spec.model_key == "wr",
                "model_training_used": False,
                "model_inference_used": True,
                "checkpoint_reselection_used": False,
            },
            "artifacts": _artifact_records(output_dir),
        }
        _write_json(output_dir / "evaluation_receipt.json", receipt)
        _write_json(
            output_dir / "artifact_manifest.json",
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "complete",
                "identity": {
                    "task": spec.task,
                    "experiment_id": spec.experiment_id,
                    "seed": spec.seed,
                },
                "files": _artifact_records(output_dir),
            },
        )
    except BaseException as exc:
        _write_json(
            lifecycle_path,
            {"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)},
        )
        raise
    _write_json(lifecycle_path, {"status": "complete"})
    return output_dir / "evaluation_receipt.json"


def _build_test_loader(cfg: DictConfig, spec: EvaluationSpec) -> tuple[DataLoader, dict[str, Any]]:
    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    if not index_path.is_file() or sha256_file(index_path) != EXPECTED_DATASET_INDEX_SHA256:
        raise RuntimeError("research-test dataset index SHA-256 漂移")
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = filter_index(
        audited,
        cfg,
        split="test",
        max_windows=None,
        sample_strategy=TEST_SAMPLE_STRATEGY,
        sample_seed=TEST_SAMPLE_SEED,
    )
    _validate_test_rows(
        rows,
        expected_count=EXPECTED_TEST_WINDOWS,
        expected_samp_ids=EXPECTED_TEST_SAMP_IDS,
        expected_row_ids_sha256=EXPECTED_TEST_ROW_IDS_SHA256,
    )
    non_test_ids = audited.loc[
        audited["split"].astype(str).isin({"train", "val"}), "dataset_row_id"
    ].to_numpy(dtype=np.int64)
    if np.intersect1d(rows["dataset_row_id"].to_numpy(dtype=np.int64), non_test_ids).size:
        raise RuntimeError("research-test rows 与 train/validation rows 重叠")
    w_cache = None
    if spec.model_key == "wr":
        w_cache = ContextLengthResearchTestWCacheReader(
            FROZEN_CACHE_ROOT,
            input_samples=spec.input_sec * 100,
            expected_manifest_sha256=FROZEN_CACHE_MANIFEST_SHA256,
        )
    parent = ResearchV2WindowDataset(index_path, rows, cfg, preload_windows=False)
    dataset = ContextLengthResearchTestDataset(
        parent,
        task=spec.task,
        input_samples=spec.input_sec * 100,
        w_cache=w_cache,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, drop_last=False)
    return loader, {
        "dataset_index": str(index_path),
        "dataset_index_sha256": EXPECTED_DATASET_INDEX_SHA256,
        "split": "test",
        "test_rows": len(rows),
        "test_samp_ids": int(rows["samp_id"].nunique()),
        "test_row_ids_sha256": EXPECTED_TEST_ROW_IDS_SHA256,
        "non_test_row_overlap_count": 0,
        "sample_strategy": TEST_SAMPLE_STRATEGY,
        "sample_seed": TEST_SAMPLE_SEED,
        "input_key": INPUT_KEY,
        "target_key": TARGET_KEY,
        "input_samples": spec.input_sec * 100,
        "output_samples": spec.output_sec * 100,
        "w_cache_path": str(FROZEN_CACHE_ROOT) if spec.model_key == "wr" else None,
        "w_cache_manifest_sha256": FROZEN_CACHE_MANIFEST_SHA256 if spec.model_key == "wr" else None,
    }


def _load_evaluation_config(spec: EvaluationSpec, *, device: str) -> DictConfig:
    path = Path(spec.run_dir) / "resolved_config.yaml"
    if sha256_file(path) != spec.resolved_config_sha256:
        raise RuntimeError("evaluation resolved config SHA-256 漂移")
    cfg = OmegaConf.load(path)
    if (
        str(cfg.model.variant) != spec.variant
        or int(cfg.window.input_sec) != spec.input_sec
        or int(cfg.window.output_sec) != spec.output_sec
        or int(cfg.training.seed) != spec.seed
    ):
        raise RuntimeError("evaluation config task/model/length/seed 漂移")
    cfg.protocol.stage = "research_test"
    cfg.protocol.run_role = "frozen_research_test_evaluation"
    cfg.protocol.execution_gate = "research_test_only"
    cfg.training.device = str(device)
    cfg.training.batch_size = 128
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    cfg.training.prefetch_factor = None
    cfg.training.show_progress = False
    cfg.training.use_amp = True
    cfg.training.allow_tf32 = False
    cfg.training.cudnn_benchmark = False
    cfg.data.center_w_cache_path = None
    cfg.data.center_w_cache_manifest_sha256 = None
    cfg.data.access_splits = ["test"]
    return cfg


def _build_model(cfg: DictConfig, spec: EvaluationSpec) -> torch.nn.Module:
    model = build_center30_model(cfg) if spec.task == "center30" else build_center_context_model(cfg)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != spec.parameter_count:
        raise RuntimeError("research-test model parameter count 漂移")
    return model


def _validate_checkpoint_payload(checkpoint: Mapping[str, Any], spec: EvaluationSpec) -> None:
    if "model_state_dict" not in checkpoint or int(checkpoint.get("epoch", -1)) != spec.selected_epoch:
        raise RuntimeError("checkpoint payload state/selected epoch 漂移")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("checkpoint payload 缺少 resolved config")
    model = config.get("model", {})
    window = config.get("window", {})
    training = config.get("training", {})
    if (
        str(model.get("variant")) != spec.variant
        or int(window.get("input_sec", -1)) != spec.input_sec
        or int(window.get("output_sec", -1)) != spec.output_sec
        or int(training.get("seed", -1)) != spec.seed
    ):
        raise RuntimeError("checkpoint payload task/model/length/seed identity 漂移")


def _evaluate_predictions(
    predictions: dict[str, np.ndarray], cfg: DictConfig, spec: EvaluationSpec
) -> pd.DataFrame:
    if spec.task == "center30":
        return evaluate_center30_predictions(predictions, cfg, method=spec.variant)
    return evaluate_center_predictions(predictions, cfg, method=spec.variant)


def _summarize_metrics(metrics: pd.DataFrame, spec: EvaluationSpec) -> pd.DataFrame:
    return summarize_center30_metrics(metrics) if spec.task == "center30" else summarize_center_metrics(metrics)


def _validate_metrics(metrics: pd.DataFrame, spec: EvaluationSpec) -> None:
    primary = CENTER30_PRIMARY_METRICS if spec.task == "center30" else CENTER_PRIMARY_METRICS
    prefix = "center30" if spec.task == "center30" else "center"
    required = {
        "method",
        "dataset_row_id",
        "split",
        "input_set",
        "samp_id",
        "coupling_state_id",
        *primary,
        f"{prefix}_ibi_coverage",
        f"{prefix}_ibi_target_eligible",
        f"{prefix}_ibi_interpretable",
    }
    if len(metrics) != EXPECTED_TEST_WINDOWS or not required.issubset(metrics.columns):
        raise RuntimeError("research-test metrics row/schema 不完整")
    if set(metrics["method"].astype(str)) != {spec.variant} or set(metrics["split"].astype(str)) != {"test"}:
        raise RuntimeError("research-test metrics method/split 漂移")
    if set(metrics["input_set"].astype(str)) != {"research_v2_waveform"}:
        raise RuntimeError("research-test metrics input_set 漂移")
    row_ids = metrics["dataset_row_id"].to_numpy(dtype=np.int64)
    if (
        np.unique(row_ids).size != EXPECTED_TEST_WINDOWS
        or hashlib.sha256(row_ids.tobytes(order="C")).hexdigest() != EXPECTED_TEST_ROW_IDS_SHA256
        or metrics["samp_id"].nunique() != EXPECTED_TEST_SAMP_IDS
    ):
        raise RuntimeError("research-test metrics row identity 漂移")
    for metric in primary:
        values = pd.to_numeric(metrics[metric], errors="coerce").to_numpy(dtype=np.float64)
        if np.isinf(values).any() or (metric != f"{prefix}_ibi_medae_sec" and not np.isfinite(values).all()):
            raise FloatingPointError(f"research-test metric 非有限: {metric}")


def _load_frozen_summary(task: str) -> tuple[dict[str, Any], str]:
    if task == "center30":
        root = CENTER30_SUMMARY_ROOT
        receipt_hash = CENTER30_SUMMARY_RECEIPT_SHA256
        manifest_hash = CENTER30_SUMMARY_MANIFEST_SHA256
        expected_runs = 24
    else:
        root = CENTER60_SUMMARY_ROOT
        receipt_hash = CENTER60_SUMMARY_RECEIPT_SHA256
        manifest_hash = CENTER60_SUMMARY_MANIFEST_SHA256
        expected_runs = 18
    receipt_path = root / "summary_receipt.json"
    manifest_path = root / "artifact_manifest.json"
    if sha256_file(receipt_path) != receipt_hash or sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError(f"{task} validation summary provenance SHA-256 漂移")
    receipt = _read_json(receipt_path)
    if (
        receipt.get("status") != "complete"
        or int(receipt.get("counts", {}).get("actual_runs", -1)) != expected_runs
        or receipt.get("access", {}).get("research_test_accessed") is not False
        or receipt.get("decision", {}).get("model_or_length_selection_performed") is not False
    ):
        raise RuntimeError(f"{task} validation summary 不满足 test 前置条件")
    return receipt, receipt_hash


def _experiment_id(task: str, model_key: str, input_sec: int) -> str:
    if task == "center30":
        return f"C30V1_{'C201' if model_key == 'c201' else 'WR'}_{input_sec}"
    return f"CCV1_{'C201' if model_key == 'c201' else 'WR'}_{input_sec}"


def _expected_run_dir(task: str, experiment_id: str, seed: int) -> Path:
    if task == "center30":
        stage = "p4s_single_seed" if seed == FORMAL_SEEDS[0] else "p4s_additional_seeds"
        return (REPO_ROOT / "runs/paper_evidence_v1/center30_context" / stage / experiment_id / f"seed_{seed}").resolve()
    stage = "p3_single_seed" if seed == FORMAL_SEEDS[0] else "p4_additional_seeds"
    return (REPO_ROOT / "runs/paper_evidence_v1/center_context" / stage / experiment_id / f"seed_{seed}").resolve()


def _expected_run_commit(task: str, seed: int) -> str:
    if task == "center30":
        return "bead33307aa79139b8124bdf515700bf7e18379b" if seed == FORMAL_SEEDS[0] else "2107cf9935229b28224035aa7272515e91be0706"
    return "f57583b91f04e5ab5ea9475042489e8aeabb4e9f" if seed == FORMAL_SEEDS[0] else "55d515e74adc551a760fdd5914f5c4a1c1ced0a8"


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {"filename": path.name, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in {"artifact_manifest.json", "lifecycle.json"}
    ]


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in ("numpy", "pandas", "torch", "omegaconf", "scipy", "mamba-ssm"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须为 object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_clean_git() -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if commit.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("上下文 research-test evaluation 必须从干净 Git commit 运行")
    return commit.stdout.strip()


__all__ = [
    "ContextLengthResearchTestDataset",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationSpec",
    "FROZEN_CACHE_MANIFEST_SHA256",
    "FROZEN_CACHE_ROOT",
    "FORMAL_SEEDS",
    "OUTPUT_ROOT",
    "TASK_MATRIX",
    "evaluate_research_test",
    "expected_research_test_evaluations",
]
