"""W0 FiLM GAMMA_040 固定 checkpoint 的复用 research-test 评价。"""

from __future__ import annotations

import fcntl
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
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.crd.model import build_crd_model
from resp_train.crd.tf_v1_research_test_data import (
    FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
    FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
    FROZEN_RESEARCH_TEST_CACHE_ROOT,
)
from resp_train.crd import w0_film_gamma_training as training
from resp_train.data.factory import build_window_data
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics


PROTOCOL = "w0-film-gamma-test-v1-20260919"
CONDITION = "GAMMA_040"
BASELINE = "GAMMA_050"
SEEDS = (20260811, 20260812, 20260813)
SELECTED_EPOCHS = (13, 30, 14)
BASELINE_EPOCHS = (13, 15, 14)
COUNT = 2310
SUBJECTS = 8
BATCH_SIZE = 128
TRAIN_LOCK_SHA = "2bf4b72a15111e1caa76b6bed19abbde3242edc451795176c307d160cc49368c"
TRAIN_SUMMARY_MANIFEST_SHA = "2df739a0a1a0d987a9c4733107230a5f74662ae78dd92c481d3bca92feae1b7b"
W0_AUDIT_SHA = "eb75cce11e1827b983f6540719e772c1e0288e8ffda06b7f3ae917bc2920d377"
ROW_ORDER_SHA = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"

CODE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/mnt/disk_code/marques/resp_reconstruction")
TRAIN_SUMMARY = SOURCE_ROOT / "runs/w0_film_gamma_training_v1/summary/20260919_115215_255816"
W0_AUDIT = SOURCE_ROOT / "runs/crd_tf_v1/research_test_summary/access_audit.csv"
LOCK_PATH = CODE_ROOT / "docs/experiments/w0_film_gamma_test_lock_20260919.json"
PROTOCOL_PATH = CODE_ROOT / "docs/experiments/w0_film_gamma_test_protocol_20260919.md"
TRAIN_RESULTS_PATH = CODE_ROOT / "docs/experiments/w0_film_gamma_training_results_20260919.md"
SCRIPT_PATH = CODE_ROOT / "scripts/eval_w0_film_gamma_test.py"
TEST_PATH = CODE_ROOT / "tests/test_w0_film_gamma_test.py"
OUTPUT_ROOT = SOURCE_ROOT / "runs/w0_film_gamma_test_v1"

IDENTITY_COLUMNS = ("dataset_row_id", "samp_id", "split")
TARGET_COLUMNS = (
    "whole_rr_target_eligible",
    "local_rr_target_eligible",
    "local_rr_target_eligible_windows",
    "joint_target_eligible",
    "envelope_spearman_target_eligible",
    "ibi_target_eligible",
)
ERROR_METRICS = training.ERROR_METRICS
PCC_METRIC = training.PCC_METRIC
PRIMARY_METRICS = (*ERROR_METRICS, PCC_METRIC)


def array_hash(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return __import__("hashlib").sha256(array.tobytes(order="C")).hexdigest()


def check_rows(
    rows: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    *,
    row_hash: str | None = None,
) -> None:
    if (
        len(rows) != COUNT
        or rows["dataset_row_id"].isna().any()
        or rows["dataset_row_id"].dtype.kind not in "iu"
        or rows["dataset_row_id"].duplicated().any()
        or rows["samp_id"].isna().any()
        or rows["samp_id"].nunique() != SUBJECTS
        or set(rows["split"].astype(str)) != {"test"}
    ):
        raise ValueError("W0 gamma test rows 数量、split、samp_id 或唯一身份错误")
    if row_hash is not None and array_hash(rows["dataset_row_id"].to_numpy()) != row_hash:
        raise ValueError("W0 gamma test row 顺序哈希漂移")
    if reference is not None:
        for key in IDENTITY_COLUMNS:
            if not np.array_equal(rows[key].to_numpy(), reference[key].to_numpy()):
                raise ValueError(f"W0 gamma test 与 W0 身份/顺序不一致: {key}")


def check_metrics(metrics: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    check_rows(metrics, reference)
    for key in TARGET_COLUMNS:
        if key not in metrics or not np.array_equal(metrics[key].to_numpy(), reference[key].to_numpy()):
            raise ValueError(f"W0 gamma test target eligibility 漂移: {key}")
    summary = summarize_task_metrics(metrics)
    for metric in PRIMARY_METRICS:
        mean_key, count_key = f"{metric}_mean", f"{metric}_n"
        if (
            mean_key not in summary
            or not np.isfinite(float(summary.iloc[0][mean_key]))
            or int(summary.iloc[0][count_key]) != COUNT
        ):
            raise FloatingPointError(f"W0 gamma test 五主指标必须完整有限: {metric}")
    return summary


def quality_flags(metrics: pd.DataFrame) -> dict[str, Any]:
    counts = {
        key: int(metrics[key].astype(bool).sum())
        for key in (
            "joint_prediction_degenerate",
            "envelope_spearman_prediction_degenerate",
        )
    }
    return {
        "prediction_degeneracy": counts,
        "quality_acceptance_passed": not any(counts.values()),
    }


def _source(
    files: dict[str, dict[str, Any]],
    path: Path,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(path).resolve()
    if expected is not None:
        training.verify_file(target, expected)
    item = training.identity(target)
    files[str(target)] = item
    return {"path": str(target), **item}


def _development_subjects(cfg: DictConfig) -> list[int]:
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    subjects: set[int] = set()
    for split, strategy, seed in (
        ("train", str(cfg.data.train_sample_strategy), int(cfg.data.train_sample_seed)),
        ("val", str(cfg.data.val_sample_strategy), int(cfg.data.val_sample_seed)),
    ):
        rows = filter_index(
            audited,
            cfg,
            split=split,
            max_windows=None,
            sample_strategy=strategy,
            sample_seed=seed,
        )
        subjects.update(map(int, rows["samp_id"]))
    return sorted(subjects)


def prepare_lock() -> Path:
    """只锁定 metadata/既有指标；不打开 test cache 数组或 test 波形。"""

    if LOCK_PATH.exists():
        raise FileExistsError(f"W0 gamma test lock 已存在: {LOCK_PATH}")
    train_lock, train_hash = training.load_lock()
    if train_hash != TRAIN_LOCK_SHA:
        raise RuntimeError("W0 gamma test training implementation identity 漂移")
    training.verify_file(
        TRAIN_SUMMARY / "artifact_manifest.json",
        json.loads((TRAIN_SUMMARY / "freeze_receipt.json").read_text(encoding="utf-8"))["manifest"],
    )
    if training.sha256_file(TRAIN_SUMMARY / "artifact_manifest.json") != TRAIN_SUMMARY_MANIFEST_SHA:
        raise RuntimeError("W0 gamma test training summary manifest 漂移")
    summary_manifest = json.loads(
        (TRAIN_SUMMARY / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in summary_manifest["files"].items():
        training.verify_file(TRAIN_SUMMARY / name, expected)
    decision = json.loads((TRAIN_SUMMARY / "decision_receipt.json").read_text(encoding="utf-8"))
    chosen = next(row for row in decision["decisions"] if row["condition"] == CONDITION)
    if (
        decision.get("matrix_complete") is not True
        or decision.get("formal_runs") != 6
        or chosen.get("quality_candidate") is not True
        or chosen.get("tolerance_aware_pareto") is not True
    ):
        raise RuntimeError("W0 gamma test validation-selected 候选身份不合格")
    if training.sha256_file(W0_AUDIT) != W0_AUDIT_SHA:
        raise RuntimeError("W0 gamma test 冻结 W0 audit identity 漂移")

    audit = pd.read_csv(W0_AUDIT)
    audit = audit.loc[audit["variant"].eq("crd_tf102_w")].sort_values("seed")
    if tuple(audit["seed"].astype(int)) != SEEDS:
        raise RuntimeError("W0 gamma test W0 audit seed 不完整")
    cache_manifest_path = FROZEN_RESEARCH_TEST_CACHE_ROOT / "cache_manifest.json"
    if training.sha256_file(cache_manifest_path) != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256:
        raise RuntimeError("W0 gamma test cache manifest identity 漂移")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    test_meta = cache_manifest["splits"]["test"]
    if (
        int(test_meta["count"]) != COUNT
        or int(test_meta["samp_id_count"]) != SUBJECTS
        or str(test_meta["row_ids_sha256"]) != ROW_ORDER_SHA
    ):
        raise RuntimeError("W0 gamma test cache split contract 漂移")

    files: dict[str, dict[str, Any]] = {}
    _source(files, training.LOCK_PATH)
    _source(files, TRAIN_SUMMARY / "artifact_manifest.json")
    _source(files, TRAIN_SUMMARY / "freeze_receipt.json")
    _source(files, TRAIN_SUMMARY / "decision_receipt.json")
    _source(files, TRAIN_SUMMARY / "run_matrix.csv")
    _source(files, W0_AUDIT)
    _source(files, cache_manifest_path)
    run_matrix = pd.read_csv(TRAIN_SUMMARY / "run_matrix.csv")
    run_matrix = run_matrix.loc[run_matrix["condition"].eq(CONDITION)].sort_values("seed")
    if tuple(run_matrix["seed"].astype(int)) != SEEDS:
        raise RuntimeError("W0 gamma test GAMMA_040 三 seed 来源不完整")

    entries, reference_rows = [], None
    for seed, epoch, baseline_epoch in zip(SEEDS, SELECTED_EPOCHS, BASELINE_EPOCHS, strict=True):
        row = run_matrix.loc[run_matrix["seed"].eq(seed)].iloc[0]
        run = Path(str(row["run_dir"])).resolve()
        receipt = training.validate_completed_run(run, lock_hash=TRAIN_LOCK_SHA)
        if (
            receipt["condition"] != CONDITION
            or int(receipt["seed"]) != seed
            or int(receipt["selected_epoch"]) != epoch
            or str(row["receipt_sha256"]) != training.sha256_file(run / training.RECEIPT_NAME)
        ):
            raise RuntimeError("W0 gamma test candidate receipt/epoch 漂移")
        cfg = training.load_resolved_training_config(run / "config.yaml")
        history = pd.read_csv(run / "train_history.csv")
        selected = int(history.loc[history["val_local_rr_mae"].idxmin(), "epoch"])
        if (
            len(history) != 80
            or int(history["optimizer_update"].iloc[-1]) != 6400
            or selected != epoch
            or float(cfg.model.film_gamma_coefficient) != 0.4
            or float(cfg.model.film_beta_coefficient) != 0.5
        ):
            raise RuntimeError("W0 gamma test checkpoint 不是冻结 validation-selected 训练结果")
        candidate = {
            "config": _source(files, run / "config.yaml", receipt["resolved_config"]),
            "history": _source(files, run / "train_history.csv", receipt["history"]),
            "checkpoint": _source(
                files,
                run / "checkpoint_best_local_rr.pt",
                receipt["best_checkpoint"],
            ),
            "receipt": _source(files, run / training.RECEIPT_NAME),
        }
        checkpoint = torch.load(
            candidate["checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        if int(checkpoint["epoch"]) != epoch or not training._tensors_finite(
            checkpoint["model_state_dict"]
        ):
            raise RuntimeError("W0 gamma test candidate checkpoint epoch/finite 漂移")

        baseline_entry = next(
            item for item in train_lock["baseline_runs"] if int(item["seed"]) == seed
        )
        audit_row = audit.loc[audit["seed"].eq(seed)].iloc[0]
        if (
            int(baseline_entry["selected_epoch"]) != baseline_epoch
            or int(audit_row["validation_selected_epoch"]) != baseline_epoch
            or str(audit_row["checkpoint_sha256"]) != baseline_entry["checkpoint"]["sha256"]
        ):
            raise RuntimeError("W0 gamma test GAMMA_050 baseline identity 漂移")
        baseline_run = Path(str(baseline_entry["run_dir"])).resolve()
        baseline_test = {}
        for filename, audit_key in (
            ("research_test_metrics.csv", "metrics_sha256"),
            ("research_test_metrics_summary.csv", "metrics_summary_sha256"),
            ("research_test_metrics_manifest.json", "evaluation_manifest_sha256"),
        ):
            path = baseline_run / filename
            if training.sha256_file(path) != str(audit_row[audit_key]):
                raise RuntimeError(f"W0 gamma test GAMMA_050 test source 漂移: {filename}")
            baseline_test[filename] = _source(files, path)
        reference = pd.read_csv(baseline_test["research_test_metrics.csv"]["path"])
        check_rows(reference, reference_rows, row_hash=ROW_ORDER_SHA)
        check_metrics(reference, reference)
        development = _development_subjects(cfg)
        if set(development) & set(map(int, reference["samp_id"])):
            raise RuntimeError("W0 gamma test 与 train/validation samp_id 交叉")
        reference_rows = reference[list(IDENTITY_COLUMNS)].copy()
        entries.append(
            {
                "seed": seed,
                "selected_epoch": epoch,
                "baseline_selected_epoch": baseline_epoch,
                "training_run": str(run),
                "training_config": OmegaConf.to_container(cfg, resolve=True),
                "candidate": candidate,
                "baseline_test": baseline_test,
                "development_samp_ids": development,
            }
        )

    cache_files = {
        name: cache_manifest["files"][name]
        for name in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy")
    }
    dataset_index = Path(cache_manifest["dataset_index"]).resolve()
    code_paths = sorted((CODE_ROOT / "resp_train").rglob("*.py"))
    code_paths.extend((SCRIPT_PATH, TEST_PATH, PROTOCOL_PATH, TRAIN_RESULTS_PATH))
    lock = {
        "protocol": PROTOCOL,
        "condition": CONDITION,
        "baseline": BASELINE,
        "seeds": list(SEEDS),
        "selected_epochs": list(SELECTED_EPOCHS),
        "baseline_selected_epochs": list(BASELINE_EPOCHS),
        "split": "test",
        "count": COUNT,
        "samp_id_count": SUBJECTS,
        "batch_size": BATCH_SIZE,
        "amp_dtype": "bfloat16",
        "include_test_only": False,
        "training_lock_sha256": TRAIN_LOCK_SHA,
        "training_summary_manifest_sha256": TRAIN_SUMMARY_MANIFEST_SHA,
        "entries": entries,
        "source_files": files,
        "cache_root": str(FROZEN_RESEARCH_TEST_CACHE_ROOT),
        "cache_files": cache_files,
        "cache_manifest_sha256": FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
        "dataset_index": {"path": str(dataset_index), **training.identity(dataset_index)},
        "row_order_sha256": ROW_ORDER_SHA,
        "preparation_git": training._git_state(require_clean=False),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_access": (
            "frozen configs, manifests, dataset metadata and prior W0 metrics; "
            "test signal/cache arrays not loaded"
        ),
        "code_files": {
            str(path.relative_to(CODE_ROOT)): training.identity(path)
            for path in sorted(set(code_paths))
        },
        "evidence_role": "reused research/development test evidence",
    }
    training.write_json(LOCK_PATH, lock)
    return LOCK_PATH


def load_lock() -> tuple[dict[str, Any], str]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(f"缺少 W0 gamma test lock: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or lock.get("condition") != CONDITION
        or lock.get("baseline") != BASELINE
        or tuple(lock.get("seeds", ())) != SEEDS
        or tuple(lock.get("selected_epochs", ())) != SELECTED_EPOCHS
        or tuple(lock.get("baseline_selected_epochs", ())) != BASELINE_EPOCHS
        or lock.get("split") != "test"
        or int(lock.get("count", -1)) != COUNT
        or int(lock.get("samp_id_count", -1)) != SUBJECTS
        or int(lock.get("batch_size", -1)) != BATCH_SIZE
        or lock.get("amp_dtype") != "bfloat16"
        or lock.get("include_test_only") is not False
        or lock.get("training_lock_sha256") != TRAIN_LOCK_SHA
        or lock.get("training_summary_manifest_sha256") != TRAIN_SUMMARY_MANIFEST_SHA
        or Path(lock.get("cache_root", "")).resolve()
        != FROZEN_RESEARCH_TEST_CACHE_ROOT.resolve()
        or lock.get("cache_manifest_sha256")
        != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256
        or lock.get("row_order_sha256") != ROW_ORDER_SHA
    ):
        raise ValueError("W0 gamma test lock 固定矩阵或来源身份漂移")
    if [
        (int(entry["seed"]), int(entry["selected_epoch"])) for entry in lock["entries"]
    ] != list(zip(SEEDS, SELECTED_EPOCHS, strict=True)):
        raise ValueError("W0 gamma test checkpoint allowlist 不完整")
    for relative, expected in lock["code_files"].items():
        training.verify_file(CODE_ROOT / relative, expected)
    return lock, training.sha256_file(LOCK_PATH)


@contextmanager
def phase_guard(parent: Path, lock_hash: str, *, completed_phase: str) -> Iterator[None]:
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / f".execution_{lock_hash}.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("W0 gamma test 相同身份正在运行") from exc
        try:
            for receipt in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((receipt.parent / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("implementation_lock_sha256") == lock_hash:
                    verify_attempt(receipt.parent, lock_hash=lock_hash, phase=completed_phase)
                    raise FileExistsError(f"W0 gamma test 相同身份已完成: {receipt.parent}")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def attempt(parent: Path, phase: str, lock_hash: str, *, seed: int | None) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    context = {
        "protocol": PROTOCOL,
        "phase": phase,
        "seed": seed,
        "split": "test",
        "condition": CONDITION,
        "baseline": BASELINE,
        "implementation_lock_sha256": lock_hash,
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    training.write_json(output / "lifecycle_started.json", {**context, "status": "running"})
    print(f"W0 gamma test attempt: {output}", flush=True)
    try:
        yield output
        training.write_json(
            output / "lifecycle_completed.json",
            {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
        )
        files = {
            str(path.relative_to(output)): training.identity(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        training.write_json(
            output / "manifest.json", {**context, "status": "completed", "files": files}
        )
        training.write_json(
            output / "freeze_receipt.json",
            {"protocol": PROTOCOL, "manifest": training.identity(output / "manifest.json")},
        )
    except BaseException as exc:
        training.write_json(
            output / "lifecycle_failed.json",
            {
                **context,
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def verify_attempt(output: Path, *, lock_hash: str, phase: str) -> dict[str, Any]:
    output = Path(output).resolve()
    if (output / "lifecycle_failed.json").exists():
        raise ValueError(f"W0 gamma test attempt 已失败: {output}")
    freeze = json.loads((output / "freeze_receipt.json").read_text(encoding="utf-8"))
    training.verify_file(output / "manifest.json", freeze["manifest"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("split") != "test"
        or manifest.get("condition") != CONDITION
        or manifest.get("baseline") != BASELINE
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
    ):
        raise ValueError("W0 gamma test attempt identity 不匹配")
    required = (
        {
            "lifecycle_completed.json",
            "summary_receipt.json",
            "seed_metrics.csv",
            "paired_seed_delta.csv",
            "three_seed_comparison.csv",
            "subject_summary.csv",
            "conclusions_zh.md",
        }
        if phase == "summary"
        else {
            "lifecycle_completed.json",
            "evaluation_receipt.json",
            "metrics.csv",
            "metrics_summary.csv",
            "test_rows.csv",
            "access_started.json",
            "access_receipt.json",
            "environment.json",
            "resolved_config.yaml",
            "implementation_lock.json",
        }
    )
    if not required.issubset(manifest["files"]):
        raise ValueError("W0 gamma test manifest 缺少必需产物")
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError("W0 gamma test manifest 路径越界")
        training.verify_file(path, expected)
    return manifest


def runtime_preflight(device: str) -> dict[str, Any]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None or not torch.cuda.is_available():
        raise RuntimeError("W0 gamma test 正式评价要求显式可用 cuda:<index>")
    git = training._git_state(require_clean=True)
    return {
        "device": str(resolved),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(resolved),
        "git": git,
    }


def evaluation_config(
    entry: Mapping[str, Any], device: str, cache_root: str
) -> tuple[DictConfig, DictConfig]:
    cfg = training.load_resolved_training_config(Path(entry["candidate"]["config"]["path"]))
    if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
        raise ValueError("W0 gamma test 训练配置与 allowlist 不一致")
    if (
        str(cfg.model.film_condition) != CONDITION
        or float(cfg.model.film_gamma_coefficient) != 0.4
        or float(cfg.model.film_beta_coefficient) != 0.5
        or int(cfg.training.seed) != int(entry["seed"])
        or int(cfg.model.initialization_seed) != int(entry["seed"])
        or str(cfg.data.test_split) != "test"
        or cfg.data.max_test_windows is not None
        or int(cfg.data.test_sample_seed) != 20260612
        or int(cfg.training.batch_size) != BATCH_SIZE
        or not bool(cfg.training.use_amp)
        or str(cfg.training.amp_dtype) != "bfloat16"
    ):
        raise ValueError("W0 gamma test config 必须使用冻结完整 test 与原生推理配置")
    cfg.training.device = device
    cfg.training.show_progress = False
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.data.tf_research_test_cache_path = str(cache_root)
    return cfg, data_cfg


def guarded_batches(loader: Any, rows: pd.DataFrame) -> Iterator[Mapping[str, Any]]:
    offset = 0
    for batch in loader:
        count = len(batch["x"])
        expected = rows.iloc[offset : offset + count]
        if count <= 0 or len(expected) != count or set(batch.get("tf", {})) != {"w"}:
            raise ValueError("W0 gamma test batch 数量或 W keys 不合格")
        for key in IDENTITY_COLUMNS:
            value = batch["meta"][key]
            actual = value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if not np.array_equal(actual, expected[key].to_numpy()):
                raise ValueError(f"W0 gamma test batch identity/order 错误: {key}")
        if tuple(batch["tf"]["w"].shape) != (count, 97, 360):
            raise ValueError("W0 gamma test W shape 错误")
        for tensor in (batch["x"], batch["target"], batch["tf"]["w"]):
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError("W0 gamma test input/target/cache 非有限")
        offset += count
        yield batch
    if offset != len(rows):
        raise ValueError("W0 gamma test loader 样本不完整")


def run_evaluation(seed: int, device: str = "cuda:0") -> Path:
    if int(seed) not in SEEDS:
        raise ValueError("W0 gamma test seed 不在冻结矩阵")
    lock, lock_hash = load_lock()
    entry = next(item for item in lock["entries"] if int(item["seed"]) == int(seed))
    parent = OUTPUT_ROOT / "evaluation" / f"seed_{int(seed)}"
    with phase_guard(parent, lock_hash, completed_phase="evaluation"):
        return _run_evaluation(int(seed), device, lock, lock_hash, entry, parent)


def _run_evaluation(
    seed: int,
    device: str,
    lock: dict[str, Any],
    lock_hash: str,
    entry: dict[str, Any],
    parent: Path,
) -> Path:
    with attempt(parent, "evaluation", lock_hash, seed=seed) as output:
        training.write_json(output / "environment.json", runtime_preflight(device))
        training.write_json(output / "implementation_lock.json", lock)
        training.write_json(
            output / "access_started.json",
            {
                "split": "test",
                "seed": seed,
                "count": COUNT,
                "checkpoint": entry["candidate"]["checkpoint"],
                "cache_root": lock["cache_root"],
                "dataset_index": lock["dataset_index"],
                "purpose": "GAMMA_040 fixed-checkpoint reused research-test evaluation",
            },
        )
        for path, expected in lock["source_files"].items():
            training.verify_file(Path(path), expected)
        cache_root = Path(lock["cache_root"])
        if training.sha256_file(cache_root / "cache_manifest.json") != lock[
            "cache_manifest_sha256"
        ]:
            raise RuntimeError("W0 gamma test cache manifest 漂移")
        for name, expected in lock["cache_files"].items():
            training.verify_file(cache_root / name, expected)
        training.verify_file(Path(lock["dataset_index"]["path"]), lock["dataset_index"])

        cfg, data_cfg = evaluation_config(entry, device, lock["cache_root"])
        OmegaConf.save(data_cfg, output / "resolved_config.yaml")
        checkpoint_path = Path(entry["candidate"]["checkpoint"]["path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config") != entry["training_config"]:
            raise ValueError("W0 gamma test checkpoint 完整训练配置 identity 漂移")
        _validate_checkpoint_config(checkpoint.get("config"), cfg)
        if int(checkpoint["epoch"]) != int(entry["selected_epoch"]):
            raise ValueError("W0 gamma test checkpoint epoch 漂移")
        if not training._tensors_finite(checkpoint["model_state_dict"]):
            raise FloatingPointError("W0 gamma test checkpoint 非有限")
        model = build_crd_model(cfg)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        reference = pd.read_csv(
            entry["baseline_test"]["research_test_metrics.csv"]["path"]
        )
        audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
        rows = filter_index(
            audited,
            cfg,
            split="test",
            max_windows=None,
            sample_strategy=str(cfg.data.test_sample_strategy),
            sample_seed=int(cfg.data.test_sample_seed),
        )
        check_rows(rows, reference, row_hash=lock["row_order_sha256"])
        if set(map(int, rows["samp_id"])) & set(entry["development_samp_ids"]):
            raise ValueError("W0 gamma test 与开发 split 的 samp_id 交叉")
        rows.to_csv(output / "test_rows.csv", index=False)
        data = build_window_data(
            data_cfg,
            split="test",
            max_windows=None,
            sample_strategy=str(cfg.data.test_sample_strategy),
            sample_seed=int(cfg.data.test_sample_seed),
            shuffle=False,
            audited=audited,
        )
        check_rows(data.rows, rows, row_hash=lock["row_order_sha256"])
        if len(data.dataset) != COUNT:
            raise ValueError("W0 gamma test dataset 缩小了样本集合")
        predictions = collect_predictions(
            model,
            guarded_batches(data.loader, rows),
            device=device,
            max_windows=COUNT,
            use_amp=True,
        )
        metrics = evaluate_task_predictions(
            predictions,
            cfg,
            include_test_only=False,
            method="w0_film_gamma040",
        )
        summary = check_metrics(metrics, reference)
        quality = quality_flags(metrics)
        metrics.insert(0, "seed", seed)
        metrics.insert(0, "condition", CONDITION)
        summary.insert(0, "seed", seed)
        summary.insert(0, "condition", CONDITION)
        summary.insert(0, "split", "test")
        summary.insert(0, "quality_acceptance_passed", quality["quality_acceptance_passed"])
        summary.insert(0, "selected_epoch", int(entry["selected_epoch"]))
        metrics.to_csv(output / "metrics.csv", index=False)
        summary.to_csv(output / "metrics_summary.csv", index=False)
        training.write_json(
            output / "access_receipt.json",
            {
                "split": "test",
                "seed": seed,
                "rows": COUNT,
                "samp_ids": SUBJECTS,
                "row_order_sha256": array_hash(rows["dataset_row_id"].to_numpy()),
                "cache_identity": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
                "cache_files_verified": lock["cache_files"],
                "checkpoint": entry["candidate"]["checkpoint"],
                "test_sample_seed": int(cfg.data.test_sample_seed),
                "evaluation_commit": training._git_state(require_clean=True)["commit"],
                "evidence_role": "reused research/development test evidence",
            },
        )
        training.write_json(
            output / "evaluation_receipt.json",
            {
                "protocol": PROTOCOL,
                "seed": seed,
                "split": "test",
                "condition": CONDITION,
                "selected_epoch": int(entry["selected_epoch"]),
                "checkpoint_sha256": entry["candidate"]["checkpoint"]["sha256"],
                "rows": COUNT,
                "samp_ids": SUBJECTS,
                "primary_finite": True,
                **quality,
                "target_eligibility_matches_w0": True,
                "row_order_sha256": lock["row_order_sha256"],
            },
        )
    return output


def _paired_tables(seed_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in SEEDS:
        baseline = seed_metrics.loc[
            seed_metrics["seed"].eq(seed) & seed_metrics["condition"].eq(BASELINE)
        ].iloc[0]
        candidate = seed_metrics.loc[
            seed_metrics["seed"].eq(seed) & seed_metrics["condition"].eq(CONDITION)
        ].iloc[0]
        for metric in PRIMARY_METRICS:
            base_value = float(baseline[f"{metric}_mean"])
            candidate_value = float(candidate[f"{metric}_mean"])
            rows.append(
                {
                    "split": "test",
                    "seed": seed,
                    "metric": metric,
                    "baseline_mean": base_value,
                    "candidate_mean": candidate_value,
                    "delta": (
                        100.0 * (candidate_value - base_value) / base_value
                        if metric in ERROR_METRICS
                        else candidate_value - base_value
                    ),
                    "delta_unit": "percent" if metric in ERROR_METRICS else "absolute",
                    "improved": bool(
                        candidate_value < base_value
                        if metric in ERROR_METRICS
                        else candidate_value > base_value
                    ),
                }
            )
    paired = pd.DataFrame(rows)
    aggregate_rows = []
    for metric, group in paired.groupby("metric", sort=True):
        values = group["delta"].to_numpy(dtype=np.float64)
        aggregate_rows.append(
            {
                "split": "test",
                "metric": metric,
                "baseline_seed_mean": float(group["baseline_mean"].mean()),
                "baseline_seed_sd": float(group["baseline_mean"].std(ddof=1)),
                "candidate_seed_mean": float(group["candidate_mean"].mean()),
                "candidate_seed_sd": float(group["candidate_mean"].std(ddof=1)),
                "paired_delta_mean": float(values.mean()),
                "paired_delta_sd": float(values.std(ddof=1)),
                "delta_unit": str(group["delta_unit"].iloc[0]),
                "improve_seed_count": int(group["improved"].sum()),
                "worse_seed_count": int((~group["improved"]).sum()),
                "n_windows_per_seed": COUNT,
            }
        )
    return paired, pd.DataFrame(aggregate_rows)


def _subject_summary(
    candidate_frames: Mapping[int, pd.DataFrame],
    baseline_frames: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        candidate = candidate_frames[seed].sort_values("dataset_row_id").reset_index(drop=True)
        baseline = baseline_frames[seed].sort_values("dataset_row_id").reset_index(drop=True)
        check_rows(candidate, baseline, row_hash=ROW_ORDER_SHA)
        for metric in PRIMARY_METRICS:
            for subject, indices in candidate.groupby("samp_id").groups.items():
                candidate_mean = float(pd.to_numeric(candidate.loc[indices, metric]).mean())
                baseline_mean = float(pd.to_numeric(baseline.loc[indices, metric]).mean())
                rows.append(
                    {
                        "split": "test",
                        "seed": seed,
                        "samp_id": int(subject),
                        "metric": metric,
                        "windows": int(len(indices)),
                        "baseline_mean": baseline_mean,
                        "candidate_mean": candidate_mean,
                        "delta": candidate_mean - baseline_mean,
                        "improved": bool(
                            candidate_mean < baseline_mean
                            if metric in ERROR_METRICS
                            else candidate_mean > baseline_mean
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _conclusion(aggregate: pd.DataFrame, subject: pd.DataFrame) -> str:
    local = aggregate.loc[aggregate["metric"].eq("local_rr_mae_bpm")].iloc[0]
    pcc = aggregate.loc[aggregate["metric"].eq(PCC_METRIC)].iloc[0]
    local_subjects = (
        subject.loc[subject["metric"].eq("local_rr_mae_bpm")]
        .groupby("samp_id")["delta"]
        .mean()
    )
    return "\n".join(
        [
            "# W0 FiLM GAMMA_040 research-test：描述性结论",
            "",
            f"- Local RR paired delta mean={local['paired_delta_mean']:.6g}%（负值改善），"
            f"improved seeds={int(local['improve_seed_count'])}/3。",
            f"- PCC paired delta mean={pcc['paired_delta_mean']:.6g}（正值改善），"
            f"improved seeds={int(pcc['improve_seed_count'])}/3。",
            f"- Local RR 三-seed主体均值改善={int((local_subjects < 0).sum())}/{len(local_subjects)}。",
            "- 证据来自复用 research/development test split，不是首次 held-out 确认。",
            "",
        ]
    )


def summarize(runs: Sequence[Path]) -> Path:
    lock, lock_hash = load_lock()
    parent = OUTPUT_ROOT / "summary"
    with phase_guard(parent, lock_hash, completed_phase="summary"):
        return _summarize(runs, lock, lock_hash, parent)


def _summarize(
    runs: Sequence[Path],
    lock: dict[str, Any],
    lock_hash: str,
    parent: Path,
) -> Path:
    if len(runs) != 3 or len({Path(path).resolve() for path in runs}) != 3:
        raise ValueError("W0 gamma test 汇总要求三个不同的完成 attempt")
    frames, sources, seeds = [], {}, set()
    candidate_frames: dict[int, pd.DataFrame] = {}
    baseline_frames: dict[int, pd.DataFrame] = {}
    for output_value in runs:
        output = Path(output_value).resolve()
        manifest = verify_attempt(output, lock_hash=lock_hash, phase="evaluation")
        seed = int(manifest["seed"])
        if seed not in SEEDS or seed in seeds:
            raise ValueError("W0 gamma test 汇总 seed 重复或越界")
        seeds.add(seed)
        entry = next(item for item in lock["entries"] if int(item["seed"]) == seed)
        receipt = json.loads((output / "evaluation_receipt.json").read_text(encoding="utf-8"))
        if (
            int(receipt["seed"]) != seed
            or int(receipt["selected_epoch"]) != int(entry["selected_epoch"])
            or receipt["checkpoint_sha256"] != entry["candidate"]["checkpoint"]["sha256"]
            or receipt["split"] != "test"
            or receipt["condition"] != CONDITION
            or int(receipt["rows"]) != COUNT
        ):
            raise ValueError("W0 gamma test 完成回执与 checkpoint allowlist 不一致")
        for item in entry["baseline_test"].values():
            training.verify_file(Path(item["path"]), item)
        baseline = pd.read_csv(
            entry["baseline_test"]["research_test_metrics.csv"]["path"]
        )
        candidate = pd.read_csv(output / "metrics.csv")
        if not candidate["seed"].eq(seed).all() or not candidate["condition"].eq(CONDITION).all():
            raise ValueError("W0 gamma test metrics seed/condition 不一致")
        check_rows(candidate, pd.read_csv(output / "test_rows.csv"), row_hash=ROW_ORDER_SHA)
        computed = check_metrics(candidate, baseline)
        quality = quality_flags(candidate)
        if any(receipt.get(key) != value for key, value in quality.items()):
            raise ValueError("W0 gamma test 质量回执与逐窗口结果不一致")
        candidate_summary = pd.read_csv(output / "metrics_summary.csv")
        baseline_summary = pd.read_csv(
            entry["baseline_test"]["research_test_metrics_summary.csv"]["path"]
        )
        baseline_computed = check_metrics(baseline, baseline)
        for frame, expected in (
            (candidate_summary, computed),
            (baseline_summary, baseline_computed),
        ):
            if len(frame) != 1:
                raise ValueError("W0 gamma test 每 seed summary 必须一行")
            for metric in PRIMARY_METRICS:
                if (
                    not np.isclose(
                        float(frame.iloc[0][f"{metric}_mean"]),
                        float(expected.iloc[0][f"{metric}_mean"]),
                        atol=1e-12,
                        rtol=0.0,
                    )
                    or int(frame.iloc[0][f"{metric}_n"]) != COUNT
                ):
                    raise ValueError("W0 gamma test summary 数值或分母不一致")
        baseline_summary.insert(0, "seed", seed)
        baseline_summary.insert(0, "condition", BASELINE)
        baseline_summary.insert(0, "split", "test")
        baseline_summary.insert(0, "selected_epoch", int(entry["baseline_selected_epoch"]))
        baseline_summary.insert(
            0, "quality_acceptance_passed", quality_flags(baseline)["quality_acceptance_passed"]
        )
        candidate_summary.insert(0, "source_sha256", training.sha256_file(output / "metrics.csv"))
        baseline_summary.insert(
            0,
            "source_sha256",
            entry["baseline_test"]["research_test_metrics.csv"]["sha256"],
        )
        frames.extend((baseline_summary, candidate_summary))
        candidate_frames[seed], baseline_frames[seed] = candidate, baseline
        sources[str(seed)] = {
            "path": str(output),
            "manifest": training.identity(output / "manifest.json"),
            "selected_epoch": int(entry["selected_epoch"]),
        }
    if seeds != set(SEEDS):
        raise ValueError("W0 gamma test 汇总 seed 不完整")
    combined = pd.concat(frames, ignore_index=True)
    paired, aggregate = _paired_tables(combined)
    subject = _subject_summary(candidate_frames, baseline_frames)
    with attempt(parent, "summary", lock_hash, seed=None) as output:
        combined.to_csv(output / "seed_metrics.csv", index=False)
        paired.to_csv(output / "paired_seed_delta.csv", index=False)
        aggregate.to_csv(output / "three_seed_comparison.csv", index=False)
        subject.to_csv(output / "subject_summary.csv", index=False)
        (output / "conclusions_zh.md").write_text(
            _conclusion(aggregate, subject), encoding="utf-8"
        )
        training.write_json(
            output / "summary_receipt.json",
            {
                "protocol": PROTOCOL,
                "split": "test",
                "seeds": list(SEEDS),
                "rows_per_seed": COUNT,
                "new_metric_rows": len(SEEDS) * COUNT,
                "source_runs": sources,
                "baseline_test_sources": {
                    str(entry["seed"]): entry["baseline_test"] for entry in lock["entries"]
                },
                "delta_definition": (
                    "error=100*(GAMMA_040-GAMMA_050)/GAMMA_050; "
                    "PCC=GAMMA_040-GAMMA_050"
                ),
                "evidence_role": "fixed validation-selected checkpoints; reused research/development test",
            },
        )
    return output
