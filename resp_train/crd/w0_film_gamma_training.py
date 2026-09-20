"""W0 FiLM gamma 系数训练验证的冻结配置、闸门与产物审计。"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import (
    CRD_TF_PROTOCOL_VERSION,
    FORMAL_SEEDS,
    _validate_crd_config,
)
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.model import build_crd_model
from resp_train.crd.tf_v1_model import trainable_parameter_count
from resp_train.crd.training import _prepare_batch
from resp_train.data.factory import build_tho_data
from resp_train.losses.task import RespirationTaskLoss
from resp_train.utils.run import create_run_dir


PROTOCOL = "w0-film-gamma-training-v1-20260918"
STAGE = "tf_film_gamma"
VARIANT = "crd_tf102_w"
CONDITIONS = {"GAMMA_030": 0.3, "GAMMA_040": 0.4}
BASELINE_CONDITION = "GAMMA_050"
BETA_COEFFICIENT = 0.5
TRAINABLE_PARAMETERS = 1_219_850
WINDOW_COUNT = 2675
FORMAL_EPOCHS = 80
FORMAL_UPDATES = 6400
RECEIPT_NAME = "gamma_training_receipt.json"

CODE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/mnt/disk_code/marques/resp_reconstruction")
CONFIG_ROOT = CODE_ROOT / "configs/w0_film_gamma_training"
OUTPUT_ROOT = SOURCE_ROOT / "runs/w0_film_gamma_training_v1"
LOCK_PATH = CODE_ROOT / "docs/experiments/w0_film_gamma_training_implementation_lock_20260918.json"
PROTOCOL_PATH = CODE_ROOT / "docs/experiments/w0_film_gamma_training_protocol_20260918.md"
RESULTS_PATH = CODE_ROOT / "docs/experiments/w0_film_local_sensitivity_results_20260918.md"
SCRIPT_PATH = CODE_ROOT / "scripts/run_w0_film_gamma_training.py"
TEST_PATH = CODE_ROOT / "tests/test_w0_film_gamma_training.py"

CONFIGS = {
    ("acceptance", "GAMMA_030"): CONFIG_ROOT / "gamma030_acceptance.yaml",
    ("formal", "GAMMA_030"): CONFIG_ROOT / "gamma030_formal.yaml",
    ("formal", "GAMMA_040"): CONFIG_ROOT / "gamma040_formal.yaml",
}

BASELINE_RUNS = {
    20260811: SOURCE_ROOT
    / "runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861",
    20260812: SOURCE_ROOT
    / "runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130",
    20260813: SOURCE_ROOT
    / "runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006",
}

SOURCE_LOCK = CODE_ROOT / "docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json"
SOURCE_LOCK_SHA256 = "6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6"
LOCAL_SUMMARY = SOURCE_ROOT / (
    "runs/w0_film_local_sensitivity_v1/summary/"
    "summary_7b11a15d39e6_20260918T092051Z_99a66682abe9"
)
LOCAL_FINAL = SOURCE_ROOT / (
    "runs/w0_film_local_sensitivity_v1/final/"
    "final_7b11a15d39e6_20260918T093454Z_78ad894bc739"
)
BEHAVIOR_SUMMARY = SOURCE_ROOT / (
    "runs/w0_cwt_film_behavior_v1/summary/"
    "summary_c4bfc1907266_20260918T055139Z_773b7065a629"
)
QUALITY_SOURCE = BEHAVIOR_SUMMARY / "window_statistics.csv"

ERROR_METRICS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
)
PCC_METRIC = "lag_aware_signed_pcc"
PRIMARY_METRICS = (*ERROR_METRICS, PCC_METRIC)
ELIGIBILITY = {
    "whole_rr_abs_error_bpm": "whole_rr_target_eligible",
    "local_rr_mae_bpm": "local_rr_target_eligible",
    "lag_aware_signed_pcc": "joint_target_eligible",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def verify_file(path: Path, expected: Mapping[str, Any]) -> None:
    target = Path(path)
    expected_identity = {key: expected[key] for key in ("size_bytes", "sha256")}
    if not target.is_file() or identity(target) != expected_identity:
        raise RuntimeError(f"W0 gamma training 文件身份漂移: {target}")


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _git_state(*, require_clean: bool) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=CODE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("W0 gamma training 必须从干净 Git commit 启动")
    return {"commit": commit, "status_porcelain": status, "dirty": bool(status)}


def _load_one_level_config(path: Path, *, overrides: Iterable[str] = ()) -> DictConfig:
    path = Path(path)
    cfg = OmegaConf.load(path)
    base_reference = cfg.pop("_base_", None)
    if base_reference is not None:
        base_path = (path.parent / str(base_reference)).resolve()
        if base_path.parent != path.resolve().parent or not base_path.is_file():
            raise ValueError("W0 gamma training _base_ 必须是同目录文件")
        base = OmegaConf.load(base_path)
        if "_base_" in base:
            raise ValueError("W0 gamma training 配置只允许一层继承")
        cfg = OmegaConf.merge(base, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def _canonical_config(cfg: DictConfig) -> DictConfig:
    """移除本实验字段后，复用原 W0 冻结 schema 做完整合同校验。"""

    canonical = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key in ("film_condition", "film_gamma_coefficient", "film_beta_coefficient"):
        del canonical.model[key]
    canonical.protocol.name = CRD_TF_PROTOCOL_VERSION
    canonical.protocol.stage = "tf"
    role = str(cfg.protocol.run_role)
    canonical.protocol.execution_gate = (
        "p3_cuda_acceptance" if role == "acceptance" else "p4_formal"
    )
    _validate_crd_config(canonical)
    return canonical


def validate_training_config(cfg: DictConfig) -> None:
    role = str(cfg.protocol.run_role)
    condition = str(cfg.model.film_condition)
    if str(cfg.protocol.name) != PROTOCOL or str(cfg.protocol.stage) != STAGE:
        raise ValueError("W0 gamma training protocol/stage 漂移")
    if role not in {"acceptance", "formal"}:
        raise ValueError("W0 gamma training 只接受 acceptance/formal")
    expected_gate = "film_gamma_cuda_acceptance" if role == "acceptance" else "film_gamma_formal"
    if str(cfg.protocol.execution_gate) != expected_gate:
        raise ValueError("W0 gamma training execution gate 漂移")
    if str(cfg.model.variant) != VARIANT or list(cfg.model.tf_representations) != ["w"]:
        raise ValueError("W0 gamma training 必须保持原生 crd_tf102_w/full-12V")
    if condition not in CONDITIONS:
        raise ValueError(f"W0 gamma training 未注册 condition={condition!r}")
    gamma = float(cfg.model.film_gamma_coefficient)
    beta = float(cfg.model.film_beta_coefficient)
    if gamma != CONDITIONS[condition] or beta != BETA_COEFFICIENT:
        raise ValueError("W0 gamma training coefficient 与 condition 不一致")
    if str(cfg.training.device) != "cuda:0":
        raise ValueError("W0 gamma training 固定 device=cuda:0")
    seed = int(cfg.training.seed)
    if int(cfg.model.initialization_seed) != seed:
        raise ValueError("W0 gamma training 初始化 seed 漂移")
    if role == "acceptance":
        if condition != "GAMMA_030" or seed != FORMAL_SEEDS[0]:
            raise ValueError("W0 gamma acceptance 固定 GAMMA_030/seed 20260811")
        expected_root = OUTPUT_ROOT / "acceptance/gamma030/seed_20260811"
    else:
        if seed not in FORMAL_SEEDS:
            raise ValueError(f"W0 gamma formal seed 只允许 {FORMAL_SEEDS}")
        expected_root = OUTPUT_ROOT / f"formal/{condition.lower().replace('_', '')}/seed_{seed}"
    if Path(str(cfg.outputs.run_root)).resolve() != expected_root.resolve():
        raise ValueError("W0 gamma training output root 漂移")
    _canonical_config(cfg)


def load_training_config(*, role: str, condition: str, seed: int | None = None) -> DictConfig:
    key = (str(role), str(condition))
    if key not in CONFIGS:
        raise ValueError(f"W0 gamma training 未注册 config={key}")
    overrides = (f"training.seed={int(seed)}",) if seed is not None else ()
    cfg = _load_one_level_config(CONFIGS[key], overrides=overrides)
    validate_training_config(cfg)
    return cfg


def load_resolved_training_config(path: Path) -> DictConfig:
    cfg = OmegaConf.load(Path(path))
    if "_base_" in cfg:
        raise ValueError("resolved W0 gamma config 不得再包含 _base_")
    OmegaConf.resolve(cfg)
    validate_training_config(cfg)
    return cfg


def model_contract(cfg: DictConfig) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "condition": str(cfg.model.film_condition),
        "variant": VARIANT,
        "gamma_coefficient": float(cfg.model.film_gamma_coefficient),
        "beta_coefficient": float(cfg.model.film_beta_coefficient),
        "seed": int(cfg.training.seed),
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "research_test_used": False,
    }


class W0FilmGammaExperiment(CRDExperiment):
    """只改变 W0 FiLM gamma/beta 系数，训练和评价复用冻结 CRD 流程。"""

    def _build_model(self) -> torch.nn.Module:
        return build_crd_model(self.cfg)


def _baseline_source_files() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    for seed, run in BASELINE_RUNS.items():
        entry: dict[str, Any] = {"seed": int(seed), "run_dir": str(run)}
        for key, filename in (
            ("checkpoint", "checkpoint_best_local_rr.pt"),
            ("config", "config.yaml"),
            ("manifest", "run_manifest.json"),
            ("history", "train_history.csv"),
            ("metrics", "metrics.csv"),
            ("metrics_summary", "metrics_summary.csv"),
        ):
            path = run / filename
            item = identity(path)
            entry[key] = item
            files[str(path)] = item
        checkpoint = torch.load(run / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False)
        entry["selected_epoch"] = int(checkpoint["epoch"])
        rows.append(entry)
    return rows, files


def prepare_lock() -> Path:
    if LOCK_PATH.exists():
        raise FileExistsError(f"W0 gamma training implementation lock 已存在: {LOCK_PATH}")
    if sha256_file(SOURCE_LOCK) != SOURCE_LOCK_SHA256:
        raise RuntimeError("W0 gamma training W0 source lock 漂移")
    baseline, source_files = _baseline_source_files()
    for path in (
        SOURCE_LOCK,
        LOCAL_SUMMARY / "artifact_manifest.json",
        LOCAL_FINAL / "artifact_manifest.json",
        BEHAVIOR_SUMMARY / "artifact_manifest.json",
        QUALITY_SOURCE,
    ):
        source_files[str(path)] = identity(path)
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    cache_manifest = SOURCE_ROOT / str(source_lock["cache_lock"]["manifest"]["path"])
    cache_payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    dataset_index = Path(cache_payload["dataset_index"]).resolve()
    source_files[str(cache_manifest)] = identity(cache_manifest)
    source_files[str(dataset_index)] = identity(dataset_index)

    code_paths = sorted((CODE_ROOT / "resp_train").rglob("*.py"))
    code_paths.extend(sorted(CONFIG_ROOT.glob("*.yaml")))
    code_paths.extend((PROTOCOL_PATH, RESULTS_PATH, SCRIPT_PATH, TEST_PATH))
    lock = {
        "protocol": PROTOCOL,
        "status": "implementation_locked_acceptance_and_formal_pending",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": _git_state(require_clean=False),
        "code_root": str(CODE_ROOT),
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "conditions": {
            "GAMMA_030": {"gamma_coefficient": 0.3, "beta_coefficient": 0.5, "new_runs": 3},
            "GAMMA_040": {"gamma_coefficient": 0.4, "beta_coefficient": 0.5, "new_runs": 3},
            "GAMMA_050": {"gamma_coefficient": 0.5, "beta_coefficient": 0.5, "new_runs": 0},
        },
        "seeds": list(FORMAL_SEEDS),
        "formal_epochs": FORMAL_EPOCHS,
        "formal_updates": FORMAL_UPDATES,
        "windows_per_condition": WINDOW_COUNT,
        "baseline_runs": baseline,
        "cache_lock": source_lock["cache_lock"],
        "dataset_index": {"path": str(dataset_index), **identity(dataset_index)},
        "local_sensitivity_summary": str(LOCAL_SUMMARY),
        "local_sensitivity_final": str(LOCAL_FINAL),
        "source_files": source_files,
        "code_files": {
            str(path.relative_to(CODE_ROOT)): identity(path)
            for path in sorted(set(code_paths))
        },
        "research_test_used": False,
    }
    write_json(LOCK_PATH, lock)
    return LOCK_PATH


def load_lock() -> tuple[dict[str, Any], str]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(f"缺少 W0 gamma training implementation lock: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or tuple(lock.get("seeds", ())) != FORMAL_SEEDS
        or int(lock.get("formal_epochs", -1)) != FORMAL_EPOCHS
        or int(lock.get("formal_updates", -1)) != FORMAL_UPDATES
        or int(lock.get("windows_per_condition", -1)) != WINDOW_COUNT
        or lock.get("research_test_used") is not False
    ):
        raise ValueError("W0 gamma training implementation lock 合同漂移")
    for path, expected in lock["source_files"].items():
        verify_file(Path(path), expected)
    for relative, expected in lock["code_files"].items():
        verify_file(CODE_ROOT / relative, expected)
    return lock, sha256_file(LOCK_PATH)


def _existing_runs(parent: Path) -> list[Path]:
    return sorted(path for path in parent.glob("20*") if path.is_dir()) if parent.is_dir() else []


def _acceptance_receipt(lock_hash: str, commit: str) -> dict[str, Any]:
    parent = OUTPUT_ROOT / "acceptance/gamma030/seed_20260811"
    runs = _existing_runs(parent)
    if len(runs) != 1:
        raise RuntimeError(f"W0 gamma formal 要求唯一 acceptance run，当前数量={len(runs)}")
    receipt_path = runs[0] / RECEIPT_NAME
    if not receipt_path.is_file():
        raise RuntimeError(f"W0 gamma acceptance 缺少 receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "role": "acceptance",
        "condition": "GAMMA_030",
        "seed": FORMAL_SEEDS[0],
        "implementation_lock_sha256": lock_hash,
        "git_commit": commit,
        "git_dirty": False,
        "status": "passed",
        "gamma_coefficient": 0.3,
        "beta_coefficient": 0.5,
        "all_history_finite": True,
        "all_checkpoint_finite": True,
        "all_optimizer_finite": True,
        "all_primary_finite": True,
        "prediction_degeneracy_count": 0,
        "input_gradient_finite": True,
        "input_gradient_nonzero": True,
        "parameter_gradients_finite": True,
        "parameter_gradient_nonzero_count_positive": True,
        "research_test_used": False,
    }
    mismatched = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"W0 gamma acceptance receipt 不合格: {mismatched}")
    return receipt


def validate_training_preflight(cfg: DictConfig) -> tuple[str, str]:
    validate_training_config(cfg)
    _, lock_hash = load_lock()
    git = _git_state(require_clean=True)
    commit = str(git["commit"])
    parent = Path(str(cfg.outputs.run_root))
    existing = _existing_runs(parent)
    if existing:
        raise RuntimeError(f"W0 gamma training 目标已存在，拒绝重复运行: {existing}")
    if str(cfg.protocol.run_role) == "formal":
        _acceptance_receipt(lock_hash, commit)
    return commit, lock_hash


def _gradient_probe(cfg: DictConfig) -> dict[str, Any]:
    device = torch.device(str(cfg.training.device))
    data = build_tho_data(cfg)
    model = W0FilmGammaExperiment(cfg)._build_model().to(device).train()
    parameter_count = trainable_parameter_count(model)
    if parameter_count != TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"W0 gamma acceptance 参数量漂移: {parameter_count} != {TRAINABLE_PARAMETERS}"
        )
    loss_fn = RespirationTaskLoss(cfg).to(device)
    sensor, target, _, tf = _prepare_batch(next(iter(data.train.loader)), device, non_blocking=True)
    sensor = sensor[:1].detach().requires_grad_(True)
    target = target[:1]
    if tf is None:
        raise RuntimeError("W0 gamma acceptance 缺少 W cache")
    tf = {key: value[:1] for key, value in tf.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(sensor, tf=tf)
    with torch.amp.autocast("cuda", enabled=False):
        loss, _ = loss_fn(prediction, target.float())
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("W0 gamma acceptance loss 非有限")
    loss.backward()
    input_gradient = sensor.grad
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if input_gradient is None or not gradients:
        raise RuntimeError("W0 gamma acceptance 未产生梯度")
    output = {
        "probe_batch_size": 1,
        "trainable_parameters_observed": parameter_count,
        "loss": float(loss.detach().cpu()),
        "waveform_shape": list(prediction["waveform"].shape),
        "waveform_10hz_shape": list(prediction["waveform_10hz"].shape),
        "input_gradient_finite": bool(torch.isfinite(input_gradient).all()),
        "input_gradient_nonzero": bool(torch.count_nonzero(input_gradient)),
        "parameter_gradients_finite": all(bool(torch.isfinite(value).all()) for value in gradients),
        "parameter_gradient_nonzero_count": sum(bool(torch.count_nonzero(value)) for value in gradients),
    }
    output["parameter_gradient_nonzero_count_positive"] = (
        int(output["parameter_gradient_nonzero_count"]) > 0
    )
    del model, loss_fn, data
    torch.cuda.empty_cache()
    if not all(
        bool(output[key])
        for key in (
            "input_gradient_finite",
            "input_gradient_nonzero",
            "parameter_gradients_finite",
            "parameter_gradient_nonzero_count_positive",
        )
    ):
        raise RuntimeError(f"W0 gamma acceptance 梯度检查失败: {output}")
    return output


def _tensors_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, Mapping):
        return all(_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tensors_finite(item) for item in value)
    return True


def _primary_values_finite(metrics: pd.DataFrame) -> bool:
    contracts = (
        ("whole_rr_abs_error_bpm", "whole_rr_target_eligible"),
        ("local_rr_mae_bpm", "local_rr_target_eligible"),
        ("lag_aware_signed_pcc", "joint_target_eligible"),
    )
    for value_column, eligible_column in contracts:
        eligible = metrics[eligible_column].astype(bool).to_numpy()
        values = pd.to_numeric(metrics.loc[eligible, value_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if values.size == 0 or not np.isfinite(values).all():
            return False
    envelope = metrics[
        ["envelope_trajectory_mae", "global_envelope_modulation_error"]
    ].to_numpy(dtype=np.float64)
    return bool(np.isfinite(envelope).all())


def _write_training_receipt(
    run_dir: Path,
    cfg: DictConfig,
    *,
    commit: str,
    lock_hash: str,
    gradient_probe: Mapping[str, Any] | None,
) -> Path:
    role = str(cfg.protocol.run_role)
    history = pd.read_csv(run_dir / "train_history.csv")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    checkpoint = torch.load(
        run_dir / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False
    )
    final_checkpoint = torch.load(
        run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    replay_cfg = load_resolved_training_config(run_dir / "config.yaml")
    if model_contract(replay_cfg) != model_contract(cfg):
        raise RuntimeError("W0 gamma training resolved config 无法回放原训练身份")
    numeric_history = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    expected_epochs = 1 if role == "acceptance" else FORMAL_EPOCHS
    expected_updates = 1 if role == "acceptance" else FORMAL_UPDATES
    expected_windows = 32 if role == "acceptance" else WINDOW_COUNT
    if len(history) != expected_epochs or int(history["optimizer_update"].iloc[-1]) != expected_updates:
        raise RuntimeError("W0 gamma training epoch/update 数不完整")
    if len(metrics) != expected_windows or metrics["dataset_row_id"].duplicated().any():
        raise RuntimeError("W0 gamma training validation 窗口不完整")
    if metrics["joint_prediction_degenerate"].astype(bool).any():
        raise RuntimeError("W0 gamma training 出现 prediction degeneracy")
    if not np.isfinite(numeric_history).all() or not _primary_values_finite(metrics):
        raise FloatingPointError("W0 gamma training history/primary 非有限")
    if manifest.get("git_commit") != commit or manifest.get("git_dirty") is not False:
        raise RuntimeError("W0 gamma training 起止 Git identity 不一致")
    if manifest.get("protocol") != PROTOCOL or manifest.get("stage") != STAGE:
        raise RuntimeError("W0 gamma training run manifest 身份漂移")
    receipt = {
        **model_contract(cfg),
        "role": role,
        "status": "passed",
        "implementation_lock_sha256": lock_hash,
        "git_commit": commit,
        "git_dirty": False,
        "epochs": expected_epochs,
        "optimizer_updates": expected_updates,
        "selected_epoch": int(checkpoint["epoch"]),
        "validation_windows": expected_windows,
        "all_history_finite": True,
        "all_checkpoint_finite": _tensors_finite(checkpoint),
        "all_optimizer_finite": _tensors_finite(final_checkpoint.get("optimizer_state_dict", {})),
        "all_primary_finite": True,
        "prediction_degeneracy_count": 0,
        "resolved_config": identity(run_dir / "config.yaml"),
        "best_checkpoint": identity(run_dir / "checkpoint_best_local_rr.pt"),
        "final_checkpoint": identity(run_dir / "checkpoint_final.pt"),
        "history": identity(run_dir / "train_history.csv"),
        "metrics": identity(run_dir / "metrics.csv"),
        "metrics_summary": identity(run_dir / "metrics_summary.csv"),
        **(dict(gradient_probe) if gradient_probe is not None else {}),
    }
    for key in ("all_checkpoint_finite", "all_optimizer_finite"):
        if receipt[key] is not True:
            raise FloatingPointError(f"W0 gamma training {key} 未通过")
    path = run_dir / RECEIPT_NAME
    write_json(path, receipt)
    return path


def run_training(*, role: str, condition: str, seed: int | None = None) -> Path:
    cfg = load_training_config(role=role, condition=condition, seed=seed)
    commit, lock_hash = validate_training_preflight(cfg)
    probe = _gradient_probe(cfg) if role == "acceptance" else None
    experiment = W0FilmGammaExperiment(cfg)
    run_dir = experiment.train()
    _write_training_receipt(
        run_dir,
        cfg,
        commit=commit,
        lock_hash=lock_hash,
        gradient_probe=probe,
    )
    return run_dir


def validate_completed_run(path: Path, *, lock_hash: str) -> dict[str, Any]:
    run = Path(path).resolve()
    receipt_path = run / RECEIPT_NAME
    if not receipt_path.is_file():
        raise ValueError(f"W0 gamma training run 缺少 receipt: {run}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("protocol") != PROTOCOL
        or receipt.get("role") != "formal"
        or receipt.get("status") != "passed"
        or receipt.get("implementation_lock_sha256") != lock_hash
        or receipt.get("research_test_used") is not False
        or int(receipt.get("epochs", -1)) != FORMAL_EPOCHS
        or int(receipt.get("optimizer_updates", -1)) != FORMAL_UPDATES
        or int(receipt.get("validation_windows", -1)) != WINDOW_COUNT
    ):
        raise ValueError(f"W0 gamma training formal receipt 不合格: {run}")
    for key, filename in (
        ("resolved_config", "config.yaml"),
        ("best_checkpoint", "checkpoint_best_local_rr.pt"),
        ("final_checkpoint", "checkpoint_final.pt"),
        ("history", "train_history.csv"),
        ("metrics", "metrics.csv"),
        ("metrics_summary", "metrics_summary.csv"),
    ):
        verify_file(run / filename, receipt[key])
    return receipt


def _paired_window_rows(
    *,
    condition: str,
    seed: int,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    candidate = candidate.sort_values("dataset_row_id").reset_index(drop=True)
    baseline = baseline.sort_values("dataset_row_id").reset_index(drop=True)
    quality = quality.sort_values("dataset_row_id").reset_index(drop=True)
    for column in ("dataset_row_id", "samp_id", "split"):
        if not np.array_equal(candidate[column].to_numpy(), baseline[column].to_numpy()):
            raise RuntimeError(f"W0 gamma summary candidate/baseline identity 漂移: {column}")
    if not np.array_equal(candidate["dataset_row_id"].to_numpy(), quality["dataset_row_id"].to_numpy()):
        raise RuntimeError("W0 gamma summary quality row identity 漂移")
    if not np.array_equal(candidate["samp_id"].to_numpy(), quality["samp_id"].to_numpy()):
        raise RuntimeError("W0 gamma summary quality subject identity 漂移")
    for column in (*ELIGIBILITY.values(), "envelope_target_stratum"):
        if not np.array_equal(candidate[column].to_numpy(), baseline[column].to_numpy()):
            raise RuntimeError(f"W0 gamma training 改变 target identity: {condition}/{seed}/{column}")

    frames = []
    for metric in PRIMARY_METRICS:
        candidate_values = pd.to_numeric(candidate[metric], errors="coerce").to_numpy(dtype=np.float64)
        baseline_values = pd.to_numeric(baseline[metric], errors="coerce").to_numpy(dtype=np.float64)
        paired = np.isfinite(candidate_values) & np.isfinite(baseline_values)
        if metric in ELIGIBILITY:
            expected = candidate[ELIGIBILITY[metric]].astype(bool).to_numpy()
            if not np.array_equal(paired, expected):
                raise RuntimeError(f"W0 gamma summary finite/eligibility 不一致: {condition}/{seed}/{metric}")
        if not paired.any():
            raise RuntimeError(f"W0 gamma summary 无有效 paired windows: {condition}/{seed}/{metric}")
        frame = pd.DataFrame(
            {
                "condition": condition,
                "seed": int(seed),
                "dataset_row_id": candidate.loc[paired, "dataset_row_id"].to_numpy(),
                "samp_id": candidate.loc[paired, "samp_id"].to_numpy(),
                "target_stratum": candidate.loc[paired, "envelope_target_stratum"].to_numpy(),
                "quality_level": quality.loc[paired, "waveform_confidence_level"].to_numpy(),
                "metric": metric,
                "baseline_value": baseline_values[paired],
                "candidate_value": candidate_values[paired],
                "delta": candidate_values[paired] - baseline_values[paired],
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _seed_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, seed, metric), group in paired.groupby(
        ["condition", "seed", "metric"], sort=True
    ):
        baseline_mean = float(group["baseline_value"].mean())
        candidate_mean = float(group["candidate_value"].mean())
        delta_mean = float(group["delta"].mean())
        rows.append(
            {
                "condition": condition,
                "seed": int(seed),
                "metric": metric,
                "windows": int(len(group)),
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "delta_mean": delta_mean,
                "relative_delta_percent": (
                    100.0 * delta_mean / baseline_mean if metric in ERROR_METRICS else math.nan
                ),
                "improved": bool(delta_mean < 0.0 if metric in ERROR_METRICS else delta_mean > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _candidate_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, metric), group in seed_summary.groupby(["condition", "metric"], sort=True):
        baseline = float(group["baseline_mean"].mean())
        candidate = float(group["candidate_mean"].mean())
        delta = candidate - baseline
        rows.append(
            {
                "condition": condition,
                "metric": metric,
                "baseline_seed_mean": baseline,
                "candidate_seed_mean": candidate,
                "delta_seed_mean": delta,
                "delta_seed_sd": float(group["delta_mean"].std(ddof=1)),
                "relative_delta_percent": (
                    100.0 * delta / baseline if metric in ERROR_METRICS else math.nan
                ),
                "improve_seed_count": int(group["improved"].sum()),
                "worse_seed_count": int((~group["improved"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _stratum_summary(paired: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for scope, column in (
        ("target_stratum", "target_stratum"),
        ("quality_level", "quality_level"),
        ("samp_id", "samp_id"),
    ):
        grouped = (
            paired.groupby(["condition", "seed", "metric", column], sort=True)
            .agg(windows=("delta", "size"), delta_mean=("delta", "mean"))
            .reset_index()
            .rename(columns={column: "level"})
        )
        grouped.insert(2, "scope", scope)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def _candidate_decisions(candidate: pd.DataFrame) -> list[dict[str, Any]]:
    decisions = []
    for condition, group in candidate.groupby("condition", sort=True):
        indexed = group.set_index("metric")
        substantive = any(
            float(indexed.loc[metric, "relative_delta_percent"]) <= -0.5
            and int(indexed.loc[metric, "improve_seed_count"]) >= 2
            for metric in ERROR_METRICS
        ) or (
            float(indexed.loc[PCC_METRIC, "delta_seed_mean"]) >= 0.002
            and int(indexed.loc[PCC_METRIC, "improve_seed_count"]) >= 2
        )
        guardrails = (
            float(indexed.loc["local_rr_mae_bpm", "relative_delta_percent"]) <= 0.5
            and all(
                float(indexed.loc[metric, "relative_delta_percent"]) <= 1.5
                for metric in (
                    "whole_rr_abs_error_bpm",
                    "envelope_trajectory_mae",
                    "global_envelope_modulation_error",
                )
            )
            and float(indexed.loc[PCC_METRIC, "delta_seed_mean"]) >= -0.003
        )
        catastrophic = any(
            float(indexed.loc[metric, "relative_delta_percent"]) > 3.0
            for metric in ERROR_METRICS
        ) or float(indexed.loc[PCC_METRIC, "delta_seed_mean"]) < -0.005
        decisions.append(
            {
                "condition": condition,
                "substantive_improvement": bool(substantive),
                "guardrails_passed": bool(guardrails),
                "catastrophic_failure": bool(catastrophic),
                "quality_candidate": bool(substantive and guardrails and not catastrophic),
            }
        )
    qualified = [row["condition"] for row in decisions if row["quality_candidate"]]

    def dominates(first: str, second: str) -> bool:
        a = candidate.loc[candidate["condition"].eq(first)].set_index("metric")
        b = candidate.loc[candidate["condition"].eq(second)].set_index("metric")
        no_worse = (
            float(a.loc["local_rr_mae_bpm", "candidate_seed_mean"])
            <= float(b.loc["local_rr_mae_bpm", "candidate_seed_mean"]) * 1.005
            and all(
                float(a.loc[metric, "candidate_seed_mean"])
                <= float(b.loc[metric, "candidate_seed_mean"]) * 1.015
                for metric in (
                    "whole_rr_abs_error_bpm",
                    "envelope_trajectory_mae",
                    "global_envelope_modulation_error",
                )
            )
            and float(a.loc[PCC_METRIC, "candidate_seed_mean"])
            >= float(b.loc[PCC_METRIC, "candidate_seed_mean"]) - 0.003
        )
        materially_better = any(
            float(a.loc[metric, "candidate_seed_mean"])
            <= float(b.loc[metric, "candidate_seed_mean"]) * 0.995
            for metric in ERROR_METRICS
        ) or float(a.loc[PCC_METRIC, "candidate_seed_mean"]) >= float(
            b.loc[PCC_METRIC, "candidate_seed_mean"]
        ) + 0.002
        return bool(no_worse and materially_better)

    for row in decisions:
        condition = row["condition"]
        row["tolerance_aware_pareto"] = bool(
            condition in qualified
            and not any(dominates(other, condition) for other in qualified if other != condition)
        )
    return decisions


def _plot_candidate_summary(output: Path, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    errors = summary.loc[summary["metric"].isin(ERROR_METRICS)]
    pivot = errors.pivot(index="condition", columns="metric", values="relative_delta_percent")
    pivot.plot.bar(ax=axes[0])
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("relative delta (%)")
    axes[0].set_title("error metrics vs GAMMA_050")
    pcc = summary.loc[summary["metric"].eq(PCC_METRIC)].set_index("condition")
    axes[1].bar(pcc.index, pcc["delta_seed_mean"])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("PCC absolute delta")
    axes[1].set_title("lag-aware signed PCC")
    figure.tight_layout()
    figure.savefig(output / "candidate_response.png", dpi=180)
    plt.close(figure)


def _summary_conclusion(decisions: list[dict[str, Any]], candidate: pd.DataFrame) -> str:
    lines = [
        "# W0 FiLM gamma 系数训练验证：描述性结论",
        "",
        "## 三 seed 主结果",
        "",
    ]
    for condition in CONDITIONS:
        local = candidate.loc[
            candidate["condition"].eq(condition)
            & candidate["metric"].eq("local_rr_mae_bpm")
        ].iloc[0]
        decision = next(row for row in decisions if row["condition"] == condition)
        lines.append(
            f"- {condition}: Local RR relative delta="
            f"{local['relative_delta_percent']:.6g}%，improved seeds="
            f"{int(local['improve_seed_count'])}/3，quality candidate="
            f"{decision['quality_candidate']}，Pareto={decision['tolerance_aware_pareto']}。"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 结果来自三个训练 seed 的完整 validation-selected checkpoints。",
            "- test 未访问；训练系数结论不能外推为独立测试集泛化结论。",
            "- 若两个候选均通过，只保留 tolerance-aware Pareto，不按单一指标选择。",
            "",
        ]
    )
    return "\n".join(lines)


def run_summary(*, runs: Iterable[Path]) -> Path:
    lock, lock_hash = load_lock()
    git = _git_state(require_clean=True)
    run_paths = [Path(path).resolve() for path in runs]
    if len(run_paths) != len(CONDITIONS) * len(FORMAL_SEEDS):
        raise ValueError("W0 gamma summary 必须提供 2 conditions × 3 seeds 的六个 runs")
    receipts: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    commits = set()
    for path in run_paths:
        receipt = validate_completed_run(path, lock_hash=lock_hash)
        key = (str(receipt["condition"]), int(receipt["seed"]))
        if key in receipts:
            raise ValueError(f"W0 gamma summary 重复 run identity: {key}")
        receipts[key] = (path, receipt)
        commits.add(str(receipt["git_commit"]))
    expected = {(condition, seed) for condition in CONDITIONS for seed in FORMAL_SEEDS}
    if set(receipts) != expected:
        raise ValueError("W0 gamma summary formal 矩阵不完整")
    if commits != {str(git["commit"])}:
        raise RuntimeError("W0 gamma summary formal runs 与当前代码不是同一 commit")

    quality_source = pd.read_csv(QUALITY_SOURCE)
    paired_frames = []
    matrix_rows = []
    for condition, seed in sorted(expected):
        path, receipt = receipts[(condition, seed)]
        candidate_metrics = pd.read_csv(path / "metrics.csv")
        baseline_metrics = pd.read_csv(BASELINE_RUNS[seed] / "metrics.csv")
        quality = quality_source.loc[quality_source["seed"].eq(seed)].copy()
        paired_frames.append(
            _paired_window_rows(
                condition=condition,
                seed=seed,
                candidate=candidate_metrics,
                baseline=baseline_metrics,
                quality=quality,
            )
        )
        matrix_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "gamma_coefficient": CONDITIONS[condition],
                "beta_coefficient": BETA_COEFFICIENT,
                "selected_epoch": int(receipt["selected_epoch"]),
                "run_dir": str(path),
                "receipt_sha256": sha256_file(path / RECEIPT_NAME),
            }
        )
    paired = pd.concat(paired_frames, ignore_index=True)
    seeds = _seed_summary(paired)
    candidates = _candidate_summary(seeds)
    strata = _stratum_summary(paired)
    decisions = _candidate_decisions(candidates)

    output = create_run_dir(OUTPUT_ROOT / "summary")
    pd.DataFrame(matrix_rows).to_csv(output / "run_matrix.csv", index=False)
    paired.to_csv(output / "paired_window_deltas.csv", index=False)
    seeds.to_csv(output / "seed_summary.csv", index=False)
    candidates.to_csv(output / "candidate_summary.csv", index=False)
    strata.to_csv(output / "stratum_subject_summary.csv", index=False)
    write_json(
        output / "decision_receipt.json",
        {
            "protocol": PROTOCOL,
            "implementation_lock_sha256": lock_hash,
            "git_commit": str(git["commit"]),
            "matrix_complete": True,
            "formal_runs": len(run_paths),
            "paired_metric_rows": int(len(paired)),
            "decisions": decisions,
            "research_test_used": False,
        },
    )
    (output / "conclusions_zh.md").write_text(
        _summary_conclusion(decisions, candidates), encoding="utf-8"
    )
    _plot_candidate_summary(output, candidates)
    write_json(output / "implementation_lock.json", lock)
    files = {
        path.name: identity(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    write_json(
        output / "artifact_manifest.json",
        {
            "protocol": PROTOCOL,
            "phase": "summary",
            "status": "completed",
            "implementation_lock_sha256": lock_hash,
            "files": files,
        },
    )
    write_json(
        output / "freeze_receipt.json",
        {"manifest": identity(output / "artifact_manifest.json")},
    )
    return output
