from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from resp_train.crd.config import CRD_TF_P6A_PROTOCOL_VERSION, load_crd_config
from resp_train.crd.model import CRDCoarseModel, build_crd_model
from resp_train.crd.tf_v1_model import (
    P6_GATE_PARAMETER_COUNT,
    TF_BRANCH_CHECKPOINT_BATCH_CHUNK,
    TF_P6_VARIANT_REPRESENTATIONS,
    TF_P6_VARIANTS,
    trainable_parameter_count,
)
from resp_train.crd.tf_v1_p3 import (
    _branch_projection_gradients_nonzero,
    _new_encoder_dropout_is_zero,
    _synthetic_batch,
    require_clean_git,
)
from resp_train.losses.task import RespirationTaskLoss
from resp_train.utils.run import resolve_device, set_seed


P6A_SYNTHETIC_SEED = 20260811
P6A_ACCEPTANCE_VARIANTS = ("crd_tf402_mws_gate", "crd_tf403_ctrl_gate")
P6A_MAX_RESERVED_FRACTION = 0.80
P6A_EXPECTED_INCREMENTAL_PARAMETERS = {
    "crd_tf401_mws_add": 450_080,
    "crd_tf402_mws_gate": 463_427,
    "crd_tf403_ctrl_gate": 463_203,
}


def run_p6a_cuda_synthetic(
    *, config_path: str | Path, device_name: str, output_root: str | Path
) -> Path:
    commit = require_clean_git()
    device = resolve_device(device_name)
    if device.type != "cuda":
        raise ValueError("CRD-TF P6a synthetic 必须使用 CUDA")
    torch.cuda.set_device(device)
    output_dir = _new_receipt_dir(Path(output_root), commit)
    receipt_path = output_dir / "synthetic_receipt.json"
    results: list[dict[str, Any]] = []
    status = "passed"
    error: str | None = None
    try:
        for variant in TF_P6_VARIANTS:
            results.append(_run_one_synthetic(config_path, variant, device))
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload = {
            "protocol": CRD_TF_P6A_PROTOCOL_VERSION,
            "phase": "p6a_cuda_synthetic",
            "status": status,
            "complete": status == "passed" and len(results) == len(TF_P6_VARIANTS),
            "git_commit": commit,
            "git_dirty": False,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "seed": P6A_SYNTHETIC_SEED,
            "tf_branch_checkpoint_batch_chunk": TF_BRANCH_CHECKPOINT_BATCH_CHUNK,
            "expected_variants": list(TF_P6_VARIANTS),
            "results": results,
            "error": error,
        }
        receipt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(receipt_path.resolve())
    return receipt_path


def _run_one_synthetic(
    config_path: str | Path, variant: str, device: torch.device
) -> dict[str, Any]:
    representations = TF_P6_VARIANT_REPRESENTATIONS[variant]
    cfg = load_crd_config(
        config_path,
        overrides=(
            f"model.variant={variant}",
            f"model.tf_representations=[{','.join(representations)}]",
        ),
    )
    set_seed(P6A_SYNTHETIC_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_crd_model(cfg).to(device).train()
    if not _new_encoder_dropout_is_zero(model):
        raise RuntimeError(f"{variant} 新增 condition encoder dropout 不为 0")
    gate_initially_identity = _gate_initially_identity(model, device)
    if variant.endswith("_gate") and not gate_initially_identity:
        raise RuntimeError(f"{variant} gate 初始 factor 不严格为 1")
    loss_fn = RespirationTaskLoss(cfg).to(device)
    target, sensor, features = _synthetic_batch(device, representations)
    started = time.perf_counter()
    with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=True):
        output = model(sensor, tf=features or None)
    with torch.amp.autocast(device.type, enabled=False):
        components = loss_fn.differentiable_component_sums(output, target.float())
        sync = components["loss_sync_sum"] / components["loss_sync_count"].clamp_min(1)
        effort = components["loss_effort_sum"] / components["loss_effort_count"].clamp_min(1)
        loss = loss_fn.sync_weight * sync + loss_fn.effort_weight * effort
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    all_finite = bool(
        torch.isfinite(output["waveform"]).all()
        and sensor.grad is not None
        and torch.isfinite(sensor.grad).all()
        and gradients
        and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
    )
    projection_gradients_nonzero = _branch_projection_gradients_nonzero(model)
    base_count = trainable_parameter_count(
        CRDCoarseModel("crd_c201_decoder_10hz_cap", P6A_SYNTHETIC_SEED)
    )
    incremental = trainable_parameter_count(model) - base_count
    if incremental != P6A_EXPECTED_INCREMENTAL_PARAMETERS[variant]:
        raise RuntimeError(
            f"{variant} incremental parameters={incremental}，期望 {P6A_EXPECTED_INCREMENTAL_PARAMETERS[variant]}"
        )
    if not all_finite or not projection_gradients_nonzero:
        raise FloatingPointError(f"{variant} synthetic output/gradient contract 失败")
    result = {
        "variant": variant,
        "representations": list(representations),
        "status": "passed",
        "waveform_shape": list(output["waveform"].shape),
        "loss": float(loss.detach().cpu()),
        "all_output_input_parameter_gradients_finite": all_finite,
        "all_branch_final_projection_gradients_nonzero": projection_gradients_nonzero,
        "all_new_encoder_dropout_zero": True,
        "gate_initially_identity": gate_initially_identity,
        "gate_parameters": P6_GATE_PARAMETER_COUNT if model.fusion_gate is not None else 0,
        "trainable_parameters": trainable_parameter_count(model),
        "incremental_parameters_vs_c201": incremental,
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }
    del model, loss_fn, target, sensor, features, output, loss, components, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _gate_initially_identity(model: torch.nn.Module, device: torch.device) -> bool:
    if model.fusion_gate is None:
        return True
    with torch.no_grad():
        factors = model.fusion_gate(torch.randn(1, 96, 1800, device=device))
    return bool(torch.equal(factors, torch.ones_like(factors)))


def p6a_acceptance_overrides(
    variant: str, *, device: str, output_root: str | Path
) -> tuple[str, ...]:
    if variant not in P6A_ACCEPTANCE_VARIANTS:
        raise ValueError(f"P6a acceptance 只允许最大 representation/control gate: {list(P6A_ACCEPTANCE_VARIANTS)}")
    representations = TF_P6_VARIANT_REPRESENTATIONS[variant]
    return (
        "protocol.run_role=acceptance",
        "protocol.execution_gate=p6a_cuda_acceptance",
        f"model.variant={variant}",
        f"model.tf_representations=[{','.join(representations)}]",
        "data.max_train_windows=128",
        "data.max_val_windows=32",
        "training.epochs=1",
        "training.batch_size=128",
        "training.gradient_accumulation_steps=1",
        f"training.device={device}",
        f"outputs.run_root={Path(output_root).resolve() / variant}",
    )


def audit_p6a_acceptance(
    *,
    synthetic_receipt: str | Path,
    run_dirs: Mapping[str, str | Path],
    output_root: str | Path,
) -> Path:
    audit_commit = require_clean_git()
    synthetic_path = Path(synthetic_receipt).resolve()
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    expected_results = list(TF_P6_VARIANTS)
    if (
        synthetic.get("protocol") != CRD_TF_P6A_PROTOCOL_VERSION
        or synthetic.get("phase") != "p6a_cuda_synthetic"
        or synthetic.get("status") != "passed"
        or synthetic.get("complete") is not True
        or synthetic.get("git_dirty") is not False
        or synthetic.get("tf_branch_checkpoint_batch_chunk") != TF_BRANCH_CHECKPOINT_BATCH_CHUNK
        or synthetic.get("expected_variants") != expected_results
        or [row.get("variant") for row in synthetic.get("results", [])] != expected_results
        or not all(_synthetic_row_passed(row) for row in synthetic.get("results", []))
    ):
        raise RuntimeError("CRD-TF P6a synthetic receipt 不合格")
    expected_commit = str(synthetic["git_commit"])
    if set(run_dirs) != set(P6A_ACCEPTANCE_VARIANTS):
        raise ValueError(f"P6a acceptance run_dirs 必须严格包含 {list(P6A_ACCEPTANCE_VARIANTS)}")
    audits = [
        _audit_acceptance_run(variant, Path(run_dirs[variant]).resolve(), expected_commit)
        for variant in P6A_ACCEPTANCE_VARIANTS
    ]
    identity = {
        "synthetic_receipt_sha256": _sha256_file(synthetic_path),
        "runs": {row["variant"]: row["run_dir"] for row in audits},
    }
    audit_id = _sha256_json(identity)
    output_dir = Path(output_root).resolve() / audit_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "p6a_acceptance.json"
    payload = {
        "protocol": CRD_TF_P6A_PROTOCOL_VERSION,
        "phase": "p6a_acceptance_audit",
        "status": "passed",
        "complete": True,
        "engineering_commit": expected_commit,
        "audit_commit": audit_commit,
        "git_dirty": False,
        "research_test_used": False,
        "synthetic_receipt": str(synthetic_path),
        "synthetic_receipt_sha256": identity["synthetic_receipt_sha256"],
        "acceptance_runs": audits,
        "batch_decision": "128x1",
        "maximum_peak_reserved_fraction": max(row["peak_reserved_fraction"] for row in audits),
        "minimum_train_samples_per_second": min(row["train_samples_per_second"] for row in audits),
        "audit_id": audit_id,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _synthetic_row_passed(row: Mapping[str, Any]) -> bool:
    variant = str(row.get("variant"))
    return bool(
        variant in P6A_EXPECTED_INCREMENTAL_PARAMETERS
        and row.get("status") == "passed"
        and row.get("all_output_input_parameter_gradients_finite") is True
        and row.get("all_branch_final_projection_gradients_nonzero") is True
        and row.get("all_new_encoder_dropout_zero") is True
        and row.get("gate_initially_identity") is True
        and int(row.get("incremental_parameters_vs_c201", -1))
        == P6A_EXPECTED_INCREMENTAL_PARAMETERS[variant]
        and int(row.get("gate_parameters", -1))
        == (P6_GATE_PARAMETER_COUNT if variant.endswith("_gate") else 0)
    )


def _audit_acceptance_run(variant: str, run_dir: Path, expected_commit: str) -> dict[str, Any]:
    required = (
        "audit.csv",
        "checkpoint_best_local_rr.pt",
        "checkpoint_final.pt",
        "config.yaml",
        "metrics.csv",
        "metrics_summary.csv",
        "optimizer_parameter_groups.json",
        "run_manifest.json",
        "runtime_summary.json",
        "train_history.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P6a acceptance 产物不完整 {variant}: {missing}")
    cfg = load_crd_config(run_dir / "config.yaml")
    if (
        str(cfg.model.variant) != variant
        or str(cfg.protocol.execution_gate) != "p6a_cuda_acceptance"
        or str(cfg.protocol.run_role) != "acceptance"
        or not str(cfg.training.device).startswith("cuda:")
        or (int(cfg.training.epochs), int(cfg.training.batch_size), int(cfg.training.gradient_accumulation_steps))
        != (1, 128, 1)
        or (int(cfg.data.max_train_windows), int(cfg.data.max_val_windows), cfg.data.max_test_windows)
        != (128, 32, None)
        or bool(cfg.training.early_stopping_enabled) is not True
        or int(cfg.training.early_stopping_patience) != 30
        or float(cfg.training.early_stopping_min_delta) != 0.0
    ):
        raise RuntimeError(f"P6a acceptance resolved config identity 不一致: {run_dir}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("git_commit") != expected_commit
        or manifest.get("git_dirty") is not False
        or manifest.get("protocol") != CRD_TF_P6A_PROTOCOL_VERSION
        or manifest.get("stage") != "tf_p6a"
        or manifest.get("run_role") != "acceptance"
    ):
        raise RuntimeError(f"P6a acceptance manifest identity 不一致: {run_dir}")
    history = pd.read_csv(run_dir / "train_history.csv")
    numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if (
        len(history) != 1
        or int(history.iloc[0]["epoch"]) != 1
        or int(history.iloc[0]["optimizer_update"]) != 1
        or int(history.iloc[0]["early_stopping_improved"]) != 1
        or int(history.iloc[0]["early_stopping_wait"]) != 0
        or int(history.iloc[0]["early_stopping_triggered"]) != 0
        or not np.isfinite(numeric).all()
        or float(history.iloc[0]["train_elapsed_seconds"]) <= 0.0
        or float(history.iloc[0]["train_samples_per_second"]) <= 0.0
    ):
        raise RuntimeError(f"P6a acceptance train history 不合格: {run_dir}")
    for filename in ("checkpoint_best_local_rr.pt", "checkpoint_final.pt"):
        checkpoint = torch.load(run_dir / filename, map_location="cpu")
        tensors = list(_iter_tensors(checkpoint.get("model_state_dict"))) + list(
            _iter_tensors(checkpoint.get("optimizer_state_dict"))
        )
        extra = checkpoint.get("extra_state", {})
        early = extra.get("early_stopping", {})
        if (
            int(checkpoint.get("epoch", -1)) != 1
            or int(extra.get("update_index", -1)) != 1
            or int(extra.get("total_updates", -1)) != 1
            or early.get("enabled") is not True
            or int(early.get("patience", -1)) != 30
            or float(early.get("min_delta", math.nan)) != 0.0
            or early.get("triggered") is not False
            or not tensors
            or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
        ):
            raise RuntimeError(f"P6a acceptance checkpoint 不合格: {run_dir / filename}")
    metrics = pd.read_csv(run_dir / "metrics_summary.csv").iloc[0]
    primary = (
        "whole_rr_abs_error_bpm_mean",
        "local_rr_mae_bpm_mean",
        "envelope_trajectory_mae_mean",
        "global_envelope_modulation_error_mean",
        "lag_aware_signed_pcc_mean",
    )
    if (
        int(metrics["n_samples"]) != 32
        or not np.isfinite([float(metrics[name]) for name in primary]).all()
        or float(metrics["joint_prediction_degenerate_fraction"]) != 0.0
    ):
        raise RuntimeError(f"P6a acceptance validation metrics 不合格: {run_dir}")
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    reserved_fraction = float(runtime.get("peak_reserved_fraction", math.nan))
    if not (0.0 < reserved_fraction <= P6A_MAX_RESERVED_FRACTION):
        raise RuntimeError(f"P6a acceptance peak reserved fraction 越界: {variant}={reserved_fraction}")
    return {
        "variant": variant,
        "run_dir": str(run_dir),
        "training_commit": expected_commit,
        "physical_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "optimizer_updates": 1,
        "validation_samples": 32,
        "train_elapsed_seconds": float(history.iloc[0]["train_elapsed_seconds"]),
        "train_samples_per_second": float(history.iloc[0]["train_samples_per_second"]),
        "peak_allocated_mib": float(runtime["peak_allocated_mib"]),
        "peak_reserved_mib": float(runtime["peak_reserved_mib"]),
        "peak_reserved_fraction": reserved_fraction,
        "checkpoint_best_sha256": _sha256_file(run_dir / "checkpoint_best_local_rr.pt"),
        "checkpoint_final_sha256": _sha256_file(run_dir / "checkpoint_final.pt"),
    }


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_tensors(nested)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _new_receipt_dir(root: Path, commit: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output = root.resolve() / f"{commit[:12]}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output
