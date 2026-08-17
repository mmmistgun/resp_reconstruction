from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from resp_train.crd.config import crd_dependency_versions, load_crd_config
from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.crd.model import build_crd_model
from resp_train.data.factory import build_window_data
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "crd-tf-w-v2-research-informed-20260817"
CANDIDATE_LOCK = REPO_ROOT / "docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json"
CANDIDATE_LOCK_SHA256 = "6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6"

P_MINUS_1_INTERVENTIONS = (
    "FULL",
    "BETA_ONLY",
    "GAMMA_ONLY",
    "CONDITION_OFF",
    "RESP",
    "CARRIER",
    "CARRIER_L",
    "CARRIER_H",
    "TIME_MEAN",
    "TIME_SHIFT_30S",
)
FILM_INTERVENTIONS = frozenset({"FULL", "BETA_ONLY", "GAMMA_ONLY", "CONDITION_OFF"})
PRIMARY_ERROR_COLUMNS = (
    "whole_rr_abs_error_bpm_mean",
    "local_rr_mae_bpm_mean",
    "envelope_trajectory_mae_mean",
    "global_envelope_modulation_error_mean",
)
PCC_COLUMN = "lag_aware_signed_pcc_mean"
DEGENERACY_COLUMN = "joint_prediction_degenerate_fraction"
FULL_ANCHOR_ATOL = 1e-6

_RESP_STOP = 56
_CARRIER_L_STOP = 72
_TIME_SHIFT_FRAMES = 60


def apply_w_intervention(value: torch.Tensor, intervention: str) -> torch.Tensor:
    """对 batch 内已复制的 W tensor 创建确定性只读视图，不修改 cache 或输入 tensor。"""

    name = _normalize_intervention(intervention)
    if value.ndim != 3 or tuple(value.shape[1:]) != (97, 360):
        raise ValueError(f"P−1 W tensor 期望 (B,97,360)，实际 {tuple(value.shape)}")
    if name in FILM_INTERVENTIONS:
        return value
    if name == "RESP":
        output = torch.zeros_like(value)
        output[:, :_RESP_STOP] = value[:, :_RESP_STOP]
        return output
    if name == "CARRIER":
        output = torch.zeros_like(value)
        output[:, _RESP_STOP:] = value[:, _RESP_STOP:]
        return output
    if name == "CARRIER_L":
        output = torch.zeros_like(value)
        output[:, _RESP_STOP:_CARRIER_L_STOP] = value[:, _RESP_STOP:_CARRIER_L_STOP]
        return output
    if name == "CARRIER_H":
        output = torch.zeros_like(value)
        output[:, _CARRIER_L_STOP:] = value[:, _CARRIER_L_STOP:]
        return output
    if name == "TIME_MEAN":
        return value.mean(dim=-1, keepdim=True).expand_as(value)
    if name == "TIME_SHIFT_30S":
        return torch.roll(value, shifts=_TIME_SHIFT_FRAMES, dims=-1)
    raise AssertionError(name)


class AuditedW0Model(nn.Module):
    """只用于 P−1 的冻结 W0 包装器；不改变 checkpoint state 或原模型实现。"""

    def __init__(self, model: nn.Module, *, intervention: str, record_film_statistics: bool = False) -> None:
        super().__init__()
        self.model = model
        self.intervention = _normalize_intervention(intervention)
        self.record_film_statistics = bool(record_film_statistics)
        self._statistics: list[dict[str, np.ndarray]] = []
        _validate_w0_model_contract(model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        tf: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if tf is None or set(tf) != {"w"}:
            raise ValueError(f"P−1 W0 只接受 tf keys=['w']，实际={sorted(tf or {})}")
        w_value = apply_w_intervention(tf["w"], self.intervention)
        latent = self.model.base.encode_local(x)
        gamma_raw, beta_raw = self.model.branches["w"]({"w": w_value})
        effective_gamma = 0.5 * torch.tanh(gamma_raw)
        effective_beta = 0.5 * torch.tanh(beta_raw)
        if self.record_film_statistics:
            self._statistics.append(_film_statistics(effective_gamma, effective_beta))

        if self.intervention in {"BETA_ONLY", "CONDITION_OFF"}:
            effective_gamma = torch.zeros_like(effective_gamma)
        if self.intervention in {"GAMMA_ONLY", "CONDITION_OFF"}:
            effective_beta = torch.zeros_like(effective_beta)
        conditioned = latent * (1.0 + effective_gamma) + effective_beta
        return self.model.base.decode_local(conditioned)

    def take_film_statistics(self) -> dict[str, np.ndarray]:
        if not self._statistics:
            return {}
        names = tuple(self._statistics[0])
        output = {name: np.concatenate([batch[name] for batch in self._statistics]) for name in names}
        self._statistics.clear()
        return output


def decide_p2_fusion(seed_summaries: pd.DataFrame) -> dict[str, Any]:
    """按冻结 near 规则把 P−1 结果映射到至多一个 P2 融合 arm。"""

    required = {"seed", "intervention", *PRIMARY_ERROR_COLUMNS, PCC_COLUMN, DEGENERACY_COLUMN}
    missing = sorted(required - set(seed_summaries.columns))
    if missing:
        raise ValueError(f"P−1 seed summary 缺少列: {missing}")
    frame = seed_summaries.copy()
    frame["intervention"] = frame["intervention"].astype(str).str.upper()
    expected_seeds = {20260811, 20260812, 20260813}
    for intervention in ("FULL", "BETA_ONLY", "GAMMA_ONLY"):
        subset = frame.loc[frame["intervention"].eq(intervention)]
        if len(subset) != 3 or set(subset["seed"].astype(int)) != expected_seeds:
            raise ValueError(f"P−1 {intervention} 必须恰含三个固定 seed")
        if subset["seed"].duplicated().any():
            raise ValueError(f"P−1 {intervention} seed 重复")

    full = frame.loc[frame["intervention"].eq("FULL")].set_index("seed").sort_index()
    if not np.isfinite(full[[*PRIMARY_ERROR_COLUMNS, PCC_COLUMN, DEGENERACY_COLUMN]].to_numpy()).all():
        raise FloatingPointError("P−1 FULL summary 包含非有限值")
    if not np.allclose(full[DEGENERACY_COLUMN].to_numpy(dtype=np.float64), 0.0, atol=0.0, rtol=0.0):
        raise RuntimeError("P−1 FULL 出现 prediction degeneracy")

    details: dict[str, Any] = {}
    for intervention in ("BETA_ONLY", "GAMMA_ONLY"):
        candidate = frame.loc[frame["intervention"].eq(intervention)].set_index("seed").sort_index()
        values = candidate[[*PRIMARY_ERROR_COLUMNS, PCC_COLUMN, DEGENERACY_COLUMN]].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"P−1 {intervention} summary 包含非有限值")
        seed_checks: list[dict[str, Any]] = []
        for seed in sorted(expected_seeds):
            error_worsening = {
                column: _relative_change(float(candidate.loc[seed, column]), float(full.loc[seed, column]))
                for column in PRIMARY_ERROR_COLUMNS
            }
            pcc_delta = float(candidate.loc[seed, PCC_COLUMN] - full.loc[seed, PCC_COLUMN])
            degenerate_fraction = float(candidate.loc[seed, DEGENERACY_COLUMN])
            passed = (
                all(value <= 0.015 for value in error_worsening.values())
                and pcc_delta >= -0.003
                and degenerate_fraction == 0.0
            )
            seed_checks.append(
                {
                    "seed": seed,
                    "error_relative_changes": error_worsening,
                    "pcc_delta": pcc_delta,
                    "prediction_degenerate_fraction": degenerate_fraction,
                    "passed": passed,
                }
            )

        candidate_mean = candidate[[*PRIMARY_ERROR_COLUMNS, PCC_COLUMN, DEGENERACY_COLUMN]].mean(axis=0)
        full_mean = full[[*PRIMARY_ERROR_COLUMNS, PCC_COLUMN, DEGENERACY_COLUMN]].mean(axis=0)
        mean_error_worsening = {
            column: _relative_change(float(candidate_mean[column]), float(full_mean[column]))
            for column in PRIMARY_ERROR_COLUMNS
        }
        mean_pcc_delta = float(candidate_mean[PCC_COLUMN] - full_mean[PCC_COLUMN])
        mean_degenerate_fraction = float(candidate_mean[DEGENERACY_COLUMN])
        passed_seed_count = sum(bool(row["passed"]) for row in seed_checks)
        mean_passed = (
            all(value <= 0.015 for value in mean_error_worsening.values())
            and mean_pcc_delta >= -0.003
            and mean_degenerate_fraction == 0.0
        )
        quality_near = bool(mean_passed and passed_seed_count >= 2)
        details[intervention] = {
            "quality_near": quality_near,
            "mean_error_relative_changes": mean_error_worsening,
            "mean_pcc_delta": mean_pcc_delta,
            "mean_prediction_degenerate_fraction": mean_degenerate_fraction,
            "passed_seed_count": passed_seed_count,
            "seed_checks": seed_checks,
        }

    beta_near = bool(details["BETA_ONLY"]["quality_near"])
    gamma_near = bool(details["GAMMA_ONLY"]["quality_near"])
    if beta_near:
        decision = "train_add"
        selected_variant = "crd_tfw_v2_f1_add_full_12v_d6"
    elif gamma_near:
        decision = "train_scale"
        selected_variant = "crd_tfw_v2_f2_scale_full_12v_d6"
    else:
        decision = "retain_film_no_p2_training"
        selected_variant = None
    return {
        "protocol": PROTOCOL,
        "phase": "p_minus_1_budget_decision",
        "decision": decision,
        "selected_p2_variant": selected_variant,
        "both_near_prefers_add": bool(beta_near and gamma_near),
        "interventions": details,
        "samp_id_analysis_used": False,
        "research_test_used": False,
    }


def run_p_minus_1_audit(
    *,
    candidate_lock_path: str | Path = CANDIDATE_LOCK,
    split: str = "val",
    device_name: str = "cuda:0",
) -> Path:
    """运行冻结 W0 三 checkpoint 的完整 validation P−1；默认由用户在干净提交上执行。"""

    _validate_audit_split(split)
    commit = _require_clean_git()
    lock_path, lock = _load_candidate_lock(candidate_lock_path)
    _verify_lock_inputs(lock)
    resolved_device = torch.device(device_name)
    if resolved_device.type != "cuda":
        raise ValueError("P−1 完整 validation 必须使用 CUDA；CPU 只允许定向单测")
    output_root = REPO_ROOT / str(lock["output_contract"]["p_minus_1_root"])
    if output_root.exists():
        raise FileExistsError(f"P−1 固定输出已存在，拒绝覆盖: {output_root}")
    staging = _create_staging_directory(output_root)
    try:
        seed_entries = _w0_seed_entries(lock)
        first_cfg = _load_audit_config(seed_entries[0], device_name=device_name)
        val_data = build_window_data(
            first_cfg,
            split=str(first_cfg.data.val_split),
            max_windows=None,
            sample_strategy=str(first_cfg.data.val_sample_strategy),
            sample_seed=int(first_cfg.data.val_sample_seed),
            shuffle=False,
        )
        _validate_validation_rows(val_data.rows, lock)
        expected_windows = int(lock["cache_lock"]["row_identity"]["val_count"])
        if len(val_data.dataset) != expected_windows:
            raise RuntimeError(f"P−1 validation windows={len(val_data.dataset)} != {expected_windows}")

        metric_frames: list[pd.DataFrame] = []
        seed_summary_frames: list[pd.DataFrame] = []
        film_frames: list[pd.DataFrame] = []
        full_anchor_checks: list[dict[str, Any]] = []
        for entry in seed_entries:
            cfg = _load_audit_config(entry, device_name=device_name)
            checkpoint_path = REPO_ROOT / entry["run_dir"] / "checkpoint_best_local_rr.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            _validate_checkpoint_config(checkpoint.get("config"), cfg)
            model = build_crd_model(cfg)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            seed = int(entry["seed"])
            for intervention in P_MINUS_1_INTERVENTIONS:
                audited_model = AuditedW0Model(
                    model,
                    intervention=intervention,
                    record_film_statistics=intervention == "FULL",
                )
                predictions = collect_predictions(
                    audited_model,
                    val_data.loader,
                    device=resolved_device,
                    max_windows=expected_windows,
                    use_amp=bool(cfg.training.use_amp),
                )
                _validate_prediction_identity(predictions, val_data.rows)
                metrics = evaluate_task_predictions(
                    predictions,
                    cfg,
                    include_test_only=False,
                    method=f"crd_tf102_w__p_minus_1__{intervention.lower()}",
                )
                metrics.insert(0, "checkpoint_sha256", entry["checkpoint"]["sha256"])
                metrics.insert(0, "selected_epoch", int(entry["selected_epoch"]))
                metrics.insert(0, "intervention", intervention)
                metrics.insert(0, "seed", seed)
                metric_frames.append(metrics)

                summary = summarize_task_metrics(metrics)
                _validate_primary_summary(summary, intervention=intervention, seed=seed)
                summary.insert(0, "checkpoint_sha256", entry["checkpoint"]["sha256"])
                summary.insert(0, "selected_epoch", int(entry["selected_epoch"]))
                summary.insert(0, "intervention", intervention)
                summary.insert(0, "seed", seed)
                seed_summary_frames.append(summary)
                if intervention == "FULL":
                    full_anchor_checks.append(_check_full_anchor(summary, entry))
                    statistics = audited_model.take_film_statistics()
                    if not statistics:
                        raise RuntimeError(f"P−1 FULL 缺少 FiLM statistics seed={seed}")
                    film = pd.DataFrame(statistics)
                    if len(film) != expected_windows:
                        raise RuntimeError(f"P−1 FiLM statistics 行数错误 seed={seed}: {len(film)}")
                    film.insert(0, "samp_id", predictions["samp_id"])
                    film.insert(0, "dataset_row_id", predictions["dataset_row_id"])
                    film.insert(0, "intervention", intervention)
                    film.insert(0, "seed", seed)
                    film_frames.append(film)
                elif audited_model.take_film_statistics():
                    raise RuntimeError(f"P−1 非 FULL 干预意外保存 FiLM statistics: {intervention}")
                print(f"P−1 validation complete seed={seed} intervention={intervention}", flush=True)
                del predictions, metrics, summary, audited_model
                if resolved_device.type == "cuda":
                    torch.cuda.empty_cache()
            del model

        all_metrics = pd.concat(metric_frames, ignore_index=True)
        seed_summaries = pd.concat(seed_summary_frames, ignore_index=True)
        film_statistics = pd.concat(film_frames, ignore_index=True)
        decision = decide_p2_fusion(seed_summaries)
        summary_output = _with_three_seed_aggregates(seed_summaries)

        metrics_path = staging / "intervention_seed_metrics.csv"
        summary_path = staging / "intervention_summary.csv"
        film_path = staging / "film_statistics.csv"
        decision_path = staging / "p_minus_1_decision.json"
        all_metrics.to_csv(metrics_path, index=False)
        summary_output.to_csv(summary_path, index=False)
        film_statistics.to_csv(film_path, index=False)
        decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "protocol": PROTOCOL,
            "phase": "p_minus_1_validation_functional_audit",
            "status": "passed",
            "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [
                "scripts/eval_crd_tf_w_v2_functional_audit.py",
                "--candidate-lock",
                "docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json",
                "--split",
                "val",
                "--device",
                str(resolved_device),
            ],
            "git_commit": commit,
            "git_dirty": False,
            "candidate_lock": str(lock_path),
            "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
            "split": "validation",
            "research_test_used": False,
            "model_training_used": False,
            "checkpoint_modified": False,
            "cache_modified": False,
            "samp_id_analysis_used": False,
            "device": str(resolved_device),
            "dependency_versions": crd_dependency_versions(),
            "expected_windows_per_evaluation": expected_windows,
            "val_sample_seed": int(first_cfg.data.val_sample_seed),
            "cache_manifest_sha256": lock["cache_lock"]["manifest"]["sha256"],
            "checkpoint_count": len(seed_entries),
            "interventions": list(P_MINUS_1_INTERVENTIONS),
            "evaluation_count": len(seed_entries) * len(P_MINUS_1_INTERVENTIONS),
            "per_sample_metric_rows": int(len(all_metrics)),
            "film_statistic_rows": int(len(film_statistics)),
            "full_anchor_checks": full_anchor_checks,
            "decision": decision["decision"],
            "selected_p2_variant": decision["selected_p2_variant"],
            "files": {
                path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
                for path in (metrics_path, summary_path, film_path, decision_path)
            },
        }
        (staging / "p_minus_1_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_root)
        return output_root
    except BaseException as exc:
        failure = {
            "protocol": PROTOCOL,
            "phase": "p_minus_1_validation_functional_audit",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (staging / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


def _normalize_intervention(intervention: str) -> str:
    name = str(intervention).strip().upper()
    if name not in P_MINUS_1_INTERVENTIONS:
        raise ValueError(f"未知 P−1 intervention={intervention!r}")
    return name


def _validate_w0_model_contract(model: nn.Module) -> None:
    if getattr(model, "tf_variant", None) != "crd_tf102_w":
        raise ValueError("P−1 只接受 crd_tf102_w")
    if set(getattr(model, "branches", {})) != {"w"}:
        raise ValueError("P−1 W0 必须恰含 W branch")
    if len(getattr(model, "controls", ())) != 0 or getattr(model, "fusion_gate", None) is not None:
        raise ValueError("P−1 W0 不得含 control 或 fusion gate")
    if not hasattr(model, "base") or not hasattr(model.base, "encode_local") or not hasattr(model.base, "decode_local"):
        raise TypeError("P−1 W0 base 缺少 encode/decode contract")


def _film_statistics(gamma: torch.Tensor, beta: torch.Tensor) -> dict[str, np.ndarray]:
    if gamma.shape != beta.shape or gamma.ndim != 3:
        raise ValueError("P−1 gamma/beta shape 不一致")
    gamma_abs = gamma.detach().float().abs()
    beta_abs = beta.detach().float().abs()
    gamma_flat = gamma_abs.flatten(1)
    beta_flat = beta_abs.flatten(1)
    output = {
        "mean_abs_gamma": gamma_flat.mean(dim=1),
        "median_abs_gamma": torch.quantile(gamma_flat, 0.5, dim=1, interpolation="linear"),
        "mean_abs_beta": beta_flat.mean(dim=1),
        "median_abs_beta": torch.quantile(beta_flat, 0.5, dim=1, interpolation="linear"),
        "gamma_time_mean_abs_difference": gamma_abs.diff(dim=-1).mean(dim=(1, 2)),
        "beta_time_mean_abs_difference": beta_abs.diff(dim=-1).mean(dim=(1, 2)),
        "gamma_saturation_fraction": (gamma_abs >= 0.49).float().mean(dim=(1, 2)),
        "beta_saturation_fraction": (beta_abs >= 0.49).float().mean(dim=(1, 2)),
    }
    return {name: value.cpu().numpy().astype(np.float64, copy=False) for name, value in output.items()}


def _relative_change(candidate: float, anchor: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(anchor) or anchor <= 0.0:
        raise ValueError(f"相对变化要求正的有限 anchor，当前 candidate={candidate}, anchor={anchor}")
    return (candidate - anchor) / anchor


def _validate_audit_split(split: str) -> None:
    if str(split).strip().lower() != "val":
        raise ValueError("P−1 只允许 split=val；禁止 research-test")


def _load_candidate_lock(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    if resolved != CANDIDATE_LOCK.resolve():
        raise ValueError(f"P−1 只接受冻结 candidate lock: {CANDIDATE_LOCK}")
    if _sha256_file(resolved) != CANDIDATE_LOCK_SHA256:
        raise RuntimeError("P−1 candidate lock SHA-256 不一致")
    lock = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or lock.get("status") != "p0_complete_downstream_implementation_not_activated"
        or lock.get("p0_outcome", {}).get("candidate_lock_complete") is not True
        or lock.get("forbidden", {}).get("samp_id_analysis") is not True
        or lock.get("forbidden", {}).get("ordinary_eval_crd_test") is not True
    ):
        raise RuntimeError("P−1 candidate lock 协议边界不合格")
    return resolved, lock


def _verify_lock_inputs(lock: Mapping[str, Any]) -> None:
    for relative, expected in lock["runtime_code_identity"].items():
        if _sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"P−1 锁定 runtime identity 漂移: {relative}")
    for anchor in lock["anchors"]:
        for entry in anchor["checkpoints"]:
            run_dir = REPO_ROOT / entry["run_dir"]
            for key, filename in {
                "checkpoint": "checkpoint_best_local_rr.pt",
                "config": "config.yaml",
                "manifest": "run_manifest.json",
                "validation_summary": "metrics_summary.csv",
            }.items():
                _verify_file_identity(run_dir / filename, entry[key])
    cache = lock["cache_lock"]
    _verify_file_identity(REPO_ROOT / cache["manifest"]["path"], cache["manifest"])
    _verify_file_identity(
        REPO_ROOT / cache["frequency_file"]["path"],
        {"size_bytes": cache["frequency_file"]["size_bytes"], "sha256": cache["frequency_file"]["file_sha256"]},
    )
    train_w_path = REPO_ROOT / cache["train_w"]["path"]
    if train_w_path.stat().st_size != int(cache["train_w"]["size_bytes"]):
        raise RuntimeError(f"P−1 train cache size 漂移: {train_w_path}")
    _verify_file_identity(REPO_ROOT / cache["val_w"]["path"], cache["val_w"])
    frequencies = np.load(REPO_ROOT / cache["frequency_file"]["path"], allow_pickle=False)
    expected_indices = {
        "resp": np.flatnonzero(frequencies <= 0.80),
        "carrier": np.flatnonzero(frequencies > 0.80),
        "carrier_l_diagnostic": np.flatnonzero((frequencies > 0.80) & (frequencies <= 2.0)),
        "carrier_h_diagnostic": np.flatnonzero(frequencies > 2.0),
        "full_6v": np.arange(0, frequencies.size, 2),
    }
    for key, indices in expected_indices.items():
        observed = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
        if observed != cache["views"][key]["indices_sha256"]:
            raise RuntimeError(f"P−1 W view identity 漂移: {key}")


def _w0_seed_entries(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    matches = [anchor for anchor in lock["anchors"] if anchor.get("role") == "w0_anchor"]
    if len(matches) != 1 or matches[0].get("variant") != "crd_tf102_w":
        raise RuntimeError("P−1 candidate lock 缺少唯一 W0 anchor")
    entries = sorted(matches[0]["checkpoints"], key=lambda item: int(item["seed"]))
    if [int(entry["seed"]) for entry in entries] != [20260811, 20260812, 20260813]:
        raise RuntimeError("P−1 W0 seed allowlist 错误")
    return entries


def _load_audit_config(entry: Mapping[str, Any], *, device_name: str):
    config_path = REPO_ROOT / str(entry["run_dir"]) / "config.yaml"
    cfg = load_crd_config(
        config_path,
        overrides=[f"training.device={device_name}", "training.show_progress=false"],
    )
    if (
        str(cfg.protocol.name) != "crd-tf-v1-research-informed-20260812"
        or str(cfg.protocol.run_role) != "formal"
        or str(cfg.model.variant) != "crd_tf102_w"
        or list(cfg.model.tf_representations) != ["w"]
        or int(cfg.training.seed) != int(entry["seed"])
        or cfg.data.max_val_windows is not None
        or cfg.data.get("tf_research_test_cache_path") is not None
    ):
        raise RuntimeError(f"P−1 W0 config identity 不合格 seed={entry['seed']}")
    return cfg


def _validate_validation_rows(rows: pd.DataFrame, lock: Mapping[str, Any]) -> None:
    if "dataset_row_id" not in rows or "split" not in rows:
        raise KeyError("P−1 validation rows 缺少 dataset_row_id/split")
    if set(rows["split"].astype(str)) != {"val"}:
        raise RuntimeError("P−1 dataset 不是纯 validation")
    row_ids = np.sort(rows["dataset_row_id"].astype(np.int64).to_numpy())
    expected = str(lock["cache_lock"]["row_identity"]["val_row_content_sha256"])
    if hashlib.sha256(row_ids.tobytes(order="C")).hexdigest() != expected:
        raise RuntimeError("P−1 validation dataset_row_id identity 不一致")


def _validate_prediction_identity(predictions: Mapping[str, np.ndarray], rows: pd.DataFrame) -> None:
    if set(np.asarray(predictions["split"]).astype(str)) != {"val"}:
        raise RuntimeError("P−1 prediction split 不是 validation")
    observed = np.asarray(predictions["dataset_row_id"], dtype=np.int64)
    expected = rows["dataset_row_id"].astype(np.int64).to_numpy()
    if not np.array_equal(observed, expected):
        raise RuntimeError("P−1 prediction row order 与 validation rows 不一致")


def _validate_primary_summary(summary: pd.DataFrame, *, intervention: str, seed: int) -> None:
    if len(summary) != 1:
        raise RuntimeError("P−1 summary 必须恰有一行")
    values = summary.loc[summary.index[0], [*PRIMARY_ERROR_COLUMNS, PCC_COLUMN]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"P−1 primary 非有限 seed={seed} intervention={intervention}")


def _check_full_anchor(summary: pd.DataFrame, entry: Mapping[str, Any]) -> dict[str, Any]:
    locked_path = REPO_ROOT / str(entry["run_dir"]) / "metrics_summary.csv"
    locked = pd.read_csv(locked_path).iloc[0]
    observed = summary.iloc[0]
    deltas = {
        column: float(observed[column]) - float(locked[column])
        for column in (*PRIMARY_ERROR_COLUMNS, PCC_COLUMN, DEGENERACY_COLUMN)
    }
    max_abs_delta = max(abs(value) for value in deltas.values())
    if max_abs_delta > FULL_ANCHOR_ATOL:
        raise RuntimeError(
            f"P−1 FULL 未复现锁定 validation summary seed={entry['seed']} max_abs_delta={max_abs_delta}"
        )
    return {
        "seed": int(entry["seed"]),
        "locked_validation_summary_sha256": entry["validation_summary"]["sha256"],
        "max_abs_delta": max_abs_delta,
        "atol": FULL_ANCHOR_ATOL,
        "passed": True,
    }


def _with_three_seed_aggregates(seed_summaries: pd.DataFrame) -> pd.DataFrame:
    seed_rows = seed_summaries.copy()
    seed_rows.insert(0, "summary_level", "seed")
    numeric_columns = [
        column
        for column in seed_summaries.columns
        if column not in {"seed", "intervention", "checkpoint_sha256", "selected_epoch"}
        and pd.api.types.is_numeric_dtype(seed_summaries[column])
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for intervention in P_MINUS_1_INTERVENTIONS:
        group = seed_summaries.loc[seed_summaries["intervention"].eq(intervention)]
        if len(group) != 3:
            raise RuntimeError(f"P−1 summary 缺少三个 seed: {intervention}")
        row: dict[str, Any] = {
            "summary_level": "three_seed_mean",
            "seed": "",
            "intervention": intervention,
            "checkpoint_sha256": "",
            "selected_epoch": "",
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
            row[column] = float(np.nanmean(values))
            row[f"{column}_seed_sd"] = float(np.nanstd(values, ddof=1))
        aggregate_rows.append(row)
    return pd.concat([seed_rows, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)


def _create_staging_directory(output_root: Path) -> Path:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    staging = output_root.with_name(f"{output_root.name}.incomplete_{stamp}")
    staging.mkdir(parents=False, exist_ok=False)
    return staging


def _require_clean_git() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        raise RuntimeError("P−1 完整 validation 要求干净 Git 工作树；请先提交或移出全部改动")
    return commit


def _verify_file_identity(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(expected["size_bytes"]):
        raise RuntimeError(f"锁定文件 size 漂移: {path}")
    if _sha256_file(path) != str(expected["sha256"]):
        raise RuntimeError(f"锁定文件 SHA-256 漂移: {path}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
