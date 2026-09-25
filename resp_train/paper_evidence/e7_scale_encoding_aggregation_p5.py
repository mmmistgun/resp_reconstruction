"""E7 P5 validation 汇总、selected-checkpoint 诊断与最终冻结。"""

from __future__ import annotations

import fcntl
import json
import sys
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.data.factory import build_tho_data
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence import e7_scale_encoding_aggregation as e7
from resp_train.paper_evidence import e7_scale_encoding_aggregation_formal as p4
from resp_train.paper_evidence.e7_scale_encoding_aggregation_model import ARMS, build_e7_model


P5_PROTOCOL = "e7-scale-encoding-aggregation-p5-validation-v1-20260926"
P4_LOCK_SHA256 = "068ba8ec6c5866ebf1b17d560b4fb5f5448e7e921e0dc8b15dda66c2f907c193"
P5_PROTOCOL_PATH = Path(
    "docs/experiments/e7_scale_encoding_aggregation_p5_validation_protocol_20260926.md"
)
P5_PATH = Path("resp_train/paper_evidence/e7_scale_encoding_aggregation_p5.py")
P5_SCRIPT_PATH = Path("scripts/run_e7_scale_encoding_aggregation_p5.py")
P5_TEST_PATH = Path("tests/test_e7_scale_encoding_aggregation_p5.py")
OUTPUT = Path("runs/e7_scale_encoding_aggregation/p5_validation")
DIAGNOSTIC_MICROBATCH = 4
DIAGNOSTIC_CHANNEL_INDICES = tuple(range(0, 96, 8))
DIAGNOSTIC_TIME_INDICES = tuple(range(0, 360, 8))


def p5_contract() -> dict[str, Any]:
    return {
        "protocol": P5_PROTOCOL,
        "p4_execution_lock_sha256": P4_LOCK_SHA256,
        "arms": list(ARMS),
        "seeds": list(e7.SEEDS),
        "cells": len(ARMS) * len(e7.SEEDS),
        "split": "val",
        "validation_rows": e7.COUNTS["val"],
        "representation_sampling": {
            "all_validation_windows": True,
            "channel_indices": list(DIAGNOSTIC_CHANNEL_INDICES),
            "time_indices": list(DIAGNOSTIC_TIME_INDICES),
            "scale_indices": "all_97",
            "microbatch": DIAGNOSTIC_MICROBATCH,
        },
        "interventions": ["full", "residual_off", "uniform_attention", "both_off"],
        "research_test_access": False,
    }


def critical_paths() -> tuple[Path, ...]:
    paths = (P5_PATH, P5_SCRIPT_PATH, P5_TEST_PATH, P5_PROTOCOL_PATH)
    missing = [str(path) for path in paths if not (e7.ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"E7 P5 缺少关键文件: {missing}")
    return paths


def runtime_identity(*, require_cuda: bool = False, device: str = "cpu") -> dict[str, Any]:
    state = e7.git_state(e7.ROOT)
    if state.get("status_porcelain"):
        raise RuntimeError("E7 P5 执行要求干净提交")
    _lock, digest = p4.load_p4_lock()
    if digest != P4_LOCK_SHA256:
        raise ValueError("E7 P5 引用的 P4 execution lock 漂移")
    if require_cuda and (torch.device(device).type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("E7 P5 selected-checkpoint 诊断需要 CUDA")
    return {
        "git": state,
        "p4_execution_lock_sha256": digest,
        "contract": p5_contract(),
        "code_files": {
            str(path): e7.identity(e7.ROOT / path)
            for path in critical_paths()
        },
    }


@contextmanager
def phase_guard(parent: Path, identity_key: str) -> Iterator[None]:
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / f".execution_{identity_key}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("E7 P5 相同阶段正在运行") from exc
        try:
            for freeze in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((freeze.parent / "manifest.json").read_text())
                if manifest.get("p4_execution_lock_sha256") == P4_LOCK_SHA256:
                    verify_attempt(freeze.parent, phase=manifest["phase"])
                    raise FileExistsError(f"E7 P5 相同阶段已完成: {freeze.parent}")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def attempt(
    parent: Path,
    phase: str,
    runtime: Mapping[str, Any],
    *,
    arm: str | None = None,
    seed: int | None = None,
) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"{phase}_{P4_LOCK_SHA256[:12]}_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    context = {
        "protocol": P5_PROTOCOL,
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "p4_execution_lock_sha256": P4_LOCK_SHA256,
        "git": runtime["git"],
        "code_files": runtime["code_files"],
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    e7.write_json(output / "lifecycle_started.json", {**context, "status": "running"})
    try:
        yield output
        e7.write_json(
            output / "lifecycle_completed.json",
            {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
        )
        manifest = {
            **context,
            "status": "completed",
            "files": {
                str(file.relative_to(output)): e7.identity(file)
                for file in sorted(output.rglob("*"))
                if file.is_file()
            },
        }
        e7.write_json(output / "manifest.json", manifest)
        e7.write_json(
            output / "freeze_receipt.json",
            {"protocol": P5_PROTOCOL, "manifest": e7.identity(output / "manifest.json")},
        )
    except BaseException as exc:
        e7.write_json(
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


def verify_attempt(path: Path, *, phase: str) -> dict[str, Any]:
    path = path.resolve()
    if (path / "lifecycle_failed.json").exists():
        raise ValueError(f"E7 P5 attempt 失败: {path}")
    freeze = json.loads((path / "freeze_receipt.json").read_text())
    e7.verify(path / "manifest.json", freeze["manifest"])
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("protocol") != P5_PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("p4_execution_lock_sha256") != P4_LOCK_SHA256
    ):
        raise ValueError("E7 P5 attempt identity 漂移")
    required = {
        "summary": {
            "per_seed.csv",
            "across_seed.csv",
            "simple_effects_per_seed.csv",
            "simple_effects_across_seed.csv",
            "factorial_contrasts_per_seed.csv",
            "factorial_contrasts_across_seed.csv",
            "materiality_per_seed.csv",
            "materiality_across_seed.csv",
            "subject_macro.csv",
            "subject_macro_materiality_per_seed.csv",
            "summary_receipt.json",
        },
        "diagnostic": {
            "representation_diagnostics.csv",
            "intervention_metrics.csv",
            "intervention_summary.csv",
            "intervention_delta.csv",
            "diagnostic_receipt.json",
        },
        "final": {
            "seed_metrics.csv",
            "factorial_contrasts.csv",
            "materiality.csv",
            "subject_macro.csv",
            "representation_diagnostics.csv",
            "checkpoint_interventions.csv",
            "final_receipt.json",
        },
    }[phase] | {"lifecycle_completed.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("E7 P5 attempt 缺少必需产物")
    for relative, expected in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path):
            raise ValueError("E7 P5 manifest 路径越界")
        e7.verify(file, expected)
    return manifest


def _formal_sources() -> list[tuple[Path, dict[str, Any], Path]]:
    sources = []
    for attempt_path in p4.completed_runs():
        receipt = json.loads((attempt_path / "formal_receipt.json").read_text())
        run_dir = (attempt_path / receipt["run_dir"]).resolve()
        if not run_dir.is_relative_to(attempt_path.resolve()):
            raise ValueError("E7 P5 formal run_dir 路径越界")
        sources.append((attempt_path, receipt, run_dir))
    return sources


def materiality_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = frame.set_index(["arm", "seed"])
    records: list[dict[str, Any]] = []
    for contrast, coefficients in e7.SIMPLE_CONTRASTS.items():
        positive = [arm for arm, coefficient in coefficients.items() if coefficient == 1]
        negative = [arm for arm, coefficient in coefficients.items() if coefficient == -1]
        if len(positive) != 1 or len(negative) != 1:
            raise ValueError("E7 P5 simple contrast 必须是单一 candidate-baseline")
        candidate_arm, baseline_arm = positive[0], negative[0]
        for seed in e7.SEEDS:
            for metric in e7.PRIMARY:
                candidate = float(indexed.loc[(candidate_arm, seed), metric + "_mean"])
                baseline = float(indexed.loc[(baseline_arm, seed), metric + "_mean"])
                if metric == e7.PCC:
                    delta = candidate - baseline
                    tolerance = e7.PCC_TOLERANCE
                    unit = "absolute_utility"
                    defined = True
                else:
                    defined = baseline != 0
                    delta = 100 * (baseline - candidate) / baseline if defined else np.nan
                    tolerance = e7.ERROR_TOLERANCE_PERCENT
                    unit = "relative_percent_utility"
                classification = (
                    "undefined"
                    if not defined
                    else "improved"
                    if delta > tolerance
                    else "degraded"
                    if delta < -tolerance
                    else "within_tolerance"
                )
                records.append(
                    {
                        "contrast": contrast,
                        "candidate_arm": candidate_arm,
                        "baseline_arm": baseline_arm,
                        "seed": seed,
                        "metric": metric,
                        "candidate": candidate,
                        "baseline": baseline,
                        "utility_delta": delta,
                        "unit": unit,
                        "tolerance": tolerance,
                        "classification": classification,
                    }
                )
    per_seed = pd.DataFrame(records)
    across = []
    for (contrast, metric), group in per_seed.groupby(["contrast", "metric"], sort=False):
        values = group.utility_delta.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        across.append(
            {
                "contrast": contrast,
                "metric": metric,
                "utility_delta_mean": finite.mean() if len(finite) else np.nan,
                "utility_delta_sample_sd": finite.std(ddof=1) if len(finite) > 1 else np.nan,
                "improved_seeds": int(group.classification.eq("improved").sum()),
                "within_tolerance_seeds": int(group.classification.eq("within_tolerance").sum()),
                "degraded_seeds": int(group.classification.eq("degraded").sum()),
                "undefined_seeds": int(group.classification.eq("undefined").sum()),
            }
        )
    return per_seed, pd.DataFrame(across)


def run_summary() -> Path:
    runtime = runtime_identity()
    parent = e7.ROOT / OUTPUT / "summary"
    with phase_guard(parent, "summary"):
        with attempt(parent, "summary", runtime) as output:
            p1_lock, _digest = e7.load_implementation_lock()
            frames = []
            window_frames = []
            sources = {}
            for formal_attempt, receipt, run_dir in _formal_sources():
                cfg = OmegaConf.load(run_dir / "config.yaml")
                checked = p4.validate_run(
                    run_dir,
                    cfg,
                    pd.read_csv(formal_attempt / "val_rows.csv"),
                )
                if any(receipt.get(key) != value for key, value in checked.items()):
                    raise ValueError("E7 P5 formal receipt 与训练产物不一致")
                p4.validate_anchor_rows(pd.read_csv(run_dir / "metrics.csv"), p1_lock, receipt["seed"])
                summary = pd.read_csv(run_dir / "metrics_summary.csv")
                summary.insert(0, "split", "val")
                summary.insert(0, "selected_epoch", receipt["selected_epoch"])
                summary.insert(0, "completed_epochs", receipt["completed_epochs"])
                summary.insert(0, "seed", receipt["seed"])
                summary.insert(0, "arm", receipt["arm"])
                frames.append(summary)
                metrics = pd.read_csv(run_dir / "metrics.csv")
                window_frames.append(metrics)
                key = f"{receipt['arm']}/{receipt['seed']}"
                sources[key] = {
                    "path": str(formal_attempt.resolve()),
                    "manifest": e7.identity(formal_attempt / "manifest.json"),
                    "metrics": e7.identity(run_dir / "metrics.csv"),
                    "selected_checkpoint": receipt["selected_checkpoint"],
                }
            per_seed = pd.concat(frames, ignore_index=True)
            e7._matrix_index(per_seed, split="val")
            across = e7.across_seed_table(per_seed)
            simple_seed, simple_across, factorial_seed, factorial_across = e7.contrast_tables(per_seed)
            material_seed, material_across = materiality_tables(per_seed)
            window_metrics = pd.concat(window_frames, ignore_index=True)
            subject_macro = e7.subject_macro_table(window_metrics)
            macro = subject_macro.loc[subject_macro.row_type.eq("macro")]
            macro_wide = macro.pivot(index=["arm", "seed"], columns="metric", values="subject_macro_mean").reset_index()
            macro_wide["split"] = "val"
            for metric in e7.PRIMARY:
                macro_wide.rename(columns={metric: metric + "_mean"}, inplace=True)
            macro_material_seed, _macro_across = materiality_tables(macro_wide)
            outputs = {
                "per_seed.csv": per_seed,
                "across_seed.csv": across,
                "simple_effects_per_seed.csv": simple_seed,
                "simple_effects_across_seed.csv": simple_across,
                "factorial_contrasts_per_seed.csv": factorial_seed,
                "factorial_contrasts_across_seed.csv": factorial_across,
                "materiality_per_seed.csv": material_seed,
                "materiality_across_seed.csv": material_across,
                "subject_macro.csv": subject_macro,
                "subject_macro_materiality_per_seed.csv": macro_material_seed,
            }
            for name, table in outputs.items():
                table.to_csv(output / name, index=False, na_rep="NA")
            e7.write_json(
                output / "summary_receipt.json",
                {
                    "protocol": P5_PROTOCOL,
                    "p4_execution_lock_sha256": P4_LOCK_SHA256,
                    "cells": len(per_seed),
                    "source_runs": sources,
                    "tables": {name: len(table) for name, table in outputs.items()},
                    "research_test_accessed": False,
                },
            )
    return output


class ScaleAccumulator:
    def __init__(self) -> None:
        self.variance_sum = 0.0
        self.variance_count = 0
        self.observation_count = 0
        self.scale_sum = np.zeros(97, dtype=np.float64)
        self.scale_cross = np.zeros((97, 97), dtype=np.float64)

    def update(self, value: torch.Tensor) -> None:
        work = value.detach().float()
        variance = work.var(dim=2, unbiased=False)
        self.variance_sum += float(variance.double().sum())
        self.variance_count += variance.numel()
        sampled = (
            work[:, DIAGNOSTIC_CHANNEL_INDICES, :, :][:, :, :, DIAGNOSTIC_TIME_INDICES]
            .permute(0, 1, 3, 2)
            .reshape(-1, 97)
            .cpu()
            .double()
            .numpy()
        )
        self.observation_count += len(sampled)
        self.scale_sum += sampled.sum(axis=0)
        self.scale_cross += sampled.T @ sampled

    def finalize(self) -> tuple[dict[str, float], np.ndarray]:
        if self.variance_count <= 0 or self.observation_count <= 1:
            raise ValueError("E7 P5 representation accumulator 为空")
        n = self.observation_count
        covariance = self.scale_cross - np.outer(self.scale_sum, self.scale_sum) / n
        diagonal = np.diag(covariance)
        if not np.isfinite(covariance).all() or np.any(diagonal <= 0):
            raise FloatingPointError("E7 P5 scale covariance 非有限或零方差")
        correlation = covariance / np.sqrt(np.outer(diagonal, diagonal))
        correlation = (correlation + correlation.T) / 2
        eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0, None)
        probabilities = eigenvalues / eigenvalues.sum()
        positive = probabilities > 0
        effective_rank = float(np.exp(-(probabilities[positive] * np.log(probabilities[positive])).sum()))

        def lag_mean(start: int, stop: int) -> float:
            return float(np.mean([np.diag(correlation, k=lag).mean() for lag in range(start, stop + 1)]))

        summary = {
            "cross_scale_variance": self.variance_sum / self.variance_count,
            "entropy_effective_rank": effective_rank,
            "correlation_lag_1": lag_mean(1, 1),
            "correlation_lag_8_16": lag_mean(8, 16),
            "correlation_lag_48_96": lag_mean(48, 96),
            "sampled_observations": n,
        }
        return summary, correlation


class RatioAccumulator:
    def __init__(self) -> None:
        self.numerator = 0.0
        self.denominator = 0.0

    def update(self, delta: torch.Tensor, reference: torch.Tensor) -> None:
        self.numerator += float(delta.detach().float().square().double().sum())
        self.denominator += float(reference.detach().float().square().double().sum())

    def finalize(self) -> float:
        if self.denominator <= 0:
            raise ZeroDivisionError("E7 P5 RMS reference 为零")
        return float(np.sqrt(self.numerator / self.denominator))


def diagnostic_conditions(arm: str) -> list[tuple[str, bool, bool]]:
    encoder, aggregation = arm.split("__", maxsplit=1)
    conditions = [("full", True, False)]
    if encoder != "s0_shallow":
        conditions.append(("residual_off", False, False))
    if aggregation == "frequency_attention":
        conditions.append(("uniform_attention", True, True))
    if encoder != "s0_shallow" and aggregation == "frequency_attention":
        conditions.append(("both_off", False, True))
    return conditions


def _environment_matches(formal_environment: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    keys = (
        "packages",
        "gpu_name",
        "cuda",
        "cudnn",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
        "deterministic_algorithms",
    )
    if any(formal_environment.get(key) != current.get(key) for key in keys):
        raise ValueError("E7 P5 diagnostic 与 formal 环境不一致")


def run_diagnostic(arm: str, seed: int, *, device: str = "cuda:0") -> Path:
    if arm not in ARMS or seed not in e7.SEEDS:
        raise ValueError("E7 P5 diagnostic cell 越界")
    runtime = runtime_identity(require_cuda=True, device=device)
    formal_attempt = next(
        path
        for path in p4.completed_runs()
        if json.loads((path / "formal_receipt.json").read_text())["arm"] == arm
        and int(json.loads((path / "formal_receipt.json").read_text())["seed"]) == seed
    )
    formal_receipt = json.loads((formal_attempt / "formal_receipt.json").read_text())
    run_dir = formal_attempt / formal_receipt["run_dir"]
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.training.device = device
    current_environment = p4.runtime_preflight(device)
    _environment_matches(
        json.loads((formal_attempt / "environment.json").read_text()),
        current_environment,
    )
    parent = e7.ROOT / OUTPUT / "diagnostic" / arm / f"seed_{seed}"
    with phase_guard(parent, f"diagnostic_{arm}_{seed}"):
        with attempt(parent, "diagnostic", runtime, arm=arm, seed=seed) as output:
            e7.write_json(output / "environment.json", current_environment)
            data = build_tho_data(cfg)
            rows = pd.read_csv(formal_attempt / "val_rows.csv")
            model = build_e7_model(cfg).to(device)
            checkpoint_path = run_dir / "checkpoint_best_local_rr.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            x0_accumulator = ScaleAccumulator()
            xe_accumulator = ScaleAccumulator()
            encoder_ratio = RatioAccumulator()
            correction_ratio = RatioAccumulator()
            block_ratios = [RatioAccumulator() for _ in range(len(model.e7_branch.scale_encoder.blocks))]
            attention_entropy_sum = attention_tv_sum = 0.0
            region_mass_sum = np.zeros(4, dtype=np.float64)
            attention_observations = 0
            with torch.no_grad():
                for batch in data.val.loader:
                    w = batch["tf"]["w"]
                    for start in range(0, len(w), DIAGNOSTIC_MICROBATCH):
                        chunk = w[start : start + DIAGNOSTIC_MICROBATCH].to(device)
                        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                            details = model.e7_branch.forward_features({"w": chunk})
                        x0_accumulator.update(details["x0"])
                        xe_accumulator.update(details["xe"])
                        encoder_ratio.update(details["xe"] - details["x0"], details["x0"])
                        correction_ratio.update(details["correction"], details["mean"])
                        for index, (block_input, block_output) in enumerate(details["block_records"]):
                            block_ratios[index].update(block_output - block_input, block_input)
                        if details["weights"] is not None:
                            stats = e7.attention_statistics(details["weights"])
                            count = int(details["weights"].shape[0] * details["weights"].shape[-1])
                            attention_entropy_sum += stats["attention_entropy"] * count
                            attention_tv_sum += stats["attention_total_variation"] * count
                            region_mass_sum += np.asarray(
                                [stats[f"region_{index}_mass"] for index in range(4)]
                            ) * count
                            attention_observations += count
            x0_summary, x0_correlation = x0_accumulator.finalize()
            xe_summary, xe_correlation = xe_accumulator.finalize()
            np.save(output / "x0_scale_correlation.npy", x0_correlation)
            np.save(output / "xe_scale_correlation.npy", xe_correlation)
            representation_rows = [
                {"arm": arm, "seed": seed, "stage": "x0", **x0_summary},
                {"arm": arm, "seed": seed, "stage": "xe", **xe_summary},
                {
                    "arm": arm,
                    "seed": seed,
                    "stage": "module_ratios",
                    "encoder_residual_rms_ratio": encoder_ratio.finalize(),
                    "attention_correction_rms_ratio": correction_ratio.finalize(),
                    **{
                        f"block_{index}_residual_rms_ratio": accumulator.finalize()
                        for index, accumulator in enumerate(block_ratios)
                    },
                    **(
                        {
                            "attention_entropy": attention_entropy_sum / attention_observations,
                            "attention_total_variation": attention_tv_sum / attention_observations,
                            **{
                                f"region_{index}_mass": region_mass_sum[index] / attention_observations
                                for index in range(4)
                            },
                        }
                        if attention_observations
                        else {}
                    ),
                },
            ]
            pd.DataFrame(representation_rows).to_csv(
                output / "representation_diagnostics.csv", index=False, na_rep="NA"
            )
            experiment = e7.ScaleFactorialExperiment(cfg, rows)
            experiment.device = torch.device(device)
            metric_frames = []
            full = pd.read_csv(run_dir / "metrics.csv")
            full.insert(0, "condition", "full")
            metric_frames.append(full)
            for condition, residual_enabled, uniform_attention in diagnostic_conditions(arm)[1:]:
                with model.intervention(
                    residual_enabled=residual_enabled,
                    uniform_attention=uniform_attention,
                ):
                    metrics = experiment._evaluate_model(model, data.val.loader)
                metrics.insert(0, "condition", condition)
                metric_frames.append(metrics)
            intervention_metrics = pd.concat(metric_frames, ignore_index=True)
            intervention_metrics.to_csv(output / "intervention_metrics.csv", index=False)
            summaries = []
            for condition, group in intervention_metrics.groupby("condition", sort=False):
                summary = summarize_task_metrics(group).iloc[0].to_dict()
                summaries.append({"arm": arm, "seed": seed, "condition": condition, **summary})
            intervention_summary = pd.DataFrame(summaries)
            intervention_summary.to_csv(output / "intervention_summary.csv", index=False)
            indexed = intervention_summary.set_index("condition")
            deltas = []
            for condition in indexed.index:
                if condition == "full":
                    continue
                for metric in e7.PRIMARY:
                    full_value = float(indexed.loc["full", metric + "_mean"])
                    changed = float(indexed.loc[condition, metric + "_mean"])
                    utility = changed - full_value if metric == e7.PCC else full_value - changed
                    deltas.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "condition": condition,
                            "metric": metric,
                            "full": full_value,
                            "intervened": changed,
                            "utility_delta": utility,
                            "positive_means": "intervention_improves",
                        }
                    )
            pd.DataFrame(deltas).to_csv(output / "intervention_delta.csv", index=False)
            e7.write_json(
                output / "diagnostic_receipt.json",
                {
                    "protocol": P5_PROTOCOL,
                    "arm": arm,
                    "seed": seed,
                    "formal_attempt": str(formal_attempt.resolve()),
                    "formal_manifest": e7.identity(formal_attempt / "manifest.json"),
                    "checkpoint": e7.identity(checkpoint_path),
                    "selected_epoch": formal_receipt["selected_epoch"],
                    "validation_rows": len(rows),
                    "conditions": [item[0] for item in diagnostic_conditions(arm)],
                    "representation_sampling": p5_contract()["representation_sampling"],
                    "research_test_accessed": False,
                },
            )
    return output


def completed_diagnostics() -> list[Path]:
    result = []
    for arm in ARMS:
        for seed in e7.SEEDS:
            parent = e7.ROOT / OUTPUT / "diagnostic" / arm / f"seed_{seed}"
            matches = []
            for freeze in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((freeze.parent / "manifest.json").read_text())
                if manifest.get("p4_execution_lock_sha256") == P4_LOCK_SHA256:
                    verify_attempt(freeze.parent, phase="diagnostic")
                    if manifest.get("arm") != arm or int(manifest.get("seed", -1)) != seed:
                        raise ValueError("E7 P5 diagnostic cell identity 漂移")
                    matches.append(freeze.parent.resolve())
            if len(matches) != 1:
                raise ValueError(f"E7 P5 diagnostic {arm}/{seed} 唯一成功数={len(matches)}")
            result.extend(matches)
    return result


def completed_summary() -> Path:
    parent = e7.ROOT / OUTPUT / "summary"
    matches = []
    for freeze in parent.glob("*/freeze_receipt.json"):
        manifest = json.loads((freeze.parent / "manifest.json").read_text())
        if manifest.get("p4_execution_lock_sha256") == P4_LOCK_SHA256:
            verify_attempt(freeze.parent, phase="summary")
            matches.append(freeze.parent.resolve())
    if len(matches) != 1:
        raise ValueError(f"E7 P5 summary 唯一成功数={len(matches)}")
    return matches[0]


def run_finalize() -> Path:
    runtime = runtime_identity()
    summary = completed_summary()
    diagnostics = completed_diagnostics()
    parent = e7.ROOT / OUTPUT / "final"
    with phase_guard(parent, "final"):
        with attempt(parent, "final", runtime) as output:
            pd.read_csv(summary / "per_seed.csv").to_csv(output / "seed_metrics.csv", index=False)
            pd.read_csv(summary / "factorial_contrasts_across_seed.csv").to_csv(
                output / "factorial_contrasts.csv", index=False
            )
            pd.read_csv(summary / "materiality_across_seed.csv").to_csv(
                output / "materiality.csv", index=False
            )
            pd.read_csv(summary / "subject_macro.csv").to_csv(
                output / "subject_macro.csv", index=False
            )
            representations = []
            interventions = []
            sources = {}
            for path in diagnostics:
                receipt = json.loads((path / "diagnostic_receipt.json").read_text())
                key = f"{receipt['arm']}/{receipt['seed']}"
                if key in sources:
                    raise ValueError("E7 P5 final diagnostic cell 重复")
                representations.append(pd.read_csv(path / "representation_diagnostics.csv"))
                interventions.append(pd.read_csv(path / "intervention_delta.csv"))
                sources[key] = {
                    "path": str(path),
                    "manifest": e7.identity(path / "manifest.json"),
                }
            pd.concat(representations, ignore_index=True).to_csv(
                output / "representation_diagnostics.csv", index=False, na_rep="NA"
            )
            pd.concat(interventions, ignore_index=True).to_csv(
                output / "checkpoint_interventions.csv", index=False
            )
            e7.write_json(
                output / "final_receipt.json",
                {
                    "protocol": P5_PROTOCOL,
                    "p4_execution_lock_sha256": P4_LOCK_SHA256,
                    "summary": {
                        "path": str(summary),
                        "manifest": e7.identity(summary / "manifest.json"),
                    },
                    "diagnostics": sources,
                    "cells": len(sources),
                    "research_test_accessed": False,
                },
            )
    return output
