"""固定 checkpoint 的 ε 推理预算实验及 plateau 选择。"""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from resp_train.data.factory import build_window_data
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.respdiff.model import finite
from .baseband import ROOT, SCHEMA, baseband_dataset, load_baseband_config, make_baseband_model
from .baseband_diagnostics import waveform_diagnostics
from .runtime import (LOGGER, Progress, check_primary_metrics, sha256, write_json)
from .sampling import SamplerSpec, ensemble_prefixes, reconstruct_ensemble_parent

NAME = "respdiff-bcg-epsilon-inference-v2-20261007"
COUNTS = {"ddim": (1, 2, 4, 8, 16), "ddpm": (1, 2, 4)}
CHECKPOINT_SHA256 = "08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6"


def load_sampler_config(path):
    actual = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    kind = actual.get("sampler", {}).get("type")
    if kind not in COUNTS:
        raise ValueError("未知 sampler 配置")
    reference = {"protocol": NAME,
        "sampler": {"type": kind, "nfe": 6 if kind == "ddim" else 50,
            "timesteps": [49, 39, 29, 19, 9, 0] if kind == "ddim" else list(range(49, -1, -1)),
            "n_trajectories": COUNTS[kind][-1], "noise_seed": 20261003, "eta": 0},
        "ensemble": {"nested_noise": True, "counts": list(COUNTS[kind])},
        "postprocessing": {"lowpass_hz": 1., "parent_fs": 100, "order": 8, "operator": "sosfiltfilt"},
        "diagnostic_subset": {"uniform_parents": 64, "forced_rows": [10382, 12226]},
        "batch_size": 64}
    if actual != reference:
        raise ValueError("配置偏离固定 ε-inference-v2 合同")
    return actual


def sampler_spec(config, *, count=None):
    values = config["sampler"]
    return SamplerSpec(sampler=values["type"], nfe=values["nfe"],
        n_trajectories=values["n_trajectories"] if count is None else count,
        noise_seed=values["noise_seed"], eta=values["eta"], postfilter=True)


def select_diagnostic_indices(rows, *, count=64, forced=(10382, 12226)):
    if len(rows) < count or rows.dataset_row_id.duplicated().any() or set(rows.split) != {"val"}:
        raise ValueError("diagnostic subset 要求足够且唯一的 validation parents")
    indices = set(np.linspace(0, len(rows) - 1, count, dtype=int).tolist())
    for row_id in forced:
        found = np.flatnonzero(rows.dataset_row_id.to_numpy() == row_id)
        if len(found) != 1:
            raise ValueError(f"强制 row {row_id} 缺失或重复")
        indices.add(int(found[0]))
    return sorted(indices)


class ParentSelection:
    def __init__(self, parents, rows, indices):
        self.parents, self.indices = parents, list(indices)
        self.rows = rows.iloc[self.indices].reset_index(drop=True).copy()
        self.index_csv_path = getattr(parents, "index_csv_path", "/")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.parents[self.indices[index]]


def load_frozen_source(source, device):
    source = Path(source).resolve()
    receipt = json.loads((source / "receipt.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("objective") != "source_equivalent":
        raise ValueError("来源必须是完成的 source-equivalent train run")
    for name in ("resolved_config.yaml", "val_parents.csv"):
        if sha256(source / name) != receipt["artifacts"][name]:
            raise ValueError(f"来源产物身份变化：{name}")
    checkpoint = source / "final.pt"
    digest = sha256(checkpoint)
    if digest != CHECKPOINT_SHA256 or digest != receipt["artifacts"]["final.pt"]:
        raise ValueError("来源 checkpoint 不匹配固定 seed20260811 final6400 身份")
    cfg = load_baseband_config(source / "resolved_config.yaml")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (state["schema"] != SCHEMA or state["update"] != 6400 or state["selector"] != "final_update"
            or state["experiment"]["objective"]["name"] != "source_equivalent"
            or state["experiment"]["seed"] != 20260811):
        raise ValueError("来源 checkpoint 元数据偏离合同")
    model = make_baseband_model(cfg)
    model.load_state_dict(state["model"], strict=True)
    for key, value in model.state_dict().items():
        finite(f"source checkpoint {key}", value)
    del state
    model.to(device).eval()
    return model, cfg, {"source_run": str(source), "checkpoint_sha256": digest,
                        "schema": SCHEMA, "update": 6400, "training_seed": 20260811}


def state_digest(model):
    import hashlib
    digest = hashlib.sha256()
    # 包含非持久化的调度/embedding buffer，覆盖推理可能意外改变的状态。
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        finite(name, value)
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validation_data(cfg, source):
    bundle = build_window_data(cfg, split="val", max_windows=cfg.data.max_val_windows,
        sample_strategy=str(cfg.data.val_sample_strategy), sample_seed=int(cfg.data.val_sample_seed), shuffle=False)
    expected = pd.read_csv(Path(source) / "val_parents.csv")
    identity = ["dataset_row_id", "samp_id", "split", "window_start_sample", "window_end_sample"]
    if not expected[identity].reset_index(drop=True).equals(bundle.rows[identity].reset_index(drop=True)):
        raise ValueError("当前 validation 身份/顺序与 checkpoint run 不一致")
    return bundle.dataset, bundle.rows, bundle.index_path


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_prefixes(model, dataset, spec, counts, output, *, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = next(model.parameters()).device
    accumulated = {count: [] for count in counts}
    elapsed = {count: 0. for count in counts}
    seen = []
    progress = Progress(f"{spec.sampler.upper()} nested ensemble", len(loader), every=1)
    LOGGER.info("%s N=%s：parents=%d chunks=%d batch=%d", spec.sampler, list(counts),
                len(dataset.rows), len(dataset), batch_size)
    for batch_index, batch in enumerate(loader, 1):
        indices = batch["index"].tolist()
        keys = []
        for index in indices:
            parent, chunk = divmod(index, 13)
            row = dataset.rows.iloc[parent]
            keys.append((str(row.split), int(row.dataset_row_id), chunk))
        condition = batch["x"].to(device)
        _synchronize(device)
        start = time.perf_counter()
        for count, mean in ensemble_prefixes(model, condition, keys, spec, checkpoints=counts):
            _synchronize(device)
            elapsed[count] += time.perf_counter() - start
            accumulated[count].append(mean.cpu().numpy())
        seen.extend(indices)
        progress.update(batch_index, f"N={spec.n_trajectories} chunks={len(seen)}/{len(dataset)}")
    if seen != list(range(len(dataset))):
        raise ValueError("inference chunk 身份缺失、重复或错序")
    result = {count: np.concatenate(parts).reshape(len(dataset.rows), 13, 1, 600)
              for count, parts in accumulated.items()}
    profiles = [{"sampler": spec.sampler, "N": count, "nfe_per_trajectory": spec.nfe,
        "denoiser_calls_per_chunk": count * spec.nfe,
        "batched_denoiser_forward_calls": len(loader) * count * spec.nfe,
        "sampling_seconds": elapsed[count], "n_parents": len(dataset.rows),
        "n_chunks": len(dataset), "batch_size": batch_size,
        "runtime_definition": "cumulative prefix wall time within shared max-N execution"} for count in counts]
    write_json(output / "sampling_identity.json", {"spec": asdict(spec), "timesteps": list(spec.timesteps),
        "counts": list(counts), "ordered_chunk_indices": seen, "profiles": profiles})
    return result, profiles


def output_diagnostics(predictions, targets, rows):
    records = []
    for prediction, target, row in zip(predictions, targets, rows.itertuples()):
        pred = waveform_diagnostics(prediction)
        ref = waveform_diagnostics(target)
        if ref["std"] == 0:
            raise ValueError("正式推理诊断出现零 target std；不允许定义外的比值缩放")
        records.append({"dataset_row_id": int(row.dataset_row_id), "samp_id": int(row.samp_id),
            **{f"prediction_{key}": value for key, value in pred.items()},
            **{f"target_{key}": value for key, value in ref.items()},
            "prediction_target_std_ratio": pred["std"] / ref["std"]})
    return pd.DataFrame(records)


def evaluate_setting(chunks, dataset, cfg, output, spec, count):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw, filtered, targets = [], [], []
    for index, parent in enumerate(chunks):
        _, prediction, post = reconstruct_ensemble_parent(parent, postfilter=spec.postfilter)
        raw.append(prediction)
        filtered.append(post)
        targets.append(dataset.parents[index]["target"].numpy().reshape(18000))
    raw, filtered, targets = map(np.stack, (raw, filtered, targets))
    identifiers = {"split": dataset.rows.split.to_numpy(dtype=str),
        **{key: dataset.rows[key].to_numpy(dtype=np.int64) for key in ("dataset_row_id", "samp_id")}}
    np.savez(output / "validation_raw_waveforms.npz", prediction_raw=raw, tho_ref=targets, **identifiers)
    np.savez(output / "validation_postfiltered_waveforms.npz", prediction_postfiltered=filtered,
             tho_ref=targets, **identifiers)
    curve = {"sampler": spec.sampler, "N": count, "nfe_per_trajectory": spec.nfe,
             "denoiser_calls_per_chunk": count * spec.nfe, "n_parents": len(dataset.rows)}
    for view, values in (("raw", raw), ("postfiltered", filtered)):
        diagnostics = output_diagnostics(values, targets, dataset.rows)
        diagnostics.to_csv(output / f"validation_{view}_diagnostics.csv", index=False)
        payload = {"r_tho_hat": values, "tho_ref": targets, **identifiers}
        metric_parts = []
        for start in range(0, len(dataset.rows), 64):
            part = {key: value[start:start + 64] for key, value in payload.items()}
            metric_parts.append(evaluate_task_predictions(part, cfg,
                method=f"RespDiff-epsilon-inference-v2/{spec.sampler}/N{count}/{view}", include_test_only=False))
        metrics = pd.concat(metric_parts, ignore_index=True)
        check_primary_metrics(metrics)
        metrics.to_csv(output / f"validation_{view}_per_parent.csv", index=False)
        summary = summarize_task_metrics(metrics)
        summary.to_csv(output / ("validation_summary.csv" if view == "postfiltered"
                                else "validation_raw_summary.csv"), index=False)
        for metric in ("whole_rr_abs_error_bpm", "local_rr_mae_bpm", "envelope_trajectory_mae",
                       "global_envelope_modulation_error", "lag_aware_signed_pcc"):
            curve[f"{view}_{metric}"] = float(summary.iloc[0][f"{metric}_mean"])
        maxima = diagnostics.prediction_max_abs
        curve.update({f"{view}_parent_max_p95": float(maxima.quantile(.95)),
                      f"{view}_parent_max_p99": float(maxima.quantile(.99)),
                      f"{view}_global_max": float(maxima.max()),
                      f"{view}_prediction_target_std_mean": float(diagnostics.prediction_target_std_ratio.mean()),
                      f"{view}_prediction_target_std_median": float(diagnostics.prediction_target_std_ratio.median())})
        for key in ("resp_energy_ratio", "above_1hz_energy_ratio", "top_1pct_sample_energy_ratio"):
            series = diagnostics[f"prediction_{key}"]
            curve[f"{view}_{key}"] = float(series.mean()) if series.notna().any() else None
            curve[f"{view}_{key}_n_defined"] = int(series.notna().sum())
        for row_id in (10382, 12226):
            found = np.flatnonzero(dataset.rows.dataset_row_id.to_numpy() == row_id)
            if len(found) == 1 and view == "postfiltered":
                index = int(found[0])
                np.savez(output / f"row_{row_id}_waveforms.npz", prediction_raw=raw[index],
                         prediction_postfiltered=filtered[index], tho_ref=targets[index],
                         dataset_row_id=row_id, sampler=spec.sampler, N=count)
    curve["reconstruction_metrics_seconds"] = time.perf_counter() - started
    write_json(output / "setting_summary.json", curve)
    LOGGER.info("%s N=%d：post PCC=%.5f WholeRR=%.5f P99max=%.5f raw max=%.5f",
        spec.sampler, count, curve["postfiltered_lag_aware_signed_pcc"],
        curve["postfiltered_whole_rr_abs_error_bpm"], curve["postfiltered_parent_max_p99"], curve["raw_global_max"])
    return curve


def _relative_change(previous, following):
    if not np.isfinite([previous, following]).all():
        raise ValueError("plateau 比较值非有限")
    if previous == 0:
        return 0. if following == 0 else None
    return abs(following - previous) / abs(previous)


def choose_ensemble(curve, *, counts=COUNTS["ddim"]):
    rows = curve.loc[curve.sampler == "ddim"].set_index("N")
    if set(rows.index) != set(counts) or not rows.index.is_unique:
        raise ValueError("plateau 选择要求完整且唯一的 DDIM N 曲线")
    comparisons = []
    chosen = None
    for count in counts:
        if count < 2 or 2 * count not in rows.index:
            continue
        left, right = rows.loc[count], rows.loc[2 * count]
        relative = {key: _relative_change(left[key], right[key]) for key in (
            "postfiltered_parent_max_p99", "postfiltered_top_1pct_sample_energy_ratio",
            "postfiltered_whole_rr_abs_error_bpm")}
        pcc_change = abs(right.postfiltered_lag_aware_signed_pcc - left.postfiltered_lag_aware_signed_pcc)
        passed = all(value is not None and value < .05 for value in relative.values()) and pcc_change < .01
        comparisons.append({"N": int(count), "next_N": 2 * int(count), "relative_changes": relative,
                            "signed_pcc_absolute_change": float(pcc_change), "plateau": bool(passed)})
        if passed and chosen is None:
            chosen = int(count)
    return {"selected_N": chosen if chosen is not None else max(counts),
        "status": "plateau_found" if chosen is not None else f"ensemble not converged by N={max(counts)}",
        "selection_view": "postfiltered", "comparisons": comparisons,
        "rule": "smallest N>=2: P99/top1%/WholeRR relative changes <.05 and abs PCC change <.01"}
