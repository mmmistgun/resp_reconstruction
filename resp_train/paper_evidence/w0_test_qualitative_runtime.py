"""固定 W0 单 checkpoint 导出与独立绘图生命周期。"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from resp_train.paper_evidence.w0_test_qualitative import (
    ANCHOR_ATOL, EPOCH, METHODS, PRIMARY, PROTOCOL, SEED, anchor_deltas,
    render_window, save_arrays, sha256, waveform_details, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_LOCKS = {
    "w0_film_gamma_training_implementation_lock_20260918.json": "2bf4b72a15111e1caa76b6bed19abbde3242edc451795176c307d160cc49368c",
    "w0_film_gamma_test_lock_20260919.json": "0d2bbbcc5ee6fee8e442475600c450de55c8e03ba31f18f5043e0557ac1ae712",
}
COUNT = 2310


def verify(path: Path, expected: dict[str, Any]) -> None:
    if (not path.is_file() or path.stat().st_size != expected["size_bytes"]
            or sha256(path) != expected["sha256"]):
        raise RuntimeError(f"来源身份不一致: {path}")


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    locks = []
    for name, expected in SOURCE_LOCKS.items():
        path = ROOT / "docs/experiments" / name
        if sha256(path) != expected:
            raise RuntimeError(f"冻结来源锁漂移: {name}")
        locks.append(json.loads(path.read_text()))
    train, test = locks
    baseline = next(item for item in train["baseline_runs"] if item["seed"] == SEED)
    entry = next(item for item in test["entries"] if item["seed"] == SEED)
    if baseline["selected_epoch"] != EPOCH or entry["baseline_selected_epoch"] != EPOCH:
        raise RuntimeError("W0 epoch 漂移")
    return baseline, test, entry


def check_rows(rows: pd.DataFrame, reference: pd.DataFrame, expected_hash: str) -> None:
    if len(rows) != COUNT or rows.dataset_row_id.duplicated().any() or rows.samp_id.nunique() != 8:
        raise ValueError("test rows 数量、唯一性或主体数不一致")
    if set(rows.split) != {"test"}:
        raise ValueError("只允许 test")
    digest = hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError("test row-order 漂移")
    for key in ("dataset_row_id", "samp_id", "coupling_state_id", "split"):
        if not np.array_equal(rows[key].to_numpy(), reference[key].to_numpy()):
            raise ValueError(f"test row identity 不一致: {key}")


def finish(output: Path, receipt: dict[str, Any]) -> None:
    write_json(output / "receipt.json", {"protocol": PROTOCOL, "status": "complete", **receipt})
    write_json(output / "artifact_manifest.json", {
        "protocol": PROTOCOL, "status": "complete",
        "files": {str(p.relative_to(output)): {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(output.rglob("*")) if p.is_file()},
    })


def export(output: Path, *, device: str, command: str) -> Path:
    """仅由显式确认的 CLI 调用。所有真实 test 访问发生在 access_started 之后。"""
    import torch
    from omegaconf import OmegaConf
    from resp_train.baselines.fixed_band import FIXED_BAND_EXPECTED_SIGNAL_KEY, FIXED_BAND_SOURCE_COLUMN
    from resp_train.baselines.iewt import IEWTConfig, IEWT_EXPECTED_SIGNAL_KEY, extract_respiration_iewt
    from resp_train.crd.config import check_crd_dependencies
    from resp_train.crd.experiment import _validate_checkpoint_config
    from resp_train.crd.model import build_crd_model
    from resp_train.crd.tf_v1_data import batch_tf_to_device
    from resp_train.crd.tf_v1_features import morlet_scales_and_frequencies
    from resp_train.data.factory import build_window_data
    from resp_train.data.index import filter_index
    from resp_train.data.research_v2 import read_research_v2_index
    from resp_train.engine.train import _waveform_output
    from resp_train.metrics.task import evaluate_task_predictions
    from resp_train.paper_evidence.w0_cwt_film_behavior import compute_batch_statistics, forward_with_capture
    from resp_train.paper_evidence.w0_cwt_film_behavior_runtime import environment

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or resolved_device.index is None or not torch.cuda.is_available():
        raise ValueError("真实导出要求显式可用 cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("; ".join(problems))
    baseline, test, entry = sources()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "execution.json", {
            "command": command, "seed": SEED, "epoch": EPOCH,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
            "environment": environment(device), "iewt_parameters": asdict(IEWTConfig()),
        })
        # 保存实际使用的源码，容纳本地尚未提交的实现，同时不改写工作树。
        code = list((ROOT / "resp_train").rglob("*.py")) + [
            ROOT / "scripts/export_w0_test_qualitative.py",
            ROOT / "docs/experiments/w0_test_qualitative_export_plan_20260927.md",
        ]
        for path in code:
            dest = output / "source_code" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        code_records = [record(p) for p in code]
        for item in code_records:
            verify(output / "source_code" / Path(item["path"]).relative_to(ROOT), item)
        run = Path(baseline["run_dir"])
        records = []
        for key, filename in (("checkpoint", "checkpoint_best_local_rr.pt"), ("config", "config.yaml"),
                              ("history", "train_history.csv"), ("manifest", "run_manifest.json")):
            path = run / filename
            verify(path, baseline[key])
            records.append(record(path))
        for item in entry["baseline_test"].values():
            verify(Path(item["path"]), item)
            records.append(item)
        cfg = OmegaConf.load(run / "config.yaml")
        OmegaConf.resolve(cfg)
        if (cfg.model.variant != "crd_tf102_w" or int(cfg.training.seed) != SEED
                or int(cfg.model.initialization_seed) != SEED or cfg.data.max_test_windows is not None
                or int(cfg.data.test_sample_seed) != 20260612 or int(cfg.training.batch_size) != 128
                or not cfg.training.use_amp or cfg.training.amp_dtype != "bfloat16"
                or bool(cfg.data.drop_nonfinite_windows)):
            raise ValueError("W0 科学配置不符合固定导出合同")
        OmegaConf.save(cfg, output / "training_config.yaml")
        checkpoint = torch.load(run / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False)
        _validate_checkpoint_config(checkpoint.get("config"), cfg)
        if checkpoint["epoch"] != EPOCH:
            raise ValueError("checkpoint epoch 漂移")
        if any(not torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values() if torch.is_tensor(v)):
            raise FloatingPointError("checkpoint 非有限")
        model = build_crd_model(cfg)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model.to(resolved_device).eval()
        cfg.training.device = device
        cfg.training.show_progress = False
        cfg.data.preload_windows = False
        cfg.data.tf_research_test_cache_path = test["cache_root"]
        OmegaConf.save(cfg, output / "resolved_config.yaml")
        write_json(output / "access_started.json", {
            "protocol": PROTOCOL, "split": "test", "seed": SEED, "expected_rows": COUNT,
            "purpose": "fixed-checkpoint qualitative export", "checkpoint": baseline["checkpoint"],
            "source_locks": SOURCE_LOCKS, "cache_root": test["cache_root"],
        })
        cache = Path(test["cache_root"])
        if sha256(cache / "cache_manifest.json") != test["cache_manifest_sha256"]:
            raise RuntimeError("cache manifest 漂移")
        records.append(record(cache / "cache_manifest.json"))
        for name, expected in test["cache_files"].items():
            verify(cache / name, expected)
            records.append(record(cache / name))
        verify(Path(test["dataset_index"]["path"]), test["dataset_index"])
        records.append(test["dataset_index"])
        audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
        rows = filter_index(audited, cfg, split="test", max_windows=None,
                            sample_strategy=str(cfg.data.test_sample_strategy), sample_seed=20260612)
        reference = pd.read_csv(entry["baseline_test"]["research_test_metrics.csv"]["path"])
        check_rows(rows, reference, test["row_order_sha256"])
        development = set(audited.loc[audited.split.isin(["train", "val"]), "samp_id"].astype(int))
        if set(rows.samp_id.astype(int)) & (development | set(entry["development_samp_ids"])):
            raise ValueError("test 与 train/val 主体交叉")
        if set(rows.bcg_signal_key) != {IEWT_EXPECTED_SIGNAL_KEY}:
            raise ValueError("宽带 BCG 信号层级漂移")
        if set(rows[FIXED_BAND_SOURCE_COLUMN]) != {FIXED_BAND_EXPECTED_SIGNAL_KEY}:
            raise ValueError("F0 信号层级漂移")
        rows.to_csv(output / "test_rows.csv", index=False)
        source_npzs = set()
        index_parent = Path(test["dataset_index"]["path"]).parent
        for key in ("source_npz", "target_source_npz"):
            source_npzs.update((index_parent / str(v)).resolve() for v in rows[key].unique())
        # 原始 NPZ 首尾哈希保证一次导出期间来源不变；全部文件只读。
        records.extend(record(path) for path in sorted(source_npzs))
        frequencies = np.load(cache / "w_frequencies_hz.npy", allow_pickle=False)
        scales, actual = morlet_scales_and_frequencies(12, 97)
        order = np.argsort(actual, kind="stable")
        if not np.array_equal(actual[order], frequencies):
            raise ValueError("CWT 实际频率映射与 cache 不一致")
        save_arrays(output / "coordinates.npz", {
            "cwt_frequency_hz": frequencies, "cwt_scales": scales[order],
            "cwt_time_s": (np.arange(360) * 50 + 24.5) / 100,
            "latent_time_s": (np.arange(1800) + .5) / 10,
            "film_time_bin_centers_s": (np.arange(36) + .5) * 5,
        })
        data = build_window_data(cfg, split="test", max_windows=None,
                                 sample_strategy=str(cfg.data.test_sample_strategy), sample_seed=20260612,
                                 shuffle=False, audited=audited)
        check_rows(data.rows, rows, test["row_order_sha256"])
        if len(data.dataset) != COUNT:
            raise ValueError("dataset 缩小了样本集合")
        (output / "windows").mkdir()
        metric_frames, film_frames, index_records = [], [], []
        offset = 0
        with torch.inference_mode():
            for batch in data.loader:
                n = len(batch["x"])
                subset = rows.iloc[offset:offset+n]
                ids = batch["meta"]["dataset_row_id"].numpy()
                if not np.array_equal(ids, subset.dataset_row_id.to_numpy()):
                    raise ValueError("batch row order 漂移")
                for value in (batch["x"], batch["target"], batch["tf"]["w"]):
                    if not torch.isfinite(value).all():
                        raise FloatingPointError("batch 非有限")
                if tuple(batch["tf"]["w"].shape) != (n, 97, 360):
                    raise ValueError("W shape 漂移")
                tf = batch_tf_to_device(batch, resolved_device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    result, captured = forward_with_capture(model, batch["x"].to(resolved_device), tf=tf)
                pred = _waveform_output(result).float().cpu().numpy().reshape(n, -1)
                stats = compute_batch_statistics(captured)
                # 有效系数保持原生 dtype 运算，转 FP32 后保存，避免重新舍入改变解释。
                g = .5 * torch.tanh(captured.gamma_raw)
                b = .5 * torch.tanh(captured.beta_raw)
                tensor_values = {
                    "gamma_raw": captured.gamma_raw, "beta_raw": captured.beta_raw,
                    "g": g, "b": b, "z": captured.z, "z_prime": captured.z_prime,
                    "scale_delta": g.float() * captured.z.float(),
                    "total_delta": captured.z_prime.float() - captured.z.float(),
                }
                film = {key: value.float().cpu().numpy() for key, value in tensor_values.items()}
                if offset == 0:
                    write_json(output / "tensor_semantics.json", {
                        "native_dtypes": {k: str(v.dtype) for k, v in tensor_values.items()},
                        "storage": "float32 lossless promotion of native tensors",
                        "film_shape": [96, 1800], "film_formula": "Z_prime = Z * (1 + g) + b",
                        "scale_delta": "float32(g) * float32(Z); descriptive decomposition",
                        "time_coordinates": "CWT: pooled sample centers; latent: nominal 0.1 s bin centers",
                        "latent_time_caveat": "nominal centers, not causal or frequency attribution",
                        "waveform_order": ["reference", *METHODS], "evaluation_interval_s": [0, 180],
                        "seed_selection": "chosen from existing test results for qualitative illustration",
                    })
                film_frame = pd.DataFrame(stats.window)
                film_frame.insert(0, "dataset_row_id", ids)
                film_frames.append(film_frame)
                del tensor_values, captured, result, tf, g, b
                bcg = batch["x"].numpy().reshape(n, -1)
                target = batch["target"].numpy().reshape(n, -1)
                f0, iewt_results = [], []
                for i, (_, row) in enumerate(subset.iterrows()):
                    raw = data.dataset.source_cache.get_arrays(str(row.source_npz), [FIXED_BAND_EXPECTED_SIGNAL_KEY])
                    f0.append(raw[FIXED_BAND_EXPECTED_SIGNAL_KEY][int(row.window_start_sample):int(row.window_end_sample)].astype(np.float32))
                    iewt_results.append(extract_respiration_iewt(bcg[i], fs=100))
                candidates = {"F0": np.stack(f0), "IEWT": np.stack([r.waveform for r in iewt_results]), "W0": pred}
                metadata = {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in batch["meta"].items()}
                for method, prediction in candidates.items():
                    frame = evaluate_task_predictions({"r_tho_hat": prediction, "tho_ref": target, **metadata},
                                                      cfg, include_test_only=False, method=method)
                    if not np.isfinite(frame[list(PRIMARY)].to_numpy(float)).all():
                        raise FloatingPointError(f"{method} 五主指标非有限")
                    if method == "IEWT":
                        for key in ("boundary_bins", "boundary_hz", "selected_band_indices"):
                            frame[f"iewt_{key}_by_block"] = [
                                json.dumps([getattr(w, key).tolist() for w in r.windows]) for r in iewt_results
                            ]
                    metric_frames.append(frame)
                for i, (_, row) in enumerate(subset.iterrows()):
                    row_id = int(ids[i])
                    wave = np.stack([target[i], *[candidates[m][i] for m in METHODS]])
                    arrays = {"dataset_row_id": np.asarray(row_id), "fs_hz": np.asarray(100.),
                              "bcg": bcg[i], "waveforms": wave, "cwt_w": batch["tf"]["w"][i].numpy(),
                              "rr_peak_valid_mask": batch["meta"]["rr_peak_valid_mask"][i].numpy(),
                              **{k: v[i] for k, v in film.items()}, **waveform_details(wave, cfg)}
                    z_norm = np.linalg.norm(arrays["z"], axis=0)
                    arrays["latent_low_energy"] = z_norm / np.sqrt(96) <= 1e-6
                    for key, value in (("scale", arrays["scale_delta"]), ("shift", arrays["b"]), ("total", arrays["total_delta"])):
                        arrays[f"r_{key}_time"] = np.linalg.norm(value, axis=0) / (z_norm + 1e-12)
                    arrays.update({f"film_time_bin_{k}": v[i] for k, v in stats.time_bin.items()})
                    arrays.update({f"film_channel_{k}": v[i] for k, v in stats.channel.items()})
                    filename = f"windows/row_{row_id}.npz"
                    save_arrays(output / filename, arrays)
                    index_records.append({"dataset_row_id": row_id, "samp_id": int(row.samp_id),
                                          "window_start_s": float(row.window_start_s),
                                          "window_end_s": float(row.window_end_s), "file": filename})
                offset += n
                print(f"exported {offset}/{COUNT} windows", flush=True)
        if offset != COUNT:
            raise ValueError("导出窗口不完整")
        metrics = pd.concat(metric_frames, ignore_index=True)
        metrics.insert(0, "seed", SEED)
        metrics.to_csv(output / "metrics.csv", index=False)
        pd.concat(film_frames, ignore_index=True).to_csv(output / "film_statistics.csv", index=False)
        deltas = anchor_deltas(metrics.loc[metrics.method == "W0"], reference)
        deltas.to_csv(output / "anchor_deltas.csv", index=False)
        if not deltas.within_atol.all():
            raise RuntimeError(f"冻结 test 指标回放失败，最大差值 {deltas.abs_delta.max():.8g}，容差 {ANCHOR_ATOL}")
        index = pd.DataFrame(index_records)
        for method in METHODS:
            view = metrics.loc[metrics.method == method, ["dataset_row_id", *PRIMARY]].rename(
                columns={k: f"{method}_{k}" for k in PRIMARY})
            index = index.merge(view, on="dataset_row_id", validate="one_to_one")
        for method in ("F0", "IEWT"):
            for key in PRIMARY:
                index[f"W0_minus_{method}_{key}"] = index[f"W0_{key}"] - index[f"{method}_{key}"]
        index.to_csv(output / "window_index.csv", index=False)
        for item in [*records, *code_records]:
            verify(Path(item["path"]), item)
        write_json(output / "source_manifest.json", {"source_locks": SOURCE_LOCKS, "files": records, "code": code_records})
        finish(output, {"phase": "export", "seed": SEED, "epoch": EPOCH, "rows": COUNT,
                        "subjects": 8, "anchor_atol": ANCHOR_ATOL,
                        "anchor_max_abs_delta": float(deltas.abs_delta.max()),
                        "test_read": True, "model_inference": True, "training": False})
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
    return output


def render(source: Path, output: Path, *, row_ids: list[int] | None,
           zoom: tuple[float, float] | None, command: str) -> Path:
    """只读完成的导出；不回访数据集、cache 或模型。"""
    source, output = source.resolve(), output.resolve()
    manifest = json.loads((source / "artifact_manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("status") != "complete":
        raise ValueError("需要完整的本协议导出")
    if output == source or source in output.parents:
        raise ValueError("绘图输出必须使用导出目录之外的独立 identity")
    def verified(relative: str) -> Path:
        path = (source / relative).resolve()
        if source not in path.parents:
            raise ValueError("导出路径越界")
        verify(path, manifest["files"][relative])
        return path
    receipt = json.loads(verified("receipt.json").read_text())
    if receipt.get("phase") != "export" or receipt.get("seed") != SEED:
        raise ValueError("需要固定 seed 的 export receipt")
    index = pd.read_csv(verified("window_index.csv"))
    metrics = pd.read_csv(verified("metrics.csv"))
    if index.dataset_row_id.duplicated().any():
        raise ValueError("绘图索引 row 重复")
    if row_ids is not None:
        if len(set(row_ids)) != len(row_ids) or not set(row_ids) <= set(index.dataset_row_id):
            raise ValueError("指定 rows 重复或不存在")
        index = index.set_index("dataset_row_id").loc[row_ids].reset_index()
    with np.load(verified("coordinates.npz"), allow_pickle=False) as blob:
        coordinates = dict(blob)
    output.mkdir(parents=True, exist_ok=False)
    try:
        table = []
        for _, row in index.iterrows():
            row_id = int(row.dataset_row_id)
            with np.load(verified(str(row.file)), allow_pickle=False) as blob:
                arrays = {**dict(blob), **coordinates}
            paths = render_window(arrays, metrics.loc[metrics.dataset_row_id == row_id], output / "figures",
                                  row_id=row_id, subject=int(row.samp_id), start_s=float(row.window_start_s),
                                  zoom=zoom)
            links = " ".join(f'<a href="{html.escape(str(p.relative_to(output)))}">{html.escape(p.stem.split("_", 2)[2])}.{p.suffix[1:]}</a>' for p in paths)
            table.append(f"<tr><td>{row_id}</td><td>{int(row.samp_id)}</td><td>{row.window_start_s:g}</td>"
                         f"<td>{row.W0_lag_aware_signed_pcc:.4f}</td><td>{row.W0_local_rr_mae_bpm:.4f}</td><td>{links}</td></tr>")
        page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>W0 定性分析</title>'
                '<style>body{font:16px system-ui;margin:2rem}td,th{padding:.5rem;border-bottom:1px solid #ddd}a{margin-right:1rem}</style>'
                '<h1>W0 seed 20260812 定性分析</h1><p>完整检索指标与相对基线差值：<a href="window_index.csv">CSV</a>。'
                '展示 seed 按已有测试结果选择。图中指标计算于完整 180 s。</p>'
                '<input id="q" placeholder="检索 row、主体、时间或指标" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.textContent.includes(this.value))">'
                '<table><thead><tr><th>Row</th><th>主体</th><th>开始时间(s)</th><th>PCC</th><th>Local RR MAE</th><th>图</th></tr></thead><tbody>'
                + "\n".join(table) + '</tbody></table></html>')
        (output / "index.html").write_text(page, encoding="utf-8")
        index.to_csv(output / "window_index.csv", index=False)
        finish(output, {"phase": "render", "command": command, "source": str(source),
                        "source_manifest_sha256": sha256(source / "artifact_manifest.json"),
                        "rows": index.dataset_row_id.astype(int).tolist(), "format": "png",
                        "zoom": zoom, "original_dataset_read": False, "model_inference": False})
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
    return output
