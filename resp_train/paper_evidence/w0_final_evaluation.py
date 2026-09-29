"""原始 W0 三 seed 的最终指标评价；新目录、单次完整生命周期。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np
import pandas as pd

from resp_train.metrics.final_evaluation import (
    EvaluationFailure, METRICS, evaluate_window, select_nonoverlap_centers,
    summarize_seeds, summarize_windows,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "w0-final-evaluation-v1-20260927"
PROTOCOL_PATH = "docs/experiments/w0_final_evaluation_protocol_20260927.md"
SEEDS = (20260811, 20260812, 20260813)
EPOCHS = (13, 15, 14)
COUNT, SUBJECTS = 2310, 8
SOURCE_LOCKS = {
    "w0_film_gamma_training_implementation_lock_20260918.json": "2bf4b72a15111e1caa76b6bed19abbde3242edc451795176c307d160cc49368c",
    "w0_film_gamma_test_lock_20260919.json": "0d2bbbcc5ee6fee8e442475600c450de55c8e03ba31f18f5043e0557ac1ae712",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def verify(path: Path, expected: dict) -> None:
    if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
        raise EvaluationFailure(f"来源文件身份漂移: {path}")


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_sources() -> tuple[list[dict], dict]:
    locks = []
    for name, expected in SOURCE_LOCKS.items():
        path = ROOT / "docs/experiments" / name
        if sha256(path) != expected:
            raise EvaluationFailure(f"历史来源锁漂移: {name}")
        locks.append(json.loads(path.read_text()))
    train, test = locks
    baselines = sorted(train["baseline_runs"], key=lambda item: item["seed"])
    if tuple(item["seed"] for item in baselines) != SEEDS:
        raise EvaluationFailure("W0 来源 seed 漂移")
    if tuple(item["selected_epoch"] for item in baselines) != EPOCHS:
        raise EvaluationFailure("W0 固定 checkpoint epoch 漂移")
    if test["count"] != COUNT or test["samp_id_count"] != SUBJECTS or tuple(test["seeds"]) != SEEDS:
        raise EvaluationFailure("W0 冻结 test 身份错误")
    return baselines, test


def check_rows(rows: pd.DataFrame, reference: pd.DataFrame, expected_hash: str) -> None:
    if len(rows) != COUNT or rows.dataset_row_id.duplicated().any() or rows.samp_id.nunique() != SUBJECTS:
        raise EvaluationFailure(f"test 必须完整包含 {COUNT} 唯一窗口、{SUBJECTS} 个 samp_id")
    if set(rows.split) != {"test"}:
        raise EvaluationFailure("评价 split 错误")
    if hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest() != expected_hash:
        raise EvaluationFailure("test row-order 漂移")
    for key in ("dataset_row_id", "samp_id", "coupling_state_id", "split"):
        if not np.array_equal(rows[key].to_numpy(), reference[key].to_numpy()):
            raise EvaluationFailure(f"test 历史来源窗口不一致: {key}")
    times = rows[["window_start_s", "window_end_s"]].to_numpy(float) * 100
    samples = rows[["window_start_sample", "window_end_sample"]].to_numpy(float)
    if not np.isfinite(times).all() or not np.allclose(times, samples, rtol=0, atol=1e-6):
        raise EvaluationFailure("绝对秒时间与采样点不一致")


def check_coverage(observed: list[int], expected: list[int]) -> None:
    if observed != expected or len(set(observed)) != len(observed):
        raise EvaluationFailure("模型未按统一顺序完整覆盖测试窗口")


def evaluate(output: Path, *, device: str, command: str) -> Path:
    """只由显式 test CLI 调用；失败保留原目录，禁止原 identity 重试。"""
    import torch
    from omegaconf import OmegaConf
    from resp_train.crd.experiment import _validate_checkpoint_config
    from resp_train.crd.model import build_crd_model
    from resp_train.crd.tf_v1_data import batch_tf_to_device
    from resp_train.data.factory import build_window_data
    from resp_train.data.index import filter_index
    from resp_train.data.research_v2 import read_research_v2_index
    from resp_train.engine.train import _waveform_output
    from resp_train.paper_evidence.w0_cwt_film_behavior_runtime import environment

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or resolved_device.index is None or not torch.cuda.is_available():
        raise ValueError("固定 BF16 推理要求可用的显式 cuda:<index>")
    baselines, test = load_sources()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    context = {"method": "W0", "seed": None, "checkpoint": None, "dataset_row_id": None, "stage": "setup"}
    try:
        write_json(output / "access_started.json", {
            "protocol": PROTOCOL, "split": "test", "seeds": list(SEEDS), "epochs": list(EPOCHS),
            "source_locks": SOURCE_LOCKS, "command": command,
            "purpose": "用户执行固定 W0 三 seed 最终指标评价", "status": "started",
        })
        write_json(output / "execution.json", {
            "command": command, "environment": environment(device),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
            "batch_size": 128, "amp_dtype": "bfloat16", "sample_seed": 20260612,
        })
        records = {}

        def track(path: Path, expected: dict | None = None) -> None:
            path = path.resolve()
            if expected is not None:
                verify(path, expected)
            records[str(path)] = record(path)

        code = sorted((ROOT / "resp_train").rglob("*.py")) + [
            ROOT / "scripts/eval_w0_final_metrics.py", ROOT / PROTOCOL_PATH,
            ROOT / "tests/test_final_evaluation.py",
        ]
        for path in code:
            destination = output / "source_code" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            track(path)
            shutil.copy2(path, destination)
            verify(destination, records[str(path.resolve())])
        for name in SOURCE_LOCKS:
            track(ROOT / "docs/experiments" / name)
        configs = []
        references = []
        # 三个来源在首次 forward 前全部验证，不根据中途结果调整矩阵。
        for baseline in baselines:
            run = Path(baseline["run_dir"])
            for key, filename in (("checkpoint", "checkpoint_best_local_rr.pt"), ("config", "config.yaml"),
                                  ("history", "train_history.csv"), ("manifest", "run_manifest.json")):
                track(run / filename, baseline[key])
            cfg = OmegaConf.load(run / "config.yaml")
            OmegaConf.resolve(cfg)
            seed = baseline["seed"]
            if (cfg.model.variant != "crd_tf102_w" or int(cfg.training.seed) != seed
                    or int(cfg.model.initialization_seed) != seed or cfg.data.max_test_windows is not None
                    or int(cfg.training.batch_size) != 128 or not cfg.training.use_amp
                    or cfg.training.amp_dtype != "bfloat16" or cfg.data.drop_nonfinite_windows
                    or int(cfg.data.test_sample_seed) != 20260612
                    or float(cfg.window.target_fs) != 100 or int(cfg.window.duration_samples) != 18000):
                raise EvaluationFailure("W0 原始科学配置漂移")
            entry = next(item for item in test["entries"] if item["seed"] == seed)
            if entry["baseline_selected_epoch"] != baseline["selected_epoch"]:
                raise EvaluationFailure("test 来源 checkpoint epoch 漂移")
            reference_file = entry["baseline_test"]["research_test_metrics.csv"]
            track(Path(reference_file["path"]), reference_file)
            references.append(pd.read_csv(reference_file["path"]))
            configs.append(cfg)
        cache = Path(test["cache_root"])
        if sha256(cache / "cache_manifest.json") != test["cache_manifest_sha256"]:
            raise EvaluationFailure("test cache manifest 漂移")
        track(cache / "cache_manifest.json")
        for name, expected in test["cache_files"].items():
            track(cache / name, expected)
        index = Path(test["dataset_index"]["path"])
        track(index, test["dataset_index"])
        if (Path(configs[0].data.dataset_root) / configs[0].data.index_csv).resolve() != index.resolve():
            raise EvaluationFailure("数据索引路径不匹配")
        audited = read_research_v2_index(configs[0].data.dataset_root, configs[0].data.index_csv, configs[0])
        rows = filter_index(audited, configs[0], split="test", max_windows=None,
                            sample_strategy=str(configs[0].data.test_sample_strategy), sample_seed=20260612)
        for reference in references:
            check_rows(rows, reference, test["row_order_sha256"])
        development = set(audited.loc[audited.split.isin(["train", "val"]), "samp_id"].astype(int))
        for entry in test["entries"]:
            development.update(entry["development_samp_ids"])
        if set(rows.samp_id.astype(int)) & development:
            raise EvaluationFailure("test 与 train/validation 主体交叉")
        # 记录身份使用参考整晚文件的规范路径；coupling_state 不是独立记录。
        selection_rows = rows.copy()
        selection_rows["target_source_npz"] = selection_rows.target_source_npz.map(
            lambda path: str((index.parent / str(path)).resolve()))
        selected = select_nonoverlap_centers(selection_rows)
        selection_rows["center_selected"] = selected
        selection_rows["center_start_sample"] = rows.window_start_sample + 6000
        selection_rows["center_end_sample"] = rows.window_start_sample + 12000
        rows.to_csv(output / "test_rows.csv", index=False)
        selection_rows[["dataset_row_id", "samp_id", "target_source_npz", "center_start_sample",
                        "center_end_sample", "center_selected"]].to_csv(output / "center_selection.csv", index=False)
        for key in ("source_npz", "target_source_npz"):
            for name in sorted(rows[key].unique()):
                path = (index.parent / str(name)).resolve()
                if str(path) not in records:
                    track(path)
        write_json(output / "source_manifest.json", {"protocol": PROTOCOL, "files": records})
        targets = np.lib.format.open_memmap(output / "reference.npy", mode="w+", dtype=np.float32, shape=(COUNT, 18000))
        all_summaries, reference_flags = [], None
        expected_ids = rows.dataset_row_id.astype(int).tolist()
        data_config = OmegaConf.to_container(configs[0].data, resolve=True)
        for baseline, cfg in zip(baselines, configs, strict=True):
            seed = baseline["seed"]
            checkpoint_path = Path(baseline["run_dir"]) / "checkpoint_best_local_rr.pt"
            context.update(seed=seed, checkpoint=str(checkpoint_path), dataset_row_id=None, stage="checkpoint")
            if OmegaConf.to_container(cfg.data, resolve=True) != data_config:
                raise EvaluationFailure("三 seed 数据配置不一致")
            seed_dir = output / f"seed_{seed}"
            seed_dir.mkdir()
            OmegaConf.save(cfg, seed_dir / "training_config.yaml")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            _validate_checkpoint_config(checkpoint.get("config"), cfg)
            if checkpoint["epoch"] != baseline["selected_epoch"]:
                raise EvaluationFailure("固定 checkpoint epoch 错误")
            if any(not torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values() if torch.is_tensor(v)):
                raise EvaluationFailure("checkpoint 包含非有限数值")
            model = build_crd_model(cfg)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            model.to(resolved_device).eval()
            cfg.training.device = device
            cfg.training.show_progress = False
            cfg.data.preload_windows = False
            cfg.data.tf_research_test_cache_path = str(cache)
            OmegaConf.save(cfg, seed_dir / "resolved_config.yaml")
            context["stage"] = "data"
            data = build_window_data(cfg, split="test", max_windows=None,
                                     sample_strategy=str(cfg.data.test_sample_strategy), sample_seed=20260612,
                                     shuffle=False, audited=audited)
            check_rows(data.rows, rows, test["row_order_sha256"])
            if len(data.dataset) != COUNT:
                raise EvaluationFailure("dataset 丢失窗口")
            predictions = np.lib.format.open_memmap(seed_dir / "prediction.npy", mode="w+", dtype=np.float32, shape=(COUNT, 18000))
            metrics, observed, offset = [], [], 0
            context.update(stage="load_batch", dataset_row_id=None, batch_row_ids=expected_ids[:128])
            with (seed_dir / "window_metrics.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
                for batch in data.loader:
                    n = len(batch["x"])
                    ids = batch["meta"]["dataset_row_id"].numpy().astype(int).tolist()
                    context.update(stage="inference", dataset_row_id=None, batch_row_ids=ids)
                    if ids != expected_ids[offset:offset+n] or n != min(128, COUNT-offset):
                        raise EvaluationFailure("batch 身份/顺序/大小漂移")
                    for key, value in (("input", batch["x"]), ("target", batch["target"]), ("CWT", batch["tf"]["w"])):
                        finite = torch.isfinite(value).reshape(n, -1).all(dim=1)
                        if not finite.all():
                            context["dataset_row_id"] = ids[int(torch.nonzero(~finite)[0].item())]
                            raise EvaluationFailure(f"{key} 包含 NaN/Inf")
                    tf = batch_tf_to_device(batch, resolved_device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        raw = model(batch["x"].to(resolved_device), tf=tf)
                    pred = _waveform_output(raw).float().cpu().numpy().reshape(n, -1)
                    target = batch["target"].float().cpu().numpy().reshape(n, -1)
                    if pred.shape != (n, 18000) or target.shape != (n, 18000):
                        raise EvaluationFailure("模型/参考输出 shape 错误")
                    if seed == SEEDS[0]:
                        targets[offset:offset+n] = target
                    elif not np.array_equal(targets[offset:offset+n], target):
                        raise EvaluationFailure("seed 间参考波形发生变化")
                    predictions[offset:offset+n] = pred
                    for local, row_id in enumerate(ids):
                        position = offset + local
                        context.update(stage="metrics", dataset_row_id=row_id)
                        metric = evaluate_window(pred[local], target[local], center_selected=bool(selected[position]))
                        metric.update(dataset_row_id=row_id, samp_id=int(rows.iloc[position].samp_id), method="W0", seed=seed)
                        stream.write(json.dumps(metric, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        metrics.append(metric)
                    observed.extend(ids)
                    offset += n
                    print(f"seed={seed} windows={offset}/{COUNT}", flush=True)
                    context.update(stage="load_batch", dataset_row_id=None, batch_row_ids=expected_ids[offset:offset+128])
            context.update(stage="seed_summary", dataset_row_id=None, batch_row_ids=[])
            check_coverage(observed, expected_ids)
            predictions.flush()
            targets.flush()
            del predictions
            flags = [{k: v for k, v in m.items() if k.endswith("target_eligible") or k == "center_selected"} for m in metrics]
            if reference_flags is None:
                reference_flags = flags
            elif flags != reference_flags:
                raise EvaluationFailure("seed 间逐窗口参考资格不一致")
            pd.DataFrame(metrics).to_csv(seed_dir / "window_metrics.csv", index=False)
            summary = {"seed": seed, "epoch": baseline["selected_epoch"], **summarize_windows(metrics)}
            write_json(seed_dir / "summary.json", summary)
            all_summaries.append(summary)
            del model, data
        del targets
        context.update(stage="finalize", dataset_row_id=None, batch_row_ids=[])
        # 首尾身份核对覆盖源码、checkpoint、cache、索引、原始 NPZ。
        for path, identity in records.items():
            verify(Path(path), identity)
        result = summarize_seeds(all_summaries, SEEDS)
        write_json(output / "summary.json", {"protocol": PROTOCOL, "method": "W0", "seeds": list(SEEDS),
                   "total_windows": COUNT, "selected_centers": int(selected.sum()), "metrics": result})
        pd.DataFrame(all_summaries).to_csv(output / "seed_metrics.csv", index=False)
        lines = ["# W0 最终指标", "", "三个固定 validation-selected checkpoint；mean ± sample SD。", "",
                 "| 指标 | 结果 | 每 seed 有效窗口数 |", "|---|---:|---:|"]
        for metric in METRICS:
            item = result[metric]
            lines.append(f"| {metric} | {item['mean']:.6f} ± {item['sample_sd']:.6f} | {item['n_per_seed']} |")
        (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(output / "receipt.json", {"protocol": PROTOCOL, "status": "complete", "seeds": list(SEEDS),
                   "test_row_order_sha256": test["row_order_sha256"], "center_selection_sha256": sha256(output / "center_selection.csv")})
        write_json(output / "artifact_manifest.json", {"protocol": PROTOCOL, "status": "complete", "files": {
            str(path.relative_to(output)): record(path) for path in sorted(output.rglob("*")) if path.is_file()}})
        return output
    except BaseException as exc:
        write_json(output / "failure.json", {"protocol": PROTOCOL, "status": "failed", **context,
                   "reason": str(exc), "exception": type(exc).__name__, "traceback": traceback.format_exc()})
        raise
