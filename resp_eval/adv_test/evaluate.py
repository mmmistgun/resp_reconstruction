from __future__ import annotations

import fcntl
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from resp_train.aligned_dual_view.artifacts import array_digest, sha256_file, write_json
from resp_train.aligned_dual_view.config import model_config
from resp_train.aligned_dual_view.experiment import RuntimeModel, _load_checkpoint, check_primary_metrics
from resp_train.aligned_dual_view.features import spec_digest
from resp_train.data.cache import WholeNightCache
from resp_train.data.research_v2 import ResearchV2WindowDataset
from resp_train.engine.train import _prediction_dict_from_arrays
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.utils.run import resolve_device

from .artifacts import check_output_location, common_manifest, output_directory, verified_artifact
from .contract import LOCK_PATH, load_lock, repo_path, require_confirmation, source_config
from .data import TestCache, read_input, select_test_rows


def evaluation_root(cache_root):
    cache_root = Path(cache_root).resolve()
    return cache_root.parent / f"{cache_root.name}_evaluations"


@contextmanager
def claim_attempt(cache_root, name, attempt):
    """同一 cache/候选只能有一个成功评价；独立失败尝试保留，锁防止并发重复。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", attempt):
        raise ValueError("attempt 只能使用安全的字母数字标识")
    root = evaluation_root(cache_root) / name
    root.mkdir(parents=True, exist_ok=True)
    # 协调文件属于可变运行控制信息，科学产物保存在各自不可覆盖的 attempt 目录。
    with (root / ".coordination.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("该候选已有正在执行的评价") from exc
        try:
            for previous in root.glob("attempt_*"):
                if (previous / "completed.json").exists():
                    raise FileExistsError("该候选已有成功评价，禁止重复执行")
                if not (previous / "failed.json").exists():
                    raise RuntimeError("存在未闭合尝试；须先核实中断原因")
            yield root / f"attempt_{attempt}"
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def check_test_metrics(metrics):
    check_primary_metrics(metrics)
    eligible = metrics.joint_target_eligible.to_numpy(dtype=bool)
    for column in ("respiratory_band_coherence", "constrained_ndtw"):
        values = metrics[column].to_numpy(dtype=np.float64)
        if np.isinf(values).any() or not np.array_equal(np.isfinite(values), eligible):
            raise FloatingPointError(f"{column} 与 target eligibility 不一致")
        if np.any(values[eligible] < 0):
            raise ValueError(f"{column} 不得为负")
    if np.any(metrics.loc[eligible, "respiratory_band_coherence"] > 1 + 1e-6):
        raise ValueError("coherence 超出 [0,1]")


def evaluate_checkpoint(name, *, cache_root, lock_path=LOCK_PATH, device="cuda:0", attempt="r1",
                        confirm_research_test=False, show_progress=True, model_factory=RuntimeModel):
    require_confirmation(confirm_research_test)
    lock = load_lock(lock_path)
    cfg = source_config(lock, name)
    entry = lock["entries"][name]
    verified_artifact(cache_root, kind="test_cache", lock=lock)
    check_output_location(evaluation_root(cache_root) / name, cfg)
    # 名称已由固定 allowlist 验证，输出位置由 cache 身份确定。
    with claim_attempt(cache_root, name, attempt) as output:
        with output_directory(output, kind="test_evaluation", cfg=cfg, lock=lock) as (root, started):
            checkpoint_path = repo_path(entry["checkpoint"])
            if sha256_file(checkpoint_path) != entry["checkpoint_sha256"]:
                raise ValueError("冻结 selected checkpoint 哈希漂移")
            model = model_factory(model_config(cfg))
            checkpoint = _load_checkpoint(checkpoint_path, model=model, identity=entry["training_identity"])
            if checkpoint["epoch"] != entry["selected_epoch"]:
                raise ValueError("checkpoint epoch 偏离 validation 选择")
            actual_device = resolve_device(device)
            if actual_device.type == "cuda":
                torch.backends.cudnn.conv.fp32_precision = "ieee"
            model = model.to(actual_device).eval()
            path, rows, identities = select_test_rows(cfg, lock, confirmed=True)
            cache = TestCache(cache_root, lock, identities)
            source = WholeNightCache(path)
            rows.to_csv(root / "test_selection.csv", index=False)
            predicted = np.empty((len(rows), 1, 18000), dtype=np.float32)
            batch_size = int(cfg.training.batch_size)
            amp_enabled = actual_device.type == "cuda"
            with (root / "access.jsonl").open("x", encoding="utf-8") as access:
                def record(event, **fields):
                    access.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")
                    access.flush()
                record("input_only_inference_started", expected_count=len(rows), target_read=False)
                with torch.no_grad():
                    for start in tqdm(range(0, len(rows), batch_size), desc=name, disable=not show_progress):
                        stop = min(start + batch_size, len(rows))
                        waveforms, features = [], []
                        for position in range(start, stop):
                            waveform = read_input(source, identities[position])
                            feature = cache.get(position, waveform, use_cwt=model.uses_cwt)
                            waveforms.append(waveform)
                            if feature is not None:
                                features.append(feature)
                        x = torch.from_numpy(np.stack(waveforms)[:, None]).to(actual_device)
                        tf = ({"adv_cwt": torch.from_numpy(np.stack(features)).to(actual_device)}
                              if model.uses_cwt else None)
                        with torch.autocast(actual_device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                            output_values = model(x, tf=tf)["waveform"].float().detach().cpu().numpy()
                        if output_values.shape != (stop - start, 1, 18000) or not np.isfinite(output_values).all():
                            raise FloatingPointError("test prediction shape/finite 错误")
                        predicted[start:stop] = output_values
                record("input_only_inference_completed", count=len(rows), target_read=False)

                # 整个候选的预测全部完成后才实例化 target dataset，阻断目标进入推理路径。
                record("target_read_started", inference_count=len(rows))
                dataset = ResearchV2WindowDataset(path, rows, cfg, preload_windows=False)
                if not dataset.rows.dataset_row_id.equals(rows.dataset_row_id):
                    raise ValueError("target dataset 改变了样本集合")
                targets = np.empty_like(predicted)
                metadata, samples = [], []
                for position, identity in enumerate(identities):
                    item = dataset[position]
                    if array_digest(item["x"]) != cache.records[position]["input_sha256"]:
                        raise ValueError("推理后 input 内容发生变化")
                    targets[position] = item["target"].numpy()
                    metadata.append(item["meta"])
                    samples.append({"identity": identity, "input_sha256": array_digest(item["x"]),
                                    "target_sha256": array_digest(item["target"]),
                                    "mask_sha256": array_digest(item["meta"]["rr_peak_valid_mask"])})
                record("target_read_completed", count=len(rows))
            predictions = _prediction_dict_from_arrays(predicted, targets, metadata,
                                                        pred_key="r_tho_hat", target_key="tho_ref")
            if (not np.array_equal(predictions["dataset_row_id"], rows.dataset_row_id.to_numpy())
                    or set(predictions["split"]) != {"test"}):
                raise ValueError("test prediction row identity 错误")
            metrics = evaluate_task_predictions(predictions, cfg, include_test_only=True, method=f"adv_v1_{entry['view']}")
            check_test_metrics(metrics)
            metrics.to_csv(root / "metrics.csv", index=False)
            summarize_task_metrics(metrics).to_csv(root / "metrics_summary.csv", index=False)
            write_json(root / "samples.json", {"samples": samples})
            precision = "bfloat16" if amp_enabled else "float32"
            comparison = {"lock_id": lock["lock_id"], "cache_id": cache.manifest["cache_id"],
                          "samples_sha256": spec_digest(samples), "code_sha256": started["code"]["sha256"],
                          "precision": precision, "batch_size": batch_size}
            write_json(root / "manifest.json", {
                **common_manifest("test_evaluation", lock, started), **comparison,
                "comparison_id": spec_digest(comparison), "entry": name, "view": entry["view"], "seed": entry["seed"],
                "selected_epoch": entry["selected_epoch"], "checkpoint_sha256": entry["checkpoint_sha256"],
                "cache_manifest_sha256": sha256_file(Path(cache_root) / "manifest.json"),
                "count": len(rows), "prediction_sha256": array_digest(predicted),
                "sample_records_file_sha256": sha256_file(root / "samples.json"),
                "metrics_sha256": sha256_file(root / "metrics.csv"),
                "summary_sha256": sha256_file(root / "metrics_summary.csv"),
                "access_sha256": sha256_file(root / "access.jsonl"),
                "device": str(actual_device), "research_test_input_used": True, "target_read": True,
                "inference_completed_before_target_read": True, "checkpoint_reselected": False,
            })
    return output
