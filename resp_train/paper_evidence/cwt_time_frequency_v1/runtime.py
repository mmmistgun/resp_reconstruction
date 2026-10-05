"""固定 80-epoch 训练、导出恢复与多卡独立 cell 调度。"""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.training import build_crd_optimizer, crd_learning_rate
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.w0_structural_factorial_v1_formal import state_dict_identity
from resp_train.paper_evidence.w0_structural_factorial_v1_engineering import finite_tree
from . import artifacts as io
from .data import data_receipt, training_data
from .engineering import environment, require_gpu, run_gpu
from .model import build_model
from .spec import ARMS, SEEDS, ROOT, OUTPUT_ROOT, PROTOCOL, config, plan


def cell_config(session, arm, seed, output, device):
    frozen = io.load_session(session)
    cfg = config(arm, seed, output, device)
    data_path, _ = data_receipt(session)
    cache = io.completed(session / "cache" / arm, io.binding(session, "cache", arm=arm,
                         data=io.identity(data_path / "manifest.json")["sha256"]))
    if cache is None:
        raise RuntimeError(f"缺少 {arm} 缓存")
    cfg.data.tf_cache_path = str(cache)
    cfg.model.cwt_representation = frozen["representations"][arm]
    return cfg


def selected_epoch(run):
    history = pd.read_csv(run / "train_history.csv")
    if len(history) != 80 or history.epoch.tolist() != list(range(1, 81)) or history.optimizer_update.tolist() != list(range(80, 6401, 80)):
        raise ValueError("训练未完成固定 80 epochs/6400 updates")
    if not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        raise FloatingPointError("history 非有限")
    for record in history.itertuples(index=False):
        for column, update in (("first_learning_rate", (record.epoch-1)*80), ("last_learning_rate", record.epoch*80-1)):
            expected = crd_learning_rate(update, total_updates=6400, max_learning_rate=3e-4, min_learning_rate=3e-5, warmup_fraction=.05)
            if not np.isclose(getattr(record, column), expected, rtol=1e-12, atol=1e-12):
                raise ValueError("学习率轨迹偏离 6400-update 合同")
    return int(history.iloc[int(np.argmin(history.val_local_rr_mae.to_numpy()))].epoch)


def verify_checkpoints(run, cfg):
    epoch = selected_epoch(run)
    model = build_model(int(cfg.training.seed), OmegaConf.to_container(cfg.model.cwt_representation, resolve=True))
    optimizer, _ = build_crd_optimizer(model, cfg)
    for name, expected_epoch in (("checkpoint_best_local_rr.pt", epoch), ("checkpoint_final.pt", 80)):
        checkpoint = torch.load(run / name, map_location="cpu", weights_only=False)
        finite_tree(checkpoint, label=name)
        if (checkpoint["epoch"] != expected_epoch or checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True)
                or checkpoint["extra_state"]["protocol"] != PROTOCOL
                or checkpoint["extra_state"]["update_index"] != expected_epoch*80
                or checkpoint["extra_state"]["total_updates"] != 6400):
            raise ValueError("checkpoint 配置/选点/更新数不一致")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for p in model.parameters():
            state = optimizer.state[p]
            if (not {"step", "exp_avg", "exp_avg_sq"}.issubset(state) or float(state["step"]) != expected_epoch*80
                    or state["exp_avg"].shape != p.shape or state["exp_avg_sq"].shape != p.shape):
                raise ValueError("optimizer state 不完整")
    return epoch


def export_validation(session, arm, seed, cfg, model, loader, device, output):
    prediction = collect_predictions(model, loader, device=device, max_windows=len(loader.dataset), use_amp=True)
    for name in ("r_tho_hat", "tho_ref"):
        finite_tree(prediction[name], label=name)
    data_path, _ = data_receipt(session)
    rows = pd.read_csv(data_path / "val_rows.csv")
    if not np.array_equal(prediction["dataset_row_id"], rows.dataset_row_id.to_numpy()):
        raise ValueError("预测 row 顺序变化")
    reference = np.load(data_path / "validation_reference.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(reference, prediction["tho_ref"]):
        raise ValueError("validation 公共参考变化")
    frame = evaluate_task_predictions(prediction, cfg, include_test_only=False, method=arm)
    sf.validate_metrics(frame, rows)
    frame.insert(0, "arm", arm)
    frame.insert(0, "seed", seed)
    with (output / "validation_prediction.npy").open("xb") as stream:
        np.save(stream, prediction["r_tho_hat"].astype(np.float32), allow_pickle=False)
    frame[["dataset_row_id", "samp_id", "split"]].to_csv(output / "prediction_rows.csv", index=False)
    io.write_json(output / "reference_source.json", {"path": str(data_path / "validation_reference.npy"),
                  **io.identity(data_path / "validation_reference.npy")})
    return frame


class Experiment(CRDExperiment):
    task_name = "cwt_time_frequency_v1"

    def __init__(self, cfg, session, arm, attempt):
        super().__init__(cfg)
        self.session, self.arm, self.attempt = session, arm, attempt

    def _build_model(self):
        model = build_model(int(self.cfg.training.seed), OmegaConf.to_container(self.cfg.model.cwt_representation, resolve=True))
        io.write_json(self.attempt / "initialization.json", state_dict_identity(model.state_dict()))
        return model

    def _build_data(self):
        return training_data(self.session, self.arm, self.cfg)

    def _evaluate_model(self, model, loader, **kwargs):
        if kwargs.get("include_test_only", False):
            raise ValueError("训练入口禁止 test")
        # trainer 仅在最终 checkpoint 完成后调用该钩子。先冻结训练完成身份，使导出失败可独立恢复。
        files = {name: io.identity(self.run_dir / name) for name in
                 ("checkpoint_best_local_rr.pt", "checkpoint_final.pt", "train_history.csv", "config.yaml", "optimizer_parameter_groups.json")}
        io.write_json(self.attempt / "training_complete.json", {"run": str(self.run_dir), "files": files,
                      "selected_epoch": selected_epoch(self.run_dir)})
        return export_validation(self.session, self.arm, int(self.cfg.training.seed), self.cfg,
                                 model, loader, self.device, self.run_dir)


def cell_parent(session, arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("未知 cell")
    return session / "formal" / arm / f"seed_{seed}"


def cell_result(session, arm, seed):
    key = io.binding(session, "formal", arm=arm, seed=seed)
    path = io.completed(cell_parent(session, arm, seed), key)
    recovery = io.completed(session / "exports" / arm / f"seed_{seed}", io.binding(session, "export", arm=arm, seed=seed))
    if path and recovery:
        raise ValueError("同 cell 存在训练与恢复两个成功结果")
    path = path or recovery
    if path is None:
        return None
    result = io.read_json(path / "result.json")
    for item in result["sources"].values():
        io.verify(item["path"], item)
    return path, result


def result_record(run, metrics, arm, seed, epoch):
    names = {"checkpoint": run / "checkpoint_best_local_rr.pt", "final_checkpoint": run / "checkpoint_final.pt",
             "history": run / "train_history.csv", "config": run / "config.yaml", "metrics": metrics / "metrics.csv"}
    return {"arm": arm, "seed": seed, "selected_epoch": epoch, "completed_epochs": 80,
            "sources": {k: {"path": str(p), **io.identity(p)} for k, p in names.items()}}


def run_formal(session, arm, seed, device, retry=False):
    frozen = io.require_data_scope(session)
    prior = cell_result(session, arm, seed)
    if prior:
        return prior[0]
    gpu = require_gpu(session, device)
    # 信号检查先于 formal，S3 分层阈值和固定案例在此门控前生成。
    if io.completed(session / "signals", io.binding(session, "signals")) is None:
        raise RuntimeError("先完成 train/validation signals 阶段")
    with io.mutex(OUTPUT_ROOT / f".device_{torch.device(device).index}.mutex"):
        with io.attempt(cell_parent(session, arm, seed), io.binding(session, "formal", arm=arm, seed=seed), retry) as output:
            if io.completed(session / "exports" / arm / f"seed_{seed}", io.binding(session, "export", arm=arm, seed=seed)):
                raise FileExistsError("该 cell 已完成导出恢复，禁止重新训练")
            io.write_json(output / "environment.json", environment(device))
            io.write_json(output / "gpu_source.json", {"path": str(gpu), **io.identity(gpu / "manifest.json")})
            cfg = cell_config(session, arm, seed, output / "training", device)
            started = time.perf_counter()
            run = Experiment(cfg, session, arm, output).train()
            epoch = verify_checkpoints(run, cfg)
            io.write_json(output / "result.json", result_record(run, run, arm, seed, epoch))
            io.write_json(output / "resource.json", {"elapsed_seconds": time.perf_counter()-started,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            io.verify_provenance(frozen["provenance"], session)
    return output


def recover_export(session, arm, seed, source, device, retry=False):
    io.require_data_scope(session)
    # 和原 formal 阶段共用 cell mutex，防止收尾仍在运行时并发恢复。
    with io.mutex(cell_parent(session, arm, seed) / ".mutex"):
        return _recover_export(session, arm, seed, source, device, retry)


def _recover_export(session, arm, seed, source, device, retry=False):
    frozen = io.load_session(session)
    if cell_result(session, arm, seed):
        raise FileExistsError("已有成功 cell")
    require_gpu(session, device)
    source = source.resolve()
    if source.parent != cell_parent(session, arm, seed).resolve():
        raise ValueError("恢复来源不属于指定 cell")
    if io.read_json(source / "started.json")["binding"] != io.binding(session, "formal", arm=arm, seed=seed):
        raise ValueError("恢复来源 session 不匹配")
    marker = io.read_json(source / "training_complete.json")
    run = Path(marker["run"]).resolve()
    if not run.is_relative_to(source):
        raise ValueError("恢复来源路径越界")
    for name, expected in marker["files"].items():
        io.verify(run / name, expected)
    cfg = OmegaConf.load(run / "config.yaml")
    if (cfg.model.cwt_time_frequency_v1.name != arm or int(cfg.training.seed) != seed
            or OmegaConf.to_container(cfg.model.cwt_representation, resolve=True) != frozen["representations"][arm]):
        raise ValueError("恢复来源的 cell/表示身份不匹配")
    epoch = verify_checkpoints(run, cfg)
    with io.mutex(OUTPUT_ROOT / f".device_{torch.device(device).index}.mutex"):
        with io.attempt(session / "exports" / arm / f"seed_{seed}", io.binding(session, "export", arm=arm, seed=seed), retry) as output:
            io.write_json(output / "environment.json", environment(device))
            io.write_json(output / "recovery_source.json", {"path": str(source), "training_complete": io.identity(source / "training_complete.json")})
            model = build_model(seed, frozen["representations"][arm]).to(device)
            checkpoint = torch.load(run / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            data = training_data(session, arm, cfg)
            frame = export_validation(session, arm, seed, cfg, model, data.val.loader, device, output)
            frame.to_csv(output / "metrics.csv", index=False)
            summarize_task_metrics(frame).to_csv(output / "metrics_summary.csv", index=False)
            io.write_json(output / "result.json", result_record(run, output, arm, seed, epoch))
            io.verify_provenance(frozen["provenance"], session)
    return output


def run_parallel(session, devices, retry=False):
    io.require_data_scope(session)
    normalized = [str(torch.device(d)) for d in devices]
    if len(set(normalized)) != len(normalized) or not normalized:
        raise ValueError("GPU 列表为空或重复")
    for d in normalized:
        if torch.device(d).type != "cuda" or torch.device(d).index is None:
            raise ValueError("要求显式 cuda:<index>")
    layout = {"devices": normalized, "shards": [plan()[i::len(normalized)] for i in range(len(normalized))]}
    with io.mutex(session / ".dispatch.mutex"):
        path = session / "dispatch.json"
        if path.exists():
            if io.read_json(path) != layout:
                raise ValueError("设备/交错分片映射已经冻结")
        else:
            io.write_json(path, layout)
        logs = session / ("dispatch_" + io.stamp())
        logs.mkdir(exist_ok=False)
        processes, streams = [], []
        try:
            for i, device in enumerate(normalized):
                stream = (logs / f"worker_{i}.log").open("x")
                streams.append(stream)
                env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONPATH=str(ROOT))
                env.pop("LD_LIBRARY_PATH", None)
                env.pop("LD_PRELOAD", None)
                command = [sys.executable, str(ROOT / "scripts/run_cwt_time_frequency_v1.py"), "run-shard", "--session", str(session),
                           "--device", device, "--shard", str(i), "--confirm-training"]
                if retry:
                    command.append("--retry-failed")
                processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
            io.write_json(logs / "workers.json", {"pids": [p.pid for p in processes], "layout": layout})
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError("worker 失败；停止其余 worker 并保留现场")
                time.sleep(1)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError("worker 失败")
        finally:
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGINT)
            for p in processes:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
            for stream in streams:
                stream.close()
    from .summary import summarize
    return summarize(session, retry)


def run_shard(session, device, index, retry=False):
    layout = io.read_json(session / "dispatch.json")
    if not 0 <= index < len(layout["devices"]) or layout["devices"][index] != device:
        raise ValueError("worker 分片/设备不匹配")
    expected = [plan()[i::len(layout["devices"])] for i in range(len(layout["devices"]))]
    if layout["shards"] != expected:
        raise ValueError("分片矩阵漂移")
    with io.mutex(session / f".shard_{index}.mutex"):
        run_gpu(session, device, retry)
        for cell in layout["shards"][index]:
            run_formal(session, cell["arm"], cell["seed"], device, retry)
