"""H-only 消融独立生命周期；历史缓存只读，训练仅 train/validation。"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.training import build_crd_optimizer
from resp_train.data.factory import build_window_data, ThoDataBundle
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.w0_structural_factorial_v1_engineering import finite_tree
from resp_train.paper_evidence.w0_structural_factorial_v1_formal import state_dict_identity
from resp_train.paper_evidence.cwt_apor_v2 import data as source_data
from resp_train.paper_evidence.cwt_apor_v2.features import representation
from resp_train.paper_evidence.cwt_apor_v2.spec import ARMS as SOURCE_ARMS, config as source_config
from resp_train.paper_evidence.cwt_apor_v2.reuse import verify_reference_stage
from resp_train.paper_evidence.cwt_apor_v2.engineering import environment
from .history import selected_epoch
from . import artifacts as io
from .spec import ROOT, PROTOCOL, ARMS, SEEDS, GAIN_ARMS, config, contract, plan
from .model import build_model
from .diagnostics import TrainingMonitor, gain_statistics


def preprocessing_paths(source_session):
    source_session = Path(source_session).resolve()
    io.verify(source_session / "session.json", io.read_json(source_session / "session_receipt.json"))
    source = io.read_json(source_session / "session.json")
    if source["protocol"] != "cwt-apor-v2-20261001":
        raise ValueError("要求 CWT-APOR-v2 来源 session")
    refs = source["preprocessing_references"]
    entries = (refs["data"], refs["caches"]["H"])
    for entry in entries:
        io.verify(Path(entry["path"]) / "manifest.json", entry)
    return tuple(Path(entry["path"]) for entry in entries)


def prepare(session, data, cache):
    """由用户执行：核验冻结 train/val 数据与 H-only 缓存，创建新 session。"""
    session, data, cache = map(lambda p: Path(p).resolve(), (session, data, cache))
    data_stage, cache_stage = verify_reference_stage(data), verify_reference_stage(cache)
    if data_stage["binding"]["phase"] != "data" or cache_stage["binding"].get("arm") != "H":
        raise ValueError("需要已完成的 development data 和 H-only cache 阶段")
    if cache_stage["binding"]["phase"] != "cache":
        raise ValueError("缓存阶段身份不符")
    metadata = io.read_json(cache / "cache.json")
    rep = representation(SOURCE_ARMS["H"])
    if (metadata["representation"] != rep or rep["shape"] != [41, 360]
            or metadata["splits"] != ["train", "val"] or metadata["research_test"] is not False
            or metadata["data_manifest"] != io.identity(data / "manifest.json")):
        raise ValueError("缓存变换/数据来源/访问范围不匹配")
    cfg = config("HA0", SEEDS[0], session / "unused")
    _, rows = source_data.audit_rows(source_data.raw_config(cfg))
    record = io.read_json(data / "data.json")
    source_data.verify_sources(record)
    for split in ("train", "val"):
        if source_data.array_hash(rows[split].dataset_row_id.to_numpy(np.int64)) != record["row_order"][split]:
            raise ValueError("冻结窗口集合/顺序漂移")
        source_data.CacheReader(cache, split, rep, rows[split].dataset_row_id.to_numpy(np.int64))
    session.mkdir(parents=True, exist_ok=False)
    payload = {"contract": contract(), "representation": rep, "provenance": io.snapshot(session),
               "sources": {k: {"path": str(p), **io.identity(p / "manifest.json")}
                           for k, p in (("data", data), ("cache", cache))},
               "command": sys.argv, "access": {"splits": ["train", "val"], "test": False}}
    io.write_json(session / "session.json", payload)
    io.write_json(session / "session_receipt.json", io.identity(session / "session.json"))
    return session


def references(session):
    frozen = io.load_session(session)
    paths = {}
    for kind, entry in frozen["sources"].items():
        path = Path(entry["path"])
        io.verify(path / "manifest.json", entry)
        verify_reference_stage(path)
        paths[kind] = path
    return frozen, paths


def training_data(session, arm, cfg):
    frozen, paths = references(session)
    data = io.read_json(paths["data"] / "data.json")
    source_data.verify_sources(data)
    raw = source_data.raw_config(cfg)
    raw.data.preload_windows = bool(cfg.data.preload_windows)
    audited, rows = source_data.audit_rows(raw)
    bundles = {}
    for split in ("train", "val"):
        if source_data.array_hash(rows[split].dataset_row_id.to_numpy(np.int64)) != data["row_order"][split]:
            raise ValueError("窗口顺序漂移")
        bundle = build_window_data(raw, split=split, max_windows=None, audited=audited,
                   sample_strategy=cfg.data[f"{split}_sample_strategy"],
                   sample_seed=cfg.data[f"{split}_sample_seed"], shuffle=split == "train")
        bundles[split] = (source_data.condition_bundle(bundle, cfg, paths["cache"], split, frozen["representation"])
                          if ARMS[arm].condition else bundle)
    if len(bundles["train"].loader) != 80 or bundles["train"].loader.batch_size != 128:
        raise ValueError("6400-update 训练预算漂移")
    return ThoDataBundle(bundles["train"], bundles["val"], audited, bundles["train"].audit_summary)


def evaluate(model, loader, cfg, output, arm, seed, rows, reference, device, split):
    gain_rows = []
    def observe(value, training):
        if training:
            raise RuntimeError("评价模型仍为 train 模式")
        for sample in value:
            gain_rows.append(gain_statistics(sample))
    if arm in GAIN_ARMS:
        model.gain_observer = observe
    try:
        prediction = collect_predictions(model, loader, device=device, max_windows=len(loader.dataset), use_amp=True)
    finally:
        model.gain_observer = None
    for name in ("r_tho_hat", "tho_ref"):
        finite_tree(prediction[name], label=name)
    if (not np.array_equal(prediction["dataset_row_id"], rows.dataset_row_id.to_numpy())
            or not np.array_equal(prediction["tho_ref"], reference)):
        raise ValueError("评价窗口/公共 target 不一致")
    frame = evaluate_task_predictions(prediction, cfg, include_test_only=split == "test", method=arm)
    sf.validate_metrics(frame, rows)
    frame.insert(0, "arm", arm)
    frame.insert(0, "seed", seed)
    frame.to_csv(output / "metrics.csv", index=False)
    with (output / f"{split}_prediction.npy").open("xb") as stream:
        np.save(stream, prediction["r_tho_hat"].astype(np.float32), allow_pickle=False)
    rows[["dataset_row_id", "samp_id", "split"]].to_csv(output / "prediction_rows.csv", index=False)
    if arm in GAIN_ARMS:
        if len(gain_rows) != len(rows):
            raise ValueError("逐窗口增益记录不完整")
        io.write_json(output / "gain_per_window.json", [
            {"dataset_row_id": int(row.dataset_row_id), "samp_id": int(row.samp_id), **stats}
            for row, stats in zip(rows.itertuples(), gain_rows)])
    return frame


class Experiment(CRDExperiment):
    task_name = "h_only_ablation_v1"

    def __init__(self, cfg, session, arm, output):
        super().__init__(cfg)
        self.session, self.arm, self.output, self.monitor = session, arm, output, None

    def _build_model(self):
        model = build_model(self.arm, int(self.cfg.training.seed))
        io.write_json(self.output / "initialization.json", state_dict_identity(model.state_dict()))
        self.monitor = TrainingMonitor(model, self.output, self.arm in GAIN_ARMS)
        return model

    def _build_data(self):
        return training_data(self.session, self.arm, self.cfg)

    def _evaluate_model(self, model, loader, **kwargs):
        if kwargs.get("include_test_only", False):
            raise PermissionError("训练入口禁止 test")
        # 训练与导出分开记账，导出失败仍保留训练完成证据。
        names = ("config.yaml", "train_history.csv", "checkpoint_best_local_rr.pt", "checkpoint_final.pt")
        io.write_json(self.output / "training_complete.json", {"run": str(self.run_dir),
                      "files": {n: io.identity(self.run_dir / n) for n in names}})
        _, paths = references(self.session)
        rows = pd.read_csv(paths["data"] / "val_rows.csv")
        reference = np.load(paths["data"] / "validation_reference.npy", mmap_mode="r", allow_pickle=False)
        return evaluate(model, loader, self.cfg, self.run_dir, self.arm, int(self.cfg.training.seed),
                        rows, reference, self.device, "val")


def verify_checkpoints(run, arm, seed, expected_cfg):
    epoch = selected_epoch(run)
    cfg = OmegaConf.load(run / "config.yaml")
    if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(expected_cfg, resolve=True):
        raise ValueError("checkpoint resolved config 不匹配")
    model = build_model(arm, seed)
    optimizer, _ = build_crd_optimizer(model, cfg)
    count = len(pd.read_csv(run / "train_history.csv"))
    for name, expected_epoch in (("checkpoint_best_local_rr.pt", epoch), ("checkpoint_final.pt", count)):
        cp = torch.load(run / name, map_location="cpu", weights_only=False)
        finite_tree(cp, label=name)
        if (cp["epoch"] != expected_epoch or cp["config"] != OmegaConf.to_container(cfg, resolve=True)
                or cp["extra_state"]["protocol"] != str(cfg.protocol.name)
                or cp["extra_state"]["total_updates"] != 6400 or cp["extra_state"]["update_index"] != expected_epoch*80):
            raise ValueError("checkpoint 更新数/选点/配置不匹配")
        model.load_state_dict(cp["model_state_dict"], strict=True)
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        for p in model.parameters():
            state = optimizer.state[p]
            if (not {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
                    or float(state["step"]) != expected_epoch*80
                    or state["exp_avg"].shape != p.shape or state["exp_avg_sq"].shape != p.shape):
                raise ValueError("optimizer state 不完整")
    return epoch


def parent(session, arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("未知 cell")
    return Path(session) / "formal" / arm / f"seed_{seed}"


def cell_result(session, arm, seed):
    path = io.completed(parent(session, arm, seed), io.binding(session, "formal", arm=arm, seed=seed))
    if path is None:
        return None
    result = io.read_json(path / "result.json")
    if result["arm"] != arm or result["seed"] != seed:
        raise ValueError("cell 结果标签漂移")
    for source in result["sources"].values():
        io.verify(source["path"], source)
    return path, result


def result_record(run, metrics, arm, seed, epoch, reused=False):
    names = {"checkpoint": run / "checkpoint_best_local_rr.pt", "final_checkpoint": run / "checkpoint_final.pt",
             "history": run / "train_history.csv", "config": run / "config.yaml", "metrics": metrics / "metrics.csv"}
    return {"arm": arm, "seed": seed, "selected_epoch": epoch, "reused": reused,
            "sources": {k: {"path": str(p), **io.identity(p)} for k, p in names.items()}}


def require_device(device):
    device = torch.device(device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ValueError("正式执行要求可用的 cuda:<index>")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("要求 BF16 支持")
    return device


def run_formal(session, arm, seed, device, retry=False):
    io.load_session(session)
    if cell_result(session, arm, seed):
        raise FileExistsError("此 cell 已完成")
    if list(parent(session, arm, seed).glob("attempt_*/training_complete.json")):
        raise RuntimeError("该 cell 已完成训练；使用 recover-export 恢复导出，禁止重新训练")
    if arm != "HA0" and any(cell_result(session, "HA0", s) is None for s in SEEDS):
        raise RuntimeError("先完成 HA0 三 seed；这些实例直接进入最终矩阵")
    device = require_device(device)
    accepted = io.completed(session / "acceptance" / arm / f"seed_{seed}",
                            io.binding(session, "acceptance", arm=arm, seed=seed))
    if accepted is None:
        raise RuntimeError("先完成该 cell 的原生 GPU/BF16 合成验收")
    with io.mutex(ROOT / "runs/h_only_ablation_v1" / f".device_{device.index}.mutex"):
        with io.attempt(parent(session, arm, seed), io.binding(session, "formal", arm=arm, seed=seed), retry) as output:
            io.write_json(output / "environment.json", environment(str(device)))
            io.write_json(output / "access.json", {"splits": ["train", "val"], "command": sys.argv})
            cfg = config(arm, seed, output / "training", str(device))
            frozen, paths = references(session)
            cfg.model.cwt_representation = frozen["representation"]
            cfg.data.tf_cache_path = str(paths["cache"]) if ARMS[arm].condition else None
            exp = Experiment(cfg, session, arm, output)
            started = time.perf_counter()
            try:
                run = exp.train()
            finally:
                if exp.monitor:
                    exp.monitor.close()
            epoch = verify_checkpoints(run, arm, seed, cfg)
            io.write_json(output / "result.json", result_record(run, run, arm, seed, epoch))
            io.write_json(output / "resource.json", {"elapsed_seconds": time.perf_counter()-started,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
            io.load_session(session)
    return output


def recover_export(session, arm, seed, source_attempt, device, retry=False):
    """只重做失败导出；checkpoint、history 和已完成训练保持只读。"""
    frozen, paths = references(session)
    source_attempt = Path(source_attempt).resolve()
    if source_attempt.parent != parent(session, arm, seed).resolve():
        raise ValueError("恢复来源不属于指定 cell")
    stage_key = io.binding(session, "formal", arm=arm, seed=seed)
    if io.read_json(source_attempt / "started.json")["binding"] != stage_key:
        raise ValueError("恢复来源身份不符")
    marker = io.read_json(source_attempt / "training_complete.json")
    run = Path(marker["run"]).resolve()
    if not run.is_relative_to(source_attempt):
        raise ValueError("训练完成来源越界")
    for name, expected in marker["files"].items():
        io.verify(run / name, expected)
    actual = OmegaConf.load(run / "config.yaml")
    expected = config(arm, seed, source_attempt / "training", actual.training.device)
    expected.model.cwt_representation = frozen["representation"]
    expected.data.tf_cache_path = str(paths["cache"]) if ARMS[arm].condition else None
    epoch = verify_checkpoints(run, arm, seed, expected)
    device = require_device(device)
    with io.mutex(ROOT / "runs/h_only_ablation_v1" / f".device_{device.index}.mutex"):
        # 源失败实例也在该 parent 中，恢复必须显式 retry，并产生新 identity。
        with io.attempt(parent(session, arm, seed), stage_key, retry) as output:
            cfg = OmegaConf.create(OmegaConf.to_container(actual, resolve=True))
            cfg.training.device = str(device)
            model = build_model(arm, seed).to(device)
            checkpoint = torch.load(run / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            bundle = training_data(session, arm, cfg)
            rows = pd.read_csv(paths["data"] / "val_rows.csv")
            reference = np.load(paths["data"] / "validation_reference.npy", mmap_mode="r", allow_pickle=False)
            evaluate(model, bundle.val.loader, cfg, output, arm, seed, rows, reference, device, "val")
            io.write_json(output / "recovery.json", {"source_attempt": str(source_attempt),
                          "training_complete": io.identity(source_attempt / "training_complete.json"), "command": sys.argv})
            io.write_json(output / "result.json", result_record(run, output, arm, seed, epoch))
            io.load_session(session)
    return output


def compatible_h65_config(actual, seed, rep):
    """除运行位置/设备外逐字段匹配历史 H 合同；明确拒绝全频 A0。"""
    expected = source_config("H", seed, actual.outputs.run_root, actual.training.device)
    # 历史入口仅增加 native representation 和缓存路径。
    expected.data.tf_cache_path = actual.data.tf_cache_path
    expected.model.cwt_representation = rep
    if OmegaConf.to_container(actual, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError("HA16 来源不是同合同的 H-only/H65；需独立重训")


def reuse_h65(session, seed, source_attempt):
    frozen = io.load_session(session)
    source_attempt = Path(source_attempt).resolve()
    manifest = verify_reference_stage(source_attempt)
    if (manifest["protocol"] != "cwt-apor-v2-20261001" or manifest["binding"].get("arm") != "H"
            or manifest["binding"].get("seed") != seed or manifest["binding"]["phase"] not in ("formal", "export")):
        raise ValueError("HA16 仅允许 CWT-APOR-v2/H 同 seed 的完成阶段")
    record = io.read_json(source_attempt / "result.json")
    for entry in record["sources"].values():
        io.verify(entry["path"], entry)
    run = Path(record["sources"]["config"]["path"]).parent
    cfg = OmegaConf.load(run / "config.yaml")
    compatible_h65_config(cfg, seed, frozen["representation"])
    epoch = verify_checkpoints(run, "HA16", seed, cfg)
    # H65 的初始 state 身份须与本轮构造逐 tensor 一致。
    initial_path = source_attempt / "initialization.json"
    if not initial_path.exists():
        raise ValueError("缺少历史初始化 receipt，暂不允许复用")
    if io.read_json(initial_path) != state_dict_identity(build_model("HA16", seed).state_dict()):
        raise ValueError("HA16 初始 tensor 身份不匹配")
    frame = pd.read_csv(record["sources"]["metrics"]["path"])
    _, paths = references(session)
    rows = pd.read_csv(paths["data"] / "val_rows.csv")
    sf.validate_metrics(frame, rows)
    if not frame.arm.eq("H").all() or not frame.seed.eq(seed).all():
        raise ValueError("历史结果标签不符")
    with io.attempt(parent(session, "HA16", seed), io.binding(session, "formal", arm="HA16", seed=seed)) as output:
        frame["arm"] = "HA16"
        frame.to_csv(output / "metrics.csv", index=False)
        io.write_json(output / "reuse.json", {"source": str(source_attempt), "manifest": io.identity(source_attempt / "manifest.json"),
                      "initialization": io.identity(initial_path), "compatibility": "passed"})
        io.write_json(output / "result.json", result_record(run, output, "HA16", seed, epoch, reused=True))
    return output
