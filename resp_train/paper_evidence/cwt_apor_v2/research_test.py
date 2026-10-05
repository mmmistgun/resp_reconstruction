"""固定全矩阵 checkpoint 的 research-test；真实 test 操作逐命令显式开放。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from resp_train.data.factory import build_window_data
from resp_train.data.cache import WholeNightCache
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from . import artifacts as io
from .data import array_hash, raw_config, source_identities, verify_sources, condition_bundle, data_receipt
from .features import transform
from .engineering import require_gpu
from .model import build_model
from .runtime import cell_result
from .signals import reference_changes
from .spec import ARMS, SEEDS, OUTPUT_ROOT, config, plan
from .summary import tables, stratified_time_compression

TEST_ROW_HASH = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
VIEW_COUNTS = {"full": (2310, 8), "exclude670": (2231, 7), "subject670": (79, 1)}


def guard(confirmed):
    if confirmed is not True:
        raise PermissionError("需要当次 --confirm-research-test；历史实验授权不适用于本轮")


def prepare_allowlist(session, retry=False):
    frozen = io.require_data_scope(session)
    summary = io.completed(session / "summary", io.binding(session, "summary"))
    from .reuse import signals_receipt
    signals = signals_receipt(session)
    if summary is None or signals is None:
        raise RuntimeError("必须先完成全 60-cell validation 汇总及分析规则冻结")
    key = io.binding(session, "allowlist")
    parent = session / "research_test" / "allowlist"
    prior = io.completed(parent, key)
    if prior:
        return prior
    entries = []
    for cell in plan():
        path, result = cell_result(session, **cell)
        entries.append({**cell, "selected_epoch": result["selected_epoch"], "sources": result["sources"],
                        "formal_manifest": {"path": str(path / "manifest.json"), **io.identity(path / "manifest.json")}})
    with io.attempt(parent, key, retry) as output:
        io.write_json(output / "allowlist.json", {"entries": entries, "validation_summary": {"path": str(summary), **io.identity(summary / "manifest.json")},
                      "analysis": io.read_json(signals / "analysis.json"), "view_counts": VIEW_COUNTS,
                      "case_rule": frozen["spec"]["case_rule"], "shift_seed": frozen["spec"]["shift_seed"],
                      "test_data_accessed": False, "test_authorized": False})
    return output


def allowlist(session):
    io.require_data_scope(session)
    path = io.completed(session / "research_test" / "allowlist", io.binding(session, "allowlist"))
    if path is None:
        raise RuntimeError("缺少已冻结 checkpoint allowlist")
    payload = io.read_json(path / "allowlist.json")
    if len(payload["entries"]) != 60 or {(c["arm"], c["seed"]) for c in payload["entries"]} != {(c["arm"], c["seed"]) for c in plan()}:
        raise ValueError("allowlist 不是完整 60-cell")
    return path, payload


def test_key(session, phase, **fields):
    path, _ = allowlist(session)
    return io.binding(session, phase, allowlist=io.identity(path / "manifest.json")["sha256"], **fields)


def test_data(session):
    path = io.completed(session / "research_test" / "data", test_key(session, "test_data"))
    if path is None:
        raise RuntimeError("先执行 test prepare-data")
    return path, io.read_json(path / "data.json")


def prepare_test_data(session, confirmed=False, retry=False):
    guard(confirmed)
    _, frozen = allowlist(session)
    key = test_key(session, "test_data")
    parent = session / "research_test" / "data"
    prior = io.completed(parent, key)
    if prior:
        return prior
    with io.attempt(parent, key, retry) as output:
        io.write_json(output / "access.json", {"split": "test", "confirmed": True, "purpose": "fixed_allowlist_data_cache_and_case_identity"})
        cfg = raw_config(config("A0", SEEDS[0], session / "unused"))
        development, source = data_receipt(session)
        io.verify(source["index"]["path"], source["index"])
        bundle = build_window_data(cfg, split="test", max_windows=None, shuffle=False,
                                   sample_strategy=cfg.data.test_sample_strategy, sample_seed=cfg.data.test_sample_seed)
        rows = bundle.rows.reset_index(drop=True)
        if (len(rows) != 2310 or rows.samp_id.nunique() != 8 or set(rows.split) != {"test"}
                or rows.dataset_row_id.duplicated().any() or array_hash(rows.dataset_row_id.to_numpy(np.int64)) != TEST_ROW_HASH):
            raise ValueError("test 样本身份/分母漂移")
        dev_subjects = set()
        for split in ("train", "val"):
            dev_subjects.update(pd.read_csv(development / f"{split}_rows.csv").samp_id.astype(int))
        if set(rows.samp_id) & dev_subjects:
            raise ValueError("research-test 与 development 受试者重叠")
        rows.to_csv(output / "test_rows.csv", index=False)
        index = Path(cfg.data.dataset_root) / cfg.data.index_csv
        payload = {"index": {"path": str(index), **io.identity(index)}, "source_files": source_identities(index, {"test": rows}),
                   "row_order": TEST_ROW_HASH, "source_key": str(cfg.data.bcg_input_key), "count": 2310, "subjects": 8}
        reference = np.lib.format.open_memmap(output / "test_reference.npy", mode="w+", dtype=np.float32, shape=(2310, 1, 18000))
        changes = []
        for i in range(len(bundle.dataset)):
            target = bundle.dataset[i]["target"].numpy()
            reference[i] = target
            changes.append({"dataset_row_id": int(rows.iloc[i].dataset_row_id), **reference_changes(target, cfg)})
        reference.flush()
        del reference
        pd.DataFrame(changes).to_csv(output / "reference_changes.csv", index=False)
        cases = []
        for _, group in rows.groupby("samp_id", sort=True):
            cases.extend(group.iloc[sorted({0, len(group)//2, len(group)-1})].dataset_row_id.astype(int).tolist())
        io.write_json(output / "cases.json", {"rule": frozen["case_rule"], "dataset_row_ids": cases})
        rng = np.random.Generator(np.random.PCG64(frozen["shift_seed"]))
        np.save(output / "test_shift_frames.npy", np.stack([rng.choice(np.arange(60, 301), 3, replace=False) for _ in range(len(rows))]), allow_pickle=False)
        io.write_json(output / "data.json", payload)
        verify_sources(payload)
    return output


def build_test_cache(session, arm, confirmed=False, retry=False):
    guard(confirmed)
    frozen = io.load_session(session)
    if arm not in ARMS:
        raise ValueError("未知 arm")
    data_path, data = test_data(session)
    key = test_key(session, "test_cache", arm=arm)
    parent = session / "research_test" / "cache" / arm
    prior = io.completed(parent, key)
    if prior:
        return prior
    rep = frozen["representations"][arm]
    with io.attempt(parent, key, retry) as output:
        io.write_json(output / "access.json", {"split": "test", "confirmed": True, "target_read": False})
        verify_sources(data)
        rows = pd.read_csv(data_path / "test_rows.csv")
        source = WholeNightCache(data["index"]["path"])
        array = np.lib.format.open_memmap(output / "test_w.npy", mode="w+", dtype=np.float32, shape=(2310, *rep["shape"]))
        for i, row in enumerate(rows.itertuples(index=False)):
            x = source.get_arrays(str(row.source_npz), [str(row.bcg_signal_key)])[str(row.bcg_signal_key)]
            array[i] = transform(x[int(row.window_start_sample):int(row.window_end_sample)], rep)
        array.flush()
        del array
        np.save(output / "test_row_ids.npy", rows.dataset_row_id.to_numpy(np.int64), allow_pickle=False)
        io.write_json(output / "cache.json", {"representation": rep, "splits": ["test"], "finite": True,
                      "data_manifest": io.identity(data_path / "manifest.json"), "row_order": TEST_ROW_HASH,
                      "source_key": data["source_key"], "research_test": True, "target_read": False})
        verify_sources(data)
        io.verify_provenance(frozen["provenance"], session)
    return output


def test_bundle(session, arm, cfg, *, confirmed=False):
    guard(confirmed)
    frozen = io.load_session(session)
    data_path, data = test_data(session)
    verify_sources(data)
    cache = io.completed(session / "research_test" / "cache" / arm, test_key(session, "test_cache", arm=arm))
    if cache is None:
        raise RuntimeError("缺少对应原生表示的 test cache")
    raw = raw_config(cfg)
    bundle = build_window_data(raw, split="test", max_windows=None, shuffle=False,
                               sample_strategy=cfg.data.test_sample_strategy, sample_seed=cfg.data.test_sample_seed)
    if array_hash(bundle.rows.dataset_row_id.to_numpy(np.int64)) != TEST_ROW_HASH:
        raise ValueError("test loader 顺序漂移")
    return condition_bundle(bundle, cfg, cache, "test", frozen["representations"][arm])


def evaluate(session, arm, seed, device, confirmed=False, retry=False):
    guard(confirmed)
    frozen = io.load_session(session)
    _, allowed = allowlist(session)
    entries = [c for c in allowed["entries"] if (c["arm"], c["seed"]) == (arm, seed)]
    if len(entries) != 1:
        raise ValueError("cell 不在 allowlist")
    entry = entries[0]
    key = test_key(session, "test_evaluation", arm=arm, seed=seed)
    parent = session / "research_test" / "evaluation" / arm / f"seed_{seed}"
    prior = io.completed(parent, key)
    if prior:
        return prior
    require_gpu(session, device)
    with io.mutex(OUTPUT_ROOT / f".device_{torch.device(device).index}.mutex"):
        with io.attempt(parent, key, retry) as output:
            io.write_json(output / "access.json", {"split": "test", "confirmed": True, "entry": entry})
            for source in entry["sources"].values():
                io.verify(source["path"], source)
            cfg = OmegaConf.load(entry["sources"]["config"]["path"])
            checkpoint = torch.load(entry["sources"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
            if checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True) or checkpoint["epoch"] != entry["selected_epoch"]:
                raise ValueError("checkpoint 偏离 allowlist")
            cfg.training.device = device
            model = build_model(seed, frozen["representations"][arm]).to(device).eval()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            bundle = test_bundle(session, arm, cfg, confirmed=confirmed)
            prediction = collect_predictions(model, bundle.loader, device=device, max_windows=2310, use_amp=True)
            for name in ("r_tho_hat", "tho_ref"):
                if not np.isfinite(prediction[name]).all():
                    raise FloatingPointError("test prediction/reference 非有限")
            data_path, _ = test_data(session)
            if not np.array_equal(prediction["tho_ref"], np.load(data_path / "test_reference.npy", mmap_mode="r", allow_pickle=False)):
                raise ValueError("test 公共参考不一致")
            metrics = evaluate_task_predictions(prediction, cfg, include_test_only=False, method=arm)
            sf.validate_metrics(metrics, bundle.rows)
            metrics.insert(0, "arm", arm)
            metrics.insert(0, "seed", seed)
            metrics.to_csv(output / "metrics.csv", index=False)
            np.save(output / "prediction.npy", prediction["r_tho_hat"].astype(np.float32), allow_pickle=False)
            metrics[["dataset_row_id", "samp_id", "split"]].to_csv(output / "prediction_rows.csv", index=False)
            io.write_json(output / "reference_source.json", {"path": str(data_path / "test_reference.npy"), **io.identity(data_path / "test_reference.npy")})
            io.verify_provenance(frozen["provenance"], session)
    return output


def summarize_test(session, retry=False):
    allowlist(session)
    key = test_key(session, "test_summary")
    parent = session / "research_test" / "summary"
    prior = io.completed(parent, key)
    if prior:
        return prior
    frames = []
    for cell in plan():
        path = io.completed(session / "research_test" / "evaluation" / cell["arm"] / f"seed_{cell['seed']}", test_key(session, "test_evaluation", **cell))
        if path is None:
            raise RuntimeError(f"test 完整矩阵未完成: {cell}")
        frames.append(pd.read_csv(path / "metrics.csv"))
    full = pd.concat(frames, ignore_index=True)
    with io.attempt(parent, key, retry) as output:
        for view, mask in (("full", np.ones(len(full), bool)), ("exclude670", full.samp_id.ne(670)), ("subject670", full.samp_id.eq(670))):
            subset = full.loc[mask].copy()
            for _, group in subset.groupby(["arm", "seed"]):
                if (len(group), group.samp_id.nunique()) != VIEW_COUNTS[view]:
                    raise ValueError(f"{view} 分母错误")
            folder = output / view
            folder.mkdir()
            for name, table in tables(subset).items():
                table.to_csv(folder / f"{name}.csv", index=False)
            data_path, _ = test_data(session)
            _, frozen = allowlist(session)
            stratified_time_compression(subset, pd.read_csv(data_path / "reference_changes.csv"), frozen["analysis"]["thresholds"]).to_csv(folder / "time_compression_strata.csv", index=False)
    return output


def mechanisms(session, seed, device, confirmed=False, retry=False):
    guard(confirmed)
    import shutil
    from torch.utils.data import DataLoader
    from .interventions import CONDITIONS, evaluate_loader
    frozen = io.load_session(session)
    require_gpu(session, device)
    _, allowed = allowlist(session)
    entries = [e for e in allowed["entries"] if (e["arm"], e["seed"]) == ("A0", seed)]
    if len(entries) != 1:
        raise ValueError("B/seed 不在 allowlist")
    entry = entries[0]
    key = test_key(session, "test_mechanisms", seed=seed)
    parent = session / "research_test" / "mechanisms" / f"seed_{seed}"
    prior = io.completed(parent, key)
    if prior:
        return prior
    reference = io.completed(session / "research_test" / "evaluation" / "A0" / f"seed_{seed}", test_key(session, "test_evaluation", arm="A0", seed=seed))
    if reference is None:
        raise RuntimeError("先完成 B 的常规 test 评价以核对 FULL 重放")
    data_path, _ = test_data(session)
    cases = io.read_json(data_path / "cases.json")["dataset_row_ids"]
    estimate = len(cases)*len(CONDITIONS)*(9*96*140+97*360+8*18000)*4 + 2*1024**3
    if shutil.disk_usage(session).free < estimate:
        raise RuntimeError(f"案例导出至少需要 {estimate} bytes 空闲磁盘")
    with io.mutex(OUTPUT_ROOT / f".device_{torch.device(device).index}.mutex"):
        with io.attempt(parent, key, retry) as output:
            io.write_json(output / "access.json", {"split": "test", "confirmed": True, "entry": entry})
            for source in entry["sources"].values():
                io.verify(source["path"], source)
            cfg = OmegaConf.load(entry["sources"]["config"]["path"])
            cfg.training.device = device
            model = build_model(seed, frozen["representations"]["A0"]).to(device).eval()
            checkpoint = torch.load(entry["sources"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            bundle = test_bundle(session, "A0", cfg, confirmed=confirmed)
            loader = DataLoader(bundle.dataset, batch_size=8, shuffle=False, num_workers=0)
            frame = evaluate_loader(model, loader, cfg, frozen["representations"]["A0"],
                                    np.load(data_path / "test_shift_frames.npy", allow_pickle=False), set(cases), output, seed)
            from .interventions import verify_full_replay
            verify_full_replay(frame, pd.read_csv(reference / "metrics.csv"))
            io.write_json(output / "source.json", {"entry": entry, "case_ids": cases, "disk_estimate_bytes": estimate, "conditions": list(CONDITIONS)})
            io.verify_provenance(frozen["provenance"], session)
    return output


def summarize_mechanisms(session, retry=False):
    from .interventions import CONDITIONS
    allowlist(session)
    key = test_key(session, "test_mechanisms_summary")
    parent = session / "research_test" / "mechanisms_summary"
    prior = io.completed(parent, key)
    if prior:
        return prior
    frames = []
    for seed in SEEDS:
        path = io.completed(session / "research_test" / "mechanisms" / f"seed_{seed}", test_key(session, "test_mechanisms", seed=seed))
        if path is None:
            raise RuntimeError("test 机制矩阵未完成")
        frames.append(pd.read_csv(path / "metrics.csv"))
    full = pd.concat(frames, ignore_index=True)
    with io.attempt(parent, key, retry) as output:
        for view, mask in (("full", np.ones(len(full), bool)), ("exclude670", full.samp_id.ne(670)), ("subject670", full.samp_id.eq(670))):
            subset = full.loc[mask]
            for _, group in subset.groupby(["arm", "seed"]):
                if (len(group), group.samp_id.nunique()) != VIEW_COUNTS[view]:
                    raise ValueError("机制视图分母不符")
            folder = output / view
            folder.mkdir()
            result = tables(subset, {(c, s) for c in CONDITIONS for s in SEEDS}, [(c, "FULL__NAT") for c in CONDITIONS if c != "FULL__NAT"])
            for name, table in result.items():
                table.to_csv(folder / f"{name}.csv", index=False)
    return output
