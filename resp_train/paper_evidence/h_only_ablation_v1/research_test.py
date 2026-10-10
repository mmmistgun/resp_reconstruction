"""全 51 个 checkpoint 固定之后，显式授权统一 research-test。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from resp_train.data.factory import build_window_data
from resp_train.data.cache import WholeNightCache
from resp_train.paper_evidence.cwt_apor_v2 import data as source_data
from resp_train.paper_evidence.cwt_apor_v2.features import transform
from resp_train.paper_evidence.cwt_apor_v2.research_test import TEST_ROW_HASH, VIEW_COUNTS, guard
from . import artifacts as io
from .spec import SEEDS, ARMS, ROOT, config, plan
from .runtime import cell_result, references, require_device, evaluate
from .model import build_model


def freeze(session):
    io.load_session(session)
    summary = io.completed(session / "val_summary", io.binding(session, "val_summary"))
    if summary is None:
        raise RuntimeError("先完成完整51-cell validation汇总")
    entries = []
    for cell in plan():
        item = cell_result(session, **cell)
        if item is None:
            raise RuntimeError(f"矩阵未完成: {cell}")
        path, result = item
        entries.append({**cell, "sources": result["sources"], "selected_epoch": result["selected_epoch"],
                        "manifest": {"path": str(path / "manifest.json"), **io.identity(path / "manifest.json")}})
    with io.attempt(session / "allowlist", io.binding(session, "allowlist")) as output:
        io.write_json(output / "allowlist.json", {"entries": entries, "view_counts": VIEW_COUNTS,
                      "summary": {"path": str(summary), **io.identity(summary / "manifest.json")}, "test_accessed": False})
    return output


def allowlist(session):
    io.load_session(session)
    path = io.completed(session / "allowlist", io.binding(session, "allowlist"))
    if path is None:
        raise RuntimeError("缺少完整 checkpoint allowlist")
    entries = io.read_json(path / "allowlist.json")["entries"]
    if len(entries) != 51 or {(e["arm"], e["seed"]) for e in entries} != {(c["arm"], c["seed"]) for c in plan()}:
        raise ValueError("allowlist不是完整51-cell矩阵")
    for entry in entries:
        io.verify(entry["manifest"]["path"], entry["manifest"])
        for source in entry["sources"].values():
            io.verify(source["path"], source)
    return path, entries


def key(session, phase, **fields):
    path, _ = allowlist(session)
    return io.binding(session, phase, allowlist=io.identity(path / "manifest.json")["sha256"], **fields)


def prepare_data(session, confirmed=False, retry=False):
    guard(confirmed)
    stage_key = key(session, "test_data")
    frozen, paths = references(session)
    with io.attempt(session / "research_test/data", stage_key, retry) as output:
        io.write_json(output / "access.json", {"split": "test", "command": sys.argv, "confirmed": True})
        cfg = source_data.raw_config(config("HA0", SEEDS[0], output / "unused"))
        development = io.read_json(paths["data"] / "data.json")
        io.verify(development["index"]["path"], development["index"])
        bundle = build_window_data(cfg, split="test", max_windows=None, shuffle=False,
                    sample_strategy=cfg.data.test_sample_strategy, sample_seed=cfg.data.test_sample_seed)
        rows = bundle.rows.reset_index(drop=True)
        if (len(rows) != 2310 or rows.samp_id.nunique() != 8 or set(rows.split) != {"test"}
                or source_data.array_hash(rows.dataset_row_id.to_numpy(np.int64)) != TEST_ROW_HASH):
            raise ValueError("test窗口身份/分母漂移")
        dev_subjects = set()
        for split in ("train", "val"):
            dev_subjects.update(pd.read_csv(paths["data"] / f"{split}_rows.csv").samp_id)
        if set(rows.samp_id) & dev_subjects:
            raise ValueError("test与development受试者重叠")
        for view, part in (("full", rows), ("exclude670", rows[rows.samp_id.ne(670)]), ("subject670", rows[rows.samp_id.eq(670)])):
            if (len(part), part.samp_id.nunique()) != VIEW_COUNTS[view]:
                raise ValueError("敏感性视图集合不符")
        rows.to_csv(output / "test_rows.csv", index=False)
        data = {"index": development["index"], "source_files": source_data.source_identities(development["index"]["path"], {"test": rows}),
                "row_order": TEST_ROW_HASH}
        rep = frozen["representation"]
        reference = np.lib.format.open_memmap(output / "test_reference.npy", mode="w+", dtype=np.float32, shape=(2310, 1, 18000))
        features = np.lib.format.open_memmap(output / "test_w.npy", mode="w+", dtype=np.float32, shape=(2310, 41, 360))
        source = WholeNightCache(data["index"]["path"])
        for i, row in enumerate(rows.itertuples(index=False)):
            target = bundle.dataset[i]["target"].numpy()
            if target.shape != (1, 18000) or not np.isfinite(target).all():
                raise FloatingPointError("test target非法")
            reference[i] = target
            x = source.get_arrays(str(row.source_npz), [str(row.bcg_signal_key)])[str(row.bcg_signal_key)]
            features[i] = transform(x[int(row.window_start_sample):int(row.window_end_sample)], rep)
        reference.flush()
        features.flush()
        del reference, features
        np.save(output / "test_row_ids.npy", rows.dataset_row_id.to_numpy(np.int64), allow_pickle=False)
        io.write_json(output / "data.json", data)
        io.write_json(output / "cache.json", {"representation": rep, "splits": ["test"], "finite": True})
        source_data.verify_sources(data)
        io.load_session(session)
    return output


def test_result(session, arm, seed):
    path = io.completed(session / "research_test" / arm / f"seed_{seed}", key(session, "test", arm=arm, seed=seed))
    if path is None:
        return None
    result = io.read_json(path / "result.json")
    for entry in result["sources"].values():
        io.verify(entry["path"], entry)
    return path, result


def run_test(session, arm, seed, device, confirmed=False, retry=False):
    guard(confirmed)
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("未知cell")
    frozen = io.load_session(session)
    _, entries = allowlist(session)
    entry = next(e for e in entries if (e["arm"], e["seed"]) == (arm, seed))
    data_path = io.completed(session / "research_test/data", key(session, "test_data"))
    if data_path is None:
        raise RuntimeError("先执行test-prepare")
    source_data.verify_sources(io.read_json(data_path / "data.json"))
    device = require_device(device)
    with io.mutex(ROOT / "runs/h_only_ablation_v1" / f".device_{device.index}.mutex"):
        with io.attempt(session / "research_test" / arm / f"seed_{seed}", key(session, "test", arm=arm, seed=seed), retry) as output:
            cfg = config(arm, seed, output / "unused", str(device))
            rows = pd.read_csv(data_path / "test_rows.csv")
            raw = source_data.raw_config(cfg)
            bundle = build_window_data(raw, split="test", max_windows=None, shuffle=False,
                       sample_strategy=cfg.data.test_sample_strategy, sample_seed=cfg.data.test_sample_seed)
            if not np.array_equal(bundle.rows.dataset_row_id, rows.dataset_row_id):
                raise ValueError("test loader顺序漂移")
            if ARMS[arm].condition:
                bundle = source_data.condition_bundle(bundle, cfg, data_path, "test", frozen["representation"])
            model = build_model(arm, seed).to(device)
            checkpoint = torch.load(entry["sources"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            reference = np.load(data_path / "test_reference.npy", mmap_mode="r", allow_pickle=False)
            evaluate(model, bundle.loader, cfg, output, arm, seed, rows, reference, device, "test")
            io.write_json(output / "access.json", {"split": "test", "confirmed": True, "command": sys.argv,
                          "checkpoint": entry["sources"]["checkpoint"]})
            io.write_json(output / "result.json", {"arm": arm, "seed": seed, "sources": {
                "metrics": {"path": str(output / "metrics.csv"), **io.identity(output / "metrics.csv")},
                "checkpoint": entry["sources"]["checkpoint"]}})
            io.load_session(session)
    return output
