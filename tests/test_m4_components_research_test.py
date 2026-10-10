"""仅使用 synthetic/disposable fixtures 验证原始 M4 test 与历史参照复用。"""
import hashlib
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from scripts import eval_m4_components as entry


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def frequencies():
    return np.concatenate((np.linspace(.03, .8, 56), np.geomspace(.81, 8, 41)))


def test_confirmation_precedes_source_access(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "load_allowlist", lambda path: pytest.fail("不得读取来源"))
    monkeypatch.setattr(entry.shared.test_cache, "TfV1ResearchTestCacheReader", lambda *a, **k: pytest.fail("不得读取 cache"))
    with pytest.raises(PermissionError):
        entry.evaluate(tmp_path, "B2_seed20260811", "cuda:0")
    with pytest.raises(PermissionError):
        entry.summarize(tmp_path)
    with pytest.raises(PermissionError):
        entry.BandTestReader(frequencies(), "B0")
    with pytest.raises(ValueError, match="NoTF"):
        entry.BandTestReader(None, "B1", confirmed=True)


@pytest.mark.parametrize("arm", ["B0", "B2", "B3", "B4", "B5", "B6", "B7"])
def test_condition_cache_exact_band_and_frequency_identity(monkeypatch, arm):
    full = frequencies()
    values = np.arange(2 * 97 * 360, dtype=np.float32).reshape(2, 97, 360)
    class Cache:
        def __init__(self, *args, **kwargs):
            self.manifest = {"fixed_transform_spec": entry.shared.fixed_transform_spec()}
            self.arrays = {"w": values}
        def _open(self, name):
            assert name == "w_frequencies_hz.npy"
            return full
        def get(self, row_id):
            return {"w": torch.from_numpy(values[row_id].copy())}
    monkeypatch.setattr(entry, "COUNT", 2)
    monkeypatch.setattr(entry.shared.test_cache, "TfV1ResearchTestCacheReader", Cache)
    reader = entry.BandTestReader(full, arm, confirmed=True)
    indices = entry.band_indices(full, entry.ARMS[arm].band)
    torch.testing.assert_close(reader.get(1)["w"], torch.from_numpy(values[1, indices]))
    with pytest.raises(ValueError, match="频率"):
        entry.BandTestReader(full + .001, arm, confirmed=True)
    values[0, 0, 0] = np.inf
    with pytest.raises(FloatingPointError):
        reader.get(0)


def test_checkpoint_frequency_identity_and_notf_exclusion():
    full = frequencies()
    assert entry.checkpoint_frequencies({}, "B1", full) is None
    with pytest.raises(ValueError, match="NoTF"):
        entry.checkpoint_frequencies({"condition.frequencies_hz": torch.tensor(full)}, "B1", full)
    for arm in entry.ARMS:
        if arm == "B1":
            continue
        f = full[entry.band_indices(full, entry.ARMS[arm].band)]
        state = {"condition.frequencies_hz": torch.tensor(f)}
        assert np.array_equal(entry.checkpoint_frequencies(state, arm, full), f)
        drift = full.copy()
        drift[10 if arm == "B5" else 60] += .001
        with pytest.raises(ValueError, match="频率"):
            entry.checkpoint_frequencies(state, arm, drift)


def synthetic_batch(arm="B0"):
    t = torch.arange(18000).float() / 100
    signal = torch.stack([(1 + .2 * torch.sin(2 * torch.pi * .02 * t)) *
        torch.sin(2 * torch.pi * f * t) for f in (.2, .25)])[:, None]
    rows = pd.DataFrame({"dataset_row_id": [10, 20], "samp_id": [101, 102], "split": ["test", "test"]})
    batch = {"x": signal.clone(), "target": signal,
        "meta": {"dataset_row_id": torch.tensor([10, 20]), "samp_id": torch.tensor([101, 102]), "split": ["test", "test"]}}
    if arm != "B1":
        scales = len(entry.band_indices(frequencies(), entry.ARMS[arm].band))
        batch["tf"] = {"w": torch.ones(2, scales, 360)}
    return batch, rows


@pytest.mark.parametrize("arm", ["B0", "B1", "B2", "B4", "B5"])
@pytest.mark.parametrize("bad", ["order", "shape", "nan", "missing", "conditions"])
def test_guarded_batches_reject_invalid_inputs(arm, bad):
    batch, rows = synthetic_batch(arm)
    loader = [batch]
    if bad == "order":
        batch["meta"]["dataset_row_id"] = torch.tensor([20, 10])
    elif bad == "shape":
        batch["x"] = batch["x"][:, :, :-1]
    elif bad == "nan":
        batch["target"][0, 0, 0] = float("nan")
    elif bad == "conditions":
        batch["tf"] = {"w": torch.ones(2, 97, 360)} if arm == "B1" else {}
    else:
        loader = []
    with pytest.raises((ValueError, FloatingPointError)):
        list(entry.guarded_batches(loader, rows, arm))


@pytest.fixture
def native_metrics(monkeypatch, tmp_path):
    from resp_train.paper_evidence import w0_structural_factorial_v1_test as checks
    batch, rows = synthetic_batch()
    monkeypatch.setattr(checks, "COUNT", 2)
    monkeypatch.setattr(checks, "SUBJECTS", 2)
    monkeypatch.setattr(checks, "ROW_ORDER_SHA256", hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest())
    class IdentityModel(nn.Module):
        def forward(self, x, *, tf=None):
            return {"waveform": x}
    outputs = {}
    for arm in ("B0", "B1", "B2", "B4", "B5"):
        batch, rows = synthetic_batch(arm)
        cfg = entry.training.config(arm, entry.training.SEEDS[0], tmp_path)
        outputs[arm] = entry.infer_metrics(IdentityModel(), [batch], rows, cfg, "cpu")
    return outputs["B0"]


def test_native_metrics_and_all_matrix_comparisons(native_metrics):
    metrics, summary = native_metrics
    assert len(metrics) == 2 and summary.iloc[0].n_samples == 2
    for key in ("ibi_coverage", "respiratory_band_coherence", "constrained_ndtw",
                "target_stratified_envelope_spearman", "joint_prediction_degenerate"):
        assert key in metrics
    frames = [metrics.assign(arm=c["arm"], seed=c["seed"]) for c in entry.training.plan(include_references=True)]
    with pytest.raises(ValueError, match="完整"):
        entry.summary_tables(pd.concat(frames[:-1], ignore_index=True))
    table = pd.concat(frames, ignore_index=True)
    result = entry.summary_tables(table)
    assert len(result["per_seed"]) == 120 and len(result["across_seed"]) == 40
    assert len(result["paired_delta"]) == 105 and len(result["native_summary"]) == 24
    assert result["denominators"].rows.eq(2).all()
    assert len(result["incremental_delta"]) == 45
    assert result["incremental_delta"].improvement.eq(0).all()
    assert result["paired_summary"].tied_seeds.eq(3).all()
    for comparison, (arm, reference) in entry.training.INCREMENTS.items():
        rows = result["incremental_delta"].query("comparison == @comparison")
        assert set(rows.arm) == {arm} and set(rows.reference) == {reference}
    for arm, reference in entry.training.COMPARISONS.items():
        assert set(result["paired_delta"].query("arm == @arm").reference) == {reference}
    table.loc[len(table) - 1, "dataset_row_id"] = 30
    with pytest.raises(ValueError):
        entry.summary_tables(table)


def test_selector_rejects_partial_history_and_retains_earliest_tie():
    history = [{"epoch": i + 1, "val_local_rr_mae": v} for i, v in enumerate([2., 1., 1.])]
    completion = {"completed_epochs": 3, "best_epoch": 2, "best_local_rr": 1.}
    assert entry.selected_epoch(history, completion) == 2
    with pytest.raises(ValueError):
        entry.selected_epoch(history, {**completion, "best_epoch": 3})
    with pytest.raises(ValueError):
        entry.selected_epoch(history[:-1], completion)
    history[0]["val_local_rr_mae"] = np.nan
    with pytest.raises(FloatingPointError):
        entry.selected_epoch(history, completion)


def test_failure_preservation_success_reuse_and_tamper(tmp_path):
    entry.write_json(tmp_path / "allowlist.json", {"fixture": True})
    parent = tmp_path / "evaluation"
    with pytest.raises(RuntimeError, match="interrupted"):
        with entry.attempt(tmp_path, parent):
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="retry"):
        entry.completed(tmp_path, parent)
    assert entry.completed(tmp_path, parent, retry=True) is None
    with entry.attempt(tmp_path, parent) as output:
        (output / "metrics.csv").write_text("value\n1\n")
    assert entry.read_json(output / "started.json")["protocol"] == entry.PROTOCOL
    assert len(list(parent.glob("attempt_*"))) == 2
    assert entry.completed(tmp_path, parent) == output
    (output / "metrics.csv").write_text("value\n2\n")
    with pytest.raises(ValueError, match="身份"):
        entry.completed(tmp_path, parent)


def test_missing_cell_blocks_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "load_allowlist", lambda root: {"entries": entry.training.plan()})
    with pytest.raises(RuntimeError, match="尚未全部完成"):
        entry.summarize(tmp_path, confirmed=True)
    assert not list((tmp_path / "summary").glob("attempt_*"))


def test_shards_cover_all_eighteen_once():
    shards = [entry.training.plan(i, 2) for i in range(2)]
    assert [len(s) for s in shards] == [9, 9]
    assert len({c["cell"] for s in shards for c in s}) == 18


def test_prepare_rejects_unmatched_session_before_test_access(monkeypatch, tmp_path):
    session = tmp_path / "train"
    session.mkdir()
    entry.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(entry.training, "load_session", lambda path: None)
    monkeypatch.setattr(entry.shared.test_cache, "TfV1ResearchTestCacheReader", lambda *a, **k: pytest.fail("不得读取 test"))
    with pytest.raises(ValueError, match="session"):
        entry.prepare(session, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["matrix", "contract", "source", "snapshot", "digest"])
def test_allowlist_rejects_identity_drift(monkeypatch, tmp_path, change):
    monkeypatch.setattr(entry, "code_identity", lambda: {"fixture.py": "source"})
    (tmp_path / "source_snapshot.tar.gz").write_bytes(b"fixture")
    references = {"cells": {c["cell"]: {"path": "fixture", "checkpoint": {"sha256": "checkpoint"},
        "config": {"sha256": "config"}, "completion": {"sha256": "completion"}, "parameters": 1,
        "selected_epoch": 1} for c in entry.training.plan(include_references=True) if c["arm"] in entry.REFERENCES}}
    monkeypatch.setattr(entry, "reference_test_sources", lambda: references)
    entries = []
    for c in entry.training.plan(include_references=True):
        reference = references["cells"].get(c["cell"])
        entries.append({**c, "evaluation_role": "reference" if reference else "evaluate",
            **({**reference, "reference_test": reference} if reference else {})})
    lock = {"protocol": entry.PROTOCOL, "contract": entry.test_contract(),
        "training_session_sha256": entry.TRAINING_SESSION_SHA256,
        "validation_receipt_sha256": entry.VALIDATION_RECEIPT_SHA256,
        "entries": entries, "reference_test_sources": references, "code_sha256": entry.code_identity(),
        "snapshot_sha256": entry.sha(tmp_path / "source_snapshot.tar.gz")}
    entry.write_json(tmp_path / "allowlist.json", lock)
    entry.write_json(tmp_path / "allowlist_receipt.json", {"sha256": entry.sha(tmp_path / "allowlist.json")})
    assert entry.load_allowlist(tmp_path) == lock
    if change == "matrix":
        lock["entries"] = lock["entries"][:-1]
    elif change == "contract":
        lock["contract"]["batch_size"] = 64
    elif change == "source":
        lock["code_sha256"] = {}
    elif change == "snapshot":
        (tmp_path / "source_snapshot.tar.gz").write_bytes(b"changed")
    else:
        lock["protocol"] = "changed"
    # 仅篡改一次性 fixture；正式写入函数禁止覆盖历史产物。
    (tmp_path / "allowlist.json").write_text(json.dumps(lock))
    if change != "digest":
        (tmp_path / "allowlist_receipt.json").write_text(json.dumps({"sha256": entry.sha(tmp_path / "allowlist.json")}))
    with pytest.raises(ValueError):
        entry.load_allowlist(tmp_path)


def test_reference_cells_never_reenter_inference(monkeypatch, tmp_path):
    c = entry.training.plan(include_references=True)[0]
    monkeypatch.setattr(entry, "load_allowlist", lambda _: {"entries": [{**c, "evaluation_role": "reference"}]})
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("历史参照不得触碰 GPU"))
    with pytest.raises(ValueError, match="只读复用"):
        entry.evaluate(tmp_path, c["cell"], "cuda:0", confirmed=True)
    assert not (tmp_path / "evaluation").exists()


@pytest.fixture
def reference_fixture(tmp_path, monkeypatch, native_metrics):
    """六个可校验的历史评价，所有文件、指标与模型输入均为临时合成产物。"""
    root = tmp_path / "old_test"
    root.mkdir()
    metrics, summary = native_metrics
    monkeypatch.setattr(entry, "COUNT", 2)
    monkeypatch.setattr(entry, "SUBJECTS", 2)
    environment = {"torch": "fixture", "cuda": None, "packages": {}, "gpu": None}
    old_entries = []
    for arm, old_arm in entry.REFERENCES.items():
        for seed in entry.training.SEEDS:
            cell = f"{old_arm}_seed{seed}"
            source = tmp_path / "development" / cell
            source.mkdir(parents=True)
            records = {}
            for key in ("checkpoint", "config", "completion"):
                path = source / f"{key}.json"
                entry.write_json(path, {"synthetic": True, "source_arm": old_arm, "seed": seed})
                records[key] = {"path": str(path), "sha256": entry.sha(path)}
            old_entries.append({"arm": old_arm, "seed": seed, "cell": cell, "selected_epoch": 1,
                "parameters": 1, "training_environment": environment, **records})
    old_lock = {"entries": old_entries, "contract": {**entry.shared.test_contract(), "batch_size": entry.BATCH_SIZE},
        "development_subjects": [1, 2]}
    entry.write_json(root / "allowlist.json", old_lock)
    monkeypatch.setattr(entry, "REFERENCE_TEST_ROOT", root)
    monkeypatch.setattr(entry, "REFERENCE_ALLOWLIST_SHA256", entry.sha(root / "allowlist.json"))
    monkeypatch.setattr(entry.p1test, "load_allowlist", lambda _: old_lock)
    sources = {}
    for old in old_entries:
        with entry.p1test.attempt(root, root / "evaluation" / old["cell"]) as result:
            metrics.assign(arm=old["arm"], seed=old["seed"]).to_csv(result / "metrics.csv", index=False)
            summary.to_csv(result / "metrics_summary.csv", index=False)
            metrics[list(entry.shared.IDENTITY_COLUMNS)].to_csv(result / "test_rows.csv", index=False)
            (result / "evaluation_config.yaml").write_text("fixture: true\n")
            entry.write_json(result / "environment.json", environment)
            entry.write_json(result / "access_started.json", {"fixture": True})
            entry.write_json(result / "evaluation.json", {"cell": old["cell"], "arm": old["arm"], "seed": old["seed"],
                "selected_epoch": 1, "checkpoint_sha256": old["checkpoint"]["sha256"],
                "rows": 2, "subjects": 2, "evidence_role": entry.EVIDENCE_ROLE})
        sources[old["cell"]] = entry.sha(result / "receipt.json")
    with entry.p1test.attempt(root, root / "summary") as result:
        entry.write_json(result / "sources.json", sources)
    monkeypatch.setattr(entry, "REFERENCE_SUMMARY_SHA256", entry.sha(result / "receipt.json"))
    return root, old_lock


def test_reference_audit_never_reads_metric_values_or_data(reference_fixture, monkeypatch):
    root, _ = reference_fixture
    before = {str(p): entry.sha(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("来源审计不解析指标值"))
    monkeypatch.setattr(entry.shared, "read_research_v2_index", lambda *a, **k: pytest.fail("不得读取 test index"))
    monkeypatch.setattr(entry.shared.test_cache, "TfV1ResearchTestCacheReader", lambda *a, **k: pytest.fail("不得读取 cache"))
    references = entry.reference_test_sources()
    assert len(references["cells"]) == 6
    assert references["cells"]["B0_seed20260811"]["source_arm"] == "M4"
    assert before == {str(p): entry.sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["metric_artifact", "checkpoint", "contract", "summary"])
def test_reference_audit_rejects_drift(reference_fixture, change):
    root, lock = reference_fixture
    if change == "metric_artifact":
        path = next((root / "evaluation").glob("*/attempt_*/metrics.csv"))
        path.write_text("changed\n")
    elif change == "checkpoint":
        Path(lock["entries"][0]["checkpoint"]["path"]).write_text("changed\n")
    elif change == "contract":
        lock["contract"]["batch_size"] = 64
    else:
        path = next((root / "summary").glob("attempt_*/receipt.json"))
        data = entry.read_json(path)
        data["note"] = "changed"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        entry.reference_test_sources()


def test_full_summary_reads_reference_without_rewriting(reference_fixture, tmp_path, monkeypatch, native_metrics):
    old_root, _ = reference_fixture
    references = entry.reference_test_sources()
    root = tmp_path / "new_test"
    root.mkdir()
    entry.write_json(root / "allowlist.json", {"fixture": True})
    metrics, _ = native_metrics
    entries = []
    for c in entry.training.plan(include_references=True):
        reference = references["cells"].get(c["cell"])
        current = {**c, "evaluation_role": "reference" if reference else "evaluate",
            "selected_epoch": 1, "parameters": 1, "checkpoint": {"sha256": "new_checkpoint"}}
        if reference:
            current.update(reference_test=reference, checkpoint=reference["checkpoint"])
        else:
            with entry.attempt(root, root / "evaluation" / c["cell"]) as result:
                metrics.assign(arm=c["arm"], seed=c["seed"]).to_csv(result / "metrics.csv", index=False)
                entry.write_json(result / "evaluation.json", {"cell": c["cell"], "arm": c["arm"], "seed": c["seed"],
                    "selected_epoch": 1, "checkpoint_sha256": "new_checkpoint", "rows": 2, "subjects": 2,
                    "evidence_role": entry.EVIDENCE_ROLE})
        entries.append(current)
    monkeypatch.setattr(entry, "load_allowlist", lambda _: {"entries": entries})
    before = {str(p): entry.sha(p) for p in old_root.rglob("*") if p.is_file()}
    output = entry.summarize(root, confirmed=True)
    assert entry.summarize(root, confirmed=True) == output
    assert len(pd.read_csv(output / "per_seed.csv")) == 120
    assert len(pd.read_csv(output / "native_summary.csv")) == 24
    sources = pd.read_csv(output / "sources.csv")
    assert sources.role.eq("reference").sum() == 6
    assert sources.role.eq("evaluate").sum() == 18
    assert len(list((root / "evaluation").iterdir())) == 18
    assert before == {str(p): entry.sha(p) for p in old_root.rglob("*") if p.is_file()}


@pytest.fixture
def prepare_fixture(tmp_path, monkeypatch):
    """用完整矩阵的微型 development checkpoint 验证 prepare 的选点与参照映射。"""
    session = tmp_path / "train"
    session.mkdir()
    entry.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(entry, "TRAINING_SESSION_SHA256", entry.sha(session / "session.json"))
    monkeypatch.setattr(entry.training, "load_session", lambda _: {})
    old_session = tmp_path / "p1"
    monkeypatch.setattr(entry.training, "REFERENCE_SESSION", old_session)
    row_frames = {"train": pd.DataFrame({"dataset_row_id": np.arange(10141), "samp_id": np.arange(10141)%32,
        "split": "train"}), "val": pd.DataFrame({"dataset_row_id": np.arange(2675)+20000,
        "samp_id": np.arange(2675)%7+100, "split": "val"})}
    source = {"cache_lock": {"row_identity": {f"{s}_row_content_sha256": hashlib.sha256(
        frame.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest() for s,frame in row_frames.items()}},
        "dataset_index": {"path": "never_opened", "sha256": "fixture"}}
    source_path = tmp_path / "source.json"
    entry.write_json(source_path, source)
    monkeypatch.setattr(entry.training.legacy, "P2_LOCK_PATH", str(source_path))
    monkeypatch.setattr(entry.training.legacy, "P2_LOCK_SHA256", entry.sha(source_path))
    validation_sources, refs = {}, {}
    for c in entry.training.plan(include_references=True):
        old_arm = entry.REFERENCES.get(c["arm"], c["arm"])
        directory = (old_session / "cells" / f"{old_arm}_seed{c['seed']}" if c["arm"] in entry.REFERENCES
                     else session / "cells" / c["cell"])
        directory.mkdir(parents=True)
        builder = entry.training.p1.config if c["arm"] in entry.REFERENCES else entry.training.config
        cfg = builder(old_arm, c["seed"], directory)
        cfg_path = directory / "config.yaml"
        OmegaConf.save(cfg, cfg_path)
        identity = entry.training.legacy.config_identity(cfg)
        state = {} if c["arm"] == "B1" else {"condition.frequencies_hz": torch.tensor(
            frequencies()[entry.band_indices(frequencies(), entry.ARMS[c['arm']].band)])}
        best, final = directory / "best.pt", directory / "final.pt"
        history = [{"epoch": 1, "val_local_rr_mae": .5}, {"epoch": 2, "val_local_rr_mae": .6}]
        torch.save({"epoch": 1, "config": identity, "model_state_dict": state}, best)
        torch.save({"epoch": 2, "config": identity, "model_state_dict": state, "history": history}, final)
        attempt = directory / "attempt"
        attempt.mkdir()
        for split, rows in row_frames.items():
            rows.to_csv(attempt / f"{split}_rows.csv", index=False)
        (attempt / "metrics.csv").write_text("fixture\n")
        entry.write_json(directory / "environment.json", {"fixture": True})
        completion = {"arm": old_arm, "seed": c["seed"], "completed_epochs": 2, "best_epoch": 1, "best_local_rr": .5,
            "best_checkpoint": "best.pt", "final_checkpoint": "final.pt", "metrics": "attempt/metrics.csv", "parameters": 1,
            "artifacts": {n: entry.sha(directory/n) for n in ("best.pt", "final.pt", "config.yaml", "attempt/metrics.csv")}}
        entry.write_json(directory / "completed.json", completion)
        validation_sources[c["cell"]] = {"path": str(directory), "source_arm": old_arm, "role": c["role"],
            "completed_sha256": entry.sha(directory / "completed.json")}
        if c["arm"] in entry.REFERENCES:
            refs[c["cell"]] = {"path": "test_metrics_never_opened", "source_arm": old_arm,
                "source_cell": f"{old_arm}_seed{c['seed']}", "selected_epoch": 1, "parameters": 1,
                **{k:{"path":str(p), "sha256":entry.sha(p)} for k,p in [("checkpoint",best), ("config",cfg_path),
                    ("completion",directory/"completed.json")]}}
    summary = session / "summary/fixture"
    summary.mkdir(parents=True)
    entry.write_json(summary / "receipt.json", {"sources": validation_sources, "artifacts": {}})
    monkeypatch.setattr(entry, "VALIDATION_RECEIPT_SHA256", entry.sha(summary / "receipt.json"))
    references = {"cells": refs, "development_subjects": list(range(32))+list(range(100,107))}
    monkeypatch.setattr(entry, "reference_test_sources", lambda: references)
    source_file = "scripts/eval_m4_components.py"
    monkeypatch.setattr(entry, "code_identity", lambda: {source_file: entry.sha(entry.ROOT/source_file)})
    return session, references


def test_prepare_complete_development_only_and_repeatable(prepare_fixture, tmp_path, monkeypatch):
    session, _ = prepare_fixture
    monkeypatch.setattr(entry.shared, "read_research_v2_index", lambda *a, **k: pytest.fail("不得读取 test index"))
    monkeypatch.setattr(entry.shared.test_cache, "TfV1ResearchTestCacheReader", lambda *a, **k: pytest.fail("不得读取 test cache"))
    output = entry.prepare(session, tmp_path / "allowlist")
    assert entry.prepare(session, output) == output
    lock = entry.load_allowlist(output)
    assert len(lock["entries"]) == 24
    assert sum(e["evaluation_role"] == "evaluate" for e in lock["entries"]) == 18
    assert not lock["test_array_read"] and not lock["test_manifest_read"] and not lock["test_metric_values_read"]
    assert not lock["model_inference_used"]


def test_prepare_rejects_reference_checkpoint_mismatch(prepare_fixture, tmp_path):
    session, references = prepare_fixture
    references["cells"]["B0_seed20260811"]["checkpoint"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="历史 test 不一致"):
        entry.prepare(session, tmp_path / "allowlist")
    assert not (tmp_path / "allowlist/allowlist_receipt.json").exists()
