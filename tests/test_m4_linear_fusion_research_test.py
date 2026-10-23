"""合成信号与一次性 provenance 验证单投影 test 入口和跨 split 0.01 判断。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts import eval_m4_linear_fusion as entry


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def test_confirmation_before_access(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "load_allowlist", lambda _: pytest.fail("提前读取来源"))
    with pytest.raises(PermissionError):
        entry.evaluate(tmp_path, "L1_seed20260811", "cuda:0")
    with pytest.raises(PermissionError):
        entry.summarize(tmp_path)
    with pytest.raises(PermissionError):
        entry.BandTestReader(None, "L1")


def test_reference_cannot_be_evaluated(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "load_allowlist", lambda _: {"entries": [
        {"cell": "L0_seed20260811", "arm": "L0", "evaluation_role": "reference"}]})
    with pytest.raises(ValueError, match="只读复用"):
        entry.evaluate(tmp_path, "L0_seed20260811", "cuda:0", confirmed=True)
    assert not (tmp_path / "evaluation").exists()


@pytest.fixture
def native(monkeypatch, tmp_path):
    from resp_train.paper_evidence import w0_structural_factorial_v1_test as checks
    t = torch.arange(18000).float() / 100
    signal = torch.stack([(1 + .2 * torch.sin(2 * torch.pi * .02 * t)) *
        torch.sin(2 * torch.pi * f * t) for f in (.2, .25)])[:, None]
    rows = pd.DataFrame({"dataset_row_id": [10, 20], "samp_id": [101, 102], "split": ["test", "test"]})
    batch = {"x": signal.clone(), "target": signal, "tf": {"w": torch.ones(2, 97, 360)},
        "meta": {"dataset_row_id": torch.tensor([10, 20]), "samp_id": torch.tensor([101, 102]), "split": ["test", "test"]}}
    monkeypatch.setattr(checks, "COUNT", 2)
    monkeypatch.setattr(checks, "SUBJECTS", 2)
    monkeypatch.setattr(checks, "ROW_ORDER_SHA256", hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest())
    class Identity(nn.Module):
        def forward(self, x, *, tf=None):
            return {"waveform": x}
    cfg = entry.training.config("L1", entry.training.SEEDS[0], tmp_path)
    metrics, _ = entry.infer_metrics(Identity(), [batch], rows, cfg, "cpu")
    return metrics, batch, rows


def test_complete_test_summary_cross_split_and_report(native, tmp_path):
    metrics, _, _ = native
    assert {"ibi_coverage", "respiratory_band_coherence", "constrained_ndtw"} <= set(metrics)
    frames = [metrics.assign(arm=c["arm"], seed=c["seed"]) for c in entry.training.plan(include_references=True)]
    with pytest.raises(ValueError, match="完整"):
        entry.summary_tables(pd.concat(frames[:-1], ignore_index=True))
    frame = pd.concat(frames, ignore_index=True)
    validation = entry.summary_tables(frame)
    test = entry.summary_tables(frame)
    comparison = entry.compare_splits(validation, test)
    assert len(comparison) == 10 and len(test["native_summary"]) == 6
    assert entry.removal_decision(comparison)["recommendation"] == "supports_single_projection"
    diagnostic = comparison.copy()
    diagnostic.loc[diagnostic.aggregation.eq("subject_macro"), "test_mean_classification"] = "worsened"
    assert entry.removal_decision(diagnostic)["recommendation"] == "supports_single_projection"
    entry.write_report(tmp_path, validation, test, comparison)
    assert len(list(tmp_path.glob("*.png"))) == 2
    assert (tmp_path / "report.md").exists() and (tmp_path / "decision.json").exists()
    frame.loc[frame.arm.eq("L1"), "local_rr_mae_bpm"] += .02
    test = entry.summary_tables(frame)
    comparison = entry.compare_splits(validation, test)
    assert entry.removal_decision(comparison)["recommendation"] == "tradeoff_exceeds_threshold"
    assert comparison.loc[comparison.metric.eq("local_rr_mae_bpm"), "test_worsened_at_threshold_seeds"].eq(3).all()
    frame.loc[len(frame)-1, "dataset_row_id"] = 30
    with pytest.raises(ValueError):
        entry.summary_tables(frame)


@pytest.mark.parametrize("bad", ["order", "shape", "nan", "missing"])
def test_batch_guards(native, bad):
    _, batch, rows = native
    if bad == "order":
        batch["meta"]["dataset_row_id"] = torch.tensor([20, 10])
    elif bad == "shape":
        batch["tf"]["w"] = batch["tf"]["w"][:, :-1]
    elif bad == "nan":
        batch["target"][0, 0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        list(entry.guarded_batches([] if bad == "missing" else [batch], rows, "L1"))


def test_cache_frequency_and_nonfinite_rejection(monkeypatch):
    frequencies = np.concatenate((np.linspace(.03, .8, 56), np.geomspace(.81, 8, 41)))
    values = np.ones((2, 97, 360), dtype=np.float32)
    class Cache:
        def __init__(self, *a, **kw):
            self.manifest = {"fixed_transform_spec": entry.shared.fixed_transform_spec()}
            self.arrays = {"w": values}
        def _open(self, name):
            return frequencies
        def get(self, row_id):
            return {"w": torch.from_numpy(values[row_id].copy())}
    monkeypatch.setattr(entry, "COUNT", 2)
    monkeypatch.setattr(entry.shared.test_cache, "TfV1ResearchTestCacheReader", Cache)
    reader = entry.BandTestReader(frequencies, "L1", confirmed=True)
    assert reader.get(0)["w"].shape == (97, 360)
    with pytest.raises(ValueError, match="频率"):
        entry.BandTestReader(frequencies + .01, "L1", confirmed=True)
    values[0, 0, 0] = np.inf
    with pytest.raises(FloatingPointError):
        reader.get(0)


@pytest.fixture
def reference_fixture(tmp_path, monkeypatch):
    root = tmp_path / "old_test"
    root.mkdir()
    entries = []
    environment = {"torch": "fixture", "cuda": None, "packages": {}, "gpu": None}
    for seed in entry.training.SEEDS:
        cell = f"A1_seed{seed}"
        directory = root / "training" / cell
        directory.mkdir(parents=True)
        files = {}
        for key in ("checkpoint", "config", "completion"):
            path = directory / key
            path.write_text("fixture")
            files[key] = {"path": str(path), "sha256": entry.sha(path)}
        entries.append({"cell": cell, "arm": "A1", "seed": seed, "selected_epoch": 2,
            "parameters": 1, "training_environment": environment, **files})
    (root / "source_snapshot.tar.gz").write_bytes(b"fixture")
    source = "scripts/m4_linear_fusion_model.py"
    lock = {"contract": {**entry.shared.test_contract(), "batch_size": entry.BATCH_SIZE}, "entries": entries,
        "development_subjects": [1, 2], "snapshot_sha256": entry.sha(root / "source_snapshot.tar.gz"),
        "code_sha256": {source: entry.sha(entry.ROOT / source)}}
    entry.write_json(root / "allowlist.json", lock)
    sources = {}
    for e in entries:
        with entry.attempt(root, root / "evaluation" / e["cell"]) as result:
            for name in ("metrics.csv", "metrics_summary.csv", "test_rows.csv", "evaluation_config.yaml", "access_started.json"):
                (result / name).write_text("fixture")
            entry.write_json(result / "environment.json", environment)
            entry.write_json(result / "evaluation.json", {"cell": e["cell"], "arm": "A1", "seed": e["seed"],
                "selected_epoch": 2, "checkpoint_sha256": e["checkpoint"]["sha256"],
                "rows": entry.COUNT, "subjects": entry.SUBJECTS, "evidence_role": entry.EVIDENCE_ROLE})
        sources[e["cell"]] = {"path": str(result), "source_arm": "A1", "source_cell": e["cell"],
            "role": "evaluate", "receipt_sha256": entry.sha(result / "receipt.json")}
    with entry.attempt(root, root / "summary") as result:
        entry.write_json(result / "sources.json", sources)
    monkeypatch.setattr(entry, "REFERENCE_TEST_ROOT", root)
    monkeypatch.setattr(entry, "REFERENCE_ALLOWLIST_SHA256", entry.sha(root / "allowlist.json"))
    monkeypatch.setattr(entry, "REFERENCE_SUMMARY_SHA256", entry.sha(result / "receipt.json"))
    return root


def test_reference_provenance_is_read_only_and_detects_tamper(reference_fixture):
    before = {str(p): entry.sha(p) for p in reference_fixture.rglob("*") if p.is_file()}
    result = entry.reference_test_sources()
    assert len(result["cells"]) == 3
    assert before == {str(p): entry.sha(p) for p in reference_fixture.rglob("*") if p.is_file()}
    path = Path(next(iter(result["cells"].values()))["path"]) / "metrics.csv"
    path.write_text("changed fixture")
    with pytest.raises(ValueError):
        entry.reference_test_sources()


def test_missing_new_cell_blocks_summary(monkeypatch, tmp_path):
    entries = [{**c, "evaluation_role": "reference" if c["arm"] == "L0" else "evaluate"}
               for c in entry.training.plan(include_references=True)]
    monkeypatch.setattr(entry, "load_allowlist", lambda _: {"entries": entries})
    with pytest.raises(RuntimeError, match="尚未全部完成"):
        entry.summarize(tmp_path, confirmed=True)
    assert not list((tmp_path / "summary").glob("attempt_*"))


def test_failed_attempt_is_preserved(tmp_path):
    entry.write_json(tmp_path / "allowlist.json", {"fixture": True})
    parent = tmp_path / "evaluation"
    with pytest.raises(RuntimeError):
        with entry.attempt(tmp_path, parent):
            raise RuntimeError("fixture failure")
    with pytest.raises(RuntimeError, match="retry"):
        entry.completed(tmp_path, parent)
    with entry.attempt(tmp_path, parent) as result:
        (result / "metrics.csv").write_text("value\n1\n")
    assert entry.completed(tmp_path, parent) == result
    assert len(list(parent.glob("attempt_*"))) == 2


def test_prepare_rejects_wrong_training_identity_before_test_access(monkeypatch, tmp_path):
    session = tmp_path / "train"
    session.mkdir()
    entry.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(entry.training, "load_session", lambda _: {})
    monkeypatch.setattr(entry, "reference_test_sources", lambda: pytest.fail("提前读取 test 来源"))
    with pytest.raises(ValueError, match="session"):
        entry.prepare(session, tmp_path / "test")


def test_fixed_three_new_checkpoints():
    cells = entry.training.plan()
    assert len(cells) == 3 and {c["arm"] for c in cells} == {"L1"}
    assert len(entry.training.plan(include_references=True)) == 6


@pytest.mark.parametrize("fail", [False, True])
def test_parallel_launcher_waits_and_preserves_logs(tmp_path, fail):
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "run_m4_linear_fusion_test.sh"
    shutil.copyfile(entry.ROOT / "scripts/run_m4_linear_fusion_test.sh", launcher)
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("""#!/usr/bin/env python3
import os,sys,time
from pathlib import Path
args=sys.argv[1:]
assert '--confirm-research-test' in args
if 'evaluate' in args:
    shard=args[args.index('--shard-index')+1]
    assert args[args.index('--shard-count')+1]=='2'
    assert args[args.index('--device')+1]=='cuda:'+shard
    time.sleep(.03 if shard=='0' else .08)
    Path('worker_'+shard).write_text('finished')
    sys.exit(7 if os.getenv('FIXTURE_FAIL')=='1' and shard=='0' else 0)
assert Path('worker_0').exists() and Path('worker_1').exists()
Path('summary_called').write_text('complete')
""")
    python.chmod(0o755)
    session = root / "session"
    session.mkdir()
    (session / "allowlist_receipt.json").write_text("{}")
    env = {**os.environ, "FIXTURE_FAIL": "1" if fail else "0"}
    command = ["bash", str(launcher), str(session), "0", "1"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == (7 if fail else 0), result.stderr
    assert (root / "worker_0").exists() and (root / "worker_1").exists()
    assert (root / "summary_called").exists() is (not fail)
    before = {p.name: p.read_bytes() for p in session.glob("gpu_*.log")}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert before == {p.name: p.read_bytes() for p in session.glob("gpu_*.log")}
