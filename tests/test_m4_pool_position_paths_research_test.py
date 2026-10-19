"""使用合成信号和一次性产物核验 test 边界与完整四条件汇总。"""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from scripts import eval_m4_pool_position_paths as entry


def frequencies():
    return np.concatenate((np.linspace(.03, .8, 56), np.geomspace(.81, 8, 41)))


def test_confirmation_before_access(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "load_allowlist", lambda path: pytest.fail("提前访问来源"))
    with pytest.raises(PermissionError):
        entry.evaluate(tmp_path, "A_seed20260811", "cuda:0")
    with pytest.raises(PermissionError):
        entry.summarize(tmp_path)
    with pytest.raises(PermissionError):
        entry.BandTestReader(frequencies(), "A")


@pytest.mark.parametrize("arm", list(entry.ARMS))
def test_stage_config_and_frequency_routing(arm, tmp_path):
    builder = entry.training.base.config if arm in ("A", "B") else entry.training.config
    cfg = builder(arm, entry.training.SEEDS[0], tmp_path)
    assert entry.configured_arm(cfg) == arm
    full = frequencies()
    state = {"condition.frequencies_hz": torch.tensor(full)}
    np.testing.assert_array_equal(entry.checkpoint_frequencies(state, arm, full), full)
    with pytest.raises(ValueError):
        entry.checkpoint_frequencies(state, arm, full + .001)


def test_full_cache_and_finite_guard(monkeypatch):
    full = frequencies()
    values = np.ones((2, 97, 360), dtype=np.float32)
    class Cache:
        def __init__(self, *a, **kw):
            self.manifest = {"fixed_transform_spec": entry.shared.fixed_transform_spec()}
            self.arrays = {"w": values}
        def _open(self, name):
            return full
        def get(self, row_id):
            return {"w": torch.from_numpy(values[row_id].copy())}
    monkeypatch.setattr(entry, "COUNT", 2)
    monkeypatch.setattr(entry.shared.test_cache, "TfV1ResearchTestCacheReader", Cache)
    for arm in entry.ARMS:
        reader = entry.BandTestReader(full, arm, confirmed=True)
        assert reader.get(0)["w"].shape == (97, 360)
    values[0, 0, 0] = np.inf
    with pytest.raises(FloatingPointError):
        reader.get(0)


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
    cfg = entry.training.base.config("A", entry.training.SEEDS[0], tmp_path)
    metrics, _ = entry.infer_metrics(Identity(), [batch], rows, cfg, "cpu")
    return metrics, batch, rows


def test_complete_summary_native_metrics_and_report(native, tmp_path):
    metrics, _, _ = native
    assert {"ibi_coverage", "respiratory_band_coherence", "constrained_ndtw"} <= set(metrics)
    frames = [metrics.assign(arm=c["arm"], seed=c["seed"]) for c in entry.plan()]
    with pytest.raises(ValueError, match="完整"):
        entry.summary_tables(pd.concat(frames[:-1], ignore_index=True))
    frame = pd.concat(frames, ignore_index=True)
    # 指标列增加已知四条件偏移，独立核对主效应和交互公式。
    offset = frame.arm.map({"A": .08, "B": 0., "C": .01, "D": .02})
    frame["local_rr_mae_bpm"] += offset
    tables = entry.summary_tables(frame)
    effects = tables["path_effects"].query("metric == 'local_rr_mae_bpm'")
    assert len(tables["per_seed"]) == 60 and len(tables["native_summary"]) == 12
    assert len(tables["path_effects"]) == 105
    for effect, expected in (("score_main", .035), ("value_main", .045),
                             ("interaction_A_minus_C_minus_D_plus_B", .05)):
        assert np.allclose(effects.loc[effects.effect.eq(effect), "delta"], expected)
    entry.write_report(tmp_path, tables)
    assert len(list(tmp_path.glob("*.png"))) == 2
    assert "评分" in (tmp_path / "report.md").read_text()
    frame.loc[len(frame)-1, "dataset_row_id"] = 30
    with pytest.raises(ValueError):
        entry.summary_tables(frame)


@pytest.mark.parametrize("bad", ["order", "shape", "nan", "missing"])
def test_batch_guard(native, bad):
    _, batch, rows = native
    if bad == "order":
        batch["meta"]["dataset_row_id"] = torch.tensor([20, 10])
    elif bad == "shape":
        batch["tf"]["w"] = batch["tf"]["w"][:, :-1]
    elif bad == "nan":
        batch["x"][0, 0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        list(entry.guarded_batches([] if bad == "missing" else [batch], rows, "A"))


def test_failed_attempt_retention_and_success_integrity(tmp_path):
    entry.write_json(tmp_path / "allowlist.json", {"fixture": True})
    parent = tmp_path / "evaluation"
    with pytest.raises(RuntimeError):
        with entry.attempt(tmp_path, parent):
            raise RuntimeError("fixture failure")
    with pytest.raises(RuntimeError, match="retry"):
        entry.completed(tmp_path, parent)
    with entry.attempt(tmp_path, parent) as result:
        (result / "metrics.csv").write_text("fixture\n1\n")
    assert entry.completed(tmp_path, parent) == result
    assert len(list(parent.glob("attempt_*"))) == 2
    (result / "metrics.csv").write_text("changed")
    with pytest.raises(ValueError):
        entry.completed(tmp_path, parent)


def test_missing_cell_blocks_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "load_allowlist", lambda _: {"entries": entry.plan()})
    with pytest.raises(RuntimeError, match="尚未全部完成"):
        entry.summarize(tmp_path, confirmed=True)


def test_plan_covers_all_sources():
    assert len(entry.plan()) == 12
    assert {c["arm"] for c in entry.plan()} == set("ABCD")
    assert len({c["cell"] for i in range(2) for c in entry.plan(i, 2)}) == 12


@pytest.mark.parametrize("change", ["matrix", "contract", "source", "snapshot", "digest"])
def test_allowlist_drift(monkeypatch, tmp_path, change):
    monkeypatch.setattr(entry, "code_identity", lambda: {"fixture.py": "source"})
    (tmp_path / "source_snapshot.tar.gz").write_bytes(b"fixture")
    lock = {"protocol": entry.PROTOCOL, "contract": entry.test_contract(),
        "training_session_sha256": entry.TRAINING_SESSION_SHA256,
        "validation_receipt_sha256": entry.VALIDATION_RECEIPT_SHA256,
        "entries": entry.plan(), "code_sha256": entry.code_identity(),
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
    (tmp_path / "allowlist.json").write_text(json.dumps(lock))
    if change != "digest":
        (tmp_path / "allowlist_receipt.json").write_text(json.dumps({"sha256": entry.sha(tmp_path / "allowlist.json")}))
    with pytest.raises(ValueError):
        entry.load_allowlist(tmp_path)
