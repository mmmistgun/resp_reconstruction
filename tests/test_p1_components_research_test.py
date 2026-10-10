"""仅使用 synthetic/disposable fixtures 验证 P1 test 边界及完整汇总。"""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts import eval_p1_components as entry


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
        entry.evaluate(tmp_path, "M0_seed20260811", "cuda:0")
    with pytest.raises(PermissionError):
        entry.summarize(tmp_path)
    with pytest.raises(PermissionError):
        entry.BandTestReader(frequencies(), "M0")
    with pytest.raises(ValueError, match="NoTF"):
        entry.BandTestReader(None, "M1", confirmed=True)


@pytest.mark.parametrize("arm", ["M0", "M2", "M3", "S1", "S2"])
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
    assert entry.checkpoint_frequencies({}, "M1", full) is None
    with pytest.raises(ValueError, match="NoTF"):
        entry.checkpoint_frequencies({"condition.frequencies_hz": torch.tensor(full)}, "M1", full)
    for arm in entry.ARMS:
        if arm == "M1":
            continue
        f = full[entry.band_indices(full, entry.ARMS[arm].band)]
        state = {"condition.frequencies_hz": torch.tensor(f)}
        assert np.array_equal(entry.checkpoint_frequencies(state, arm, full), f)
        drift = full.copy()
        drift[10 if arm == "S1" else 60] += .001
        with pytest.raises(ValueError, match="频率"):
            entry.checkpoint_frequencies(state, arm, drift)


def synthetic_batch(arm="M0"):
    t = torch.arange(18000).float() / 100
    signal = torch.stack([(1 + .2 * torch.sin(2 * torch.pi * .02 * t)) *
        torch.sin(2 * torch.pi * f * t) for f in (.2, .25)])[:, None]
    rows = pd.DataFrame({"dataset_row_id": [10, 20], "samp_id": [101, 102], "split": ["test", "test"]})
    batch = {"x": signal.clone(), "target": signal,
        "meta": {"dataset_row_id": torch.tensor([10, 20]), "samp_id": torch.tensor([101, 102]), "split": ["test", "test"]}}
    if arm != "M1":
        scales = len(entry.band_indices(frequencies(), entry.ARMS[arm].band))
        batch["tf"] = {"w": torch.ones(2, scales, 360)}
    return batch, rows


@pytest.mark.parametrize("arm", ["M0", "M1", "M2", "S1"])
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
        batch["tf"] = {"w": torch.ones(2, 97, 360)} if arm == "M1" else {}
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
    for arm in ("M0", "M1", "M2", "S1"):
        batch, rows = synthetic_batch(arm)
        cfg = entry.training.config(arm, entry.training.SEEDS[0], tmp_path)
        outputs[arm] = entry.infer_metrics(IdentityModel(), [batch], rows, cfg, "cpu")
    return outputs["M0"]


def test_native_metrics_and_all_matrix_comparisons(native_metrics):
    metrics, summary = native_metrics
    assert len(metrics) == 2 and summary.iloc[0].n_samples == 2
    for key in ("ibi_coverage", "respiratory_band_coherence", "constrained_ndtw",
                "target_stratified_envelope_spearman", "joint_prediction_degenerate"):
        assert key in metrics
    frames = [metrics.assign(arm=c["arm"], seed=c["seed"]) for c in entry.training.plan()]
    with pytest.raises(ValueError, match="完整"):
        entry.summary_tables(pd.concat(frames[:-1], ignore_index=True))
    table = pd.concat(frames, ignore_index=True)
    result = entry.summary_tables(table)
    assert len(result["per_seed"]) == 150 and len(result["across_seed"]) == 50
    assert len(result["paired_delta"]) == 135 and len(result["native_summary"]) == 30
    assert result["denominators"].rows.eq(2).all()
    assert len(result["representation_attention_interaction"]) == 15
    assert result["representation_attention_interaction"].interaction.eq(0).all()
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


def test_shards_cover_all_thirty_once():
    shards = [entry.training.plan(i, 2) for i in range(2)]
    assert [len(s) for s in shards] == [15, 15]
    assert len({c["cell"] for s in shards for c in s}) == 30


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
    lock = {"protocol": entry.PROTOCOL, "contract": entry.test_contract(),
        "training_session_sha256": entry.TRAINING_SESSION_SHA256,
        "validation_receipt_sha256": entry.VALIDATION_RECEIPT_SHA256,
        "entries": entry.training.plan(), "code_sha256": entry.code_identity(),
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
