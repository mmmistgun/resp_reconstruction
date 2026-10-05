"""research-test 实现的合成 CPU 检查，不读取实际 test 数据或缓存。"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("patch_tf_test_entry", ROOT / "scripts/eval_patch_aligned_tf_mamba.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def test_confirmation_checked_before_any_source_access(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "load_allowlist", lambda path: pytest.fail("不应读取 allowlist"))
    monkeypatch.setattr(entry.test_cache, "TfV1ResearchTestCacheReader", lambda *a, **k: pytest.fail("不应读取 cache"))
    with pytest.raises(PermissionError):
        entry.evaluate(tmp_path, "p1s_seed20260811", "cuda:0")
    with pytest.raises(PermissionError):
        entry.summarize(tmp_path)
    with pytest.raises(PermissionError):
        entry.HTestReader(np.ones(41))


def test_selector_uses_full_history_and_earliest_tie():
    history = [{"epoch": i + 1, "val_local_rr_mae": score} for i, score in enumerate([2., 1., 1., 1.5])]
    receipt = {"completed_epochs": 4, "best_epoch": 2, "best_local_rr": 1.}
    assert entry.selected_epoch(history, receipt) == 2
    with pytest.raises(ValueError):
        entry.selected_epoch(history, {**receipt, "best_epoch": 3})
    with pytest.raises(ValueError):
        entry.selected_epoch(history[:-1], receipt)
    history[0]["val_local_rr_mae"] = np.nan
    with pytest.raises(FloatingPointError):
        entry.selected_epoch(history, receipt)


def test_h_test_cache_selects_physical_frequency_and_rejects_drift(monkeypatch):
    frequencies = np.concatenate((np.linspace(.03, .8, 56), np.geomspace(.81, 8, 41)))
    values = np.arange(2 * 97 * 360, dtype=np.float32).reshape(2, 97, 360)
    class Cache:
        def __init__(self, *args, **kwargs):
            self.manifest = {"fixed_transform_spec": entry.fixed_transform_spec()}
            self.arrays = {"w": values}
        def _open(self, name):
            assert name == "w_frequencies_hz.npy"
            return frequencies
        def get(self, row_id):
            return {"w": torch.from_numpy(values[row_id].copy())}
    monkeypatch.setattr(entry, "COUNT", 2)
    monkeypatch.setattr(entry.test_cache, "TfV1ResearchTestCacheReader", Cache)
    reader = entry.HTestReader(frequencies[56:], confirmed=True)
    torch.testing.assert_close(reader.get(1)["w"], torch.from_numpy(values[1, 56:]))
    with pytest.raises(ValueError, match="频率"):
        entry.HTestReader(frequencies[56:] + .001, confirmed=True)
    values[0, 0, 0] = np.inf
    with pytest.raises(FloatingPointError):
        reader.get(0)


def synthetic_batch():
    t = torch.arange(18000).float() / 100
    signal = torch.stack([(1 + .2 * torch.sin(2 * torch.pi * .02 * t)) *
        torch.sin(2 * torch.pi * f * t) for f in (.2, .25)])[:, None]
    rows = pd.DataFrame({"dataset_row_id": [10, 20], "samp_id": [101, 102], "split": ["test", "test"]})
    batch = {"x": signal.clone(), "target": signal,
        "tf": {"w": torch.ones(2, 41, 360)}, "meta": {"dataset_row_id": torch.tensor([10, 20]),
            "samp_id": torch.tensor([101, 102]), "split": ["test", "test"]}}
    return batch, rows


@pytest.mark.parametrize("bad", ["order", "shape", "nan", "missing"])
def test_guarded_batches_fail_on_incomplete_or_invalid_input(bad):
    batch, rows = synthetic_batch()
    loader = [batch]
    if bad == "order":
        batch["meta"]["dataset_row_id"] = torch.tensor([20, 10])
    elif bad == "shape":
        batch["tf"]["w"] = torch.ones(2, 97, 360)
    elif bad == "nan":
        batch["target"][0, 0, 0] = float("nan")
    else:
        loader = []
    with pytest.raises((ValueError, FloatingPointError)):
        list(entry.guarded_batches(loader, rows))


@pytest.fixture
def native_metrics(monkeypatch, tmp_path):
    from resp_train.paper_evidence import w0_structural_factorial_v1_test as checks
    batch, rows = synthetic_batch()
    monkeypatch.setattr(checks, "COUNT", 2)
    monkeypatch.setattr(checks, "SUBJECTS", 2)
    monkeypatch.setattr(checks, "ROW_ORDER_SHA256", hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest())
    class IdentityModel(nn.Module):
        def forward(self, x, *, tf):
            assert tf["w"].shape == (len(x), 41, 360)
            return {"waveform": x}
    cfg = entry.training.config(2, entry.training.SEEDS[0], tmp_path)
    return entry.infer_metrics(IdentityModel(), [batch], rows, cfg, "cpu")


def test_native_inference_includes_test_auxiliary_metrics(native_metrics):
    metrics, summary = native_metrics
    assert len(metrics) == 2 and summary.iloc[0].n_samples == 2
    for name in ("ibi_coverage", "respiratory_band_coherence", "constrained_ndtw",
                 "target_stratified_envelope_spearman", "joint_prediction_degenerate"):
        assert name in metrics
    assert np.isfinite(metrics["local_rr_mae_bpm"]).all()
    assert metrics["lag_aware_signed_pcc"].min() > .99


def test_complete_matrix_summary_and_target_identity(native_metrics):
    metrics, _ = native_metrics
    frames = []
    for cell in entry.training.plan():
        frame = metrics.copy()
        frame["arm"] = f"p{cell['patch_seconds']}s"
        frame["seed"] = cell["seed"]
        frames.append(frame)
    with pytest.raises(ValueError, match="完整"):
        entry.summary_tables(pd.concat(frames[:-1], ignore_index=True))
    table = pd.concat(frames, ignore_index=True)
    results = entry.summary_tables(table)
    assert len(results["per_seed"]) == 45 and len(results["across_seed"]) == 15
    assert len(results["paired_delta"]) == 30
    assert set(results["paired_delta"].reference) == {"p2s"}
    assert len(results["native_summary"]) == 9
    assert results["denominators"].rows.eq(2).all()
    table.loc[len(table) - 1, "dataset_row_id"] = 30
    with pytest.raises(ValueError):
        entry.summary_tables(table)


def test_artifact_reuse_retry_and_tamper_detection(tmp_path):
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
    assert entry.completed(tmp_path, parent) == output
    (output / "metrics.csv").write_text("value\n2\n")
    with pytest.raises(ValueError, match="身份"):
        entry.completed(tmp_path, parent)


def test_missing_cell_blocks_summary_before_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "load_allowlist", lambda root: {"entries": entry.training.plan()})
    with pytest.raises(RuntimeError, match="尚未全部完成"):
        entry.summarize(tmp_path, confirmed=True)
    assert not list((tmp_path / "summary").glob("attempt_*"))


def test_relative_artifact_paths_are_confined(tmp_path):
    with pytest.raises(ValueError, match="越界"):
        entry.contained(tmp_path, "../outside.pt")
