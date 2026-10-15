"""research-test 入口的人工选点、访问门禁、频率与推理检查。"""
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from scripts import eval_bcg_cwt_patch_mamba as evaluate
from scripts.check_bcg_cwt_patch_mamba import synthetic_config, SyntheticMamba
from resp_train.models.bcg_cwt_patch_mamba import build_bcg_cwt_patch_mamba


@pytest.fixture(autouse=True)
def cpu_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def test_test_access_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate.cache, "TfV1ResearchTestCacheReader", lambda *a, **k: pytest.fail("未授权时不得读取缓存"))
    with pytest.raises(PermissionError): evaluate.WTestReader(np.ones(97))
    with pytest.raises(PermissionError): evaluate.evaluate(tmp_path, "cuda:0")


def test_selection_uses_complete_validation_history_and_earliest_tie(tmp_path):
    directories = {}
    for epoch in range(1, 4):
        directory = tmp_path / str(epoch)
        directory.mkdir()
        (directory / "checkpoint.pt").write_bytes(str(epoch).encode())
        directories[epoch] = directory
    records = [{"epoch": i, "val_local_rr_mae": rr, "validation": {"loss": loss}}
               for i, rr, loss in ((1, .5, .4), (2, .6, .3), (3, .5, .3))]
    completion = {"epochs": 3, "best_epoch": 1, "best_local_rr_mae": .5, "best_loss_epoch": 2, "best_val_loss": .3}
    entries = evaluate.select_entries(records, completion, directories)
    assert [(e["role"], e["epoch"]) for e in entries] == [("best_rr", 1), ("best_loss", 2)]
    completion["best_epoch"] = 3
    with pytest.raises(ValueError, match="轨迹"):
        evaluate.select_entries(records, completion, directories)
    with pytest.raises(ValueError, match="不完整"):
        evaluate.select_entries(records[:-1], completion, directories)
    records[0]["val_local_rr_mae"] = np.nan
    with pytest.raises(FloatingPointError): evaluate.select_entries(records, completion, directories)


def test_full_cwt_reader_preserves_97_rows_and_rejects_frequency_drift(monkeypatch):
    frequencies = np.linspace(.03, 8, 97, dtype=np.float64)
    value = torch.arange(97 * 360).reshape(97, 360).float()
    source = SimpleNamespace(manifest={"fixed_transform_spec": evaluate.fixed_transform_spec()},
        arrays={"w": SimpleNamespace(shape=(2310, 97, 360), dtype=np.dtype("float32"))},
        _open=lambda name: frequencies, get=lambda row: {"w": value.clone()})
    monkeypatch.setattr(evaluate.cache, "TfV1ResearchTestCacheReader", lambda *a, **k: source)
    reader = evaluate.WTestReader(frequencies, confirmed=True)
    torch.testing.assert_close(reader.get(1)["w"], value)
    with pytest.raises(ValueError, match="频率"):
        evaluate.WTestReader(frequencies * 1.01, confirmed=True)
    value[0, 0] = float("nan")
    with pytest.raises(FloatingPointError): reader.get(1)


def test_synthetic_inference_metrics_and_row_guard(monkeypatch):
    from omegaconf import OmegaConf
    from resp_train.paper_evidence import w0_structural_factorial_v1_test as checks
    cfg = synthetic_config()
    cfg.evaluation = OmegaConf.load(evaluate.ROOT / "configs/crd_tf_v1/crd_tf101_m_smoke.yaml").evaluation
    model = build_bcg_cwt_patch_mamba(cfg, mamba_factory=SyntheticMamba)
    rows = pd.DataFrame({"dataset_row_id": [11, 12], "samp_id": [71, 72], "split": ["test", "test"]})
    monkeypatch.setattr(checks, "COUNT", 2)
    monkeypatch.setattr(checks, "SUBJECTS", 2)
    monkeypatch.setattr(checks, "ROW_ORDER_SHA256", hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest())
    t = torch.arange(18000) / 100
    target = ((1 + .3 * torch.sin(2 * torch.pi * .015 * t)) * torch.sin(2 * torch.pi * .25 * t))[None]
    base = [{"x": torch.randn(1, 18000), "target": target,
             "meta": row} for row in rows.to_dict("records")]
    reader = SimpleNamespace(get=lambda row: {"w": torch.rand(97, 360)})
    dataset = evaluate.WTestDataset(base, reader)
    predictions, metrics, summary = evaluate.infer(model, DataLoader(dataset, batch_size=2), rows, cfg, "cpu")
    assert predictions["r_tho_hat"].shape == (2, 1, 18000)
    assert metrics.dataset_row_id.tolist() == [11, 12]
    assert "constrained_ndtw" in metrics and "respiratory_band_coherence" in metrics
    assert np.isfinite(summary["local_rr_mae_bpm_mean"].iloc[0])
    bad = rows.copy()
    bad.loc[0, "dataset_row_id"] = 13
    with pytest.raises(ValueError, match="identity/order"):
        evaluate.infer(model, DataLoader(dataset, batch_size=2), bad, cfg, "cpu")
