"""固定12选点的research-test矩阵、loss最早选点及完整配对汇总。"""
import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from scripts import eval_m4_dual_selector as e


def test_matrix_and_confirmation_before_access(tmp_path, monkeypatch):
    assert len(e.plan()) == 12 and len({c["cell"] for c in e.plan()}) == 12
    assert {c["selector"] for c in e.plan(0, 2)} == {"rr"}
    assert {c["selector"] for c in e.plan(1, 2)} == {"loss"}
    monkeypatch.setattr(e, "load_allowlist", lambda path: pytest.fail("尚未确认不得读取来源"))
    with pytest.raises(PermissionError):
        e.evaluate(tmp_path, "M4_seed20260811_loss", "cpu")
    with pytest.raises(PermissionError):
        e.summarize(tmp_path)


def test_selectors_are_separate_earliest_minima():
    history = [{"epoch": i + 1, "val_local_rr_mae": rr, "validation": {"loss": loss}}
        for i, (rr, loss) in enumerate([(1., 4.), (2., 3.), (3., 3.)])]
    completion = {"completed_epochs": 3, "best_epoch": 1, "best_local_rr": 1.,
        "best_loss_epoch": 2, "best_validation_loss": 3.}
    assert e.selector_epoch(history, completion, "rr") == 1
    assert e.selector_epoch(history, completion, "loss") == 2
    with pytest.raises(ValueError):
        e.selector_epoch(history, {**completion, "best_loss_epoch": 3}, "loss")
    history[1]["validation"]["loss"] = float("nan")
    with pytest.raises(ValueError):
        e.selector_epoch(history, completion, "loss")


def test_native_metrics_twelve_cell_summary(tmp_path, monkeypatch):
    from resp_train.paper_evidence import w0_structural_factorial_v1_test as checks
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rows = pd.DataFrame({"dataset_row_id": [10, 20], "samp_id": [101, 102], "split": ["test", "test"]})
    monkeypatch.setattr(checks, "COUNT", 2)
    monkeypatch.setattr(checks, "SUBJECTS", 2)
    monkeypatch.setattr(checks, "ROW_ORDER_SHA256", hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest())
    t = torch.arange(18000).float() / 100
    signal = torch.stack([(1 + .2 * torch.sin(2 * torch.pi * .02 * t)) * torch.sin(2 * torch.pi * f * t)
        for f in (.2, .25)])[:, None]
    batch = {"x": signal, "target": signal, "tf": {"w": torch.ones(2, 97, 360)},
        "meta": {"dataset_row_id": torch.tensor([10, 20]), "samp_id": torch.tensor([101, 102]), "split": ["test", "test"]}}
    class Identity(nn.Module):
        def forward(self, x, **kwargs):
            return {"waveform": x}
    cfg = e.training.config("M4", e.training.SEEDS[0], tmp_path)
    metrics, _ = e.infer_metrics(Identity(), [batch], rows, cfg, "cpu")
    frames = []
    for cell in e.plan():
        frame = metrics.assign(arm=cell["arm"], seed=cell["seed"], selector=cell["selector"])
        frame["local_rr_mae_bpm"] += .1 if cell["selector"] == "loss" else .2
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    result = e.summary_tables(table)
    assert len(result["native_summary"]) == 12
    assert len(result["rr_per_seed"]) == len(result["loss_per_seed"]) == 30
    assert len(result["seed20260811_selector_delta"]) == 10
    delta = result["selector_paired_delta"]
    np.testing.assert_allclose(delta.loc[delta.metric == "local_rr_mae_bpm", "delta"], -.1)
    with pytest.raises(ValueError, match="完整"):
        e.summary_tables(pd.concat(frames[:-1], ignore_index=True))
    table.loc[len(table) - 1, "dataset_row_id"] = 99
    with pytest.raises(ValueError):
        e.summary_tables(table)


@pytest.mark.parametrize("change", ["selector", "source", "snapshot"])
def test_allowlist_drift_rejected(tmp_path, monkeypatch, change):
    monkeypatch.setattr(e, "code_identity", lambda: {"synthetic": "sha"})
    (tmp_path / "source_snapshot.tar.gz").write_bytes(b"synthetic")
    lock = {"protocol": e.PROTOCOL, "contract": e.test_contract(), "entries": e.plan(),
        "training_session_sha256": e.TRAINING_SESSION_SHA256,
        "validation_receipt_sha256": e.VALIDATION_RECEIPT_SHA256,
        "code_sha256": e.code_identity(), "snapshot_sha256": e.sha(tmp_path / "source_snapshot.tar.gz")}
    e.write_json(tmp_path / "allowlist.json", lock)
    e.write_json(tmp_path / "allowlist_receipt.json", {"sha256": e.sha(tmp_path / "allowlist.json")})
    assert e.load_allowlist(tmp_path) == lock
    if change == "selector":
        lock["entries"][0]["selector"] = "loss"
    elif change == "source":
        lock["code_sha256"] = {}
    else:
        lock["snapshot_sha256"] = "different"
    (tmp_path / "allowlist.json").write_text(__import__("json").dumps(lock))
    (tmp_path / "allowlist_receipt.json").write_text(__import__("json").dumps({"sha256": e.sha(tmp_path / "allowlist.json")}))
    with pytest.raises(ValueError):
        e.load_allowlist(tmp_path)
