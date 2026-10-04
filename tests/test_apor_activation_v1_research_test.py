from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from scripts import eval_apor_activation_v1_research_test as rt
from resp_train.paper_evidence import w0_structural_factorial_v1_test as legacy


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def fixture_lock():
    return {"protocol": rt.PROTOCOL, "entries": rt.training.plan(), "split": "test", "count": 2310,
            "samp_ids": 8, "row_order_sha256": rt.ROW_HASH, "sample_seed": 20260612, "batch_size": 128,
            "amp_dtype": "bfloat16", "include_test_only": True, "test_array_read": False,
            "references": [{"arm": a,"seed": s} for a in rt.REFERENCE_ARMS for s in rt.SEEDS],
            "views": {k: list(v) for k,v in rt.VIEW_COUNTS.items()}}


def test_fixed_matrix_and_selector():
    lock = fixture_lock()
    rt.validate_contract(lock)
    assert len(lock["entries"]) == 3
    lock["entries"] = lock["entries"][:-1]
    with pytest.raises(ValueError):
        rt.validate_contract(lock)
    history = pd.DataFrame({"epoch": [1, 2, 3], "val_local_rr_mae": [.4, .2, .2]})
    assert rt.selected_epoch(history, {"best_epoch": 2, "completed_epochs": 3}) == 2
    with pytest.raises(ValueError):
        rt.selected_epoch(history, {"best_epoch": 3, "completed_epochs": 3})
    with pytest.raises(ValueError):
        rt.entry_for(fixture_lock(), "U0", rt.SEEDS[0])


@pytest.mark.parametrize("arm", ["U1"])
def test_evaluation_config_keeps_training_contract(tmp_path, arm):
    cfg = rt.training.config(arm, rt.SEEDS[0], tmp_path)
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)
    entry = {"arm": arm, "seed": rt.SEEDS[0], "training_config": OmegaConf.to_container(cfg, resolve=True),
             "config": {"path": str(config_path), **rt.sf.identity(config_path)}}
    eval_cfg, data_cfg = rt.evaluation_config(entry, "cpu")
    assert eval_cfg.loss == cfg.loss and eval_cfg.evaluation == cfg.evaluation
    assert OmegaConf.to_container(eval_cfg.model) == OmegaConf.to_container(cfg.model)
    if arm == "M1":
        assert data_cfg.data.tf_cache_path is None and not data_cfg.model.tf_representations
        assert data_cfg.data.get("tf_research_test_cache_path") is None
    else:
        assert data_cfg.data.tf_research_test_cache_path == str(rt.FROZEN_RESEARCH_TEST_CACHE_ROOT)


def synthetic_rows(batch):
    return pd.DataFrame({key: batch["meta"][key] for key in rt.IDENTITY_COLUMNS})


@pytest.mark.parametrize("condition", [True, False])
def test_batch_guards_cover_both_input_paths(condition):
    batch = rt.training.synthetic_batch(2, 91, split="test")
    rows = synthetic_rows(batch)
    if not condition:
        batch.pop("tf")
    assert len(list(rt.guarded_batches([batch], rows, condition=condition))) == 1
    with pytest.raises(ValueError):
        list(rt.guarded_batches([batch], rows.iloc[::-1], condition=condition))
    with pytest.raises(ValueError):
        list(rt.guarded_batches([], rows, condition=condition))
    batch["target"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        list(rt.guarded_batches([batch], rows, condition=condition))


def test_batch_rejects_wrong_view():
    batch = rt.training.synthetic_batch(2, 91, split="test")
    with pytest.raises(ValueError):
        list(rt.guarded_batches([batch], synthetic_rows(batch), condition=False))


def patch_row_contract(monkeypatch, ids):
    monkeypatch.setattr(legacy, "COUNT", len(ids))
    monkeypatch.setattr(legacy, "SUBJECTS", 2)
    monkeypatch.setattr(legacy, "ROW_ORDER_SHA256", rt.training.array_hash(np.asarray(ids, dtype=np.int64)))


@pytest.mark.parametrize("arm", ["U1"])
def test_synthetic_model_to_native_test_metrics(arm, tmp_path, monkeypatch):
    cfg = rt.training.config(arm, rt.SEEDS[0], tmp_path)
    model = rt.build_model(cfg)
    # 原生Mamba由已完成GPU验收覆盖；此fixture只验证评价数据流与新入口。
    model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in model.base.local_blocks])
    batch = rt.training.synthetic_batch(2, 71, split="test", row_offset=100)
    condition = rt.ARM_SPECS[arm].condition
    if not condition:
        batch.pop("tf")
    rows = synthetic_rows(batch)
    patch_row_contract(monkeypatch, rows.dataset_row_id.to_numpy())
    predictions = rt.collect_predictions(model, rt.guarded_batches([batch], rows, condition=condition),
                                          device="cpu", max_windows=2, use_amp=False)
    metrics = rt.evaluate_task_predictions(predictions, cfg, include_test_only=True, method=arm)
    summary = rt.check_test_metrics(metrics, rows)
    assert len(metrics) == 2 and summary.iloc[0].n_samples == 2
    assert all(column in metrics for column in [*rt.sf.PRIMARY, "constrained_ndtw", "ibi_coverage"])


def metric_fixture():
    rows = []
    for cell in rt.training.scientific_plan():
        for index in range(4):
            row = {**cell, "dataset_row_id": index, "samp_id": 670 if index % 2 == 0 else 1, "split": "test",
                   "joint_prediction_degenerate": False, "envelope_spearman_prediction_degenerate": False,
                   "respiratory_band_coherence": .8, "constrained_ndtw": .2}
            for key in rt.TARGET_COLUMNS:
                row[key] = 1 if key == "local_rr_target_eligible_windows" else True
            row.update({metric: (.8 if metric == rt.sf.PCC else {"U0": 1., "U1": .9}[cell["arm"]])
                        for metric in rt.sf.PRIMARY})
            rows.append(row)
    return pd.DataFrame(rows)


def test_complete_summary_pairing_and_no_sample_filtering(monkeypatch):
    patch_row_contract(monkeypatch, range(4))
    monkeypatch.setattr(rt, "VIEW_COUNTS", {"full": (4, 2), "exclude670": (2, 1), "subject670": (2, 1)})
    frame = metric_fixture()
    tables = rt.summary_tables(frame)
    assert len(tables["per_seed"]) == 90
    assert len(tables["across_seed"]) == 30
    assert len(tables["paired_delta"]) == 45
    assert set(tables["per_seed"].view) == {"full", "exclude670", "subject670"}
    assert tables["per_seed"].loc[lambda x: x.view.eq("exclude670"), "n"].eq(2).all()
    assert np.allclose(tables["paired_delta"].loc[lambda x: x.metric.ne(rt.sf.PCC), "delta"], -.1)
    with pytest.raises(ValueError):
        rt.summary_tables(frame.loc[~frame.arm.eq("U1")])
    bad = frame.copy()
    bad.loc[0, "local_rr_mae_bpm"] = np.nan
    with pytest.raises(FloatingPointError):
        rt.summary_tables(bad)


def test_evidence_preserves_failure_and_reuses_success(tmp_path):
    rt.write_json(tmp_path / "allowlist.json", {"fixture": True})
    parent = tmp_path / "evaluation/M0/seed_1"
    with pytest.raises(RuntimeError):
        with rt.evidence_attempt(tmp_path, parent, "evaluation", arm="M0", seed=1):
            raise RuntimeError("fixture")
    with pytest.raises(RuntimeError, match="retry-failed"):
        rt.completed(tmp_path, parent, "evaluation", arm="M0", seed=1)
    with rt.evidence_attempt(tmp_path, parent, "evaluation", arm="M0", seed=1) as output:
        rt.write_json(output / "metrics.json", {"value": .2})
    assert rt.completed(tmp_path, parent, "evaluation", arm="M0", seed=1) == output
    (output / "metrics.json").write_text("tampered")
    with pytest.raises(RuntimeError):
        rt.completed(tmp_path, parent, "evaluation", arm="M0", seed=1)


@pytest.mark.parametrize("phase", ["parallel", "evaluate", "summary"])
def test_cli_requires_current_test_confirmation(tmp_path, phase):
    command = [sys.executable, str(Path(rt.__file__)), phase, "--allowlist", str(tmp_path)]
    if phase == "evaluate":
        command += ["--arm", "U1", "--seed", str(rt.SEEDS[0])]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "--confirm-research-test" in result.stderr


@pytest.mark.parametrize("exit_codes", [(0, 0), (0, 1)])
def test_parallel_test_controller_freezes_partition_and_gates_summary(tmp_path, monkeypatch, exit_codes):
    rt.write_json(tmp_path / "allowlist.json", {"fixture": True})
    monkeypatch.setattr(rt, "load_allowlist", lambda path: {})
    monkeypatch.setattr(rt.training, "cuda_environment", lambda device: {"device_uuid": device})
    calls, summaries = [], []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.returncode = exit_codes[len(calls)]
            self.pid = 100 + len(calls)
            calls.append(command)

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(rt.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(rt, "summarize", lambda *args: summaries.append(args) or tmp_path / "summary")
    if exit_codes == (0, 0):
        rt.parallel(tmp_path, ["cuda:0", "cuda:1"])
        assert len(summaries) == 1
    else:
        with pytest.raises(RuntimeError, match="worker失败"):
            rt.parallel(tmp_path, ["cuda:0", "cuda:1"])
        assert summaries == []
    layout = rt.read_json(tmp_path / "parallel_plan.json")
    assert layout["protocol"] == rt.PROTOCOL
    assert [len(worker["cells"]) for worker in layout["workers"]] == [2, 1]
    assert all("--confirm-research-test" in command for command in calls)
