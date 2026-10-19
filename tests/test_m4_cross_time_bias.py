"""仅用 synthetic/disposable fixtures 验证 Δt 消融、初始化、来源和配对汇总。"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from scripts import m4_cross_time_bias_model as mod
from scripts import m4_cross_time_bias_runtime as run
from scripts.p1_components_model import P1ComponentsModel
from resp_train.crd.training import build_crd_optimizer
from test_p1_components import FixtureMamba, frequencies, cpu_only
from test_m4_components import make_cell


def small_architecture():
    return replace(mod.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)


def model(arm, seed=run.SEEDS[0]):
    return mod.CrossTimeBiasModel(arm, seed, frequencies(), cfg=small_architecture(), mamba_factory=FixtureMamba)


def test_scientific_contract_and_fixed_matrix(tmp_path):
    assert len(run.plan()) == 3 and len(run.plan(include_references=True)) == 6
    assert {c["arm"] for c in run.plan()} == {"A1"}
    assert len({c["cell"] for i in range(3) for c in run.plan(i, 3)}) == 3
    for cell in run.plan(include_references=True):
        cfg = run.config(cell["arm"], cell["seed"], tmp_path)
        old = run.p1.config("M4", cell["seed"], tmp_path)
        for section in ("data", "window", "training", "loss", "evaluation"):
            assert cfg[section] == old[section]
        assert cfg.model.architecture == old.model.architecture
        assert dict(cfg.model.m4_cross_time_bias) == mod.contract(cell["arm"])
    with pytest.raises(ValueError, match="冻结参照"):
        run.run_cell(tmp_path, "A0", run.SEEDS[0], "cpu")


@pytest.mark.parametrize("seed", run.SEEDS)
def test_only_time_bias_removed_and_rng_dispatch_unchanged(seed, monkeypatch):
    calls = []
    original_seed = torch.manual_seed
    def record(value):
        calls.append(value)
        return original_seed(value)
    monkeypatch.setattr(torch, "manual_seed", record)
    old = P1ComponentsModel("M4", seed, frequencies(), cfg=small_architecture(), mamba_factory=FixtureMamba)
    expected_calls = list(calls)
    calls.clear()
    state = torch.get_rng_state().clone()
    candidate = model("A1", seed)
    assert calls == expected_calls and torch.equal(state, torch.get_rng_state())
    left, right = old.state_dict(), candidate.state_dict()
    removed = {name for name in left if name.startswith("condition.time_bias.")}
    assert set(left) - set(right) == removed and not (set(right) - set(left))
    for name, value in right.items():
        torch.testing.assert_close(value, left[name], atol=0, rtol=0)
    assert sum(p.numel() for p in old.parameters()) - sum(p.numel() for p in candidate.parameters()) == 100
    report = run.audit_initialization(cfg=small_architecture(), mamba_factory=FixtureMamba)
    assert all(r["common_exact"] and r["original_m4_exact"] for r in report.values())


def test_zero_time_bias_equivalence_and_retained_frequency_geometry():
    full, candidate = model("A0"), model("A1")
    with torch.no_grad():
        full.condition.time_bias[-1].weight.zero_()
        full.condition.time_bias[-1].bias.zero_()
        tokens, condition = torch.randn(1, 359, 32), torch.rand(1, 97, 360)
        torch.testing.assert_close(full.condition(tokens, condition), candidate.condition(tokens, condition), atol=0, rtol=0)
        torch.testing.assert_close(candidate.condition.relative_seconds, torch.tensor([-.25, .25]), atol=0, rtol=0)
    assert not list(candidate.condition.time_bias.parameters())
    assert sum(p.numel() for p in candidate.condition.frequency_bias.parameters()) == 100
    assert candidate.condition.output.weight.shape == (32, 32)
    assert candidate.fusion.projection.weight.shape == (32, 32)


@pytest.mark.parametrize("bf16", [False, True])
def test_complete_gradients_and_optimizer_excludes_removed_parameters(bf16, tmp_path):
    net = model("A1")
    with torch.no_grad():
        net.fusion.projection.weight.normal_(std=.02)
    x, tf = torch.randn(1, 1, 18000), {"w": torch.rand(1, 97, 360, requires_grad=True)}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        output = net(x, tf=tf)["waveform"]
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert tf["w"].grad.abs().sum() > 0
    assert net.condition.frequency_bias[0].weight.grad.abs().sum() > 0
    _, partition = build_crd_optimizer(net, run.config("A1", run.SEEDS[0], tmp_path))
    names = partition.decay_names + partition.no_decay_names
    assert not any(name.startswith("condition.time_bias.") for name in names)
    assert any(name.startswith("condition.frequency_bias.") for name in names)


def test_two_update_official_task_loss_fixture(monkeypatch, tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    run.write_json(session / "session.json", {"synthetic": True})
    monkeypatch.setattr(run, "load_session", lambda _: {})
    monkeypatch.setattr(run, "architecture", small_architecture)
    monkeypatch.setattr(run, "cwt_magnitude_features", lambda _: (np.random.default_rng(42).random((97, 360), dtype=np.float32), frequencies()))
    result = run.smoke(session, "A1", "cpu", mamba_factory=FixtureMamba, batch_size=1)
    receipt = run.read_json(result / "receipt.json")
    assert receipt["passed"] and receipt["updates"] == 2 and receipt["accumulation"] == 4


def metrics_fixture():
    rows = []
    for cell in run.plan(include_references=True):
        for row_id, subject in ((10, 101), (20, 102)):
            error = 1. if cell["arm"] == "A0" else 1.02
            row = {"arm": cell["arm"], "seed": cell["seed"], "dataset_row_id": row_id, "samp_id": subject,
                "split": "val", "whole_rr_target_eligible": True, "local_rr_target_eligible": True,
                "local_rr_target_eligible_windows": 9, "joint_target_eligible": True,
                "envelope_spearman_target_eligible": True, "joint_prediction_degenerate": False,
                "envelope_spearman_prediction_degenerate": False}
            row.update({m: .9 if m == run.sf.PCC else error for m in run.sf.PRIMARY})
            rows.append(row)
    return pd.DataFrame(rows)


def test_complete_matrix_pairing_threshold_and_identity():
    frame = metrics_fixture()
    tables = run.summary_tables(frame)
    assert len(tables["per_seed"]) == 30 and len(tables["paired_delta"]) == 15
    assert len(tables["subject_macro_paired_delta"]) == 15
    threshold = tables["threshold_summary"]
    assert threshold.loc[threshold.metric != run.sf.PCC, "mean_classification"].eq("worsened").all()
    assert threshold.loc[threshold.metric == run.sf.PCC, "mean_classification"].eq("no_obvious_mean_difference").all()
    with pytest.raises(ValueError, match="完整"):
        run.summary_tables(frame.iloc[:-2])
    frame.loc[len(frame)-1, "dataset_row_id"] = 999
    with pytest.raises(ValueError):
        run.summary_tables(frame)


@pytest.mark.parametrize("improvement,expected", [(.009, "no_obvious_mean_difference"), (.01, "improved"),
    (-.009, "no_obvious_mean_difference"), (-.01, "worsened")])
def test_threshold_boundary(improvement, expected):
    frame = pd.DataFrame({"metric": [run.sf.PCC]*3, "delta": [improvement]*3, "improvement": [improvement]*3})
    table = run.threshold_summary(frame)
    assert table.iloc[0].mean_classification == expected


@pytest.fixture
def reference_fixture(tmp_path, monkeypatch):
    source = tmp_path / "p1"
    source.mkdir()
    run.write_json(source / "session.json", {"synthetic": True})
    monkeypatch.setattr(run, "REFERENCE_SESSION", source)
    monkeypatch.setattr(run, "REFERENCE_SHA", run.sha(source / "session.json"))
    value = OmegaConf.load(run.SPEC)
    value.reference_session_sha256 = run.REFERENCE_SHA
    path = tmp_path / "spec.yaml"
    OmegaConf.save(value, path)
    monkeypatch.setattr(run, "SPEC", path)
    monkeypatch.setattr(run, "verify_reference_session", lambda _: {})
    code = "scripts/m4_cross_time_bias_model.py"
    monkeypatch.setattr(run, "code_identity", lambda: {code: run.sha(run.ROOT / code)})
    original_audit = run.audit_initialization
    monkeypatch.setattr(run, "audit_initialization", lambda: original_audit(cfg=small_architecture(), mamba_factory=FixtureMamba))
    frame = metrics_fixture()
    references = {}
    for seed in run.SEEDS:
        directory = source / "cells" / f"M4_seed{seed}"
        rows = frame.query("arm == 'A0' and seed == @seed").assign(arm="M4")
        make_cell(directory, "M4", seed, rows, run.p1.config("M4", seed, directory), 1056357)
        references[f"B0_seed{seed}"] = {"completed_sha256": run.sha(directory / "completed.json"),
            "environment_sha256": run.sha(directory / "environment.json")}
    component = tmp_path / "components.json"
    run.write_json(component, {"references": {"cells": references}})
    monkeypatch.setattr(run, "REFERENCE_COMPONENTS_SESSION", component)
    monkeypatch.setattr(run, "REFERENCE_COMPONENTS_SHA", run.sha(component))
    return source


def test_reference_hash_tamper_rejected(reference_fixture):
    refs = run.audit_references()
    assert len(refs["cells"]) == 3
    checkpoint = reference_fixture / "cells" / f"M4_seed{run.SEEDS[0]}" / "checkpoint.pt"
    checkpoint.write_bytes(b"changed disposable fixture")
    with pytest.raises(ValueError, match="身份变化"):
        run.audit_references()


def test_reference_output_alias_requires_same_cell(reference_fixture, tmp_path, monkeypatch):
    seed = run.SEEDS[0]
    directory = reference_fixture / "cells" / f"M4_seed{seed}"
    alias = tmp_path / "legacy_cell"
    alias.symlink_to(directory, target_is_directory=True)
    cfg = OmegaConf.load(directory / "config.yaml")
    cfg.outputs.run_root = str(alias)
    OmegaConf.save(cfg, directory / "config.yaml")
    receipt = run.read_json(directory / "completed.json")
    receipt["artifacts"]["config.yaml"] = run.sha(directory / "config.yaml")
    # 一次性 fixture 重写以模拟历史记录中的兼容路径。
    (directory / "completed.json").write_text(__import__("json").dumps(receipt))
    source = run.read_json(run.REFERENCE_COMPONENTS_SESSION)
    source["references"]["cells"][f"B0_seed{seed}"]["completed_sha256"] = run.sha(directory / "completed.json")
    run.REFERENCE_COMPONENTS_SESSION.write_text(__import__("json").dumps(source))
    monkeypatch.setattr(run, "REFERENCE_COMPONENTS_SHA", run.sha(run.REFERENCE_COMPONENTS_SESSION))
    assert len(run.audit_references()["cells"]) == 3
    alias.unlink()
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="输出路径"):
        run.audit_references()


def test_prepare_summary_and_source_preservation(reference_fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "build_tho_data", lambda _: pytest.fail("不得读取真实数据"))
    before = {str(p): run.sha(p) for p in reference_fixture.rglob("*") if p.is_file()}
    session = run.prepare(tmp_path / "new")
    assert run.prepare(session) == session
    assert len(list((session / "resolved_configs").glob("*.yaml"))) == 3
    assert sum(c["status"] == "reference_verified" for c in run.status(session)) == 3
    with pytest.raises(RuntimeError, match="尚未全部完成"):
        run.summarize(session)
    frame = metrics_fixture()
    for c in run.plan():
        directory = session / "cells" / c["cell"]
        rows = frame[(frame.arm == c["arm"]) & (frame.seed == c["seed"])]
        make_cell(directory, c["arm"], c["seed"], rows, run.config(c["arm"], c["seed"], directory))
    manifest = run.load_session(session)
    monkeypatch.setattr(run, "load_session", lambda _: manifest)
    original_spec = run.spec
    def small_spec():
        result = original_spec()
        result["data"] = {**result["data"], "val_windows": 2, "val_subjects": 2}
        return result
    monkeypatch.setattr(run, "spec", small_spec)
    monkeypatch.setattr(run, "summarize_task_metrics", lambda group: pd.DataFrame({"n_samples": [len(group)]}))
    result = run.summarize(session)
    assert run.summarize(session) == result
    assert len(pd.read_csv(result / "per_seed.csv")) == 30
    assert len(list(result.glob("*.png"))) == 2 and (result / "report.md").is_file()
    assert before == {str(p): run.sha(p) for p in reference_fixture.rglob("*") if p.is_file()}
    receipt = run.read_json(result / "receipt.json")
    for name, digest in receipt["artifacts"].items():
        assert run.sha(result / name) == digest


def test_environment_drift_rejected_before_registration(reference_fixture, tmp_path):
    session = run.prepare(tmp_path / "new")
    run.register_training_environment(session, {"torch": "fixture", "cuda": None, "packages": {}, "gpu": None})
    with pytest.raises(ValueError, match="环境"):
        run.register_training_environment(session, {"torch": "changed", "cuda": None, "packages": {}, "gpu": None})


def test_historical_sources_checked_with_scoped_later_additions(monkeypatch, tmp_path):
    (tmp_path / "source_snapshot.tar.gz").write_bytes(b"fixture")
    saved = {"protocol": run.p1.PROTOCOL, "spec": run.p1.spec(), "cells": run.p1.plan(),
        "arms": {a: run.p1.contract(a) for a in run.p1.ARMS}, "code_sha256": {"old.py": "old"},
        "snapshot_sha256": run.sha(tmp_path / "source_snapshot.tar.gz")}
    run.write_json(tmp_path / "session.json", saved)
    sources = {"old.py": "old", "resp_train/paper_evidence/h_only_ablation_v1/new.py": "new"}
    monkeypatch.setattr(run.p1, "code_identity", lambda: sources)
    assert run.verify_reference_session(tmp_path)["historical_source_files_verified"] == 1
    sources["old.py"] = "changed"
    with pytest.raises(ValueError, match="历史源码"):
        run.verify_reference_session(tmp_path)
    sources["old.py"] = "old"
    sources["unrecognized.py"] = "new"
    with pytest.raises(ValueError, match="未核对"):
        run.verify_reference_session(tmp_path)
