"""合成 CPU 验证单投影、共有初始化、双 GPU 调度及冻结基线配对。"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from scripts import m4_linear_fusion_model as mod
from scripts import m4_linear_fusion_runtime as run
from scripts.m4_cross_time_bias_model import CrossTimeBiasModel
from resp_train.crd.training import build_crd_optimizer
from test_p1_components import FixtureMamba, frequencies, cpu_only
from test_m4_components import make_cell


def small_architecture():
    return replace(mod.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)


def model(arm, seed=run.SEEDS[0]):
    return mod.LinearFusionModel(arm, seed, frequencies(), cfg=small_architecture(), mamba_factory=FixtureMamba)


def test_contract_and_gpu_shards(tmp_path):
    assert len(run.plan()) == 3 and len(run.plan(include_references=True)) == 6
    assert [c["seed"] for c in run.plan(0, 2)] == [20260811, 20260813]
    assert [c["seed"] for c in run.plan(1, 2)] == [20260812]
    for cell in run.plan(include_references=True):
        cfg = run.config(cell["arm"], cell["seed"], tmp_path)
        old = run.base.config("A1", cell["seed"], tmp_path)
        for section in ("data", "window", "training", "loss", "evaluation"):
            assert cfg[section] == old[section]
        assert cfg.model.architecture == old.model.architecture
        assert dict(cfg.model.m4_linear_fusion) == mod.contract(cell["arm"])
    with pytest.raises(ValueError, match="冻结参照"):
        run.run_cell(tmp_path, "L0", run.SEEDS[0], "cpu")


@pytest.mark.parametrize("seed", run.SEEDS)
def test_common_initialization_rng_sequence_and_zero_residual(seed, monkeypatch):
    calls = []
    original_seed = torch.manual_seed
    def record(value):
        calls.append(value)
        return original_seed(value)
    monkeypatch.setattr(torch, "manual_seed", record)
    old = CrossTimeBiasModel("A1", seed, frequencies(), cfg=small_architecture(), mamba_factory=FixtureMamba)
    expected_calls = list(calls)
    calls.clear()
    before = torch.get_rng_state().clone()
    single = model("L1", seed)
    assert calls == expected_calls and torch.equal(before, torch.get_rng_state())
    original, state = old.state_dict(), single.state_dict()
    assert set(original) - set(state) == {"fusion.projection.weight", "fusion.projection.bias"}
    assert not set(state) - set(original)
    for name, value in state.items():
        if name in ("condition.output.weight", "condition.output.bias"):
            assert torch.count_nonzero(value) == 0
        else:
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    assert sum(p.numel() for p in old.parameters()) - sum(p.numel() for p in single.parameters()) == 32*32+32
    x, tf = torch.randn(1, 1, 18000), {"w": torch.rand(1, 97, 360)}
    old.eval(); single.eval()
    with torch.no_grad():
        torch.testing.assert_close(old(x, tf=tf)["waveform"], single(x, tf=tf)["waveform"], atol=0, rtol=0)
    audit = run.audit_initialization(cfg=small_architecture(), mamba_factory=FixtureMamba)
    assert all(r["common_exact_except_output"] and r["reference_exact"] for r in audit.values())


def test_composed_affine_matches_dual_projection_function():
    dual, single = model("L0").eval(), model("L1").eval()
    with torch.no_grad():
        dual.fusion.projection.weight.normal_(std=.02)
        dual.fusion.projection.bias.normal_(std=.01)
        single.condition.output.weight.copy_(dual.fusion.projection.weight @ dual.condition.output.weight)
        single.condition.output.bias.copy_(dual.fusion.projection.weight @ dual.condition.output.bias + dual.fusion.projection.bias)
        x, tf = torch.randn(1, 1, 18000), {"w": torch.rand(1, 97, 360)}
        torch.testing.assert_close(dual(x, tf=tf)["waveform"], single(x, tf=tf)["waveform"], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("bf16", [False, True])
def test_finite_gradients_and_optimizer_groups(bf16, tmp_path):
    net = model("L1")
    with torch.no_grad():
        net.condition.output.weight.normal_(std=.02)
    tf = {"w": torch.rand(1, 97, 360, requires_grad=True)}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        output = net(torch.randn(1, 1, 18000), tf=tf)["waveform"]
    assert output.shape == (1, 1, 18000) and torch.isfinite(output).all()
    output.square().mean().backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert tf["w"].grad.abs().sum() > 0 and net.condition.frequency_bias[0].weight.grad.abs().sum() > 0
    _, partition = build_crd_optimizer(net, run.config("L1", run.SEEDS[0], tmp_path))
    names = partition.decay_names + partition.no_decay_names
    assert not any(n.startswith(("fusion.", "condition.time_bias.")) for n in names)
    assert any(n.startswith("condition.output.") for n in names)


def test_two_update_task_loss_opens_single_projection(monkeypatch, tmp_path):
    session = tmp_path / "new"
    session.mkdir()
    run.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(run, "load_session", lambda _: {})
    monkeypatch.setattr(run, "architecture", small_architecture)
    monkeypatch.setattr(run, "cwt_magnitude_features", lambda _: (np.random.default_rng(42).random((97, 360), dtype=np.float32), frequencies()))
    output = run.smoke(session, "L1", "cpu", mamba_factory=FixtureMamba, batch_size=1)
    receipt = run.read_json(output / "receipt.json")
    assert receipt["passed"] and receipt["updates"] == 2 and receipt["accumulation"] == 4


def metrics_fixture():
    records = []
    for c in run.plan(include_references=True):
        for row_id, subject in ((10, 101), (20, 102)):
            error = 1. if c["arm"] == "L0" else 1.005
            row = {"arm": c["arm"], "seed": c["seed"], "dataset_row_id": row_id, "samp_id": subject,
                "split": "val", "whole_rr_target_eligible": True, "local_rr_target_eligible": True,
                "local_rr_target_eligible_windows": 9, "joint_target_eligible": True,
                "envelope_spearman_target_eligible": True, "joint_prediction_degenerate": False,
                "envelope_spearman_prediction_degenerate": False}
            row.update({m: .9 if m == run.sf.PCC else error for m in run.sf.PRIMARY})
            records.append(row)
    return pd.DataFrame(records)


def test_summary_pairing_and_window_weighted_decision(tmp_path):
    frame = metrics_fixture()
    tables = run.summary_tables(frame)
    assert len(tables["paired_delta"]) == 15 and len(tables["subject_macro_paired_delta"]) == 15
    assert tables["threshold_summary"].mean_classification.eq("no_obvious_mean_difference").all()
    # 宏平均为诊断口径，结构选择的预设入口是窗口加权表。
    tables["subject_macro_threshold_summary"]["mean_classification"] = "worsened"
    run.write_report(tmp_path, tables)
    assert run.read_json(tmp_path / "decision.json")["recommendation"] == "supports_single_projection"
    assert len(list(tmp_path.glob("*.png"))) == 2
    with pytest.raises(ValueError, match="完整"):
        run.summary_tables(frame.iloc[:-2])
    frame.loc[0, "dataset_row_id"] = 99
    with pytest.raises(ValueError):
        run.summary_tables(frame)


@pytest.fixture
def references_fixture(monkeypatch, tmp_path):
    source = tmp_path / "baseline"
    source.mkdir()
    audit = run.audit_initialization(cfg=small_architecture(), mamba_factory=FixtureMamba)
    manifest = {"paired_initialization": {str(seed): {"A1": {"state_sha256": audit[str(seed)]["L0"]["state_sha256"]}} for seed in run.SEEDS}}
    run.write_json(source / "session.json", manifest)
    monkeypatch.setattr(run, "REFERENCE_SESSION", source)
    monkeypatch.setattr(run, "REFERENCE_SHA", run.sha(source / "session.json"))
    monkeypatch.setattr(run.base, "load_session", lambda _: manifest)
    monkeypatch.setattr(run, "audit_initialization", lambda: audit)
    value = OmegaConf.load(run.SPEC)
    value.reference_session_sha256 = run.REFERENCE_SHA
    path = tmp_path / "spec.yaml"
    OmegaConf.save(value, path)
    monkeypatch.setattr(run, "SPEC", path)
    code = "scripts/m4_linear_fusion_model.py"
    monkeypatch.setattr(run, "code_identity", lambda: {code: run.sha(run.ROOT / code)})
    sources = {}
    frame = metrics_fixture()
    for seed in run.SEEDS:
        d = source / "cells" / f"A1_seed{seed}"
        rows = frame[(frame.arm == "L0") & (frame.seed == seed)].assign(arm="A1")
        make_cell(d, "A1", seed, rows, run.base.config("A1", seed, d), 1056257)
        run.write_json(d / "initialization.json", {"state_sha256": audit[str(seed)]["L0"]["state_sha256"]})
        receipt = run.read_json(d / "completed.json")
        receipt["artifacts"]["initialization.json"] = run.sha(d / "initialization.json")
        (d / "completed.json").write_text(json.dumps(receipt))
        sources[f"A1_seed{seed}"] = {"path": str(d), "completed_sha256": run.sha(d / "completed.json")}
    summary = source / "summary" / "fixture"
    summary.mkdir(parents=True)
    run.write_json(summary / "receipt.json", {"sources": sources, "artifacts": {}})
    monkeypatch.setattr(run, "REFERENCE_SUMMARY", summary)
    monkeypatch.setattr(run, "REFERENCE_SUMMARY_SHA", run.sha(summary / "receipt.json"))
    return source


def test_reference_tamper_rejected(references_fixture):
    assert len(run.audit_references()["cells"]) == 3
    (references_fixture / "cells" / f"A1_seed{run.SEEDS[0]}" / "checkpoint.pt").write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="身份变化"):
        run.audit_references()


def test_prepare_complete_summary_and_source_preservation(references_fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "build_tho_data", lambda _: pytest.fail("不得访问真实数据"))
    before = {str(p): run.sha(p) for p in references_fixture.rglob("*") if p.is_file()}
    session = run.prepare(tmp_path / "new")
    assert run.prepare(session) == session
    assert len(list((session / "resolved_configs").glob("*.yaml"))) == 3
    with pytest.raises(RuntimeError, match="尚未全部完成"):
        run.summarize(session)
    frame = metrics_fixture()
    for c in run.plan():
        d = session / "cells" / c["cell"]
        make_cell(d, c["arm"], c["seed"], frame[(frame.arm == c["arm"]) & (frame.seed == c["seed"])], run.config(c["arm"], c["seed"], d))
    manifest = run.load_session(session)
    monkeypatch.setattr(run, "load_session", lambda _: manifest)
    original_spec = run.spec
    def small_spec():
        value = original_spec()
        value["data"] = {**value["data"], "val_windows": 2, "val_subjects": 2}
        return value
    monkeypatch.setattr(run, "spec", small_spec)
    monkeypatch.setattr(run, "summarize_task_metrics", lambda group: pd.DataFrame({"n_samples": [len(group)]}))
    output = run.summarize(session)
    assert run.summarize(session) == output
    assert len(pd.read_csv(output / "per_seed.csv")) == 30
    assert before == {str(p): run.sha(p) for p in references_fixture.rglob("*") if p.is_file()}


def test_parallel_environment_registration(references_fixture, tmp_path):
    session = run.prepare(tmp_path / "new")
    env = {"torch": "fixture", "cuda": None, "packages": {}, "gpu": None}
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: run.register_training_environment(session, env), range(2)))
    assert run.read_json(session / "training_environment.json") == env
    with pytest.raises(ValueError, match="环境"):
        run.register_training_environment(session, {**env, "torch": "other"})


@pytest.mark.parametrize("fail", [False, True])
def test_parallel_launcher_waits_for_both_and_preserves_logs(tmp_path, fail):
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "run_m4_linear_fusion_parallel.sh"
    shutil.copyfile(run.ROOT / "scripts/run_m4_linear_fusion_parallel.sh", launcher)
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("""#!/usr/bin/env python3
import os,sys,time
from pathlib import Path
args=sys.argv[1:]
if 'run' in args:
    shard=args[args.index('--shard-index')+1]
    time.sleep(.03 if shard=='0' else .08)
    Path('worker_'+shard).write_text('finished')
    sys.exit(7 if os.getenv('FIXTURE_FAIL')=='1' and shard=='0' else 0)
assert Path('worker_0').exists() and Path('worker_1').exists()
Path('summary_called').write_text('complete')
""")
    python.chmod(0o755)
    session = root / "session"
    session.mkdir()
    (session / "session.json").write_text("{}")
    env = {**os.environ, "FIXTURE_FAIL": "1" if fail else "0"}
    result = subprocess.run(["bash", str(launcher), str(session), "0", "1"], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == (7 if fail else 0), result.stderr
    assert (root / "worker_0").exists() and (root / "worker_1").exists()
    assert (root / "summary_called").exists() is (not fail)
    before = {p.name: p.read_bytes() for p in session.glob("gpu_*.log")}
    result = subprocess.run(["bash", str(launcher), str(session), "0", "1"], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert before == {p.name: p.read_bytes() for p in session.glob("gpu_*.log")}
