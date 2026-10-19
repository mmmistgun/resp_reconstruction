"""位置评分/值路径的 CPU 合同、配对、来源复用和交互验收。"""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from scripts import m4_pool_position_paths_model as mod
from scripts import m4_pool_position_paths_runtime as run
from scripts.p1_components_model import P1ComponentsModel
from test_m4_pool_position import small, fixture_frame
from test_p1_components import FixtureMamba, frequencies, cpu_only


def model(arm, seed=20260811):
    return mod.PoolPositionModel(arm, seed, frequencies(), cfg=small(), mamba_factory=FixtureMamba)


def test_training_matrix_and_frozen_reference_boundary(tmp_path, monkeypatch):
    assert len(run.plan()) == 6 and len(run.plan(include_references=True)) == 12
    assert set(c["arm"] for c in run.plan()) == {"C", "D"}
    assert len(run.plan(0, 2)) == len(run.plan(1, 2)) == 3
    for cell in run.plan():
        cfg = run.config(cell["arm"], cell["seed"], tmp_path)
        old = run.base.config("A", cell["seed"], tmp_path)
        for key in ("data", "window", "training", "loss", "evaluation"):
            assert cfg[key] == old[key]
        cfg.model.architecture = asdict(small())
        monkeypatch.setattr(mod, "frequency_metadata", lambda _: frequencies())
        net = mod.build_model(cfg, mamba_factory=FixtureMamba)
        assert net.waveform_encoder.score_position == (cell["arm"] == "C")
        assert net.waveform_encoder.value_position == (cell["arm"] == "D")
    for arm in mod.REFERENCES:
        with pytest.raises(ValueError, match="冻结参照"):
            run.run_cell(tmp_path, arm, run.SEEDS[0], "cpu")
        with pytest.raises(ValueError, match="C/D"):
            run.smoke(tmp_path, arm, "cpu")


@pytest.mark.parametrize("seed", run.SEEDS)
def test_shared_state_and_cuda_seed_dispatch(seed, monkeypatch):
    dispatch = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", dispatch.append)
    before = torch.get_rng_state().clone()
    original = P1ComponentsModel("M4", seed, frequencies(), cfg=small(), mamba_factory=FixtureMamba)
    expected = list(dispatch)
    for arm in mod.TRAIN_ARMS:
        dispatch.clear()
        candidate = model(arm, seed)
        assert dispatch == expected
        assert torch.equal(before, torch.get_rng_state())
        assert candidate.state_dict().keys() == original.state_dict().keys()
        for name, value in candidate.state_dict().items():
            assert torch.equal(value, original.state_dict()[name]), name


@pytest.mark.parametrize("arm", ["C", "D"])
@pytest.mark.parametrize("bf16", [False, True])
def test_complete_gradients(arm, bf16):
    net = model(arm)
    with torch.no_grad():
        net.fusion.projection.weight.normal_(std=.02)
    tf = torch.rand(1, 97, 360, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        y = net(torch.randn(1, 1, 18000), tf={"w": tf})["waveform"]
    y.square().mean().backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert tf.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", ["C", "D"])
def test_formal_task_loss_two_updates(arm, tmp_path, monkeypatch):
    run.write_json(tmp_path / "session.json", {"fixture": True})
    monkeypatch.setattr(run, "load_session", lambda _: {})
    monkeypatch.setattr(run, "architecture", small)
    monkeypatch.setattr(run, "cwt_magnitude_features", lambda _: (np.ones((97, 360), dtype=np.float32), frequencies()))
    output = run.smoke(tmp_path, arm, "cpu", mamba_factory=FixtureMamba, batch_size=1)
    receipt = run.read_json(output / "receipt.json")
    assert receipt["passed"] and receipt["updates"] == 2


def full_frame():
    source = fixture_frame().query("arm=='A'")
    pieces = []
    for arm, error in {"A": 4., "B": 10., "C": 8., "D": 7.}.items():
        frame = source.copy()
        frame["arm"] = arm
        for metric in run.sf.PRIMARY:
            frame[metric] = 1-.01*error if metric == run.sf.PCC else error
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def test_conditional_effects_main_effects_and_interaction():
    frame = full_frame()
    tables = run.summary_tables(frame)
    assert len(tables["per_seed"]) == 60 and len(tables["paired_delta"]) == 45
    effects = tables["path_effects"]
    expected = {"score_at_value0_C_minus_B": -2., "score_at_value1_A_minus_D": -3.,
        "value_at_score0_D_minus_B": -3., "value_at_score1_A_minus_C": -4.,
        "score_main": -2.5, "value_main": -3.5, "interaction_A_minus_C_minus_D_plus_B": -1.}
    for effect, value in expected.items():
        selected = effects.query("effect==@effect")
        np.testing.assert_allclose(selected.loc[selected.metric!=run.sf.PCC, "delta"], value)
        np.testing.assert_allclose(selected.loc[selected.metric==run.sf.PCC, "delta"], -.01*value)
    assert tables["path_effects_across_seed"].improved_seeds.eq(3).all()
    assert len(tables["subject_path_effects"]) == 2*len(effects)
    with pytest.raises(ValueError, match="完整"):
        run.summary_tables(frame.query("arm!='D'"))
    broken = frame.copy()
    broken.loc[0, run.sf.PRIMARY[0]] = np.inf
    with pytest.raises((ValueError, FloatingPointError)):
        run.summary_tables(broken)


def make_cell(directory, arm, seed, frame, cfg, initialization):
    directory.mkdir(parents=True)
    OmegaConf.save(cfg, directory / "config.yaml")
    frame.to_csv(directory / "validation_metrics.csv", index=False)
    (directory / "checkpoint.pt").write_bytes(b"synthetic checkpoint")
    run.write_json(directory / "environment.json", {"torch": "fixture", "cuda": None, "packages": {}, "gpu": "fixture"})
    run.write_json(directory / "initialization.json", initialization)
    wave = np.sin(np.arange(18000, dtype=np.float32)/100)[None,None].repeat(2, axis=0)
    np.savez(directory / "cases.npz", x=wave, target=wave, prediction=wave,
        dataset_row_id=np.array([10,20]), samp_id=np.array([101,102]))
    run.write_json(directory / "completed.json", {"arm": arm, "seed": seed, "parameters": initialization["parameters"],
        "best_epoch": 2, "metrics": "validation_metrics.csv", "best_checkpoint": "checkpoint.pt", "final_checkpoint": "checkpoint.pt",
        "artifacts": {name: run.sha(directory / name) for name in ("config.yaml", "validation_metrics.csv", "checkpoint.pt",
            "environment.json", "initialization.json", "cases.npz")}})


@pytest.fixture
def references(tmp_path, monkeypatch):
    directory = tmp_path / "references"
    directory.mkdir()
    initialization = run.audit_initialization(cfg=small(), mamba_factory=FixtureMamba)
    source = {"paired_initialization": initialization}
    run.write_json(directory / "session.json", source)
    monkeypatch.setattr(run, "REFERENCE_SESSION", directory)
    monkeypatch.setattr(run, "REFERENCE_SHA", run.sha(directory / "session.json"))
    monkeypatch.setattr(run.base, "load_session", lambda _: source)
    spec = OmegaConf.load(run.SPEC)
    spec.reference_session_sha256 = run.REFERENCE_SHA
    path = tmp_path / "spec.yaml"
    OmegaConf.save(spec, path)
    monkeypatch.setattr(run, "SPEC", path)
    frame = full_frame()
    for arm in mod.REFERENCES:
        for seed in run.SEEDS:
            cell = directory / "cells" / f"{arm}_seed{seed}"
            make_cell(cell, arm, seed, frame.query("arm==@arm and seed==@seed"), run.base.config(arm, seed, cell), initialization[str(seed)][arm])
    monkeypatch.setattr(run, "audit_initialization", lambda: initialization)
    relative = "scripts/m4_pool_position_paths_model.py"
    monkeypatch.setattr(run, "code_identity", lambda: {relative: run.sha(run.ROOT / relative)})
    return directory, initialization


def test_reference_integrity_and_environment(references, tmp_path):
    directory, _ = references
    references_record = run.audit_references()
    assert len(references_record["cells"]) == 6
    session = run.prepare(tmp_path / "new")
    environment = {"torch": "fixture", "cuda": None, "packages": {}, "gpu": "fixture", "device": "cuda:0"}
    run.register_training_environment(session, environment)
    with pytest.raises(ValueError, match="冻结 A/B"):
        run.register_training_environment(session, {**environment, "gpu": "changed"})
    (directory / "cells" / f"A_seed{run.SEEDS[0]}" / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="身份变化"):
        run.load_session(session)


def test_prepare_and_full_summary_preserve_references(references, tmp_path, monkeypatch):
    directory, initialization = references
    before = {str(p): run.sha(p) for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(run, "build_tho_data", lambda _: pytest.fail("准备和汇总只读取已保存产物"))
    session = run.prepare(tmp_path / "new")
    assert run.prepare(session) == session
    assert sum(c["status"] == "reference_verified" for c in run.status(session)) == 6
    assert len(list((session / "resolved_configs").glob("*.yaml"))) == 6
    frame = full_frame()
    for cell in run.plan():
        arm, seed = cell["arm"], cell["seed"]
        target = session / "cells" / cell["cell"]
        make_cell(target, arm, seed, frame.query("arm==@arm and seed==@seed"), run.config(arm, seed, target), initialization[str(seed)][arm])
    manifest = run.load_session(session)
    original = run.spec
    def small_spec():
        value = original()
        value["data"].update(val_windows=2, val_subjects=2)
        return value
    monkeypatch.setattr(run, "load_session", lambda _: manifest)
    monkeypatch.setattr(run, "spec", small_spec)
    output = run.summarize(session)
    assert run.summarize(session) == output
    assert len(pd.read_csv(output / "selection.csv")) == 12
    assert len(pd.read_csv(output / "path_effects.csv")) == 105
    assert len(list(output.glob("*.png"))) == 6
    assert before == {str(p): run.sha(p) for p in directory.rglob("*") if p.is_file()}
