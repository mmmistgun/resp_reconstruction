"""仅 synthetic CPU：正式参数构造、配对初始化、梯度及独立生命周期。"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from scripts import m4_single_projection_model as mod
from scripts import m4_single_projection_runtime as run
from scripts import p1_components_runtime as original_run
from scripts.p1_components_model import P1ComponentsModel
from scripts.m4_residual_model import M4ResidualModel


class FixtureMamba(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.project = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.project(x)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(before)


def frequencies():
    return np.r_[np.geomspace(.03, .8, 56), np.geomspace(.81, 8, 41)]


def small():
    return replace(mod.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)


def model(arm, seed=20260811, *, full=False):
    return mod.M4SingleProjectionModel(arm, seed, frequencies(),
        cfg=mod.architecture() if full else small(), mamba_factory=FixtureMamba)


@pytest.mark.parametrize("seed", run.SEEDS)
def test_official_cpu_parameter_counts_and_every_shared_tensor(seed):
    # 官方 Mamba2 只构造 CPU 参数，不调用其 GPU forward。
    historical = P1ComponentsModel("M4", seed, frequencies())
    control = mod.M4SingleProjectionModel("M4", seed, frequencies())
    candidate = mod.M4SingleProjectionModel("SP", seed, frequencies())
    assert sum(p.numel() for p in control.parameters()) == 1056357
    assert sum(p.numel() for p in candidate.parameters()) == 1047045
    assert all(p.device.type == "cpu" for p in candidate.parameters())
    assert historical.state_dict().keys() == control.state_dict().keys()
    for key, value in historical.state_dict().items():
        assert torch.equal(value, control.state_dict()[key]), key
    common = mod.verify_shared_weights(control, candidate)
    assert len(common) == len(candidate.state_dict()) - 2
    for name in ("waveform_encoder", "condition.encoder", "condition.query", "condition.key",
                 "condition.value", "condition.frequency_bias", "condition.time_bias", "trunk", "decoder"):
        assert any(key.startswith(name + ".") for key in common)


def test_v2_is_structurally_reusable_but_initialization_not_matched():
    old = M4ResidualModel("A0", run.SEEDS[0], frequencies(), cfg=small(), mamba_factory=FixtureMamba)
    new = model("SP")
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    assert not torch.equal(old.waveform_encoder.project.weight, new.waveform_encoder.project.weight)
    assert not torch.equal(old.condition.query.weight, new.condition.query.weight)
    # 对齐权重后比较已打开的残差路径，验证 A0 的注意力数学与本实验相同。
    old.load_state_dict(new.state_dict(), strict=True)
    with torch.no_grad():
        old.condition.output.weight.normal_(std=.01)
        new.condition.output.load_state_dict(old.condition.output.state_dict())
    x, w = torch.randn(1, 1, 18000), {"w": torch.randn(1, 97, 360)}
    old.eval(); new.eval()
    torch.testing.assert_close(old(x, tf=w)["waveform"], new(x, tf=w)["waveform"], rtol=0, atol=0)


@pytest.mark.parametrize("arm,expected", [("M4", 2), ("SP", 1)])
def test_real_projection_calls_full_geometry_and_initial_identity(arm, expected):
    net = model(arm, full=True).eval()
    cfg = net.config
    assert (cfg.patch_samples, cfg.patch_hop_samples, cfg.patch_count, cfg.dimension,
            cfg.cwt_bins_per_patch, cfg.heads, cfg.layers) == (100, 50, 359, 96, 2, 4, 6)
    projection_names = [name for name, layer in net.named_modules()
        if isinstance(layer, nn.Linear) and (name == "condition.output" or name.startswith("fusion."))]
    assert len(projection_names) == expected
    calls, fused = [], []
    handles = [net.get_submodule(name).register_forward_hook(
        lambda layer, args, output, name=name: calls.append((name, tuple(args[0].shape)))) for name in projection_names]
    handles.append(net.fusion.register_forward_hook(lambda layer, args, output:
        fused.append((args[0].detach(), output.detach()))))
    with torch.no_grad():
        output = net(torch.randn(1, 1, 18000), tf={"w": torch.randn(1, 97, 360)})["waveform"]
    for handle in handles:
        handle.remove()
    assert len(calls) == expected and all(shape == (1, 359, 96) for _, shape in calls)
    assert output.shape == (1, 1, 18000) and torch.isfinite(output).all()
    assert torch.equal(*fused[0])
    if arm == "SP":
        assert list(net.fusion.parameters()) == []
        assert torch.count_nonzero(net.condition.output.weight) == 0
        assert torch.count_nonzero(net.condition.output.bias) == 0
        z, delta = torch.randn(1, 359, 96), torch.randn(1, 359, 96)
        assert torch.equal(net.fusion(z, delta), z + delta)
    assert torch.all(net.overlap.weight > 0) and torch.all(net.overlap.denominator > 0)


@pytest.mark.parametrize("arm", ["M4", "SP"])
def test_first_step_gradient_gate_and_second_step_upstream(arm):
    torch.manual_seed(91)
    net = model(arm).eval()
    x, w = torch.randn(1, 1, 18000), {"w": torch.randn(1, 97, 360)}
    target = torch.randn_like(x)
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    (net(x, tf=w)["waveform"] - target).square().mean().backward()
    opening = net.fusion.projection if arm == "M4" else net.condition.output
    assert opening.weight.grad.abs().sum() > 0 and opening.bias.grad.abs().sum() > 0
    for name, parameter in net.condition.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if arm == "M4" or not name.startswith("output."):
            assert torch.count_nonzero(parameter.grad) == 0, name
    for module in (net.waveform_encoder, net.trunk, net.decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    (net(x, tf=w)["waveform"] - target).square().mean().backward()
    for name in ("encoder.1.weight", "query.weight", "key.weight", "value.weight",
                 "frequency_bias.0.weight", "time_bias.0.weight"):
        grad = dict(net.condition.named_parameters())[name].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0, name


@pytest.mark.parametrize("seed", run.SEEDS)
def test_config_all_scientific_sections_match_original(seed, tmp_path):
    old = original_run.config("M4", seed, tmp_path)
    for arm in run.ARMS:
        new = run.config(arm, seed, tmp_path)
        for section in ("data", "window", "loss", "evaluation", "training"):
            assert old[section] == new[section], section
        assert old.model.architecture == new.model.architecture
        assert new.model.name == "m4_single_projection_v1"
        assert dict(new.model.m4_single_projection) == mod.contract(arm)
    assert len(run.plan()) == 6
    assert {x["arm"] for x in run.plan(1, 2)} == {"SP"}


@pytest.mark.parametrize("arm", ["M4", "SP"])
def test_task_loss_two_updates_with_four_microbatches(arm, tmp_path, monkeypatch):
    session = tmp_path / arm
    session.mkdir()
    run.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(run, "load_session", lambda path: {})
    monkeypatch.setattr(run, "architecture", small)
    monkeypatch.setattr(run, "cwt_magnitude_features", lambda x:
        (np.random.default_rng(42).random((97, 360), dtype=np.float32), frequencies()))
    result = run.smoke(session, arm, "cpu", mamba_factory=FixtureMamba, batch_size=1)
    receipt = run.read_json(result / "receipt.json")
    assert receipt["passed"] and receipt["updates"] == 2 and receipt["accumulation"] == 4
    assert receipt["mamba"] == "fixture" and receipt["peak_allocated_bytes"] is None


@pytest.mark.parametrize("arm", ["M4", "SP"])
def test_nonfinite_inputs_explicit_failure(arm):
    net = model(arm)
    x, w = torch.randn(1, 1, 18000), torch.ones(1, 97, 360)
    w[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        net(x, tf={"w": w})


def metric_frame():
    records = []
    for cell in run.plan():
        for row, subject in [(10, 101), (20, 102)]:
            record = {"arm": cell["arm"], "seed": cell["seed"], "dataset_row_id": row,
                "samp_id": subject, "split": "val", "whole_rr_target_eligible": True,
                "local_rr_target_eligible": True, "local_rr_target_eligible_windows": 9,
                "joint_target_eligible": True, "envelope_spearman_target_eligible": True,
                "joint_prediction_degenerate": False, "envelope_spearman_prediction_degenerate": False}
            for metric in run.sf.PRIMARY:
                record[metric] = (.8 if cell["arm"] == "SP" else .7) if metric == run.sf.PCC else (1 if cell["arm"] == "SP" else 2)
            records.append(record)
    return pd.DataFrame(records)


def test_summary_all_metrics_pair_sign_sd_and_missing_cell():
    frame = metric_frame()
    tables = run.summary_tables(frame)
    assert len(tables["per_seed"]) == 30 and len(tables["paired_delta"]) == 15
    paired = tables["paired_across_seed"]
    assert len(paired) == 5 and paired.improved_seeds.eq(3).all()
    assert paired.delta_sd.eq(0).all()
    np.testing.assert_allclose(paired.loc[paired.metric == run.sf.PCC, "delta_mean"], .1)
    np.testing.assert_allclose(paired.loc[paired.metric != run.sf.PCC, "delta_mean"], -1)
    with pytest.raises(ValueError, match="完整"):
        run.summary_tables(frame.iloc[:-2])
    frame.loc[0, run.sf.PRIMARY[0]] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        run.summary_tables(frame)


def test_prepare_does_not_read_data_freezes_sources_and_rejects_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "build_tho_data", lambda *a: pytest.fail("禁止真实数据访问"))
    session = run.prepare(tmp_path / "session")
    assert run.prepare(session) == session
    record = run.load_session(session)
    assert record["splits"] == ["train", "val"] and not record["research_test"]
    assert "scripts/p1_components_model.py" in record["code_sha256"]
    partial = tmp_path / "partial"
    partial.mkdir(); (partial / "failed.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        run.prepare(partial)


def test_environment_and_completion_integrity(tmp_path):
    directory = tmp_path / "cells/M4_seed20260811"
    directory.mkdir(parents=True)
    environment = {"torch": "fixture", "cuda": "fixture", "packages": {}, "gpu": "fixture",
        "LD_LIBRARY_PATH_unset": True, "LD_PRELOAD_unset": True}
    run.write_json(directory / "environment.json", environment)
    run.check_environments(tmp_path, environment)
    with pytest.raises(ValueError, match="环境"):
        run.check_environments(tmp_path, {**environment, "gpu": "other"})
    (directory / "artifact").write_text("fixture")
    run.write_json(directory / "completed.json", {"artifacts": {"artifact": run.sha(directory / "artifact")}})
    run.checked_completion(directory)
    (directory / "artifact").write_text("changed")
    with pytest.raises(ValueError, match="身份"):
        run.checked_completion(directory)


def fixture_cell(parent, arm, seed):
    """仅构造可校验的文件生命周期，checkpoint 是不会加载的占位字节。"""
    directory = parent / "cells" / f"{arm}_seed{seed}"
    attempt = directory / "attempts/fixture"
    attempt.mkdir(parents=True)
    environment = {"torch": "fixture", "cuda": "fixture", "packages": {}, "gpu": "fixture",
        "LD_LIBRARY_PATH_unset": True, "LD_PRELOAD_unset": True}
    run.write_json(directory / "environment.json", environment)
    cfg = original_run.config("M4", seed, directory) if arm == "M4" else run.config(arm, seed, directory)
    OmegaConf.save(cfg, directory / "config.yaml")
    (directory / "checkpoint.pt").write_bytes(b"synthetic placeholder; never loaded")
    metrics = metric_frame().query("arm == @arm and seed == @seed")
    metrics.to_csv(attempt / "validation_metrics.csv", index=False)
    run.summarize_task_metrics(metrics).to_csv(attempt / "validation_summary.csv", index=False)
    pd.DataFrame([{"epoch": 1, "elapsed_seconds": 2.}]).to_csv(attempt / "train_history.csv", index=False)
    engineering = parent / "engineering" / arm
    engineering.mkdir(parents=True, exist_ok=True)
    er = engineering / "receipt.json"
    if not er.exists():
        run.write_json(er, {"environment": environment, "passed": True, "arm": arm,
            "timing_scope": "two_update_engineering_check_including_initial_kernel_cost",
            "batch_size": 32, "accumulation": 4,
            "updates": 2, "mamba": "official_mamba2", "train_elapsed_seconds": 3.,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 200})
    run.write_json(attempt / "engineering.json", {"path": str(engineering), "sha256": run.sha(er)})
    files = [directory / "checkpoint.pt", directory / "config.yaml", attempt / "validation_metrics.csv",
        attempt / "validation_summary.csv", attempt / "train_history.csv"]
    run.write_json(directory / "completed.json", {"arm": arm, "seed": seed,
        "metrics": "attempts/fixture/validation_metrics.csv", "parameters": 1056357 if arm == "M4" else 1047045,
        "best_epoch": 1, "best_local_rr": float(metrics.local_rr_mae_bpm.mean()),
        "completed_epochs": 1, "optimizer_updates": 80, "early_stopped": False,
        "best_checkpoint": "checkpoint.pt", "final_checkpoint": "checkpoint.pt",
        "artifacts": {str(path.relative_to(directory)): run.sha(path) for path in files}})
    return directory


def test_historical_reference_and_complete_summary_lifecycle(tmp_path, monkeypatch):
    session = run.prepare(tmp_path / "new")
    historical = tmp_path / "historical"
    for seed in run.SEEDS:
        fixture_cell(historical, "M4", seed)
    source = "scripts/p1_components_model.py"
    run.write_json(historical / "session.json", {"code_sha256": {source: run.sha(run.ROOT / source)}})
    audit = {"sessions": {"p1": {"path": str(historical), "session_sha256": run.sha(historical / "session.json")}}}
    read = run.read_json
    monkeypatch.setattr(run, "read_json", lambda path: audit if Path(path) == run.ROOT / "configs/m4_single_projection/source_audit.json" else read(path))
    before = {str(p): run.sha(p) for p in historical.rglob("*") if p.is_file()}
    run.reuse_m4(session)
    run.reuse_m4(session)
    assert sum(c["status"] == "historical_reference" for c in run.status(session)) == 3
    for seed in run.SEEDS:
        assert run.cell_directory(session, f"M4_seed{seed}") == historical / "cells" / f"M4_seed{seed}"
        fixture_cell(session, "SP", seed)
    # 仅 summary 使用两条 synthetic 行，训练合同没有变化。
    spec = run.spec()
    monkeypatch.setattr(run, "load_session", lambda path: {})
    monkeypatch.setattr(run, "spec", lambda: {**spec, "data": {**spec["data"], "val_windows": 2, "val_subjects": 2}})
    result = run.summarize(session)
    assert len(pd.read_csv(result / "efficiency.csv")) == 6
    assert len(pd.read_csv(result / "selection.csv")) == 6
    assert len(pd.read_csv(result / "paired_across_seed.csv")) == 5
    assert run.summarize(session) == result
    assert {str(p): run.sha(p) for p in historical.rglob("*") if p.is_file()} == before
    checked = run.checked_completion
    with monkeypatch.context() as patch:
        patch.setattr(run, "checked_completion", lambda directory:
            {**checked(directory), "best_local_rr": float("nan")})
        with pytest.raises(ValueError, match="选点回执"):
            run.summarize(session)
    with (historical / "cells/M4_seed20260811/checkpoint.pt").open("ab") as stream:
        stream.write(b"changed fixture")
    with pytest.raises(ValueError, match="身份"):
        run.summarize(session)


def test_control_reuse_mode_cannot_change_after_training_starts(tmp_path):
    session = run.prepare(tmp_path / "session")
    cell = session / "cells/SP_seed20260811"
    cell.mkdir(parents=True)
    (cell / "config.yaml").write_text("fixture: true\n")
    with pytest.raises(ValueError, match="首个训练"):
        run.reuse_m4(session)


@pytest.mark.parametrize("field,value", [
    ("train_elapsed_seconds", float("nan")), ("train_elapsed_seconds", float("inf")),
    ("peak_allocated_bytes", None), ("peak_reserved_bytes", 0),
    ("passed", False), ("arm", "SP"), ("timing_scope", "different_scope"),
])
def test_summary_rejects_invalid_engineering_evidence(field, value):
    receipt = {"passed": True, "arm": "M4", "batch_size": 32, "accumulation": 4,
        "updates": 2, "mamba": "official_mamba2", "train_elapsed_seconds": 1.,
        "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
        "timing_scope": "two_update_engineering_check_including_initial_kernel_cost"}
    run.validate_engineering_receipt(receipt, "M4")
    with pytest.raises(ValueError):
        run.validate_engineering_receipt({**receipt, field: value}, "M4")
