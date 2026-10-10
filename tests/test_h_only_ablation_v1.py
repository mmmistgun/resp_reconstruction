"""仅 synthetic/disposable 的 CPU 验收；原生 Mamba CUDA 另由用户验收。"""
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from resp_train.paper_evidence.cwt_apor_v2.model import build_model as historical
from resp_train.paper_evidence.cwt_apor_v2.spec import config as old_config
from resp_train.paper_evidence.h_only_ablation_v1.spec import ARMS, SEEDS, config, contract, plan
from resp_train.paper_evidence.h_only_ablation_v1.model import build_model, gain, HCondition
from resp_train.paper_evidence.h_only_ablation_v1.history import validate_history
from resp_train.paper_evidence.h_only_ablation_v1.diagnostics import gain_statistics, TrainingMonitor
from resp_train.paper_evidence.h_only_ablation_v1.runtime import compatible_h65_config
from resp_train.paper_evidence.h_only_ablation_v1.report import report_tables
from resp_train.paper_evidence.h_only_ablation_v1 import artifacts as io
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.crd.training import crd_learning_rate

COUNTS = [1077448, 994312, 1076712, 1075240, 126064, 1076584, 1077448, 1065064,
          1068136, 1068136, 1068136, 1077448, 1077448, 1077448, 1077448, 1077448, 1077640]


def cpu_fixture(arm, seed=SEEDS[0]):
    model = build_model(arm, seed)
    # 此替换只用于本地 CPU 前向/反向测试；构造和参数核验保留真实 Mamba。
    model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in model.base.local_blocks])
    return model


def test_matrix_unique_change_and_config(tmp_path):
    assert len(plan()) == 51 and len({(c["arm"], c["seed"]) for c in plan()}) == 51
    baseline = asdict(ARMS["HA0"])
    for name, spec in ARMS.items():
        if name != "HA0":
            assert sum(value != baseline[key] for key, value in asdict(spec).items()) == 1
        cfg = config(name, SEEDS[0], tmp_path)
        assert cfg.loss.effort_weight == (0 if name == "HA12" else .25)
        assert cfg.training.epochs == 80 and cfg.training.batch_size == 128
        assert cfg.training.early_stopping_min_epoch == 30 and cfg.training.early_stopping_patience == 15
        assert cfg.training.max_learning_rate == 3e-4 and cfg.training.min_learning_rate == 3e-5
    assert OmegaConf.to_container(OmegaConf.load(io.ROOT / "configs/h_only_ablation_v1/experiment.yaml")) == contract()


@pytest.mark.parametrize("seed", SEEDS)
def test_nested_initialization_and_h65_exact(seed):
    old = historical(seed, {"shape": [41, 360], "arm": {"pool_samples": 50}})
    reference, candidate = build_model("HA16", seed), build_model("HA0", seed)
    assert old.state_dict().keys() == reference.state_dict().keys()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(reference.state_dict()[name], value, rtol=0, atol=0)
        expected = value
        if name == "branches.w.parameter_fill.expand.weight":
            expected = value[:64]
        if name == "branches.w.parameter_fill.project.weight":
            expected = value[:, :64]
        torch.testing.assert_close(candidate.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("arm,expected", zip(ARMS, COUNTS))
def test_parameter_counts_shared_initialization_and_graph(arm, expected):
    baseline, model = build_model("HA0", SEEDS[0]), build_model(arm, SEEDS[0])
    assert model.parameter_count == expected
    common = dict(baseline.named_parameters())
    for name, value in model.named_parameters():
        if name in common and value.shape == common[name].shape:
            torch.testing.assert_close(value, common[name], rtol=0, atol=0)
    if arm == "HA1":
        assert len(model.branches) == 0
    if arm == "HA2":
        assert not any("patch_mixer" in n or ".norm1." in n for n, _ in model.named_parameters())
    if arm == "HA3":
        assert not any("channel_mixer" in n or ".norm2." in n for n, _ in model.named_parameters())
    if arm == "HA4":
        assert not model.base.local_blocks and model.base.frontend.adapter.out_channels == 96
    if arm == "HA5":
        assert model.branches["w"].conv_in.kernel_size == model.branches["w"].depthwise.kernel_size == (1, 3)
    if arm == "HA7":
        assert isinstance(model.branches["w"].parameter_fill, nn.Identity)
    if arm in ("HA8", "HA9"):
        assert model.branches["w"].final_projection.out_channels == 96


@pytest.mark.parametrize("arm", ARMS)
def test_cpu_forward_backward_all_registered_parameters(arm):
    model = cpu_fixture(arm)
    if ARMS[arm].condition:
        with torch.no_grad():
            model.branches["w"].final_projection.weight.normal_(0, .01)
    x = torch.randn(1, 1, 18000, requires_grad=True)
    tf = {"w": torch.rand(1, 41, 360)} if ARMS[arm].condition else None
    result = model(x, tf=tf)["waveform"]
    assert result.shape == (1, 1, 18000) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_center_five_slots_and_unchanged_projection():
    baseline, model = build_model("HA0", SEEDS[0]), build_model("HA6", SEEDS[0])
    branch = model.branches["w"]
    times = 24.5 + torch.arange(360)*50
    value = branch.sample(times[None, None].float())
    expected = (128*torch.arange(140)+127.5).float()[:, None].expand(140, 5)
    torch.testing.assert_close(value[0, 0], expected, rtol=1e-6, atol=.002)
    assert value.shape == (1, 1, 140, 5) and branch.local_projection.in_channels == 480
    torch.testing.assert_close(branch.local_projection.weight, baseline.branches["w"].local_projection.weight, rtol=0, atol=0)


@pytest.mark.parametrize("arm,start", [("HA8", 96), ("HA9", 0)])
def test_single_film_copies_correct_projection_half(arm, start):
    source = build_model("HA16", SEEDS[0]).branches["w"]
    with torch.no_grad():
        source.final_projection.weight.normal_()
        source.final_projection.bias.normal_()
    branch = HCondition(source, ARMS[arm], SEEDS[0])
    torch.testing.assert_close(branch.final_projection.weight, source.final_projection.weight[start:start+96])
    torch.testing.assert_close(branch.final_projection.bias, source.final_projection.bias[start:start+96])


@pytest.mark.parametrize("mode,limits", [("bounded", (.5, 1.5)), ("positive", (0, 2)), ("signed", (-1, 3)), ("unbounded", None)])
def test_gain_initial_derivative_and_limits(mode, limits):
    raw = torch.tensor(0., requires_grad=True)
    result = gain(raw, mode)
    assert result.item() == 1
    result.backward()
    assert raw.grad.item() == .5
    extreme = gain(torch.tensor([-1e6, 1e6]), mode)
    if limits:
        torch.testing.assert_close(extreme, torch.tensor(limits, dtype=torch.float32))
    else:
        assert extreme[0] < -1000 and extreme[1] > 1000


def test_diagnostics_and_gradient_monitor(tmp_path):
    stats = gain_statistics(torch.tensor([-4., -.05, 0., .05, 4.]))
    assert stats["min"] == -4 and stats["max"] == 4
    assert stats["negative_fraction"] == pytest.approx(.4)
    assert stats["near_zero_fraction"] == pytest.approx(.6)
    assert stats["extreme_fraction"] == pytest.approx(.4)
    with pytest.raises(FloatingPointError):
        gain_statistics(torch.tensor([float("nan")]))
    model = nn.Linear(2, 1, bias=False)
    monitor = TrainingMonitor(model, tmp_path, False)
    model(torch.ones(1, 2)).sum().backward()
    monitor.close()
    record = json.loads((tmp_path / "diagnostics.jsonl").read_text())
    assert record["pre_clip_norm"] == pytest.approx(np.sqrt(2))


def test_checkpointed_condition_monitor_counts_one_backward(tmp_path):
    model = cpu_fixture("HA0")
    monitor = TrainingMonitor(model, tmp_path, True)
    try:
        # batch>8 覆盖正式训练使用的非 reentrant 条件分块重算。
        output = model(torch.randn(9, 1, 18000), tf={"w": torch.rand(9, 41, 360)})
        output["waveform"].square().mean().backward()
    finally:
        monitor.close()
    records = [json.loads(line) for line in (tmp_path / "diagnostics.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in records] == ["train_gain", "gradient"]
    assert records[0]["min"] == records[0]["max"] == 1.
    assert records[0]["n"] == 9*96*140
    assert monitor.update == 1


def test_real_data_and_test_commands_require_explicit_flags(tmp_path):
    from scripts.run_h_only_ablation_v1 import parser
    from resp_train.paper_evidence.h_only_ablation_v1.report import summarize
    from resp_train.paper_evidence.h_only_ablation_v1.research_test import prepare_data, run_test
    for command, extra in (("prepare", []), ("train", ["--arms", "HA0", "--device", "cuda:0"]),
                           ("acceptance", ["--arms", "HA0", "--device", "cuda:0"]), ("test-prepare", [])):
        with pytest.raises(SystemExit):
            parser().parse_args([command, "--session", str(tmp_path), *extra])
    with pytest.raises(PermissionError):
        summarize(tmp_path, "test")
    with pytest.raises(PermissionError):
        prepare_data(tmp_path)
    with pytest.raises(PermissionError):
        run_test(tmp_path, "HA0", SEEDS[0], "cuda:0")


def test_incomplete_matrix_cannot_freeze_or_access_test(tmp_path, monkeypatch):
    from resp_train.paper_evidence.h_only_ablation_v1 import research_test as rt
    monkeypatch.setattr(io, "load_session", lambda session: {})
    (tmp_path / "session.json").write_text("{}")
    with pytest.raises(RuntimeError, match="51-cell"):
        rt.freeze(tmp_path)
    with pytest.raises(RuntimeError, match="allowlist"):
        rt.prepare_data(tmp_path, confirmed=True)


def test_completed_training_cannot_be_retrained(tmp_path, monkeypatch):
    from resp_train.paper_evidence.h_only_ablation_v1 import runtime
    monkeypatch.setattr(io, "load_session", lambda session: {})
    (tmp_path / "session.json").write_text("{}")
    parent = runtime.parent(tmp_path, "HA0", SEEDS[0]) / "attempt_interrupted"
    parent.mkdir(parents=True)
    (parent / "training_complete.json").write_text("{}")
    with pytest.raises(RuntimeError, match="recover-export"):
        runtime.run_formal(tmp_path, "HA0", SEEDS[0], "cuda:0", retry=True)


def history_fixture(cfg):
    records, best, wait = [], float("inf"), 0
    for epoch in range(1, 31):
        val = 2. if epoch == 1 else 1.
        improved = val < best
        best, wait = min(best, val), 0 if improved else wait + 1
        records.append({"epoch": epoch, "optimizer_update": epoch*80, "train_loss_sync": 1., "train_loss_effort": 2.,
            "train_loss_total": 1+2*cfg.loss.effort_weight, "val_local_rr_mae": val,
            "first_learning_rate": crd_learning_rate((epoch-1)*80, total_updates=6400, max_learning_rate=3e-4, min_learning_rate=3e-5),
            "last_learning_rate": crd_learning_rate(epoch*80-1, total_updates=6400, max_learning_rate=3e-4, min_learning_rate=3e-5),
            "early_stopping_improved": int(improved), "early_stopping_wait": wait,
            "early_stopping_triggered": int(epoch >= 30 and wait >= 15), "early_stopping_min_epoch": 30})
    return pd.DataFrame(records)


@pytest.mark.parametrize("arm", ("HA0", "HA12"))
def test_history_own_loss_earliest_tie_and_lr(arm, tmp_path):
    cfg = config(arm, SEEDS[0], tmp_path)
    frame = history_fixture(cfg)
    assert validate_history(frame, cfg) == 2
    wrong = frame.copy()
    wrong.loc[0, "train_loss_total"] += .1
    with pytest.raises(ValueError, match="loss"):
        validate_history(wrong, cfg)
    wrong = frame.copy()
    wrong.loc[29, "last_learning_rate"] = 3e-5
    with pytest.raises(ValueError, match="LR"):
        validate_history(wrong, cfg)


def test_h65_compatibility_rejects_full_band_and_changed_contract(tmp_path):
    rep = {"shape": [41, 360], "arm": {"name": "H"}}
    cfg = old_config("H", SEEDS[0], tmp_path)
    cfg.model.cwt_representation = rep
    compatible_h65_config(cfg, SEEDS[0], rep)
    bad = deepcopy(cfg)
    bad.model.cwt_apor_v2.high_only = False
    with pytest.raises(ValueError):
        compatible_h65_config(bad, SEEDS[0], rep)
    bad = deepcopy(cfg)
    bad.training.batch_size = 64
    with pytest.raises(ValueError):
        compatible_h65_config(bad, SEEDS[0], rep)


def test_complete_matrix_summary_and_subject_weighting():
    records = []
    for cell in plan():
        for index, subject in enumerate((1, 1, 2)):
            row = {**cell, "dataset_row_id": index, "samp_id": subject, "split": "val"}
            row.update({m: float(index+1) if m != sf.PCC else .5 for m in sf.PRIMARY})
            row.update({flag: True for flag in set(sf.ELIGIBILITY.values()) | {"local_rr_target_eligible", "joint_target_eligible"}})
            row.update(joint_prediction_degenerate=False, envelope_spearman_prediction_degenerate=False)
            records.append(row)
    frame = pd.DataFrame(records)
    result = report_tables(frame)
    metric = sf.ERRORS[0]
    assert result["per_seed"].query("arm == 'HA0' and metric == @metric")["mean"].iloc[0] == 2
    assert result["subject_macro"].query("arm == 'HA0' and metric == @metric")["mean"].iloc[0] == 2.25
    assert result["across_seed"].seed_sd.eq(0).all()
    with pytest.raises(ValueError, match="完整"):
        report_tables(frame[frame.arm.ne("HA15")])


def test_failure_preserved_and_retry_has_new_identity(tmp_path):
    key = {"phase": "synthetic"}
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with io.attempt(tmp_path / "stage", key) as output:
            (output / "partial.txt").write_text("retained")
            raise RuntimeError("synthetic failure")
    with pytest.raises(RuntimeError, match="失败"):
        with io.attempt(tmp_path / "stage", key):
            pass
    with io.attempt(tmp_path / "stage", key, retry=True) as second:
        (second / "result.txt").write_text("complete")
    assert second != output and (output / "partial.txt").read_text() == "retained"
    assert io.completed(tmp_path / "stage", key) == second
    with pytest.raises(FileExistsError):
        with io.attempt(tmp_path / "stage", key, retry=True):
            pass
