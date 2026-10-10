"""M4 高频机制合成测试：控制的数值正确性、严格配对和完整生命周期。"""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts import m4_high_frequency_core as core
from scripts import run_m4_high_frequency as run
from scripts import plot_m4_high_frequency as plot
from scripts.p1_components_model import P1ComponentsModel, architecture


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def frequencies():
    return np.r_[np.geomspace(.03, .8, 56), np.geomspace(.81, 8, 41)]


class FixtureMamba(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.projection(x)


def model():
    cfg = replace(architecture(), dimension=16, waveform_channels=8, condition_channels=8, layers=1)
    net = P1ComponentsModel("M4", core.SEEDS[0], frequencies(), cfg=cfg, mamba_factory=FixtureMamba).eval()
    # 正式初始 residual 为零；打开残差才能验证干预确实沿真实读取路径传递。
    with torch.no_grad():
        net.fusion.projection.weight.normal_(std=.05)
    return net


def test_row_shifts_are_batch_order_seed_independent():
    ids = [88, 7, 42]
    shifts = core.row_shifts(ids)
    np.testing.assert_array_equal(shifts[[2, 0]], core.row_shifts([42, 88]))
    assert shifts.shape == (3, 3) and shifts.min() >= 60 and shifts.max() <= 300
    assert all(len(set(s)) == 3 for s in shifts)
    with pytest.raises(ValueError):
        core.row_shifts([-1])


@pytest.mark.parametrize("kind", core.TRANSFORMS)
def test_transform_preserves_low_band_and_high_shift_distribution(kind):
    original = torch.rand(2, 97, 360)
    before = original.clone()
    shifts = core.row_shifts([1, 2])
    changed = core.transform(original, frequencies(), kind, shifts)
    torch.testing.assert_close(original, before, rtol=0, atol=0)
    torch.testing.assert_close(changed[:, :56], original[:, :56], rtol=0, atol=0)
    if "SHIFT" in kind:
        for i in range(2):
            torch.testing.assert_close(changed[i, 56:], original[i, 56:].roll(int(shifts[i, int(kind[-1])-1]), -1), rtol=0, atol=0)
            torch.testing.assert_close(changed[i, 56:].sort(-1).values, original[i, 56:].sort(-1).values, rtol=0, atol=0)
    if kind == "H_MEAN":
        torch.testing.assert_close(changed[:, 56:], original[:, 56:].mean(-1, keepdim=True).expand(-1, -1, 360))


def test_invalid_transform_and_nonfinite_fail():
    w = torch.rand(1, 97, 360)
    with pytest.raises(ValueError):
        core.transform(w, frequencies(), "H_SHIFT_1", np.array([[1, 2, 3]]))
    w[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        core.transform(w, frequencies(), "FULL", core.row_shifts([1]))


@pytest.mark.parametrize("amp", [False, True])
def test_actual_m4_path_fixed_gn_replay_and_hooks_cleanup(amp):
    net = model()
    x, w = torch.randn(2, 1, 18000), torch.rand(2, 97, 360)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        outputs = list(core.paired_batch(net, x, w, frequencies(), core.row_shifts([1, 2])))
    assert [r[0] for r in outputs] == list(core.CONDITIONS)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
    assert float((outputs[3][2].points["residual"] - outputs[0][2].points["residual"]).abs().max()) > 0
    assert all(not m._forward_hooks for m in net.modules())
    for k, v in net.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        core.traced_forward(net, x, w)


def test_exception_also_removes_hooks():
    net = model()
    x, w = torch.randn(1, 1, 18000), torch.rand(1, 97, 360)
    w[0, 1, 0] = float("inf")
    with torch.no_grad(), pytest.raises(FloatingPointError):
        core.traced_forward(net, x, w)
    assert all(not m._forward_hooks for m in net.modules())


def test_modulation_psd_detects_synthetic_amplitude_frequency():
    t = np.arange(360) / 2
    w = np.tile(3 + np.sin(2*np.pi*.25*t), (97, 1))
    freq, spectrum, valid = core.modulation_psd(w)
    assert valid.all()
    np.testing.assert_allclose(freq[np.argmax(spectrum, axis=1)], .25)
    np.testing.assert_allclose(spectrum.sum(-1), 1)
    _, constant, valid = core.modulation_psd(np.ones_like(w))
    assert not valid.any() and np.isnan(constant).all()


def metric_fixture():
    frames = []
    for seed in core.SEEDS:
        for condition in core.CONDITIONS:
            frame = pd.DataFrame({"dataset_row_id": [1, 2, 3], "samp_id": [10, 10, 20], "split": "val",
                "seed": seed, "condition": condition, "joint_prediction_degenerate": False,
                "envelope_spearman_prediction_degenerate": False})
            for key in set(run.sf.ELIGIBILITY.values()):
                frame[key] = True
            for metric in run.sf.PRIMARY:
                base = np.array([1., 1., 4.]) if metric != run.sf.PCC else np.array([.9, .9, .8])
                delta = np.array([.1, .1, .5]) if not condition.startswith("FULL") else np.zeros(3)
                frame[metric] = base + (-delta if metric == run.sf.PCC else delta)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_paired_summary_distinguishes_subject_and_window_weights():
    paired, subject, seeds, summary = run.paired_tables(metric_fixture())
    row = summary[(summary.condition == "H_SHIFT_1__FIXED") & (summary.metric == run.sf.PCC)]
    assert row[row.aggregation == "window"].iloc[0].deterioration == pytest.approx(.7/3)
    assert row[row.aggregation == "subject_macro"].iloc[0].deterioration == pytest.approx(.3)
    assert (row.worse_seeds == 3).all()
    assert len(paired) == 3*10*5*3


@pytest.mark.parametrize("corruption", ["seed", "condition", "duplicate", "nan", "eligibility"])
def test_summary_rejects_incomplete_or_corrupt_pairs(corruption):
    metrics = metric_fixture()
    if corruption == "seed":
        metrics = metrics[metrics.seed != core.SEEDS[-1]]
    elif corruption == "condition":
        metrics = metrics[metrics.condition != core.CONDITIONS[-1]]
    elif corruption == "duplicate":
        metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    elif corruption == "nan":
        metrics.loc[4, run.sf.PRIMARY[0]] = np.nan
    else:
        metrics.loc[4, run.sf.ELIGIBILITY[run.sf.PRIMARY[0]]] = False
    with pytest.raises((ValueError, FloatingPointError)):
        run.paired_tables(metrics)


def test_gate_precedes_loading_and_data_access(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"contract": {"split": "test"}}))
    monkeypatch.setattr(run, "load_manifest", lambda _: pytest.fail("门控前不得加载来源"))
    with pytest.raises(PermissionError):
        run.evaluate(tmp_path, core.SEEDS[0], "cuda:0")


def test_summary_rejects_cross_seed_population_change():
    metrics = metric_fixture()
    metrics.loc[metrics.seed == core.SEEDS[-1], "samp_id"] += 100
    with pytest.raises(ValueError, match="跨 seed"):
        run.paired_tables(metrics)


def test_signal_association_records_constant_case_ineligibility(tmp_path):
    cfg = run.training.config("M4", core.SEEDS[0], tmp_path / "unused")
    t = np.arange(18000) / 100
    signal = (1+.3*np.sin(2*np.pi*.02*t))*np.sin(2*np.pi*.25*t)
    records = run.signal_diagnostic(np.ones((97, 360)), signal, cfg, frequencies(), core.row_shifts([0])[0])
    assert len(records) == 4
    assert all(not r["valid"] and np.isnan(r["signed_r"]) and r["reason"] for r in records)


def test_lifecycle_retains_failure_and_rejects_corruption(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    parent = tmp_path / "evaluation"
    with pytest.raises(RuntimeError):
        with run.attempt(tmp_path, parent) as failed:
            raise RuntimeError("synthetic failure")
    assert (failed / "failed.json").exists()
    with pytest.raises(RuntimeError, match="retry"):
        run.completed(tmp_path, parent)
    assert run.completed(tmp_path, parent, retry=True) is None
    with run.attempt(tmp_path, parent) as success:
        (success / "data.txt").write_text("original")
    assert run.completed(tmp_path, parent) == success
    (success / "data.txt").write_text("corrupted")
    with pytest.raises(ValueError, match="产物变化"):
        run.completed(tmp_path, parent)


def test_synthetic_inference_metrics_cases_signal_and_plots(tmp_path):
    # 完整 180 秒合成信号、两个 subject、一个尾批；不打开真实 index/cache。
    cfg = run.training.config("M4", core.SEEDS[0], tmp_path / "unused")
    t = torch.arange(18000) / 100
    samples = []
    for i in range(3):
        target = ((1 + .3*torch.sin(2*torch.pi*.02*t)) * torch.sin(2*torch.pi*.25*t))[None]
        w = torch.rand(97, 360) + torch.sin(2*torch.pi*.25*torch.arange(360)/2)[None]*.2
        samples.append({"x": target + .1*torch.randn_like(target), "target": target,
            "tf": {"w": w}, "meta": {"dataset_row_id": i, "samp_id": 10+i//2, "split": "val"}})
    rows = pd.DataFrame([v["meta"] for v in samples])
    assert core.select_cases(rows) == [0, 1, 2]
    loader = DataLoader(samples, batch_size=2, shuffle=False)
    with pytest.raises(ValueError, match="错序"):
        list(run.guarded_batches(loader, rows.iloc[::-1]))
    run.infer(model(), loader, rows, cfg, "cpu", tmp_path, frequencies(), core.SEEDS[0])
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert len(metrics) == 30
    assert len(list((tmp_path / "cases").glob("*.npz"))) == 30
    assert run.read_json(tmp_path / "checks.json")["batches"] == 2
    # 相同 fixture 复制 seed 标签，仅验证汇总/图件接口，不作科研重复。
    full = pd.concat([metrics.assign(seed=s) for s in core.SEEDS], ignore_index=True)
    _, subjects, _, _ = run.paired_tables(full)
    plot.plot_effects(subjects, tmp_path)
    plot.plot_signals(tmp_path, tmp_path)
    plot.plot_case(tmp_path, 0, core.SEEDS[0], frequencies(), cfg, tmp_path)
    assert (tmp_path / "subject_paired_effects.png").stat().st_size > 1000
    assert (tmp_path / "modulation_and_association.svg").stat().st_size > 1000
