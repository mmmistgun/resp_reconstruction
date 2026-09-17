from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from resp_train.crd import experiment as native
from resp_train.crd import tf_v1_model as tf_model
from resp_train.crd.config import load_crd_config
from resp_train.crd.initialization import module_seed
from resp_train.crd.training import partition_weight_decay_parameters
from resp_train.paper_evidence import e4_scale_aggregation as e4
from resp_train.paper_evidence.e4_scale_aggregation_model import (
    FourRegionAggregation, E4CwtBranch, E4ScaleAggregationModel, REGIONS, validate_frequency_grid, build_e4_model,
)
from resp_train.paper_evidence.e4_scale_aggregation_engineering import synthetic_batch


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def baseline(seed=e4.SEEDS[0]):
    root = Path(__file__).resolve().parents[1]
    return load_crd_config(root / "configs/crd_tf_v1/crd_tf102_w_formal.yaml", overrides=[
        f"training.seed={seed}", f"model.initialization_seed={seed}"])


def test_config_is_isolated_and_generic_loader_stays_strict(tmp_path):
    original = baseline()
    before = OmegaConf.to_container(original, resolve=True)
    cfg = e4.derived_config(original, output_root=tmp_path, device="cpu")
    e4.validate_config(cfg, original, output_root=tmp_path, device="cpu")
    assert OmegaConf.to_container(original, resolve=True) == before
    assert cfg.loss == original.loss and cfg.data == original.data and cfg.evaluation == original.evaluation
    OmegaConf.save(cfg, tmp_path / "e4.yaml")
    with pytest.raises(ValueError, match="冻结要求"):
        load_crd_config(tmp_path / "e4.yaml")
    with pytest.raises((ValueError, TypeError)):
        build_e4_model(original)


@pytest.mark.parametrize("key,value", [
    ("model.e4_aggregation.fill_hidden_channels", 64), ("model.e4_aggregation.channels", 48),
    ("model.e4_aggregation.regions", [[0, 24], [24, 48], [48, 72], [72, 97]]),
    ("training.batch_size", 64), ("loss.effort_weight", 0), ("training.epochs", 79),
    ("training.max_learning_rate", .001), ("training.early_stopping_enabled", True),
    ("model.initialization_seed", 1), ("data.train_split", "test"),
    ("evaluation.local_rr_step_sec", 30), ("protocol.name", "legacy"),
])
def test_scientific_drift_is_rejected(tmp_path, key, value):
    reference = baseline()
    cfg = e4.derived_config(reference, output_root=tmp_path, device="cpu")
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        e4.validate_config(cfg, reference, output_root=tmp_path, device="cpu")


def test_frequency_metadata_partition_and_duplicate_identity():
    # 来源文档中的频率元数据，不读取真实 cache、波形或 target。
    path = Path(__file__).resolve().parents[1] / "docs/experiments/e4_w0_scale_aggregation_source_audit_20260917.json"
    frequency = np.asarray(json.loads(path.read_text())["frequency"]["values_hz"], dtype=np.float64)
    info = validate_frequency_grid(frequency)
    assert info["regions"] == [list(pair) for pair in REGIONS]
    assert np.bincount(info["membership"]).tolist() == [25, 24, 24, 24]
    assert frequency[1] == frequency[2] and info["membership"][1] == info["membership"][2]
    for invalid in (frequency.astype(np.float32), frequency[::-1].copy(), frequency + 1e-8):
        with pytest.raises(ValueError):
            validate_frequency_grid(invalid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_projection_identity_and_all_new_parameter_gradients(dtype):
    module = FourRegionAggregation(7).to(dtype=dtype)
    x = torch.randn(1, 96, 97, 360, generator=torch.Generator().manual_seed(8), dtype=torch.float32).to(dtype).requires_grad_()
    output = module(x)
    assert torch.equal(output, x.mean(dim=2))
    output.float().square().mean().backward()
    assert module.projection.weight.grad.ne(0).all()
    assert torch.isfinite(module.projection.weight.grad).all() and torch.isfinite(x.grad).all()


def test_region_means_concat_order_and_no_boundary_loss():
    module = FourRegionAggregation(7)
    x = torch.zeros(1, 96, 97, 360)
    for k, (start, stop) in enumerate(REGIONS):
        x[:, :, start:stop] = k + 1
    # 输出通道 0 从区域 2 的通道 17 读取，检验区域/通道排列。
    with torch.no_grad():
        module.projection.weight[0, 2 * 96 + 17, 0] = 2
    actual = module(x)
    expected_mean = (25 + 24 * 2 + 24 * 3 + 24 * 4) / 97
    torch.testing.assert_close(actual[:, 0], torch.full_like(actual[:, 0], expected_mean + 6))
    torch.testing.assert_close(actual[:, 1:], torch.full_like(actual[:, 1:], expected_mean))
    for index in (0, 24, 25, 48, 49, 72, 73, 96):
        pulse = torch.zeros_like(x)
        pulse[:, 17, index] = 1
        result = module(pulse)
        expected = 2 / 24 if 49 <= index < 73 else 0
        torch.testing.assert_close(result[:, 0], torch.full_like(result[:, 0], expected))


@pytest.mark.parametrize("bad", ["shape", "nan", "inf", "weight_inf"])
def test_bad_aggregation_inputs_fail(bad):
    module = FourRegionAggregation(7)
    x = torch.zeros(1, 96, 97, 360)
    if bad == "shape":
        x = x[:, :, :49]
    elif bad == "weight_inf":
        with torch.no_grad():
            module.projection.weight[0, 0, 0] = float("inf")
    else:
        x[0, 0, 0, 0] = float(bad)
    with pytest.raises((ValueError, FloatingPointError)):
        module(x)


@pytest.mark.parametrize("seed", e4.SEEDS)
def test_shared_initialization_rng_parameters_and_optimizer_groups(seed):
    torch.manual_seed(71)
    start = torch.get_rng_state()
    reference = tf_model.CRDTfV1Model("crd_tf102_w", seed)
    ref_rng = torch.get_rng_state()
    torch.set_rng_state(start)
    candidate = E4ScaleAggregationModel(seed)
    assert torch.equal(torch.get_rng_state(), ref_rng)
    expected = reference.state_dict()
    actual = candidate.state_dict()
    assert set(actual) - set(expected) == {"branches.w.aggregation.projection.weight"}
    assert all(torch.equal(value, actual[name]) for name, value in expected.items())
    assert sum(p.numel() for p in candidate.parameters()) == 1256714
    assert candidate.branches["w"].parameter_fill.expand.out_channels == 65
    assert len(candidate.base.local_blocks) == 6
    full, new = map(partition_weight_decay_parameters, (reference, candidate))
    assert set(new.decay_names) - set(full.decay_names) == {"branches.w.aggregation.projection.weight"}
    assert new.no_decay_names == full.no_decay_names
    with pytest.raises(RuntimeError):
        candidate.load_state_dict(reference.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        reference.load_state_dict(candidate.state_dict(), strict=True)
    restored = E4ScaleAggregationModel(seed)
    restored.load_state_dict(actual, strict=True)


def test_branch_context_before_zero_film_and_multistep_gradient():
    seed = e4.SEEDS[0]
    with module_seed(seed, "tf_branch_w"):
        full = tf_model.CwtBranch()
    with module_seed(seed, "tf_branch_w"):
        candidate = E4CwtBranch(seed)
    context = {}
    hooks = [full.temporal.register_forward_pre_hook(lambda _, args: context.update(full=args[0].clone())),
             candidate.temporal.register_forward_pre_hook(lambda _, args: context.update(candidate=args[0].clone()))]
    w = {"w": torch.rand(1, 97, 360)}
    with torch.no_grad():
        full(w)
        candidate(w)
    assert torch.equal(context["full"], context["candidate"])
    for hook in hooks:
        hook.remove()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4)
    gradients = []
    target = torch.randn(1, 96, 1800)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        gamma, beta = candidate(w)
        ((gamma - target).square().mean() + (beta + target).square().mean()).backward()
        gradients.append([float(g.norm()) for g in candidate.aggregation.projection.weight.grad.split(96, dim=1)])
        optimizer.step()
    assert gradients[0] == [0.] * 4
    assert all(g > 0 for g in gradients[-1])
    assert candidate.aggregation.projection.weight.ne(0).any()


def test_chunking_nonzero_projection_preserves_output_and_gradient(monkeypatch):
    monkeypatch.setattr(tf_model, "TF_BRANCH_CHECKPOINT_BATCH_CHUNK", 1)
    with module_seed(9, "tf_branch_w"):
        direct = E4CwtBranch(9)
    with torch.no_grad():
        direct.aggregation.projection.weight.normal_(0, .005)
        direct.final_projection.weight.normal_(0, .005)
    chunked = copy.deepcopy(direct)
    data = {"w": torch.rand(2, 97, 360)}
    a, b = direct(data), tf_model._checkpointed_mapping_branch(chunked, data)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
    sum(value.square().mean() for value in a).backward()
    sum(value.square().mean() for value in b).backward()
    for x, y in zip(direct.parameters(), chunked.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=2e-6)


def comparison_frame():
    rows = []
    for arm in ("W0_FULL", e4.ARM):
        for i, seed in enumerate(e4.SEEDS):
            value = [1., 2., 4.][i]
            change = 0 if arm == "W0_FULL" else [0.1, -0.2, 1.2][i]
            rows.append({"arm": arm, "seed": seed, **{key + "_mean": value + change for key in e4.ERRORS},
                         e4.PCC + "_mean": .8 if arm == "W0_FULL" else [.7, .9, .6][i]})
    return pd.DataFrame(rows)


def test_paired_statistics_and_zero_denominator_keep_all_seeds():
    frame = comparison_frame()
    paired, aggregate = e4.paired_tables(frame)
    assert len(paired) == 15 and len(aggregate) == 5
    row = aggregate.iloc[0]
    assert row.paired_delta_mean == pytest.approx(10)
    assert row.paired_delta_sample_sd == pytest.approx(20)
    assert row.delta_of_seed_means == pytest.approx(100 * 1.1 / 7)
    assert row.full_better_seeds == 2 and row.candidate_better_seeds == 1
    frame.loc[0, e4.ERRORS[0] + "_mean"] = 0
    paired, aggregate = e4.paired_tables(frame)
    assert np.isnan(paired.iloc[0].delta) and paired.iloc[0].raw_delta == 1.1
    assert len(paired) == 15 and np.isnan(aggregate.iloc[0].paired_delta_mean)
    assert aggregate.iloc[0].relative_delta_defined_seeds == 2
    for invalid in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]), frame.assign(lag_aware_signed_pcc_mean=np.nan)):
        with pytest.raises((ValueError, FloatingPointError)):
            e4.paired_tables(invalid)


def test_lifecycle_preserves_failures_and_rejects_concurrent_execution(tmp_path):
    with e4.phase_guard(tmp_path, "a" * 64):
        with pytest.raises(RuntimeError, match="正在运行"):
            with e4.phase_guard(tmp_path, "a" * 64):
                pass
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with e4.attempt(tmp_path, "a" * 64, "formal", e4.SEEDS[0]) as path:
            (path / "partial.txt").write_text("保留")
            raise RuntimeError("synthetic failure")
    assert (path / "lifecycle_failed.json").exists() and (path / "partial.txt").read_text() == "保留"
    assert not (path / "freeze_receipt.json").exists()
    with pytest.raises(ValueError, match="失败"):
        e4.verify_attempt(path, phase="formal", lock_hash="a" * 64)


class SmallFixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 5, padding=2)

    def forward(self, x, **kwargs):
        return {"waveform": x + self.conv(x)}


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(e4, "ROOT", tmp_path)
    monkeypatch.setattr(e4, "EPOCHS", 2)
    monkeypatch.setattr(e4, "UPDATES_PER_EPOCH", 2)
    monkeypatch.setattr(e4, "COUNTS", {"train": 3, "val": 2})
    monkeypatch.setattr(e4, "runtime_preflight", lambda device: {"fixture": "CPU", "git": {"commit": "synthetic"}})
    monkeypatch.setattr(e4, "build_e4_model", lambda cfg: SmallFixtureModel())
    batch = synthetic_batch(3, 56)
    def samples(count, split):
        return [{"x": batch["x"][i], "target": batch["target"][i], "tf": {"w": batch["tf"]["w"][i]},
                 "meta": {"dataset_row_id": i, "samp_id": i, "split": split}} for i in range(count)]
    rows = pd.DataFrame({"dataset_row_id": [0, 1], "samp_id": [0, 1], "split": "val"})
    data = SimpleNamespace(train=SimpleNamespace(loader=DataLoader(samples(3, "train"), batch_size=2)),
                           val=SimpleNamespace(loader=DataLoader(samples(2, "val"), batch_size=2)),
                           audit_summary=pd.DataFrame({"fixture": [True]}))
    monkeypatch.setattr(native, "build_tho_data", lambda cfg: data)
    baselines, entries, source_files = {}, [], {}
    for seed in e4.SEEDS:
        cfg = baseline(seed)
        cfg.training.epochs = 2
        baselines[str(seed)] = OmegaConf.to_container(cfg, resolve=True)
        source = tmp_path / f"full_{seed}"
        source.mkdir()
        reference = rows.copy()
        for key in ("whole_rr_target_eligible", "local_rr_target_eligible", "joint_target_eligible"):
            reference[key] = True
        reference.to_csv(source / "metrics.csv", index=False)
        source_files[f"full_{seed}/metrics.csv"] = e4.identity(source / "metrics.csv")
        summary = {key + "_mean": [.5 if key == e4.PCC else 1.] for key in e4.PRIMARY}
        summary.update({key + "_n": [2] for key in e4.PRIMARY})
        pd.DataFrame(summary).to_csv(source / "metrics_summary.csv", index=False)
        entries.append({"seed": seed, "run_dir": source.name, "selected_epoch": 1,
                        "validation_summary": e4.identity(source / "metrics_summary.csv")})
    lock = {"baselines": baselines, "w0_entries": entries, "source_files": source_files}
    lock_hash = "a" * 64
    monkeypatch.setattr(e4, "load_lock", lambda: (lock, lock_hash))
    def audit(lock, cfg, output):
        rows.to_csv(output / "val_rows.csv", index=False)
        e4.write_json(output / "access_receipt.json", {"fixture": True})
        return {"val": rows}
    monkeypatch.setattr(e4, "audit_sources", audit)
    with e4.attempt(tmp_path / "gpu", lock_hash, "gpu_acceptance") as gpu:
        e4.write_json(gpu / "environment.json", {"fixture": "CPU", "git": {"commit": "synthetic"}})
        e4.write_json(gpu / "access_receipt.json", {"fixture": True})
        e4.write_json(gpu / "gpu_acceptance.json", {"passed": True, "aggregation_contract": e4.AGGREGATION_CONTRACT,
                       "seeds": list(e4.SEEDS), "physical_batch": 128})
    return tmp_path, gpu


def test_native_three_seed_training_selector_summary_and_immutability(native_fixture):
    root, gpu = native_fixture
    with pytest.raises(FileExistsError, match="阶段已完成"):
        with e4.phase_guard(gpu.parent, "a" * 64, completed_phase="gpu_acceptance"):
            pass
    completed = []
    for seed in e4.SEEDS:
        path = e4.run_formal(seed, gpu_receipt=gpu, device="cpu")
        receipt = json.loads((path / "formal_receipt.json").read_text())
        assert receipt["epochs"] == 2 and receipt["updates"] == 4
        assert receipt["quality_acceptance_passed"]
        run = path / receipt["run_dir"]
        history = pd.read_csv(run / "train_history.csv")
        assert receipt["selected_epoch"] == int(history.loc[history.val_local_rr_mae.idxmin(), "epoch"])
        completed.append(path)
    with pytest.raises(FileExistsError):
        e4.run_formal(e4.SEEDS[0], gpu_receipt=gpu, device="cpu")
    output = e4.summarize(completed)
    assert len(pd.read_csv(output / "seed_metrics.csv")) == 6
    assert len(pd.read_csv(output / "paired_seed_delta.csv")) == 15
    assert len(pd.read_csv(output / "three_seed_comparison.csv")) == 5
    with pytest.raises(FileExistsError):
        e4.summarize(completed)
    with pytest.raises(ValueError):
        e4.summarize(completed[:2])
    with pytest.raises(ValueError):
        e4.verify_attempt(completed[0], phase="formal", lock_hash="b" * 64)


def test_history_checkpoint_optimizer_and_metrics_reject_tampering(native_fixture):
    root, gpu = native_fixture
    path = e4.run_formal(e4.SEEDS[0], gpu_receipt=gpu, device="cpu")
    receipt = json.loads((path / "formal_receipt.json").read_text())
    run = path / receipt["run_dir"]
    cfg = OmegaConf.load(run / "config.yaml")
    history = pd.read_csv(run / "train_history.csv")
    tied = history.copy()
    tied["val_local_rr_mae"] = 1.
    assert e4.validate_history(tied, cfg) == 1
    with pytest.raises(ValueError):
        e4.validate_history(history.iloc[:1], cfg)
    drift = history.copy()
    drift.loc[0, "last_learning_rate"] *= 2
    with pytest.raises(ValueError, match="schedule"):
        e4.validate_history(drift, cfg)
    rows = pd.read_csv(path / "val_rows.csv")
    metrics = pd.read_csv(run / "metrics.csv")
    with pytest.raises(ValueError):
        e4.validate_metrics(metrics.iloc[:1], rows)
    with pytest.raises(ValueError):
        e4.validate_metrics(pd.concat([metrics.iloc[:1]] * 2), rows)
    metrics.loc[0, "joint_prediction_degenerate"] = True
    assert e4.validate_metrics(metrics, rows)["joint_prediction_degenerate"] == 1
    metrics.loc[0, e4.PCC] = float("nan")
    with pytest.raises(FloatingPointError):
        e4.validate_metrics(metrics, rows)
    checkpoint_path = run / "checkpoint_best_local_rr.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    next(iter(checkpoint["optimizer_state_dict"]["state"].values()))["step"] += 1
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="optimizer state/step"):
        e4.validate_run(run, cfg, rows)
    with pytest.raises(RuntimeError, match="身份漂移"):
        e4.verify_attempt(path, phase="formal", lock_hash="a" * 64)


def test_lock_configuration_and_frequency_drift_fail(tmp_path):
    audit_path = Path(__file__).resolve().parents[1] / "docs/experiments/e4_w0_scale_aggregation_source_audit_20260917.json"
    frequency = np.asarray(json.loads(audit_path.read_text())["frequency"]["values_hz"], dtype=np.float64)
    baselines = {str(seed): OmegaConf.to_container(baseline(seed), resolve=True) for seed in e4.SEEDS}
    templates = {seed: OmegaConf.to_container(e4.derived_config(OmegaConf.create(cfg),
                 output_root=tmp_path / e4.OUTPUT / "formal" / f"seed_{seed}", device="cuda:0"), resolve=True)
                 for seed, cfg in baselines.items()}
    lock = {"protocol": e4.PROTOCOL, "arm": e4.ARM, "seeds": list(e4.SEEDS), "counts": e4.COUNTS,
            "epochs": 80, "updates_per_epoch": 80, "aggregation_contract": e4.AGGREGATION_CONTRACT,
            "frequency": validate_frequency_grid(frequency), "code_files": {}, "baselines": baselines,
            "resolved_templates": templates}
    path = tmp_path / e4.LOCK_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(lock))
    e4.load_lock(tmp_path)
    lock["frequency"]["membership"][24] = 1
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        e4.load_lock(tmp_path)


def test_summary_denominator_and_anchor_eligibility_fail(native_fixture):
    root, gpu = native_fixture
    output = e4.run_formal(e4.SEEDS[0], gpu_receipt=gpu, device="cpu")
    receipt = json.loads((output / "formal_receipt.json").read_text())
    run = output / receipt["run_dir"]
    metrics = pd.read_csv(run / "metrics.csv")
    lock, _ = e4.load_lock()
    metrics.loc[0, "joint_target_eligible"] = False
    with pytest.raises(ValueError, match="eligibility"):
        e4.validate_anchor_rows(metrics, lock, e4.SEEDS[0])
    summary = pd.read_csv(run / "metrics_summary.csv")
    summary.loc[0, e4.ERRORS[0] + "_n"] = 1
    summary.to_csv(run / "metrics_summary.csv", index=False)
    with pytest.raises(ValueError, match="分母"):
        e4.validate_run(run, OmegaConf.load(run / "config.yaml"), pd.read_csv(output / "val_rows.csv"))


def test_parameter_report_matches_branch_convolution_accounting():
    # 固定特征 shape 下解析核对；这些值只代表分支卷积乘加。
    report = e4.parameter_compute_report()
    before = (720 + 432 + 48 * 96) * 97 * 360
    after = (3 * (96 * 5 + 96 * 192 + 192 * 96) + 96 * 65 + 65 * 96 + 96 * 192) * 1800
    assert report["branch_conv_only_macs_w0"] == before + after
    assert report["branch_conv_only_macs_e4"] == before + after + 384 * 96 * 360
    assert report["whole_model_flops"] is None
