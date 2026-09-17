"""四候选控制器的 synthetic CPU 验证，临时目录隔离全部训练产物。"""

from __future__ import annotations

import copy
import json
import shutil
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
from resp_train.crd.config import load_crd_config
from resp_train.paper_evidence import e4_aggregation_v2 as e4
from resp_train.paper_evidence.e4_scale_aggregation_engineering import synthetic_batch

REPO = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def frequencies():
    path = REPO / "docs/experiments/e4_w0_scale_aggregation_source_audit_20260917.json"
    return json.loads(path.read_text())["frequency"]["values_hz"]


def baseline(seed=e4.SEEDS[0]):
    return load_crd_config(REPO / "configs/crd_tf_v1/crd_tf102_w_formal.yaml", overrides=[
        f"training.seed={seed}", f"model.initialization_seed={seed}"])


class TinyFixtureModel(nn.Module):
    def __init__(self, observed):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 5, padding=2)
        self.observed = observed

    def forward(self, x, **kwargs):
        if self.training and torch.is_grad_enabled():
            self.observed.append(x.shape[0])
        return {"waveform": x + self.conv(x)}


@pytest.fixture
def native_fixture(tmp_path, monkeypatch, frequencies):
    for module in (e4, e4.previous):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "COUNTS", {"train": 3, "val": 2})
    monkeypatch.setattr(e4, "EPOCHS", 2)
    monkeypatch.setattr(e4, "UPDATES_PER_EPOCH", 2)
    environment = {"fixture": "CPU", "git": {"commit": "synthetic"}}
    monkeypatch.setattr(e4, "runtime_preflight", lambda device: copy.deepcopy(environment))
    observed = []
    monkeypatch.setattr(e4, "build_model", lambda cfg: TinyFixtureModel(observed))
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
    templates = {arm: {seed: OmegaConf.to_container(e4.derived_config(OmegaConf.create(cfg), arm, frequencies,
                 output_root=tmp_path / e4.OUTPUT / "formal" / arm / f"seed_{seed}", device="cuda:0"), resolve=True)
                 for seed, cfg in baselines.items()} for arm in e4.ARMS}
    lock = {"protocol": e4.PROTOCOL, "arms": list(e4.ARMS), "seeds": list(e4.SEEDS), "counts": e4.COUNTS,
            "epochs": 2, "updates_per_epoch": 2, "contracts": {arm: e4.aggregation_contract(arm) for arm in e4.ARMS},
            "baselines": baselines, "resolved_templates": templates, "w0_entries": entries, "source_files": source_files,
            "frequency": e4.validate_frequency_grid(np.asarray(frequencies, dtype=np.float64)), "code_files": {}}
    original_load_lock = e4.load_lock
    monkeypatch.setattr(e4, "load_lock", lambda: (lock, DIGEST))

    def audit(lock, cfg, output):
        rows.to_csv(output / "val_rows.csv", index=False)
        e4.write_json(output / "access_receipt.json", {"fixture": True})
        return {"val": rows}

    monkeypatch.setattr(e4, "audit_sources", audit)
    with e4.attempt(tmp_path / "gpu", DIGEST, "gpu_acceptance") as gpu:
        e4.write_json(gpu / "environment.json", environment)
        e4.write_json(gpu / "access_receipt.json", {"fixture": True})
        e4.write_json(gpu / "gpu_acceptance.json", {"passed": True, "arms": list(e4.ARMS),
                       "seeds": list(e4.SEEDS), "physical_batch": 128})
    return SimpleNamespace(root=tmp_path, gpu=gpu, rows=rows, lock=lock, observed=observed, load_lock=original_load_lock)


def test_complete_twelve_run_native_matrix_summary_and_identity(native_fixture):
    fixture = native_fixture
    completed = []
    for arm in e4.ARMS:
        for seed in e4.SEEDS:
            start = len(fixture.observed)
            path = e4.run_formal(arm, seed, gpu_receipt=fixture.gpu, device="cpu")
            assert fixture.observed[start:] == [2, 1, 2, 1]
            receipt = json.loads((path / "formal_receipt.json").read_text())
            assert (receipt["arm"], receipt["seed"], receipt["epochs"], receipt["updates"]) == (arm, seed, 2, 4)
            assert receipt["quality_acceptance_passed"]
            history = pd.read_csv(path / receipt["run_dir"] / "train_history.csv")
            assert receipt["selected_epoch"] == int(history.loc[history.val_local_rr_mae.idxmin(), "epoch"])
            completed.append(path)
            if len(completed) == 1:
                with pytest.raises(ValueError):
                    e4.completed_runs()
    assert set(e4.completed_runs()) == set(completed)
    with pytest.raises(FileExistsError):
        e4.run_formal(e4.ARMS[0], e4.SEEDS[0], gpu_receipt=fixture.gpu, device="cpu")
    with pytest.raises(ValueError):
        e4.summarize(completed[:-1])
    with pytest.raises(ValueError):
        e4.summarize(completed[:-1] + [completed[0]])
    with pytest.raises(ValueError):
        e4.verify_attempt(completed[0], phase="formal", lock_hash="b" * 64)
    # 重签名仅限 disposable fixture：验证语义身份，而非只碰巧被文件哈希拦截。
    for wrong_arm in (False, True):
        with e4.attempt(fixture.root / "invalid_identity", DIGEST, "formal", e4.ARMS[0], e4.SEEDS[0]) as cloned:
            for source in completed[0].iterdir():
                if source.name.startswith("lifecycle_") or source.name in {"manifest.json", "freeze_receipt.json"}:
                    continue
                if source.is_dir():
                    shutil.copytree(source, cloned / source.name)
                else:
                    shutil.copy2(source, cloned / source.name)
            if wrong_arm:
                receipt_path = cloned / "formal_receipt.json"
                altered = json.loads(receipt_path.read_text())
                altered["arm"] = "other"
                receipt_path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="身份重复或越界"):
            e4.summarize(completed[:-1] + [cloned])
    output = e4.summarize(completed)
    combined = pd.read_csv(output / "seed_metrics.csv")
    assert len(combined) == 15
    assert len(pd.read_csv(output / "paired_seed_delta.csv")) == 60
    assert len(pd.read_csv(output / "four_arm_comparison.csv")) == 20
    assert set(combined.arm) == {"W0_FULL", *e4.ARMS}
    e4.verify_attempt(output, phase="summary", lock_hash=DIGEST)
    with pytest.raises(FileExistsError):
        e4.summarize(completed)


def test_history_checkpoint_optimizer_and_metric_tampering(native_fixture):
    fixture = native_fixture
    path = e4.run_formal(e4.ARMS[0], e4.SEEDS[0], gpu_receipt=fixture.gpu, device="cpu")
    receipt = json.loads((path / "formal_receipt.json").read_text())
    run = path / receipt["run_dir"]
    cfg = OmegaConf.load(run / "config.yaml")
    history = pd.read_csv(run / "train_history.csv")
    tied = history.assign(val_local_rr_mae=1.)
    assert e4.validate_history(tied, cfg) == 1
    for bad in (history.iloc[:-1], history.assign(optimizer_update=0), history.assign(train_loss_total=np.nan),
                history.assign(train_loss_total=history.train_loss_total + 1), history.assign(last_learning_rate=1.)):
        with pytest.raises((ValueError, FloatingPointError)):
            e4.validate_history(bad, cfg)
    checkpoint_path = run / "checkpoint_best_local_rr.pt"
    original = checkpoint_path.read_bytes()
    for corruption in ("epoch", "metric", "step", "nonfinite", "missing_parameter"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if corruption == "epoch":
            checkpoint["epoch"] = 99
        elif corruption == "metric":
            checkpoint["metrics"]["train_loss_total"] += 1
        elif corruption == "step":
            next(iter(checkpoint["optimizer_state_dict"]["state"].values()))["step"] += 1
        elif corruption == "nonfinite":
            next(iter(checkpoint["model_state_dict"].values())).flatten()[0] = np.nan
        else:
            checkpoint["model_state_dict"].pop(next(iter(checkpoint["model_state_dict"])))
        torch.save(checkpoint, checkpoint_path)
        with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
            e4.validate_run(run, cfg, fixture.rows)
        with pytest.raises(RuntimeError):
            e4.verify_attempt(path, phase="formal", lock_hash=DIGEST)
        checkpoint_path.write_bytes(original)
    metrics_path = run / "metrics.csv"
    original_metrics = metrics_path.read_bytes()
    metrics = pd.read_csv(metrics_path)
    bad_frames = [metrics.iloc[:-1], pd.concat([metrics.iloc[:1]] * 2), metrics.assign(arm="wrong"),
                  metrics.assign(seed=0), metrics.assign(**{e4.PCC: np.nan})]
    for bad in bad_frames:
        bad.to_csv(metrics_path, index=False)
        with pytest.raises((ValueError, FloatingPointError)):
            e4.validate_run(run, cfg, fixture.rows)
        metrics_path.write_bytes(original_metrics)
    groups_path = run / "optimizer_parameter_groups.json"
    groups = json.loads(groups_path.read_text())
    groups["decay"] = []
    groups_path.write_text(json.dumps(groups))
    with pytest.raises(ValueError, match="分组"):
        e4.validate_run(run, cfg, fixture.rows)


@pytest.mark.parametrize("key,value", [("model.aggregation_v2.hidden_channels", 9), ("model.aggregation_v2.arm", "wrong"),
    ("model.aggregation_v2.fill_hidden_channels", 64), ("loss.effort_weight", 0.), ("training.batch_size", 64),
    ("training.epochs", 79), ("training.early_stopping_enabled", True), ("training.max_learning_rate", .001),
    ("model.initialization_seed", 1), ("data.train_split", "test"), ("protocol.name", "legacy")])
def test_scientific_config_drift_is_rejected(tmp_path, frequencies, key, value):
    original = baseline()
    before = OmegaConf.to_container(original, resolve=True)
    cfg = e4.derived_config(original, e4.ARMS[0], frequencies, output_root=tmp_path, device="cpu")
    e4.validate_config(cfg, original, e4.ARMS[0], frequencies, output_root=tmp_path, device="cpu")
    assert OmegaConf.to_container(original, resolve=True) == before
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        e4.validate_config(cfg, original, e4.ARMS[0], frequencies, output_root=tmp_path, device="cpu")


def comparison_frame():
    records = []
    for arm in ("W0_FULL", *e4.ARMS):
        for index, seed in enumerate(e4.SEEDS):
            full = [1., 2., 4.][index]
            delta = 0 if arm == "W0_FULL" else [.1, -.2, 1.2][index]
            records.append({"arm": arm, "seed": seed, "split": "val", "selected_epoch": index + 1,
                "source_sha256": "a" * 64, **{key + "_mean": full + delta for key in e4.ERRORS},
                e4.PCC + "_mean": .8 if arm == "W0_FULL" else [.7, .9, .6][index],
                **{key + "_n": 2 for key in e4.PRIMARY}})
    return pd.DataFrame(records)


def test_paired_direction_sample_sd_and_zero_denominator_na(tmp_path):
    frame = comparison_frame()
    paired, aggregate = e4.paired_tables(frame, split="val")
    assert len(paired) == 60 and len(aggregate) == 20
    row = aggregate.iloc[0]
    assert row.paired_delta_mean == pytest.approx(10.)
    assert row.paired_delta_sample_sd == pytest.approx(20.)
    assert row.delta_of_seed_means == pytest.approx(100 * 1.1 / 7)
    assert (row.full_better_seeds, row.candidate_better_seeds) == (2, 1)
    pcc = paired.loc[paired.metric.eq(e4.PCC)].iloc[:3]
    assert pcc.unit.eq("absolute_drop").all()
    np.testing.assert_allclose(pcc.delta, [.1, -.1, .2])
    assert paired.full_n.eq(2).all() and paired.candidate_n.eq(2).all()
    frame.loc[0, e4.ERRORS[0] + "_mean"] = 0
    paired, aggregate = e4.paired_tables(frame, split="val")
    assert np.isnan(paired.iloc[0].delta) and paired.iloc[0].raw_delta == 1.1
    assert aggregate.iloc[0].relative_delta_defined_seeds == 2
    assert np.isnan(aggregate.iloc[0].paired_delta_mean)
    path = tmp_path / "paired.csv"
    paired.to_csv(path, index=False, na_rep="NA")
    assert ",NA," in path.read_text()


@pytest.mark.parametrize("corruption", ["missing_arm", "duplicate_seed", "wrong_arm", "wrong_split", "nonfinite", "negative_error"])
def test_pair_matrix_rejects_identity_or_metric_corruption(corruption):
    frame = comparison_frame()
    if corruption == "missing_arm":
        frame = frame[~frame.arm.eq(e4.ARMS[-1])]
    elif corruption == "duplicate_seed":
        frame.loc[4, "seed"] = frame.loc[3, "seed"]
    elif corruption == "wrong_arm":
        frame.loc[3, "arm"] = "other"
    elif corruption == "wrong_split":
        frame.loc[3, "split"] = "test"
    else:
        frame.loc[3, e4.ERRORS[0] + "_mean"] = np.nan if corruption == "nonfinite" else -1
    with pytest.raises((ValueError, FloatingPointError)):
        e4.paired_tables(frame, split="val")


def test_lock_matrix_and_frequency_drift_rejected(native_fixture):
    fixture = native_fixture
    path = fixture.root / e4.LOCK_PATH
    path.parent.mkdir(parents=True)
    e4.write_json(path, fixture.lock)
    fixture.load_lock(fixture.root)
    for key in ("arms", "frequency", "template"):
        lock = copy.deepcopy(fixture.lock)
        if key == "arms":
            lock["arms"] = lock["arms"][:-1]
        elif key == "frequency":
            lock["frequency"]["membership"][24] = 1
        else:
            lock["resolved_templates"][e4.ARMS[0]][str(e4.SEEDS[0])]["loss"]["effort_weight"] = 0
        path.write_text(json.dumps(lock))
        with pytest.raises(ValueError):
            fixture.load_lock(fixture.root)


def test_phase_mutex_failure_artifacts_and_invalid_run_identity(tmp_path):
    with e4.phase_guard(tmp_path, DIGEST):
        with pytest.raises(RuntimeError, match="正在运行"):
            with e4.phase_guard(tmp_path, DIGEST):
                pass
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with e4.attempt(tmp_path, DIGEST, "formal", e4.ARMS[0], e4.SEEDS[0]) as output:
            (output / "partial.txt").write_text("保留")
            raise RuntimeError("synthetic failure")
    assert (output / "partial.txt").read_text() == "保留"
    assert (output / "lifecycle_failed.json").exists()
    assert not (output / "freeze_receipt.json").exists()
    with pytest.raises(ValueError, match="失败"):
        e4.verify_attempt(output, phase="formal", lock_hash=DIGEST)
    for arm, seed in (("wrong", e4.SEEDS[0]), (e4.ARMS[0], 0)):
        with pytest.raises(ValueError, match="固定矩阵"):
            e4.run_formal(arm, seed, gpu_receipt=tmp_path, device="cpu")
