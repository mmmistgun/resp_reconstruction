"""独立测试集入口的 synthetic CPU 合同；所有数据及来源位于 disposable fixture。"""

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

from resp_train.crd.config import load_crd_config
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.paper_evidence import e4_aggregation_v2_test as test_eval

REPO = Path(__file__).resolve().parents[1]
EPOCHS = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 80]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, *, tf):
        assert tf["w"].shape[1:] == (97, 360)
        return {"waveform": x * self.gain}


def make_history(cfg, selected_epoch):
    epochs = np.arange(1, 81)
    history = pd.DataFrame({"epoch": epochs, "optimizer_update": epochs * 80,
                           "train_loss_total": .225, "train_loss_sync": .2, "train_loss_effort": .1,
                           "val_local_rr_mae": 1 + np.abs(epochs - selected_epoch) * .001})
    # selector 必须选择相同最优指标第一次出现的 epoch。
    if selected_epoch < 80:
        history.loc[history.epoch.eq(selected_epoch + 1), "val_local_rr_mae"] = 1.
    for key, indices in (("first_learning_rate", (epochs - 1) * 80), ("last_learning_rate", epochs * 80 - 1)):
        history[key] = [test_eval.training.crd_learning_rate(int(index), total_updates=6400,
                        max_learning_rate=cfg.training.max_learning_rate, min_learning_rate=cfg.training.min_learning_rate,
                        warmup_fraction=cfg.training.warmup_fraction) for index in indices]
    return history


@pytest.fixture
def fixture_world(tmp_path, monkeypatch):
    module = test_eval
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for helper in (module, module.reference_tools):
        monkeypatch.setattr(helper, "COUNT", 4)
        monkeypatch.setattr(helper, "SUBJECTS", 2)
    monkeypatch.setattr(module, "git_state", lambda *args, **kwargs: {"commit": "synthetic"})
    monkeypatch.setattr(module.training, "runtime_preflight", lambda device: {"git": {"commit": "synthetic"}, "device": "cpu"})
    monkeypatch.setattr(module, "build_model", lambda cfg: FixtureModel())
    cfg_path = REPO / "configs/crd_tf_v1/crd_tf102_w_formal.yaml"
    baselines = {str(seed): load_crd_config(cfg_path, overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"])
                 for seed in module.SEEDS}
    frequencies = json.loads((REPO / "docs/experiments/e4_w0_scale_aggregation_source_audit_20260917.json").read_text())["frequency"]["values_hz"]
    rows = pd.DataFrame({"dataset_row_id": np.arange(100, 104), "samp_id": [8, 8, 9, 9], "split": "test"})
    time = torch.arange(18000) / 100
    target = ((1 + .4 * torch.sin(2 * torch.pi * .025 * time)) * torch.sin(2 * torch.pi * .23 * time))[None]
    sensor = target + .05 * torch.sin(2 * torch.pi * .19 * time)[None]
    items = [{"x": sensor, "target": target, "tf": {"w": torch.zeros(97, 360)},
              "meta": {key: row[key] for key in module.IDENTITY_COLUMNS}} for _, row in rows.iterrows()]
    data = SimpleNamespace(rows=rows, dataset=items, loader=DataLoader(items, batch_size=3))
    monkeypatch.setattr(module, "read_research_v2_index", lambda *args, **kwargs: rows.copy())
    monkeypatch.setattr(module, "filter_index", lambda *args, **kwargs: rows.copy())
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(module, "CACHE_ROOT", cache)

    def build_data(cfg, **kwargs):
        assert cfg.data.tf_research_test_cache_path == str(cache)
        assert cfg.loss.effort_weight == .25
        arm = str(cfg.model.aggregation_v2.arm)
        assert OmegaConf.to_container(cfg.model.aggregation_v2, resolve=True) == module.aggregation_contract(arm)
        assert kwargs["split"] == "test" and kwargs["max_windows"] is None and kwargs["shuffle"] is False
        assert kwargs["sample_seed"] == 20260612
        return data

    monkeypatch.setattr(module, "build_window_data", build_data)
    index_path = tmp_path / "index.csv"
    rows.to_csv(index_path, index=False)
    cache_files = {}
    for filename in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy"):
        # 特意不是有效 npy，prepare 若越过 metadata 边界读取数组必然失败。
        (cache / filename).write_bytes(b"synthetic metadata-only bytes")
        cache_files[filename] = module.identity(cache / filename)
    monkeypatch.setattr(module, "FREQUENCY_FILE_SHA", cache_files["w_frequencies_hz.npy"]["sha256"])
    module.write_json(cache / "cache_manifest.json", {"splits": {"test": {"count": 4, "samp_id_count": 2,
        "row_ids_sha256": module.array_hash(rows.dataset_row_id.to_numpy())}}, "files": cache_files,
        "dataset_index": str(index_path), "dataset_index_sha256": module.sha256_file(index_path)})
    monkeypatch.setattr(module, "CACHE_MANIFEST_SHA", module.sha256_file(cache / "cache_manifest.json"))
    predictions = {"r_tho_hat": np.repeat(sensor.numpy()[None], 4, axis=0),
                   "tho_ref": np.repeat(target.numpy()[None], 4, axis=0),
                   **{key: rows[key].to_numpy() for key in module.IDENTITY_COLUMNS}}
    predictions["split"] = np.asarray(rows.split.tolist(), dtype=str)
    reference = evaluate_task_predictions(predictions, baselines[str(module.SEEDS[0])], include_test_only=False, method="crd_tf102_w")
    train_lock_path = tmp_path / module.training.LOCK_PATH
    train_lock_path.parent.mkdir(parents=True, exist_ok=True)
    module.write_json(train_lock_path, {"fixture": "training lock"})
    train_digest = module.sha256_file(train_lock_path)
    full_entries, audit_rows, sources = [], [], {}
    for seed in module.SEEDS:
        old_root = tmp_path / f"old_full_{seed}"
        old_root.mkdir()
        reference.to_csv(old_root / "research_test_metrics.csv", index=False)
        summarize_task_metrics(reference).to_csv(old_root / "research_test_metrics_summary.csv", index=False)
        module.write_json(old_root / "research_test_metrics_manifest.json", {"fixture": "old full", "seed": seed})
        full_entries.append({"seed": seed, "selected_epoch": 13, "checkpoint": {"sha256": "synthetic"}, "run_dir": old_root.name})
        audit_rows.append({"seed": seed, "variant": "crd_tf102_w", "checkpoint_sha256": "synthetic", "validation_selected_epoch": 13,
            "metrics_sha256": module.sha256_file(old_root / "research_test_metrics.csv"),
            "metrics_summary_sha256": module.sha256_file(old_root / "research_test_metrics_summary.csv"),
            "evaluation_manifest_sha256": module.sha256_file(old_root / "research_test_metrics_manifest.json")})
    for index, (arm, seed) in enumerate((arm, seed) for arm in module.ARMS for seed in module.SEEDS):
        epoch = EPOCHS[index]
        with module.training.attempt(tmp_path / "formal" / arm / str(seed), train_digest, "formal", arm, seed) as formal:
            run = formal / "training" / "fixture"
            run.mkdir(parents=True)
            cfg = module.training.derived_config(baselines[str(seed)], arm, frequencies, output_root=formal / "training", device="cpu")
            OmegaConf.save(cfg, run / "config.yaml")
            make_history(cfg, epoch).to_csv(run / "train_history.csv", index=False)
            torch.save({"epoch": epoch, "config": OmegaConf.to_container(cfg, resolve=True),
                        "model_state_dict": FixtureModel().state_dict()}, run / "checkpoint_best_local_rr.pt")
            for split, ids in (("train", [0, 1]), ("val", [2, 3])):
                pd.DataFrame({"samp_id": ids}).to_csv(formal / f"{split}_rows.csv", index=False)
            module.write_json(formal / "formal_receipt.json", {"arm": arm, "seed": seed, "epochs": 80, "updates": 6400,
                "selected_epoch": epoch, "run_dir": "training/fixture"})
            for filename in ("environment.json", "access_receipt.json", "implementation_lock.json"):
                module.write_json(formal / filename, {"fixture": "CPU"})
        sources[f"{arm}/{seed}"] = {"arm": arm, "seed": seed, "path": str(formal),
            "manifest": module.identity(formal / "manifest.json"), "selected_epoch": epoch}
    with module.training.attempt(tmp_path / module.training.OUTPUT / "summary", train_digest, "summary") as summary:
        module.write_json(summary / "summary_receipt.json", {"arms": list(module.ARMS), "seeds": list(module.SEEDS),
            "split": "val", "source_runs": sources, "implementation_lock_sha256": train_digest})
        for filename in ("seed_metrics.csv", "paired_seed_delta.csv", "four_arm_comparison.csv"):
            (summary / filename).write_text("synthetic fixture\n")
    (tmp_path / module.W0_AUDIT).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(tmp_path / module.W0_AUDIT, index=False)
    monkeypatch.setattr(module, "W0_AUDIT_SHA", module.sha256_file(tmp_path / module.W0_AUDIT))
    prior = {"baselines": {seed: OmegaConf.to_container(cfg, resolve=True) for seed, cfg in baselines.items()},
             "frequency": {"values_hz": frequencies}, "w0_entries": full_entries}
    monkeypatch.setattr(module.training, "load_lock", lambda root: (prior, train_digest))
    for relative in (module.SCRIPT_PATH, module.TEST_PATH, module.PROTOCOL_PATH, Path("resp_train/fixture.py")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic code or document\n")
    module.prepare_lock(summary, tmp_path)
    original_loader = module.load_lock
    monkeypatch.setattr(module, "load_lock", lambda: original_loader(tmp_path))
    return SimpleNamespace(root=tmp_path, summary=summary, rows=rows, reference=reference, data=data, lock=module.load_lock()[0])


def test_prepare_and_complete_twelve_checkpoint_test_matrix_preserves_sources(fixture_world):
    world = fixture_world
    assert [entry["selected_epoch"] for entry in world.lock["entries"]] == EPOCHS
    assert world.lock["count"] == 4 and world.lock["split"] == "test"
    hashes = {path: test_eval.sha256_file(Path(path)) for path in world.lock["source_files"]}
    with pytest.raises(FileExistsError):
        test_eval.prepare_lock(world.summary, world.root)
    runs = []
    for arm in test_eval.ARMS:
        for seed in test_eval.SEEDS:
            runs.append(test_eval.run_evaluation(arm, seed, "cpu"))
            if len(runs) == 1:
                with pytest.raises(ValueError):
                    test_eval.completed_runs()
    assert set(test_eval.completed_runs()) == set(runs)
    for run, entry in zip(runs, world.lock["entries"], strict=True):
        receipt = json.loads((run / "evaluation_receipt.json").read_text())
        assert (receipt["arm"], receipt["seed"], receipt["selected_epoch"], receipt["rows"]) == (
            entry["arm"], entry["seed"], entry["selected_epoch"], 4)
        assert (run / "access_started.json").is_file() and (run / "access_receipt.json").is_file()
    with pytest.raises(FileExistsError):
        test_eval.run_evaluation(test_eval.ARMS[0], test_eval.SEEDS[0], "cpu")
    with pytest.raises(ValueError):
        test_eval.summarize(runs[:-1])
    with pytest.raises(ValueError):
        test_eval.summarize(runs[:-1] + [runs[0]])
    output = test_eval.summarize(runs[::-1])
    for filename, count in (("seed_metrics.csv", 15), ("paired_seed_delta.csv", 60), ("four_arm_comparison.csv", 20)):
        frame = pd.read_csv(output / filename)
        assert len(frame) == count and frame.split.eq("test").all()
    assert json.loads((output / "summary_receipt.json").read_text())["new_metric_rows"] == 48
    assert all(test_eval.sha256_file(Path(path)) == digest for path, digest in hashes.items())
    with pytest.raises(FileExistsError):
        test_eval.summarize(runs)


@pytest.mark.parametrize("error", ["split", "row_order", "duplicate", "count", "subjects"])
def test_rows_reject_incompatible_test_identity(fixture_world, error):
    rows = fixture_world.rows.copy()
    if error == "split":
        rows["split"] = "val"
    elif error == "row_order":
        rows = rows.iloc[::-1]
    elif error == "duplicate":
        rows.loc[1, "dataset_row_id"] = rows.loc[0, "dataset_row_id"]
    elif error == "count":
        rows = rows.iloc[:-1]
    else:
        rows["samp_id"] = 8
    with pytest.raises(ValueError):
        test_eval.check_rows(rows, fixture_world.rows)


def test_nonfinite_and_eligibility_fail_but_degeneracy_is_retained(fixture_world):
    reference = fixture_world.reference
    for column, value, exception in [("envelope_trajectory_mae", np.nan, FloatingPointError),
                                     ("local_rr_target_eligible_windows", 0, ValueError)]:
        frame = reference.copy()
        frame.loc[0, column] = value
        with pytest.raises(exception):
            test_eval.check_metrics(frame, reference)
    frame = reference.copy()
    frame.loc[0, "joint_prediction_degenerate"] = True
    assert len(test_eval.check_metrics(frame, reference)) == 1
    assert test_eval.quality_flags(frame)["quality_acceptance_passed"] is False


def test_input_target_finite_and_tail_batch_identity(fixture_world):
    batches = list(fixture_world.data.loader)
    assert [len(batch["x"]) for batch in test_eval.guarded_batches(batches, fixture_world.rows)] == [3, 1]
    batches[-1]["target"][0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        list(test_eval.guarded_batches(batches, fixture_world.rows))


@pytest.mark.parametrize("kind", ["frequency", "aggregation", "epoch", "missing_arm", "duplicate_seed", "wrong_arm"])
def test_allowlist_scientific_matrix_drift_fails(fixture_world, kind):
    path = fixture_world.root / test_eval.LOCK_PATH
    lock = json.loads(path.read_text())
    if kind == "frequency":
        lock["cache_files"]["w_frequencies_hz.npy"]["sha256"] = "bad"
    elif kind == "aggregation":
        lock["contracts"][test_eval.ARMS[0]]["regions"][0][1] = 24
    elif kind == "epoch":
        lock["entries"][0]["selected_epoch"] = 81
    elif kind == "missing_arm":
        lock["entries"] = lock["entries"][:-3]
    elif kind == "duplicate_seed":
        lock["entries"][1]["seed"] = lock["entries"][0]["seed"]
    else:
        lock["entries"][0]["arm"] = "other"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        test_eval.load_lock()


@pytest.mark.parametrize("drift", ["checkpoint_bytes", "cache_bytes", "epoch", "model_config", "state", "scientific_arm"])
def test_drift_fails_before_index_or_waveform_access(fixture_world, monkeypatch, drift):
    world = fixture_world
    lock_path = world.root / test_eval.LOCK_PATH
    lock = json.loads(lock_path.read_text())
    entry = lock["entries"][0]
    checkpoint_path = Path(entry["candidate"]["checkpoint_best_local_rr.pt"]["path"])
    if drift == "cache_bytes":
        (Path(lock["cache_root"]) / "test_w.npy").write_bytes(b"changed synthetic cache")
    elif drift == "checkpoint_bytes":
        checkpoint_path.write_bytes(b"changed synthetic checkpoint")
    elif drift == "scientific_arm":
        config_path = Path(entry["candidate"]["config.yaml"]["path"])
        cfg = OmegaConf.load(config_path)
        cfg.model.aggregation_v2.arm = test_eval.ARMS[1]
        OmegaConf.save(cfg, config_path)
        entry["training_config"] = OmegaConf.to_container(cfg, resolve=True)
        updated = test_eval.identity(config_path)
        entry["candidate"]["config.yaml"].update(updated)
        lock["source_files"][str(config_path)] = updated
        lock_path.write_text(json.dumps(lock))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if drift == "epoch":
            checkpoint["epoch"] = 80
        elif drift == "model_config":
            checkpoint["config"]["model"]["aggregation_v2"]["fill_hidden_channels"] = 64
        else:
            checkpoint["model_state_dict"] = {}
        torch.save(checkpoint, checkpoint_path)
        updated = test_eval.identity(checkpoint_path)
        entry["candidate"]["checkpoint_best_local_rr.pt"].update(updated)
        lock["source_files"][str(checkpoint_path)] = updated
        lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(test_eval, "read_research_v2_index", lambda *args, **kwargs: pytest.fail("must not read index"))
    monkeypatch.setattr(test_eval, "build_window_data", lambda *args, **kwargs: pytest.fail("must not read waveforms"))
    with pytest.raises((ValueError, RuntimeError)):
        test_eval.run_evaluation(test_eval.ARMS[0], test_eval.SEEDS[0], "cpu")
    parent = world.root / test_eval.OUTPUT / "evaluation" / test_eval.ARMS[0] / f"seed_{test_eval.SEEDS[0]}"
    outputs = [path for path in parent.iterdir() if path.is_dir()]
    assert len(outputs) == 1 and (outputs[0] / "lifecycle_failed.json").exists()
    assert (outputs[0] / "access_started.json").exists()
    assert not (outputs[0] / "freeze_receipt.json").exists()


def test_development_subject_overlap_fails_before_waveforms(fixture_world, monkeypatch):
    path = fixture_world.root / test_eval.LOCK_PATH
    lock = json.loads(path.read_text())
    lock["entries"][0]["development_samp_ids"].append(8)
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(test_eval, "build_window_data", lambda *args, **kwargs: pytest.fail("must not read waveforms"))
    with pytest.raises(ValueError, match="samp_id 交叉"):
        test_eval.run_evaluation(test_eval.ARMS[0], test_eval.SEEDS[0], "cpu")


def test_concurrent_evaluation_is_rejected(fixture_world):
    _, digest = test_eval.load_lock()
    parent = fixture_world.root / test_eval.OUTPUT / "evaluation" / test_eval.ARMS[0] / f"seed_{test_eval.SEEDS[0]}"
    with test_eval.training.phase_guard(parent, digest):
        with pytest.raises(RuntimeError, match="正在运行"):
            test_eval.run_evaluation(test_eval.ARMS[0], test_eval.SEEDS[0], "cpu")


def test_cache_override_preserves_training_science(fixture_world):
    entry = fixture_world.lock["entries"][0]
    cfg, data_cfg = test_eval.evaluation_config(entry, "cpu", fixture_world.lock["cache_root"])
    assert cfg.loss.effort_weight == .25 and cfg.data.get("tf_research_test_cache_path") is None
    assert data_cfg.data.tf_research_test_cache_path == fixture_world.lock["cache_root"]
    assert data_cfg.model == cfg.model and data_cfg.loss == cfg.loss
    wrong = copy.deepcopy(entry)
    wrong["training_config"]["loss"]["effort_weight"] = 0.
    with pytest.raises(ValueError):
        test_eval.evaluation_config(wrong, "cpu", fixture_world.lock["cache_root"])


@pytest.mark.parametrize("drift", ["missing_slot", "duplicate_source", "wrong_epoch"])
def test_prepare_rejects_resigned_disposable_source_matrix(fixture_world, monkeypatch, drift):
    world = fixture_world
    receipt_path = world.summary / "summary_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    slots = list(receipt["source_runs"])
    if drift == "missing_slot":
        receipt["source_runs"].pop(slots[-1])
    elif drift == "duplicate_source":
        receipt["source_runs"][slots[1]] = copy.deepcopy(receipt["source_runs"][slots[0]])
    else:
        receipt["source_runs"][slots[0]]["selected_epoch"] = 79
    receipt_path.write_text(json.dumps(receipt))
    # 对临时来源重新签名，确保验证的是矩阵语义而非旧哈希。
    manifest_path = world.summary / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["summary_receipt.json"] = test_eval.identity(receipt_path)
    manifest_path.write_text(json.dumps(manifest))
    freeze_path = world.summary / "freeze_receipt.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["manifest"] = test_eval.identity(manifest_path)
    freeze_path.write_text(json.dumps(freeze))
    monkeypatch.setattr(test_eval, "LOCK_PATH", Path("docs/experiments/invalid_fixture_lock.json"))
    monkeypatch.setattr(test_eval, "build_window_data", lambda *args, **kwargs: pytest.fail("must not read waveforms"))
    with pytest.raises(ValueError):
        test_eval.prepare_lock(world.summary, world.root)
