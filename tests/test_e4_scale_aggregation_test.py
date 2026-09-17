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
from resp_train.paper_evidence import e4_scale_aggregation_test as test_eval


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
                            "train_loss_total": 0.225, "train_loss_sync": 0.2, "train_loss_effort": 0.1,
                            "val_local_rr_mae": 1 + np.abs(epochs - selected_epoch) * 0.001})
    for key, indices in (("first_learning_rate", (epochs - 1) * 80), ("last_learning_rate", epochs * 80 - 1)):
        history[key] = [test_eval.training.crd_learning_rate(int(i), total_updates=6400,
                         max_learning_rate=cfg.training.max_learning_rate, min_learning_rate=cfg.training.min_learning_rate,
                         warmup_fraction=cfg.training.warmup_fraction) for i in indices]
    return history


@pytest.fixture
def fixture_world(tmp_path, monkeypatch):
    """全部来源、checkpoint 与 cache 身份都是临时合成文件；不接触真实 test。"""
    module = test_eval
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "COUNT", 4)
    monkeypatch.setattr(module, "SUBJECTS", 2)
    monkeypatch.setattr(module, "git_state", lambda *a, **k: {"commit": "synthetic", "status_porcelain": ""})
    monkeypatch.setattr(module.training, "runtime_preflight", lambda device: {"git": {"commit": "synthetic"}, "device": "cpu"})
    monkeypatch.setattr(module, "build_e4_model", lambda cfg: FixtureModel())
    cfg_path = Path(__file__).resolve().parents[1] / "configs/crd_tf_v1/crd_tf102_w_formal.yaml"
    baselines = {str(seed): load_crd_config(cfg_path, overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"])
                 for seed in module.SEEDS}
    rows = pd.DataFrame({"dataset_row_id": np.arange(100, 104), "samp_id": [8, 8, 9, 9], "split": "test"})
    t = torch.arange(18000) / 100
    target = ((1 + 0.4 * torch.sin(2 * torch.pi * 0.025 * t)) * torch.sin(2 * torch.pi * 0.23 * t))[None]
    sensor = target + 0.05 * torch.sin(2 * torch.pi * 0.19 * t)[None]
    items = [{"x": sensor, "target": target, "tf": {"w": torch.zeros(97, 360)},
              "meta": {key: row[key] for key in module.IDENTITY_COLUMNS}} for _, row in rows.iterrows()]
    batch = SimpleNamespace(rows=rows, dataset=items, loader=DataLoader(items, batch_size=3))
    monkeypatch.setattr(module, "read_research_v2_index", lambda *a, **k: rows.copy())
    monkeypatch.setattr(module, "filter_index", lambda *a, **k: rows.copy())
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(module, "FROZEN_RESEARCH_TEST_CACHE_ROOT", cache)

    def build_data(cfg, **kwargs):
        assert cfg.data.tf_research_test_cache_path == str(cache)
        assert cfg.loss.effort_weight == 0.25
        assert OmegaConf.to_container(cfg.model.e4_aggregation, resolve=True) == module.AGGREGATION_CONTRACT
        assert kwargs["split"] == "test" and kwargs["max_windows"] is None and not kwargs["shuffle"]
        assert kwargs["sample_seed"] == 20260612
        return batch

    monkeypatch.setattr(module, "build_window_data", build_data)
    index_path = tmp_path / "index.csv"
    rows.to_csv(index_path, index=False)
    cache_files = {}
    for name in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy"):
        # 不是有效 npy；prepare 若尝试读取数组，测试必须失败。
        (cache / name).write_bytes(b"synthetic cached bytes; metadata access only")
        cache_files[name] = module.identity(cache / name)
    monkeypatch.setattr(module, "FREQUENCY_FILE_SHA", cache_files["w_frequencies_hz.npy"]["sha256"])
    cache_manifest = {"splits": {"test": {"count": 4, "samp_id_count": 2,
                       "row_ids_sha256": module.array_hash(rows.dataset_row_id.to_numpy())}}, "files": cache_files,
                       "dataset_index": str(index_path), "dataset_index_sha256": module.sha256_file(index_path)}
    module.write_json(cache / "cache_manifest.json", cache_manifest)
    monkeypatch.setattr(module, "FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256", module.sha256_file(cache / "cache_manifest.json"))
    predictions = {"r_tho_hat": np.repeat(sensor.numpy()[None], 4, axis=0),
                   "tho_ref": np.repeat(target.numpy()[None], 4, axis=0),
                   **{key: rows[key].to_numpy() for key in module.IDENTITY_COLUMNS}}
    predictions["split"] = np.asarray(rows["split"].tolist(), dtype=str)
    reference = evaluate_task_predictions(predictions, baselines[str(module.SEEDS[0])], include_test_only=False, method="crd_tf102_w")
    full_entries, audit_rows, sources = [], [], {}
    for seed, epoch in zip(module.SEEDS, module.SELECTED_EPOCHS, strict=True):
        with module.training.attempt(tmp_path / "formal" / str(seed), module.TRAIN_LOCK_SHA, "formal", seed) as formal:
            run = formal / "training" / "fixture"
            run.mkdir(parents=True)
            cfg = module.training.derived_config(baselines[str(seed)], output_root=formal / "training", device="cpu")
            OmegaConf.save(cfg, run / "config.yaml")
            make_history(cfg, epoch).to_csv(run / "train_history.csv", index=False)
            torch.save({"epoch": epoch, "config": OmegaConf.to_container(cfg, resolve=True),
                        "model_state_dict": FixtureModel().state_dict()}, run / "checkpoint_best_local_rr.pt")
            for split, ids in (("train", [0, 1]), ("val", [2, 3])):
                pd.DataFrame({"samp_id": ids}).to_csv(formal / f"{split}_rows.csv", index=False)
            module.write_json(formal / "formal_receipt.json", {"seed": seed, "epochs": 80, "updates": 6400,
                               "selected_epoch": epoch, "run_dir": "training/fixture"})
            module.write_json(formal / "environment.json", {"fixture": "CPU"})
            module.write_json(formal / "access_receipt.json", {"fixture": "CPU"})
            module.write_json(formal / "implementation_lock.json", {"fixture": "CPU"})
        sources[str(seed)] = {"path": str(formal), "manifest": module.identity(formal / "manifest.json"), "selected_epoch": epoch}
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
    with module.training.attempt(tmp_path / "summary", module.TRAIN_LOCK_SHA, "summary") as summary:
        module.write_json(summary / "summary_receipt.json", {"seeds": list(module.SEEDS), "source_runs": sources})
        for name in ("seed_metrics.csv", "paired_seed_delta.csv", "three_seed_comparison.csv"):
            (summary / name).write_text("synthetic fixture\n")
    monkeypatch.setattr(module, "SOURCE_SUMMARY", summary.relative_to(tmp_path))
    monkeypatch.setattr(module, "SOURCE_MANIFEST_SHA", module.sha256_file(summary / "manifest.json"))
    (tmp_path / module.W0_AUDIT).parent.mkdir(parents=True)
    pd.DataFrame(audit_rows).to_csv(tmp_path / module.W0_AUDIT, index=False)
    monkeypatch.setattr(module, "W0_AUDIT_SHA", module.sha256_file(tmp_path / module.W0_AUDIT))
    (tmp_path / module.training.LOCK_PATH).parent.mkdir(parents=True, exist_ok=True)
    module.write_json(tmp_path / module.training.LOCK_PATH, {"fixture": "training lock"})
    prior = {"baselines": {seed: OmegaConf.to_container(cfg, resolve=True) for seed, cfg in baselines.items()}, "w0_entries": full_entries}
    monkeypatch.setattr(module.training, "load_lock", lambda root: (prior, module.TRAIN_LOCK_SHA))
    for relative in (module.SCRIPT_PATH, module.TEST_PATH, module.PROTOCOL_PATH, Path("resp_train/fixture.py")):
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic code or document\n")
    module.prepare_lock(tmp_path)
    original_loader = module.load_lock
    monkeypatch.setattr(module, "load_lock", lambda: original_loader(tmp_path))
    return SimpleNamespace(root=tmp_path, rows=rows, reference=reference, data=batch, lock=module.load_lock()[0])


def test_prepare_locks_all_three_epochs_without_loading_test_arrays(fixture_world):
    lock = fixture_world.lock
    assert [e["selected_epoch"] for e in lock["entries"]] == [10, 5, 17]
    assert lock["count"] == 4 and lock["split"] == "test"
    assert set(lock["cache_files"]) == {"test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy"}
    with pytest.raises(FileExistsError):
        test_eval.prepare_lock(fixture_world.root)


def test_complete_test_pipeline_uses_three_fixed_checkpoints_and_frozen_full(fixture_world):
    hashes = {p: test_eval.sha256_file(Path(p)) for p in fixture_world.lock["source_files"]}
    runs = [test_eval.run_evaluation(seed, "cpu") for seed in test_eval.SEEDS]
    for run, seed, epoch in zip(runs, test_eval.SEEDS, test_eval.SELECTED_EPOCHS, strict=True):
        receipt = json.loads((run / "evaluation_receipt.json").read_text())
        assert (receipt["seed"], receipt["selected_epoch"], receipt["rows"]) == (seed, epoch, 4)
        assert (run / "access_started.json").is_file() and (run / "access_receipt.json").is_file()
    with pytest.raises(FileExistsError):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")
    with pytest.raises(ValueError):
        test_eval.summarize(runs[:2])
    with pytest.raises(ValueError):
        test_eval.summarize([runs[0], runs[0], runs[2]])
    output = test_eval.summarize(runs[::-1])
    assert len(pd.read_csv(output / "seed_metrics.csv")) == 6
    assert len(pd.read_csv(output / "paired_seed_delta.csv")) == 15
    assert pd.read_csv(output / "seed_metrics.csv").split.eq("test").all()
    assert pd.read_csv(output / "paired_seed_delta.csv").split.eq("test").all()
    assert pd.read_csv(output / "three_seed_comparison.csv").split.eq("test").all()
    receipt = json.loads((output / "summary_receipt.json").read_text())
    assert receipt["new_metric_rows"] == 12
    assert all(test_eval.sha256_file(Path(p)) == value for p, value in hashes.items())
    with pytest.raises(FileExistsError):
        test_eval.summarize(runs)


@pytest.mark.parametrize("error", ["split", "row_order", "duplicate", "count", "subjects"])
def test_rows_reject_incompatible_test_identity(fixture_world, error):
    rows = fixture_world.rows.copy()
    if error == "split": rows["split"] = "val"
    elif error == "row_order": rows = rows.iloc[::-1]
    elif error == "duplicate": rows.loc[1, "dataset_row_id"] = rows.loc[0, "dataset_row_id"]
    elif error == "count": rows = rows.iloc[:-1]
    else: rows["samp_id"] = 8
    with pytest.raises(ValueError):
        test_eval.check_rows(rows, fixture_world.rows)


def test_metric_nonfinite_and_target_eligibility_fail_but_degeneracy_is_retained(fixture_world):
    reference = fixture_world.reference
    for column, value, exception in [("envelope_trajectory_mae", np.nan, FloatingPointError),
                                     ("local_rr_target_eligible_windows", 0, ValueError)]:
        frame = reference.copy()
        frame.loc[0, column] = value
        with pytest.raises(exception):
            test_eval.check_metrics(frame, reference)
    frame = reference.copy()
    frame.loc[0, "joint_prediction_degenerate"] = True
    test_eval.check_metrics(frame, reference)
    assert test_eval.quality_flags(frame)["quality_acceptance_passed"] is False


def test_config_cache_override_preserves_training_science(fixture_world):
    entry = fixture_world.lock["entries"][0]
    cfg, data_cfg = test_eval.evaluation_config(entry, "cpu", fixture_world.lock["cache_root"])
    assert cfg.loss.effort_weight == 0.25 and cfg.data.get("tf_research_test_cache_path") is None
    assert data_cfg.data.tf_research_test_cache_path == fixture_world.lock["cache_root"]
    assert data_cfg.model == cfg.model and data_cfg.loss == cfg.loss
    wrong = copy.deepcopy(entry)
    wrong["training_config"]["loss"]["effort_weight"] = 0.0
    with pytest.raises(ValueError):
        test_eval.evaluation_config(wrong, "cpu", fixture_world.lock["cache_root"])


def test_checkpoint_corruption_fails_before_prediction_and_preserves_attempt(fixture_world, monkeypatch):
    entry = fixture_world.lock["entries"][0]
    Path(entry["candidate"]["checkpoint_best_local_rr.pt"]["path"]).write_bytes(b"corrupt synthetic checkpoint")
    monkeypatch.setattr(test_eval, "collect_predictions", lambda *a, **k: pytest.fail("must not infer"))
    with pytest.raises(RuntimeError, match="身份漂移"):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")
    parent = fixture_world.root / test_eval.OUTPUT / "evaluation" / f"seed_{test_eval.SEEDS[0]}"
    outputs = [p for p in parent.iterdir() if p.is_dir()]
    assert len(outputs) == 1 and (outputs[0] / "lifecycle_failed.json").exists()
    assert (outputs[0] / "access_started.json").exists()
    assert not (outputs[0] / "freeze_receipt.json").exists()


def test_test_cache_drift_fails_before_waveforms(fixture_world, monkeypatch):
    (Path(fixture_world.lock["cache_root"]) / "test_w.npy").write_bytes(b"changed synthetic cache")
    monkeypatch.setattr(test_eval, "build_window_data", lambda *a, **k: pytest.fail("must not read test waveforms"))
    with pytest.raises(RuntimeError, match="身份漂移"):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")


def test_input_target_finite_and_tail_batch_identity(fixture_world):
    batches = list(fixture_world.data.loader)
    assert [len(b["x"]) for b in test_eval.guarded_batches(batches, fixture_world.rows)] == [3, 1]
    batches[-1]["target"][0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        list(test_eval.guarded_batches(batches, fixture_world.rows))


def test_validation_output_cannot_enter_test_summary(fixture_world):
    from resp_train.paper_evidence.e4_scale_aggregation import attempt
    _, lock_hash = test_eval.load_lock()
    with attempt(fixture_world.root / "unrelated", lock_hash, "formal", test_eval.SEEDS[0]) as wrong:
        test_eval.write_json(wrong / "formal_receipt.json", {"fixture": True})
    with pytest.raises(ValueError, match="identity"):
        test_eval.verify_attempt(wrong, lock_hash, "evaluation")


def test_epoch_guard_rejects_nonselected_checkpoint_even_if_byte_hash_matches(fixture_world, monkeypatch):
    lock_path = fixture_world.root / test_eval.LOCK_PATH
    lock = json.loads(lock_path.read_text())
    entry = lock["entries"][0]
    checkpoint_path = Path(entry["candidate"]["checkpoint_best_local_rr.pt"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["epoch"] = 80
    torch.save(checkpoint, checkpoint_path)
    updated = test_eval.identity(checkpoint_path)
    entry["candidate"]["checkpoint_best_local_rr.pt"].update(updated)
    lock["source_files"][str(checkpoint_path)] = updated
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(test_eval, "collect_predictions", lambda *a, **k: pytest.fail("must not infer"))
    monkeypatch.setattr(test_eval, "build_window_data", lambda *a, **k: pytest.fail("must not read waveforms"))
    with pytest.raises(ValueError, match="epoch 漂移"):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")


def test_development_subject_overlap_is_rejected_before_waveform_access(fixture_world, monkeypatch):
    lock_path = fixture_world.root / test_eval.LOCK_PATH
    lock = json.loads(lock_path.read_text())
    lock["entries"][0]["development_samp_ids"].append(8)
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(test_eval, "build_window_data", lambda *a, **k: pytest.fail("must not read waveforms"))
    with pytest.raises(ValueError, match="samp_id 交叉"):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")


def test_e4_native_factory_and_scientific_identity(tmp_path):
    from resp_train.paper_evidence.e4_scale_aggregation_model import E4ScaleAggregationModel
    from resp_train.crd.model import build_crd_model
    path = Path(__file__).resolve().parents[1] / "configs/crd_tf_v1/crd_tf102_w_formal.yaml"
    cfg = test_eval.training.derived_config(load_crd_config(path), output_root=tmp_path, device="cpu")
    candidate = test_eval.build_e4_model(cfg)
    assert isinstance(candidate, E4ScaleAggregationModel)
    assert sum(p.numel() for p in candidate.parameters()) == 1256714
    assert candidate.branches["w"].aggregation.projection.weight.shape == (96, 384, 1)
    with pytest.raises(RuntimeError):
        candidate.load_state_dict(build_crd_model(cfg).state_dict(), strict=True)


def rewrite_checkpoint_identity(world, checkpoint):
    lock_path = world.root / test_eval.LOCK_PATH
    lock = json.loads(lock_path.read_text())
    entry = lock["entries"][0]
    path = Path(entry["candidate"]["checkpoint_best_local_rr.pt"]["path"])
    torch.save(checkpoint, path)
    identity = test_eval.identity(path)
    entry["candidate"]["checkpoint_best_local_rr.pt"].update(identity)
    lock["source_files"][str(path)] = identity
    lock_path.write_text(json.dumps(lock))


@pytest.mark.parametrize("drift", ["model_config", "state"])
def test_wrong_model_fails_before_real_data_even_with_matching_hash(fixture_world, monkeypatch, drift):
    entry = fixture_world.lock["entries"][0]
    checkpoint = torch.load(entry["candidate"]["checkpoint_best_local_rr.pt"]["path"], weights_only=False)
    if drift == "model_config":
        checkpoint["config"]["model"]["e4_aggregation"]["fill_hidden_channels"] = 64
    else:
        checkpoint["model_state_dict"] = {}
    rewrite_checkpoint_identity(fixture_world, checkpoint)
    monkeypatch.setattr(test_eval, "build_window_data", lambda *a, **k: pytest.fail("must not read test signals"))
    with pytest.raises((ValueError, RuntimeError)):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")


def test_concurrent_evaluation_is_rejected(fixture_world):
    _, digest = test_eval.load_lock()
    parent = fixture_world.root / test_eval.OUTPUT / "evaluation" / f"seed_{test_eval.SEEDS[0]}"
    with test_eval.training.phase_guard(parent, digest):
        with pytest.raises(RuntimeError, match="正在运行"):
            test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")


def test_test_lock_rejects_frequency_and_aggregation_drift(fixture_world):
    path = fixture_world.root / test_eval.LOCK_PATH
    original = json.loads(path.read_text())
    for kind in ("frequency", "aggregation", "epoch"):
        lock = copy.deepcopy(original)
        if kind == "frequency":
            lock["cache_files"]["w_frequencies_hz.npy"]["sha256"] = "bad"
        elif kind == "aggregation":
            lock["aggregation_contract"]["regions"][0][1] = 24
        else:
            lock["entries"][0]["selected_epoch"] = 11
        path.write_text(json.dumps(lock))
        with pytest.raises(ValueError):
            test_eval.load_lock()
