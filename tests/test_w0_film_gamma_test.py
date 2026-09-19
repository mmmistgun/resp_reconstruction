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

from resp_train.crd import w0_film_gamma_training as training
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.paper_evidence import w0_film_gamma_test as test_eval


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


class FixtureModel(nn.Module):
    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, x, *, tf):
        assert tf["w"].shape[1:] == (97, 360)
        return {"waveform": x * self.gain}


def _history(selected_epoch: int) -> pd.DataFrame:
    epochs = np.arange(1, 81)
    return pd.DataFrame(
        {
            "epoch": epochs,
            "optimizer_update": epochs * 80,
            "val_local_rr_mae": 1.0 + np.abs(epochs - selected_epoch) * 0.001,
        }
    )


@pytest.fixture
def fixture_world(tmp_path, monkeypatch):
    module = test_eval
    monkeypatch.setattr(module, "CODE_ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(module, "COUNT", 4)
    monkeypatch.setattr(module, "SUBJECTS", 2)
    monkeypatch.setattr(module, "ROW_ORDER_SHA", "placeholder")
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(module, "LOCK_PATH", tmp_path / "docs/test_lock.json")
    monkeypatch.setattr(module, "PROTOCOL_PATH", tmp_path / "docs/protocol.md")
    monkeypatch.setattr(module, "TRAIN_RESULTS_PATH", tmp_path / "docs/results.md")
    monkeypatch.setattr(module, "SCRIPT_PATH", tmp_path / "scripts/eval.py")
    monkeypatch.setattr(module, "TEST_PATH", tmp_path / "tests/test_eval.py")
    for path in (
        module.PROTOCOL_PATH,
        module.TRAIN_RESULTS_PATH,
        module.SCRIPT_PATH,
        module.TEST_PATH,
        tmp_path / "resp_train/fixture.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source\n", encoding="utf-8")

    monkeypatch.setattr(
        module.training,
        "_git_state",
        lambda require_clean: {"commit": "synthetic", "status_porcelain": "", "dirty": False},
    )
    monkeypatch.setattr(module, "runtime_preflight", lambda device: {"device": "cpu", "git": {"commit": "synthetic"}})
    monkeypatch.setattr(module, "build_crd_model", lambda cfg: FixtureModel(0.98))
    monkeypatch.setattr(module.training, "load_resolved_training_config", lambda path: OmegaConf.load(path))

    rows = pd.DataFrame(
        {
            "dataset_row_id": np.arange(100, 104, dtype=np.int64),
            "samp_id": [8, 8, 9, 9],
            "split": ["test"] * 4,
        }
    )
    row_hash = module.array_hash(rows["dataset_row_id"].to_numpy())
    monkeypatch.setattr(module, "ROW_ORDER_SHA", row_hash)
    development_rows = pd.DataFrame(
        {"dataset_row_id": [1, 2], "samp_id": [1, 2], "split": ["train", "val"]}
    )
    monkeypatch.setattr(module, "read_research_v2_index", lambda *args, **kwargs: development_rows.copy())

    def filtered(index, cfg, *, split, **kwargs):
        return rows.copy() if split == "test" else development_rows.loc[development_rows["split"].eq(split)].copy()

    monkeypatch.setattr(module, "filter_index", filtered)

    t = torch.arange(18000, dtype=torch.float32) / 100.0
    target = torch.sin(2 * torch.pi * 0.23 * t)[None]
    sensor = target + 0.05 * torch.sin(2 * torch.pi * 0.19 * t)[None]
    items = [
        {
            "x": sensor.clone(),
            "target": target.clone(),
            "tf": {"w": torch.zeros(97, 360)},
            "meta": {key: row[key] for key in module.IDENTITY_COLUMNS},
        }
        for _, row in rows.iterrows()
    ]
    data = SimpleNamespace(rows=rows, dataset=items, loader=DataLoader(items, batch_size=3))

    def build_data(cfg, **kwargs):
        assert cfg.data.tf_research_test_cache_path == str(cache)
        assert kwargs["split"] == "test" and kwargs["max_windows"] is None
        assert kwargs["shuffle"] is False and kwargs["sample_seed"] == 20260612
        return data

    monkeypatch.setattr(module, "build_window_data", build_data)

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(module, "FROZEN_RESEARCH_TEST_CACHE_ROOT", cache)
    monkeypatch.setattr(module, "FROZEN_RESEARCH_TEST_CACHE_IDENTITY", "synthetic-cache")
    cache_files = {}
    for name in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy"):
        (cache / name).write_bytes(b"not a decodable npy; metadata fixture only")
        cache_files[name] = training.identity(cache / name)
    index_path = tmp_path / "dataset_index.csv"
    rows.to_csv(index_path, index=False)
    cache_manifest = {
        "splits": {
            "test": {
                "count": 4,
                "samp_id_count": 2,
                "row_ids_sha256": row_hash,
            }
        },
        "files": cache_files,
        "dataset_index": str(index_path),
        "dataset_index_sha256": training.sha256_file(index_path),
    }
    training.write_json(cache / "cache_manifest.json", cache_manifest)
    monkeypatch.setattr(
        module,
        "FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256",
        training.sha256_file(cache / "cache_manifest.json"),
    )

    train_lock_path = tmp_path / "docs/training_lock.json"
    training.write_json(train_lock_path, {"synthetic": True})
    monkeypatch.setattr(module.training, "LOCK_PATH", train_lock_path)
    baseline_entries = []
    audit_rows = []
    matrix_rows = []
    receipt_by_path = {}
    baseline_predictions = {
        "r_tho_hat": np.repeat(sensor.numpy()[None], 4, axis=0),
        "tho_ref": np.repeat(target.numpy()[None], 4, axis=0),
        **{key: rows[key].to_numpy() for key in module.IDENTITY_COLUMNS},
    }
    baseline_predictions["split"] = np.asarray(rows["split"], dtype=str)
    base_cfg = training.load_training_config(
        role="formal", condition="GAMMA_040", seed=module.SEEDS[0]
    )
    reference = evaluate_task_predictions(
        baseline_predictions, base_cfg, include_test_only=False, method="crd_tf102_w"
    )
    for seed, epoch, baseline_epoch in zip(
        module.SEEDS, module.SELECTED_EPOCHS, module.BASELINE_EPOCHS, strict=True
    ):
        run = tmp_path / f"candidate_{seed}"
        run.mkdir()
        cfg = training.load_training_config(role="formal", condition="GAMMA_040", seed=seed)
        OmegaConf.save(cfg, run / "config.yaml")
        _history(epoch).to_csv(run / "train_history.csv", index=False)
        checkpoint = {
            "epoch": epoch,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_state_dict": FixtureModel(0.98).state_dict(),
        }
        torch.save(checkpoint, run / "checkpoint_best_local_rr.pt")
        (run / training.RECEIPT_NAME).write_text("{}\n", encoding="utf-8")
        receipt = {
            "condition": module.CONDITION,
            "seed": seed,
            "selected_epoch": epoch,
            "resolved_config": training.identity(run / "config.yaml"),
            "history": training.identity(run / "train_history.csv"),
            "best_checkpoint": training.identity(run / "checkpoint_best_local_rr.pt"),
        }
        receipt_by_path[run.resolve()] = receipt
        matrix_rows.append(
            {
                "condition": module.CONDITION,
                "seed": seed,
                "run_dir": str(run),
                "receipt_sha256": training.sha256_file(run / training.RECEIPT_NAME),
            }
        )

        baseline = tmp_path / f"baseline_{seed}"
        baseline.mkdir()
        reference.to_csv(baseline / "research_test_metrics.csv", index=False)
        summarize_task_metrics(reference).to_csv(
            baseline / "research_test_metrics_summary.csv", index=False
        )
        training.write_json(
            baseline / "research_test_metrics_manifest.json",
            {"fixture": True, "seed": seed},
        )
        baseline_entries.append(
            {
                "seed": seed,
                "selected_epoch": baseline_epoch,
                "checkpoint": {"sha256": "synthetic-checkpoint"},
                "run_dir": str(baseline),
            }
        )
        audit_rows.append(
            {
                "seed": seed,
                "variant": "crd_tf102_w",
                "checkpoint_sha256": "synthetic-checkpoint",
                "validation_selected_epoch": baseline_epoch,
                "metrics_sha256": training.sha256_file(
                    baseline / "research_test_metrics.csv"
                ),
                "metrics_summary_sha256": training.sha256_file(
                    baseline / "research_test_metrics_summary.csv"
                ),
                "evaluation_manifest_sha256": training.sha256_file(
                    baseline / "research_test_metrics_manifest.json"
                ),
            }
        )

    monkeypatch.setattr(
        module.training,
        "load_lock",
        lambda: ({"baseline_runs": baseline_entries}, module.TRAIN_LOCK_SHA),
    )
    monkeypatch.setattr(
        module.training,
        "validate_completed_run",
        lambda path, lock_hash: receipt_by_path[Path(path).resolve()],
    )

    summary = tmp_path / "training_summary"
    summary.mkdir()
    pd.DataFrame(matrix_rows).to_csv(summary / "run_matrix.csv", index=False)
    training.write_json(
        summary / "decision_receipt.json",
        {
            "matrix_complete": True,
            "formal_runs": 6,
            "decisions": [
                {
                    "condition": module.CONDITION,
                    "quality_candidate": True,
                    "tolerance_aware_pareto": True,
                }
            ],
        },
    )
    files = {
        name: training.identity(summary / name)
        for name in ("run_matrix.csv", "decision_receipt.json")
    }
    training.write_json(
        summary / "artifact_manifest.json",
        {"protocol": training.PROTOCOL, "files": files},
    )
    training.write_json(
        summary / "freeze_receipt.json",
        {"manifest": training.identity(summary / "artifact_manifest.json")},
    )
    monkeypatch.setattr(module, "TRAIN_SUMMARY", summary)
    monkeypatch.setattr(
        module,
        "TRAIN_SUMMARY_MANIFEST_SHA",
        training.sha256_file(summary / "artifact_manifest.json"),
    )

    audit_path = tmp_path / "w0_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    monkeypatch.setattr(module, "W0_AUDIT", audit_path)
    monkeypatch.setattr(module, "W0_AUDIT_SHA", training.sha256_file(audit_path))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("prepare must not decode cache"))

    module.prepare_lock()
    return SimpleNamespace(
        root=tmp_path,
        rows=rows,
        reference=reference,
        data=data,
        cache=cache,
        lock=module.load_lock()[0],
    )


def test_prepare_locks_three_checkpoints_without_decoding_cache(fixture_world):
    lock = fixture_world.lock
    assert [entry["selected_epoch"] for entry in lock["entries"]] == [13, 30, 14]
    assert lock["count"] == 4 and lock["samp_id_count"] == 2
    assert set(lock["cache_files"]) == {
        "test_w.npy",
        "test_row_ids.npy",
        "w_frequencies_hz.npy",
    }
    assert "test signal/cache arrays not loaded" in lock["preparation_access"]
    with pytest.raises(FileExistsError):
        test_eval.prepare_lock()


def test_complete_synthetic_test_pipeline_and_summary(fixture_world):
    runs = [test_eval.run_evaluation(seed, "cpu") for seed in test_eval.SEEDS]
    for run, seed, epoch in zip(runs, test_eval.SEEDS, test_eval.SELECTED_EPOCHS, strict=True):
        receipt = json.loads((run / "evaluation_receipt.json").read_text(encoding="utf-8"))
        assert (receipt["seed"], receipt["selected_epoch"], receipt["rows"]) == (
            seed,
            epoch,
            4,
        )
        assert receipt["condition"] == test_eval.CONDITION
    with pytest.raises(FileExistsError):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")
    with pytest.raises(ValueError):
        test_eval.summarize(runs[:2])
    output = test_eval.summarize(runs[::-1])
    assert len(pd.read_csv(output / "seed_metrics.csv")) == 6
    assert len(pd.read_csv(output / "paired_seed_delta.csv")) == 15
    assert len(pd.read_csv(output / "three_seed_comparison.csv")) == 5
    assert len(pd.read_csv(output / "subject_summary.csv")) == 3 * 2 * 5
    assert pd.read_csv(output / "seed_metrics.csv")["split"].eq("test").all()
    receipt = json.loads((output / "summary_receipt.json").read_text(encoding="utf-8"))
    assert receipt["new_metric_rows"] == 12
    with pytest.raises(FileExistsError):
        test_eval.summarize(runs)


@pytest.mark.parametrize("error", ["split", "order", "duplicate", "count", "subjects"])
def test_rows_reject_incompatible_identity(fixture_world, error):
    rows = fixture_world.rows.copy()
    if error == "split":
        rows["split"] = "val"
    elif error == "order":
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
    frame = reference.copy()
    frame.loc[0, "envelope_trajectory_mae"] = np.nan
    with pytest.raises(FloatingPointError):
        test_eval.check_metrics(frame, reference)
    frame = reference.copy()
    frame.loc[0, "local_rr_target_eligible_windows"] = 0
    with pytest.raises(ValueError):
        test_eval.check_metrics(frame, reference)
    frame = reference.copy()
    frame.loc[0, "joint_prediction_degenerate"] = True
    test_eval.check_metrics(frame, reference)
    assert test_eval.quality_flags(frame)["quality_acceptance_passed"] is False


def test_config_replays_gamma040_and_adds_only_cache_override(fixture_world):
    entry = fixture_world.lock["entries"][0]
    cfg, data_cfg = test_eval.evaluation_config(entry, "cpu", str(fixture_world.cache))
    assert cfg.model.film_gamma_coefficient == 0.4
    assert cfg.model.film_beta_coefficient == 0.5
    assert cfg.data.get("tf_research_test_cache_path") is None
    assert data_cfg.data.tf_research_test_cache_path == str(fixture_world.cache)
    wrong = copy.deepcopy(entry)
    wrong["training_config"]["model"]["film_gamma_coefficient"] = 0.3
    with pytest.raises(ValueError):
        test_eval.evaluation_config(wrong, "cpu", str(fixture_world.cache))


def test_checkpoint_corruption_fails_before_prediction(fixture_world, monkeypatch):
    entry = fixture_world.lock["entries"][0]
    Path(entry["candidate"]["checkpoint"]["path"]).write_bytes(b"corrupt checkpoint")
    monkeypatch.setattr(
        test_eval,
        "collect_predictions",
        lambda *args, **kwargs: pytest.fail("must not infer"),
    )
    with pytest.raises(RuntimeError, match="身份漂移"):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")
    parent = fixture_world.root / "outputs/evaluation" / f"seed_{test_eval.SEEDS[0]}"
    attempts = [path for path in parent.iterdir() if path.is_dir()]
    assert len(attempts) == 1 and (attempts[0] / "lifecycle_failed.json").is_file()


def test_cache_drift_fails_before_dataset_access(fixture_world, monkeypatch):
    (fixture_world.cache / "test_w.npy").write_bytes(b"changed")
    monkeypatch.setattr(
        test_eval,
        "build_window_data",
        lambda *args, **kwargs: pytest.fail("must not access dataset"),
    )
    with pytest.raises(RuntimeError, match="身份漂移"):
        test_eval.run_evaluation(test_eval.SEEDS[0], "cpu")


def test_guarded_batches_checks_tail_and_finite(fixture_world):
    batches = list(fixture_world.data.loader)
    assert [len(batch["x"]) for batch in test_eval.guarded_batches(batches, fixture_world.rows)] == [
        3,
        1,
    ]
    batches[-1]["target"][0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        list(test_eval.guarded_batches(batches, fixture_world.rows))


def test_paired_delta_signs_are_metric_specific() -> None:
    rows = []
    for seed in test_eval.SEEDS:
        for condition, error, pcc in (
            (test_eval.BASELINE, 1.0, 0.8),
            (test_eval.CONDITION, 0.9, 0.82),
        ):
            row = {"seed": seed, "condition": condition}
            for metric in test_eval.ERROR_METRICS:
                row[f"{metric}_mean"] = error
            row[f"{test_eval.PCC_METRIC}_mean"] = pcc
            rows.append(row)
    paired, aggregate = test_eval._paired_tables(pd.DataFrame(rows))
    errors = paired.loc[paired["metric"].isin(test_eval.ERROR_METRICS)]
    pcc = paired.loc[paired["metric"].eq(test_eval.PCC_METRIC)]
    assert np.allclose(errors["delta"].to_numpy(), -10.0, atol=1e-12, rtol=0.0)
    assert np.allclose(pcc["delta"].to_numpy(), 0.02, atol=1e-12, rtol=0.0)
    assert len(paired) == 15 and len(aggregate) == 5
