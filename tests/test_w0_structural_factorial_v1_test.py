from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence import w0_structural_factorial_v1_test as test_eval


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def synthetic_rows(monkeypatch):
    rows = pd.DataFrame(
        {
            "dataset_row_id": np.arange(100, 104, dtype=np.int64),
            "samp_id": [8, 8, 9, 9],
            "split": ["test"] * 4,
        }
    )
    monkeypatch.setattr(test_eval, "COUNT", len(rows))
    monkeypatch.setattr(test_eval, "SUBJECTS", rows.samp_id.nunique())
    monkeypatch.setattr(
        test_eval,
        "ROW_ORDER_SHA256",
        test_eval.formal.array_hash(rows.dataset_row_id.to_numpy()),
    )
    return rows


def synthetic_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    count = len(rows)
    metrics = rows.copy()
    metrics["whole_rr_abs_error_bpm"] = np.linspace(0.1, 0.4, count)
    metrics["local_rr_mae_bpm"] = np.linspace(0.2, 0.5, count)
    metrics["envelope_trajectory_mae"] = np.linspace(0.01, 0.04, count)
    metrics["global_envelope_modulation_error"] = np.linspace(0.02, 0.05, count)
    metrics["lag_aware_signed_pcc"] = np.linspace(0.7, 0.9, count)
    metrics["local_rr_prediction_valid_fraction"] = 1.0
    metrics["whole_rr_target_eligible"] = True
    metrics["local_rr_target_eligible"] = True
    metrics["local_rr_target_eligible_windows"] = 5
    metrics["joint_target_eligible"] = True
    metrics["envelope_spearman_target_eligible"] = True
    metrics["ibi_target_eligible"] = True
    metrics["joint_prediction_degenerate"] = False
    metrics["envelope_spearman_prediction_degenerate"] = False
    metrics["respiratory_band_coherence"] = np.linspace(0.5, 0.8, count)
    metrics["constrained_ndtw"] = np.linspace(0.2, 0.3, count)
    return metrics


def test_rows_and_metrics_accept_complete_synthetic_fixture(synthetic_rows):
    metrics = synthetic_metrics(synthetic_rows)
    test_eval.check_test_rows(synthetic_rows)
    summary = test_eval.check_test_metrics(metrics, synthetic_rows)
    assert int(summary.iloc[0].n_samples) == len(synthetic_rows)
    assert all(
        int(summary.iloc[0][metric + "_n"]) == len(synthetic_rows)
        for metric in sf.PRIMARY
    )
    assert test_eval.quality_flags(metrics)["quality_acceptance_passed"] is True


@pytest.mark.parametrize("drift", ["split", "order", "duplicate", "subject", "count"])
def test_rows_reject_identity_drift(synthetic_rows, drift):
    rows = synthetic_rows.copy()
    if drift == "split":
        rows.loc[0, "split"] = "val"
    elif drift == "order":
        rows = rows.iloc[::-1].reset_index(drop=True)
    elif drift == "duplicate":
        rows.loc[1, "dataset_row_id"] = rows.loc[0, "dataset_row_id"]
    elif drift == "subject":
        rows["samp_id"] = 8
    else:
        rows = rows.iloc[:-1]
    with pytest.raises(ValueError, match="research-test rows"):
        test_eval.check_test_rows(rows)


@pytest.mark.parametrize(
    ("column", "value", "exception"),
    [
        ("envelope_trajectory_mae", np.nan, FloatingPointError),
        ("respiratory_band_coherence", np.inf, FloatingPointError),
        ("constrained_ndtw", np.nan, FloatingPointError),
        ("joint_target_eligible", "invalid", ValueError),
        ("local_rr_target_eligible_windows", -1, ValueError),
        ("local_rr_target_eligible_windows", 0, ValueError),
    ],
)
def test_metrics_reject_nonfinite_and_target_drift(
    synthetic_rows, column, value, exception
):
    metrics = synthetic_metrics(synthetic_rows)
    if isinstance(value, str):
        metrics[column] = metrics[column].astype(object)
    metrics.loc[0, column] = value
    with pytest.raises(exception):
        test_eval.check_test_metrics(metrics, synthetic_rows)


def test_prediction_degeneracy_is_reported_without_dropping_rows(synthetic_rows):
    metrics = synthetic_metrics(synthetic_rows)
    metrics.loc[0, "joint_prediction_degenerate"] = True
    test_eval.check_test_metrics(metrics, synthetic_rows)
    quality = test_eval.quality_flags(metrics)
    assert quality["prediction_degeneracy"]["joint_prediction_degenerate"] == 1
    assert quality["quality_acceptance_passed"] is False


def test_guarded_batches_checks_tail_identity_shape_and_finite(synthetic_rows):
    items = [
        {
            "x": torch.full((1, 16), float(index)),
            "target": torch.full((1, 16), float(index + 1)),
            "tf": {"w": torch.zeros(97, 360)},
            "meta": {
                key: row[key]
                for key in test_eval.IDENTITY_COLUMNS
            },
        }
        for index, (_, row) in enumerate(synthetic_rows.iterrows())
    ]
    loader = DataLoader(items, batch_size=3, shuffle=False)
    batches = list(test_eval.guarded_batches(loader, synthetic_rows))
    assert [len(batch["x"]) for batch in batches] == [3, 1]

    bad = list(items)
    bad[-1] = {**bad[-1], "tf": {"w": torch.full((97, 360), float("nan"))}}
    with pytest.raises(FloatingPointError, match="非有限"):
        list(test_eval.guarded_batches(DataLoader(bad, batch_size=3), synthetic_rows))


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, *, tf):
        return {"waveform": x * self.gain}


def test_complete_evaluation_lifecycle_uses_only_synthetic_fixture(
    tmp_path, monkeypatch, synthetic_rows
):
    arm, seed = test_eval.ARMS[0], sf.SEEDS[0]
    formal_attempt = tmp_path / "formal"
    formal_attempt.mkdir()
    sf.write_json(formal_attempt / "environment.json", {"device": "cpu"})
    cfg = sf.derived_config(
        sf.load_w0_baseline(seed),
        arm=arm,
        output_root=formal_attempt / "training",
        device="cpu",
    )
    training_config = OmegaConf.to_container(cfg, resolve=True)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 7,
            "config": training_config,
            "model_state_dict": FixtureModel().state_dict(),
        },
        checkpoint_path,
    )
    entry = {
        "arm": arm,
        "seed": seed,
        "selected_epoch": 7,
        "formal_attempt": str(formal_attempt),
        "training_config": training_config,
        "development_samp_ids": [1, 2],
        "files": {
            "checkpoint": {
                "path": str(checkpoint_path),
                **sf.identity(checkpoint_path),
            }
        },
    }
    lock = {
        "entries": [entry],
        "cache": {"root": str(tmp_path / "cache")},
        "evidence_role": "synthetic fixture",
    }
    allowlist_path = tmp_path / "allowlist"
    allowlist_path.mkdir()
    sf.write_json(allowlist_path / "allowlist.json", {"fixture": True})
    monkeypatch.setattr(test_eval, "load_allowlist", lambda path: (lock, "b" * 64))
    monkeypatch.setattr(test_eval, "_research_test_root", lambda: tmp_path / "outputs")
    monkeypatch.setattr(
        test_eval.formal,
        "runtime_preflight",
        lambda device: {"device": device, "fixture": True},
    )
    monkeypatch.setattr(
        test_eval.formal,
        "_runtime_compatibility",
        lambda runtime, training: {"compatible": True},
    )
    monkeypatch.setattr(test_eval, "_verify_evaluation_sources", lambda lock, entry: None)
    monkeypatch.setattr(
        test_eval,
        "evaluation_config",
        lambda entry, lock, device: (cfg, OmegaConf.create(training_config)),
    )
    monkeypatch.setattr(test_eval, "build_w0_structural_factorial_model", lambda cfg: FixtureModel())
    monkeypatch.setattr(test_eval, "read_research_v2_index", lambda *args, **kwargs: synthetic_rows.copy())
    monkeypatch.setattr(test_eval, "filter_index", lambda *args, **kwargs: synthetic_rows.copy())
    items = [
        {
            "x": torch.ones(1, 16),
            "target": torch.ones(1, 16),
            "tf": {"w": torch.zeros(97, 360)},
            "meta": {key: row[key] for key in test_eval.IDENTITY_COLUMNS},
        }
        for _, row in synthetic_rows.iterrows()
    ]
    data = SimpleNamespace(
        rows=synthetic_rows.copy(),
        dataset=items,
        loader=DataLoader(items, batch_size=3, shuffle=False),
    )
    monkeypatch.setattr(test_eval, "build_window_data", lambda *args, **kwargs: data)

    def collect(model, loader, **kwargs):
        assert [len(batch["x"]) for batch in loader] == [3, 1]
        assert kwargs == {"device": "cpu", "max_windows": 4, "use_amp": True}
        return {"synthetic": True}

    monkeypatch.setattr(test_eval, "collect_predictions", collect)
    monkeypatch.setattr(
        test_eval,
        "evaluate_task_predictions",
        lambda predictions, cfg, **kwargs: synthetic_metrics(synthetic_rows),
    )
    output = test_eval.run_evaluation(
        allowlist_path,
        arm=arm,
        seed=seed,
        device="cpu",
    )
    manifest = test_eval.verify_attempt(
        output,
        phase="evaluation",
        evidence_hash="b" * 64,
        arm=arm,
        seed=seed,
    )
    assert manifest["status"] == "completed"
    receipt = pd.read_json(output / "evaluation_receipt.json", typ="series")
    assert int(receipt["rows"]) == len(synthetic_rows)
    assert bool(receipt["primary_finite"])
    assert len(pd.read_csv(output / "metrics.csv")) == len(synthetic_rows)


def test_evaluation_config_changes_only_runtime_and_cache(tmp_path, monkeypatch):
    arm = test_eval.ARMS[0]
    seed = sf.SEEDS[0]
    formal_attempt = tmp_path / "formal"
    output_root = formal_attempt / "training"
    baseline = sf.load_w0_baseline(seed)
    cfg = sf.derived_config(
        baseline,
        arm=arm,
        output_root=output_root,
        device="cpu",
    )
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)
    entry = {
        "arm": arm,
        "seed": seed,
        "formal_attempt": str(formal_attempt),
        "files": {"config": {"path": str(config_path), **sf.identity(config_path)}},
        "training_config": OmegaConf.to_container(cfg, resolve=True),
    }
    monkeypatch.setattr(
        test_eval.formal,
        "load_formal_contract",
        lambda: ({"baselines": {str(seed): OmegaConf.to_container(baseline, resolve=True)}}, "fixture"),
    )
    evaluated, data_cfg = test_eval.evaluation_config(
        entry,
        {"cache": {"root": str(tmp_path / "cache")}},
        device="cpu",
    )
    assert evaluated.loss == cfg.loss and evaluated.model == cfg.model
    assert evaluated.data.get("tf_research_test_cache_path") is None
    assert data_cfg.data.tf_research_test_cache_path == str(tmp_path / "cache")
    assert int(data_cfg.training.batch_size) == 128
    assert str(data_cfg.training.amp_dtype) == "bfloat16"


def test_evidence_attempt_is_immutable_and_verifiable(tmp_path):
    evidence_hash = "a" * 64
    parent = tmp_path / "allowlist"
    with test_eval.evidence_attempt(
        parent,
        phase="allowlist",
        evidence_hash=evidence_hash,
        reject_completed=True,
    ) as output:
        for filename in (
            "allowlist.json",
            "allowlist_receipt.json",
            "source_manifest.json",
            "access_receipt.json",
            "source_code.json",
        ):
            sf.write_json(output / filename, {"fixture": filename})
    manifest = test_eval.verify_attempt(
        output,
        phase="allowlist",
        evidence_hash=evidence_hash,
    )
    assert manifest["status"] == "completed"
    with pytest.raises(FileExistsError, match="已完成"):
        with test_eval.evidence_attempt(
            parent,
            phase="allowlist",
            evidence_hash=evidence_hash,
            reject_completed=True,
        ):
            pass


def test_allowlist_load_binds_preparation_identity(tmp_path, monkeypatch, synthetic_rows):
    arm, seed = test_eval.ARMS[0], sf.SEEDS[0]
    preparation_identity = "c" * 64
    monkeypatch.setattr(test_eval, "ARMS", (arm,))
    monkeypatch.setattr(test_eval.sf, "SEEDS", (seed,))
    monkeypatch.setattr(test_eval.formal, "P2_LOCK_SHA256", "formal-lock")
    monkeypatch.setattr(test_eval, "FROZEN_RESEARCH_TEST_CACHE_IDENTITY", "cache-id")
    allowlist = {
        "schema_version": 1,
        "protocol": test_eval.PROTOCOL,
        "allowlist_preparation_identity_sha256": preparation_identity,
        "arms": [arm],
        "seeds": [seed],
        "entries": [{"arm": arm, "seed": seed}],
        "split": "test",
        "count": len(synthetic_rows),
        "samp_id_count": synthetic_rows.samp_id.nunique(),
        "test_sample_seed": test_eval.TEST_SAMPLE_SEED,
        "row_order_sha256": test_eval.ROW_ORDER_SHA256,
        "batch_size": 128,
        "amp_dtype": "bfloat16",
        "include_test_only": True,
        "training_implementation_lock_sha256": "formal-lock",
        "cache": {"identity": "cache-id"},
        "code_files": {},
    }
    with test_eval.evidence_attempt(
        tmp_path / "allowlists",
        phase="allowlist",
        evidence_hash=preparation_identity,
    ) as output:
        sf.write_json(output / "allowlist.json", allowlist)
        for filename in (
            "allowlist_receipt.json",
            "source_manifest.json",
            "access_receipt.json",
            "source_code.json",
        ):
            sf.write_json(output / filename, {"fixture": filename})
    loaded, digest = test_eval.load_allowlist(output)
    assert loaded["entries"] == [{"arm": arm, "seed": seed}]
    assert digest == sf.sha256_file(output / "allowlist.json")


@pytest.mark.parametrize("phase", ["evaluate", "summary"])
def test_cli_requires_explicit_research_test_confirmation(tmp_path, phase):
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/run_w0_structural_factorial_v1_test.py"),
        phase,
        "--allowlist",
        str(tmp_path / "missing"),
    ]
    if phase == "evaluate":
        command.extend(["--arm", test_eval.ARMS[0], "--seed", str(sf.SEEDS[0])])
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "--confirm-research-test" in result.stderr
    assert "No such file" not in result.stderr
