from __future__ import annotations

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
from resp_train.crd.config import load_crd_config
from resp_train.crd.experiment import _early_stopping_should_stop
from resp_train.crd.tf_v1_model import CRDTfV1Model
from resp_train.crd.training import crd_learning_rate, partition_weight_decay_parameters
from resp_train.paper_evidence import e6_temporal_frontend as e6
from resp_train.paper_evidence.e6_temporal_frontend_engineering import synthetic_batch
from resp_train.paper_evidence.e6_temporal_frontend_model import (
    ARM,
    FRONTEND_CONTRACT,
    FRONTEND_PARAMETERS,
    MODEL_PARAMETERS,
    E6TemporalFrontend,
    E6TemporalFrontendModel,
    build_e6_model,
)
from resp_train.temporal.blocks import ChannelLayerNorm1D, TemporalStem
from resp_train.temporal_signal_audit import ResampleSpec, fixed_resample


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def baseline(seed=e6.SEEDS[0]):
    root = Path(__file__).resolve().parents[1]
    return load_crd_config(
        root / "configs/crd_tf_v1/crd_tf102_w_formal.yaml",
        overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"],
    )


def test_spec_config_is_strict_and_generic_loader_unchanged(tmp_path):
    spec = e6.load_experiment_spec()
    assert OmegaConf.to_container(spec.frontend, resolve=True) == FRONTEND_CONTRACT
    reference = baseline()
    before = OmegaConf.to_container(reference, resolve=True)
    cfg = e6.derived_config(reference, output_root=tmp_path, device="cpu")
    e6.validate_config(cfg, reference, output_root=tmp_path, device="cpu")
    assert OmegaConf.to_container(reference, resolve=True) == before
    assert cfg.loss == reference.loss and cfg.data == reference.data
    assert cfg.training.early_stopping_enabled is True
    assert cfg.training.early_stopping_min_epoch == 30
    assert cfg.training.early_stopping_patience == 15
    assert cfg.training.early_stopping_min_delta == 0.0
    OmegaConf.save(cfg, tmp_path / "e6.yaml")
    with pytest.raises(ValueError, match="冻结要求"):
        load_crd_config(tmp_path / "e6.yaml")
    with pytest.raises((ValueError, TypeError)):
        build_e6_model(reference)


def test_frozen_mechanism_sources_and_critical_paths_exist():
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / e6.SOURCE_AUDIT).read_text())
    assert audit["protocol"] == e6.PROTOCOL
    assert audit["decision"]["candidate"] == ARM
    assert e6.sha256_file(root / e6.SOURCE_LOCK) == e6.SOURCE_LOCK_SHA256
    for item in audit["frozen_sources"].values():
        assert e6.sha256_file(root / item["path"]) == item["sha256"]
    assert all((root / path).is_file() for path in e6._critical_paths())


@pytest.mark.parametrize(
    "key,value",
    [
        ("model.e6_temporal_frontend.stage_100_to_20.numtaps", 127),
        ("model.e6_temporal_frontend.learned_20hz.carrier_filter.kernel_size", 19),
        ("model.e6_temporal_frontend.learned_20hz.normalization", "none"),
        ("model.e6_temporal_frontend.stage_20_to_10.cutoff_hz", 4.0),
        ("training.batch_size", 64),
        ("training.gradient_accumulation_steps", 2),
        ("training.epochs", 79),
        ("training.early_stopping_enabled", False),
        ("training.early_stopping_min_epoch", 29),
        ("training.early_stopping_patience", 14),
        ("training.early_stopping_min_delta", 0.01),
        ("loss.effort_weight", 0.0),
        ("model.initialization_seed", 1),
        ("data.train_split", "test"),
        ("protocol.name", "legacy"),
    ],
)
def test_scientific_config_drift_is_rejected(tmp_path, key, value):
    reference = baseline()
    cfg = e6.derived_config(reference, output_root=tmp_path, device="cpu")
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        e6.validate_config(cfg, reference, output_root=tmp_path, device="cpu")


def test_two_fixed_decimators_match_train_audit_operators():
    rng = np.random.default_rng(20260923)
    signals = rng.normal(size=(2, 18000)) + np.linspace(-0.4, 0.7, 18000)
    frontend = E6TemporalFrontend().double()
    actual_20 = (
        frontend.decimate_100_to_20(torch.from_numpy(signals)[:, None])
        .detach()
        .numpy()[:, 0]
    )
    expected_20 = np.stack(
        [
            fixed_resample(
                signal,
                input_fs=100.0,
                spec=ResampleSpec(up=1, down=5, numtaps=255, cutoff_hz=9.0),
                beta=8.6,
                padtype="line",
            )
            for signal in signals
        ]
    )
    np.testing.assert_allclose(actual_20, expected_20, atol=5e-12, rtol=5e-12)
    latent = rng.normal(size=(2, 3, 3600))
    actual_10 = frontend.decimate_20_to_10(torch.from_numpy(latent)).detach().numpy()
    expected_10 = np.stack(
        [
            np.stack(
                [
                    fixed_resample(
                        channel,
                        input_fs=20.0,
                        spec=ResampleSpec(up=1, down=2, numtaps=127, cutoff_hz=4.5),
                        beta=8.6,
                        padtype="line",
                    )
                    for channel in sample
                ]
            )
            for sample in latent
        ]
    )
    np.testing.assert_allclose(actual_10, expected_10, atol=5e-12, rtol=5e-12)


def test_frontend_is_exact_rtm_stem_with_strict_shape_and_parameters():
    frontend = E6TemporalFrontend()
    assert isinstance(frontend, TemporalStem)
    assert sum(parameter.numel() for parameter in frontend.parameters()) == FRONTEND_PARAMETERS
    assert sum(isinstance(module, ChannelLayerNorm1D) for module in frontend.modules()) == 3
    assert not any(isinstance(module, (nn.GroupNorm, nn.BatchNorm1d)) for module in frontend.modules())
    assert frontend.depthwise_20.groups == 48
    assert frontend.decimate_100_to_20.taps.shape == (255,)
    assert frontend.decimate_20_to_10.taps.shape == (127,)
    x = torch.randn(2, 1, 18000, generator=torch.Generator().manual_seed(7))
    sampled = frontend.decimate_100_to_20(x)
    assert sampled.shape == (2, 1, 3600)
    actual = frontend(x)
    assert actual.shape == (2, 96, 1800)
    assert torch.isfinite(actual).all()


def test_frontend_initialization_is_exact_temporal_stem_initialization():
    torch.manual_seed(913)
    expected = TemporalStem()
    torch.manual_seed(913)
    actual = E6TemporalFrontend()
    assert expected.state_dict().keys() == actual.state_dict().keys()
    assert all(
        torch.equal(value, actual.state_dict()[name])
        for name, value in expected.state_dict().items()
    )


def test_all_learned_stem_parameters_have_immediate_finite_gradient():
    frontend = E6TemporalFrontend()
    x = torch.randn(2, 1, 18000, generator=torch.Generator().manual_seed(8)).requires_grad_()
    frontend(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in frontend.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.ne(0).any()


@pytest.mark.parametrize("bad", ["shape", "nan", "inf", "weight_inf"])
def test_frontend_invalid_or_nonfinite_values_fail(bad):
    frontend = E6TemporalFrontend()
    x = torch.zeros(1, 1, 18000)
    if bad == "shape":
        x = x[..., :-1]
    elif bad == "weight_inf":
        with torch.no_grad():
            frontend.carrier_filter_20.weight[0, 0, 0] = float("inf")
    else:
        x[0, 0, 0] = float(bad)
    with pytest.raises((ValueError, FloatingPointError)):
        frontend(x)


@pytest.mark.parametrize("seed", e6.SEEDS)
def test_shared_initialization_rng_parameter_and_optimizer_contract(seed):
    torch.manual_seed(71)
    start = torch.get_rng_state()
    reference = CRDTfV1Model("crd_tf102_w", seed)
    reference_rng = torch.get_rng_state()
    torch.set_rng_state(start)
    candidate = E6TemporalFrontendModel(seed)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    ref_shared = {
        key: value
        for key, value in reference.state_dict().items()
        if not key.startswith("base.frontend.")
    }
    candidate_shared = {
        key: value
        for key, value in candidate.state_dict().items()
        if not key.startswith("base.frontend.")
    }
    assert len(ref_shared) == 165
    assert ref_shared.keys() == candidate_shared.keys()
    assert all(torch.equal(value, candidate_shared[key]) for key, value in ref_shared.items())
    assert sum(parameter.numel() for parameter in candidate.parameters()) == MODEL_PARAMETERS
    assert candidate.branch_parameter_counts() == {"w": 150048}
    assert candidate.base.local_block_count == 6
    full_partition = partition_weight_decay_parameters(reference)
    new_partition = partition_weight_decay_parameters(candidate)
    removed = set(full_partition.decay_names) - set(new_partition.decay_names)
    added = set(new_partition.decay_names) - set(full_partition.decay_names)
    assert removed == {
        name for name in full_partition.decay_names if name.startswith("base.frontend.")
    }
    assert added == {
        "base.frontend.carrier_filter_20.weight",
        "base.frontend.depthwise_20.weight",
        "base.frontend.project_20.weight",
    }
    assert {
        name for name in new_partition.no_decay_names if name.startswith("base.frontend.")
    } == {
        f"base.frontend.norm_20_{stage}.norm.{parameter}"
        for stage in ("a", "b", "c")
        for parameter in ("weight", "bias")
    }
    with pytest.raises(RuntimeError):
        candidate.load_state_dict(reference.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        reference.load_state_dict(candidate.state_dict(), strict=True)
    restored = E6TemporalFrontendModel(seed)
    restored.load_state_dict(candidate.state_dict(), strict=True)


def test_builder_and_parameter_compute_report(tmp_path):
    reference = baseline()
    cfg = e6.derived_config(reference, output_root=tmp_path, device="cpu")
    model = build_e6_model(cfg)
    assert isinstance(model, E6TemporalFrontendModel)
    report = e6.parameter_compute_report()
    assert report["frontend_parameters_e6"] == 24672
    assert report["model_parameters_e6"] == MODEL_PARAMETERS
    assert report["frontend_covered_macs_e6"] == 110_300_400
    assert report["whole_model_flops"] is None


def test_early_stopping_min_epoch_gate_and_patience_semantics():
    assert not _early_stopping_should_stop(
        epoch=29, min_epoch=30, epochs_without_improvement=99, patience=15
    )
    assert not _early_stopping_should_stop(
        epoch=30, min_epoch=30, epochs_without_improvement=14, patience=15
    )
    assert _early_stopping_should_stop(
        epoch=30, min_epoch=30, epochs_without_improvement=15, patience=15
    )
    with pytest.raises(ValueError):
        _early_stopping_should_stop(
            epoch=30, min_epoch=0, epochs_without_improvement=15, patience=15
        )


def test_history_audit_accepts_epoch30_stop_and_rejects_early_trigger(tmp_path):
    reference = baseline()
    cfg = e6.derived_config(reference, output_root=tmp_path, device="cpu")
    rows = []
    for epoch in range(1, 31):
        first_update = (epoch - 1) * e6.UPDATES_PER_EPOCH
        last_update = epoch * e6.UPDATES_PER_EPOCH - 1
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update": epoch * e6.UPDATES_PER_EPOCH,
                "train_loss_total": 1.25,
                "train_loss_sync": 1.0,
                "train_loss_effort": 1.0,
                "first_learning_rate": crd_learning_rate(
                    first_update,
                    total_updates=e6.EPOCHS * e6.UPDATES_PER_EPOCH,
                    max_learning_rate=float(cfg.training.max_learning_rate),
                    min_learning_rate=float(cfg.training.min_learning_rate),
                    warmup_fraction=float(cfg.training.warmup_fraction),
                ),
                "last_learning_rate": crd_learning_rate(
                    last_update,
                    total_updates=e6.EPOCHS * e6.UPDATES_PER_EPOCH,
                    max_learning_rate=float(cfg.training.max_learning_rate),
                    min_learning_rate=float(cfg.training.min_learning_rate),
                    warmup_fraction=float(cfg.training.warmup_fraction),
                ),
                "val_local_rr_mae": 1.0,
                "early_stopping_improved": int(epoch == 1),
                "early_stopping_wait": epoch - 1,
                "early_stopping_triggered": int(epoch == 30),
                "early_stopping_min_epoch": 30,
            }
        )
    history = pd.DataFrame(rows)
    assert e6.validate_history(history, cfg) == 1
    history.loc[28, "early_stopping_triggered"] = 1
    with pytest.raises(ValueError, match="early stopping"):
        e6.validate_history(history, cfg)


def comparison_frame():
    rows = []
    for arm in ("W0_FULL", ARM):
        for index, seed in enumerate(e6.SEEDS):
            value = [1.0, 2.0, 4.0][index]
            change = 0 if arm == "W0_FULL" else [0.1, -0.2, 1.2][index]
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    **{key + "_mean": value + change for key in e6.ERRORS},
                    e6.PCC + "_mean": 0.8 if arm == "W0_FULL" else [0.7, 0.9, 0.6][index],
                }
            )
    return pd.DataFrame(rows)


def test_paired_statistics_keep_all_seeds_and_direction():
    frame = comparison_frame()
    paired, aggregate = e6.paired_tables(frame)
    assert len(paired) == 15 and len(aggregate) == 5
    row = aggregate.iloc[0]
    assert row.paired_delta_mean == pytest.approx(10)
    assert row.paired_delta_sample_sd == pytest.approx(20)
    assert row.full_better_seeds == 2 and row.candidate_better_seeds == 1
    invalid = comparison_frame().iloc[:-1]
    with pytest.raises(ValueError):
        e6.paired_tables(invalid)
    frame.loc[0, e6.ERRORS[0] + "_mean"] = 0.0
    paired, aggregate = e6.paired_tables(frame)
    assert np.isnan(paired.iloc[0].delta)
    assert np.isnan(aggregate.iloc[0].paired_delta_mean)
    assert aggregate.iloc[0].relative_delta_defined_seeds == 2


def metric_frame(offset: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "dataset_row_id": [0, 1, 2, 3],
            "samp_id": [10, 10, 20, 20],
            "split": "val",
            "whole_rr_target_eligible": True,
            "local_rr_target_eligible": True,
            "joint_target_eligible": True,
        }
    )
    for metric in e6.ERRORS:
        frame[metric] = np.asarray([1.0, 2.0, 3.0, 4.0]) + offset
    frame[e6.PCC] = np.asarray([0.5, 0.6, 0.7, 0.8]) - offset
    return frame


def test_subject_tables_use_equal_subject_macro_and_same_rows():
    sources = {
        seed: (metric_frame(0.0), metric_frame(0.1)) for seed in e6.SEEDS
    }
    paired, macro = e6.subject_tables(sources)
    assert len(paired) == 3 * 2 * 5
    assert len(macro) == 3 * 5
    assert set(macro.subject_count) == {2}
    assert (macro.full_better_subjects == 2).all()
    broken = dict(sources)
    candidate = metric_frame(0.1)
    candidate.loc[0, "dataset_row_id"] = 99
    broken[e6.SEEDS[0]] = (metric_frame(0), candidate)
    with pytest.raises(ValueError, match="identity"):
        e6.subject_tables(broken)


def test_lifecycle_preserves_failure_and_rejects_concurrent_execution(tmp_path):
    with e6.phase_guard(tmp_path, "a" * 64):
        with pytest.raises(RuntimeError, match="正在运行"):
            with e6.phase_guard(tmp_path, "a" * 64):
                pass
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with e6.attempt(tmp_path, "a" * 64, "formal", e6.SEEDS[0]) as path:
            (path / "partial.txt").write_text("保留")
            raise RuntimeError("synthetic failure")
    assert (path / "lifecycle_failed.json").exists()
    assert (path / "partial.txt").read_text() == "保留"
    with pytest.raises(ValueError, match="失败"):
        e6.verify_attempt(path, phase="formal", lock_hash="a" * 64)


class SmallFixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 5, padding=2)

    def forward(self, x, **_kwargs):
        return {"waveform": x + self.conv(x)}


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(e6, "ROOT", tmp_path)
    monkeypatch.setattr(e6, "EPOCHS", 2)
    monkeypatch.setattr(e6, "UPDATES_PER_EPOCH", 2)
    monkeypatch.setattr(e6, "EARLY_STOP_MIN_EPOCH", 1)
    monkeypatch.setattr(e6, "EARLY_STOP_PATIENCE", 1)
    monkeypatch.setattr(e6, "EARLY_STOP_MIN_DELTA", 0.0)
    monkeypatch.setattr(e6, "COUNTS", {"train": 3, "val": 2})
    monkeypatch.setattr(e6, "SAMP_IDS", {"train": 3, "val": 2})
    monkeypatch.setattr(
        e6,
        "runtime_preflight",
        lambda device: {"fixture": "CPU", "git": {"commit": "synthetic"}},
    )
    monkeypatch.setattr(e6, "build_e6_model", lambda cfg: SmallFixtureModel())
    batch = synthetic_batch(3, 56)

    def samples(count, split):
        return [
            {
                "x": batch["x"][index],
                "target": batch["target"][index],
                "tf": {"w": batch["tf"]["w"][index]},
                "meta": {
                    "dataset_row_id": index,
                    "samp_id": index,
                    "split": split,
                },
            }
            for index in range(count)
        ]

    rows = pd.DataFrame(
        {"dataset_row_id": [0, 1], "samp_id": [0, 1], "split": "val"}
    )
    data = SimpleNamespace(
        train=SimpleNamespace(loader=DataLoader(samples(3, "train"), batch_size=2)),
        val=SimpleNamespace(loader=DataLoader(samples(2, "val"), batch_size=2)),
        audit_summary=pd.DataFrame({"fixture": [True]}),
    )
    monkeypatch.setattr(native, "build_tho_data", lambda cfg: data)
    baselines, entries, source_files = {}, [], {}
    for seed in e6.SEEDS:
        cfg = baseline(seed)
        cfg.training.epochs = 2
        baselines[str(seed)] = OmegaConf.to_container(cfg, resolve=True)
        source = tmp_path / f"full_{seed}"
        source.mkdir()
        reference = metric_frame(0).iloc[:2].copy()
        reference["samp_id"] = [0, 1]
        for key in ("joint_prediction_degenerate", "envelope_spearman_prediction_degenerate"):
            reference[key] = False
        reference.to_csv(source / "metrics.csv", index=False)
        source_files[f"full_{seed}/metrics.csv"] = e6.identity(source / "metrics.csv")
        summary = {
            key + "_mean": [float(reference[key].mean())] for key in e6.PRIMARY
        }
        summary.update({key + "_n": [2] for key in e6.PRIMARY})
        pd.DataFrame(summary).to_csv(source / "metrics_summary.csv", index=False)
        source_files[f"full_{seed}/metrics_summary.csv"] = e6.identity(
            source / "metrics_summary.csv"
        )
        entries.append(
            {
                "seed": seed,
                "run_dir": source.name,
                "selected_epoch": 1,
            }
        )
    lock = {
        "baselines": baselines,
        "w0_entries": entries,
        "source_files": source_files,
    }
    lock_hash = "a" * 64
    monkeypatch.setattr(e6, "load_lock", lambda: (lock, lock_hash))

    def audit(_lock, _cfg, output):
        rows.to_csv(output / "val_rows.csv", index=False)
        e6.write_json(output / "access_receipt.json", {"fixture": True})
        return {"val": rows}

    monkeypatch.setattr(e6, "audit_sources", audit)
    with e6.attempt(tmp_path / "gpu", lock_hash, "gpu_acceptance") as gpu:
        e6.write_json(
            gpu / "environment.json",
            {"fixture": "CPU", "git": {"commit": "synthetic"}},
        )
        e6.write_json(gpu / "access_receipt.json", {"fixture": True})
        e6.write_json(
            gpu / "gpu_acceptance.json",
            {
                "passed": True,
                "frontend_contract": FRONTEND_CONTRACT,
                "seeds": list(e6.SEEDS),
                "physical_batch": 128,
            },
        )
    return tmp_path, gpu


def test_native_three_seed_lifecycle_selector_and_summary(native_fixture):
    root, gpu = native_fixture
    completed = []
    for seed in e6.SEEDS:
        output = e6.run_formal(seed, gpu_receipt=gpu, device="cpu")
        receipt = json.loads((output / "formal_receipt.json").read_text())
        assert receipt["planned_epochs"] == 2
        assert receipt["completed_epochs"] == 2
        assert receipt["planned_updates"] == 4
        assert receipt["completed_updates"] == 4
        assert receipt["quality_acceptance_passed"]
        history = pd.read_csv(output / receipt["run_dir"] / "train_history.csv")
        assert receipt["selected_epoch"] == int(
            history.loc[history.val_local_rr_mae.idxmin(), "epoch"]
        )
        completed.append(output)
    with pytest.raises(FileExistsError):
        e6.run_formal(e6.SEEDS[0], gpu_receipt=gpu, device="cpu")
    summary = e6.summarize(completed)
    assert len(pd.read_csv(summary / "seed_metrics.csv")) == 6
    assert len(pd.read_csv(summary / "paired_seed_delta.csv")) == 15
    assert len(pd.read_csv(summary / "three_seed_comparison.csv")) == 5
    assert len(pd.read_csv(summary / "paired_seed_subject_delta.csv")) == 30
    assert len(pd.read_csv(summary / "subject_macro_by_seed.csv")) == 15
    with pytest.raises(FileExistsError):
        e6.summarize(completed)
