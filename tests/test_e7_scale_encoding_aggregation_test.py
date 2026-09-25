from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from resp_train.paper_evidence import e7_scale_encoding_aggregation_test as test_eval


def synthetic_seed_metrics() -> pd.DataFrame:
    rows = []
    for arm_index, arm in enumerate(test_eval.ARMS):
        for seed_index, seed in enumerate(test_eval.e7.SEEDS):
            row = {
                "arm": arm,
                "seed": seed,
                "split": "test",
                "selected_epoch": 10 + seed_index,
                "quality_acceptance_passed": True,
            }
            for metric_index, metric in enumerate(test_eval.e7.PRIMARY):
                row[metric + "_mean"] = 0.1 + 0.01 * arm_index + 0.001 * seed_index + 0.0001 * metric_index
                row[metric + "_n"] = 4
            rows.append(row)
    return pd.DataFrame(rows)


def test_contract_freezes_complete_matrix():
    contract = test_eval.test_contract()
    assert contract["cells"] == 18
    assert contract["arms"] == list(test_eval.ARMS)
    assert contract["seeds"] == list(test_eval.e7.SEEDS)
    assert contract["checkpoint_selector"] == "validation_local_rr_strict_minimum_earliest_tie"
    assert contract["validation_decision_frozen"] is True
    assert contract["count"] == 2310


def test_prepare_lock_is_metadata_only():
    source = inspect.getsource(test_eval.prepare_lock)
    for forbidden in ("np.load", "torch.load", "build_window_data", "collect_predictions"):
        assert forbidden not in source
    assert "cache_manifest" in source
    assert "completed_runs" in source


def test_materiality_and_w0_pairing_cover_all_cells():
    frame = synthetic_seed_metrics()
    per_seed, across = test_eval.materiality_tables(frame)
    assert len(per_seed) == 7 * 3 * 5
    assert len(across) == 7 * 5
    assert set(per_seed.classification) <= {
        "improved",
        "degraded",
        "within_tolerance",
        "undefined",
    }
    w0 = []
    for seed in test_eval.e7.SEEDS:
        row = {"seed": seed}
        for metric in test_eval.e7.PRIMARY:
            row[metric + "_mean"] = 0.2
        w0.append(row)
    paired, grouped = test_eval.w0_paired_tables(frame, pd.DataFrame(w0))
    assert len(paired) == 18 * 5
    assert len(grouped) == 6 * 5
    assert set(paired.positive_means) == {"candidate_improves"}


def test_e7_factorial_test_tables_preserve_test_split():
    frame = synthetic_seed_metrics()
    across = test_eval.e7.across_seed_table(frame, split="test")
    simple_seed, simple_across, factorial_seed, factorial_across = test_eval.e7.contrast_tables(
        frame, split="test"
    )
    assert len(across) == 30
    assert len(simple_seed) == 105
    assert len(simple_across) == 35
    assert len(factorial_seed) == 75
    assert len(factorial_across) == 25
    assert set(across.split) == {"test"}


def test_subject_macro_requires_each_subject_metric_to_be_eligible():
    records = []
    for arm in test_eval.ARMS:
        for seed in test_eval.e7.SEEDS:
            for subject in (1, 2):
                for window in range(2):
                    row = {"arm": arm, "seed": seed, "split": "test", "samp_id": subject}
                    row.update({metric: 0.1 + window for metric in test_eval.e7.PRIMARY})
                    records.append(row)
    macro = test_eval.e7.subject_macro_table(pd.DataFrame(records), split="test")
    assert len(macro) == 18 * 5 * 3
    assert set(macro.row_type) == {"subject", "macro"}
    broken = pd.DataFrame(records)
    mask = (
        broken.arm.eq(test_eval.ARMS[0])
        & broken.seed.eq(test_eval.e7.SEEDS[0])
        & broken.samp_id.eq(1)
    )
    broken.loc[mask, test_eval.e7.PRIMARY[0]] = np.nan
    with pytest.raises(ValueError, match="没有 eligible"):
        test_eval.e7.subject_macro_table(broken, split="test")


def test_evaluation_config_replays_frozen_training_config(tmp_path):
    p1_lock, _digest = test_eval.e7.load_implementation_lock()
    arm = test_eval.ARMS[0]
    seed = test_eval.e7.SEEDS[0]
    cfg = test_eval.e7.derived_config(
        OmegaConf.create(p1_lock["baselines"][str(seed)]),
        arm,
        p1_lock["frequency"]["values_hz"],
        output_root=tmp_path / "training",
        device="cuda:0",
    )
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path)
    entry = {
        "arm": arm,
        "seed": seed,
        "training_config": OmegaConf.to_container(cfg, resolve=True),
        "candidate": {"config.yaml": {"path": str(path)}},
    }
    runtime, data_cfg = test_eval.evaluation_config(entry, "cpu", "/synthetic/cache")
    assert runtime.training.device == "cpu"
    assert data_cfg.data.tf_research_test_cache_path == "/synthetic/cache"
    changed = dict(entry)
    changed["training_config"] = dict(entry["training_config"])
    changed["training_config"]["protocol"] = {"name": "drift"}
    with pytest.raises(ValueError, match="allowlist"):
        test_eval.evaluation_config(changed, "cpu", "/synthetic/cache")


def test_guarded_batches_checks_tail_order_shape_and_finite():
    rows = pd.DataFrame(
        {
            "dataset_row_id": [10, 11, 12],
            "samp_id": [1, 1, 2],
            "split": ["test"] * 3,
        }
    )

    def batch(start: int, stop: int):
        selected = rows.iloc[start:stop]
        count = len(selected)
        return {
            "x": torch.zeros(count, 1, 8),
            "target": torch.zeros(count, 1, 8),
            "tf": {"w": torch.zeros(count, 97, 360)},
            "meta": {key: selected[key].to_numpy() for key in test_eval.IDENTITY_COLUMNS},
        }

    loader = [batch(0, 2), batch(2, 3)]
    assert [len(item["x"]) for item in test_eval.guarded_batches(loader, rows)] == [2, 1]
    bad = batch(0, 3)
    bad["tf"]["w"][0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError):
        list(test_eval.guarded_batches([bad], rows))


def test_attempt_lifecycle_and_required_products(tmp_path):
    lock_hash = "a" * 64
    with test_eval.attempt(
        tmp_path / "attempts",
        "evaluation",
        lock_hash,
        arm=test_eval.ARMS[0],
        seed=test_eval.e7.SEEDS[0],
    ) as output:
        required = {
            "evaluation_receipt.json": "{}\n",
            "metrics.csv": "x\n1\n",
            "metrics_summary.csv": "x\n1\n",
            "test_rows.csv": "x\n1\n",
            "resolved_config.yaml": "x: 1\n",
            "access_started.json": "{}\n",
            "access_receipt.json": "{}\n",
            "environment.json": "{}\n",
            "test_lock.json": "{}\n",
        }
        for filename, content in required.items():
            (output / filename).write_text(content, encoding="utf-8")
    manifest = test_eval.verify_attempt(output, lock_hash, "evaluation")
    assert manifest["arm"] == test_eval.ARMS[0]
    assert manifest["seed"] == test_eval.e7.SEEDS[0]
    (output / "metrics.csv").write_text("drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="身份漂移"):
        test_eval.verify_attempt(output, lock_hash, "evaluation")


def test_failed_attempt_is_retained(tmp_path):
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with test_eval.attempt(tmp_path, "evaluation", "b" * 64):
            raise RuntimeError("synthetic failure")
    failed = next(tmp_path.glob("*/lifecycle_failed.json"))
    payload = json.loads(failed.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"


def test_validate_summary_enforces_full_denominator(monkeypatch):
    monkeypatch.setattr(test_eval, "COUNT", 4)
    expected = pd.DataFrame(
        {
            **{metric + "_mean": [0.1] for metric in test_eval.e7.PRIMARY},
            **{metric + "_n": [4] for metric in test_eval.e7.PRIMARY},
        }
    )
    test_eval._validate_summary(expected.copy(), expected)
    broken = expected.copy()
    broken.loc[0, test_eval.e7.PRIMARY[0] + "_n"] = 3
    with pytest.raises(ValueError, match="分母漂移"):
        test_eval._validate_summary(broken, expected)
