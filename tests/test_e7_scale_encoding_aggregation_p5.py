from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from resp_train.paper_evidence import e7_scale_encoding_aggregation_p5 as p5


def metric_matrix() -> pd.DataFrame:
    rows = []
    for arm_index, arm in enumerate(p5.ARMS):
        for seed_index, seed in enumerate(p5.e7.SEEDS):
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "split": "val",
                    **{
                        metric + "_mean": 1.0 + seed_index + 0.1 * arm_index
                        for metric in p5.e7.ERRORS
                    },
                    p5.e7.PCC + "_mean": 0.9 - 0.01 * arm_index,
                }
            )
    return pd.DataFrame(rows)


def test_p5_contract_and_formal_matrix_are_available():
    contract = p5.p5_contract()
    assert contract["cells"] == 18
    assert len(contract["representation_sampling"]["channel_indices"]) == 12
    assert len(contract["representation_sampling"]["time_indices"]) == 45
    _lock, digest = p5.p4.load_p4_lock()
    assert digest == p5.P4_LOCK_SHA256
    assert len(p5.p4.completed_runs()) == 18


def test_materiality_direction_and_tolerance():
    per_seed, across = p5.materiality_tables(metric_matrix())
    assert len(per_seed) == 7 * 3 * 5
    assert len(across) == 7 * 5
    error = per_seed.loc[
        per_seed.contrast.eq("mean_local")
        & per_seed.seed.eq(p5.e7.SEEDS[0])
        & per_seed.metric.eq(p5.e7.ERRORS[0])
    ].iloc[0]
    assert error.utility_delta < -0.5
    assert error.classification == "degraded"
    pcc = per_seed.loc[
        per_seed.contrast.eq("mean_local")
        & per_seed.seed.eq(p5.e7.SEEDS[0])
        & per_seed.metric.eq(p5.e7.PCC)
    ].iloc[0]
    assert pcc.utility_delta < -0.002
    assert pcc.classification == "degraded"


def test_scale_accumulator_and_ratio_are_finite():
    generator = torch.Generator().manual_seed(9)
    value = torch.randn(2, 96, 97, 360, generator=generator)
    accumulator = p5.ScaleAccumulator()
    accumulator.update(value)
    summary, correlation = accumulator.finalize()
    assert summary["sampled_observations"] == 2 * 12 * 45
    assert 1 <= summary["entropy_effective_rank"] <= 97
    assert correlation.shape == (97, 97)
    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1e-10, rtol=0)
    ratio = p5.RatioAccumulator()
    ratio.update(value * 0.2, value)
    assert ratio.finalize() == pytest.approx(0.2, rel=1e-6)


def test_attention_statistics_uses_fp32_reduction_bound():
    weights = torch.full((1, 1, 97, 3), 1 / 97, dtype=torch.float32)
    weights[:, :, 0] += 1.1e-6
    stats = p5.attention_statistics(weights)
    assert stats["attention_max_abs_sum_error"] > 1e-6
    assert stats["attention_max_abs_sum_error"] < stats["attention_sum_tolerance"]
    invalid = weights.clone()
    invalid[:, :, 0] += 1e-3
    with pytest.raises(FloatingPointError, match="归一化误差"):
        p5.attention_statistics(invalid)


@pytest.mark.parametrize(
    "arm,expected",
    [
        ("s0_shallow__mean", ["full"]),
        ("s0_shallow__frequency_attention", ["full", "uniform_attention"]),
        ("s1_deep_local__mean", ["full", "residual_off"]),
        (
            "s2_axis_spanning__frequency_attention",
            ["full", "residual_off", "uniform_attention", "both_off"],
        ),
    ],
)
def test_diagnostic_conditions_are_fixed(arm, expected):
    assert [item[0] for item in p5.diagnostic_conditions(arm)] == expected


def test_attempt_keeps_failed_lifecycle(tmp_path):
    runtime = {
        "git": {"commit": "fixture", "status_porcelain": ""},
        "code_files": {},
    }
    with pytest.raises(RuntimeError, match="fixture"):
        with p5.attempt(tmp_path, "summary", runtime) as output:
            (output / "partial.txt").write_text("kept", encoding="utf-8")
            raise RuntimeError("fixture failure")
    assert (output / "partial.txt").read_text() == "kept"
    lifecycle = json.loads((output / "lifecycle_failed.json").read_text())
    assert lifecycle["status"] == "failed"
