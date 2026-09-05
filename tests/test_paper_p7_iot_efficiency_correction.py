import pandas as pd

from resp_train.paper_evidence import p7_iot_efficiency as p7
from resp_train.paper_evidence import p7_iot_efficiency_correction as p7v2


def test_correction_contract_keeps_matrix_and_changes_measurement_order_only():
    correction, base = p7v2.load_correction_contract()
    assert correction["parent_contract"]["sha256"] == p7.CONFIG_SHA256
    assert correction["authorization"] == base["authorization"]
    benchmark = correction["benchmark"]
    assert (
        benchmark["warmup_iterations"],
        benchmark["timed_iterations"],
        benchmark["rounds"],
    ) == (20, 100, 5)
    assert benchmark["scenarios"] == list(p7.SCENARIO_ORDER)
    assert benchmark["profiler_position"] == "after_all_latency_and_memory_measurements"
    assert correction["output"]["directory"] != base["output"]["directory"]


def test_round_schedule_is_complete_and_position_balanced():
    correction, _ = p7v2.load_correction_contract()
    orders = correction["benchmark"]["round_model_order"]
    assert len(orders) == 5
    for order in orders:
        assert set(order) == set(p7.MODEL_ORDER)
    for position in range(3):
        counts = {candidate: sum(order[position] == candidate for order in orders) for candidate in p7.MODEL_ORDER}
        assert max(counts.values()) - min(counts.values()) <= 1


def _cwt_summary(p95_values, median_values=None):
    if median_values is None:
        median_values = p95_values
    return pd.DataFrame(
        {
            "candidate_id": list(p7.MODEL_ORDER),
            "scenario": "online_w",
            "stage": "w_feature_extraction_97scale_log_pool",
            "n": 500,
            "median_ms": median_values,
            "p95_ms": p95_values,
            "mean_ms": median_values,
            "sample_sd_ms": 1.0,
            "min_ms": 30.0,
            "max_ms": 50.0,
        }
    )


def test_online_cwt_consistency_passes_at_locked_relative_spread_boundary():
    correction, _ = p7v2.load_correction_contract()
    rows, passed = p7v2.build_online_cwt_consistency(
        _cwt_summary([40.0, 42.0, 44.0], [35.0, 36.0, 38.5]), correction
    )
    assert passed
    assert rows["cross_model_consistency_passed"].all()
    assert rows["p95_relative_spread_across_models"].iloc[0] == 0.1


def test_online_cwt_consistency_rejects_prior_observed_divergence():
    correction, _ = p7v2.load_correction_contract()
    rows, passed = p7v2.build_online_cwt_consistency(
        _cwt_summary([66.26067845, 41.5362737, 39.8182617]), correction
    )
    assert not passed
    assert not rows["cross_model_consistency_passed"].any()
    assert rows["p95_relative_spread_across_models"].iloc[0] > 0.6
