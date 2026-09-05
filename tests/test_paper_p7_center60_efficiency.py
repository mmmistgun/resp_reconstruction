import pandas as pd

from resp_train.paper_evidence import p7_center60_efficiency as p760


def test_contract_locks_separate_60_to_60_gpu_only_matrix():
    contract = p760.load_contract()
    assert contract["scientific_scope"] == "separate_60s_input_60s_output_task"
    assert contract["authorization"]["cpu_model_benchmark"] is False
    assert contract["authorization"]["research_test_signal_or_target_access"] is False
    assert [item["candidate_id"] for item in contract["models"]] == list(p760.MODEL_ORDER)
    assert {item["seed"] for item in contract["models"]} == {20260811}
    assert contract["input"]["input_samples"] == 6000
    assert contract["input"]["output_samples"] == 6000
    assert contract["benchmark"]["expected_timed_pipeline_iterations"] == 1500


def test_balanced_model_and_w_scenario_schedules():
    benchmark = p760.load_contract()["benchmark"]
    model_orders = benchmark["round_model_order"]
    for candidate in p760.MODEL_ORDER:
        position_counts = [
            sum(order[position] == candidate for order in model_orders)
            for position in (0, 1)
        ]
        assert abs(position_counts[0] - position_counts[1]) <= 1
    assert benchmark["w_round_scenario_order"] == [
        ["cached_w", "online_w"],
        ["online_w", "cached_w"],
        ["cached_w", "online_w"],
        ["online_w", "cached_w"],
        ["cached_w", "online_w"],
    ]


def _latency_summary():
    rows = []
    for candidate, scenarios in p760.SCENARIOS.items():
        for scenario in scenarios:
            rows.append(
                {
                    "candidate_id": candidate,
                    "scenario": scenario,
                    "stage": "end_to_end",
                    "median_ms": 10.0,
                    "p95_ms": 12.0,
                    "mean_ms": 10.5,
                    "sample_sd_ms": 0.5,
                    "min_ms": 9.0,
                    "max_ms": 13.0,
                    "p95_slack_ms": 29988.0,
                    "p95_updates_per_second": 83.333,
                    "context_wait_seconds": 60,
                }
            )
    return pd.DataFrame(rows)


def _resources(contract):
    return pd.DataFrame(
        [
            {
                "candidate_id": item["candidate_id"],
                "experiment_id": item["experiment_id"],
                "model_label": item["model_label"],
                "model_variant": item["variant"],
                "checkpoint_seed": item["seed"],
                "trainable_parameters": item["trainable_parameters"],
                "total_parameters": item["trainable_parameters"],
                "bcg_input_elements": item["bcg_input_elements"],
                "w_input_elements": item["w_input_elements"],
                "total_model_input_elements": item["bcg_input_elements"]
                + item["w_input_elements"],
                "w_scales": item["w_scales"],
                "checkpoint_size_bytes": item["checkpoint_size_bytes"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "config_sha256": item["config_sha256"],
            }
            for item in contract["models"]
        ]
    )


def test_frozen_center60_quality_is_joined_without_test_signal_access():
    contract = p760.load_contract()
    table = p760.build_quality_efficiency_table(
        contract,
        _latency_summary(),
        _resources(contract),
        repo_root=p760.REPO_ROOT,
    )
    assert len(table) == 3
    assert set(table["quality_scope"]) == {"center60_independent_test_three_seed_mean"}
    assert table.filter(regex="center_.*_(mean|sample_sd)$").notna().all().all()
    assert set(table.loc[table["candidate_id"] == p760.C201, "scenario"]) == {
        "waveform_only"
    }


def test_descriptive_context_table_keeps_task_boundaries_explicit():
    contract = p760.load_contract()
    quality = p760.build_quality_efficiency_table(
        contract,
        _latency_summary(),
        _resources(contract),
        repo_root=p760.REPO_ROOT,
    )
    table = p760.build_descriptive_efficiency_context_table(
        contract, quality, repo_root=p760.REPO_ROOT
    )
    assert set(table["input_seconds"]) == {60, 180}
    assert set(table["output_seconds"]) == {60, 180}
    assert not table["cross_task_causal_window_claim_allowed"].any()
