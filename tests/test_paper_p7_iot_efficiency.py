import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from resp_train.crd.tf_w_v2 import FULL_6V_INDICES, VIEW_INDEX_SHA256
from resp_train.paper_evidence import p7_iot_efficiency as p7


def test_frozen_contract_is_gpu_only_single_seed_exact_matrix():
    contract = p7.load_contract()
    assert contract["authorization"] == {
        "gpu_benchmark": True,
        "cpu_model_benchmark": False,
        "training": False,
        "cache_build": False,
        "research_test_access": False,
    }
    assert [item["candidate_id"] for item in contract["models"]] == list(p7.MODEL_ORDER)
    assert {item["seed"] for item in contract["models"]} == {20260811}
    assert contract["benchmark"]["scenarios"] == list(p7.SCENARIO_ORDER)
    assert (
        contract["benchmark"]["warmup_iterations"],
        contract["benchmark"]["timed_iterations"],
        contract["benchmark"]["rounds"],
    ) == (20, 100, 5)


def test_w3_view_is_frozen_even_index_subset_of_full97():
    assert FULL_6V_INDICES.tolist() == list(range(0, 97, 2))
    assert len(FULL_6V_INDICES) == 49
    observed = hashlib.sha256(FULL_6V_INDICES.astype("<i8", copy=False).tobytes()).hexdigest()
    assert observed == VIEW_INDEX_SHA256["full_6v"]
    assert p7.load_contract()["w_source"]["w3_view_index_sha256"] == observed


def _latency_iterations():
    rows = []
    for candidate in p7.MODEL_ORDER:
        for scenario in p7.SCENARIO_ORDER:
            stages = ["host_to_device_pageable_bcg_and_full97_w", "model_forward_including_w_view", "Pi_180", "end_to_end"]
            if scenario == "online_w":
                stages.insert(0, "w_feature_extraction_97scale_log_pool")
            for round_index in range(1, 6):
                for iteration in range(1, 101):
                    for stage_index, stage in enumerate(stages, start=1):
                        rows.append(
                            {
                                "candidate_id": candidate,
                                "scenario": scenario,
                                "round": round_index,
                                "iteration": iteration,
                                "stage": stage,
                                "latency_ms": float(stage_index),
                            }
                        )
    return pd.DataFrame(rows)


def test_stage_summary_exact_counts_and_end_to_end_derived_values():
    summary = p7.summarize_stage_latencies(_latency_iterations(), p7.load_contract())
    assert summary["n"].eq(500).all()
    end_to_end = summary.loc[summary["stage"] == "end_to_end"]
    assert len(end_to_end) == 6
    assert np.isfinite(end_to_end["p95_slack_ms"]).all()
    assert np.isfinite(end_to_end["p95_updates_per_second"]).all()
    assert end_to_end["context_wait_seconds"].eq(180).all()
    non_end = summary.loc[summary["stage"] != "end_to_end"]
    assert non_end["p95_slack_ms"].isna().all()


def test_stage_summary_rejects_partial_matrix():
    partial = _latency_iterations().iloc[:-1]
    with pytest.raises(RuntimeError, match="count"):
        p7.summarize_stage_latencies(partial, p7.load_contract())


def test_memory_summary_reports_five_rounds_and_mib():
    rows = []
    for candidate in p7.MODEL_ORDER:
        for scenario in p7.SCENARIO_ORDER:
            for round_index in range(1, 6):
                rows.append(
                    {
                        "candidate_id": candidate,
                        "scenario": scenario,
                        "round": round_index,
                        "baseline_allocated_bytes": 1024**2,
                        "baseline_reserved_bytes": 2 * 1024**2,
                        "peak_allocated_bytes": 3 * 1024**2,
                        "peak_reserved_bytes": 4 * 1024**2,
                    }
                )
    summary = p7.summarize_memory(pd.DataFrame(rows))
    assert len(summary) == 6
    assert summary["round_count"].eq(5).all()
    assert summary["peak_allocated_mean_mib"].eq(3.0).all()
    assert summary["peak_reserved_max_mib"].eq(4.0).all()


def test_quality_table_keeps_test_and_validation_scopes_separate(tmp_path):
    contract = p7.load_contract()
    test_rows = []
    variants = {item["candidate_id"]: item["variant"] for item in contract["models"]}
    for candidate in p7.MODEL_ORDER[:2]:
        for index, metric in enumerate(p7.PRIMARY_METRICS):
            test_rows.append(
                {
                    "variant": variants[candidate], "metric": p7.TEST_METRIC_NAMES[metric],
                    "mean": index + 0.1, "sample_sd": 0.01,
                }
            )
    test_path = tmp_path / "test.csv"
    pd.DataFrame(test_rows).to_csv(test_path, index=False)
    validation_path = tmp_path / "validation.csv"
    d4 = {"candidate_id": "D4_W0"}
    for index, metric in enumerate(p7.PRIMARY_METRICS):
        d4[f"{metric}_mean_seed_mean"] = index + 0.2
        d4[f"{metric}_mean_sample_sd"] = 0.02
    pd.DataFrame([d4]).to_csv(validation_path, index=False)
    contract = json.loads(json.dumps(contract))
    contract["quality_sources"] = {
        "w0_w3": {"scope": "independent_test_three_seed_mean", "path": test_path.name, "sha256": p7.sha256_file(test_path)},
        "d4": {"scope": "validation_three_seed_mean", "path": validation_path.name, "sha256": p7.sha256_file(validation_path)},
    }
    latency = p7.summarize_stage_latencies(_latency_iterations(), contract)
    resources = pd.DataFrame(
        {
            "candidate_id": list(p7.MODEL_ORDER),
            "model_variant": [variants[value] for value in p7.MODEL_ORDER],
            "trainable_parameters": [1, 1, 1],
        }
    )
    table = p7.build_quality_efficiency_table(contract, latency, resources, repo_root=tmp_path)
    assert len(table) == 6
    scopes = table.groupby("candidate_id")["quality_scope"].first().to_dict()
    assert scopes["D4_W0"] == "validation_three_seed_mean"
    assert scopes["W0_FULL_12V_FILM_D6"] == "independent_test_three_seed_mean"


def test_frozen_quality_sources_are_complete_and_scoped():
    contract = p7.load_contract()
    variants = {item["candidate_id"]: item["variant"] for item in contract["models"]}
    resources = pd.DataFrame(
        {
            "candidate_id": list(p7.MODEL_ORDER),
            "model_variant": [variants[value] for value in p7.MODEL_ORDER],
            "trainable_parameters": [item["trainable_parameters"] for item in contract["models"]],
        }
    )
    latency = p7.summarize_stage_latencies(_latency_iterations(), contract)
    table = p7.build_quality_efficiency_table(contract, latency, resources, repo_root=p7.REPO_ROOT)
    assert len(table) == 6
    assert table.filter(regex="_(mean|sample_sd)$").notna().all().all()
    assert table.loc[table["candidate_id"] == "D4_W0", "quality_scope"].eq(
        "validation_three_seed_mean"
    ).all()
