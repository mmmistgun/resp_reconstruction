from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from resp_train.crd.tf_v1_model import TF_VARIANTS


PRIMARY_METRICS = (
    "whole_rr_mae",
    "local_rr_mae",
    "trajectory_mae",
    "global_envelope_error",
    "signed_pcc",
)
ERROR_METRICS = frozenset(PRIMARY_METRICS[:-1])
EXPECTED_SEEDS = (20260811, 20260812, 20260813)


def validate_formal_matrix(available: Mapping[str, Sequence[int]]) -> None:
    """P5 前置完整性闸门；缺臂或缺 seed 时拒绝汇总。"""

    expected_variants = set(TF_VARIANTS)
    actual_variants = set(available)
    if actual_variants != expected_variants:
        missing = sorted(expected_variants - actual_variants)
        extra = sorted(actual_variants - expected_variants)
        raise RuntimeError(f"CRD-TF formal matrix 不完整: missing={missing}, extra={extra}")
    expected_seeds = set(EXPECTED_SEEDS)
    for variant, seeds in available.items():
        if set(map(int, seeds)) != expected_seeds:
            raise RuntimeError(f"CRD-TF {variant} seeds 不完整: {sorted(map(int, seeds))}")


def to_utility(metric: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    if metric not in PRIMARY_METRICS:
        raise KeyError(f"未知 CRD-TF primary metric={metric!r}")
    if not np.isfinite(values_array).all():
        raise FloatingPointError(f"CRD-TF metric {metric} 包含 NaN/Inf")
    return -values_array if metric in ERROR_METRICS else values_array


def pair_interaction(
    metric: str,
    *,
    base: Sequence[float],
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    pair: Sequence[float],
) -> np.ndarray:
    arrays = [to_utility(metric, values) for values in (base, arm_a, arm_b, pair)]
    _require_paired_seed_shape(arrays)
    base_u, a_u, b_u, pair_u = arrays
    return pair_u - a_u - b_u + base_u


def triple_interaction(
    metric: str,
    *,
    base: Sequence[float],
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    arm_c: Sequence[float],
    pair_ab: Sequence[float],
    pair_ac: Sequence[float],
    pair_bc: Sequence[float],
    triple: Sequence[float],
) -> np.ndarray:
    arrays = [
        to_utility(metric, values)
        for values in (base, arm_a, arm_b, arm_c, pair_ab, pair_ac, pair_bc, triple)
    ]
    _require_paired_seed_shape(arrays)
    base_u, a_u, b_u, c_u, ab_u, ac_u, bc_u, triple_u = arrays
    return triple_u - ab_u - ac_u - bc_u + a_u + b_u + c_u - base_u


def interaction_summary(values: Sequence[float]) -> dict[str, float | int | bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("interaction 必须包含三个 finite paired-seed 值")
    positive_seeds = int(np.sum(array > 0.0))
    mean = float(np.mean(array))
    return {
        "mean": mean,
        "sample_sd": float(np.std(array, ddof=1)),
        "positive_seed_count": positive_seeds,
        "descriptive_positive": bool(mean > 0.0 and positive_seeds >= 2),
    }


def passes_base_guardrails(candidate: Mapping[str, float], base: Mapping[str, float]) -> bool:
    _require_metric_mapping(candidate)
    _require_metric_mapping(base)
    return bool(
        candidate["local_rr_mae"] <= base["local_rr_mae"] * 1.015
        and candidate["trajectory_mae"] <= base["trajectory_mae"] * 1.015
        and candidate["signed_pcc"] >= base["signed_pcc"] - 0.005
    )


def materially_improves(metric: str, candidate: float, reference: float) -> bool:
    if metric not in PRIMARY_METRICS:
        raise KeyError(f"未知 CRD-TF primary metric={metric!r}")
    if not np.isfinite([candidate, reference]).all():
        raise FloatingPointError("candidate/reference 包含 NaN/Inf")
    if metric == "signed_pcc":
        return bool(candidate - reference >= 0.002)
    denominator = abs(reference)
    if denominator == 0.0:
        raise ValueError(f"{metric} reference=0，无法应用 0.5% 相对阈值")
    return bool((reference - candidate) / denominator >= 0.005)


def paired_material_improvement(
    metric: str,
    candidate: Sequence[float],
    reference: Sequence[float],
) -> dict[str, float | int | bool]:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != (3,) or reference_array.shape != (3,):
        raise ValueError("capacity comparison 必须包含三个 paired seeds")
    if not np.isfinite(candidate_array).all() or not np.isfinite(reference_array).all():
        raise FloatingPointError("capacity comparison 包含 NaN/Inf")
    utility_delta = to_utility(metric, candidate_array) - to_utility(metric, reference_array)
    direction_count = int(np.sum(utility_delta > 0.0))
    candidate_mean = float(np.mean(candidate_array))
    reference_mean = float(np.mean(reference_array))
    threshold_passed = materially_improves(metric, candidate_mean, reference_mean)
    return {
        "candidate_mean": candidate_mean,
        "reference_mean": reference_mean,
        "utility_delta_mean": float(np.mean(utility_delta)),
        "positive_seed_count": direction_count,
        "threshold_passed": threshold_passed,
        "passed": bool(threshold_passed and direction_count >= 2),
    }


def tolerance_dominates(candidate: Mapping[str, float], reference: Mapping[str, float]) -> bool:
    """五轴容差支配：所有轴不过容差且至少一轴超过实质改善阈值。"""

    _require_metric_mapping(candidate)
    _require_metric_mapping(reference)
    no_worse = True
    strictly_better = False
    for metric in PRIMARY_METRICS:
        candidate_value = float(candidate[metric])
        reference_value = float(reference[metric])
        if metric == "signed_pcc":
            no_worse &= candidate_value >= reference_value - 0.002
        else:
            no_worse &= candidate_value <= reference_value * 1.005
        strictly_better |= materially_improves(metric, candidate_value, reference_value)
    return bool(no_worse and strictly_better)


def tolerance_pareto_set(rows: Mapping[str, Mapping[str, float]]) -> tuple[str, ...]:
    for metrics in rows.values():
        _require_metric_mapping(metrics)
    retained = []
    for name, metrics in rows.items():
        dominated = any(
            other_name != name and tolerance_dominates(other_metrics, metrics)
            for other_name, other_metrics in rows.items()
        )
        if not dominated:
            retained.append(name)
    return tuple(sorted(retained))


def _require_paired_seed_shape(arrays: Sequence[np.ndarray]) -> None:
    if any(array.shape != (3,) for array in arrays):
        raise ValueError("interaction 输入必须逐一对应三个 frozen training seeds")


def _require_metric_mapping(values: Mapping[str, float]) -> None:
    if set(values) != set(PRIMARY_METRICS):
        raise ValueError(f"primary metrics 必须严格为 {list(PRIMARY_METRICS)}")
    if not np.isfinite([float(values[name]) for name in PRIMARY_METRICS]).all():
        raise FloatingPointError("primary metrics 包含 NaN/Inf")
