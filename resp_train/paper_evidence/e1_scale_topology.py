"""E1 的固定尺度操作、原生 forward 包装器及描述性配对统计。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

PROTOCOL = "e1-w0-scale-topology-v1-20260915"
SEEDS = (20260811, 20260812, 20260813)
CONDITIONS = ("FULL", "SCALE_SHIFT_12", "SCALE_REVERSE", "SCALE_PERMUTE_FIXED")
PERMUTATION_SEED = 20260915
WINDOW_COUNT = 2675
ERRORS = (
    "whole_rr_abs_error_bpm", "local_rr_mae_bpm",
    "envelope_trajectory_mae", "global_envelope_modulation_error",
)
PCC = "lag_aware_signed_pcc"
PRIMARY = (*ERRORS, PCC)
FILM_COLUMNS = ("gamma_raw_pair_mae", "beta_raw_pair_mae")


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes(order="C")).hexdigest()


def make_index_lock() -> dict[str, Any]:
    """仅在实现准备时生成；正式评价读取保存的数组。"""
    slots = np.arange(97, dtype=np.int64)
    indices = (
        slots, (slots - 12) % 97, slots[::-1],
        np.random.Generator(np.random.PCG64(PERMUTATION_SEED)).permutation(97),
    )
    lock: dict[str, Any] = {
        "protocol": PROTOCOL, "algorithm": "numpy.random.Generator(PCG64).permutation",
        "seed": PERMUTATION_SEED, "numpy_version": np.__version__,
        "array_encoding": "little-endian int64, C-order",
        "gather": "output[b,s,t] = input[b,index[s],t]", "conditions": {},
    }
    for name, index in zip(CONDITIONS, indices, strict=True):
        inverse = np.argsort(index)
        lock["conditions"][name] = {
            "index": index.tolist(), "inverse": inverse.tolist(),
            "index_sha256": array_hash(index), "inverse_sha256": array_hash(inverse),
        }
    validate_index_lock(lock)
    lock["self_checks"] = index_self_checks(lock)
    return lock


def validate_index_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("protocol") != PROTOCOL or lock.get("seed") != PERMUTATION_SEED:
        raise ValueError("E1 索引锁身份不匹配")
    if tuple(lock.get("conditions", {})) != CONDITIONS:
        raise ValueError("E1 索引锁条件矩阵不完整")
    slots = np.arange(97)
    for name, entry in lock["conditions"].items():
        index, inverse = np.asarray(entry["index"]), np.asarray(entry["inverse"])
        for value in (index, inverse):
            if value.shape != (97,) or value.dtype.kind not in "iu" or not np.array_equal(np.sort(value), slots):
                raise ValueError(f"E1 索引必须是 97 项双射: {name}")
        if not np.array_equal(index[inverse], slots) or not np.array_equal(inverse[index], slots):
            raise ValueError(f"E1 逆索引不匹配: {name}")
        if array_hash(index) != entry["index_sha256"] or array_hash(inverse) != entry["inverse_sha256"]:
            raise ValueError(f"E1 索引哈希不匹配: {name}")
    for name, expected in zip(CONDITIONS[:3], (slots, (slots - 12) % 97, slots[::-1]), strict=True):
        if not np.array_equal(lock["conditions"][name]["index"], expected):
            raise ValueError(f"E1 操作语义漂移: {name}")


def index_self_checks(lock: Mapping[str, Any]) -> dict[str, bool]:
    validate_index_lock(lock)
    value = np.arange(2 * 97 * 360).reshape(2, 97, 360)
    original = value.copy()
    for entry in lock["conditions"].values():
        index, inverse = np.asarray(entry["index"]), np.asarray(entry["inverse"])
        output = value[:, index, :]
        if not np.array_equal(output[:, inverse, :], value):
            raise RuntimeError("E1 正逆重排自检失败")
        if not np.array_equal(np.sort(output, axis=1), np.sort(value, axis=1)):
            raise RuntimeError("E1 每时间点数值多重集自检失败")
        for slot in range(97):
            if not np.array_equal(output[:, slot, :], value[:, index[slot], :]):
                raise RuntimeError("E1 尺度时间序列搬移失败")
        if output.shape != value.shape or output.dtype != value.dtype or not np.array_equal(value, original):
            raise RuntimeError("E1 输入 identity 自检失败")
    return {key: True for key in ("bijection", "inverse_exact", "shape_dtype", "source_unchanged",
                                  "per_time_multiset", "whole_scale_series")}


def apply_scale(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or tuple(value.shape[1:]) != (97, 360):
        raise ValueError("E1 W shape 必须为 (B,97,360)")
    if index.shape != (97,) or index.dtype != torch.int64:
        raise ValueError("E1 gather index 必须是 int64[97]")
    return value.index_select(1, index.to(value.device))


class ScaleAuditModel(nn.Module):
    """所有条件调用原模型；hook 只捕获引用，callback 在完整输出后执行。"""

    def __init__(self, model: nn.Module, condition: str, index_lock: Mapping[str, Any],
                 on_film: Callable[[tuple[torch.Tensor, torch.Tensor]], None]) -> None:
        super().__init__()
        validate_index_lock(index_lock)
        if condition not in CONDITIONS:
            raise ValueError(f"未知 E1 condition: {condition}")
        if (getattr(model, "tf_variant", None) != "crd_tf102_w"
                or set(getattr(model, "branches", {})) != {"w"}
                or len(getattr(model, "controls", ())) != 0
                or getattr(model, "fusion_gate", None) is not None):
            raise ValueError("E1 wrapper 只接受原生 W0")
        self.model, self.condition, self.on_film = model, condition, on_film
        self.register_buffer("index", torch.tensor(index_lock["conditions"][condition]["index"], dtype=torch.int64),
                             persistent=False)

    def forward(self, x: torch.Tensor, *, tf: Mapping[str, torch.Tensor] | None = None) -> Any:
        if tf is None or set(tf) != {"w"}:
            raise ValueError("E1 仅接受 W feature")
        feature = tf if self.condition == "FULL" else {"w": apply_scale(tf["w"], self.index)}
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []

        def capture(_: nn.Module, __: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
            captured.append(output)

        handle = self.model.branches["w"].register_forward_hook(capture)
        try:
            result = self.model(x, tf=feature)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"E1 期望一次原生 W branch 输出，实际 {len(captured)}")
        self.on_film(captured[0])
        return result


def raw_pair_mae(current: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    if current.shape != anchor.shape or current.ndim != 3 or current.size == 0:
        raise ValueError("FiLM 配对 shape 不一致或为空")
    if not np.isfinite(current).all() or not np.isfinite(anchor).all():
        raise FloatingPointError("FiLM 原始张量非有限")
    with np.errstate(over="raise", invalid="raise"):
        values = np.abs(current.astype(np.float64) - anchor.astype(np.float64)).mean(axis=(1, 2))
    if not np.isfinite(values).all():
        raise FloatingPointError("FiLM 配对 MAE 非有限")
    return values


def validate_rows(rows: pd.DataFrame, *, expected_count: int = WINDOW_COUNT) -> None:
    if len(rows) != expected_count or rows["dataset_row_id"].isna().any() or rows["dataset_row_id"].duplicated().any():
        raise ValueError("E1 row 数量、身份缺失或重复")
    if rows["dataset_row_id"].dtype.kind not in "iu" or rows["samp_id"].dtype.kind not in "iu":
        raise ValueError("E1 row/samp identity 必须为整数")
    if set(rows["split"].astype(str)) != {"val"} or rows["samp_id"].isna().any():
        raise ValueError("E1 当前矩阵要求完整 validation identity")


def validate_metrics(metrics: pd.DataFrame, rows: pd.DataFrame) -> None:
    """保留冻结 target-only 缺失语义，禁止非资格原因缩小分母。"""
    if len(metrics) != len(rows):
        raise ValueError("E1 metrics row 数量不一致")
    for key in ("dataset_row_id", "split", "samp_id"):
        if not np.array_equal(metrics[key].to_numpy(), rows[key].to_numpy()):
            raise ValueError(f"E1 metrics identity/order 不一致: {key}")
    eligibility = {ERRORS[0]: "whole_rr_target_eligible", ERRORS[1]: "local_rr_target_eligible",
                   PCC: "joint_target_eligible"}
    for column in PRIMARY:
        values = metrics[column].to_numpy(dtype=np.float64)
        expected_finite = np.ones(len(metrics), dtype=bool)
        if column in eligibility:
            flags = metrics[eligibility[column]]
            if flags.isna().any() or not flags.isin([True, False]).all():
                raise ValueError("E1 target eligibility 非法")
            expected_finite = flags.to_numpy(dtype=bool)
        if np.isinf(values).any() or not np.array_equal(np.isfinite(values), expected_finite):
            raise FloatingPointError(f"E1 主指标 finite/eligibility 不一致: {column}")
    for column in ("joint_prediction_degenerate", "envelope_spearman_prediction_degenerate"):
        values = metrics[column]
        if values.isna().any() or not values.isin([True, False]).all() or values.astype(bool).any():
            raise RuntimeError(f"E1 prediction degeneracy: {column}")


def check_full_anchor(observed: pd.DataFrame, reference: pd.DataFrame, seed: int) -> dict[str, Any]:
    if len(observed) != 1 or len(reference) != 1:
        raise ValueError("FULL summary 必须各含一行")
    columns = [f"{key}_mean" for key in PRIMARY]
    left, right = (frame[columns].to_numpy(dtype=np.float64)[0] for frame in (observed, reference))
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise FloatingPointError("FULL 复现主指标非有限")
    delta = left - right
    receipt = {"seed": seed, "absolute_deltas": dict(zip(columns, np.abs(delta).tolist(), strict=True)),
               "atol": 1e-6, "rtol": 0.0, "passed": bool(np.all(np.abs(delta) <= 1e-6))}
    if not receipt["passed"]:
        raise RuntimeError(f"FULL 复现超差: {receipt}")
    for key in PRIMARY:
        count = f"{key}_n"
        if int(observed.iloc[0][count]) != int(reference.iloc[0][count]):
            raise ValueError(f"FULL 指标分母漂移: {key}")
    return receipt


def summarize_pairs(seed_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回原始聚合表、逐 seed delta、delta 聚合表；保持两种相对变化的区别。"""
    keys = ["condition", "seed"]
    expected = {(name, seed) for name in CONDITIONS for seed in SEEDS}
    if (seed_summary[keys].duplicated().any()
            or set(seed_summary[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError("E1 summary 条件×seed 矩阵不完整或重复")
    columns = [f"{key}_mean" for key in PRIMARY] + list(FILM_COLUMNS)
    if not np.isfinite(seed_summary[columns].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("E1 summary 非有限")
    full = seed_summary[seed_summary.condition.eq("FULL")].set_index("seed").loc[list(SEEDS)]
    if (full[[f"{key}_mean" for key in ERRORS]].to_numpy() <= 0).any():
        raise ValueError("E1 FULL error 分母必须大于零")
    if (seed_summary[list(FILM_COLUMNS)].to_numpy() < 0).any() or (full[list(FILM_COLUMNS)].to_numpy() != 0).any():
        raise ValueError("E1 FiLM MAE 必须非负，FULL 必须为零")
    aggregate, paired, delta_aggregate = [], [], []
    for name in CONDITIONS:
        frame = seed_summary[seed_summary.condition.eq(name)].set_index("seed").loc[list(SEEDS)]
        for column in columns:
            values = frame[column].to_numpy(dtype=np.float64)
            aggregate.append({"condition": name, "metric": column, "mean": float(values.mean()),
                              "sample_sd": float(values.std(ddof=1)), "n_seeds": 3})
        for metric in PRIMARY:
            column = f"{metric}_mean"
            values, anchors = frame[column].to_numpy(dtype=np.float64), full[column].to_numpy(dtype=np.float64)
            delta = anchors - values if metric == PCC else 100.0 * (values - anchors) / anchors
            aggregate_delta = anchors.mean() - values.mean() if metric == PCC else 100.0 * (values.mean() - anchors.mean()) / anchors.mean()
            if not np.isfinite(delta).all() or not np.isfinite(aggregate_delta):
                raise FloatingPointError("E1 paired delta 非有限")
            unit = "absolute_drop" if metric == PCC else "relative_percent"
            for seed, value, anchor, change in zip(SEEDS, values, anchors, delta, strict=True):
                paired.append({"condition": name, "seed": seed, "metric": metric, "value": float(value),
                               "full_value": float(anchor), "delta": float(change), "unit": unit})
            delta_aggregate.append({"condition": name, "metric": metric, "unit": unit,
                                    "paired_delta_mean": float(delta.mean()), "paired_delta_sample_sd": float(delta.std(ddof=1)),
                                    "delta_of_seed_means": float(aggregate_delta), "worse_seeds": int((delta > 0).sum()),
                                    "better_seeds": int((delta < 0).sum()), "equal_seeds": int((delta == 0).sum())})
    return pd.DataFrame(aggregate), pd.DataFrame(paired), pd.DataFrame(delta_aggregate)
