"""训练与采样计划 v1 的显式预算；历史入口继续使用原合同。"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

NAME = "respdiff-bcg-training-and-sampling-v1-20261010"
SCHEMA = "respdiff-bcg-training-and-sampling-v1"
TRAIN_SEEDS = (20260811, 20260812, 20260813)
PROBE_STEPS = (0, 9, 19, 29, 39, 49)
PROBE_SEEDS = (20261010, 20261011, 20261012)
PAIRING_SEEDS = (20261020, 20261021, 20261022)
PRESSURE_ROWS = (10382, 12226, 10322)
GRID = {("ddim", 6): (1, 2, 4, 8, 16, 32),
        ("ddim", 10): (1, 2, 4, 8, 16),
        ("ddim", 20): (1, 2, 4, 8),
        ("ddim", 50): (1, 2, 4),
        ("ddpm", 50): (1, 2, 4, 8)}
METRICS = ("postfiltered_lag_aware_signed_pcc", "postfiltered_whole_rr_abs_error_bpm",
           "postfiltered_parent_max_p99", "postfiltered_top_1pct_sample_energy_ratio")


@dataclass(frozen=True)
class StudySamplerSpec:
    sampler: str = "ddim"
    nfe: int = 6
    n_trajectories: int = 16
    noise_seed: int = 20261003
    postfilter: bool = True
    eta: float = 0.0

    def __post_init__(self):
        key = (self.sampler, self.nfe)
        if type(self.nfe) is not int or key not in GRID:
            raise ValueError("本轮支持 DDIM6/10/20/50 或 DDPM50")
        if (type(self.n_trajectories) is not int
                or not 1 <= self.n_trajectories <= GRID[key][-1]):
            raise ValueError("N 超出首轮预算；扩展需要独立合同修订")
        if type(self.noise_seed) is not int or self.noise_seed < 0:
            raise ValueError("noise_seed 必须为非负整数")
        if self.eta != 0 or self.postfilter is not True:
            raise ValueError("本轮固定 eta=0 和 parent LPF")

    @property
    def timesteps(self):
        if self.nfe == 6:
            return (49, 39, 29, 19, 9, 0)
        if self.nfe == 50:
            return tuple(range(49, -1, -1))
        # 显式 half-up，完整列表同时进入运行身份。
        return tuple(math.floor(49 * (self.nfe - 1 - i) / (self.nfe - 1) + .5)
                     for i in range(self.nfe))

    def identity(self):
        return {**asdict(self), "timesteps": list(self.timesteps)}


@dataclass(frozen=True)
class TrainingSpec:
    regime: str
    seed: int = 20260811

    def __post_init__(self):
        if self.regime not in ("overfit", "baseline", "long_absolute", "long_stretched"):
            raise ValueError("未知训练方案")
        if type(self.seed) is not int or self.seed not in TRAIN_SEEDS:
            raise ValueError("train seed 偏离计划")
        if self.regime == "overfit" and self.seed != TRAIN_SEEDS[0]:
            raise ValueError("小样本诊断只使用首个 seed")

    @property
    def updates(self):
        return 3200 if self.regime == "overfit" else (6400 if self.regime == "baseline" else 16000)

    @property
    def milestones(self):
        return (() if self.regime == "overfit" else
                (11200, 15840) if self.regime == "long_stretched" else (4480, 6336))

    @property
    def snapshots(self):
        if self.regime == "overfit":
            return (0, 400, 800, 1600, 3200)
        return tuple(v for v in (0, 1600, 3200, 4800, 6400, 9600, 12800, 16000) if v <= self.updates)

    @property
    def validation_updates(self):
        return self.snapshots[1:] if self.regime == "overfit" else tuple(
            v for v in (6400, 9600, 12800, 16000) if v <= self.updates)

    def identity(self):
        return {**asdict(self), "updates": self.updates, "milestones": list(self.milestones),
                "snapshots": list(self.snapshots), "validation_updates": list(self.validation_updates),
                "selector": "final_update", "learning_rate": 1e-4, "gamma": .1,
                "batch_size": 64, "num_workers": 0}


def grid_specs(noise_seed=20261003):
    return [(StudySamplerSpec(kind, steps, counts[-1], noise_seed), counts)
            for (kind, steps), counts in GRID.items()]


def _valid_curve(curve):
    required = ["sampler", "nfe_per_trajectory", "N", "denoiser_calls_per_chunk", *METRICS]
    if curve.empty or any(key not in curve for key in required):
        raise ValueError("选择曲线缺少必要字段或为空")
    keys = ["sampler", "nfe_per_trajectory", "N"]
    if curve.duplicated(keys).any() or not np.isfinite(curve[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("选择曲线存在重复配置或非有限指标")
    for row in curve.itertuples():
        spec = StudySamplerSpec(str(row.sampler), int(row.nfe_per_trajectory), int(row.N))
        if (row.N != spec.n_trajectories or row.nfe_per_trajectory != spec.nfe
                or row.denoiser_calls_per_chunk != spec.nfe * spec.n_trajectories):
            raise ValueError("曲线采样预算不一致")
    if (curve[list(METRICS[1:])] < 0).any().any() or (curve[METRICS[0]].abs() > 1 + 1e-6).any():
        raise ValueError("指标范围非法")


def local_stability(curve):
    _valid_curve(curve)
    records, stable = [], []
    for (kind, steps), frame in curve.groupby(["sampler", "nfe_per_trajectory"], sort=True):
        rows = frame.set_index("N")
        passed_at = set()
        for n in sorted(rows.index):
            if 2 * n not in rows.index:
                continue
            left, right = rows.loc[n], rows.loc[2 * n]
            relative = {}
            for key in METRICS[1:]:
                a, b = float(left[key]), float(right[key])
                relative[key] = (abs(b - a) / abs(a) if a else (0. if b == 0 else None))
            pcc = abs(float(right[METRICS[0]] - left[METRICS[0]]))
            passed = pcc < .01 and all(v is not None and v < .05 for v in relative.values())
            records.append({"sampler": kind, "S": int(steps), "N": int(n), "next_N": int(2*n),
                            "relative_changes": relative, "pcc_change": pcc, "passed": passed})
            if passed:
                passed_at.add(int(n))
        starts = sorted(n for n in passed_at if 2*n in passed_at)
        stable.append({"sampler": kind, "S": int(steps), "stable_candidate_N": starts[0] if starts else None,
                       "status": "local_stability" if starts else "preliminary" if passed_at else "budget_limited",
                       "max_N": int(rows.index.max())})
    return {"comparisons": records, "status_by_sampler": stable}


def tolerance_candidates(frame):
    """按计划逐层筛选，保留完整候选身份供审核。"""
    result = frame[frame[METRICS[0]] >= frame[METRICS[0]].max() - .01]
    for key in METRICS[1:]:
        best = float(result[key].min())
        result = result[result[key] <= best * 1.05]
    return result.copy()


def choose_configurations(curve, *, require_full_grid=True):
    _valid_curve(curve)
    if require_full_grid:
        expected = {(kind, steps, n) for (kind, steps), counts in GRID.items() for n in counts}
        actual = set(zip(curve.sampler, curve.nfe_per_trajectory, curve.N))
        if actual != expected:
            raise ValueError("选择要求完整首轮矩阵，不能从 partial 结果选配置")
    selected = []
    for kind in ("ddpm", "ddim"):
        frame = curve[curve.sampler == kind]
        if frame.empty:
            raise ValueError("最终选择必须包含 DDPM 与 DDIM")
        candidates = tolerance_candidates(frame)
        # 仅使用全部候选均有的独立计时；共享 prefix 时间不参与独立耗时排序。
        keys, ascending = ["denoiser_calls_per_chunk"], [True]
        timed = ("independent_sampling_seconds" in candidates
                 and np.isfinite(candidates.independent_sampling_seconds.to_numpy(dtype=float)).all())
        if timed:
            if (candidates.independent_sampling_seconds < 0).any():
                raise ValueError("独立采样时间不能为负")
            keys.append("independent_sampling_seconds")
            ascending.append(True)
        keys += [METRICS[0], "nfe_per_trajectory", "N"]
        ascending += [False, True, True]
        row = candidates.sort_values(keys, ascending=ascending).iloc[0]
        selected.append({"sampler": kind, "S": int(row.nfe_per_trajectory), "N": int(row.N),
                         "role": "selected", "independent_timing_used": timed})
    if not any(v["sampler"] == "ddim" and v["S"] == 6 and v["N"] == 16 for v in selected):
        selected.append({"sampler": "ddim", "S": 6, "N": 16, "role": "fixed_reference"})
    return {"protocol": NAME, "selected": selected, "stability": local_stability(curve),
            "selection_noise_seed": 20261003, "selection_rule": "pcc_0.01_rr_tail_5pct_then_cost"}


def choose_training(anchors):
    required = {"baseline", "long_absolute", "long_stretched"}
    if set(anchors.regime) != required:
        raise ValueError("训练选择要求 B0/B1/B2 的完整 anchor")
    combined = []
    for regime, frame in anchors.groupby("regime"):
        _valid_curve(frame)
        if set(zip(frame.sampler, frame.nfe_per_trajectory, frame.N)) != {("ddim", 6, 16), ("ddpm", 50, 4)}:
            raise ValueError("训练选择要求两个固定 anchor")
        combined.append({"regime": regime, **{key: float(frame[key].mean() if i < 2 else frame[key].max())
                                             for i, key in enumerate(METRICS)},
                         "updates": TrainingSpec(regime).updates})
    frame = pd.DataFrame(combined)
    eligible = tolerance_candidates(frame).sort_values(["updates", METRICS[0], "regime"],
                                                       ascending=[True, False, True])
    return {"selected_regime": str(eligible.iloc[0].regime), "anchor_selection_statistics": combined}
