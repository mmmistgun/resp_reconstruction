"""完整84-cell的配对统计、μ×voices及池化位置2×2效应。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from scripts import m4_s1_cwt_parameters_spec as spec


def comparisons():
    pairs = [("baseline", f"{a}_vs_B", a, "B") for a in spec.ARMS if a != "B"]
    extra = [
        ("frequency_cap", f"{a}_vs_{b}", a, b)
        for a, b in zip(("C_2", "C_4", "B", "C_12", "C_20"), ("C_0p8", "C_2", "C_4", "B", "C_12"))
    ]
    extra += [("bands", f"{a}_vs_{b}", a, b) for a, b in (("B", "H"), ("B", "C_0p8"), ("H", "C_0p8"))]
    extra += [("pooling", "P_025_vs_P_100", "P_025", "P_100")]
    extra += [("modules", f"{a}_vs_{b}", a, b) for a, b in spec.spec()["analysis"]["module_additional_pairs"]]
    extra += [("pool_position_conditional", name, a, b) for name, (a, b) in spec.POOL_PAIRS.items()]
    # 相邻μ和voices比较覆盖整个Q网格，与交互项共同描述参数敏感性。
    def q(mu, voices):
        return "B" if (mu, voices) == (13.4, 12) else f"Q_mu{str(mu).replace('.', 'p')}_v{voices}"
    for mu in (6, 13.4, 20):
        for left, right in ((8, 4), (12, 8), (24, 12)):
            a, b = q(mu, left), q(mu, right)
            extra.append(("voices", f"{a}_vs_{b}", a, b))
    for voices in (4, 8, 12, 24):
        for left, right in ((13.4, 6), (20, 13.4)):
            a, b = q(left, voices), q(right, voices)
            extra.append(("mu", f"{a}_vs_{b}", a, b))
    return pairs + extra


def effect_coefficients():
    terms = {f"pool_position_{name}": weights for name, weights in spec.POOL_EFFECTS.items()}
    for mu in (6, 20):
        for voices in (4, 8, 24):
            name = f"mu_voices_mu{mu}_v{voices}"
            terms[name] = {f"Q_mu{mu}_v{voices}": 1, f"Q_mu13p4_v{voices}": -1,
                           f"Q_mu{mu}_v12": -1, "B": 1}
    return terms


def paired_table(means, *, subject=False):
    keys = ["arm", "seed", "metric"] + (["samp_id"] if subject else [])
    indexed = means.set_index(keys)["mean"]
    if indexed.index.has_duplicates or not np.isfinite(indexed.to_numpy(float)).all():
        raise ValueError("配对均值重复或非有限")
    rows = []
    subjects = sorted(means.samp_id.unique()) if subject else [None]
    for group, name, arm, reference in comparisons():
        for seed in spec.SEEDS:
            for metric in sf.PRIMARY:
                for sid in subjects:
                    suffix = (seed, metric) + ((sid,) if subject else ())
                    left, right = (arm, *suffix), (reference, *suffix)
                    if left not in indexed or right not in indexed:
                        raise ValueError("配对缺少指标或受试者")
                    delta = float(indexed[left] - indexed[right])
                    record = {"group": group, "comparison": name, "arm": arm, "reference": reference,
                        "seed": seed, "metric": metric, "delta": delta,
                        "improvement": delta if metric == sf.PCC else -delta}
                    if subject:
                        record["samp_id"] = sid
                    rows.append(record)
    return pd.DataFrame(rows)


def pair_summary(pairs, *, subject=False):
    keys = ["group", "comparison", "arm", "reference", "metric"] + (["samp_id"] if subject else [])
    result = pairs.groupby(keys).agg(delta_mean=("delta", "mean"), delta_sd=("delta", "std"),
        improvement_mean=("improvement", "mean"), improved_seeds=("improvement", lambda s: int((s > 0).sum())),
        worsened_seeds=("improvement", lambda s: int((s < 0).sum())),
        tied_seeds=("improvement", lambda s: int((s == 0).sum())),
        improved_at_threshold_seeds=("improvement", lambda s: int((s >= .01).sum())),
        worsened_at_threshold_seeds=("improvement", lambda s: int((s <= -.01).sum()))).reset_index()
    result["threshold"] = .01
    result["mean_classification"] = np.where(result.improvement_mean.abs() < .01, "no_obvious_mean_difference",
                                             np.where(result.improvement_mean > 0, "improved", "worsened"))
    return result


def effect_tables(means):
    indexed = means.set_index(["arm", "seed", "metric"])["mean"]
    rows = []
    for name, coefficients in effect_coefficients().items():
        for seed in spec.SEEDS:
            for metric in sf.PRIMARY:
                values = [indexed[arm, seed, metric] * weight for arm, weight in coefficients.items()]
                if not np.isfinite(values).all():
                    raise FloatingPointError("效应项非有限")
                rows.append({"effect": name, "seed": seed, "metric": metric, "value": float(sum(values))})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["effect", "metric"]).agg(mean=("value", "mean"), sd=("value", "std"),
        positive_seeds=("value", lambda s: int((s > 0).sum())),
        negative_seeds=("value", lambda s: int((s < 0).sum())),
        zero_seeds=("value", lambda s: int((s == 0).sum()))).reset_index()
    return detail, summary


def summary_tables(frame):
    expected = {(a, s) for a in spec.ARMS for s in spec.SEEDS}
    if set(frame[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None)) != expected:
        raise ValueError("汇总要求完整28配置×3seed，共84个cell")
    first, records = None, []
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=False):
        group = group.reset_index(drop=True)
        identity_columns = ["dataset_row_id", "samp_id", "split"]
        if set(group.split) != {"val"}:
            raise ValueError("汇总只允许validation")
        sf.validate_metrics(group, group[identity_columns])
        identity = group[identity_columns + sorted(c for c in group if c.endswith("_target_eligible")
                                                    or c == "local_rr_target_eligible_windows")]
        if first is None:
            first = identity
        elif not first.equals(identity):
            raise ValueError("跨cell样本顺序、身份或target资格变化")
        for metric in sf.PRIMARY:
            values = group.loc[sf._metric_mask(group, metric), metric].to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError("主指标缺少有效有限值")
            records.append({"arm": arm, "name": spec.ARMS[arm].name, "seed": seed, "metric": metric,
                            "mean": float(values.mean()), "n": len(values)})
    per_seed = pd.DataFrame(records)
    subjects = sf.subject_stratified_metrics(frame)
    macro = subjects.groupby(["arm", "seed", "metric"])["mean"].mean().reset_index()
    # 禁止某个臂因缺少受试者指标而静默改变宏平均人群。
    subject_pairs = paired_table(subjects, subject=True)
    tables = {"per_seed": per_seed,
        "across_seed": per_seed.groupby(["arm", "name", "metric"])["mean"].agg(seed_mean="mean", seed_sd="std").reset_index(),
        "per_subject": subjects, "subject_macro": macro,
        "subject_macro_across_seed": macro.groupby(["arm", "metric"])["mean"].agg(seed_mean="mean", seed_sd="std").reset_index(),
        "per_subject_paired_delta": subject_pairs,
        "per_subject_paired_summary": pair_summary(subject_pairs, subject=True),
        "local_rr_tail": sf.local_rr_tail_summary(frame), "denominators": sf.metric_denominators(frame)}
    for prefix, source in (("", per_seed), ("subject_macro_", macro)):
        paired = paired_table(source)
        tables[prefix + "paired_delta"] = paired
        tables[prefix + "paired_summary"] = pair_summary(paired)
        tables[prefix + "effects"], tables[prefix + "effect_summary"] = effect_tables(source)
    return tables


def write_report(output, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    lines = ["# M4-S1 CWT参数与模块消融：validation", "",
        "28配置×seeds 20260812/20260813/20260814，全部从头训练；checkpoint按完整validation Local RR严格最小、并列最早选择。",
        "窗口加权为主视图，受试者宏平均为诊断；SD为训练seed样本标准差。各臂结果属于开发阶段证据。",
        "配对delta为候选减参照，误差负值/PCC正值为改善。0.01为原生单位上的描述性阈值，不表示统计等效。",
        "池化位置效应和μ×voices交互先按seed计算；效应符号结合各臂原值解释。完整数据见同目录CSV。", ""]
    for prefix, title in (("", "窗口加权"), ("subject_macro_", "受试者宏平均")):
        table = tables[prefix + "across_seed"].set_index(["arm", "metric"])
        lines += [f"## {title}", "", "| 配置 | " + " | ".join(sf.PRIMARY) + " |",
                  "|---|" + "---:|" * len(sf.PRIMARY)]
        for arm in spec.ARMS:
            values = [f"{table.loc[(arm, m), 'seed_mean']:.6f} ± {table.loc[(arm, m), 'seed_sd']:.6f}"
                      for m in sf.PRIMARY]
            lines.append(f"| {arm} | " + " | ".join(values) + " |")
        lines.append("")
        paired = tables[prefix + "paired_summary"]
        paired = paired[paired.group == "baseline"].set_index(["arm", "metric"])
        arms = [a for a in spec.ARMS if a != "B"]
        fig, axes = plt.subplots(1, 5, figsize=(18, 10), sharey=True, constrained_layout=True)
        for axis, metric in zip(axes, sf.PRIMARY):
            delta = [paired.loc[(a, metric), "delta_mean"] for a in arms]
            sd = [paired.loc[(a, metric), "delta_sd"] for a in arms]
            axis.errorbar(delta, np.arange(len(arms)), xerr=sd, fmt="o", markersize=3)
            axis.axvline(0, color="grey", linewidth=.8)
            axis.set_title(metric.replace("_", " "), fontsize=8)
            axis.set_yticks(np.arange(len(arms)), arms, fontsize=7)
            axis.grid(axis="x", alpha=.2)
        axes[0].invert_yaxis()
        fig.savefig(output / f"{prefix}baseline_deltas.png", dpi=150)
        plt.close(fig)
    lines += ["## 池化位置主效应与交互（窗口加权）", "",
              "| 效应 | 指标 | mean ± SD | 正/负/零seed数 |", "|---|---|---:|---|"]
    for row in tables["effect_summary"].itertuples(index=False):
        if row.effect.startswith("pool_position_"):
            lines.append(f"| {row.effect} | {row.metric} | {row.mean:+.6f} ± {row.sd:.6f} | "
                         f"{row.positive_seeds}/{row.negative_seeds}/{row.zero_seeds} |")
    lines += ["", "本报告保留全部配置及预设比较；模型默认配置的采用需单独决策。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
