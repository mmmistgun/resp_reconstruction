"""完整矩阵配对汇总；保持 target-only 资格、分母和受试者单位。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from . import artifacts as io
from .spec import plan, comparisons


def paired_table(table, pairs, keys):
    records = []
    for candidate, reference in pairs:
        left = table[table.arm.eq(candidate)]
        right = table[table.arm.eq(reference)]
        merged = left.merge(right, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one", how="outer", indicator=True)
        if not merged._merge.eq("both").all():
            raise ValueError("配对指标/seed/subject 集合不一致")
        for row in merged.to_dict("records"):
            base, value = row["mean_reference"], row["mean_candidate"]
            raw = value-base
            degradation = -raw if row["metric"] == sf.PCC else raw
            records.append({**{k: row[k] for k in keys}, "arm": candidate, "reference": reference,
                            "candidate_mean": value, "reference_mean": base, "raw_delta": raw,
                            "degradation": degradation, "relative_degradation_percent":
                            (100*degradation/base if base > 0 and row["metric"] != sf.PCC else np.nan),
                            "relative_defined": bool(base > 0 and row["metric"] != sf.PCC)})
    return pd.DataFrame(records)


def tables(frame, expected_cells=None, pairs=None):
    expected = expected_cells if expected_cells is not None else {(c["arm"], c["seed"]) for c in plan()}
    observed = set(frame[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("汇总要求完整预定义矩阵")
    per_seed, base_identity = [], None
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=False):
        group = group.reset_index(drop=True)
        sf.validate_metrics(group, group[["dataset_row_id", "samp_id", "split"]])
        flags = sorted(set(sf.ELIGIBILITY.values()) | {c for c in group if c.endswith("_target_eligible")})
        row_identity = group[["dataset_row_id", "samp_id", "split", *flags]]
        if base_identity is None:
            base_identity = row_identity
        elif not row_identity.equals(base_identity):
            raise ValueError("跨 cell 的样本/资格/顺序不同")
        for metric in sf.PRIMARY:
            values = group.loc[sf._metric_mask(group, metric), metric].to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError("主指标为空或非有限")
            per_seed.append({"arm": arm, "seed": seed, "metric": metric, "mean": values.mean(), "n": len(values)})
    per_seed = pd.DataFrame(per_seed)
    across = per_seed.groupby(["arm", "metric"], sort=False)["mean"].agg(seed_mean="mean", seed_sd="std").reset_index()
    pairs = comparisons() if pairs is None else pairs
    paired = paired_table(per_seed, pairs, ["seed", "metric"])
    paired_summary = paired.groupby(["arm", "reference", "metric"], sort=False).degradation.agg(
        mean="mean", seed_sd="std", improved_seeds=lambda x: int((x < 0).sum())).reset_index()
    subjects = sf.subject_stratified_metrics(frame)
    macro = subjects.groupby(["arm", "seed", "metric"], sort=False)["mean"].mean().reset_index()
    return {"per_seed": per_seed, "across_seed": across, "paired_delta": paired, "paired_summary": paired_summary,
            "per_subject": subjects, "subject_macro": macro,
            "per_subject_paired_delta": paired_table(subjects, pairs, ["seed", "samp_id", "metric"]),
            "subject_macro_paired_delta": paired_table(macro, pairs, ["seed", "metric"]),
            "local_rr_tail": sf.local_rr_tail_summary(frame), "denominators": sf.metric_denominators(frame)}


def stratified_time_compression(frame, changes, thresholds):
    subset = frame[frame.arm.isin(["A0", "P_025", "P_100"])].merge(
        changes[["dataset_row_id", "rr_change", "envelope_change"]], on="dataset_row_id", how="left", validate="many_to_one", indicator=True)
    if not subset._merge.eq("both").all():
        raise ValueError("缺少参考分层身份")
    records = []
    for attribute in ("rr_change", "envelope_change"):
        values = subset[attribute].to_numpy(float)
        low, high = thresholds[attribute]["cuts"]
        labels = np.where(~np.isfinite(values), "undefined", np.where(values < low, "low", np.where(values < high, "medium", "high")))
        for (arm, seed), group in subset.assign(stratum=labels).groupby(["arm", "seed"], sort=False):
            for label in ("low", "medium", "high", "undefined"):
                part = group[group.stratum.eq(label)]
                for metric in sf.PRIMARY:
                    valid = part.loc[sf._metric_mask(part, metric), metric].to_numpy(float)
                    if not np.isfinite(valid).all():
                        raise FloatingPointError("分层指标非有限")
                    records.append({"attribute": attribute, "stratum": label, "arm": arm, "seed": seed, "metric": metric,
                                    "mean": valid.mean() if len(valid) else np.nan, "eligible_n": len(valid),
                                    "window_n": len(part), "subject_n": part.samp_id.nunique()})
    return pd.DataFrame(records)


def summarize(session, retry=False):
    from .runtime import cell_result
    io.load_session(session)
    key = io.binding(session, "summary")
    prior = io.completed(session / "summary", key)
    if prior:
        return prior
    frames, sources = [], []
    for cell in plan():
        result = cell_result(session, **cell)
        if result is None:
            raise RuntimeError(f"60-cell 矩阵未完成: {cell}")
        path, record = result
        frame = pd.read_csv(record["sources"]["metrics"]["path"])
        if not frame.arm.eq(cell["arm"]).all() or not frame.seed.eq(cell["seed"]).all():
            raise ValueError("cell 标签不符")
        frames.append(frame)
        sources.append({**cell, "path": str(path), "manifest": io.identity(path / "manifest.json"), "result": record})
    full = pd.concat(frames, ignore_index=True)
    with io.attempt(session / "summary", key, retry) as output:
        for name, table in tables(full).items():
            table.to_csv(output / f"{name}.csv", index=False)
        from .reuse import signals_receipt
        signals = signals_receipt(session)
        changes = pd.read_csv(signals / "reference_changes.csv")
        analysis = io.read_json(signals / "analysis.json")
        stratified_time_compression(full, changes[changes.split.eq("val")], analysis["thresholds"]).to_csv(output / "time_compression_strata.csv", index=False)
        io.write_json(output / "sources.json", sources)
        io.write_json(output / "completion.json", {"cells": 60, "research_test_used": False,
                      "comparison_direction": "error=candidate-reference; PCC=reference-candidate"})
    return output
