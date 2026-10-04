"""从既有 test CSV 计算 670 排除敏感性；不读取波形或模型。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PRIMARY = (
    "whole_rr_abs_error_bpm", "local_rr_mae_bpm", "envelope_trajectory_mae",
    "global_envelope_modulation_error", "lag_aware_signed_pcc",
)
EXPECTED_SUBJECTS = {"220", "229", "286", "670", "671", "704", "726", "1006"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_spec(columns):
    if set(PRIMARY) <= set(columns):
        return "full180", list(PRIMARY), {
            PRIMARY[0]: "whole_rr_target_eligible",
            PRIMARY[1]: "local_rr_target_eligible",
            PRIMARY[4]: "joint_target_eligible",
        }
    for prefix, task in (("center30_", "center30"), ("center_", "center60"), ("center90_", "center90")):
        metrics = [prefix + x for x in ("rr_mae_bpm", "envelope_trajectory_mae", "global_envelope_modulation_error", "lag_aware_signed_pcc")]
        if set(metrics) <= set(columns):
            return task, metrics, {metrics[0]: prefix + "rr_target_eligible", metrics[-1]: prefix + "joint_target_eligible"}
    return None


def truth(value):
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise ValueError(f"非法资格字段 {value!r}")


def quantile(values, q):
    ordered = sorted(values)
    x = (len(ordered) - 1) * q
    i = math.floor(x)
    return ordered[i] + (ordered[math.ceil(x)] - ordered[i]) * (x - i)


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    sources, skipped, seed_rows, subject_rows, audits, tails = [], [], [], [], [], []
    cells = set()
    identities = {}
    paths = []
    for folder, dirs, files in os.walk(root / "runs"):
        dirs[:] = [d for d in dirs if d not in ("source_snapshot", ".git", "__pycache__") and not (Path(folder) / d).resolve().is_relative_to(output.resolve())]
        for name in files:
            p = Path(folder) / name
            if name.endswith(".csv") and "test" in str(p.relative_to(root)).lower() and "metric" in name and "summary" not in name:
                paths.append(p)
    for path in sorted(paths):
        relative = path.relative_to(root)
        if relative.parts[1] == "w0_test_qualitative_v1":
            skipped.append({"path": str(relative), "reason": "定性展示副本，不重复计入"})
            continue
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            spec = metric_spec(reader.fieldnames or [])
            if not spec or "samp_id" not in (reader.fieldnames or []):
                skipped.append({"path": str(relative), "reason": "非当前逐窗口主指标格式"})
                continue
            rows = list(reader)
        task, metrics, eligibility = spec
        family = relative.parts[1] if task == "full180" else task
        path_seed = re.search(r"seed_?(\d{8})", str(relative))
        groups = defaultdict(list)
        for row in rows:
            if row["split"] != "test":
                raise ValueError(f"非 test 行: {path}")
            arm = row.get("arm") or row.get("candidate_id") or row["method"]
            if task != "full180":
                arm = path.parent.parent.name
            seed = row.get("seed") or (path_seed.group(1) if path_seed else "deterministic")
            groups[arm, seed].append(row)
        sources.append({"path": str(relative), "sha256": digest(path), "rows": len(rows), "cells": len(groups), "family": family, "task": task})
        for (arm, seed), group in groups.items():
            cell = (family, arm, seed)
            if cell in cells:
                raise ValueError(f"重复 cell，需要人工确认来源: {cell}")
            cells.add(cell)
            identity = sorted((int(r["dataset_row_id"]), r["samp_id"]) for r in group)
            assert len(identity) == len(set(x[0] for x in identity)), cell
            assert {r["samp_id"] for r in group} == EXPECTED_SUBJECTS, cell
            assert len(group) == 2310, (cell, len(group))
            identity_hash = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            if task in identities:
                assert identities[task] == identity_hash, cell
            identities[task] = identity_hash
            base = dict(family=family, task=task, arm=arm, seed=seed)
            counts = Counter(r["samp_id"] for r in group)
            assert counts["670"] == 79, (cell, counts)
            # 保留指标定义中的 target eligibility；不按预测质量筛选。
            values = {m: defaultdict(list) for m in metrics}
            for r in group:
                for metric in metrics:
                    flag = eligibility.get(metric)
                    eligible = truth(r[flag]) if flag and flag in r else True
                    raw = r[metric]
                    if not eligible:
                        if raw and math.isfinite(float(raw)):
                            raise ValueError(f"ineligible 主指标有有限值，需核对聚合合同: {cell}/{metric}")
                        continue
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError(f"eligible 主指标非有限: {cell}/{metric}")
                    values[metric][r["samp_id"]].append(value)
            for metric in metrics:
                for subject in sorted(counts, key=int):
                    v = values[metric][subject]
                    if not v:
                        raise ValueError(f"subject 主指标分母为零: {cell}/{subject}/{metric}")
                    subject_rows.append({**base, "samp_id": subject, "metric": metric, "total_n": counts[subject], "eligible_n": len(v), "mean": statistics.fmean(v)})
                for cohort in ("all", "exclude_670", "only_670"):
                    subjects = [s for s in counts if cohort == "all" or (cohort == "exclude_670" and s != "670") or (cohort == "only_670" and s == "670")]
                    v = [x for s in subjects for x in values[metric][s]]
                    for aggregation in ("window", "subject_macro"):
                        mean = statistics.fmean(v) if aggregation == "window" else statistics.fmean(statistics.fmean(values[metric][s]) for s in subjects)
                        seed_rows.append({**base, "cohort": cohort, "aggregation": aggregation, "metric": metric, "total_n": sum(counts[s] for s in subjects), "eligible_n": len(v), "n_subjects": len(subjects), "mean": mean, "row_identity_sha256": identity_hash})
                    if "rr_" in metric:
                        tails.append({**base, "cohort": cohort, "metric": metric, "n": len(v), "median": quantile(v, .5), "p90": quantile(v, .9), "p95": quantile(v, .95), "gt2_fraction": sum(x > 2 for x in v) / len(v), "gt5_fraction": sum(x > 5 for x in v) / len(v)})
            # 与来源原生 summary 逐项对账；混合 cell 的 RTM 表单独保留检查状态。
            summary_path = path.with_name(path.stem + "_summary.csv")
            summary_status = "no_compatible_summary"
            if len(groups) == 1 and summary_path.exists():
                with summary_path.open() as stream:
                    summary = list(csv.DictReader(stream))
                if len(summary) == 1 and all(m + "_mean" in summary[0] for m in metrics):
                    for m in metrics:
                        observed = statistics.fmean(x for vs in values[m].values() for x in vs)
                        assert math.isclose(observed, float(summary[0][m + "_mean"]), rel_tol=1e-10, abs_tol=1e-12), (cell, m)
                    summary_status = "matched"
                    sources.append({"path": str(summary_path.relative_to(root)), "sha256": digest(summary_path), "rows": 1, "cells": 1, "family": family, "task": task})
            audits.append({**base, "source": str(relative), "n_all": len(group), "n_removed": counts["670"], "n_retained": len(group)-counts["670"], "identity_sha256": identity_hash, "summary_check": summary_status})
    across = []
    grouped = defaultdict(list)
    for r in seed_rows:
        grouped[tuple(r[k] for k in ("family", "task", "arm", "cohort", "aggregation", "metric"))].append(r)
    for key, rows in sorted(grouped.items()):
        seeds = sorted(r["seed"] for r in rows)
        assert seeds == ["20260811", "20260812", "20260813"] or seeds == ["deterministic"], (key, seeds)
        v = [r["mean"] for r in rows]
        across.append({**dict(zip(("family", "task", "arm", "cohort", "aggregation", "metric"), key)), "n_seeds": len(v), "mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else "", "n_windows_per_seed": rows[0]["total_n"], "n_subjects": rows[0]["n_subjects"]})
    comparisons = []
    lookup = {(r["family"], r["arm"], r["cohort"], r["aggregation"], r["metric"], r["seed"]): r["mean"] for r in seed_rows}
    family_arms = defaultdict(set)
    for family, arm, seed in cells:
        if seed != "deterministic":
            family_arms[family].add(arm)
    # 对同一实验的所有固定候选列出描述性配对；不依据排除结果选择候选。
    for family, arms in sorted(family_arms.items()):
        for reference, candidate in itertools.combinations(sorted(arms), 2):
            for cohort, aggregation in itertools.product(("all", "exclude_670"), ("window", "subject_macro")):
                ms = sorted({r["metric"] for r in across if r["family"] == family})
                for metric in ms:
                    seeds = ["20260811", "20260812", "20260813"]
                    ref = [lookup[family, reference, cohort, aggregation, metric, s] for s in seeds]
                    cand = [lookup[family, candidate, cohort, aggregation, metric, s] for s in seeds]
                    delta = [c-r for c, r in zip(cand, ref)]
                    higher = metric.endswith("pcc")
                    comparisons.append(dict(family=family, reference=reference, candidate=candidate, cohort=cohort, aggregation=aggregation, metric=metric, delta_mean=statistics.fmean(delta), delta_sd=statistics.stdev(delta), ratio_of_means_pct=(statistics.fmean(cand)/statistics.fmean(ref)-1)*100 if not higher else "", mean_paired_relative_change_pct=statistics.fmean((c/r-1)*100 for c, r in zip(cand,ref)) if not higher else "", improved_seeds=sum(d > 0 if higher else d < 0 for d in delta), n_seeds=3))
    for name, rows in (("per_seed.csv", seed_rows), ("per_subject.csv", subject_rows), ("across_seed.csv", across), ("pairwise_contrasts.csv", comparisons), ("rr_tail_per_seed.csv", tails), ("audit.csv", audits), ("skipped.csv", skipped)):
        write_csv(output / name, rows)
    manifest = {"protocol": "test-subject670-sensitivity-20260930", "created_utc": datetime.now(timezone.utc).isoformat(), "excluded_subject": "670", "sources": sources, "n_cells": len(cells), "families": dict(Counter(c[0] for c in cells)), "script_sha256": digest(Path(__file__)), "summary_check_counts": dict(Counter(r["summary_check"] for r in audits)), "outputs": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("sources", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output_dir.resolve())
