"""核验 670 敏感性统计，导出重点对比、完整表和科研图。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from analyze_test_subject670_sensitivity import PRIMARY, digest, write_csv

SEEDS = ("20260811", "20260812", "20260813")
E7 = "e7_scale_encoding_aggregation"
E8 = "e8_film_decoder_redesign_v1"
E9 = "e9_latent_width_condition_refiner_v1"
SF = "w0_structural_factorial_v1_es30p15"
AP = "patch_apor_v1"
W0 = ("crd_tf_v1", "crd_tf102_w")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def build(root, analysis, output, appendix):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((analysis / "manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        assert digest(analysis / name) == expected, name
    for source in manifest["sources"]:
        assert digest(root / source["path"]) == source["sha256"], source["path"]
    seed_rows = read_csv(analysis / "per_seed.csv")
    across = read_csv(analysis / "across_seed.csv")
    subjects = read_csv(analysis / "per_subject.csv")
    lookup = {(r["family"], r["arm"], r["cohort"], r["aggregation"], r["metric"], r["seed"]): float(r["mean"]) for r in seed_rows}
    means = {(r["family"], r["arm"], r["cohort"], r["aggregation"], r["metric"]): float(r["mean"]) for r in across}
    per_subject = defaultdict(list)
    for r in subjects:
        per_subject[r["family"], r["arm"], r["seed"], r["metric"]].append(r)
    # 用 subject 均值和分母独立重构直接平均，验证两种聚合及排除集合。
    max_residual = 0.0
    for r in seed_rows:
        sub = per_subject[r["family"], r["arm"], r["seed"], r["metric"]]
        sub = [s for s in sub if r["cohort"] == "all" or (r["cohort"] == "exclude_670" and s["samp_id"] != "670") or (r["cohort"] == "only_670" and s["samp_id"] == "670")]
        n = sum(int(s["eligible_n"]) for s in sub)
        observed = sum(float(s["mean"]) * int(s["eligible_n"]) for s in sub) / n if r["aggregation"] == "window" else sum(float(s["mean"]) for s in sub) / len(sub)
        residual = abs(observed - float(r["mean"]))
        max_residual = max(max_residual, residual)
        assert residual < 1e-12
        assert n == int(r["eligible_n"])
    additional_summary = []
    for r in read_csv(analysis / "audit.csv"):
        if r["summary_check"] == "matched":
            continue
        parent = (root / r["source"]).parent
        p = parent / ("research_test_seed_summary.csv" if r["family"] == "resp_temporal_v1" else "summary.csv")
        summary_rows = read_csv(p)
        row = next(x for x in summary_rows if x.get("candidate_id", r["arm"]) == r["arm"] and x.get("seed", r["seed"]) == r["seed"])
        for m in PRIMARY:
            value = row[m] if m in row else row[m + "_mean"]
            assert math.isclose(float(value), lookup[r["family"], r["arm"], "all", "window", m, r["seed"]], rel_tol=1e-10, abs_tol=1e-12)
        additional_summary.append({"path": str(p.relative_to(root)), "sha256": digest(p), "arm": r["arm"], "seed": r["seed"]})
    receipt_checks = []
    for source in manifest["sources"]:
        p = root / source["path"]
        if "summary" in p.name:
            continue
        checked = []
        for name in ("manifest.json", "artifact_manifest.json", "research_test_manifest.json"):
            receipt = p.with_name(name)
            if not receipt.exists():
                continue
            content = json.loads(receipt.read_text())
            item = content.get("files", {}).get(p.name) if isinstance(content.get("files", {}), dict) else None
            expected = item.get("sha256") if isinstance(item, dict) else item
            if expected:
                assert expected == source["sha256"], p
                checked.append({"path": str(receipt.relative_to(root)), "sha256": digest(receipt)})
        receipt_checks.append({"source": source["path"], "matched_receipts": checked})

    focused = []
    pairs = []
    def pair(label, cf, ca, rf, ra):
        pairs.append((label, (cf, ca), (rf, ra)))
    for family in ("e2_w0_effort_test_v1", "e4_scale_aggregation_v2_test", "e4_w0_scale_aggregation_test_v1", "e5_temporal_frontend_test", "e6_temporal_frontend_test", "w0_film_gamma_test_v1", "crd_tf_w_v2", "crd_tf_v1"):
        for arm in sorted({r["arm"] for r in across if r["family"] == family}):
            if (family, arm) != W0:
                pair(arm + " vs historical W0", family, arm, *W0)
    for fam, ref in ((E7, "s0_shallow__mean"), (E8, "e8_fill65_pointwise"), (E9, "e9a_d96_h64"), (SF, "sfv1_patch_tm3_ref2"), (AP, "M0")):
        for arm in sorted({r["arm"] for r in across if r["family"] == fam}):
            if arm != ref:
                pair(arm + " vs " + ref, fam, arm, fam, ref)
    for arm in ("A1", "A2", "A3", "A4"):
        pair(arm + " vs A0", AP, arm, AP, "A0")
    pair("E9 H64 vs H65", E9, "e9a_d96_h64", E9, "e9a_d96_h65")
    for label, cand, ref in pairs:
        for cohort in ("all", "exclude_670"):
            for agg in ("window", "subject_macro"):
                for m in PRIMARY:
                    cv = [lookup[*cand, cohort, agg, m, s] for s in SEEDS]
                    rv = [lookup[*ref, cohort, agg, m, s] for s in SEEDS]
                    higher = m == PRIMARY[-1]
                    delta = [c-r for c, r in zip(cv, rv)]
                    pct = (st.fmean(cv)/st.fmean(rv)-1)*100
                    paired = st.fmean((c/r-1)*100 for c,r in zip(cv,rv))
                    focused.append(dict(comparison=label, candidate_family=cand[0], candidate=cand[1], reference_family=ref[0], reference=ref[1], cohort=cohort, aggregation=agg, metric=m, candidate_mean=st.fmean(cv), reference_mean=st.fmean(rv), delta=st.fmean(delta), ratio_of_means_pct=pct if not higher else "", mean_paired_relative_change_pct=paired if not higher else "", improved_seeds=sum(d > 0 if higher else d < 0 for d in delta), materially_improved_seeds=sum(d > .002 for d in delta) if higher else sum((c/r-1) < -.005 for c,r in zip(cv,rv)), materially_worsened_seeds=sum(d < -.002 for d in delta) if higher else sum((c/r-1) > .005 for c,r in zip(cv,rv))))
    write_csv(output / "focused_contrasts.csv", focused)
    changes = []
    for r in across:
        if r["cohort"] != "exclude_670":
            continue
        key = (r["family"], r["arm"])
        m, agg = r["metric"], r["aggregation"]
        old = means[*key, "all", agg, m]
        new = float(r["mean"])
        changes.append(dict(family=key[0], arm=key[1], aggregation=agg, metric=m, all_mean=old, exclude_670_mean=new, exclude_670_sd=r["sd"], only_670_mean=means[*key, "only_670", agg, m], change=new-old, relative_change_pct=(new/old-1)*100, n_seeds=r["n_seeds"]))
    write_csv(output / "before_after.csv", changes)

    lines = ["# 排除受试者 670：全部实验指标附表", "", "2026-09-30。各单元格为完整 test → 排除 670 的跨 seed 均值；完整 mean/SD、资格分母、逐 subject 与同 seed 对比见机器可读产物。RR 单位 bpm，前三/四个误差指标越低越好，PCC 越高越好。", "", "每个 cell 均为 2310→2231 窗口、8→7 个受试者。除 F0/IEWT 为确定性单次基线，其余均为三个训练 seed 等权。窗口与受试者等权分表；不同任务和不同实验轮次不构成统一模型排名。", ""]
    families = sorted({r["family"] for r in across})
    for family in families:
        arms = sorted({r["arm"] for r in across if r["family"] == family})
        if family.startswith("center"):
            prefix = "center_" if family == "center60" else family + "_"
            ms = [prefix+x for x in ("rr_mae_bpm", "envelope_trajectory_mae", "global_envelope_modulation_error", "lag_aware_signed_pcc")]
            headers = ["RR", "轨迹", "调制", "PCC"]
        else:
            ms = PRIMARY
            headers = ["Whole RR", "Local RR", "轨迹", "调制", "PCC"]
        lines += ["## " + family, ""]
        for agg, name in (("window", "窗口直接平均"), ("subject_macro", "受试者等权")):
            lines += ["### " + name, "", "| Arm | " + " | ".join(headers) + " |", "|---|" + "---:|"*len(ms)]
            for arm in arms:
                vals = [f"{means[family,arm,'all',agg,m]:.6f} → {means[family,arm,'exclude_670',agg,m]:.6f}" for m in ms]
                lines.append("| " + arm + " | " + " | ".join(vals) + " |")
            lines.append("")
    with appendix.open("x") as stream:
        stream.write("\n".join(lines))

    # 科研图使用标准绘图库；每个点均来自同 seed 对比的跨 seed 相对变化均值。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [
        ("E8 res96/temporal vs W0", "e8_res96_temporal vs e8_fill65_pointwise"),
        ("E8 res96/single vs W0", "e8_res96_single vs e8_fill65_pointwise"),
        ("E9 D64-H64 vs D96-H64", "e9b_d64_h64 vs e9a_d96_h64"),
        ("E9 H64 vs H65", "E9 H64 vs H65"),
        ("E7 S1-MEAN vs S0-MEAN", "s1_deep_local__mean vs s0_shallow__mean"),
        ("APOR A0 vs M0", "A0 vs M0"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for ax, metric, title in zip(axes, PRIMARY[:2], ("Whole RR", "Local RR")):
        for i, (_, label) in enumerate(labels):
            values = [next(r["mean_paired_relative_change_pct"] for r in focused if r["comparison"] == label and r["cohort"] == co and r["aggregation"] == "window" and r["metric"] == metric) for co in ("all", "exclude_670")]
            ax.plot(values, [i,i], color="#b6b9c0", zorder=1)
            ax.scatter(values[0], i, color="#c47c42", label="All 8 subjects" if i == 0 else None, s=50, zorder=2)
            ax.scatter(values[1], i, color="#167d9a", label="Excluding subject 670" if i == 0 else None, s=50, zorder=2)
        ax.axvline(0, color="#444444", linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel("Paired-seed relative error change (%)\nnegative = candidate improves")
        ax.grid(axis="x", alpha=.2)
        ax.set_yticks(range(len(labels)), [x[0] for x in labels])
        ax.invert_yaxis() if ax is axes[0] else None
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Subject 670 sensitivity: fixed-checkpoint research-test comparisons")
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "rr_comparison_sensitivity.png", dpi=180)
    fig.savefig(output / "rr_comparison_sensitivity.pdf")
    plt.close(fig)
    verification = {"analysis_manifest_sha256": digest(analysis / "manifest.json"), "all_source_hashes_verified": len(manifest["sources"]), "native_summary_cells_matched": manifest["summary_check_counts"]["matched"] + len(additional_summary), "additional_summary_checks": additional_summary, "per_seed_aggregation_rows_verified": len(seed_rows), "max_reaggregation_absolute_residual": max_residual, "receipt_hash_matches": sum(bool(r["matched_receipts"]) for r in receipt_checks), "receipt_checks": receipt_checks, "script_sha256": digest(Path(__file__)), "appendix": {"path": str(appendix.relative_to(root)), "sha256": digest(appendix)}, "outputs": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in verification.items() if k not in ("additional_summary_checks", "receipt_checks", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--appendix", type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.analysis.resolve(), args.output_dir.resolve(), args.appendix.resolve())
