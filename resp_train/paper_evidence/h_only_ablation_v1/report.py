"""五指标与固定三视图；完整矩阵、窗口平均和受试者等权并列。"""
from pathlib import Path
import numpy as np
import pandas as pd
from resp_train.paper_evidence.cwt_apor_v2.summary import tables
from . import artifacts as io
from .spec import ARMS, GAIN_ARMS, plan
from .runtime import cell_result


def report_tables(frame):
    expected = {(c["arm"], c["seed"]) for c in plan()}
    result = tables(frame, expected_cells=expected, pairs=[(a, "HA0") for a in ARMS if a != "HA0"])
    result["paired_summary"] = result["paired_delta"].groupby(["arm", "reference", "metric"], sort=False).degradation.agg(
        mean="mean", seed_sd="std", improved_seeds=lambda x: int((x < 0).sum()),
        worsened_seeds=lambda x: int((x > 0).sum()), tied_seeds=lambda x: int((x == 0).sum())).reset_index()
    macro = result["subject_macro"]
    result["subject_macro_across_seed"] = macro.groupby(["arm", "metric"], sort=False)["mean"].agg(
        seed_mean="mean", seed_sd="std").reset_index()
    for prefix in ("subject_macro", "per_subject"):
        paired = result[prefix + "_paired_delta"]
        keys = ["arm", "reference", "metric"] + (["samp_id"] if prefix == "per_subject" else [])
        result[prefix + "_paired_summary"] = paired.groupby(keys, sort=False).degradation.agg(
            mean="mean", seed_sd="std", improved_seeds=lambda x: int((x < 0).sum()),
            worsened_seeds=lambda x: int((x > 0).sum()), tied_seeds=lambda x: int((x == 0).sum())).reset_index()
    result["subject_direction_counts"] = result["per_subject_paired_delta"].groupby(
        ["arm", "seed", "metric"], sort=False).degradation.agg(
            improved_subjects=lambda x: int((x < 0).sum()), worsened_subjects=lambda x: int((x > 0).sum()),
            tied_subjects=lambda x: int((x == 0).sum())).reset_index()
    return result


def summarize(session, split="val", retry=False, confirmed=False):
    if split == "test":
        from resp_train.paper_evidence.cwt_apor_v2.research_test import guard
        guard(confirmed)
    io.load_session(session)
    frames, sources, gains = [], [], []
    for cell in plan():
        if split == "val":
            item = cell_result(session, **cell)
        elif split == "test":
            from .research_test import test_result
            item = test_result(session, **cell)
        else:
            raise ValueError("未知 split")
        if item is None:
            raise RuntimeError(f"51-cell 矩阵未完成: {cell}")
        path, result = item
        frame = pd.read_csv(result["sources"]["metrics"]["path"])
        if not frame.arm.eq(cell["arm"]).all() or not frame.seed.eq(cell["seed"]).all() or not frame.split.eq(split).all():
            raise ValueError("cell 标签/split 不一致")
        frames.append(frame)
        sources.append({**cell, "path": str(path), "manifest": io.identity(path / "manifest.json")})
        if cell["arm"] in GAIN_ARMS:
            records = io.read_json(Path(result["sources"]["metrics"]["path"]).parent / "gain_per_window.json")
            gain_frame = pd.json_normalize(records)
            if (not np.array_equal(gain_frame.dataset_row_id, frame.dataset_row_id)
                    or not np.array_equal(gain_frame.samp_id, frame.samp_id)
                    or not np.isfinite(gain_frame.select_dtypes(include=np.number).to_numpy()).all()):
                raise ValueError("增益统计行身份/finite不匹配")
            gain_frame.insert(0, "arm", cell["arm"])
            gain_frame.insert(0, "seed", cell["seed"])
            gains.append(gain_frame)
    full = pd.concat(frames, ignore_index=True)
    with io.attempt(session / (split + "_summary"), io.binding(session, split + "_summary"), retry) as output:
        full.to_csv(output / "per_window.csv", index=False)
        pd.concat(gains, ignore_index=True).to_csv(output / "gain_per_window.csv", index=False)
        views = {"full": full}
        if split == "test":
            views.update(exclude670=full[full.samp_id.ne(670)], subject670=full[full.samp_id.eq(670)])
        for view, frame in views.items():
            folder = output / view
            folder.mkdir()
            for name, table in report_tables(frame).items():
                table.to_csv(folder / f"{name}.csv", index=False)
        io.write_json(output / "sources.json", sources)
        io.write_json(output / "groups.json", {"结构消融": list(ARMS)[:12], "损失消融": ["HA0", "HA12"],
                      "增益形式": ["HA0", "HA13", "HA14", "HA15"], "64/65宽度对照": ["HA0", "HA16"]})
    return output
