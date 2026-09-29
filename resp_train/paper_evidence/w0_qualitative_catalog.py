"""保存产物的窗口检索、案例选择和 E4 证据导航。"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from resp_train.paper_evidence.w0_test_qualitative import PRIMARY, PROTOCOL, sha256, write_json
from resp_train.paper_evidence.w0_test_qualitative_runtime import finish, verify

ANALYSIS_PROTOCOL = "w0-qualitative-analysis-v1-20260927"


class Bundle:
    """只允许读取完成清单中的文件，每次消费验证相应文件身份。"""
    def __init__(self, root: Path, phase: str):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "artifact_manifest.json").read_text())
        if self.manifest.get("status") != "complete" or self.manifest.get("protocol") != PROTOCOL:
            raise ValueError("需要本协议已完成产物")
        self.receipt = json.loads(self.path("receipt.json").read_text())
        if self.receipt.get("phase") != phase:
            raise ValueError(f"产物阶段应为 {phase}")

    def path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if self.root not in path.parents:
            raise ValueError("产物路径越界")
        verify(path, self.manifest["files"][name])
        return path

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(name))

    def identity(self) -> dict:
        return {"path": str(self.root), "manifest_sha256": sha256(self.root / "artifact_manifest.json")}


def selection_rows(cases: Path, source: Path) -> list[int]:
    bundle = Bundle(cases, "case_selection")
    source = source.resolve()
    identity = bundle.receipt["export_source"]
    if identity != {"path": str(source), "manifest_sha256": sha256(source / "artifact_manifest.json")}:
        raise ValueError("案例清单与导出来源不匹配")
    rows = bundle.csv("selected_cases.csv").dataset_row_id.astype(int).tolist()
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("案例清单为空或重复")
    return rows


def select_frame(frame: pd.DataFrame, *, quality: str, strata: list[str], subjects: list[int] | None,
                 per_subject: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个主体×分层按参考调制量距组内中位数选例，row id 打破并列。"""
    if quality not in ("zero-markers", "all") or not strata or set(strata) - {"low", "medium", "high"}:
        raise ValueError("质量/分层参数无效")
    if per_subject < 1 or frame.dataset_row_id.duplicated().any():
        raise ValueError("per-subject 必须为正，窗口身份必须唯一")
    required = ["transient_motion_ratio", "posture_transition_ratio", "target_envelope_modulation"]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError("质量标记或参考调制量缺失")
    mask = frame.envelope_target_stratum.isin(strata)
    if quality == "zero-markers":
        mask &= frame.transient_motion_ratio.eq(0) & frame.posture_transition_ratio.eq(0)
    if subjects is not None:
        if not set(subjects) <= set(frame.samp_id):
            raise ValueError("指定主体不存在")
        mask &= frame.samp_id.isin(subjects)
    candidates = frame.loc[mask].copy()
    if candidates.empty:
        raise ValueError("筛选结果为空")
    groups = ["samp_id", "envelope_target_stratum"]
    candidates["group_target_median"] = candidates.groupby(groups).target_envelope_modulation.transform("median")
    candidates["distance_to_median"] = abs(candidates.target_envelope_modulation - candidates.group_target_median)
    # 数学上等距的两侧值可能因浮点减法相差末位；排序精度固定，原始距离仍保留。
    candidates["ranking_distance"] = candidates.distance_to_median.round(12)
    candidates = candidates.sort_values([*groups, "ranking_distance", "dataset_row_id"], kind="stable")
    candidates["selection_rank"] = candidates.groupby(groups).cumcount() + 1
    return candidates, candidates.loc[candidates.selection_rank <= per_subject].copy()


def table_page(frame: pd.DataFrame, title: str, links: dict[int, list[tuple[str, str]]] | None = None) -> str:
    """本地 HTML，无后台依赖；下拉筛选与列排序不改写选择清单。"""
    columns = list(frame.columns)
    options = []
    for col in ("samp_id", "envelope_target_stratum", "zero_markers", "seed", "condition"):
        if col in columns:
            values = sorted(frame[col].astype(str).unique())
            options.append(f'<label>{html.escape(col)} <select data-col="{columns.index(col)}" onchange="filterRows()">'
                           '<option value="">全部</option>' + ''.join(
                               f'<option>{html.escape(v)}</option>' for v in values) + '</select></label>')
    heads = ''.join(f'<th onclick="sortRows({i})">{html.escape(c)} ↕</th>' for i, c in enumerate(columns))
    body = []
    for position, (_, row) in enumerate(frame.iterrows()):
        cells = ''.join(f'<td>{html.escape(str(v))}</td>' for v in row)
        if links is not None:
            cells += '<td>' + ' '.join(f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'
                                      for label, url in links.get(position, [])) + '</td>'
        body.append('<tr>' + cells + '</tr>')
    return ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
            '<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}th{cursor:pointer}label,a{margin-right:12px}</style>'
            '<h1>' + html.escape(title) + '</h1><p>点击列标题排序；页面筛选不改变已保存案例清单。</p>'
            + ''.join(options) + '<input id="query" placeholder="搜索" oninput="filterRows()">'
            '<table><thead><tr>' + heads + ('<th>查看</th>' if links is not None else '')
            + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table>'
            '<script>function filterRows(){let q=document.getElementById("query").value;let ss=[...document.querySelectorAll("select")];document.querySelectorAll("tbody tr").forEach(r=>{r.hidden=!r.textContent.includes(q)||ss.some(s=>s.value&&r.cells[+s.dataset.col].textContent!==s.value)})}'
            'let direction={};function sortRows(i){direction[i]=-(direction[i]||-1);let b=document.querySelector("tbody");[...b.rows].sort((a,c)=>{let x=a.cells[i].textContent,y=c.cells[i].textContent;return direction[i]*(x!==""&&y!==""&&Number.isFinite(+x)&&Number.isFinite(+y)?+x-+y:x.localeCompare(y))}).forEach(r=>b.append(r))}</script></html>')


def evidence_index(root: Path) -> tuple[pd.DataFrame, list[dict]]:
    """读取 E4 冻结汇总与预选窗口元数据，不重算评价或解码信号。"""
    records, sources = [], []
    manifests = sorted((root / "evaluation").glob("seed_*/*/manifest.json"))
    for manifest_path in manifests:
        directory = manifest_path.parent
        freeze = json.loads((directory / "freeze_receipt.json").read_text())
        verify(manifest_path, freeze["manifest"])
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "completed" or manifest.get("phase") != "evaluation":
            continue
        if manifest.get("implementation_lock_sha256") != "eb15d3e67d9309f22e219361c18611a7eb7b6615b5b5c84f35fcbd1b118115a6":
            continue
        sources.append({"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)})
        verify(directory / "gn_statistics.csv", manifest["files"]["gn_statistics.csv"])
        frames = {}
        for relative, identity in manifest["files"].items():
            path = Path(relative)
            if len(path.parts) != 3 or path.parts[0] != "examples" or path.suffix != ".npz":
                continue
            row_id = int(path.parts[1].removeprefix("row_"))
            condition = path.stem
            metric_file = f"{condition}/metrics.csv"
            if condition not in frames:
                verify(directory / metric_file, manifest["files"][metric_file])
                frame = pd.read_csv(directory / metric_file)
                if frame.dataset_row_id.duplicated().any():
                    raise ValueError("E4 指标窗口重复")
                frames[condition] = frame.set_index("dataset_row_id")
            if not (directory / path).is_file() or (directory / path).stat().st_size != identity["size_bytes"]:
                raise ValueError(f"E4 示例缺失或大小漂移: {path}")
            row = frames[condition].loc[row_id]
            records.append({"seed": manifest["seed"], "split": "val", "dataset_row_id": row_id,
                            "samp_id": int(row.samp_id), "condition": condition,
                            **{key: float(row[key]) for key in PRIMARY},
                            "example_path": str((directory / path).resolve()), "example_sha256": identity["sha256"],
                            "example_verification": "manifest identity and size", "metrics_path": str((directory / metric_file).resolve()),
                            "gn_statistics_path": str((directory / 'gn_statistics.csv').resolve())})
    if not records or {r["seed"] for r in records} != {20260811, 20260812, 20260813}:
        raise ValueError("E4 三 seed 完成来源不完整")
    frame = pd.DataFrame(records)
    if frame.duplicated(["seed", "dataset_row_id", "condition"]).any():
        raise ValueError("E4 同一 seed 存在重复完成来源")
    return frame, sources


def create_index(source: Path, output: Path, *, figures: Path | None, e4_root: Path | None, command: str) -> Path:
    bundle = Bundle(source, "export")
    rows = bundle.csv("test_rows.csv")
    frame = bundle.csv("window_index.csv")
    metrics = bundle.csv("metrics.csv")
    quality = ["dataset_row_id", "transient_motion_ratio", "posture_transition_ratio"]
    frame = frame.merge(rows[quality], on="dataset_row_id", validate="one_to_one")
    frame = frame.merge(metrics.loc[metrics.method.eq("W0"), ["dataset_row_id", "target_envelope_modulation", "envelope_target_stratum"]],
                        on="dataset_row_id", validate="one_to_one")
    if len(frame) != len(rows) or frame.dataset_row_id.duplicated().any():
        raise ValueError("统一索引窗口不完整")
    frame["zero_markers"] = frame.transient_motion_ratio.eq(0) & frame.posture_transition_ratio.eq(0)
    output = output.resolve()
    if output == bundle.root or bundle.root in output.parents:
        raise ValueError("索引使用导出目录之外的独立输出目录")
    evidence, evidence_sources = evidence_index(e4_root) if e4_root else (None, [])
    links = {i: [] for i in range(len(frame))}
    if figures is not None:
        fig = Bundle(figures, "render")
        if fig.receipt.get("source_manifest_sha256") != bundle.identity()["manifest_sha256"]:
            raise ValueError("图形与导出身份不匹配")
        by_id = {int(row_id): i for i, row_id in enumerate(frame.dataset_row_id)}
        for relative in fig.manifest["files"]:
            path = Path(relative)
            if path.suffix == ".png" and path.name.startswith("row_"):
                parts = path.stem.split("_", 2)
                row_id = int(parts[1])
                target = fig.root / path
                if row_id in by_id:
                    if not target.is_file():
                        raise ValueError("图形链接缺失")
                    links[by_id[row_id]].append((parts[2], os.path.relpath(target, output)))
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "windows.csv", index=False)
    shown = frame.drop(columns=["file"])
    page = table_page(shown, "W0 测试窗口索引", links)
    if evidence is not None:
        evidence.to_csv(output / "evidence.csv", index=False)
        evidence_links = {i: [("示例 NPZ", os.path.relpath(r.example_path, output)),
                              ("逐窗指标", os.path.relpath(r.metrics_path, output)),
                              ("GN 统计", os.path.relpath(r.gn_statistics_path, output))]
                          for i, r in evidence.iterrows()}
        visible = evidence[["seed", "split", "dataset_row_id", "samp_id", "condition", *PRIMARY]]
        (output / "evidence.html").write_text(table_page(visible, "E4 validation 干预证据索引", evidence_links), encoding="utf-8")
        page = page.replace('<table>', '<p><a href="evidence.html">E4 干预证据</a> <a href="windows.csv">完整窗口 CSV</a></p><table>', 1)
    (output / "index.html").write_text(page, encoding="utf-8")
    finish(output, {"phase": "analysis_index", "command": command, "export_source": bundle.identity(),
                    "analysis_protocol": ANALYSIS_PROTOCOL,
                    "rows": len(frame), "evidence_sources": evidence_sources,
                    "figures": str(figures.resolve()) if figures else None})
    return output


def select_cases(index: Path, output: Path, *, quality: str, strata: list[str], subjects: list[int] | None,
                 per_subject: int, command: str) -> Path:
    bundle = Bundle(index, "analysis_index")
    candidates, selected = select_frame(bundle.csv("windows.csv"), quality=quality, strata=strata,
                                        subjects=subjects, per_subject=per_subject)
    output = output.resolve()
    if output == bundle.root or bundle.root in output.parents:
        raise ValueError("案例清单使用索引目录之外的独立输出目录")
    output.mkdir(parents=True, exist_ok=False)
    candidates.to_csv(output / "candidates.csv", index=False)
    selected.to_csv(output / "selected_cases.csv", index=False)
    write_json(output / "selection.json", {"quality": quality, "strata": strata, "subjects": subjects,
               "per_subject_per_stratum": per_subject, "criterion": "distance to subject-stratum target modulation median",
               "distance_ranking_decimal_places": 12, "tie_break": "dataset_row_id ascending",
               "prediction_metrics_used_for_selection": False})
    (output / "index.html").write_text(table_page(selected, "W0 案例选择清单"), encoding="utf-8")
    finish(output, {"phase": "case_selection", "command": command, "index_source": bundle.identity(),
                    "analysis_protocol": ANALYSIS_PROTOCOL,
                    "export_source": bundle.receipt["export_source"], "rows": selected.dataset_row_id.astype(int).tolist()})
    return output
