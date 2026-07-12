from __future__ import annotations

import argparse
import json
from pathlib import Path

from tho_group_meeting_ppt.build import (
    build_discussion_assets,
    build_discussion_presentation,
    build_presentation,
)
from tho_group_meeting_ppt.charts import build_all_charts
from tho_group_meeting_ppt.evidence import build_evidence_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "docs/stage_reports/20260708"


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 THO research v2 组会汇报 PPT")
    parser.add_argument("--template", type=Path, default=REPORT_DIR / "组会汇报.pptx")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "THO_research_v2_全流程研究讨论版.pptx")
    parser.add_argument("--no-backup", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evidence-only", action="store_true", help="只审计并打印证据目录，不生成 PPT")
    mode.add_argument("--assets-only", action="store_true", help="只生成任务 3–6 的真实图资产")
    mode.add_argument("--discussion-deck", action="store_true", help="显式组装研究讨论版（默认模式）")
    mode.add_argument("--legacy-summary", action="store_true", help="兼容旧 25 页摘要入口")
    mode.add_argument("--charts-only", action="store_true", help="只生成旧摘要图表")
    args = parser.parse_args()

    if args.evidence_only:
        catalog = build_evidence_catalog(REPO_ROOT)
        print(json.dumps({
            "dataset_index": str(catalog.dataset_index),
            "result_root": str(catalog.result_root),
            "harmonic_root": str(catalog.harmonic_root),
            "case_row_ids": catalog.case_row_ids,
            "run_configs": {key: str(value.path) for key, value in catalog.run_configs.items()},
        }, ensure_ascii=False, indent=2))
        return
    if args.assets_only:
        paths = build_discussion_assets(REPO_ROOT)
        for path in paths.values():
            print(path)
        return
    if args.charts_only:
        paths = build_all_charts(REPO_ROOT, REPORT_DIR / "generated_assets")
        for path in paths.values():
            print(path)
        return
    if args.legacy_summary:
        output = build_presentation(
            template=args.template,
            output=args.output,
            include_backup=not args.no_backup,
            repo_root=REPO_ROOT,
        )
    else:
        output = build_discussion_presentation(
            template=args.template,
            output=args.output,
            repo_root=REPO_ROOT,
        )
    print(output)


if __name__ == "__main__":
    main()
