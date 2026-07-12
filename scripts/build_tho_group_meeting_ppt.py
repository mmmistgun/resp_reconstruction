from __future__ import annotations

import argparse
from pathlib import Path

from tho_group_meeting_ppt.build import build_presentation
from tho_group_meeting_ppt.charts import build_all_charts


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "docs/stage_reports/20260708"


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 THO research v2 组会汇报 PPT")
    parser.add_argument("--template", type=Path, default=REPORT_DIR / "组会汇报.pptx")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "THO_research_v2_阶段进展_组会汇报.pptx")
    parser.add_argument("--charts-only", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if args.charts_only:
        paths = build_all_charts(REPO_ROOT, REPORT_DIR / "generated_assets")
        for path in paths.values():
            print(path)
        return
    output = build_presentation(
        template=args.template,
        output=args.output,
        include_backup=not args.no_backup,
        repo_root=REPO_ROOT,
    )
    print(output)


if __name__ == "__main__":
    main()
