from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "docs/stage_reports/20260708/组会汇报.pptx"
FINAL_DECK = REPO_ROOT / "docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx"


def test_body_text_is_black_and_table_has_only_three_horizontal_rules():
    from scripts.tho_group_meeting_ppt.theme import (
        BODY_BLACK,
        add_body_text,
        add_three_line_table,
        is_three_line_table,
        new_content_slide,
    )

    prs = Presentation()
    slide = new_content_slide(prs, "测试标题", page_number=1)
    box = add_body_text(
        slide,
        ["正文内容"],
        x=Inches(1.0),
        y=Inches(1.5),
        width=Inches(5.0),
        height=Inches(1.0),
    )
    table = add_three_line_table(
        slide,
        ["方案", "指标"],
        [["A", "0.1"]],
        x=Inches(1.0),
        y=Inches(2.8),
        width=Inches(6.0),
        height=Inches(1.2),
    )

    assert box.text_frame.paragraphs[0].runs[0].font.color.rgb == BODY_BLACK
    assert is_three_line_table(table)
    assert all(cell.fill.type is None for row in table.rows for cell in row.cells)

