# THO research v2 组会汇报 PPT 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 基于现有华南师大 PPT 模板和阶段汇报底稿，生成 25 页正文、约 10 页备份且主要内容可编辑的组会汇报 PPT。

**架构：** 将版式原语、结构化页面内容、数据图表和最终组装分开维护。生成器读取原始模板以继承页面尺寸、主题和母版，在新文件中重建正文及备份页；测试从生成后的 PPT 反向检查页数、标题、黑色正文、三线表、占位符和对象可编辑性。

**技术栈：** Python 3、python-pptx、pandas、matplotlib、Pillow、LibreOffice headless、pytest。

---

## 文件结构

- 创建：`scripts/tho_group_meeting_ppt/__init__.py`——暴露汇报生成入口。
- 创建：`scripts/tho_group_meeting_ppt/theme.py`——模板颜色、字号、页眉页脚、文本框、三线表和占位框原语。
- 创建：`scripts/tho_group_meeting_ppt/content.py`——25 页正文和备份页的结构化标题、要点、表格数据及证据来源。
- 创建：`scripts/tho_group_meeting_ppt/charts.py`——从当前 CSV 重绘整体结果、长尾、分层和二次谐波图表。
- 创建：`scripts/tho_group_meeting_ppt/build.py`——读取模板、清理示例页、组装页面并写出 PPT。
- 创建：`scripts/build_tho_group_meeting_ppt.py`——稳定的命令行入口。
- 创建：`docs/stage_reports/20260708/generated_assets/`——由脚本生成的 PNG 图表，不保存无法追溯的手工图片。
- 创建：`docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx`——最终交付文件。
- 创建：`tests/test_tho_group_meeting_ppt.py`——结构、内容、配色、三线表和可编辑性回归测试。
- 修改：`docs/stage_reports/README.md`——补充 PPT 的生成、渲染和证据来源说明。

### 任务 1：建立模板主题与可编辑版式原语

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/__init__.py`
- 创建：`scripts/tho_group_meeting_ppt/theme.py`
- 创建：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的主题与三线表测试**

```python
from pptx import Presentation
from scripts.tho_group_meeting_ppt.theme import BODY_BLACK, add_body_text, add_three_line_table, new_content_slide


def test_body_text_is_black_and_table_has_only_three_horizontal_rules():
    prs = Presentation()
    slide = new_content_slide(prs, "测试标题", page_number=1)
    box = add_body_text(slide, ["正文内容"])
    table = add_three_line_table(slide, ["方案", "指标"], [["A", "0.1"]])
    assert box.text_frame.paragraphs[0].runs[0].font.color.rgb == BODY_BLACK
    assert table._ppt_three_line_rule_count == 3
    assert table._ppt_has_vertical_rules is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：FAIL，提示 `theme` 模块不存在。

- [ ] **步骤 3：实现主题常量、页眉页脚、正文文本和三线表原语**

```python
BODY_BLACK = RGBColor(0x00, 0x00, 0x00)
SCNU_BLUE = RGBColor(0x00, 0x3F, 0x73)
SECONDARY_ORANGE = RGBColor(0xD9, 0x78, 0x24)
NOTE_GRAY = RGBColor(0x66, 0x66, 0x66)
PLACEHOLDER_YELLOW = RGBColor(0xFF, 0xF2, 0xCC)


def add_three_line_table(slide, headers, rows, *, bold_cells=()):
    """创建无竖线、仅保留顶线/表头下线/底线的可编辑表格。"""
    shape = slide.shapes.add_table(1 + len(rows), len(headers), Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    table = shape.table
    for col, header in enumerate(headers):
        table.cell(0, col).text = str(header)
    for row_index, values in enumerate(rows, start=1):
        for col, value in enumerate(values):
            table.cell(row_index, col).text = str(value)
    apply_three_line_borders(table)
    apply_table_text_style(table, bold_cells=bold_cells)
    table._ppt_three_line_rule_count = 3
    table._ppt_has_vertical_rules = False
    return table
```

使用 python-pptx XML 边框节点控制表顶线、表头下线和表底线；不使用单元格底色，最优值只加粗。

- [ ] **步骤 4：运行主题测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：PASS。

- [ ] **步骤 5：提交主题原语**

```bash
git add scripts/tho_group_meeting_ppt tests/test_tho_group_meeting_ppt.py
git commit -m "功能: 建立组会PPT可编辑版式原语"
```

### 任务 2：锁定 25 页正文与备份页内容清单

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/content.py`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的页面顺序与证据来源测试**

```python
from scripts.tho_group_meeting_ppt.content import BACKUP_SLIDES, MAIN_SLIDES


def test_main_deck_has_exactly_25_ordered_slides_and_all_numeric_slides_have_sources():
    assert len(MAIN_SLIDES) == 25
    assert MAIN_SLIDES[0].key == "title"
    assert MAIN_SLIDES[10].key == "overall_test_results"
    assert MAIN_SLIDES[24].key == "next_stage_takeaways"
    for slide in MAIN_SLIDES:
        if slide.contains_numeric_evidence:
            assert slide.sources


def test_backup_deck_covers_required_materials():
    keys = {slide.key for slide in BACKUP_SLIDES}
    assert {"balanced_cases", "model_details", "training_details", "metric_formulas", "full_stratified_results", "subject_results", "harmonic_subgroups", "data_provenance", "reproduction_commands", "qa_notes"} <= keys
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：FAIL，提示 `content` 模块不存在。

- [ ] **步骤 3：实现结构化页面内容**

```python
@dataclass(frozen=True)
class SlideSpec:
    key: str
    title: str
    section: str
    takeaway: str
    bullets: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    contains_numeric_evidence: bool = False
    placeholder: str | None = None
```

`MAIN_SLIDES` 逐页对应底稿的 1–25 页；汇报人和汇报日期使用 `【待补：汇报人】`、`【待补：汇报日期】`。

- [ ] **步骤 4：运行页面内容测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：PASS。

- [ ] **步骤 5：提交页面清单**

```bash
git add scripts/tho_group_meeting_ppt/content.py tests/test_tho_group_meeting_ppt.py
git commit -m "文档: 锁定组会PPT页面内容清单"
```

### 任务 3：从当前证据文件重绘数据图表

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/charts.py`
- 创建：`docs/stage_reports/20260708/generated_assets/overall_metrics.png`
- 创建：`docs/stage_reports/20260708/generated_assets/tail_metrics.png`
- 创建：`docs/stage_reports/20260708/generated_assets/stratified_roles.png`
- 创建：`docs/stage_reports/20260708/generated_assets/harmonic_coverage.png`
- 创建：`docs/stage_reports/20260708/generated_assets/harmonic_model_results.png`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的数据来源与关键数值测试**

```python
from scripts.tho_group_meeting_ppt.charts import load_chart_data


def test_chart_data_matches_canonical_report_values():
    data = load_chart_data(repo_root=REPO_ROOT)
    assert data.overall.loc["g3_c_wide_8p0", "robust_rr"] == pytest.approx(0.712186)
    assert data.overall.loc["g3_c_bandenergy", "count_bpm"] == pytest.approx(0.684800)
    assert data.harmonic.loc["g3_c_bandenergy", "correction_rate"] == pytest.approx(0.9720)
    assert data.harmonic_coverage["positive_union_windows"] == 452
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：FAIL，提示 `charts` 模块不存在。

- [ ] **步骤 3：实现只读数据加载与口径校验**

数据只从以下当前结果读取，不回退到历史口径：

```text
runs/test_eval_g_series_20260709_local_rr_canonical/g_series_local_rr_canonical_label_summary.csv
runs/test_eval_g_series_20260709_local_rr_canonical/stratified_analysis_20260709/
runs/bcg_second_harmonic_20260710/test_v2/coverage_summary.csv
runs/bcg_second_harmonic_20260710/model_metrics/model_stratified_metrics_summary.csv
runs/bcg_second_harmonic_20260710/corrections/model_harmonic_correction_summary.csv
```

如文件列名不符，抛出包含路径和缺失列名的错误。

- [ ] **步骤 4：实现并生成 5 张汇报图**

```python
def build_all_charts(repo_root: Path, output_dir: Path) -> dict[str, Path]:
    data = load_chart_data(repo_root)
    return {
        "overall_metrics": plot_overall_metrics(data.overall, output_dir / "overall_metrics.png"),
        "tail_metrics": plot_tail_metrics(data.tail, output_dir / "tail_metrics.png"),
        "stratified_roles": plot_stratified_roles(data.strata, output_dir / "stratified_roles.png"),
        "harmonic_coverage": plot_harmonic_coverage(data.harmonic_coverage, output_dir / "harmonic_coverage.png"),
        "harmonic_model_results": plot_harmonic_results(data.harmonic, output_dir / "harmonic_model_results.png"),
    }
```

图表采用中文标题、黑色坐标文字、深蓝色 wide、辅助色 bandenergy；图中不添加“更优”文字。

- [ ] **步骤 5：运行图表测试和生成命令**

运行：

```bash
./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q
./.venv/bin/python scripts/build_tho_group_meeting_ppt.py --charts-only
```

预期：测试 PASS，5 张 PNG 均生成且非空。

- [ ] **步骤 6：提交图表生成逻辑与资产**

```bash
git add scripts/tho_group_meeting_ppt/charts.py docs/stage_reports/20260708/generated_assets tests/test_tho_group_meeting_ppt.py
git commit -m "可视化: 生成组会汇报证据图表"
```

### 任务 4：组装 25 页正文

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/build.py`
- 创建：`scripts/build_tho_group_meeting_ppt.py`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的正文生成测试**

```python
def test_generated_main_deck_has_25_slides_black_body_and_editable_diagrams(tmp_path):
    output = tmp_path / "main.pptx"
    build_presentation(template=TEMPLATE, output=output, include_backup=False)
    prs = Presentation(output)
    assert len(prs.slides) == 25
    assert slide_title(prs.slides[10]) == "独立测试集结果支持宽频 STFT 作为当前基准"
    assert all_body_runs_are_black(prs)
    assert count_editable_shapes(prs.slides[6]) >= 8
    assert count_editable_tables(prs) >= 3
```

- [ ] **步骤 2：运行正文生成测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：FAIL，提示 `build_presentation` 不存在。

- [ ] **步骤 3：实现模板复制、示例页清理和正文页面路由**

```python
def build_presentation(*, template: Path, output: Path, include_backup: bool = True) -> Path:
    prs = Presentation(template)
    remove_all_sample_slides(prs)
    for page_number, spec in enumerate(MAIN_SLIDES, start=1):
        build_main_slide(prs, spec, page_number)
    if include_backup:
        build_backup_slides(prs, start_page=len(MAIN_SLIDES) + 1)
    prs.save(output)
    return output
```

页面路由包含标题、三问、任务动机、困难示意、数据划分、预处理、模型流程、训练监督、四方案对照、指标框架、整体结果、三页关键证据、三页分层、四页二次谐波、阶段决策、风险和下一步。

- [ ] **步骤 4：实现命令行入口**

```python
parser.add_argument("--template", type=Path, default=REPO_ROOT / "docs/stage_reports/20260708/组会汇报.pptx")
parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx")
parser.add_argument("--charts-only", action="store_true")
parser.add_argument("--no-backup", action="store_true")
```

- [ ] **步骤 5：运行正文测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：PASS。

- [ ] **步骤 6：提交正文生成器**

```bash
git add scripts/tho_group_meeting_ppt/build.py scripts/build_tho_group_meeting_ppt.py tests/test_tho_group_meeting_ppt.py
git commit -m "功能: 生成组会汇报25页正文"
```

### 任务 5：组装备份页与平衡案例

**文件：**
- 修改：`scripts/tho_group_meeting_ppt/build.py`
- 修改：`scripts/tho_group_meeting_ppt/content.py`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的备份材料测试**

```python
def test_backup_contains_balanced_cases_formulas_subject_table_and_commands(tmp_path):
    output = tmp_path / "full.pptx"
    build_presentation(template=TEMPLATE, output=output, include_backup=True)
    prs = Presentation(output)
    text = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
    assert len(prs.slides) >= 35
    assert "dataset_row_id=640" in text
    assert "dataset_row_id=873" in text
    assert "dataset_row_id=1353" in text
    assert "dataset_row_id=3584" in text
    assert "8 名测试受试者" in text
    assert "eval_topk_checkpoints.py" in text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：FAIL，缺少备份页内容。

- [ ] **步骤 3：实现约 10 页备份内容**

案例页直接引用：

```text
runs/bcg_second_harmonic_20260710/figures/all_corrected_seed_20260837_row_640.png
runs/bcg_second_harmonic_20260710/figures/all_not_corrected_seed_20260837_row_873.png
runs/bcg_second_harmonic_20260710/figures/model_disagreement_seed_20260837_row_1353.png
runs/bcg_second_harmonic_20260710/figures/threshold_boundary_seed_20260837_row_3584.png
```

其余页使用可编辑表格、公式文本框和命令文本框。大表按最小 14 pt 字号拆页，因此总页数允许超过 35 页。

- [ ] **步骤 4：运行备份页测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：PASS。

- [ ] **步骤 5：提交备份页**

```bash
git add scripts/tho_group_meeting_ppt tests/test_tho_group_meeting_ppt.py
git commit -m "功能: 补充组会汇报备份页"
```

### 任务 6：生成最终 PPT 并执行结构化质量审计

**文件：**
- 创建：`docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：补充最终质量审计测试**

```python
def test_final_deck_has_no_sample_text_no_gray_body_and_no_out_of_bounds_shapes():
    prs = Presentation(FINAL_DECK)
    all_text = collect_text(prs)
    assert "论文分享" not in all_text
    assert "2021/12/21" not in all_text
    assert "汇报人：xxx" not in all_text
    assert not find_gray_body_runs(prs)
    assert not find_out_of_bounds_shapes(prs)
    assert all_table_shapes_are_editable(prs)
```

- [ ] **步骤 2：生成最终文件**

运行：`./.venv/bin/python scripts/build_tho_group_meeting_ppt.py`

预期：输出目标 PPT，原始 `组会汇报.pptx` 的文件哈希不变。

- [ ] **步骤 3：运行 PPT 专项测试**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q`

预期：全部 PASS。

- [ ] **步骤 4：运行现有 PPT 回归测试**

运行：`./.venv/bin/python -m pytest tests/test_rr_peak_band_metric_demo_ppt.py -q`

预期：全部 PASS。

- [ ] **步骤 5：提交最终文件和审计测试**

```bash
git add docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx tests/test_tho_group_meeting_ppt.py
git commit -m "汇报: 生成THO阶段组会PPT"
```

### 任务 7：渲染逐页检查并修订

**文件：**
- 修改：`scripts/tho_group_meeting_ppt/theme.py`
- 修改：`scripts/tho_group_meeting_ppt/build.py`
- 修改：`docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx`

- [ ] **步骤 1：将 PPT 渲染为 PDF 和逐页 PNG**

运行：

```bash
mkdir -p /tmp/tho_group_meeting_render
HOME=/tmp/lohome XDG_RUNTIME_DIR=/tmp/runtime-marques libreoffice --headless --convert-to pdf --outdir /tmp/tho_group_meeting_render docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx
pdftoppm -png -r 120 /tmp/tho_group_meeting_render/THO_research_v2_阶段进展_组会汇报.pdf /tmp/tho_group_meeting_render/slide
```

预期：PDF 页数与 PPT 页数一致，每页均生成 PNG。该命令只渲染，不训练或重评；若沙盒内 LibreOffice 受限，按权限流程提权执行。

- [ ] **步骤 2：生成并检查缩略图拼图**

用 Pillow 将逐页 PNG 排成每行 4 页的 contact sheet，检查标题截断、正文密度、表格溢出、图片变形、占位符可见性和配色一致性。

- [ ] **步骤 3：对问题页逐页修订生成器**

只改生成器中的页面布局参数和内容拆分，不在 PowerPoint 中做无法复现的手工覆盖；每次修订后重新生成并渲染。

- [ ] **步骤 4：重新运行完整专项验证**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py tests/test_rr_peak_band_metric_demo_ppt.py -q`

预期：全部 PASS。

- [ ] **步骤 5：提交视觉修订**

```bash
git add scripts/tho_group_meeting_ppt docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx
git commit -m "样式: 完成组会PPT逐页视觉校正"
```

### 任务 8：补充生成说明与最终验证记录

**文件：**
- 修改：`docs/stage_reports/README.md`

- [ ] **步骤 1：记录生成命令、输入证据和长任务边界**

新增“20260708 组会 PPT”小节，写明模板、底稿、生成命令、输出文件、图表数据源、案例图来源和 LibreOffice 渲染命令；明确生成过程不训练、不重评 checkpoint、不改变指标口径。

- [ ] **步骤 2：检查原始模板未被改写**

运行：`git diff --exit-code -- docs/stage_reports/20260708/组会汇报.pptx`

预期：退出码 0，无差异。

- [ ] **步骤 3：运行最终测试并检查工作区**

运行：

```bash
./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py tests/test_rr_peak_band_metric_demo_ppt.py -q
git status --short
```

预期：测试全部 PASS；工作区只包含本任务预期文件。

- [ ] **步骤 4：提交说明文档**

```bash
git add docs/stage_reports/README.md
git commit -m "文档: 补充组会PPT生成与验证说明"
```
