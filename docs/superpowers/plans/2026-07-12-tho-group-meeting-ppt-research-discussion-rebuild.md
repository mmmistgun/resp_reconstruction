# THO research v2 研究讨论型组会 PPT 重构实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将现有阶段摘要型 PPT 重构为面向第一次接触项目听众的全流程研究讨论型 PPT，具体解释数据、预处理、模型、训练、指标、对照、结果、困难案例和待讨论设计。

**架构：** 先建立可追溯证据目录和结构化讨论单元，再分别生成信号/预处理、模型/训练、指标、结果/案例四类资产，最后按完整讨论单元组装 PPT。数值与 shape 从正式 run 配置、代码、CSV 或 NPZ 读取；无法确认的内容生成包含缺失字段和建议入口的可编辑证据缺口框。

**技术栈：** Python 3、python-pptx、pandas、NumPy、SciPy、PyTorch、matplotlib、Pillow、LibreOffice headless、pytest。

---

## 文件结构

- 创建：`scripts/tho_group_meeting_ppt/evidence.py`——读取正式 run 配置、数据索引、信号资产和案例 NPZ。
- 创建：`scripts/tho_group_meeting_ppt/discussion_content.py`——定义 11 个全流程章节及讨论单元。
- 创建：`scripts/tho_group_meeting_ppt/signal_figures.py`——生成 BCG / THO、预处理、STFT 与 bandenergy 图。
- 创建：`scripts/tho_group_meeting_ppt/model_figures.py`——生成 patch token、STFT 分支、融合与损失调度图。
- 创建：`scripts/tho_group_meeting_ppt/metric_figures.py`——生成五类指标的逐步计算示例。
- 创建：`scripts/tho_group_meeting_ppt/case_figures.py`——生成 delta、稳定性、分层和完整案例图。
- 创建：`scripts/tho_group_meeting_ppt/detail_slides.py`——组装信号、流程、公式、结果、案例和讨论页。
- 修改：`scripts/tho_group_meeting_ppt/content.py`——切换到新版讨论单元入口。
- 修改：`scripts/tho_group_meeting_ppt/build.py`——调用新版证据、资产与页面组装器。
- 修改：`scripts/tho_group_meeting_ppt/theme.py`——增加公式、证据边界、讨论框与多面板原语。
- 修改：`scripts/build_tho_group_meeting_ppt.py`——增加分阶段生成入口。
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/`——保存可追溯重绘图。
- 修改：`docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx`——最终研究讨论型汇报。
- 修改：`tests/test_tho_group_meeting_ppt.py`——覆盖证据、内容、资产、页面语义和最终 PPT。
- 修改：`docs/stage_reports/README.md`——记录新版定位、证据来源和生成方式。

## 任务 1：建立可追溯证据目录

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/evidence.py`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的证据目录测试**

```python
def test_discussion_evidence_catalog_resolves_formal_configs_and_signal_sources():
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    catalog = build_evidence_catalog(REPO_ROOT)
    assert catalog.dataset_index.exists()
    assert catalog.general_signal_npz.name == "f0_visual_sample_signals.npz"
    assert catalog.general_sample_row_id == 8025
    assert catalog.run_configs["g3_c_wide_8p0"].stft_win == 2000
    assert catalog.run_configs["g3_c_wide_8p0"].stft_hop == 250
    assert catalog.run_configs["g3_c_bandenergy"].stft_encoder_type == "bandenergy"
    assert catalog.case_row_ids == (640, 873, 1353, 3584)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k evidence_catalog`

预期：FAIL，提示 `evidence` 模块不存在。

- [ ] **步骤 3：实现证据数据结构**

```python
@dataclass(frozen=True)
class RunConfigEvidence:
    path: Path
    patch_len: int
    patch_stride: int
    mixer_layers: int
    base_channels: int
    stft_win: int
    stft_hop: int
    stft_low_hz: float
    stft_high_hz: float
    stft_encoder_type: str
    stft_inject_position: str


@dataclass(frozen=True)
class EvidenceCatalog:
    dataset_root: Path
    dataset_index: Path
    general_signal_npz: Path
    general_sample_row_id: int
    run_configs: dict[str, RunConfigEvidence]
    result_root: Path
    harmonic_root: Path
    case_row_ids: tuple[int, int, int, int]
```

`build_evidence_catalog()` 必须从 `g_series_test_eval_manifest.csv` 追溯 checkpoint 和 run `config.yaml`，不得用 smoke 配置替代正式配置。

- [ ] **步骤 4：增加缺字段错误测试与实现**

```python
def test_read_run_config_reports_missing_field_with_path(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import read_run_config

    config = tmp_path / "config.yaml"
    config.write_text("model:\n  patch_len: 256\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"stft_win.*config.yaml"):
        read_run_config(config)
```

- [ ] **步骤 5：验证并提交**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k evidence`

预期：PASS。

```bash
git add scripts/tho_group_meeting_ppt/evidence.py tests/test_tho_group_meeting_ppt.py
git commit -m "功能: 建立研究讨论PPT证据目录（任务1）"
```

## 任务 2：定义全流程讨论单元

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/discussion_content.py`
- 修改：`scripts/tho_group_meeting_ppt/content.py`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的章节覆盖测试**

```python
def test_discussion_units_cover_full_pipeline_without_fixed_page_target():
    from scripts.tho_group_meeting_ppt.discussion_content import DISCUSSION_UNITS

    sections = {unit.section for unit in DISCUSSION_UNITS}
    assert sections == {
        "任务与信号直觉", "数据来源与样本形成", "输入与目标预处理", "模型输入与计算图",
        "训练与损失", "指标计算与失效场景", "对照实验设计", "整体结果与稳定性",
        "分层分析与完整案例", "BCG 二次谐波问题", "研究议题与下一步",
    }
    assert len(DISCUSSION_UNITS) >= 42
    technical = [unit for unit in DISCUSSION_UNITS if unit.kind not in {"title", "section"}]
    assert all(unit.method_steps or unit.visual_keys for unit in technical)
    assert all(unit.sources for unit in technical)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k discussion_units`

预期：FAIL，提示 `discussion_content` 不存在。

- [ ] **步骤 3：实现讨论单元类型和内容**

```python
@dataclass(frozen=True)
class DiscussionUnit:
    key: str
    section: str
    title: str
    kind: str
    question: str
    method_steps: tuple[str, ...]
    parameters: tuple[str, ...]
    rationale: tuple[str, ...]
    evidence: tuple[str, ...]
    limits: tuple[str, ...]
    discussion_prompt: str | None
    visual_keys: tuple[str, ...]
    sources: tuple[str, ...]
```

`content.py` 改为导出 `DISCUSSION_UNITS`；旧 25 页摘要清单退出默认路径。

- [ ] **步骤 4：加入禁止流水账与泛问答测试**

```python
def test_discussion_units_exclude_progress_reproduction_and_generic_qa_pages():
    from scripts.tho_group_meeting_ppt.discussion_content import DISCUSSION_UNITS

    text = "\n".join(unit.title + " " + unit.question for unit in DISCUSSION_UNITS)
    for phrase in ("工作进度", "实验编号汇总", "复现实验命令", "现场问答"):
        assert phrase not in text
```

- [ ] **步骤 5：验证并提交**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k discussion`

预期：PASS。

```bash
git add scripts/tho_group_meeting_ppt/discussion_content.py scripts/tho_group_meeting_ppt/content.py tests/test_tho_group_meeting_ppt.py
git commit -m "文档: 定义全流程研究讨论单元（任务2）"
```

## 任务 3：生成信号、预处理与时频表示资产

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/signal_figures.py`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/signal_overview.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/preprocessing_comparison.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/softz_mapping.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/stft_resolution_comparison.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/bandenergy_response.png`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的真实信号与 STFT shape 测试**

```python
def test_signal_assets_use_real_sample_and_expected_stft_shapes(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.signal_figures import build_signal_assets

    assets, metadata = build_signal_assets(REPO_ROOT, tmp_path)
    assert metadata["dataset_row_id"] == 8025
    assert metadata["signal_length"] == 18000
    assert metadata["f0_frames"] == 37
    assert metadata["wide_frames"] == 73
    assert metadata["wide_frequency_resolution_hz"] == pytest.approx(0.05)
    assert set(assets) == {"signal_overview", "preprocessing_comparison", "softz_mapping", "stft_resolution_comparison", "bandenergy_response"}
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k signal_assets`

预期：FAIL，提示 `signal_figures` 不存在。

- [ ] **步骤 3：实现真实信号加载和只读切片**

通用示例使用 `docs/figure/F0_native_stft_pre_mixer/signals/f0_visual_sample_signals.npz`。robust-z / soft-z 对比从 dataset row 8025 对应 NPZ 按 `window_start_s:window_end_s` 切片；字段缺失时写出 `evidence_gap_preprocessing.json`，包含缺失字段、row id 和建议生成入口，不构造伪数据。

- [ ] **步骤 4：实现与模型一致的 STFT 和 bandenergy**

```python
def bandenergy_from_logmag(logmag: np.ndarray, frequencies: np.ndarray):
    bands = ((0.05, 0.3), (0.1, 0.7), (0.3, 1.2), (0.7, 3.0), (3.0, 8.0))
    traces = []
    for low_hz, high_hz in bands:
        keep = (frequencies >= low_hz) & (frequencies < high_hz)
        traces.append(logmag[keep].mean(axis=0))
    return np.stack(traces)
```

STFT 必须验证与 `torch.stft(center=True)` 的帧数和频率 bin 一致。

- [ ] **步骤 5：生成资产并提交**

运行：

```bash
./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k signal
./.venv/bin/python -c "from pathlib import Path; from scripts.tho_group_meeting_ppt.signal_figures import build_signal_assets; build_signal_assets(Path('.').resolve(), Path('docs/stage_reports/20260708/generated_assets/discussion'))"
```

预期：测试 PASS，5 张图存在且尺寸不小于 1600×900。

```bash
git add scripts/tho_group_meeting_ppt/signal_figures.py docs/stage_reports/20260708/generated_assets/discussion tests/test_tho_group_meeting_ppt.py
git commit -m "可视化: 生成信号预处理与时频细节图（任务3）"
```

## 任务 4：生成模型、张量与训练机制资产

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/model_figures.py`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/token_geometry.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/stft_branch_shapes.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/loss_schedule.png`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的正式 run shape 测试**

```python
def test_model_detail_metadata_comes_from_formal_run_configs(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.model_figures import build_model_assets

    assets, metadata = build_model_assets(REPO_ROOT, tmp_path)
    assert metadata["patch_len"] == 256
    assert metadata["patch_stride"] == 128
    assert metadata["patch_count"] == 140
    assert metadata["token_shape"] == ["B", 16, 140]
    assert metadata["signed_corr_weights"] == [0.6, 0.52, 0.44, 0.36, 0.28, 0.2]
    assert set(assets) == {"token_geometry", "stft_branch_shapes", "loss_schedule"}
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k model_detail`

预期：FAIL，提示 `model_figures` 不存在。

- [ ] **步骤 3：实现 shape 计算和调度图**

patch count 调用 `PatchMixer1D.token_count_for_length(18000)`；STFT shape 使用正式 run 配置和 `torch.stft` 空输入验证；loss 权重读取正式 `config.yaml`，展示 epoch 1–10 的 signed corr / cos 调度及启用/关闭分项。

- [ ] **步骤 4：运行测试并提交**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k 'model or loss_schedule'`

预期：PASS。

```bash
git add scripts/tho_group_meeting_ppt/model_figures.py docs/stage_reports/20260708/generated_assets/discussion tests/test_tho_group_meeting_ppt.py
git commit -m "可视化: 生成模型张量与训练机制图（任务4）"
```

## 任务 5：生成五类指标逐步计算示例

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/metric_figures.py`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/metric_robust_rr.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/metric_cycle_count.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/metric_lag_corr.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/metric_relative_envelope.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/metric_local_rr.png`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的指标一致性测试**

```python
def test_metric_examples_match_repository_metric_functions(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.metric_figures import build_metric_assets

    assets, values = build_metric_assets(REPO_ROOT, tmp_path)
    assert values["dataset_row_id"] == 8025
    assert values["robust_rr_abs_error"] == pytest.approx(values["csv_robust_rr_abs_error"], abs=1e-6)
    assert values["cycle_count_abs_error"] >= 0
    assert -1.0 <= values["lag4_corr"] <= 1.0
    assert 0.0 <= values["local_rr_valid_frac"] <= 1.0
    assert len(assets) == 5
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k metric_examples`

预期：FAIL，提示 `metric_figures` 不存在。

- [ ] **步骤 3：复用当前指标实现生成中间量**

稳健 RR 复用 `docs/figure/rr_peak_band_metric/plot_rr_peak_band_metric_demo.py`；其余调用 `resp_train.metrics.signal` 和 `resp_train.metrics.evaluate` 当前函数。每张图包含输入、中间量、最终数值和一条来自底稿的失效说明。

- [ ] **步骤 4：运行测试并提交**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py tests/test_rr_peak_band_metric_demo_ppt.py -q -k 'metric or rr_peak'`

预期：PASS。

```bash
git add scripts/tho_group_meeting_ppt/metric_figures.py docs/stage_reports/20260708/generated_assets/discussion tests/test_tho_group_meeting_ppt.py
git commit -m "可视化: 生成呼吸任务指标逐步示例（任务5）"
```

## 任务 6：生成结果 delta、稳定性与完整案例资产

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/case_figures.py`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/overall_delta.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/seed_subject_stability.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/strata_tradeoffs.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/case_row_640.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/case_row_873.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/case_row_1353.png`
- 创建：`docs/stage_reports/20260708/generated_assets/discussion/case_row_3584.png`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的 delta 与案例对齐测试**

```python
def test_case_assets_align_four_models_by_dataset_row_id(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.case_figures import build_case_assets

    assets, metadata = build_case_assets(REPO_ROOT, tmp_path)
    assert metadata["case_row_ids"] == [640, 873, 1353, 3584]
    assert metadata["models_per_case"] == 4
    assert metadata["signal_length"] == 18000
    assert metadata["wide_robust_rr_delta_vs_time"] == pytest.approx(0.712186 - 0.773193, abs=1e-6)
    assert metadata["bandenergy_count_bpm_delta_vs_time"] == pytest.approx(0.684800 - 0.778259, abs=1e-6)
    assert len(assets) == 7
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k case_assets`

预期：FAIL，提示 `case_figures` 不存在。

- [ ] **步骤 3：实现 row id 对齐和完整案例图**

按 `dataset_row_id` 连接 4 个 `*20260837_harmonic_predictions.npz`、`test_harmonic_labels.csv`、dataset index 和 canonical metrics CSV。对重复或缺失 row id 报错。每个案例图包含 BCG、THO、四模型预测、呼吸带波形、频谱和指标小表，并提供整窗概览与 30–60 秒局部放大。

- [ ] **步骤 4：运行测试并提交**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k 'case or stability or delta'`

预期：PASS。

```bash
git add scripts/tho_group_meeting_ppt/case_figures.py docs/stage_reports/20260708/generated_assets/discussion tests/test_tho_group_meeting_ppt.py
git commit -m "可视化: 生成结果稳定性与完整案例图（任务6）"
```

## 任务 7：组装研究讨论型正文

**文件：**
- 创建：`scripts/tho_group_meeting_ppt/detail_slides.py`
- 修改：`scripts/tho_group_meeting_ppt/theme.py`
- 修改：`scripts/tho_group_meeting_ppt/build.py`
- 修改：`scripts/build_tho_group_meeting_ppt.py`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：编写失败的页面语义测试**

```python
def test_discussion_deck_contains_detailed_method_and_case_pages(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.build import build_discussion_presentation

    output = tmp_path / "discussion.pptx"
    build_discussion_presentation(template=TEMPLATE, output=output, repo_root=REPO_ROOT)
    prs = Presentation(output)
    titles = [_shape_text(slide, "页面标题") for slide in list(prs.slides)[1:]]
    assert len(prs.slides) >= 43
    assert any("样本如何从整晚数据形成" in title for title in titles)
    assert any("soft-z 压缩解决了什么" in title for title in titles)
    assert any("20 秒窗、2.5 秒步长" in title for title in titles)
    assert any("L_signed_corr" in title for title in titles)
    assert any("稳健带通 RR 具体怎么算" in title for title in titles)
    assert any("dataset_row_id=640" in title for title in titles)
    assert any("下一轮最小判别性实验" in title for title in titles)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k discussion_deck`

预期：FAIL，提示 `build_discussion_presentation` 不存在。

- [ ] **步骤 3：实现页面原语和七种 builder**

`theme.py` 增加 method panel、evidence boundary、discussion box；`detail_slides.py` 为 `signal`、`method`、`model`、`formula`、`result`、`case`、`discussion` 七种 kind 提供 builder。全部使用 PowerPoint 原生文本框、表格和形状；数据图保持高分辨率图片。

- [ ] **步骤 4：实现显式拆页规则**

页面仅在 `DiscussionUnit` 明确表示为两个独立单元时拆分；不得按字符数自动切页。测试检查正文最小 18 pt、表格最小 14 pt、图片实际尺寸不小于 4.5×2.4 英寸。

- [ ] **步骤 5：实现分阶段 CLI**

```python
parser.add_argument("--evidence-only", action="store_true")
parser.add_argument("--assets-only", action="store_true")
parser.add_argument("--discussion-deck", action="store_true")
```

默认命令生成新版 discussion deck，旧摘要 deck 不再作为默认交付物。

- [ ] **步骤 6：运行测试并提交**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py -q -k 'discussion_deck or editable or three_line'`

预期：PASS。

```bash
git add scripts/tho_group_meeting_ppt/detail_slides.py scripts/tho_group_meeting_ppt/theme.py scripts/tho_group_meeting_ppt/build.py scripts/build_tho_group_meeting_ppt.py tests/test_tho_group_meeting_ppt.py
git commit -m "功能: 组装全流程研究讨论型PPT（任务7）"
```

## 任务 8：生成最终 PPT 并逐章节渲染检查

**文件：**
- 修改：`docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx`
- 修改：`tests/test_tho_group_meeting_ppt.py`

- [ ] **步骤 1：补充最终文件审计测试**

```python
def test_final_discussion_deck_is_detailed_editable_and_has_no_generic_qa():
    prs = Presentation(FINAL_DECK)
    text = collect_text(prs)
    assert len(prs.slides) >= 43
    assert "复现实验命令" not in text
    assert "现场问答" not in text
    assert "【待补：" not in text or "所需字段" in text
    assert all_body_runs_are_black(prs)
    assert all_tables_are_powerpoint_compatible_three_line(prs)
    assert not find_out_of_bounds_shapes(prs)
```

- [ ] **步骤 2：生成并测试最终文件**

运行：

```bash
./.venv/bin/python scripts/build_tho_group_meeting_ppt.py --discussion-deck
./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py tests/test_rr_peak_band_metric_demo_ppt.py -q
```

预期：目标 PPT 生成，测试全部 PASS，原始模板哈希不变。

- [ ] **步骤 3：渲染全部页面**

运行：

```bash
mkdir -p /tmp/tho_group_meeting_discussion_render
HOME=/tmp/lohome XDG_RUNTIME_DIR=/tmp/runtime-marques libreoffice --headless --convert-to pdf \
  --outdir /tmp/tho_group_meeting_discussion_render \
  docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx
pdftoppm -png -r 120 \
  /tmp/tho_group_meeting_discussion_render/THO_research_v2_阶段进展_组会汇报.pdf \
  /tmp/tho_group_meeting_discussion_render/slide
```

- [ ] **步骤 4：逐章节检查并修订**

分别生成 11 个章节 contact sheet，检查标题、最小字号、公式、中间量、三线表、案例曲线、来源、证据边界和讨论框。问题只在生成器中修订，不做不可复现的手工覆盖。

- [ ] **步骤 5：重新验证并提交**

运行：`./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py tests/test_rr_peak_band_metric_demo_ppt.py -q`

预期：全部 PASS。

```bash
git add -f docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx
git add scripts/tho_group_meeting_ppt tests/test_tho_group_meeting_ppt.py
git commit -m "汇报: 完成全流程研究讨论型PPT（任务8）"
```

## 任务 9：更新说明并执行科研边界核对

**文件：**
- 修改：`docs/stage_reports/README.md`

- [ ] **步骤 1：更新新版定位和生成说明**

记录研究讨论型定位、11 个章节、证据目录、资产生成入口、最终输出和渲染方式；明确不运行训练、不重评 checkpoint、不修改数据、split、标签或指标。

- [ ] **步骤 2：执行最终验证**

运行：

```bash
./.venv/bin/python -m pytest tests/test_tho_group_meeting_ppt.py tests/test_rr_peak_band_metric_demo_ppt.py -q
./.venv/bin/python -m py_compile scripts/build_tho_group_meeting_ppt.py scripts/tho_group_meeting_ppt/*.py
git diff --exit-code -- docs/stage_reports/20260708/组会汇报.pptx
git diff --check
git status --short
```

预期：测试全部 PASS；编译成功；原始模板无差异；工作区仅包含本任务预期文件。

- [ ] **步骤 3：提交说明文档**

```bash
git add docs/stage_reports/README.md
git commit -m "文档: 记录研究讨论型PPT证据与生成方式（任务9）"
```
