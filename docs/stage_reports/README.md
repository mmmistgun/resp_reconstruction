# 阶段汇报整理约定

本目录用于保存面向组会、中期检查或阶段复盘的科研汇报底稿与汇报组织约定。这里记录的是后续整理阶段汇报时应优先遵守的个人偏好和协作习惯，不替代 `docs/experiments/` 中的实验计划、指标口径和证据账本。

## 文档定位

- 阶段汇报优先服务听众理解当前研究判断，不追求覆盖完整实验历史。
- 正文采用“研究问题 -> 关键证据 -> 阶段决策 -> 下一步”的问题驱动结构。
- 不服务当前问题的历史分支不进入正文；必要时只在备份页或证据索引中说明其退场原因。
- PPT 可以后置组织；先沉淀可读的 Markdown 阶段底稿，再从底稿拆出正文页和备份页。
- 阶段底稿不直接改变长期实验结论；稳定结论应同步沉淀到 `findings.md` 或对应 `docs/experiments/*.md`。

## 内容取舍

- 第一次面向组会汇报时，先讲清任务动机、数据、预处理、模型看什么、怎么训练、怎么评价、对照方案回答什么问题。
- 正文只放会影响当前判断的实验细节：控制变量、核心比较方案、关键指标、测试集复评结果、风险和下一步。
- 不把“不能只看验证损失”这类显而易见的评价前提包装成阶段发现；它应放在指标页作为读表前提。
- 需要主动说明对照中哪些设置保持不变、哪些真正变化，避免听众误以为多个因素同时改变。
- 多次训练结果可以写作“3 次独立训练”或“多次独立训练”，汇报正文不必强调“随机种子”字样。
- 统计单位要说明清楚：窗口级指标、受试者级独立性、窗口重叠等限制应进入风险或方法边界。

## 证据与实验范围

- 科研汇报不背历史包袱；已经不服务当前问题的数据口径、模型路线或历史实验。
- 测试集复评只覆盖当前 active shortlist；复评目的必须对应阶段问题，而不是重新给所有历史 run 排名。
- 用户手动运行正式实验时，文档应给出 dry-run 和正式运行命令；agent 负责汇总、生成派生 summary，并把结论写回阶段底稿。
- 汇总时保留原始 metrics、manifest 和 summary，不覆盖用户刚跑出的实验产物；派生 CSV 用清晰文件名单独保存。
- 若 manifest 中实际运行参数和文档命令不同，以实际 manifest 为准并同步修正文档复现命令。

## 讲述顺序

推荐正文顺序：

1. 汇报问题和一句话结论。
2. 任务动机：为什么要从 BCG 恢复 THO-like 呼吸信息。
3. 任务定义与主要难点。
4. 数据集、预处理和受试者划分。
5. 整体流程与模型结构。
6. 训练流程、损失函数和训练配置。
7. 指标解释和读表原则。
8. 对照设计与比较方案差异。
9. 测试集复评结果和关键 delta。
10. 当前阶段决策。
11. 风险、不确定性和统计限制。
12. 下一步计划。

备份页优先放：代表性预测图、完整指标公式、完整训练配置、复现实验命令、数据 provenance、现场问答。

## 表述习惯

- 优先使用自然中文科研表达，避免直译式、工程味过重或内部黑话式表述。
- 第一次出现的概念必须给出来源或计算方式；例如频带能量特征应说明由 BCG 输入的 STFT 频带汇总得到。
- 实验代号不能替代解释；第一次出现时要说明该方案回答什么问题。
- 结论不要写成“默认方案”“更强”“更稳”这类口语判断，优先写成“本阶段基准方案”“误差更低”“综合表现更稳定”。
- 代码实现词只在需要追溯时保留；面向汇报时，`注入` 可改为“融合”，`辅助输入` 可改为“辅助信息”，`比较对象` 可改为“比较方案”。
- 避免使用不自然表达，例如“低维频带摘要”“一对一波形翻译”“拓扑变化”“呼吸相关位移”。无法确定是否自然时，先列问题词和建议替换，再修改正文。

## 修改流程

- 往阶段报告补内容前，先说明准备补什么、放在哪里、为什么值得补；用户确认取舍后再改。
- 用户要求“先看看”时，只检查并列出建议，不直接写入文档。
- 修改时同步正文、PPT 骨架和备份页建议，避免后续转 PPT 时残留旧说法。
- 每次修改后做轻量验证：扫描占位符、问题术语、行尾空白和关键段落是否存在。
- 纯文档修改不需要跑代码测试，但最终说明中要明确没有改数据、split、指标、训练配置或实验结果。

## 20260708 组会 PPT

本轮交付定位为面向首次接触项目听众的研究讨论型汇报：完整讲清任务、数据、预处理、模型、训练、指标、证据、限制与具体决策，不是按时间编排的进度流水账。正文按以下 11 章组织：

1. 任务与信号直觉。
2. 数据来源与样本形成。
3. 输入与目标预处理。
4. 模型输入与计算图。
5. 训练与损失。
6. 指标计算与失效场景。
7. 对照实验设计。
8. 整体结果与稳定性。
9. 分层分析与完整案例。
10. BCG 二次谐波问题。
11. 研究议题与下一步。

正式模板是 `docs/stage_reports/20260708/组会汇报.pptx`，内容底稿是 `docs/stage_reports/2026-07-08-tho-research-v2-group-meeting-stage-report.md`，最终输出固定为 `docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx`。成品共 65 页；流程、公式、文字和表格等主体内容是 PowerPoint native editable 对象，并引用 19 张经 manifest 校验的证据图和 5 张三线表。

### 证据目录与资产

`scripts/tho_group_meeting_ppt/evidence.py` 负责解析并校验正式 run config、canonical test manifest、二次谐波证据和代表性信号。任务 3–6 的四个 builder 及其冻结清单是：

- `signal_figures.py::build_signal_assets` → `signal_assets_manifest.json`。
- `model_figures.py::build_model_assets` → `model_assets_manifest.json`。
- `metric_figures.py::build_metric_assets` → `metric_assets_manifest.json`。
- `case_figures.py::build_case_assets` → `case_assets_manifest.json`。

四份 manifest 均位于 `docs/stage_reports/20260708/generated_assets/discussion/`；组装前会校验 19 张图的路径、SHA-256 和关键上游证据，不接受静默替换。

### 生成命令与 CLI 边界

默认 CLI 直接生成上述正式文件：

```bash
./.venv/bin/python scripts/build_tho_group_meeting_ppt.py
```

若资产源已更新，先刷新四份资产清单及 19 张图，再组装正式 PPT：

```bash
./.venv/bin/python scripts/build_tho_group_meeting_ppt.py --assets-only
./.venv/bin/python scripts/build_tho_group_meeting_ppt.py
```

`--evidence-only`、`--assets-only`、`--discussion-deck`、`--legacy-summary` 和 `--charts-only` 是互斥模式。其中 `--evidence-only` 只审计并打印证据目录，`--assets-only` 只刷新任务 3–6 资产，`--discussion-deck` 是与默认正式生成等价的显式模式，`--legacy-summary` 只保留旧 25 页摘要入口，`--charts-only` 也只服务旧摘要图表。不应把后两者当作正式交付的默认流程。

### 渲染、测试与确定性

可在独立 LibreOffice profile 中将成品转换为 PDF，避免复用桌面会话状态：

```bash
render_dir="$(mktemp -d /tmp/tho_group_meeting_render.XXXXXX)"
lo_home="$(mktemp -d /tmp/tho_group_meeting_lohome.XXXXXX)"
lo_profile="$(mktemp -d /tmp/tho_group_meeting_profile.XXXXXX)"
runtime_dir="$(mktemp -d /tmp/tho_group_meeting_runtime.XXXXXX)"
chmod 700 "$runtime_dir"
HOME="$lo_home" XDG_RUNTIME_DIR="$runtime_dir" SAL_USE_VCLPLUGIN=svp \
  libreoffice "-env:UserInstallation=file://$lo_profile" --headless --nologo \
  --nodefault --nofirststartwizard --norestore --convert-to pdf \
  --outdir "$render_dir" \
  docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx
pdfinfo "$render_dir/THO_research_v2_阶段进展_组会汇报.pdf" | rg '^Pages:'
```

常规回归测试中 LibreOffice integration 默认 skip；需要真实 PDF/PNG 渲染检查时显式执行：

```bash
RUN_LIBREOFFICE_INTEGRATION=1 ./.venv/bin/python -m pytest \
  tests/test_tho_group_meeting_ppt.py tests/test_rr_peak_band_metric_demo_ppt.py
```

最近一次显式 LibreOffice integration 验证为 `104 passed`（约 116 s）。生成器会固定 ZIP 成员排序、时间戳、平台属性和压缩方式，并清理模板残留的 `docProps/core.xml` / `docProps/app.xml` metadata；连续 build 必须产生相同 SHA-256，并与正式文件一致。

### 科研边界与未解决证据缺口

当前数据 provenance 仍缺少 `dirty=false` 的数据制作 commit 与据此重新导出的 manifest；该缺口尚未解决，不得将现有证据改写为已经具备完全干净的数据制作链。

本轮没有启动训练，没有重评 checkpoint，没有修改数据、split、标签或核心指标口径。新增 lag trace 只是为了可视化指标搜索过程而抽取的中间量，保持既有 lag-aware 指标数值不变，不构成指标口径变更。
