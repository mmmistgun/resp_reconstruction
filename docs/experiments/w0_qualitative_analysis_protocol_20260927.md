# W0 按需定性分析与 R3 案例干预

协议 ID：`w0-qualitative-analysis-v1-20260927`。状态：2026-09-28 经用户明确授权，固定 8 案例的四条件真实 GPU 干预已完成，成功产物为 `runs/w0_qualitative_analysis_v1/r3_cases_02`。

已生成 `runs/w0_qualitative_analysis_v1/index_01`（2310 个 test 窗口、1386 条 E4 validation 证据记录）和
`cases_01`（5 个主体、low/medium 分层共 8 个案例）。现有三类图形链接可从统一索引打开。

## 1. 范围与执行边界

本协议承接 W0 seed 20260812、epoch 15 的完整测试窗口导出与用户确认的索引/选例/按需绘图需求。
基础导出合同见 `w0_test_qualitative_export_plan_20260927.md`。已有 E4 validation 干预保持关闭，本入口仅读取其冻结指标与示例索引。

索引、选例与离线绘图读取已保存产物。`intervene-r3` 是本协议新增的选定 research-test 窗口推理，必须显式传入
`--confirm-research-test-intervention`，由用户执行；不改变 checkpoint、数据划分、指标或训练配置。
当前实现检查、synthetic 测试、已有表格的索引和案例清单生成由 Codex 执行，不启动真实 GPU 干预。

所有新输出使用独立目录并拒绝覆盖。异常直接报错；完成状态以 receipt 和 artifact manifest 为准。
产物 envelope 的 `protocol` 复用导出格式标识，新阶段以 `phase` 和 `analysis_protocol` 标识本协议。

## 2. 接口与文件

统一入口：`scripts/export_w0_test_qualitative.py`。

| 命令 | 主要参数 | 产物 |
|---|---|---|
| `index` | `--source --output [--figures] [--e4-root] [--skip-e4]` | `windows.csv`、可筛选排序的 `index.html`、E4 `evidence.csv/html` |
| `select-cases` | `--index --output --quality --strata --subjects --per-subject` | `candidates.csv`、`selected_cases.csv`、`selection.json`、HTML |
| `render` | `--source --output [--cases 或 --rows] [--views] [--zoom] [--channels]` | `figures/waveforms`、`conditioning`、`trajectories` 中所选 PNG 和导航页 |
| `intervene-r3` | `--source --cases --output --device --confirm-research-test-intervention` | 配置、四条件逐窗张量、指标、配对差、GN 统计及来源 |
| `render-intervention` | `--source --output [--rows] [--zoom]` | R3 输入变化、波形、包络、特征变化、预测变化组合 PNG 与导航 |

统一窗口索引按 dataset_row_id 一对一关联主体/窗口时间、体动与体位转换标记、参考包络调制量/分层、五主指标及相对基线差值。
页面下拉筛选主体、参考分层和零标记状态，列标题排序；页面浏览不会修改选择清单。
图形链接兼容原平铺目录和分类目录。`--figures` 必须指向与导出 manifest 匹配的完成图形目录。
`--channels 0 12 40` 按实际 latent 通道编号限制 conditioning 中的 g/b 热图；强度曲线仍按全部通道计算，并在图中标注。通道不作为频带解释。

E4 索引读取三 seed 的已完成 r2 evaluation manifest，并核对 freeze receipt。
逐条件指标及 GN 统计进行 SHA 核验；示例 NPZ 核对存在与大小，记录冻结 SHA，不解码全部示例数组。
索引保留 validation split、seed、row、condition、五指标、示例路径和 GN 统计路径，避免与 test 数值混合汇总。

## 3. 案例选择

默认质量条件为 `transient_motion_ratio=0` 且 `posture_transition_ratio=0`；默认分层为 low 和 medium。
在每个主体×分层内部，按参考 `target_envelope_modulation` 到组内中位数的绝对距离排序。
排序距离固定保留 12 位小数，随后以 dataset_row_id 升序处理并列；完整未舍入距离仍保存。
`--per-subject 1` 表示每个主体×分层各取一例。主体或分层无候选时不补选其他层，整体为空则报错。

模型性能不参与默认选例。候选表保留全部方法指标供阅览，选择清单记录来源 manifest 和规则。
重叠窗口与同主体不同案例不视为独立受试者；零标记案例仍需目视检查信号。
`render --cases` 和 GPU 干预都要求案例清单绑定的导出路径与 manifest 完全匹配。

## 4. 固定 R3 四条件

冻结 checkpoint 为 seed 20260812、epoch 15、原始 W0。输入来自已完成导出的 BCG 与实际 CWT 输入，target 也复用保存数组。
R3 使用 E4 的 `[73:97]` 尺度槽位，实际中心频率约 2.191–7.996 Hz，其余 CWT 和 BCG 不变。
沿用 `e4_r3_norm.make_shifts(2310)`：PCG64 seed=20260920，对完整导出行序产生偏移，取 SHIFT_1。
每个窗口的偏移在 60–300 个 CWT 帧内（每帧 0.5 s），全部 R3 尺度共享循环偏移；明确的窗口/偏移映射保存到 config。

四条件固定为：

1. `FULL__NAT`；
2. `SHIFT_1__NAT`；
3. `FULL__ALL_W_GN_FIXED`；
4. `SHIFT_1__ALL_W_GN_FIXED`。

batch 固定为 8，尾 batch 保留；eval、inference_mode、BF16 AMP。每批先计算 FULL，四层 W 分支 GN 的固定统计来自同批、同窗口 FULL。
复用现有 `NormIntervention` 的原生统计、独立公式和重放检查。固定 FULL 与同批自然 FULL 的波形要求 rtol=1e-5、atol=1e-6；
调制前 Z 在各条件间也按相同容差检查。固定 GN 实际使用的 mean/rstd 与 FULL 逐元素一致。
旧导出 FULL 与本次 FULL 的指标差另作描述记录；所有干预效应与本次同批 FULL 配对，避免把运行环境或 batch 差异计入干预响应。

每窗口/条件保存：实际干预 CWT、prediction、target、BCG、gamma_raw/beta_raw、g/b、Z/Z_prime、scale_delta/total_delta、包络、
`prediction_minus_full`、`z_prime_minus_full` 及相对特征变化时间曲线。
相对特征变化为各时刻 `||Z_prime_condition-Z_prime_full||₂/(||Z_full||₂+1e-12)`。
GN 表保留每窗口/条件/层/组的 natural/used/full mean 和 rstd。
`paired_deltas.csv` 使用条件减 FULL 原值差：四项 error 正值表示变差，PCC 负值表示变差。

这些结果属于选定实例对指定 R3 时间干预的响应，不把差值解释为高频信息的贡献百分比。
不得根据干预结果更换案例或错位规则，以追求更大的效果；新增条件需独立修订。

## 5. 用户命令

以下在仓库根目录执行。各输出目录必须尚不存在；环境变量清理沿用当前运行环境。
本次 `index_01` 和 `cases_01` 已完成，可直接从 `render` 或 `intervene-r3` 开始。
下面 `index` / `select-cases` 命令作为首次生成示例保留；再次生成时使用新的输出目录名。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python scripts/export_w0_test_qualitative.py index \
  --source runs/w0_test_qualitative_v1/seed_20260812_export_01 \
  --figures runs/w0_test_qualitative_v1/seed_20260812_figures_01 \
  --output runs/w0_qualitative_analysis_v1/index_01

env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python scripts/export_w0_test_qualitative.py select-cases \
  --index runs/w0_qualitative_analysis_v1/index_01 \
  --output runs/w0_qualitative_analysis_v1/cases_01

env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python scripts/export_w0_test_qualitative.py render \
  --source runs/w0_test_qualitative_v1/seed_20260812_export_01 \
  --cases runs/w0_qualitative_analysis_v1/cases_01 \
  --views waveforms conditioning trajectories \
  --output runs/w0_qualitative_analysis_v1/case_figures_01
```

选定清单的真实 GPU 干预及离线绘图：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python scripts/export_w0_test_qualitative.py intervene-r3 \
  --source runs/w0_test_qualitative_v1/seed_20260812_export_01 \
  --cases runs/w0_qualitative_analysis_v1/cases_01 \
  --output runs/w0_qualitative_analysis_v1/r3_cases_01 \
  --device cuda:0 --confirm-research-test-intervention

env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python scripts/export_w0_test_qualitative.py render-intervention \
  --source runs/w0_qualitative_analysis_v1/r3_cases_01 \
  --output runs/w0_qualitative_analysis_v1/r3_figures_01
```

如需指定主体、分层或局部放大，先保存新的案例清单或图形目录。完整数组可直接复用，图形变化不触发模型推理。

## 6. 验收

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv/bin/python -m pytest tests/test_w0_test_qualitative.py tests/test_w0_qualitative_catalog.py -q
```

synthetic CPU 验证覆盖索引一对一关联、选例不依赖预测指标、主体/分层选择与并列、来源绑定、输出防覆盖、
HTML 转义、分类图形、确认门控、R3 变换保护区域、原生 FiLM 捕获、FULL 回放和四层 GN 控制。
真实 GPU 验收由用户运行干预命令：所选全部窗口覆盖四条件，主指标有限，配对基准与统计控制通过，完成清单存在后再生成机制图。


## 7. 2026-09-28 授权运行记录

用户明确要求 Codex 代跑高频干预。本次严格使用 cases_01 的 8 个案例、seed 20260812、epoch 15、原四条件与 SHIFT_1 规则，在 cuda:0 执行推理，不训练。

首次运行目录 `r3_cases_01` 在首批指标计算时失败：pandas split 列导出 object 数组后，评价入口对 Python str 调用 `.item()`。失败目录保留。修复限定在干预入口的元数据类型转换：split 显式转为 NumPy 字符串，其余身份字段转为 int64；指标公式及实验条件未改。新增合成回归测试后，`tests/test_w0_qualitative_catalog.py` 的 7 项测试通过。

按同一授权在新目录 `r3_cases_02` 重试成功。receipt 与 artifact manifest 均为 complete，8×4=32 条条件窗口记录完整。45 个清单文件的大小与 SHA-256 核验通过；四项误差及 PCC 均有限，FULL 固定统计与自然统计的配对指标差为零，704 条固定统计 GN 记录的 used/full mean 与 rstd 逐元素一致。

固定 GN 的高频错位使 6/8 案例包络轨迹误差增加、5/8 案例 PCC 下降。此前按参考包络规则选定的 row_2751，其包络轨迹 MAE 由 0.129749 增至 0.140038，PCC 由 0.938911 降至 0.935963。这些是预选案例的配对响应，不替代完整测试集总体评价。绘图读取本次同批 FULL 与 SHIFT 配对产物，不混用旧导出 FULL。

成功运行命令：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python scripts/export_w0_test_qualitative.py intervene-r3 \
  --source runs/w0_test_qualitative_v1/seed_20260812_export_01 \
  --cases runs/w0_qualitative_analysis_v1/cases_01 \
  --output runs/w0_qualitative_analysis_v1/r3_cases_02 \
  --device cuda:0 --confirm-research-test-intervention
```
