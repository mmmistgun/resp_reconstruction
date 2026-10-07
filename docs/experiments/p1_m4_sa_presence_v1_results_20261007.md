# Patch-aligned TF-Mamba M4：窗口 SA 事件有无 research-test 结果

日期：2026-10-07（Asia/Shanghai）。协议：`p1-m4-sa-presence-v1-20261007`。状态：**完成并关闭**。

**M4在含SA事件的430个窗口上，五项指标均较无SA标注事件的1880个窗口差，三个训练seed方向全部一致。** 本结果为固定research-test窗口分布上的性能描述。

## 模型、分组与性能

模型是P1组件实验的M4（P1-Full-Add）：1秒patch、W-full CWT、对齐cross-attention、零初始化线性residual-add、六层BiMamba2。固定seed为20260811/20260812/20260813，validation-selected checkpoint epoch为25/5/8。

分组复用[既有SA窗口分析](h_only_sa_presence_v1_results_20261006.md)的冻结组别，范围包括阻塞性、中枢性、混合性暂停和低通气。事件Start/End与180秒THO窗口有正时长交集即为含SA事件；其余为无SA标注事件，仅端点接触不算。逐行核对M4和分组来源的THO文件、key、采样率、采样起止、受试者及row ID完全一致。完整合同见[专项协议](p1_m4_sa_presence_v1_protocol_20261007.md)。

表中先逐seed计算窗口平均，再报告三seed均值±sample SD（ddof=1）。RR单位bpm；四项误差越低越好，PCC越高越好。每组、每seed五项指标有效分母均等于窗口数。

| 组别 | 窗口数（占比） | 受试者数 | Whole RR误差 | Local RR MAE | 包络轨迹MAE | 全局调制误差 | PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 含SA事件 | 430（18.61%） | 8 | 2.0985±0.2206 | 1.9928±0.1313 | 0.2412±0.0122 | 0.2992±0.0222 | 0.7903±0.0134 |
| 无SA标注事件 | 1880（81.39%） | 7 | 0.2636±0.0423 | 0.3311±0.0599 | 0.1237±0.0086 | 0.1374±0.0099 | 0.8904±0.0053 |
| 全量 | 2310 | 8 | 0.6051±0.0582 | 0.6404±0.0668 | 0.1455±0.0093 | 0.1675±0.0064 | 0.8718±0.0067 |

三seed内，含事件组的四项误差均更高，PCC均更低。评价对象为整个180秒窗口。“无SA标注事件”表示源标签没有与窗口相交的事件，不能等同于正常人群。

## 受试者构成

| 受试者 | 含SA事件窗口 | 无SA标注事件窗口 |
|---|---:|---:|
| 220 | 22 | 261 |
| 229 | 50 | 248 |
| 286 | 91 | 211 |
| 670 | 79 | 0 |
| 671 | 60 | 444 |
| 704 | 106 | 469 |
| 726 | 13 | 201 |
| 1006 | 9 | 46 |

670全部79窗属于含事件组，占该组18.37%，无标注组不包含该主体。组间指标差异同时包含主体构成和潜在信号质量差异，不能仅归因于SA的独立影响。三个训练seed描述优化随机性，独立受试者仍为8人；窗口存在时间依赖，本轮不作窗口独立性显著性推断。

## 来源与验证

M4来源：`/home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction/runs/p1_components_v1/research_test_v1_20261007`。Allowlist SHA256为`8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47`；每份评价receipt绑定checkpoint、metrics、test_rows和evaluation身份；固定全量参照为`summary/attempt_b4769c4f0e724872b110fe20c6db99eb/per_seed.csv`。

SA组别来源：本worktree的`runs/h_only_sa_presence_v1/attempt_20261005T175626Z_3c7db0d6459b/window_groups.csv`，复用其manifest/receipt及逐窗口target资格。事件覆盖审计与标签时间轴证据沿用该已完成来源。

新产物绝对目录：

`/mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-medical-rr-strata-v1/runs/p1_m4_sa_presence_v1/attempt_20261007T150849Z_4d5ba53300e8`

目录保存`window_groups.csv`、`window_metrics.csv`、`metrics_per_seed.csv`、`metrics_across_seed.csv`、`subject_composition.csv`、`checkpoints.json`、`sources.json`、`verification.json`、脚本/协议副本、manifest与receipt。基础commit为`5ab273e`，执行脚本及依赖脚本以文件SHA256绑定。

验证：4项synthetic CPU定向测试通过，包含新M4的THO定位/模型身份/顺序漂移拒绝及复用SA汇总的区间、权重、非有限失败测试。全部2310行、THO定位、430/1880分组与三seedtarget资格一致；两组加权恢复15项逐seed全量均值，并与冻结P1汇总一致，最大绝对差`1.11e-16`；读取文件执行前后哈希一致。分析执行约6秒，只消费既存产物，历史评价与checkpoint保持原身份。
