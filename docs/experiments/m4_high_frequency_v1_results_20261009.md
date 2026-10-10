# 原始 M4 高频时间结构机制：validation 与 research-test 结果

日期：2026-10-09。协议：`m4-high-frequency-v1-20261009`。**完整三 seed×十条件的 validation、research-test、配对汇总和图册均完成，执行阶段关闭。** 原协议与源码保留运行时冻结字节，其“尚未运行”描述是实现阶段快照；当前完成状态由本文承接。

## 1. 核心结论

**原始 M4 利用了高频 CWT 幅度与当前窗口的时间对应。破坏 (0.8,8] Hz 条件的时间对应后，包络轨迹和 PCC 在两个 split 的三个 seed 上稳定变差；固定条件分支 GN 后效应保持。** test 平均三个错位及三个 seed 后，八名受试者的包络误差均增加，PCC 在六名受试者下降。

这条证据支持高频条件的时间对应具有模型内任务价值。时间均值替换呈现不同结果：包络 MAE 在两个 split 的均值均略有改善，test 的窗口加权 PCC 也略有改善。因此不能把“错位造成损失”进一步等同于“动态高频一定优于静态高频”，或解释成普遍的高频额外信息增益。结合已完成 Full/H/L/NoTF 对照，当前最稳妥的论点是 **M4 对高频条件的正确时间对应具有功能依赖，包络重建提供了较一致的受试者证据**。

## 2. 完成范围与数值控制

原始 P1-M4 的 seeds 20260811/20260812/20260813，checkpoint epoch 25/5/8；未重训或重新选点。固定 batch8、BF16、cuda:0。H 按实际中心频率选取 (0.8,8] Hz 的 41 个尺度，保留原 97×360 网格。每窗口三个偏移依 row identity 预先确定，与 seed 和 batch 无关。

| split | 窗口/受试者 | 条件×seed | 条件窗口指标行数 | 案例图（PNG 和 SVG 各） |
|---|---|---|---:|---:|
| validation | 2675/7 | 10×3 | 80250 | 63 |
| research-test | 2310/8 | 10×3 | 69300 | 72 |

后台任务 `j-je0h3d` 于北京时间 2026-10-09 15:05:11 开始、16:52:28 完成，退出码 0，12 个流水线阶段均成功；未发现 failed marker。六份 checks 均记录 FULL/native 与 FULL/FIXED 波形差为 0，固定 GN 与当批 FULL 统计一致，时域 z 不变；每 seed 覆盖 validation 335 批/test 289 批，均包含尾批。

完成后只读核验 evaluation、summary 和图册 receipt 所列 **1714 项文件 SHA256**，全部通过。核验只读取既有产物，未新增推理、重跑指标或修改结果。两套图册及汇总完整；test 受试者效应图已目视检查。

## 3. 固定 GN 的五指标配对变化

以下先同 seed 同窗口与本次 FULL 配对，再对三个 seed 等权平均。Whole/Local RR 单位 bpm，其余为原指标单位；PCC 列为 FULL−干预，四项误差为干预−FULL，正值统一表示变差。括号为变差 seed 数，分母 3。

### Validation

| 条件 | Whole RR | Local RR | 包络轨迹 MAE | 全局包络误差 | PCC 下降 |
|---|---:|---:|---:|---:|---:|
| SHIFT_1/FIXED | −0.008345 (0) | +0.027229 (3) | +0.006098 (3) | +0.008883 (3) | +0.008634 (3) |
| SHIFT_2/FIXED | +0.000489 (2) | +0.033118 (3) | +0.006745 (3) | +0.011081 (3) | +0.009116 (3) |
| SHIFT_3/FIXED | −0.001451 (1) | +0.036290 (3) | +0.006559 (3) | +0.008643 (1) | +0.008951 (3) |
| H_MEAN/FIXED | −0.014141 (0) | +0.019312 (3) | −0.002485 (0) | +0.010290 (1) | +0.004567 (3) |

三个错位的包络 MAE 相对增加 **3.978%、4.404%、4.288%**；Local RR 相对增加 **5.228%、6.373%、6.976%**。百分比为同 seed 相对变化的三 seed 平均，不是跨 seed 均值之比。

### Research-test

| 条件 | Whole RR | Local RR | 包络轨迹 MAE | 全局包络误差 | PCC 下降 |
|---|---:|---:|---:|---:|---:|
| SHIFT_1/FIXED | +0.054030 (3) | +0.025112 (3) | +0.004320 (3) | +0.005303 (2) | +0.002657 (3) |
| SHIFT_2/FIXED | +0.043964 (3) | +0.016717 (3) | +0.003891 (3) | +0.004191 (3) | +0.002557 (3) |
| SHIFT_3/FIXED | +0.062886 (3) | +0.020395 (3) | +0.004411 (3) | +0.006468 (3) | +0.002535 (3) |
| H_MEAN/FIXED | +0.055873 (3) | +0.013830 (2) | −0.000684 (1) | +0.004392 (1) | −0.000622 (1) |

三个错位的包络 MAE 相对增加 **2.993%、2.716%、3.069%**；Local RR 相对增加 **3.969%、2.584%、3.108%**。自然 GN 得到相近的包络/PCC 方向和量级，完整 NAT/FIXED、seed SD、逐 seed 与 subject-macro 结果保留在原汇总中。本次 FULL 基准为包络 0.145531、PCC 0.871759、Local RR 0.639980；不混用旧 batch32 常规 test 基准。

## 4. 受试者差异与 H_MEAN 的解释

将三个预定 SHIFT/FIXED 和三个 seed 等权平均后：

| split | 包络误差增加 | PCC 下降 | Local RR 误差增加 |
|---|---:|---:|---:|
| validation | 6/7 人 | 6/7 人 | 6/7 人 |
| research-test | 8/8 人 | 6/8 人 | 4/8 人 |

这些是平均重复测量后的方向数，不意味着每个 subject×seed×shift 均同向。test 各 SHIFT 的包络恶化受试者数分别为 seeds 11/12/13 的 **8/8/5、8/8/6、8/8/6**。三种 SHIFT 的 subject-macro 包络与 PCC 亦在三个 seed 上均变差。

test 的 RR 总体退化主要集中于 670：平均三个错位与三个 seed 后，670 的 Local RR 增加 0.618199 bpm，对全体窗口均值贡献 +0.021142 bpm；全体净变化为 +0.020741 bpm，其余七人的窗口加权变化为 −0.000415 bpm。Whole RR 亦有相同集中趋势。这是基于已保存受试者表的完成后描述性分解，不改变主表样本集合或评价合同，不构成新增排除规则。

包络效应分布更广：test 全体平均增量 +0.004207，670 之外七人的窗口加权增量仍为 +0.003811。PCC 相应为下降 0.002583 与 0.001582。

H_MEAN/FIXED 的 validation 包络误差平均下降 1.622%（3/3 seed 改善）；test 平均下降 0.449%（2/3 seed 改善）。test PCC 的窗口加权均值提高 0.000622，但受试者等权均值下降 0.002363。结果表明各干预检验的问题不同：高频错位引入的跨路径不一致，不能完全等同于抹去高频动态信息。当前没有证据把高频时间起伏的全部成分都描述为有益。

## 5. 信号对应与图件

高频尺度平均 log 幅度经 10 秒均值后，与正式 THO 中心化 log-RMS 包络作零时滞有符号相关。受试者等权平均如下，SHIFT 为三个预定平移平均：

| split | FULL r | SHIFT r | 配对有效窗口 |
|---|---:|---:|---:|
| validation | 0.358164 | −0.039352 | 2675/2675 |
| research-test | 0.286706 | −0.013731 | 2310/2310 |

test 八人均为 FULL 高于 SHIFT。该指标描述幅度与包络的对应，不是模型信息贡献率；与旧 APOR 先投影呼吸调制带再取包络的定义不同。载频×调制频率周期图、逐尺度有效支持和所有预选案例均保存。

图件推荐先查看完整受试者效应和调制谱；案例用于解释条件读取—加性残差—输出的变化，不替代总体效应。全部案例按受试者首/中/末窗口预选，未按干预效果重选。

## 6. 证据边界与论文使用

可支持的表述：**在冻结的 M4 模型中，高频 CWT 幅度与原窗口的时间对应对呼吸重建具有功能价值；固定条件分支 GN 后，时间错位仍稳定损害包络轨迹和波形相关。**

均值替换和既有 Full/H/L/NoTF 对照一并保留，避免将功能依赖解释为所有指标的额外性能收益。当前主路径始终使用宽带 BCG；本实验没有识别特定生理来源，也没有完成整个输入的高频移除重训练。seed SD 描述训练波动，不是独立人群置信区间。research-test 曾用于开发，仍保留 reused research/development evidence 属性。

## 7. 产物入口

输出根：`/mnt/disk_code/marques/resp_reconstruction/runs/m4_high_frequency_v1/background_20261009_01`。

- [完整启动与阶段日志](/mnt/disk_code/marques/resp_reconstruction/runs/m4_high_frequency_v1/background_20261009_01/launcher.log)。
- [Validation 汇总](/mnt/disk_code/marques/resp_reconstruction/runs/m4_high_frequency_v1/background_20261009_01/validation/summary/attempt_ea85f8a5a9914188ac7e3a899e3fa79e/three_seed_summary.csv)。
- [Research-test 汇总](/mnt/disk_code/marques/resp_reconstruction/runs/m4_high_frequency_v1/background_20261009_01/research_test/summary/attempt_c6756642581b4d83852d889742131f21/three_seed_summary.csv)。
- [Validation 图册](/mnt/disk_code/marques/resp_reconstruction/runs/m4_high_frequency_v1/background_20261009_01/validation_figures/index.html)。
- [Research-test 图册](/mnt/disk_code/marques/resp_reconstruction/runs/m4_high_frequency_v1/background_20261009_01/research_test_figures/index.html)。

每个 session 的 manifest/源码快照、每 seed 的 environment/config/rows/shifts/checks/metrics/案例及 receipt、summary 的 sources 和图册 receipt 均保留。后续只读整理使用这些产物；当前完成的推理、汇总和图册不自行重跑。
