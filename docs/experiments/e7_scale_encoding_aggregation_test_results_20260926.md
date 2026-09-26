# E7 聚合前尺度编码 × 尺度聚合：research-test 结果与收口

日期：2026-09-26。状态：**固定 18-checkpoint research-test 评价与完整矩阵汇总均已完成并冻结。**

机器可读收口为 `docs/experiments/e7_scale_encoding_aggregation_test_closeout_20260926.json`。

## 1. 结论

**Research-test 没有发现五指标稳定 Pareto 改善，与 validation 的整体决定一致：在当前结构族、训练预算和容差口径内，结果为保留浅层尺度编码提供有限支持。**

具体而言：

1. S1-MEAN 相对 S0-MEAN 没有稳定改善。Whole/Local RR 各有两个 seed 材料性退化；global envelope modulation 的均值改善伴随 seed 间混合方向。Subject-macro 中 global modulation 三 seed 改善，但 Whole/Local RR 仍各有两个 seed 退化，因此仍是属性权衡。
2. S2-MEAN 相对 S1-MEAN 没有提供完整尺度上下文的稳定增量收益。Global modulation 在 sample-direct 与 subject-macro 均为三个 seed 全部退化，其他指标也没有共同改善方向。
3. Frequency attention 的作用依赖 encoder。S0 下 attention 的 Whole/Local RR 三 seed 全部退化；S2 下 attention 的 global modulation 三 seed改善、envelope trajectory 两个 seed 改善，但 Whole/Local RR 主要退化，PCC 方向不稳定。
4. `encoder × aggregation` 交互存在属性结构：S2 相对 S1 的 attention 交互在 envelope、global modulation 和 PCC 上三 seed同向为正，但 Local RR 三 seed同向为负。这说明完整尺度上下文改变了 attention 的收益分布，没有形成五指标整体优势。

这里的“有限支持”表示六臂矩阵没有证明更深或完整轴尺度编码能改善完整五指标 Pareto 前沿；深层 encoder 与 attention 在 P5 机制诊断中已被证实会改变表示并被模型利用。

## 2. 冻结产物

Test lock SHA-256：

```text
0fcba39436c9fbc5390d483787a6987fc8fb7a5a25e0e126932941076d421188
```

完整汇总：

```text
runs/e7_scale_encoding_aggregation/research_test/summary/
summary_0fcba39436c9_20260926T040914Z_8436abbf29e6
```

Summary manifest SHA-256 为 `197b266d73ed6fc8bbe45950c4ffb5c77e5d0132f3e444dcec48b838ca0f9e10`；summary receipt SHA-256 为 `d48556580475f657e4eefe348464228a8cad13a83b6700aa1f3fe42b04ae17f2`。

18 个 cell 均完成质量验收，joint prediction 与 envelope Spearman prediction degeneration 计数均为零。全部评价使用提交 `27e1d37f0b92d03fac81efb5a2cf7adbd07cdcfc`、同一 GPU/软件环境和干净工作树。

主分支合入后续结构实验后，共享 trainer 的增量接口由
`e7_scale_encoding_aggregation_mainline_compatibility_20260926.json` 固定。该兼容记录只允许已列出的两个源码身份共存，不改变 E7 科学合同、历史产物或上述运行提交，也不开放重跑。

## 3. 六臂三 seed 均值

四项 error 越低越好，PCC 越高越好。

| Arm | Whole RR | Local RR | Envelope trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| S0-MEAN | 0.6178 | 0.6072 | 0.13947 | 0.17394 | 0.87646 |
| S0-ATTN | 0.6961 | 0.6696 | 0.13944 | 0.17094 | 0.87428 |
| S1-MEAN | 0.6582 | 0.6321 | 0.14071 | 0.16890 | 0.87807 |
| S1-ATTN | 0.6541 | 0.6349 | 0.14210 | 0.17265 | 0.87364 |
| S2-MEAN | 0.6601 | 0.6126 | 0.14233 | 0.18152 | 0.87816 |
| S2-ATTN | 0.6864 | 0.6522 | 0.13936 | 0.17246 | 0.87696 |

没有单一 arm 同时最优五项指标。S0-MEAN 的 Whole/Local RR 最低；S2-ATTN 的 envelope trajectory 最低；S1-MEAN 的 global modulation 最低；S2-MEAN 的 PCC 最高。

## 4. 核心效应

下表正值表示改善。四项 error 为相对 utility 百分比，PCC 为绝对 utility 差；括号为 `改善/容差内/退化` seed 数。

| 效应 | Whole RR | Local RR | Envelope trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| S1-MEAN − S0-MEAN | −6.960 (1/0/2) | −4.256 (1/0/2) | −0.883 (1/1/1) | +2.752 (1/1/1) | +0.00162 (1/2/0) |
| S2-MEAN − S1-MEAN | −0.592 (1/0/2) | +3.001 (2/0/1) | −1.151 (0/1/2) | −7.446 (0/0/3) | +0.00008 (0/3/0) |
| S0-ATTN − S0-MEAN | −12.573 (0/0/3) | −10.170 (0/0/3) | +0.013 (1/1/1) | +1.513 (2/0/1) | −0.00218 (0/1/2) |
| S2-ATTN − S2-MEAN | −4.509 (1/0/2) | −6.841 (1/0/2) | +2.007 (2/0/1) | +4.897 (3/0/0) | −0.00120 (1/1/1) |

Endpoint interaction 的 Whole RR utility 三 seed同向为正，但 Local RR、两项 envelope 与 PCC 均为混合方向。更局部的 S1→S2 interaction 在 envelope trajectory、global modulation 与 PCC 上三 seed同向为正，同时 Local RR 三 seed同向为负；该交互属于属性重分配。

## 5. Subject-macro

Subject-macro 保留同一结论：

- S1-MEAN 相对 S0-MEAN 的 global modulation 三 seed改善，平均 utility 为 `+4.855%`；Whole/Local RR 平均 utility 为 `−14.828%/−4.866%`，各有两个 seed 退化。
- S2-MEAN 相对 S1-MEAN 的 global modulation 三 seed全部退化，平均 utility 为 `−11.296%`；envelope trajectory 与 PCC 也以退化为主。
- S0 attention 的 Whole/Local RR 三 seed全部退化，平均 utility 为 `−21.632%/−14.386%`。
- 其余 attention 比较均未形成跨五指标稳定方向。

因此结果不是由窗口数量较多的 subject 单独驱动；受试者等权口径仍不支持替换 S0-MEAN。

## 6. 与 W0 及 validation 的关系

同 seed 历史 W0 只作为冻结参照。S0-MEAN 相对 W0 的三 seed平均 utility 为：Whole RR `−0.071%`、Local RR `+0.386%`、envelope trajectory `+0.054%`、global modulation `−0.300%`、PCC `−0.00012`，说明本轮同结构对照在三 seed均值层面与 W0 接近。

Research-test 的个别属性方向与 validation 有变化，例如 S1-MEAN 的 envelope trajectory 收益没有在 sample-direct test 上复现；整体判断保持一致：S1/S2 与 attention 会改变属性分布，但没有稳定改善完整五指标 Pareto 前沿。

## 7. 证据边界

本轮新增 `41,580` 条逐窗口指标，test split 包含 2,310 个窗口和 8 个 `samp_id`。该 split 已参与多个历史科研阶段，因此本结果属于复用 research-test 上的开发性描述。Validation 的 checkpoint 选择、容差、指标和结论在 test 访问前已冻结，test 结果没有参与重选或训练调整。
