# E7 聚合前尺度编码 × 尺度聚合：P5 validation 收口

日期：2026-09-26。状态：**18-cell formal、完整 validation 汇总、selected-checkpoint 表征诊断和模块干预均已完成并冻结。当前证据限于 validation；research-test 尚未访问。**

机器可读收口为 `docs/experiments/e7_scale_encoding_aggregation_p5_closeout_20260926.json`。

## 1. 结论

**在当前结构族、训练预算、五指标容差和 validation 人群内，结果为保留浅层尺度编码提供有限支持，但不支持“深层模块没有提取信息”。**

具体而言：

1. S1/S2 显著改变了聚合前表示并被下游模型利用；它们提高尺度有效秩、降低尺度相关，关闭 residual 会稳定损害包络、全局调制和 PCC。
2. S1 的收益集中于包络属性，同时伴随 RR 代价；没有形成五指标 tolerance-aware Pareto 改善。
3. S2 相对 S1 没有稳定增量收益，当前证据不支持把完整尺度轴跨度视为主要瓶颈。
4. Frequency attention 学到了幅度很大的输入依赖修正，深层 encoder 下恢复均匀通常显著退化，但从头训练的结构比较仍存在 PCC/RR 代价。因此 attention 是被利用的模块，不是整体替换 W0 的依据。

这里的“浅层基本充分”只表示当前六臂矩阵没有证明更深/更宽编码能改善完整五指标 Pareto 前沿；它不表示浅层表示与深层表示等价，也不表示包络属性已经达到上限。

## 2. 冻结产物

CPU summary：

```text
runs/e7_scale_encoding_aggregation/p5_validation/summary/
summary_068ba8ec6c58_20260925T161725Z_58cf841bf690
```

Final：

```text
runs/e7_scale_encoding_aggregation/p5_validation/final/
final_068ba8ec6c58_20260925T184514Z_2cb10271e506
```

Summary/final manifest SHA-256 分别为 `89584f471443cb47ab8945720b95e3c711336259be3267026e1ac576543a00a6` 和 `aae2d23fecd116db73b239ea028b4e394e72300b09e7077d54bc8e4235ad9837`。Final 包含 18 行 seed metrics、25 行析因 contrast、35 行材料性汇总、720 行 subject-macro、54 行表征诊断及 135 行 checkpoint 干预差值。

## 3. 结构比较

以下效用变化正值表示改善；四项 error 使用相对百分比，PCC 使用绝对差。

### S1：局部加深

在 MEAN 下，S1 相对 S0：

- 包络轨迹平均改善 1.773%，三个 seed 均为材料性改善。
- 全局包络调制平均改善 5.117%，两个 seed 改善、一个处于容差内。
- Whole/Local RR 平均分别变化 −0.716%/−0.735%，均为两个 seed 材料性退化。
- PCC 平均变化 +0.000319，整体处于小幅混合方向。

因此 S1 提供的是包络属性收益与 RR 代价之间的权衡。

### S2：扩大尺度跨度

在 MEAN 下，S2 相对 S1 的五指标均没有稳定改善：Whole RR、全局调制和 PCC 各有两个 seed 材料性退化，PCC 平均下降 0.002438。Subject-macro 同样没有形成跨属性稳定方向。S2 的理论轴跨度没有转化为额外整体收益。

### Frequency attention

- S0 attention 的 Whole RR 平均改善 4.403%，但 PCC 平均下降 0.003785，且全局调制有两个 seed 退化。
- S1 attention 相对 S1 mean 的 PCC 三 seed 均材料性退化，平均下降 0.006577；包络轨迹和全局调制也以退化为主。
- S2 attention 的 Local RR 有两个 seed 改善，但 Whole RR、包络轨迹和 PCC 均以退化为主。

Attention 没有在任一 encoder 深度下形成稳定五指标 Pareto 改善。

## 4. 表征机制

诊断覆盖每个 cell 的完整 2,675 个 validation 窗口；相关矩阵在每窗口固定通道/时间网格上累计 1,444,500 个观测。

MEAN arm 的三 seed 平均：

| Arm | residual RMS/input RMS | effective rank `X0→XE` | lag-1 correlation `X0→XE` |
|---|---:|---:|---:|
| S1 | 2.851 | 14.384→29.206 | 0.879→0.747 |
| S2 | 2.731 | 13.482→34.983 | 0.885→0.744 |

新增 encoder 不是微小扰动：残差 RMS 超过输入 RMS 两倍，并显著扩展尺度有效秩。S2 比 S1 产生更高的有效秩，但该表征变化没有转化为稳定任务收益。

Attention correction RMS/mean-feature RMS 三 seed 平均为：S0 `0.507`、S1 `0.833`、S2 `0.654`。对应 attention entropy 从 S0 的 `4.345` 降至 S1/S2 的 `3.141/3.077`；均匀 97 槽的理论 entropy 为约 `4.575`。这说明深层 encoder 下 attention 学到了更集中的尺度选择，而不是接近均匀的小修正。

## 5. Checkpoint 内干预

- S1-MEAN 关闭 residual 后，包络轨迹、全局调制和 PCC 在三个 seed 全部退化；Whole RR 在三个 seed 全部改善。
- S2-MEAN 呈相同属性分化：关闭 residual 后全局调制和 PCC 三 seed 全部退化，Whole RR 三 seed全部改善。
- S1-ATTN 关闭 residual 后，Local RR、包络轨迹、全局调制和 PCC 三 seed全部退化。
- S2-ATTN 恢复均匀聚合后，五主指标在三个 seed 全部退化。
- S1/S2-ATTN 同时关闭 encoder residual 与 attention 后，包络轨迹、全局调制和 PCC 均在三个 seed 全部退化。

这些结果证明模块在各自已训练 checkpoint 内被强烈利用，但 checkpoint 内移除属于分布外干预；它解释功能利用，不能覆盖从头训练的六臂结构比较。

## 6. 实现修订与失败生命周期

17 个 diagnostics 使用提交 `137891e`；最后一个 cell 使用 `2bbad33`，只把 97 项 FP32 attention 求和审计从固定 `1e-6` 改为机器精度归约界，模型和诊断数值公式未变。首次 diagnostic 失败现场保留。

首次 finalize 因 S0-MEAN 没有适用干预、历史零行 CSV 无列头而失败；`329b4f6` 只增加固定 schema 兼容，随后 final 成功。失败 final 目录保留，未覆盖任何成功产物。

## 7. 后续边界

E7 validation 阶段至此关闭。若进入 research-test，必须另建专项协议并固定全部 18 个 validation-selected checkpoint；test 不参与结构筛选、checkpoint 选择、阈值或诊断规则修改。既有 E4 与本轮 validation 已参与设计，因此后续 test 只能解释为 research-test-informed development 证据。
