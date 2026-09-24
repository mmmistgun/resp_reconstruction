# E6 20-Hz 学习解调前端：validation 结果

日期：2026-09-24。协议：`e6-temporal-frontend-v1-20260923`。

## 1. 结论

`e6_tfe201_aa20_demod10_w0` 在冻结 validation 口径下形成明确的属性权衡，不替换 W0。Pooled envelope trajectory MAE 改善 `2.6263%`，但 Whole RR、Local RR、global envelope error 分别恶化 `1.9856% / 4.5246% / 4.5215%`，signed PCC 下降 `0.015607`。Local RR 与 PCC 三个 seed 全部退化。

受试者等权结果没有复现 pooled trajectory 收益：trajectory 平均恶化 `2.0269%`，W0 在 `14/21` 个 seed×samp_id 单元更优。Local RR、global envelope 和 PCC 的 subject-macro 也分别恶化 `7.5204% / 6.6744% / 0.018241`。因此 trajectory 的 pooled 改善不足以支持模型替换。

相对 E5，E6 五项候选均值均恢复：Whole/Local/trajectory/global error 分别降低约 `1.7434% / 2.3420% / 4.5268% / 0.9093%`，PCC 提高 `0.001298`。这与“在最终降至 10 Hz 前完成学习非线性”具有结构价值的假设一致，但不是顺序的单因素因果证明，因为 E5 与 E6 还同时改变了前端宽度、归一化和局部编码。

## 2. 来源与完整性

三个 formal attempt 为：

| Seed | Completed epoch | Selected epoch | Attempt |
|---|---:|---:|---|
| 20260811 | 30 | 4 | `runs/e6_temporal_frontend/formal/seed_20260811/formal_d139aca0a26b_20260923T071617Z_61bfe5eca7d9` |
| 20260812 | 30 | 4 | `runs/e6_temporal_frontend/formal/seed_20260812/formal_d139aca0a26b_20260923T111319Z_21267b72bee0` |
| 20260813 | 30 | 4 | `runs/e6_temporal_frontend/formal/seed_20260813/formal_d139aca0a26b_20260923T114928Z_c3aa1615a422` |

三者均绑定 implementation lock SHA-256 `d139aca0a26bd7fb1715e7f1fbd802e113e2e9cbc347ed17b4d2133ff613aea3`，完成 2,400 optimizer updates、2,675 条 validation，并按 `min_epoch=30 / patience=15 / min_delta=0` 在 epoch 30 触发 early stopping。checkpoint/history 全有限，prediction-degeneracy count 均为 0，validation row-order SHA-256 均为 `b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a`。

一次性汇总位于：

```text
runs/e6_temporal_frontend/summary/summary_d139aca0a26b_20260923T143709Z_4382db584e0d
```

汇总 manifest SHA-256 为 `66b93fee3e1a66d7807528e80a4c7236895501a0d38848653944063861d1b0c4`。三个 seed-20260812 工程失败 attempt 均保留，但没有进入正式汇总。

## 3. 三 seed 主指标

| 指标 | W0 mean ± sample SD | E6 mean ± sample SD | E6 变化 | E6 改善 seed |
|---|---:|---:|---:|---:|
| Whole RR absolute error | 0.499298 ± 0.018320 | 0.509212 ± 0.041626 | +1.9856% | 1/3 |
| Local RR MAE | 0.551309 ± 0.012064 | 0.576253 ± 0.013811 | +4.5246% | 0/3 |
| Envelope trajectory MAE | 0.152729 ± 0.001423 | 0.148718 ± 0.006460 | −2.6263% | 2/3 |
| Global envelope modulation error | 0.191051 ± 0.003022 | 0.199689 ± 0.009397 | +4.5215% | 1/3 |
| Lag-aware signed PCC | 0.865300 ± 0.001491 | 0.849693 ± 0.002342 | −0.015607 | 0/3 |

四项 error 的变化为三 seed 均值之差相对 W0 均值；PCC 报告绝对下降。Error 正值表示 E6 更差。

## 4. 受试者等权描述

| 指标 | Subject-macro 平均变化 | W0 更优 | E6 更优 |
|---|---:|---:|---:|
| Whole RR | +15.4901% | 10/21 | 11/21 |
| Local RR | +7.5204% | 14/21 | 7/21 |
| Envelope trajectory | +2.0269% | 14/21 | 7/21 |
| Global envelope modulation | +6.6744% | 13/21 | 8/21 |
| Signed PCC drop | +0.018241 | 20/21 | 1/21 |

窗口重叠、同一受试者的多个窗口和三个模型 seed 不作为新增独立受试者；方向数只用于描述稳健性，不是确认性显著性检验。

## 5. 当前决定

- 保留冻结 W0，E6 不进入主模型替换。
- E6 validation、formal attempts、工程失败现场和 summary 保持不可覆盖。
- E6 test 若执行，只能评价固定 epoch-4 checkpoint，并作为复用 research-test 上的开发性描述；不得据 test 重选 checkpoint、改变 validation 决定或声称新的独立确认。
- W0 `2×2×2` 结构因子实验使用独立 worktree、协议和输出 identity，不与 E6 生命周期混合。
