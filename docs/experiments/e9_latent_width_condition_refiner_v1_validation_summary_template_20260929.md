# E9：validation 一次性汇总模板

状态：**模板；仅在18/18 formal train/validation 与唯一 implementation lock 完整回载后填写并冻结。**

## 1. 冻结身份

| 字段 | 值 |
|---|---|
| Protocol | `e9-latent-width-condition-refiner-v1` |
| Implementation lock path/SHA | `<pending>` |
| Formal completed matrix | `<must be 18/18>` |
| Validation windows / subjects | `<pending>` |
| Summary attempt / manifest SHA | `<pending>` |
| Source commit / environment | `<pending>` |
| Research-test accessed | `false` |

## 2. E9 五主指标

三 seed window-direct arithmetic mean ± sample SD；前四项越低越好，PCC 越高越好。

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `e9a_d96_h65` | `<mean ± SD>` |  |  |  |  |
| `e9a_d96_h64` |  |  |  |  |  |
| `e9a_d96_h48` |  |  |  |  |  |
| `e9b_d64_direct` |  |  |  |  |  |
| `e9b_d64_h64` |  |  |  |  |  |
| `e9b_d64_h48` |  |  |  |  |  |

## 3. 冻结 E8 历史参照

这些数值只提供跨实验背景，不进入 E9 paired-seed 计划对比，不替代 E9-A 同轮 H65 锚点。来源为 E8 冻结 validation summary。

| Historical E8 arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `direct/pointwise` | 0.492749 ± 0.031169 | 0.545727 ± 0.021832 | 0.152145 ± 0.005044 | 0.190626 ± 0.006415 | 0.861968 ± 0.004021 |
| `fill65/pointwise` | 0.500708 ± 0.018403 | 0.553073 ± 0.012575 | 0.152950 ± 0.001474 | 0.191168 ± 0.002073 | 0.865227 ± 0.001423 |
| `res96/pointwise` | 0.498948 ± 0.025619 | 0.545092 ± 0.009639 | 0.154157 ± 0.000559 | 0.185980 ± 0.002262 | 0.863997 ± 0.003827 |
| `res192/pointwise` | 0.490762 ± 0.014062 | 0.544298 ± 0.011659 | 0.154646 ± 0.004454 | 0.193695 ± 0.007865 | 0.865859 ± 0.001329 |

## 4. 计划对比

Error 同时报告原始差与相对改善率；PCC 报告绝对差。正向统一后的效用值仅用于方向显示，不构造跨指标总分。

| Contrast | Whole RR | Local RR | Trajectory | Global modulation | PCC | paired-seed directions | Classification |
|---|---:|---:|---:|---:|---:|---|---|
| `H64−H65` |  |  |  |  |  |  |  |
| `H48−H65` |  |  |  |  |  |  |  |
| `H48−H64` |  |  |  |  |  |  |  |
| `D64-H64−D64-direct` |  |  |  |  |  |  |  |
| `D64-H48−D64-direct` |  |  |  |  |  |  |  |
| `D64-H48−D64-H64` |  |  |  |  |  |  |  |

跨宽度描述表：每个 D64 arm 分别对 E9 同轮 D96-H65、D96-H64；同一行必须同时包含五主指标、参数与 covered MAC，分类固定为“容量—效率联合变化”，不得标为纯 refiner 效应。

## 5. Local RR 尾部与 subject-macro

| Arm | median | P90 | P95 | >2 bpm | >5 bpm |
|---|---:|---:|---:|---:|---:|
| `<six E9 arms>` |  |  |  |  |  |

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC | subjects |
|---|---:|---:|---:|---:|---:|---:|
| `<six E9 arms>` |  |  |  |  |  |  |

任何 window-direct Pareto 判断都必须并列陈述 subject-macro 与 Local RR tail 是否出现反向证据。

## 6. 容量与工程验收

| Arm | Params | state bytes | covered MACs | acceptance batch | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| `<six E9 arms>` |  |  |  |  |  |  |

## 7. 决策

固定阈值：error 相对0.5%，PCC 绝对0.002。按主协议分别填写 E9-A、E9-B、跨宽度保护线、Pareto 集、paired-seed方向与反向证据。不按单项最好值强选赢家，不写跨指标加权总分。

建议结论句：

> 在当前数据与训练合同下，`<arm>` 相对 `<reference>` 为 `<容差内等效/Pareto改善/属性交换/材料性退化>`。其容量证据为 `<params/MAC>`；结论限于所测候选，不意味着64或48具有理论最优性。
