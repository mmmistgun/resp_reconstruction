# E9：Latent Width and Condition Refiner Study closeout

日期：2026-09-29。训练协议 ID：`e9-latent-width-condition-refiner-v1`；research-test 协议 ID：`e9-latent-width-condition-refiner-v1-research-test-20260929`。

状态：**18/18 formal train/validation、18/18 固定 checkpoint research-test 与两阶段一次性汇总均已完成并冻结；E9 至此关闭。本文件是 E9 当前状态、结果与证据身份的统一入口。**

## 1. 结论

E9 的稳定结论不是某个条件末端宽度全面占优，而是潜在宽度与条件通道重组形成了可解释的节律—形态—效率取舍。

Validation 预先选择了 `D96-H64`：它相对同轮 `D96-H65` 在五项主指标上均改善，因此把任意的 65 维瓶颈规整为 64 维。固定 checkpoint research-test 没有复现这一五属性 Pareto 关系：`D96-H64` 的 trajectory、global modulation 与 PCC 更好，但 Whole/Local RR 分别恶化 5.4765% 和 4.7668%。因此，**保留 validation 已冻结的 `D96-H64` 主结构决定，但把其证据表述收窄为 validation 选择；reused research-test 显示它相对 H65 是节律—形态交换，而不是跨 split 的全面优势。**

`D96-H48` 没有形成替换 H64 的一致依据。Research-test 中它改善 Whole/Local RR，却使 trajectory、global modulation 与 PCC 退化；这与 validation 中 H48 的形态/调制代价方向一致。进一步压缩条件末端只能作为目标特定取舍，不能升级为默认结构。

D=64 下也不存在全面赢家：

- `D64-direct` 保持最低参数量，并在 research-test 获得 D64 组最低 Whole/Local RR；
- `D64-H64` 获得全矩阵最低 trajectory 与 global modulation，说明降到 64 维后，等宽残差重组仍能显著改善形态与调制；
- `D64-H48` 相对 H64 在 Whole RR、trajectory 与 global modulation 上均材料性退化，不支持继续缩窄。

三个 D64 arm 在 research-test 中仍未相对 `D96-H64` 通过预注册的五属性质量保护线，因而都不能替换当前主结构。它们保留为容量—效率候选：`D64-direct` 面向节律与最小容量，`D64-H64` 面向形态与调制。

## 2. 冻结身份与完整性

### 2.1 Train/validation

- Implementation lock：`8bb3b1992f64fc90959d3c16116e5ee9ac6463131ac01bb0f2473c068298367b`。
- Formal runtime amendment：`1c8b58249d7173743128f8c8b603c5ca643d9cfabf97f5e521b207d51fb79566`。
- Formal：6 arms × 3 seeds = 18 个唯一成功 attempt；`completed=18 / failed=0 / pending=0 / running=0`。
- Validation summary：`runs/e9_latent_width_condition_refiner_v1/validation_summary/summary_8bb3b1992f64_20260929T144128Z_f1242dd035e9`。
- Validation manifest SHA-256：`93ae65b2a19c3b76bb3fb6656ce4190d26e994abb0028e810a9c1c2e1a29c81f`。
- Validation decision SHA-256：`ed7159d43894acea366beac74bdc1c056636bde02a3131f861fa353e85b690a6`。

### 2.2 Research-test

- Allowlist：`runs/e9_latent_width_condition_refiner_v1/research_test/allowlist/allowlist_c14442d9b1ae_20260929T145212Z_286df79b0c0f`。
- Allowlist SHA-256：`0b5d0ea7e2f8be356a3827d053d6504c6647e568e3c6b17757114a1bb7add998`；固定 18 个 validation-selected checkpoint。
- 完整矩阵：`completed=18 / failed=0 / pending=0 / running=0`。
- 每个 attempt 为 2,310 windows、8 个 `samp_id`；共 41,580 行 test metrics，所有五项主指标 eligibility 均为 2,310/attempt，非有限/退化计数为 0。
- Test sample seed：`20260612`；split=`test`。
- Summary：`runs/e9_latent_width_condition_refiner_v1/research_test/summary/summary_0b5d0ea7e2f8_20260929T151109Z_c4a3508d4028`。
- Summary manifest SHA-256：`455c97396863a16be315102cd53390b53d7ce6d4c483b0bdc17f4ae0c8edfbea`。
- Decision SHA-256：`554e297d61eea4ff871b466458b436cd59cc7cae53eed4cbd43396e5266f9213`。
- Summary receipt SHA-256：`dac85274de8e75104718d5fee8a43fe3c6e966e6d062e929ac6bad4af56c928e`。
- Allowlist 阶段未读取 test arrays 或 dataset index；最终 summary 只读取 18 个冻结 evaluation 产物，未加载 checkpoint、未执行推理、未读取 test arrays。

## 3. Research-test 五项主指标

表中为三 seed 的 window-direct arithmetic mean ± sample SD。前四项越低越好，PCC 越高越好。

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `e9a_d96_h65` | **0.635974 ± 0.008030** | 0.610474 ± 0.014933 | 0.141833 ± 0.003701 | 0.174757 ± 0.005496 | 0.872766 ± 0.005497 |
| `e9a_d96_h64` | 0.670622 ± 0.020942 | 0.639657 ± 0.020953 | 0.138622 ± 0.002618 | 0.169883 ± 0.003182 | **0.878613 ± 0.002290** |
| `e9a_d96_h48` | 0.655574 ± 0.019636 | 0.610678 ± 0.016493 | 0.140736 ± 0.003441 | 0.172564 ± 0.003777 | 0.875269 ± 0.003382 |
| `e9b_d64_direct` | 0.664593 ± 0.035315 | **0.605825 ± 0.024000** | 0.138718 ± 0.001877 | 0.182278 ± 0.006096 | 0.876785 ± 0.002783 |
| `e9b_d64_h64` | 0.677947 ± 0.038432 | 0.627674 ± 0.039369 | **0.136739 ± 0.001559** | **0.168244 ± 0.004065** | 0.876880 ± 0.002440 |
| `e9b_d64_h48` | 0.697629 ± 0.035652 | 0.627608 ± 0.050590 | 0.137895 ± 0.001729 | 0.178803 ± 0.005174 | 0.877701 ± 0.001176 |

单项最优分散在四个 arm：Whole RR=`D96-H65`，Local RR=`D64-direct`，trajectory/global modulation=`D64-H64`，PCC=`D96-H64`。这正是本轮应报告的属性分工，不构造跨指标加权总分。

## 4. 预注册计划对比

Error 列为相对改善率，PCC 为绝对差；正值表示候选改善。材料性阈值为 error 相对 0.5%、PCC 绝对 0.002。

### 4.1 E9-A：D=96 条件末端

| Contrast | Whole RR | Local RR | Trajectory | Global modulation | PCC | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `H64−H65` | -5.4765% | -4.7668% | +2.1887% | +2.7190% | +0.005847 | 节律—形态交换 |
| `H48−H65` | -3.0742% | -0.0317% | +0.7702% | +1.2342% | +0.002503 | 属性交换 |
| `H48−H64` | +2.1218% | +4.5175% | -1.5789% | -1.6113% | -0.003345 | H48 不替换 H64 |

`H64−H65` 的 paired-seed方向为 Whole/Local RR 0/3，trajectory/global modulation/PCC 2/3。它说明 H64 的形态和相关性优势具有一定跨 seed 一致性，但 RR 代价同样是 3/3；research-test 不支持“五属性全面改善”的跨 split 延伸。

### 4.2 E9-B：D=64 条件末端

| Contrast | Whole RR | Local RR | Trajectory | Global modulation | PCC | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `D64-H64−direct` | -2.1862% | -3.7398% | +1.4233% | +7.6429% | +0.000095 | RR—形态交换 |
| `D64-H48−direct` | -5.0777% | -3.6761% | +0.5901% | +1.8884% | +0.000916 | direct 更适合节律/简洁目标 |
| `D64-H48−H64` | -2.9400% | +0.0722% | -0.8515% | -6.2830% | +0.000821 | H64 形态保护更好 |

`D64-H64−direct` 的 trajectory 与 global modulation 均在 3/3 seeds 改善；Whole/Local RR仅 1/3 seeds 改善。Direct 并非“无材料性退化”于两个 refiner，因为它相对 H64 明显损失形态和调制；H64 也不是全面 Pareto 改善，因为 RR 同时转差。H48 相对 H64 被 tolerance-aware 判为材料性退化。

## 5. 跨宽度质量—效率关系

以 validation-selected `D96-H64` 为参照：

| D64 arm | Whole RR | Local RR | Trajectory | Global modulation | PCC | 质量保护线 |
|---|---:|---:|---:|---:|---:|---|
| `D64-direct` | +0.8308% | +5.2996% | -0.0948% | -7.2837% | -0.001828 | 未通过：global modulation |
| `D64-H64` | -1.0398% | +1.8199% | +1.3418% | +0.9636% | -0.001734 | 未通过：Whole RR |
| `D64-H48` | -3.9984% | +1.8814% | +0.4894% | -5.2374% | -0.000913 | 未通过：Whole RR、global modulation |

效率数据沿用冻结 formal 记录，不从 test 重新 benchmark：

| Arm | 参数量 | 协议覆盖 MACs | 相对 D96-H64 参数减少 | 相对 D96-H64 MAC 减少 |
|---|---:|---:|---:|---:|
| `D96-H64` | 1,219,658 | 855,878,400 | — | — |
| `D64-direct` | 592,706 | 445,985,280 | 51.40% | 47.89% |
| `D64-H64` | 600,962 | 460,730,880 | 50.73% | 46.17% |
| `D64-H48` | 598,914 | 457,044,480 | 50.89% | 46.60% |

D64 formal peak allocated/reserved约为 6,529/10,140 MiB，D96约为 8,773/12,990–12,992 MiB。MAC为协议声明的覆盖范围，不是完整模型 FLOPs。质量变化不能单独归因于条件末端，因为 D64 同时改变了全局表示容量。

## 6. Subject-macro 与 Local RR 尾部

Subject-macro保持相同的属性分化。`D96-H65→H64` 时 Whole RR `1.406792→1.513487`、Local RR `1.196777→1.240862`，而 trajectory `0.164448→0.156760`、global modulation `0.218242→0.207645`、PCC `0.829406→0.835974`。这与 window-direct 的 RR—形态交换一致。

Local RR 尾部中，H65/H64/H48 的 median 分别为 `0.083047/0.081280/0.081640`，P90为 `1.448299/1.509885/1.467510`，P95为 `3.075630/3.293732/3.166041`，`>2 bpm`比例为 `7.3593%/7.6190%/7.5036%`，`>5 bpm`比例为 `3.0447%/3.2035%/2.9437%`。H64仅改善中心位置，P90/P95与阈值尾部均不支持其 RR 优势。

D64-direct 的 Local RR median/P90/P95为 `0.081372/1.385938/2.976356`，`>2/>5 bpm`比例为 `7.1429%/3.0447%`；其节律候选定位同时得到 window-direct 与尾部统计支持。

## 7. Validation—test关系与证据边界

Validation与research-test的方向不是完全一致：

- `D96-H64−H65` 从 validation 五属性改善变为 test 中 RR退化、形态/PCC改善；
- H48 相对 H64 的形态与调制代价保持，但 RR在test中转为明显改善；
- D64-direct 的节律优势与 D64-H64 的形态/调制优势保持；
- 所有 D64 arm 未通过相对 D96-H64 的五属性质量保护线这一决定保持。

本轮 test split 已参与既往研究开发，证据角色固定为 **reused research/development evidence**。它用于检查冻结结构方向在该 split 上的表现，不构成独立 held-out 确认，不用于重选 checkpoint、阈值、arm或追加宽度候选。Seed仅描述优化随机性；重叠 windows 不是独立统计样本，结果不作总体推断。

## 8. 最终冻结决定

1. 保留 validation 已冻结的 `D96-H64` 主结构选择，但结论限定为 validation-supported；research-test 将其相对 H65 定义为 RR—形态交换。
2. 不将 `D96-H48` 升级为默认条件末端；它没有跨 validation/test 形成相对 H64 的全面改善。
3. D64 内部保留两个用途明确的效率工作点：`D64-direct` 面向节律与最小容量，`D64-H64` 面向形态与调制；不强选全面赢家。
4. `D64-H48` 不优先于上述两个工作点。
5. 三个 D64 arm 均不替换 D96 主结构；任何部署选择必须同时报告质量、参数量、覆盖 MACs 与 formal 显存。
6. E9 不再根据本轮 test 追加 D/H 候选、重训、重选或修改材料性阈值。

结论只适用于当前任务、数据、模型与训练合同；64和48均是经验候选，不声称理论最优。
