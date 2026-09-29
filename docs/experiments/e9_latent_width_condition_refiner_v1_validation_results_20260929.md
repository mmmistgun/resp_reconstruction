# E9：Latent Width and Condition Refiner Study validation结果

日期：2026-09-29。协议 ID：`e9-latent-width-condition-refiner-v1`。

状态：**18/18 formal train/validation与一次性validation汇总均已完成并冻结；本文件是E9当前validation证据入口。Research-test未访问。**

## 1. 核心结论

**D=96下采用规整的H=64条件末端。** 相对同轮H=65锚点，H64在五项主指标上全部形成材料性改善：Whole RR、Local RR、trajectory、global modulation分别改善3.6471%、0.6916%、2.3811%、1.2901%，PCC提高0.002624。Subject-macro同步支持这一选择，Whole/Local RR与trajectory改善，PCC提高0.003909；Local RR尾部的median、P90和P95也更低。H64因此是本轮最清晰的结构结论：历史参数预算产生的65维瓶颈可以规整为64维，并获得更好的validation质量。

H48相对H64没有继续压缩的依据。两者Whole RR、Local RR与PCC处于容差内，但H48的trajectory和global modulation分别恶化1.8008%和2.0337%。E9-A最终选择`e9a_d96_h64`，不把H48升级为主候选。

D=64揭示了清楚的节律—形态取舍：

- `D64-direct`获得D64组最低Whole/Local RR，同时参数最少；
- `D64-H64`获得全矩阵最低trajectory与global modulation，并进入tolerance-aware Pareto集合；
- `D64-H48`相对H64主要交换为PCC改善，同时Whole RR、trajectory和global modulation转差。

因此D64组不强选全面赢家。若目标是D64内部的节律与简洁性，优先`D64-direct`；若目标是形态与全局调制，`D64-H64`提供对应优势。三种D64结构相对同轮D96-H64均未通过五属性质量保护线，当前不替换D96-H64主结构；它们保留为容量—效率候选。

## 2. 冻结身份与完整性

- Implementation lock：`8bb3b1992f64fc90959d3c16116e5ee9ac6463131ac01bb0f2473c068298367b`。
- Formal runtime amendment：`1c8b58249d7173743128f8c8b603c5ca643d9cfabf97f5e521b207d51fb79566`。
- Formal：6 arms × 3 seeds = 18个唯一成功attempt；`completed=18 / failed=0 / pending=0 / running=0`。
- Validation：每个attempt 2,675行、7个`samp_id`；总计48,150行。
- Checkpoint：18个best与18个final checkpoint均完成config、state、optimizer、epoch/update和finite回放。
- 训练合计632 epochs、50,560 optimizer updates；逐run完成30–50 epochs，selected epoch为9–35。
- 逐run wall-time求和约10.89 h；该值不代表双GPU并行后的端到端时间。
- 汇总只读取冻结validation产物并在CPU回放合同；没有训练、推理、GPU或research-test访问。

冻结汇总路径：

```text
runs/e9_latent_width_condition_refiner_v1/validation_summary/
summary_8bb3b1992f64_20260929T144128Z_f1242dd035e9
```

- Manifest SHA-256：`93ae65b2a19c3b76bb3fb6656ce4190d26e994abb0028e810a9c1c2e1a29c81f`。
- Decision SHA-256：`ed7159d43894acea366beac74bdc1c056636bde02a3131f861fa353e85b690a6`。
- Summary receipt SHA-256：`53d0757449b10ee326c5b977f40ae84dcfa48fb8d1b19ac9e506fe93add23ebf`。
- Summary源码commit：`7320de8`。

## 3. 五项主指标

表中为三seed的window-direct arithmetic mean ± sample SD。前四项越低越好，PCC越高越好。

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `e9a_d96_h65` | 0.503649 ± 0.013405 | 0.550888 ± 0.012843 | 0.155127 ± 0.005501 | 0.190241 ± 0.003154 | 0.862778 ± 0.002992 |
| `e9a_d96_h64` | **0.485259 ± 0.021906** | 0.547016 ± 0.008780 | 0.151244 ± 0.002975 | 0.187743 ± 0.000935 | **0.865402 ± 0.001759** |
| `e9a_d96_h48` | 0.486917 ± 0.025570 | **0.545870 ± 0.015271** | 0.153883 ± 0.003577 | 0.191536 ± 0.006771 | 0.863891 ± 0.004620 |
| `e9b_d64_direct` | 0.490740 ± 0.020984 | 0.554204 ± 0.014322 | 0.154800 ± 0.005003 | 0.189612 ± 0.003070 | 0.862399 ± 0.001723 |
| `e9b_d64_h64` | 0.496876 ± 0.025203 | 0.560259 ± 0.011898 | **0.149109 ± 0.004556** | **0.186114 ± 0.009400** | 0.860759 ± 0.001638 |
| `e9b_d64_h48` | 0.508173 ± 0.021189 | 0.559118 ± 0.017567 | 0.153186 ± 0.003782 | 0.191416 ± 0.006657 | 0.863183 ± 0.002928 |

单项seed mean最优分布在三个arm：Whole RR/PCC=`D96-H64`，Local RR=`D96-H48`，trajectory/global modulation=`D64-H64`。Tolerance-aware Pareto集合为`D96-H64`与`D64-H64`；前者提供最均衡质量，后者代表形态导向的低容量解。

## 4. 预注册计划对比

Error列为相对改善率，PCC为绝对差；正值表示候选改善。

### 4.1 E9-A：D=96条件末端

| Contrast | Whole RR | Local RR | Trajectory | Global modulation | PCC | 分类 |
|---|---:|---:|---:|---:|---:|---|
| `H64−H65` | +3.6471% | +0.6916% | +2.3811% | +1.2901% | +0.002624 | tolerance-aware Pareto改善 |
| `H48−H65` | +3.3607% | +0.9176% | +0.7736% | -0.6663% | +0.001113 | 属性交换 |
| `H48−H64` | -0.4449% | +0.2220% | -1.8008% | -2.0337% | -0.001512 | H64质量保护更好 |

H64相对H65的改善覆盖五项主属性。Paired seed方向为Whole RR 2/3、Local RR 2/3、trajectory 1/3、global modulation 3/3、PCC 3/3；mean效应与global modulation/PCC的跨seed方向最稳定。H48相对H64的额外参数/MAC节省没有覆盖形态与调制质量差异。

### 4.2 E9-B：D=64条件末端

| Contrast | Whole RR | Local RR | Trajectory | Global modulation | PCC | 分类 |
|---|---:|---:|---:|---:|---:|---|
| `D64-H64−direct` | -1.2369% | -1.1006% | +3.6159% | +1.7756% | -0.001640 | RR—形态交换 |
| `D64-H48−direct` | -3.5985% | -0.8842% | +1.0256% | -0.9966% | +0.000784 | direct更适合节律/简洁目标 |
| `D64-H48−H64` | -2.3235% | +0.2126% | -2.8004% | -2.9162% | +0.002424 | H64形态优势，H48 PCC优势 |

结果没有支持D64条件末端的单一普遍选择。Direct删除额外重组后保持最好的D64 RR与最低容量；H64把收益集中到trajectory/global modulation；H48没有形成相对direct或H64的Pareto改善。

## 5. 跨宽度质量—容量关系

所有D64 arm相对D96-H64都出现至少一项材料性质量变化，因此不满足主模型替换保护线。最简洁的`D64-direct`相对D96-H64：Whole RR、Local RR、trajectory、global modulation分别变化-1.1881%、-1.3051%、-2.4199%、-0.9921%，PCC变化-0.003003。

相对历史同形态锚点D96-H65，`D64-direct`形成更集中的效率候选：Whole RR改善2.5914%，trajectory/global modulation/PCC处于容差内，Local RR变化-0.6001%。它没有跨过预注册的五属性保护线，但明确展示了D64容量下降后的节律—效率工作点。

以D96-H64为参照，三个D64 arm的工程缩减为：

| D64 arm | 参数减少 | covered MAC减少 | formal peak allocated减少 | formal peak reserved减少 |
|---|---:|---:|---:|---:|
| `D64-direct` | 51.40% | 47.89% | 25.58% | 21.94% |
| `D64-H64` | 50.73% | 46.17% | 25.58% | 21.94% |
| `D64-H48` | 50.89% | 46.60% | 25.58% | 21.94% |

上述MAC为协议声明覆盖范围，不是完整模型FLOPs。显存来自formal原生训练路径；D96 peak allocated约8,773 MiB、reserved约12,990–12,992 MiB，D64分别约6,529 MiB与10,140 MiB。

## 6. Subject-macro与Local RR尾部

D96-H64相对H65的subject-macro证据与window-direct主结论一致：Whole RR `0.641201→0.593745`、Local RR `0.646107→0.633117`、trajectory `0.166938→0.162334`、PCC `0.850515→0.854424`，global modulation基本持平（`0.222669→0.222710`）。

Local RR尾部中，H64相对H65的median `0.114596→0.113337`、P90 `1.604051→1.555023`、P95 `2.364866→2.327027`、`>2 bpm`比例 `6.7040%→6.6417%`；`>5 bpm`比例为`0.8972%→0.9969%`。尾部主体支持H64，极端阈值比例提醒最终评价继续并列呈现完整尾部，而不改变五主属性结论。

D64-direct的Local RR median/P90/P95分别为`0.113552/1.560945/2.310073`，`>2/>5 bpm`比例为`6.5421%/0.8598%`，显示其RR尾部仍具竞争力；这与其作为节律—效率候选的定位一致。

## 7. E8历史参照边界

冻结E8 pointwise历史列已纳入汇总：`direct/fill65/res96/res192`。这些结果来自另一轮12-arm训练，只用于说明E9候选所在的历史范围，不进入E9 paired-seed计划对比，也不替代本轮H65锚点。E9最重要的新证据是同轮H64对H65的五属性Pareto改善，而不是跨实验单项数值排序。

## 8. Validation决定与后续边界

1. E9-A选择`e9a_d96_h64`：它把任意H=65规整为H=64，并形成五属性validation Pareto改善。
2. E9-A不保留H48作为主替换候选；其额外压缩对应trajectory/global modulation交换。
3. E9-B报告两种目标明确的工作点：`D64-direct`面向节律与最小容量，`D64-H64`面向trajectory/global modulation；不强选全面赢家。
4. D64 arm均未相对D96-H64满足五属性质量保护线，因此当前不建议替换D96-H64主结构。
5. Validation checkpoint选择与结构决定现已冻结。Research-test仍需独立allowlist与当次明确授权；当前结果不包含test信息。

结论限于当前任务、数据与训练合同；H64是所测候选中的经验选择，不声称理论最优。

