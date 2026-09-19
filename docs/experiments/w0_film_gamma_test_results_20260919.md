# W0 FiLM gamma 系数 research-test 比较结果

协议：`w0-film-gamma-test-v1-20260919`  
性质：固定 validation-selected checkpoints 在复用 research/development test split 上的描述性比较

## 1. 实验范围与证据属性

本轮固定比较 validation 阶段唯一 quality candidate `GAMMA_040`（$c_\gamma=0.4,c_\beta=0.5$）与同 seed 的既有 W0 基线 `GAMMA_050`（$c_\gamma=0.5,c_\beta=0.5$）。三个 `GAMMA_040` checkpoint 均在训练阶段按完整 validation Local RR 最小值选定；test 结果未参与 epoch、条件、指标或统计口径选择。

| Seed | GAMMA_040 selected epoch | GAMMA_050 selected epoch | Test windows | Subjects |
|---|---:|---:|---:|---:|
| 20260811 | 13 | 13 | 2310 | 8 |
| 20260812 | 30 | 15 | 2310 | 8 |
| 20260813 | 14 | 14 | 2310 | 8 |

`GAMMA_040` 新执行三次固定 checkpoint evaluation，共生成 6930 条逐窗口 metrics；`GAMMA_050` 复用同 seed 的既有冻结 research-test metrics，不重复推理。

本项目的 test split 已用于既往 CRD-TF 研究评价。因此，本轮证据仅用于描述两个固定条件在复用 research/development test split 上的相对表现，不是首次触碰的 held-out 评价，也不构成无偏独立泛化确认。

## 2. 运行完整性与复现身份

三次 evaluation 与一次 summary 均已完成，且各自产物 manifest 通过哈希核验。三次 evaluation 均满足：2310 个唯一 test windows、8 个 `samp_id`、五主指标 finite、target eligibility 与 W0 一致、prediction degeneracy count 为 0，row-order SHA-256 均为 `184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。

复现身份如下：

- implementation lock SHA-256：`0d2bbbcc5ee6fee8e442475600c450de55c8e03ba31f18f5043e0557ac1ae712`；
- summary：`/mnt/disk_code/marques/resp_reconstruction/runs/w0_film_gamma_test_v1/summary/summary_0d2bbbcc5ee6_20260919T082329Z_78d16bc7986e`；
- summary manifest SHA-256：`a544ec5079c8bf93232ddcd81cd3eb848c34c9ec53bb007b87118b3e3a800bf4`；
- evaluation attempts：
  - seed 20260811：`evaluation_0d2bbbcc5ee6_20260919T074642Z_8dfb018706a5`；
  - seed 20260812：`evaluation_0d2bbbcc5ee6_20260919T074805Z_6b9323dcea14`；
  - seed 20260813：`evaluation_0d2bbbcc5ee6_20260919T074923Z_4a3f4228b0df`。

summary 主表规模为 `seed_metrics.csv` 6 行、`paired_seed_delta.csv` 15 行、`three_seed_comparison.csv` 5 行，符合协议预期并通过 `verify_attempt`。三次 evaluation 和 summary 的 lifecycle、receipt、artifact manifest 与 freeze receipt 均完整，来源哈希核验通过。

## 3. 五主指标的三-seed 汇总

下表原值为每个 seed 在完整 2310-window test 上聚合后，再对三个 seed 求均值。四项 error 的 paired delta 定义为

$$
100\times\frac{E_{040}-E_{050}}{E_{050}},
$$

因此负值表示改善、正值表示恶化；PCC 报绝对差 `PCC_040-PCC_050`，正值表示改善。`±` 后为三个 paired-seed delta 的 sample SD。

| 指标 | GAMMA_050 | GAMMA_040 | Paired delta mean ± SD | 改善 seeds |
|---|---:|---:|---:|---:|
| Whole RR absolute error（bpm，↓） | 0.617234 | 0.636965 | +3.304981% ± 4.259956% | 1/3 |
| Local RR MAE（bpm，↓） | 0.609566 | 0.613446 | +0.625082% ± 1.823883% | 2/3 |
| Envelope trajectory MAE（↓） | 0.139546 | 0.140160 | +0.445230% ± 2.133100% | 2/3 |
| Global envelope modulation error（↓） | 0.173418 | 0.177256 | +2.259124% ± 2.865089% | 0/3 |
| Lag-aware signed PCC（↑） | 0.876577 | 0.874237 | -0.00233988 ± 0.00570599 | 2/3 |

四项 error 的三-seed paired mean 均转为恶化；PCC 也平均下降，且下降幅度大于 validation 的 -0.000547。Local RR、trajectory 和 PCC 虽各有 2/3 seeds 按指标方向改善，但少数 seed 的较大反向变化主导了均值。Global modulation 则为 0/3 seeds 改善，是方向最一致的恶化轴。

## 4. 逐 seed 稳定性

| Seed | Whole RR | Local RR | Trajectory | Global modulation | PCC absolute delta |
|---|---:|---:|---:|---:|---:|
| 20260811 | +1.812% | -0.046% | -1.019% | +0.030% | +0.001789 |
| 20260812 | +8.110% | -0.768% | +2.893% | +5.491% | -0.008851 |
| 20260813 | -0.008% | +2.690% | -0.538% | +1.257% | +0.0000419 |

seed 20260811 在 Local RR、trajectory 和 PCC 上改善，但 Whole RR 与 global modulation 轻微恶化。seed 20260812 同时出现 Whole RR、trajectory、global modulation 和 PCC 的较大反向变化，仅 Local RR 改善；它是 Whole RR、global modulation 和 PCC 三-seed均值恶化的主要贡献者。seed 20260813 的 Whole RR、trajectory 和 PCC 略有改善，但 Local RR 与 global modulation 恶化。

因此，GAMMA_040 在复用 test 上没有形成跨 seed 的五轴一致收益。两个 RR 轴也未在同一组 seeds 上稳定改善：Local RR 的改善 seeds 为 20260811/20260812，Whole RR 仅 seed 20260813 轻微改善。

## 5. Subject 异质性

test 包含 8 个 `samp_id`。下表同时给出“三个 seed 的主体 delta 均值按指标方向改善”的主体数，以及“至少 2/3 seeds 在该主体上同向改善”的主体数。

| 指标 | 三-seed主体均值改善 | ≥2/3 seeds 同向改善 |
|---|---:|---:|
| Whole RR | 4/8 | 6/8 |
| Local RR | 4/8 | 3/8 |
| Trajectory | 3/8 | 8/8 |
| Global modulation | 3/8 | 3/8 |
| PCC | 1/8 | 5/8 |

主体均值与方向计数存在明显不一致。例如 trajectory 在全部 8 个主体上均有至少 2/3 seeds 改善，但只有 3/8 主体的三-seed均值改善；PCC 也从 5/8 的多数 seed 同向改善降为仅 1/8 的主体均值改善。这说明少数 seed 的较大反向幅度足以覆盖多数 seed 的较小正向变化。Whole RR 则出现 4/8 主体均值改善、但 6/8 主体至少 2/3 seeds 改善的相同幅度不对称现象。

这些统计用于描述主体异质性。各主体的窗口数不等且窗口相互重叠，主体数仅为 8；不能把窗口当作独立重复，也不据此进行独立显著性推断。

## 6. 与 validation 方向对照

训练验证阶段，GAMMA_040 相对 GAMMA_050 的结果为：

| 指标 | Validation paired delta mean | Validation 改善 seeds | Test paired delta mean | Test 改善 seeds |
|---|---:|---:|---:|---:|
| Whole RR | -2.297% | 2/3 | +3.304981% | 1/3 |
| Local RR | -1.678% | 3/3 | +0.625082% | 2/3 |
| Trajectory | -0.712% | 3/3 | +0.445230% | 2/3 |
| Global modulation | -2.179% | 1/3 | +2.259124% | 0/3 |
| PCC | -0.000547 | 2/3 | -0.00233988 | 2/3 |

validation 上四项 error 的 paired mean 均为改善；在复用 test 上四项均转为恶化。Whole RR 的改善 seed 数从 2/3 降为 1/3，Local RR 与 trajectory 从 3/3 降为 2/3，global modulation 从 1/3 降为 0/3。PCC 在两个 split 上均为平均下降，test 的负向幅度更大；其 2/3 改善 seeds 与负均值并存，再次体现幅度不对称。

test 协议没有定义新的 quality selector，因此这里不作“test gate 失败”的判定。适当表述是：validation 上支持 GAMMA_040 的平均 error 改善方向未在这次复用 test 比较中保持。

## 7. 观察与机制假设

直接观察是：降低 $c_\gamma$ 至 0.4 后，validation 的多 error 轴收益没有在复用 test 上复现为三-seed平均收益；差异同时具有明显的 seed 和 subject 异质性，尤其受 seed 20260812 的较大反向变化影响。

一种工作假设是，较低 gamma 可能限制 FiLM 乘性调制幅度，并在 validation 分布上形成更合适的误差折衷；但该折衷可能依赖训练优化路径、checkpoint 时点或主体组成，在复用 test 分布上不够稳健。另一种相容解释是，GAMMA_040 的部分 validation 收益来自少数主体或幅度较大的局部变化，换到不同主体组合后被反向幅度抵消。

这些都是机制假设，而非已证实的解释。当前比较同时包含完整重训练、不同优化轨迹和 validation checkpoint 选择的影响，不能从结果中分离出 gamma 系数的因果效应；现有三 seed 与 8 个主体也不足以区分分布差异、优化方差和主体构成各自的贡献。

## 8. 证据边界与阶段决策

本轮结果支持以下有限结论：

1. 在固定 validation-selected checkpoints、三个训练 seed 和完整复用 research/development test split 下，GAMMA_040 的四项 error 三-seed均值均不优于 GAMMA_050，PCC 平均值也更低。
2. seed 与 subject 方向计数表明部分比较仍有局部改善，但其幅度不对称，不能覆盖总体均值的反向变化，也不能外推为所有训练实例或主体受益。
3. 本结果不是首次 held-out 确认，不证明独立泛化或部署收益，也不支持关于 gamma 调制机制的因果结论。
4. checkpoint 由 validation Local RR 选择；test 只评价固定 checkpoint，不进行 test 驱动的 epoch、模型或指标选择。
5. 本 test 协议未预注册新的 quality selector，故不追加 test gate 标签或据此改写 validation 阶段的既有候选判定。

阶段决策为：现有复用 test 证据不支持将 GAMMA_040 替换 GAMMA_050 作为 W0 默认；保留 GAMMA_040 的 validation candidate 身份，冻结本轮固定比较，不基于本次 test 结果继续追加 gamma 系数、调参、换 checkpoint 或回看其他条件。
