# W0 FiLM gamma 系数训练验证结果

协议：`w0-film-gamma-training-v1-20260918`  
性质：结果知情、validation-only 的训练可行性实验

## 1. 实验范围与冻结矩阵

本实验检验降低 W0 FiLM gamma 系数后，重新训练得到的 validation-selected checkpoint 是否保留冻结模型局部扰动所提示的改善方向。

冻结矩阵如下：

| 条件 | $c_\gamma$ | $c_\beta$ | Seeds | 角色 |
|---|---:|---:|---|---|
| GAMMA_030 | 0.3 | 0.5 | 20260811 / 20260812 / 20260813 | 新训练候选 |
| GAMMA_040 | 0.4 | 0.5 | 20260811 / 20260812 / 20260813 | 新训练候选 |
| GAMMA_050 | 0.5 | 0.5 | 同上 | 复用冻结 W0 基线 |

除 gamma 系数外，训练数据、train/validation split、初始化规则、模型结构、优化器、学习率计划、loss、80-epoch 训练长度及 checkpoint selector 均保持冻结。checkpoint 按完整 validation Local RR 最小值选择。GAMMA_050 三个基线 checkpoint 的 selected epoch 分别为 13、15、14。

本结果只使用 validation，`research_test_used=false`。

## 2. 运行完整性与复现身份

六个新增 formal run 均完成并通过产物审计：

| 条件 | Seed | Selected epoch | 状态 |
|---|---:|---:|---|
| GAMMA_030 | 20260811 | 13 | passed |
| GAMMA_030 | 20260812 | 5 | passed |
| GAMMA_030 | 20260813 | 14 | passed |
| GAMMA_040 | 20260811 | 13 | passed |
| GAMMA_040 | 20260812 | 30 | passed |
| GAMMA_040 | 20260813 | 14 | passed |

每个 run 均完成 80 epochs、6400 optimizer updates，并保存 2675 个 validation windows 的 selected-checkpoint metrics。六个 run 的 history、checkpoint、optimizer state 和全部 primary metrics 均为 finite，prediction degeneracy count 均为 0。

复现身份：

- 训练 commit：`0f9686ccbc64a8b4ee9578b96eae88f5b3bdf4eb`；
- implementation lock SHA-256：`2bf4b72a15111e1caa76b6bed19abbde3242edc451795176c307d160cc49368c`；
- summary：`runs/w0_film_gamma_training_v1/summary/20260919_115215_255816`；
- summary manifest SHA-256：`2df739a0a1a0d987a9c4733107230a5f74662ae78dd92c481d3bca92feae1b7b`。

summary receipt 确认 `matrix_complete=true`、`formal_runs=6`，配对统计共 80250 行，即 $2\times3\times2675\times5$。六个 formal receipt、summary artifact manifest 和 freeze receipt 均已逐文件复核。

## 3. 整体五轴结果

以下 $\Delta$ 均定义为候选减 GAMMA_050 基线。四项 error 指标中，负值表示改善；PCC 中，正值表示改善。均值和 SD 均在三个 paired seeds 上计算。

| 指标 | GAMMA_050 | GAMMA_030 | GAMMA_030 $\Delta$ mean ± SD | 改善 seeds | GAMMA_040 | GAMMA_040 $\Delta$ mean ± SD | 改善 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whole RR absolute error（bpm，↓） | 0.499298 | 0.479549 | -0.019749 ± 0.014748（-3.955%） | 3/3 | 0.487829 | -0.011469 ± 0.029410（-2.297%） | 2/3 |
| Local RR MAE（bpm，↓） | 0.551309 | 0.546693 | -0.004616 ± 0.005391（-0.837%） | 2/3 | 0.542056 | -0.009253 ± 0.002995（-1.678%） | 3/3 |
| Envelope trajectory MAE（↓） | 0.152729 | 0.151108 | -0.001621 ± 0.002610（-1.061%） | 2/3 | 0.151642 | -0.001087 ± 0.000872（-0.712%） | 3/3 |
| Global envelope modulation error（↓） | 0.191051 | 0.189843 | -0.001208 ± 0.001267（-0.632%） | 3/3 | 0.186888 | -0.004163 ± 0.010693（-2.179%） | 1/3 |
| Lag-aware signed PCC（↑） | 0.865300 | 0.861204 | -0.004097 ± 0.009379 | 2/3 | 0.864754 | -0.000547 ± 0.002717 | 2/3 |

GAMMA_030 在四项 error 的三-seed 均值上均改善，其中 Whole RR 改善最大且三个 seed 同方向；但 PCC 平均下降 0.004097。其 PCC 虽有 2/3 seeds 改善，seed 20260812 的下降为 -0.014916，主导了三-seed 均值。

GAMMA_040 同样在四项 error 的三-seed 均值上均改善。Local RR 和 trajectory 为 3/3 seeds 同方向，Whole RR 为 2/3；global modulation 的平均改善主要来自 seed 20260812 的较大下降，只有 1/3 seeds 同方向，因此该轴不具备跨 seed 稳定改善。PCC 平均下降 0.000547，仍在预注册容差内。

## 4. 预注册门槛判定

| 条件 | 实质改善 | Guardrails | 明显失败 | Quality candidate | Tolerance-aware Pareto |
|---|---:|---:|---:|---:|---:|
| GAMMA_030 | 通过 | 未通过 | 否 | 否 | 否 |
| GAMMA_040 | 通过 | 通过 | 否 | 是 | 是 |

GAMMA_030 满足实质改善要求，但 PCC 平均下降 0.004097，超过预注册的最大允许下降 0.003，因此未通过 PCC guardrail。该下降尚未超过 0.005 的明显失败线，故不记为 catastrophic failure。

GAMMA_040 的 Local RR、Whole RR 和 trajectory 均达到实质改善阈值并具备至少 2/3 seeds 的支持；各 error guardrail 均通过，PCC 下降也未超过 0.003。生命周期、finite 和 prediction-degeneracy 条件全部满足。

因此，GAMMA_040 是本冻结矩阵中唯一的 quality candidate，也是唯一的 tolerance-aware Pareto 条件。GAMMA_030 的单轴收益不足以覆盖其 PCC guardrail 失败。

## 5. Target-modulation 分层

三-seed 分层 $\Delta$ 显示，两种较低 gamma 条件的 RR 收益均主要集中在 high target-modulation 层。

GAMMA_040 的 Local RR $\Delta$ 在 high、medium、low 层分别为 -0.02521、-0.00332、-0.00020 bpm；high 层为 3/3 seeds 改善。Whole RR 对应为 -0.03677、+0.00194、+0.00059 bpm，说明整体 Whole RR 收益主要来自 high 层，而 medium/low 层接近零且方向混合。high 层 trajectory 为 -0.00303，global modulation error 为 -0.00884，但后者仅 1/3 seeds 同方向。各层 PCC 均值均轻微下降，约为 -0.00052 至 -0.00060，且每层都有 2/3 seeds 改善，体现出 seed 间幅度不对称。

GAMMA_030 的 high 层 Whole RR、Local RR、trajectory 和 global modulation error 分别为 -0.05501、-0.01814、-0.00583 和 -0.00821；Whole RR、Local RR、global modulation 为 3/3 seeds 改善。其 medium/low 层 Local RR 均值分别为 +0.00248 和 +0.00187 bpm。high 层 PCC 均值下降 -0.00893，仍由 seed 20260812 的较大负向变化主导。

这些结果与 high target-modulation 窗口响应更强的既有观察一致，但分层结果仍是描述性关联，不能据此认定 target modulation 是收益产生的原因。

## 6. 独立 waveform-confidence 质量分层

在独立于 target modulation 的 waveform-confidence 分层中，GAMMA_040 的主要 RR 收益没有局限于单一质量层：

- high / medium 质量层的 Whole RR $\Delta$ 分别为 -0.00523 / -0.01480 bpm；
- Local RR 分别为 -0.00154 / -0.01337 bpm；
- trajectory 分别为 -0.000672 / -0.001309；
- global modulation error 分别为 -0.001450 / -0.005612；
- PCC 分别为 +0.000540 / -0.001127。

其中 high-quality Whole RR 为 3/3 seeds 改善，medium-quality Local RR 为 3/3 seeds 改善。global modulation 两层的负均值均只有 1/3 seeds 同方向，仍主要反映单个 seed 的大幅改善。

GAMMA_030 的收益更偏向 medium-quality 层：该层 Whole RR 和 Local RR 分别下降 0.02858 和 0.00726 bpm，均为 3/3 seeds 改善；high-quality Local RR 均值则轻微恶化 0.00034 bpm。其 high/medium 两层 PCC 均值分别下降 0.00256 和 0.00492，再次表明 PCC 风险并非仅由某一质量层解释。

## 7. Subject 稳定性

validation 共包含 7 个 `samp_id`。下表同时报告三-seed 主体均值改善的主体数，以及至少 2/3 seeds 同方向的主体数。

| 条件 | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| GAMMA_030：主体均值改善 | 5/7 | 4/7 | 3/7 | 2/7 | 2/7 |
| GAMMA_030：≥2/3 seeds 同方向 | 5/7 | 5/7 | 4/7 | 3/7 | 4/7 |
| GAMMA_040：主体均值改善 | 4/7 | 5/7 | 3/7 | 6/7 | 3/7 |
| GAMMA_040：≥2/3 seeds 同方向 | 5/7 | 4/7 | 5/7 | 3/7 | 3/7 |

GAMMA_040 的 Local RR 和 global modulation 改善覆盖了多数主体，但五轴均未达到所有主体一致改善。部分主体的三-seed 均值与多数 seed 方向不同，说明少数大幅变化可以主导均值。主体数仅为 7，且 validation windows 存在重叠，因此这些计数用于描述异质性，不构成窗口独立的统计推断。

## 8. 观察与机制假设

整体结果支持一个有限的工作假设：将 $c_\gamma$ 从 0.5 适度降至 0.4，可能限制 FiLM 乘性调制幅度，在保留条件建模能力的同时降低部分 RR 和轨迹误差。high target-modulation 层较强的响应与这一解释相容。

该解释不是因果结论。系数变化后网络经过完整重新优化，观察差异同时包含表示学习、优化路径和 checkpoint 选择的影响。GAMMA_030 的 PCC 风险及 seed 20260812 的突出偏离表明，更强的 gamma 压缩可能增加形态相关性与 error 指标之间的折衷，或使结果对优化轨迹更敏感。selected epoch 从 5 到 30 的跨度也提示不同条件和 seed 的最优验证时点并不完全稳定。

## 9. 证据边界与阶段决策

本实验支持以下 validation-only 结论：

1. 在冻结训练合同和三个训练 seed 下，GAMMA_040 是唯一通过预注册质量门槛的候选。
2. GAMMA_040 的主要优势是 Local RR 和 trajectory 的跨 seed 一致改善，以及 Whole RR 的平均改善；global modulation 的较大均值收益不具备同等 seed 稳定性。
3. GAMMA_030 的 error 收益不能抵消 PCC guardrail 失败，因此不进入质量候选集合。
4. 结果不能写作独立测试集泛化、部署收益或 gamma 系数的因果效应，也不能外推为所有主体均受益。
5. checkpoint 按 validation Local RR 选择，故 Local RR 以外各轴是同一 selected checkpoint 上的伴随描述性结果。

阶段决策为：冻结 GAMMA_040 作为本矩阵唯一 quality candidate 和 tolerance-aware Pareto 条件；GAMMA_030 不晋级。若后续需要访问独立测试集，应另立匹配协议并取得当次授权，预先固定 GAMMA_040 与 GAMMA_050 的比较、指标和判定规则，不以本次 validation 分层或主体结果继续调整方案。

