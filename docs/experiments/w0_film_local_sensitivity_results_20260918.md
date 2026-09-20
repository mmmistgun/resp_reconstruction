# W0 FiLM gamma/beta 局部系数敏感性结果

协议：`w0-film-local-sensitivity-v1-20260918`  
性质：结果知情、探索性、validation-only、冻结 checkpoint 的局部 inference 扰动

## 1. 范围与统计口径

本分析复用三个冻结的 W0 `crd_tf102_w` validation-selected checkpoint：

| Seed | Selected epoch |
|---|---:|
| 20260811 | 13 |
| 20260812 | 15 |
| 20260813 | 14 |

每个 checkpoint 使用完整 validation，共 2675 个窗口、7 个 `samp_id`。FULL 保持训练时系数 $c_\gamma=0.5,c_\beta=0.5$ 并复用冻结指标；新增四个单路径干预：

- `GAMMA_040`：$c_\gamma$ 从 0.5 调到 0.4，$c_\beta$ 固定为 0.5；
- `GAMMA_060`：$c_\gamma$ 从 0.5 调到 0.6，$c_\beta$ 固定为 0.5；
- `BETA_040`：$c_\beta$ 从 0.5 调到 0.4，$c_\gamma$ 固定为 0.5；
- `BETA_060`：$c_\beta$ 从 0.5 调到 0.6，$c_\gamma$ 固定为 0.5。

所有结果均与同 seed、同 `dataset_row_id` 的 FULL 配对。本文统一报告 error-aligned delta：

$$
\Delta=E_{\mathrm{candidate}}-E_{\mathrm{FULL}},
$$

因此负值表示改善，正值表示恶化；PCC 轴使用 $1-\mathrm{PCC}$。seed 汇总为三个 seed 的算术均值和样本标准差，不计算 p-value 或窗口级置信区间。窗口存在重叠，不能视为独立受试者；主体层仅有 7 个 `samp_id`。

本结果只描述冻结模型附近的局部输出敏感性。局部 inference perturbation 不等价于按新系数重新训练，也不构成结构消融或因果机制证明。

## 2. 矩阵完整性与复现身份

矩阵完整：

- 3 个冻结 W0 checkpoint；
- 每 seed、每条件完整覆盖 2675 个 validation 窗口；
- 四个新干预共 12 次 inference、32100 条新逐窗口指标记录；
- 32100 条记录均完成 FULL 配对；
- `prediction_degeneracy_count=0`；
- 未训练、未修改 checkpoint 或 cache；
- 未访问 test，`research_test_used=false`。

r2 smoke 的同次融合硬锚点、explicit FULL 输出和原生重复 forward 均逐 tensor 完全一致，最大绝对差均为 0。实现锁 SHA-256 为 `7b11a15d39e67dae598464893ab68ee7e7d2f37ed6d7131adef3b808269df4d9`，代码提交为 `8f59862`。

权威产物：

- smoke：`runs/w0_film_local_sensitivity_v1/smoke/seed_20260811/smoke_7b11a15d39e6_20260918T070554Z_08c33e2b4d6f`；
- summary：`runs/w0_film_local_sensitivity_v1/summary/summary_7b11a15d39e6_20260918T092051Z_99a66682abe9`；
- final：`runs/w0_film_local_sensitivity_v1/final/final_7b11a15d39e6_20260918T093454Z_78ad894bc739`。

主证据表为 `local_response.csv`、`seed_summary.csv`、`quality_subject_summary.csv` 和 `matrix_receipt.json`。summary 与 final 的 manifest/freeze receipt 已通过文件身份复核。

## 3. 整体结果

FULL 的三 seed 平均值依次为：Whole RR 0.49930 bpm、Local RR 0.55131 bpm、trajectory 0.15273、global modulation 0.19105、$1-\mathrm{PCC}$ 0.13470。

下表为 paired delta 的 seed mean ± sample SD；括号为改善 seed 数/3。

| 条件 | Whole RR | Local RR | Trajectory | Global modulation | $1-\mathrm{PCC}$ |
|---|---:|---:|---:|---:|---:|
| GAMMA_040 | -0.01240 ± 0.00614 (3/3) | -0.01155 ± 0.00349 (3/3) | -0.002252 ± 0.001631 (3/3) | +0.000619 ± 0.007519 (1/3) | -0.001079 ± 0.000415 (3/3) |
| GAMMA_060 | +0.01992 ± 0.00901 (0/3) | +0.01832 ± 0.00305 (0/3) | +0.003051 ± 0.001815 (0/3) | +0.002042 ± 0.007893 (2/3) | +0.001990 ± 0.000704 (0/3) |
| BETA_040 | -0.006012 ± 0.007066 (2/3) | -0.004019 ± 0.003675 (2/3) | -0.000197 ± 0.000671 (2/3) | -0.000184 ± 0.003363 (1/3) | -0.000346 ± 0.000325 (3/3) |
| BETA_060 | +0.007980 ± 0.004439 (0/3) | +0.003627 ± 0.001531 (0/3) | +0.000340 ± 0.000658 (1/3) | +0.000629 ± 0.003051 (2/3) | +0.000635 ± 0.000373 (0/3) |

整体上，gamma 的局部响应强于 beta。降低 gamma 在 Whole RR、Local RR、trajectory 和 $1-\mathrm{PCC}$ 上均为 3/3 seed 改善；提高 gamma 在相同四轴上均为 3/3 seed 恶化。beta 呈同方向但幅度较小、跨 seed 稳定性较弱。Global modulation 对两条路径均不稳定，不能据此给出统一方向。

## 4. Target modulation strata

冻结分层包含 low 1136、medium 652、high 887 个窗口。下表为中心斜率：

$$
S_{\mathrm{central}}=\frac{M(0.6)-M(0.4)}{0.2},
$$

正值表示增大该路径系数局部趋向恶化。

| Stratum / path | Whole RR | Local RR | Trajectory | Global modulation | $1-\mathrm{PCC}$ |
|---|---:|---:|---:|---:|---:|
| low / gamma | 0.00363 | 0.01145 | 0.00910 | -0.00062 | 0.01327 |
| medium / gamma | 0.12082 | 0.07765 | 0.01337 | 0.01461 | 0.01890 |
| high / gamma | 0.39399 | 0.37874 | 0.05848 | 0.01151 | 0.01540 |
| low / beta | 0.00461 | 0.01024 | 0.00274 | 0.00121 | 0.00370 |
| medium / beta | 0.04033 | -0.00426 | 0.00684 | 0.02041 | 0.00827 |
| high / beta | 0.17543 | 0.10531 | -0.00044 | -0.00428 | 0.00398 |

gamma 的 Whole RR、Local RR 和 trajectory 斜率随 target modulation 层级明显增大，high 层响应最强。high 层中：

- `GAMMA_040` 的 Whole RR、Local RR、trajectory delta 分别为 -0.03214、-0.02839、-0.004954，均为 3/3 seed 改善；
- `GAMMA_060` 对应为 +0.04666、+0.04736、+0.006742，均为 3/3 seed 恶化；
- `BETA_040` 的 Whole RR、Local RR 为 -0.01217、-0.01123，但只有 2/3 seed 改善；
- `BETA_060` 对应为 +0.02292、+0.009828，均为 3/3 seed 恶化。

low/medium 层总体也更支持降低 gamma，而不是与 high 层完全相反。例外主要是 global modulation；beta 在 medium Local RR、high trajectory 和 high global modulation 上的中心斜率接近零或跨 seed 变号。因此，更准确的描述是 gamma 敏感性随 target modulation 增大，而 beta 响应较弱且更依赖指标。

## 5. 独立质量层

独立 `waveform_confidence_level` 包含 high 931 个窗口、5 个主体，medium 1744 个窗口、7 个主体。下表每格依次为 high / medium 层的 paired delta seed mean。

| 条件 | Whole RR | Local RR | Trajectory | Global modulation | $1-\mathrm{PCC}$ |
|---|---:|---:|---:|---:|---:|
| GAMMA_040 | +0.000696 / -0.01939 | -0.002184 / -0.01655 | -0.000754 / -0.003052 | +0.001712 / +0.000036 | -0.001252 / -0.000987 |
| GAMMA_060 | +0.000936 / +0.03006 | +0.000839 / +0.02766 | +0.000979 / +0.004157 | -0.001919 / +0.004156 | +0.001623 / +0.002186 |
| BETA_040 | -0.000861 / -0.008761 | -0.001552 / -0.005336 | -0.000524 / -0.000022 | -0.000087 / -0.000236 | -0.000901 / -0.000050 |
| BETA_060 | +0.000858 / +0.01178 | +0.000669 / +0.005206 | +0.000593 / +0.000206 | -0.000181 / +0.001062 | +0.001094 / +0.000390 |

质量分层仍显示 gamma 降低的主要收益集中于 Local RR、trajectory 和 $1-\mathrm{PCC}$；medium 质量层的 Whole RR 收益也较明显。需要保留两个反向信号：

- high-quality 中 `GAMMA_040` 的 Whole RR 略增 +0.000696，2/3 seed 变差；
- high-quality 中 global modulation 增加 +0.001712，3/3 seed 变差。

这两个变化相对较小，但说明 gamma 降低尚不是五轴一致改善。high target × high confidence 的交叉层仅有 44 个窗口、4 个主体，不宜单独作机制推断。

## 6. Subject 与 seed 稳定性

按每个 `samp_id` 先对三个 seed 的 delta 取均值：

- `GAMMA_040` 在 Whole RR、Local RR、trajectory、$1-\mathrm{PCC}$ 上均为 6/7 主体改善；global modulation 仅 3/7 改善；
- `GAMMA_060` 在 Local RR、trajectory、$1-\mathrm{PCC}$ 上为 7/7 主体恶化，Whole RR 为 5/7 恶化；global modulation 为 4/7 恶化；
- `BETA_040` 在 Local RR、trajectory 上为 6/7 主体改善，$1-\mathrm{PCC}$ 为 5/7 改善，但 Whole RR 和 global modulation 均只有 3/7 改善；
- `BETA_060` 在 Local RR、trajectory 上为 6/7 主体恶化，$1-\mathrm{PCC}$ 为 7/7 恶化，Whole RR 为 5/7 恶化。

gamma 方向在主要 RR、trajectory 和相关性轴上同时具有较强的跨 seed 与跨主体一致性；beta 的方向性较弱。主体窗口数从 57 到 820 不等，上述主体计数是稳定性描述，不是等权主体统计推断。

## 7. 观察、机制假设与待验证问题

### 7.1 观察

1. 冻结 W0 附近，提高 $c_\gamma$ 至 0.6 在多数主轴上稳定恶化，降低至 0.4 则稳定改善。
2. gamma 响应幅度明显大于 beta，且 Whole RR、Local RR、trajectory 的 gamma 敏感性在 high target-modulation 层显著增强。
3. beta 降低可能带来较小收益，但跨 seed、主体和指标的一致性不足以支持其作为首选训练变量。
4. Global modulation 与其他四轴不完全同向；high-quality 层也出现 gamma 降低时的小幅反向代价。

### 7.2 机制假设

结果与“gamma scale 路径是当前冻结 W0 在 high target-modulation 窗口上的主要局部敏感轴”相容。较强 target modulation 可能放大乘法调制对局部频率和轨迹误差的影响，而 beta shift 路径主要提供较弱、指标依赖的修正。

当前实验不能区分以下可能性：

- 训练得到的 gamma 表示在冻结 decoder 中略偏强；
- gamma 边缘程度只是 high target-modulation 窗口的伴随标记；
- 重新训练后网络会通过条件头、local encoder 或 decoder 重新补偿系数变化。

### 7.3 待验证问题

- $c_\gamma=0.4$ 参与完整训练后，当前 inference 层面的 RR、trajectory 和 PCC 收益是否仍存在；
- high-quality global modulation 的小幅反向变化会在重训练后消失、保留还是扩大；
- 收益是否来自少数高误差主体，或在主体等权统计下仍成立；
- target modulation 是否真正调节 gamma 的训练响应；
- 若后续需要不确定性估计，应按 `samp_id` 重采样，并明确只有 7 个主体。

## 8. 后续最小决策

当前证据支持开展一个窄范围、gamma-only 的训练可行性实验，但不支持宣布 $c_\gamma=0.4$ 是更优训练系数，也不支持广泛系数扫描。

建议最小下一步为：

1. 另立训练协议，仅比较训练系数 $c_\gamma=0.4$ 与冻结基线 $c_\gamma=0.5$；$c_\beta$ 均保持 0.5。
2. 使用相同 split、三个训练 seed 和 validation-only 评价，采用独立且不可覆盖的输出 identity。
3. 在运行前冻结五轴验收口径，并显式检查 high-quality global modulation 与 Whole RR 的反向代价；不得根据部分 seed 调整矩阵。
4. 首轮不扩展系数矩阵；其他路径或组合干预根据这一步结果另行立项。
5. 只有训练阶段得到预先定义且跨 seed/主体稳定的 validation 结果后，才另行讨论 test 附件。

该决策的含义是允许验证 gamma 降低这一训练假设，而不是接受 0.4 为最终系数。
