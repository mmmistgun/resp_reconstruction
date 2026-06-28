# F 系列 STFT 约束、时频表征与输出空间实验计划（修订版）

本文档用于规划 F 系列实验。核心目标不是把 STFT loss 做低，也不是把输出空间尽快换成 STFT，而是判断：

1. 更细的目标时频监督是否能减少呼吸主频/谐波误拣和长尾 RR 错误。
2. 目标 STFT 监督是否能真正传导到最终 waveform 输出，而不是形成 auxiliary 旁路。
3. STFT / CWT / SST 这类时频表示应作为 loss、内部表征、修正分支还是输出空间。
4. 新增时频约束是否只改善频谱指标，却破坏 peak-band RR、呼吸计数、相位/极性或 easy windows。

修订原则：历史实验结果是强先验，但不是定理。旧 L3、E3、E4、E5 的负结果只排除“当时那组设置”；不能直接推出“STFT loss 无效”“CWT/SST 无效”或“更复杂表示无效”。F 系列应把历史失败拆成可检验假设，而不是把它们固化成结构禁令。

## 0. 背景判断（修订版）

当前记录支持以下判断，但措辞需要保持边界：

- E 系列提示：在当前 `tho_research_v2` soft-z 口径和 `patch_mixer1d` / `native_inject` 路线中，STFT 分支确实被模型使用；收益主要集中在 baseline hard、低 `spectrum_similarity` 或部分 RR 失败窗口。它对 `rr_peak_band_abs_error` 不是单调正贡献，因此不应写成“STFT 已稳定降低所有 peak-band 长尾错误”。
- `stft_only` magnitude 或 magnitude-dominant 路线存在两类结构性风险：一是 magnitude 本身对 polarity/phase 盲，二是模型可能学到“频谱形状合理但 waveform、计数或低频方向不稳”的解。因此它不能直接升级成最终输出空间。
- 历史 L3 低频 STFT magnitude loss 失败，说明“强匹配低频 STFT 图像”在当时 loss 权重、STFT 参数、模型路径和数据口径下不可取；它不等价于所有 STFT 派生 loss 都不可取。F-A 应避免重复 L3 的全图/强权重 log-magnitude 匹配，而应改成小权重、分频带、分布/能量轨迹和 hard-window 分层验证。
- E5 中 gated / cross-attention 出现“频谱小改善，但 peak-band RR 或呼吸计数变差”的模式。这说明第一阶段不宜继续扩大复杂融合结构；但不能据此排除所有条件融合。更合理的处理是要求任何新增时频路径必须有低扰动初始化、paired time-only/substrate 对照和分层诊断。
- CWT/SST 的历史记录应作为风险提示，而不是完全否定。CWT/SST 不适合作为第一阶段最终输出空间，但仍可作为高频心冲击输入、ridge/调制诊断或后置候选。

## 1. 主线 anchor 与固定对照口径

F 系列不回退到早期 `tho_small` 或旧 rawish L0/L1 口径。默认沿用当前 research v2 soft-z 口径。

固定项：

- 数据：`configs/tho_research_v2.yaml` 当前 research v2 soft-z 数据口径。
- 输入主信号：沿用当前主线 BCG soft-z 输入。
- 输出：第一阶段仍输出一维 waveform，推理接口保持不变。
- 数据 seed、验证集、方向门控：沿用当前主线口径，baseline 与候选保持一致。
- seed：pilot 每个候选 2-3 seed；只有 paired delta 一致后再扩到 6+ seed。
- 评价：不以 STFT loss 或 aux loss 选模。主判断仍看 `rr_peak_band_abs_error`、`rr_spec_abs_error`、`breath_count_zero_cross_abs_error`、`relative_envelope_corr`、`band_limited_corr`、`best_lag_corr`。

F0 anchor 建议拆成两个表项，而不是只保留一个：

| label | 模型 | 目的 |
|---|---|---|
| `F0_native_time_only` | `patch_mixer1d + native_time_only + 当前原 loss` | 解释 substrate 本身的强度，避免把 native substrate 收益误算到 STFT loss |
| `F0_native_stft_pre_mixer` | `patch_mixer1d + conv2d fullband 8Hz + native_inject pre_mixer + 当前原 loss` | F 系列主 anchor，复现当前最稳 STFT 主线 |

F-A 的新增 loss 应优先相对 `F0_native_stft_pre_mixer` 做 paired comparison；同时保留 `F0_native_time_only`，用于判断新增目标 STFT loss 是否只是补偿了已有 STFT 输入路径，还是确实改善 waveform 输出。

## 2. F-A：waveform 输出 + STFT 派生 loss

F-A 只改变训练目标。模型仍输出 `y_hat`，再由 `STFT(y_hat)` 与 `STFT(y)` 构造额外损失。

### 2.1 STFT 参数

第一批使用单一主口径，避免把窗长变量和 loss 变量混在一起：

- `fs = 100Hz`
- `win_length = 3000`，即 30s
- `hop_length = 500`，即 5s
- `n_fft = 3000`
- `window = Hann`
- `center = False`，用于 loss 和当前 E 系列 STFT frame 口径对齐
- 频率主监督区域：`0.067-1.2Hz`
- 辅助/诊断区域：`0.033-3Hz`

30s/5s 适合约束呼吸主频和低阶谐波。不要第一批用 5s/10s 短窗作为主频 loss；其频率分辨率太粗，容易鼓励 envelope artifact 或宽带能量相似，而不是稳定主频。

### 2.2 第一批矩阵（建议改成小型析因，而不是纯串联）

原草稿采用 `F-A0 -> F-A1 -> F-A2` 串联设计，归因不够清楚。建议第一批改成最小析因：分别验证 distribution、band-energy、二者组合，再考虑 tiny logmag/sc。

| 实验 | 改动 | 目的 | 阶段判断 |
|---|---|---|---|
| `F0` | 当前主线 anchor，原 loss | 固定同口径对照 | 必跑 |
| `F-A0_dist` | `L_base + L_dist` | 验证呼吸候选频带内的主峰/谐波分布约束是否有益 | 第一优先级之一 |
| `F-A1_bandE` | `L_base + L_bandE` | 验证呼吸相关频带能量轨迹是否改善 hard / low-spectrum 窗口 | 第一优先级之一 |
| `F-A2_dist_bandE` | `L_base + L_dist + L_bandE` | 验证分布约束与能量轨迹是否互补 | 第一批 |
| `F-A3_tiny_logmag_sc` | `F-A2 + tiny L_logmag/sc` | 只补细节，不让 magnitude loss 主导训练 | 第一批可选或第二批 |

如果资源只能跑三个候选，保留 `F-A0_dist`、`F-A1_bandE`、`F-A2_dist_bandE`，暂缓 `F-A3`。

### 2.3 `L_dist`：呼吸候选频带分布约束

目的不是让完整 STFT 图更像，而是减少呼吸主峰/谐波误拣。建议在 `0.05-1.2Hz` 或 `0.067-1.2Hz` 内计算 normalized spectral distribution，而不是只在 `0.05-0.7Hz` 内计算。原因是：如果模型把谐波误判为主峰，谐波通常落在主呼吸带上方；loss 必须“看见”谐波候选区域。

建议形式：

```text
A_y      = log1p(abs(STFT(y)))
A_yhat   = log1p(abs(STFT(y_hat)))
B        = frequency bins in 0.067-1.2Hz
p_y      = softmax(beta * A_y[B, t])
p_yhat   = softmax(beta * A_yhat[B, t])
L_dist   = weighted_JSD(p_y, p_yhat)
```

实现注意：

- `beta` 不宜过大；第一版可试 `beta=2-5`。过大时 loss 近似硬 peak，会把梯度集中到单 bin。
- 对低能量或目标 STFT 主峰不明确的 frame 应降权。可用 target respiratory-band energy、`rr_peak_valid_ratio` 或现有质量指标生成 frame weight。
- 不建议第一版使用 soft-RR centroid 作为主项。centroid 容易把多峰结构压成均值，可能改善 `rr_spec_abs_error` 但不一定改善 peak-band RR。

### 2.4 `L_bandE`：分频带能量轨迹约束

`L_bandE` 用少量重叠频带的 log energy trajectory 做 SmoothL1，目标是约束呼吸能量随时间的慢变化，而不是逐 bin 匹配图像。

候选频带：

| 频带 | 含义 | 初始权重倾向 |
|---|---|---:|
| `0.05-0.3Hz` | 慢呼吸 / 主节律低端 | 0.5 |
| `0.1-0.7Hz` | 主要呼吸频带 | 1.0 |
| `0.3-1.2Hz` | 呼吸谐波 / 波形尖锐度 | 0.3-0.5 |
| `1.2-3.0Hz` | 输出高频污染诊断或弱 suppression | 0.0-0.05 |

可增加一个可选的 harmonic-ratio 诊断项，不建议第一批强监督：

```text
R_harm = log(E_0.3-1.2 / (E_0.1-0.7 + eps))
```

如果 hard windows 的错误确实来自二倍频/谐波误拣，再把 `R_harm` 变成小权重 loss。

### 2.5 `L_logmag/sc`：只能作为小权重辅助

`L_logmag` 和 spectral convergence 可以参考音频 waveform generation 中的 multi-resolution STFT loss，但不能照搬全频强约束。第一版只作为 F-A3 的小权重辅助：

- 只在 `0.067-1.2Hz` 主区域给高权重。
- `1.2-3Hz` 只做弱约束或 suppression。
- 不匹配 `>3Hz` target 细节。
- STFT 派生 loss 总梯度贡献控制在 `L_base` 的 5-10%。若有正信号，再试 10-20%。

建议实现 gradient-norm logging：记录每个 loss 对 `y_hat` 或 decoder 末层参数的梯度范数。这样可避免 “loss 数值很小但梯度很大” 或 “loss 数值大但训练无效”。

### 2.6 进入 F-B 前的固定项与后续候选

| 实验 | 改动 | 进入条件 | 风险 |
|---|---|---|---|
| `F-A4_mrstft` | `F-A0/F-A1/F-A2` 的 multi-resolution STFT 版本 | F-A0/F-A1/F-A2 至少一个有 paired 正信号 | 窗长变量增加归因难度 |
| `F-A5_low_complex` | 极小权重 low-band complex STFT | magnitude/distribution 改善频谱但相位、polarity 或 lag 仍不稳 | phase 对时移和低能量 bin 敏感 |

MR-STFT 第一版只建议 `20s/30s/60s`，其中 30s 为主权重。不要把 5s/10s 作为主频 loss；如果要用短窗，它只能作为 envelope/quality 辅助，不应主导 RR 选择。

进入 F-B/F-C/F-D 前，以下口径应保持冻结，避免把训练目标、结构和数据变量混到同一批结论里：

- anchor 固定：`F0_native_stft_pre_mixer` 是 F-A/F-B/F-D 的主 paired anchor；`F0_native_time_only` 只用于解释 native substrate 本身强度。
- 数据和验证固定：继续使用 `configs/tho_research_v2.yaml`、当前 split、验证窗口、方向门控和同 seed paired comparison；不改变数据口径或验证集。
- 输出定义固定：F-A/F-B/F-D 的第一阶段仍输出一维 waveform；不把低通 waveform、magnitude-only STFT 或 complex STFT 当成同任务主输出。
- 目标 STFT 口径固定：第一阶段保持 `30s/5s`、`n_fft=3000`、Hann、`center=False`、主监督 `0.067-1.2Hz`、诊断 `0.033-3Hz`；多分辨率和 complex 只作为 F-A4/F-A5 的后续候选。
- 评价和护栏固定：不以 STFT/aux loss 选模；继续用第 6 节的 peak-band RR、计数、频谱、相关性、easy/fast-RR 和 hard/low-spectrum 分层规则。
- 当前 F-A 状态固定：截至 8.5，`F-A2b_dist_bandE_w005` 只保留为“有 hard/low-spectrum 正信号、未过 easy/fast 护栏”的候选；`F-A2d/e` 没有解决护栏，不能作为进入 F-B 的通过信号。

因此，若后续启动 F-B，默认应从 `F-B0_aux` 这类诊断性 auxiliary head 开始，而不是直接把 F-A2 系列视为已经通过并升级到 residual 或输出空间路线。

### 2.7 F-A2 护栏诊断优先级

截至 2026-06-27，`F-A2b/F-A2d/F-A2e` 的共同模式是：overall、baseline hard 和 low-spectrum 仍有正信号，但 baseline easy 与 fast-RR 护栏仍系统性变差。下一步不应先改变 STFT 分辨率、扩 seed、进入 F-B/F-C，或继续沿 `waveform_confidence_score/level` 做 scalar sweep；应先用现有 run 做窗口级诊断，定位退化窗口的共同特征。

第一步只生成离线诊断表，不训练新模型：

- `window_delta.csv`：按 `dataset_row_id` 合并 F0 与候选 run，记录 `rr_peak_band_abs_error`、`rr_spec_abs_error`、zero-cross count、`spectrum_similarity`、`best_lag_sec`、预测 RR shift 等 paired delta。
- `bucket_summary.csv`：按 `target_rr_bin`、baseline `spectrum_similarity` tertile、baseline count-error bin、baseline lag bin，以及 `baseline_easy / clean_easy_highspec / dirty_easy_lowspec / baseline_hard / low_spectrum / fast_rr` 做分桶汇总。
- `top_degraded_easy.csv` 与 `top_improved_hard.csv`：分别列出 easy 退化最大和 hard 改善最大的窗口，作为后续 paired 波形图、STFT band-energy 和 harmonic-ratio 复核清单。

初步判断口径：

- 如果 clean easy、高 `spectrum_similarity`、baseline count-error 为 0 的窗口几乎不退化，而退化集中在 peak-band easy 但 count/spectrum 已经不干净的窗口，下一批 gate 应区分“clean easy”和“dirty easy”，不能继续把所有 `rr_peak_band_abs_error <= 0.25` 都当作同一护栏集合。
- 如果严重退化集中在 fast RR 或高 harmonic-ratio 窗口，再考虑把 `1.2-3Hz` 作为弱诊断项或 harmonic-ratio gating；不要直接扩大主监督频带。
- 如果退化主要来自 best-lag、phase 或 polarity，而不是 RR peak-bin 迁移，则 magnitude/band-energy loss 继续调权重意义有限，应转向 lag-aware / complex 小权重，或只对 hard 窗口开放 STFT 派生 loss。
- 如果 hard 改善和 easy 退化来自同一类窗口，说明当前 `L_bandE` 选择性不足，优先重写 sample/window proxy，而不是进入 F-B residual 或输出空间路线。

诊断完成后才设计下一批最小训练矩阵。候选方向优先级：

1. `F-A2f_peak_anchor_w005`：保留 `stft_band_energy_weight=0.005`，增强或重写 `L_dist` / peak distribution 约束，目标是减少 band-energy 推动下的错误 peak-bin 迁移。
2. `F-A2g_hard_proxy_gate_w005`：保留 `F-A2b` 权重，但只对训练期可用的 hard/ambiguous proxy 开启或强化 STFT 派生 loss；proxy 不能来自验证集 paired delta，避免验证泄漏。
3. `F-A2h_lag_or_complex_probe`：只有当诊断显示主要矛盾是 lag/phase/polarity 时才进入，且必须保持小权重和 hard-window 分层验证。

通过条件仍沿用第 6 节护栏：hard / low-spectrum 收益必须保留，baseline easy 不再三 seed 同向变差，fast RR 不系统性恶化，`frac_gt_1/frac_gt_2`、count 和 `rr_spec_abs_error` 不能以明显牺牲换取局部改善。

## 3. F-B：辅助 STFT 表征与低扰动 STFT path

F-B 改变训练结构，但最终推理输出仍优先保持一维 waveform。它不应先理解成“把输出空间从 waveform 换成 STFT”，而应按网络角色分层：

1. auxiliary head：只做目标时频表征监督。
2. consistency：检查目标 STFT 监督是否传导到 waveform 输出。
3. residual correction：允许 STFT path 低扰动参与最终输出。
4. direct decoder：最高风险，最后考虑。

F-B0 可以作为轻量诊断提前做；F-B1/F-B2 应在 F-A 或 F-B0 出现可解释信号后再推进。

### 3.1 STFT 参数口径

第一版保持 30s/5s 单分辨率，减少新自由度：

- `fs = 100Hz`
- `win_length = 3000`
- `hop_length = 500`
- `n_fft = 3000`
- `window = Hann`
- auxiliary log-magnitude head：`center=False`，约 31 帧，便于和 E 系列 STFT frame 口径对齐。
- iSTFT 输出路径：`center=True` 或显式 padding 后 crop，并显式传入 `length=18000`，避免边界和长度不一致。

频带选择不直接照搬输入 STFT 的 8Hz/12Hz 解释。输入 STFT 的高频价值来自 BCG 中心冲击/扰动上下文；target/output STFT 是胸带呼吸波形，应更保守。

| 用途 | 预测或监督频带 | 主权重区域 | 说明 |
|---|---|---|---|
| auxiliary log-magnitude head | `0.033-3Hz` | `0.067-1.2Hz` | 可输出 k=1..90，但 k=1 低权重 |
| consistency | 同 auxiliary head | `0.067-1.2Hz` | 验证 aux 是否传导到 `STFT(y_hat)` |
| residual correction | `0.067-1.2Hz` | `0.067-1.2Hz` | 第一版避免高频/谐波接管计数 |
| wider residual | `0.067-3Hz` | `0.067-1.2Hz` 为主 | 仅在 1.2Hz residual 过度平滑时再试 |

### 3.2 F-B 矩阵

编号规则：`F-B0/F-B1/F-B2` 表示推进阶段；字母后缀表示同一阶段的变体，不表示已经进入下一阶段。

| 实验/变体 | 改动 | 目的 | 是否进入第一批 F-B |
|---|---|---|---|
| `F-B0_aux` | 主 waveform head 不变，增加 aux target-STFT logmag head | 判断 shared latent 是否含有目标时频结构 | 可做，但只作诊断 |
| `F-B1_aux_consistency` | F-B0 + delayed consistency：`STFT(y_hat)` 对齐 aux head | 判断目标 STFT 监督能否传导到 waveform 输出 | 更有判断价值 |
| `F-B2_low_complex_residual` | `y_hat = y0 + alpha * iSTFT(delta_S)` | 让 STFT path 低扰动修正 waveform | F-B1 有信号后再做 |
| `F-B2b_wide_residual` | residual 扩到 `0.067-3Hz` | 处理 1.2Hz residual 过度平滑 | F-B2 成功后再做 |

F-B0 的通过条件不能是 aux loss 下降。aux head 可能只是旁路任务，尤其当前主线已经有 BCG STFT 输入。F-B0 最多证明 latent 有目标时频信息；真正推进信号是 F-B1 是否改善 `STFT(y_hat)` 和任务指标。

F-B1 的 consistency 不应第一步就双向回传。建议：

1. warmup aux head，只开 `L_aux`；
2. 随后开小权重 consistency；
3. 第一版 consistency 用 `detach(Z_aux)` 去拉 `STFT(y_hat)`，避免当前 waveform 反向污染 aux head；
4. 若 F-B1 有正信号，再试不 detach 或双向 consistency。

### 3.3 F-B2 residual correction 的稳定性约束

F-B2 需要避免 residual branch 接管主 decoder。

结构：

```text
y0 = waveform_decoder(features)
delta_S = stft_residual_head(features)
delta_y = iSTFT(delta_S)
y_hat = y0 + alpha * delta_y
```

注意：`alpha=0` 会让 residual head 参数初始梯度也为 0，只剩 `alpha` 自己能动。更稳的做法：

1. 使用小非零 gate，例如 `alpha=1e-3`；
2. 或使用 `alpha = alpha_max * sigmoid(a)`，bias 初始化到约 `1e-3`；
3. 给 residual head 加直接 STFT auxiliary loss，使 final waveform 通路很弱时 head 仍有梯度；
4. 使用 staged training：先训练 F-B1/aux 表征，再解锁 residual。

F-B2 loss：

```text
L = L_base(y_hat, y)
  + beta * L_base(y0, y)
  + lambda_stft * L_stft(y_hat, y)
  + lambda_aux * L_aux(delta_S or STFT head, target_STFT)
  + lambda_res * L_residual
```

其中 `beta` 先取 `0.3-1.0`，确保 `y0` 仍是可用主输出。`L_residual` 可用 `||delta_S||_1` 或 low-band residual energy ratio，防止 STFT path 过度扰动。

保留 ReZero / small-gate 的原因不是形式偏好，而是归因和稳定性要求：

- 初始模型应尽量接近 F0，避免新增 STFT path 一开始破坏 easy windows。
- E5 已出现“频谱小改善，但 peak-band RR / 计数变差”的模式，residual path 必须先作为小扰动进入。
- residual branch 若自由接管，F-B2 就不再回答“STFT 是否修正 hard windows”，而会退化成新 decoder 搜索。

### 3.4 F-B 诊断指标

F-B 不能用 aux/STFT loss 选模。除通用任务指标外，需要额外保存：

- `metrics_y0_base.csv`：residual 前 waveform 指标。
- `metrics_yhat_final.csv`：residual 后 waveform 指标。
- `stft_head_error_by_band.csv`：aux/residual head 对 target STFT 的分频带误差。
- `waveform_stft_error_by_band.csv`：`STFT(y_hat)` 对 target STFT 的分频带误差。
- `residual_energy_by_window.csv`：`||delta_y|| / ||y0||` 或同类低频能量占比。
- `aux_to_wave_consistency_by_band.csv`：aux head 与 `STFT(y_hat)` 的分频带一致性。

判断规则：

- aux head 的 STFT error 下降但 `STFT(y_hat)` 不变：旁路。
- residual energy 很大但主指标没有改善：STFT path 接管。
- `rr_spec_abs_error` 改善但 `breath_count_zero_cross_abs_error` 变差：停止该路线。
- easy windows 或 fast RR 档系统性变差：停止或退回 small-gate。

### 3.5 F-B 特征提取器选择

F-B 的 target/output STFT 图很小：30s/5s 下 180s 窗口只有约 31 或 37 个 frame；第一版主要关注 `0.067-1.2Hz`，最多扩到 3Hz。因此只参考语音时频模型的结构思想，不直接搬 full TF-GridNet、full U-Net、大 attention、Mamba 或大 Conformer。

优先结构偏置：

- 小型 TF-CNN：第一优先级。当前 E 系列中 `conv2d fullband 8Hz` 是最清楚的强 STFT 参照，提示当前尺度下局部 TF 卷积足够强。但这只能作为结构先验，不能写成“卷积必然最优”。
- band-aware TF-CNN：用少量生理频带 head 或 band gate 注入先验，减少逐 bin 自由度。
- TF-Grid-lite：只取 full-band / sub-band / temporal 分工思想，用频率局部卷积、时间卷积和全频带 gate，而不是完整 cross-frame attention 堆叠。
- dual-path TCN/GRU：可作为 residual head 的中等复杂度候选，把频率内结构和时间动态拆开。

后置或不进第一批：

- mini-Conformer：只可放在 F-B0/F-B1 auxiliary token 上试，不直接控制最终 waveform。
- SincNet/LEAF：更适合 raw BCG learned frontend probe，不应和 F-B STFT output head 同批改。
- Conv-TasNet/SEANet：适合另开 waveform 主干或 encoder probe，不适合替代 F-B STFT head。
- Mamba/SSM：更适合 long raw/patch sequence；F-B 的 31/37 frame 太短，长序列优势发挥不出来。
- full U-Net / full TF-GridNet / 大 attention：第一批不建议，参数和结构自由度过高。

建议最多保留三个 F-B head 版本：

| 版本 | 用途 | 结构 | 进入条件 |
|---|---|---|---|
| `Enc1_min_aux` | 最小 auxiliary head | pool 到 31 帧 + 2 层 Conv1d + Linear 到频率 bin | F-B0 默认 |
| `Enc2_band_aware_aux` | band-aware auxiliary head | pool 到 31 帧 + shared temporal Conv1d + 少量 band heads | F-B0/F-B1 有旁路疑虑或需要频带先验 |
| `Enc3_tfgrid_residual` | TF-Grid-lite complex residual | pool/interpolate 到 37 帧 + 频率局部卷积 + 时间卷积 + band global gate | F-B1 证明监督能传导到 waveform 后 |

## 4. F-C：STFT 作为主输出

F-C 是高风险路线。正常情况下只有在 F-A/F-B 出现稳定正信号后才考虑；截至 2026-06-27，F-B
residual/head 扩展路线没有通过 easy/fast 护栏，因此 F-C 不应被解释为“F-B 已经成功后的升级”。
若用户明确要求尝试，只能按受控输出空间 pilot 做，不扩成 full-band 或新 target 口径搜索。

| 实验 | 改动 | 判断 |
|---|---|---|
| `F-C0_low_complex_stft_output` | 从 native token 输出 `0-3Hz` low-band complex STFT real/imag，经 iSTFT 回 waveform | 第一批唯一训练候选；只验证输出空间本身 |
| `F-C1_lowpass_target_baseline` | waveform decoder 对齐同一 low-pass target 口径 | 需要先确认 target/metric 口径；不能静默加入 |
| `F-C2_full_complex_stft` | 输出 full-band complex STFT，经 iSTFT 回 waveform | 维度和过拟合空间过大，暂不建议 |
| `F-C3_mag_only_aux_only` | magnitude-only STFT output | 不建议作为最终输出，只能作为 auxiliary head |

如果进入 F-C，应优先输出 real/imag，而不是 magnitude/phase angle。phase angle 有 wrapping 问题；real/imag 更适合普通实值网络优化。magnitude-only 不能决定符号、相位和时移，不适合作为最终输出路径。

F-C0 若只输出 `0-3Hz` 并把高频置零，最终 waveform 是低通结果。这会改变输出定义，必须明确两件事：

1. waveform loss 是否也对 low-pass target 计算；
2. 任务指标是否仍在完整 target 上解释。

当前 F-C0 准备版不改 target 或指标口径：训练 loss 和 `metrics.csv` 仍对完整 target 解释，只把模型输出空间限制为
`0-3Hz` complex STFT。因此它可以和 F0 做 paired 指标比较，但结论只能回答“低频 STFT 主输出在现有任务口径下是否可用”，
不能回答“低通目标是否更适合呼吸任务”。如果后续改成 low-pass target，必须新增明确 baseline，例如
`F-C1_lowpass_target_baseline`，并在文档中标注旧结论不可直接横比。

full-band 维度约为 `1501 freq bins * 37 frames * 2 real/imag`，远大于 18000 点 waveform，且大量高频 bin
对呼吸任务没有价值，容易增加过拟合空间。因此 F-C 第一批不做 full-band。

## 5. F-D：CWT / SST 与多分辨率高频心冲击分支（不作为第一阶段输出空间）

F-D 的定位是“表示输入 / 诊断分支”，不是 F-C 的输出空间替代。CWT/SST 的多分辨率和 ridge 能力有合理信号基础，但历史 E4/CWT/SST 结果只能作为风险提示，不能直接推出“CWT/SST 无效”。同样，SST 可视化分离度更清晰也不能直接推出“任务指标必然改善”。

本阶段的原则：

- 低频呼吸主分支第一版仍保留当前 `30s/5s` STFT 或 native waveform tokens。它负责呼吸主频、低频形态和谐波。
- CWT 第一优先用途是 high-frequency cardiac branch，用于 `1-8Hz` 心冲击分量的短时调制、心率 ridge、能量/质量提示。
- SST 第一优先用途是 ridge / instantaneous-frequency 特征提取和诊断，不作为第一版 dense map 输入，也不作为网络输出空间。
- CWT/SST 不要求和 STFT 的 image grid 对齐，只要求经独立 encoder 后对齐到 respiratory-scale tokens，例如 31 个 low/native tokens。
- 所有 high-CWT/SST 信息只通过 small-init gate、FiLM 或 gated residual 影响 low/native tokens，避免新增高频分支直接接管 waveform 重建。

对历史 E4/CWT/SST 记录的解释应更保守：

- 如果 E4 前置中低频 CWT 在 hard 窗口的谐波分离不如固定窗 STFT，只能说明当时 wavelet、scale grid、归一化、边界处理和模型路径组合没有收益；不能排除 high-frequency CWT、capped low-CWT 或 modulation-feature CWT。
- 如果 SST 可视化分离度更清晰但训练打平，说明 ridge 清晰度没有自动转化成任务收益；不等价于 SST 不能作为后置 ridge feature 或诊断。
- CWT 是冗余表示，inverse CWT 的数值稳定性、边界 cone-of-influence、尺度归一化和重建一致性都比 iSTFT 更难控；SST 的反变换 / ridge reconstruction 更不适合作为第一版网络输出路径。

### 5.1 CWT 图像分辨率口径

CWT 图像不应被视为“另一张同尺寸 STFT 图”。STFT 使用固定窗长，每个频率 bin 的时间/频率分辨率相同；CWT 的低频 scale 有更长有效窗，高频 scale 有更短有效窗。因此 CWT 可以采样成 `[scale, time]` 矩阵，但每一行的物理感受野不同。

建议在文档和配置里区分两件事：

```text
representation grid:
  high-CWT 自己的 scale/time 采样网格，例如 36 scales × 180 time steps。

fusion token grid:
  high-CWT encoder 输出的 respiratory-scale context tokens，例如 31 tokens。
```

F-D 不要求 CWT 和 STFT image-grid 对齐。正确做法是：high-CWT 保留更密时间采样和 log-scale 频率轴，经独立 encoder 压缩到 `31` 个 respiratory-scale tokens，再与 low-STFT/native tokens 融合。

### 5.2 F-D0：短窗 high-STFT anchor

在进入 high-CWT 前，建议先保留一个短窗 STFT anchor，确认“高频心冲击需要更高时间分辨率”这个变量本身是否有价值。

```text
low branch:
  当前 30s/5s STFT 或 native waveform tokens。

high-STFT branch:
  fs = 100Hz
  win_length = 8s = 800 samples
  hop = 1s 或 2s
  n_fft = 800
  freq = 1-8Hz，可选 0.7-8Hz
  representation = log1p magnitude + per-frequency train z-score
```

解释：`8s` 窗对 `1-8Hz` 心冲击有足够的频率解析能力，并且比 `30s/5s` 更能保留心冲击幅度、质量和心率轨迹的短时变化。`4s/1s` 可作为后续质量/运动分支 probe，不建议第一批作为主高频心冲击分支。

### 5.3 F-D1：high-CWT dense map branch

F-D1 只验证 high-CWT 是否优于短窗 high-STFT，不同时替换低频 STFT，不引入新 decoder。

第一版主参数：

```text
input:
  bcg_rawish_wideband_state_aligned_segment_soft_z

freq range:
  1-8Hz

wavelet:
  complex Morlet
  main = cmor1.5-1.0
  probe = cmor3.0-1.0

scale spacing:
  log-frequency spacing
  voices_per_octave = 12 main
  voices_per_octave = 8 probe

time sampling after aggregation:
  1s main
  0.5s only after 1s shows signal

representation:
  log1p(|CWT|)
  per-scale train z-score
  optional scale-wise energy normalization

expected grid:
  1-8Hz = 3 octaves
  12 voices/octave -> about 36 scales
  1s time grid -> about 180 time steps
```

第二版 probe 只改变一个主变量：

```text
probe A:
  freq = 0.7-8Hz
  voices_per_octave = 8
  wavelet = cmor3.0-1.0
  time_step = 1s

probe B:
  freq = 1-8Hz
  voices_per_octave = 12
  wavelet = cmor1.5-1.0
  time_step = 0.5s
```

不要同批同时改变 wavelet、频率范围、voices/octave、time step 和融合结构。CWT 参数自由度很大，每批最多改变一个主变量。

尺度实现必须显式记录：

```text
frequencies_hz: 目标中心频率列表
scales: 由 wavelet center frequency、fs 和 frequencies_hz 反推得到
wavelet: 具体字符串或参数
padding: reflect padding 长度
crop: 回裁到 180s 窗口
edge_mask / cone-of-influence proxy: 供诊断使用
normalization_stats: 只由 train split 估计
```

### 5.4 F-D2：high-CWT modulation-feature branch

F-D2 用同一套 high-CWT 预计算作为输入来源，但不把完整 CWT 图交给 CNN。它先提取少量可解释的高频调制、质量和 ridge/peak proxy 特征，再用轻量 TCN 对齐到 respiratory-scale tokens。

这一项用于回答：F-D1 的收益如果存在，究竟来自完整 CWT 图的局部时频纹理，还是主要来自心冲击包络、能量调制、质量变化和主 ridge 稳定性。

F-D2 的判断价值：

- 若 F-D2 接近或超过 F-D1，优先保留 modulation-feature 路线，因为它更轻、更可解释，也更容易做 leakage-safe 诊断。
- 若 F-D1 明显优于 F-D2，说明完整 CWT 图里可能有调制特征之外的局部结构，再继续扩 CWT map 参数才有意义。
- 若 F-D1/F-D2 都没有收益，不进入 F-D3 SST ridge 或 F-D4 low-CWT。

### 5.5 F-D3：SST ridge feature extractor，而不是 dense map 第一版输入

SST 不作为第一版 dense neural input。它的用途是从 CWT 中提取更清晰的 cardiac ridge、ridge energy、ridge stability、瞬时频率轨迹和调制特征。

`F-D3_cardiac_sst_ridge` 的建议参数：

```text
base transform:
  沿用 F-D1 high-CWT 主参数
  freq = 1-8Hz
  wavelet = cmor1.5-1.0 或 cmor3.0-1.0
  voices_per_octave = 12

SST frequency bins:
  1-8Hz
  64 bins main
  96 bins optional probe
  log-frequency bins 或较粗 linear bins 均可，但需固定记录

coefficient threshold:
  对低幅值 CWT 系数做 mask
  建议每窗口 p20/p30 magnitude threshold 作为第一版
  或使用固定 epsilon，但必须记录阈值敏感性

saved dense SST map:
  只作为诊断图和 ridge extraction 中间结果
  不直接作为第一版 CNN 输入
```

建议提取的 SST / ridge 特征：

```text
cardiac_ridge_freq(t)
cardiac_ridge_energy(t)
ridge_concentration(t)
ridge_stability(t)
second_ridge_ratio(t)
high_band_entropy(t)
ridge_energy_modulation_power_0.05_0.7Hz
band_energy_modulation_power_0.05_0.7Hz
```

这些特征应先在 `T_high=180` 或 `360` 的高频时间网格上计算，再用 TCN/pooling 压缩到 `31` 个 respiratory-scale tokens。

### 5.6 F-D4：Low-CWT 后置，不进入第一批

低频呼吸不要第一批用 CWT 替换 STFT。最低频 scale 的有效时间支撑过长，容易产生明显边界效应和局部节律变化迟钝。当前 `30s/5s` STFT 正是为了呼吸主频和谐波的频率分辨率而设计，第一批应保留。

若后续做 low-CWT，建议作为 `F-D4_low_capped_cwt`：

```text
freq:
  0.067-1.2Hz
  不建议第一版从 0.05Hz 起步

voices_per_octave:
  12 或 16

wavelet:
  cmor1.5-1.0 或 generalized Morse

time grid:
  聚合到 5s
  最终对齐 31 respiratory tokens

cap / edge handling:
  最大有效时间支撑控制在 45-60s
  或显式记录 cone-of-influence / edge mask

use:
  只和 low-STFT 对照
  不与 high-CWT、SST、新 decoder 同批叠加
```

如果没有 capped-Q、edge mask 或边界诊断，low-CWT 的结果很难解释。

### 5.7 F-D 特征提取网络

#### Enc1：high-CWT small CWT-CNN + TCN（用于 `F-D1_high_cwt`）

默认用于 dense high-CWT map。

```text
input:
  X_high = log1p(|CWT_high|)
  shape = [B, 1, S, T_high]
  S ≈ 36
  T_high ≈ 180

Conv2D block 1:
  Conv2D or depthwise-separable Conv2D
  kernel = (3 scale, 7 time)
  channels = 16 or 32
  activation = GELU

Conv2D block 2:
  kernel = (3 scale, 5 time)
  channels = 32

scale/band pooling:
  S -> 4-6 band groups
  example groups: 1-1.5, 1.5-2.5, 2.5-4, 4-6, 6-8Hz

temporal TCN:
  Conv1D over T_high
  dilation = 1, 2, 4
  channels = 64

downsample:
  T_high 180 -> 31 respiratory tokens
  use average pooling or learned local attention pooling

output:
  C_high [B, C, 31]
```

#### Enc2：modulation-feature TCN（用于 `F-D2_high_cwt_modulation`）

用于判断 dense CWT map 是否必要。先从 high-CWT 提取少量可解释调制特征，再送入 TCN。

```text
features from high-CWT:
  band_energy_1_2Hz(t)
  band_energy_2_4Hz(t)
  band_energy_4_8Hz(t)
  spectral_centroid_1_8Hz(t)
  spectral_entropy_1_8Hz(t)
  peak_or_ridge_freq(t)
  peak_or_ridge_energy(t)
  high_band_quality_ratio(t)

network:
  features [B, K, T_high]
  -> Conv1D / TCN, dilation 1/2/4
  -> downsample to 31 tokens
  -> FiLM or gated residual into low/native tokens
```

如果 `F-D2` 接近或超过 `F-D1`，说明模型主要需要心冲击包络、质量和 ridge 稳定性，而不是完整 CWT 图。

#### Enc3：CWT map + SST ridge feature dual path（用于 `F-D3_cardiac_sst_ridge`）

只在 `F-D1` 或 `F-D2` 有正信号后进入。

```text
CWT map path:
  log1p(|CWT|)
  -> small CWT-CNN + TCN
  -> C_map [B, C, 31]

SST/ridge feature path:
  ridge features [B, K, T_high]
  -> feature TCN
  -> C_ridge [B, C, 31]

high context:
  C_high = C_map + gate_ridge * proj(C_ridge)

fusion to low/native:
  small-init FiLM 或 gated residual
```

Dense SST map 不进入第一版主融合。SST path 只作为 ridge/instantaneous-frequency feature path。

#### Enc4：local attention pooling（用于 `F-D5_local_attention_pooling`，后置）

如果 `0.5s` time grid 或普通 pooling 成为瓶颈，再试 local attention pooling。它只用于高频 tokens 到低频 tokens 的压缩，不作为 full cross-attention 融合主干。

```text
for each low token j:
  query = H_low[j]
  keys/values = high tokens within corresponding local time window
  output = C_high[j]

constraints:
  local window only
  no full 180s global cross-attention
  no bidirectional attention in first probe
```

### 5.8 高频到低频的融合方式

首选 small-init FiLM 或 gated residual，而不是 full cross-attention。

FiLM：

```text
gamma, beta = MLP(C_high)
H_low' = (1 + alpha * gamma) * H_low + alpha * beta
```

Gated residual：

```text
gate = sigmoid(Wg([H_low, C_high]))
delta = Wd(C_high)
H_low' = H_low + alpha * gate * delta
```

`alpha` 不建议严格为 0；第一版可用 `1e-3` 或等价 small-init gate，避免高频路径完全无梯度。高频分支只能作为条件修正，不应直接替代 low/native tokens。

### 5.9 F-D 实验矩阵

编号规则：`F-D0` 到 `F-D5` 是实验阶段/变量编号；`Enc1` 到 `Enc4` 是特征提取器版本，不和 F-D 实验编号一一对应。

| 实验 | 表示 | 特征网络 | 融合 | 目的 |
|---|---|---|---|---|
| `F-D0_high_stft_anchor` | low-STFT/native + short high-STFT `8s/1s, 1-8Hz` | small TF-CNN + TCN | small-init gate/FiLM | 多分辨率 STFT anchor |
| `F-D1_high_cwt` | low-STFT/native + high-CWT `1-8Hz` | Enc1 small CWT-CNN + TCN | small-init gate/FiLM | 验证 CWT 是否优于 high-STFT |
| `F-D2_high_cwt_modulation` | low-STFT/native + high-CWT modulation features | Enc2 feature TCN | small-init gate/FiLM | 验证高频是否主要通过调制/质量起作用 |
| `F-D3_cardiac_sst_ridge` | low-STFT/native + high-CWT + SST ridge features | Enc3 dual path | small-init gate/FiLM | 验证 SST ridge 是否带来额外收益 |
| `F-D4_low_capped_cwt` | low capped-CWT + high-CWT | small CWT-CNN | small-init gate/FiLM | 只有 high-CWT 有益后才替换低频 |
| `F-D5_local_attention_pooling` | high-CWT with 0.5s grid | Enc4 local attention pooling | small-init gate/FiLM | 只在 pooling 成为瓶颈后做 |

第一批若启动 F-D，只跑：

```text
F-D0_high_stft_anchor
F-D1_high_cwt
F-D2_high_cwt_modulation
```

第二批再考虑：

```text
F-D3_cardiac_sst_ridge
F-D4_low_capped_cwt
```

`F-D5` 最后考虑。不要第一批做 full CWT map、dense SST map、attention 和新 decoder 的叠加实验。

### 5.10 F-D 实现和诊断记录

第一阶段建议离线预计算 CWT/SST，不要一开始做 differentiable transform。

```text
precompute:
  high_cwt_logmag.{npy,zarr,hdf5}
  high_cwt_modulation_features.{npy,zarr,hdf5}
  high_sst_ridge_features.{npy,zarr,hdf5}

stats:
  per-scale mean/std from train split only

padding:
  reflect padding before CWT
  crop back to 180s
  save edge_mask or cone-of-influence proxy

diagnostics:
  high_zero
  high_shuffle_time
  high_shuffle_batch
  gate_strength_by_layer
  gate_strength_by_rr_bin
  hard/easy window delta
  low-spectrum window delta
  high-CWT feature ablation
  F-D1 dense CWT vs F-D2 modulation-feature delta
```

F-D 的推进条件：

- `F-D1` 必须相对 `F-D0` 至少不恶化主护栏，且在 hard / low-spectrum 窗口有合理收益，才能继续扩 high-CWT 参数。
- `F-D2` 若接近 `F-D1`，优先保留 modulation-feature 路线，因为它更轻、更可解释。
- `F-D3` 只有在 high-CWT 或 modulation features 已有正信号后才进入；SST ridge 不作为第一批 dense input。
- `F-D4` 只有在 high-CWT 证明有价值后才考虑；low-CWT 不应和 high-CWT 首次验证同批混入。

## 6. 阶段通过规则

候选进入扩 seed 或下一阶段前，至少满足：

- paired `rr_peak_band_abs_error_mean/median` 不劣于 F0，优先改善。
- `frac_gt_1` 或 `frac_gt_2` 至少一个明确下降。
- `rr_spec_abs_error` 不能明显恶化。
- `breath_count_zero_cross_abs_error` 不能明显恶化。
- `relative_envelope_mae/corr` 至少不明显恶化。
- `band_limited_corr`、`best_lag_corr`、`best_lag_sec` 作为关键诊断；不能单独决定胜负，但若明显恶化需要解释。
- 分层上至少在 baseline hard、low `spectrum_similarity`、low confidence 或长尾错误窗口中出现合理收益。
- easy windows 和 fast RR 档不能系统性变差。

停止条件：

- 只改善 STFT 相关 loss 或 `rr_spec_abs_error`，但 peak-band RR / 计数明显恶化。
- 多 seed 方差大到无法区分候选收益和训练随机性。
- F-A0/F-A1/F-A2 没有正信号时，不继续 F-A4/F-A5，也不升级到 F-B2/F-C。
- F-B0 aux loss 下降但 `STFT(y_hat)` 和任务指标无改善，不进入 F-B2。
- F-B2 residual energy ratio 明显升高但主指标无改善，判为接管风险。

## 7. 第一阶段建议执行清单

1. 为 `F-A0_dist`、`F-A1_bandE`、`F-A2_dist_bandE` 补充 loss 实现和单元测试。
2. 为 STFT bin mask、frame count、`center=False/True`、`iSTFT(length=18000)`、梯度路径、detach 策略做单元测试。
3. 增加 runner/manifest，固定 F0 与 F-A 候选的同 seed 配对。
4. 训练 2-3 seed pilot。
5. 汇总整体指标、paired delta、hard/low-spectrum/easy/fast-RR 分层指标。
6. 记录每个新增 loss 的梯度范数，控制 STFT 派生 loss 总梯度占比。
7. 将 manifest、summary、paired delta、分层 delta 路径和主结论回写到本文档，明确 F-A0/F-A1/F-A2 是否通过、停止或需要补跑。
8. 只有 F-A0/F-A1/F-A2 至少一个通过阶段规则，才扩 seed 或设计 F-B1/F-B2。
9. 若单独启动 F-D，需要先预计算并校验 high-STFT / high-CWT / modulation-feature 三类输入，确认 scale-frequency 映射、padding/crop、edge mask 和 31-token 对齐无误。

执行分工：当前 F-A 准备阶段由 agent 补实现、测试、runner、manifest、summary 和 dry-run；正式 2-3 seed 全量 pilot 训练由用户手动运行。agent 不自动启动全量训练，除非用户后续明确要求。
结果记录分工：全量训练完成并产生汇总后，需要把实验范围、关键指标、paired/分层结论、风险解释和下一步决策回写到本文档；若只得到部分 run，也应记录缺口和补跑条件，避免结论只停留在临时 CSV 或对话中。

### 7.1 F-A pilot 结果记录（2026-06-26）

运行范围：

- manifest：`runs/f_a_stft_loss_manifest.csv`
- summary：`runs/f_a_stft_loss_summary.csv`
- paired delta：`runs/f_a_stft_loss_paired_delta.csv`
- strata delta：`runs/f_a_stft_loss_strata_delta.csv`
- split 审计：`runs/audits/split_independence_f_a_stft_loss/summary.csv`
- seed：`20260700`、`20260837`、`20260901`
- 验证窗口：`2675`
- split 审计结果：`overlap_samp_id_count=0`、`overlap_segment_count=0`、`has_samp_id_leakage=False`、`has_segment_leakage=False`

汇总口径说明：首次汇总脚本按 arm 级 `run_root` 取 latest run，会把同一 arm 的 3 个 seed 错配到同一个最新目录；已修正为按 `config.yaml` 中 `training.seed` 匹配 run，并重新生成上述 summary/paired/strata 文件。修正后 `summary` 为 15 行 complete，且 15 个 `run_dir` 唯一。

主结论：

- `F0_native_stft_pre_mixer` 相对 `F0_native_time_only` 仍有明确 anchor 收益：`rr_peak_band_abs_error_mean` 从 `0.5412` 降到 `0.4909`，`frac_gt_1` 从 `0.1363` 降到 `0.1198`，`frac_gt_2` 从 `0.0552` 降到 `0.0465`。这支持继续把 native STFT pre-mixer 作为 F-A 主 anchor。
- `F-A0_dist` 基本中性，不构成继续推进信号：paired `rr_peak_band_abs_error_mean` 平均 `-0.0005`，`frac_gt_1` 反而平均 `+0.0009`。当前 `stft_dist_weight=0.02` 的 tail STFT/base 梯度比约 `0.16%`，约束太弱或目标形式没有打到主问题。
- `F-A1_bandE` 有 hard-window 信号但不稳：overall paired `rr_peak_band_abs_error_mean` 平均 `-0.0075`，baseline hard 分层平均 `-0.1768`；但 seed `20260837` overall 变差 `+0.0285`，baseline easy 分层平均变差 `+0.0236`。
- `F-A2_dist_bandE` 是本批最强但仍未完全通过护栏的候选：overall paired `rr_peak_band_abs_error_mean` 平均 `-0.0164`，median 平均 `-0.0056`，`rr_spec_abs_error_mean` 平均 `-0.0152`，`breath_count_zero_cross_abs_error_mean` 平均 `-0.0188`，`relative_envelope_corr_mean` 平均 `+0.0031`，`band_limited_corr_mean` 平均 `+0.0016`，`frac_gt_1` 平均 `-0.0059`，`frac_gt_2` 平均 `-0.0021`。分层上 baseline hard 平均 `-0.2134`、low-spectrum 平均 `-0.0134`，说明 band-energy 约束确实能打到长尾和低谱相似窗口。
- 主要风险是 easy / fast-RR 护栏：`F-A2_dist_bandE` 的 baseline easy 分层 `rr_peak_band_abs_error_mean` 三个 seed 全部变差，平均 `+0.0218`；fast-RR 分层有 2/3 seed 变差，平均 `+0.0022`。按第 6 节规则，它不能直接升级到扩 seed 或 F-B/F-C。
- 梯度规模没有过强迹象：`F-A2_dist_bandE` tail STFT/base 梯度比平均约 `3.7%`，低于原计划 5-10% 上限；效果主要来自 band-energy，dist 分量梯度约 `0.000004`，远小于 band-energy。

阶段判断：

- 不进入 F-B1/F-B2，也不扩到 6+ seed。
- 保留 `F-A2_dist_bandE` 作为正信号候选，但必须先做护栏修正：降低或调度 `stft_band_energy_weight`，或让 STFT 派生 loss 更偏向 hard/low-spectrum 窗口，目标是保留 hard/low-spectrum 收益，同时消除 easy 与 fast-RR 系统性恶化。
- `F-A0_dist` 不单独继续；若继续使用 distribution 项，应作为 `F-A2` 的小辅助，并重新评估权重或 beta，否则当前梯度贡献太小。

### 7.2 F-A2 guard probe 计划（2026-06-27）

目的：只检验 `F-A2_dist_bandE` 的护栏修正，不改模型结构、数据 split、评价指标或原始 F0/F-A2 结果。核心问题是：降低 `stft_band_energy_weight` 后，能否保留 baseline hard / low-spectrum 收益，同时消除 baseline easy 和 fast-RR 的系统性退化。

runner：`scripts/run_f_a2_guard_probe.py`

manifest：`runs/f_a2_guard_manifest.csv`

矩阵：

| label | 训练方式 | `stft_dist_weight` | `stft_band_energy_weight` | 目的 |
|---|---|---:|---:|---|
| `F0_native_stft_pre_mixer` | 复用 `runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual` | 0.0 | 0.0 | 同 seed anchor |
| `F-A2_dist_bandE` | 复用 `runs/f_a_stft_loss/f_a2_dist_bande/dual` | 0.02 | 0.01 | 原 F-A2 对照 |
| `F-A2b_dist_bandE_w005` | 新训练 | 0.02 | 0.005 | band-energy 减半，观察 easy/fast 护栏 |
| `F-A2c_dist_bandE_w003` | 新训练 | 0.02 | 0.003 | 更弱 band-energy，观察 hard 收益是否仍保留 |

执行范围：每个新候选 3 seed，即 6 个新训练 run；旧 F0 和旧 F-A2 只写入 manifest 供同表汇总，不重新训练。

训练后汇总路径：

- summary：`runs/f_a2_guard_summary.csv`
- paired delta：`runs/f_a2_guard_paired_delta.csv`
- strata delta：`runs/f_a2_guard_strata_delta.csv`

阶段判断：

- 若 `F-A2b` 或 `F-A2c` 在 baseline hard / low-spectrum 仍明显优于 F0，且 baseline easy 不再系统性变差，才考虑扩 seed。
- 若降低权重后 hard 收益消失，说明原 F-A2 的收益主要依赖较强 band-energy 约束；此时不应直接进入 F-B，而应考虑 hard/low-spectrum 加权或阶段性调度。
- 若 easy/fast-RR 仍系统性变差，即使 overall 指标改善，也不进入扩 seed 或 F-B。

### 7.3 F-A2 guard probe 结果记录（2026-06-27）

运行范围：

- manifest：`runs/f_a2_guard_manifest.csv`
- summary：`runs/f_a2_guard_summary.csv`
- paired delta：`runs/f_a2_guard_paired_delta.csv`
- strata delta：`runs/f_a2_guard_strata_delta.csv`
- 新训练 run：`F-A2b_dist_bandE_w005`、`F-A2c_dist_bandE_w003`，各 3 seed
- 复用 run：`F0_native_stft_pre_mixer`、`F-A2_dist_bandE`，各 3 seed
- 汇总状态：`summary` 为 12 行 complete，12 个 `run_dir` 唯一

主结果：

- `F-A2b_dist_bandE_w005` 是本轮最稳的折中候选：相对 F0 的 overall paired `rr_peak_band_abs_error_mean` 三个 seed 全部改善，平均 `-0.0192`；`rr_peak_band_abs_error_median` 平均 `-0.0070`；`breath_count_zero_cross_abs_error_mean` 平均 `-0.0208`；`frac_gt_1` 平均 `-0.0059`；`frac_gt_2` 平均 `-0.0037`。baseline hard 分层平均 `-0.1960`，low-spectrum 平均 `-0.0185`，说明 hard/low-spectrum 收益基本保留。
- `F-A2b` 相对原 `F-A2_dist_bandE` 的 overall peak mean 平均再降 `-0.0028`，且原 F-A2 在 seed `20260837` 上 overall peak 变差的问题消失；但 `rr_spec_abs_error_mean` 比原 F-A2 平均差 `+0.0040`，`band_limited_corr_mean` 平均差 `-0.0006`。
- `F-A2c_dist_bandE_w003` 护栏更轻但收益缩水：overall paired `rr_peak_band_abs_error_mean` 平均 `-0.0104`，baseline hard 平均 `-0.1116`，low-spectrum 平均 `-0.0149`；`relative_envelope_corr_mean` 平均 `-0.0003`。它降低了 easy 退化幅度，但不像 `F-A2b` 那样保留强 hard 收益。
- easy / fast-RR 护栏仍未完全通过：`F-A2b` baseline easy 三个 seed 均变差，平均 `+0.0182`；fast-RR 平均 `+0.0019`，2/3 seed 变差。`F-A2c` baseline easy 也三 seed 均变差，但平均降到 `+0.0090`；fast-RR 平均 `+0.0010`，仍有 2/3 seed 变差。
- 梯度规模符合权重降低预期：原 `F-A2` tail STFT/base 梯度比约 `3.7%`；`F-A2b` 约 `0.8%`；`F-A2c` 约 `0.4%`。dist 分量梯度仍约 `0.000004`，主要作用仍来自 band-energy。

阶段判断：

- 不直接进入 F-B，也不扩 seed。`F-A2b_dist_bandE_w005` 具有比原 F-A2 更稳定的 overall / hard 正信号，但 easy 和 fast-RR 仍系统性变差，不满足第 6 节护栏。
- 不建议继续只扫更小 scalar 权重。`0.003` 已显示继续降权会削弱 hard 收益；下一步应改变监督选择方式，例如 hard/low-spectrum 加权、阶段性 band-energy 调度，或对 easy/fast-RR 档降权/关闭 STFT 派生 loss。
- 若资源允许，下一批更有价值的候选是保留 `stft_band_energy_weight=0.005`，但把 STFT loss 只强化到 baseline hard / low-spectrum 或低 confidence 窗口；成功标准必须要求 baseline easy 不再三 seed 同向变差。

### 7.4 F-A2 confidence guard probe 计划（2026-06-27）

目的：接受 7.3 的判断，不继续做更小 scalar 权重 sweep；保留 `F-A2b_dist_bandE_w005` 的权重强度，但改变 STFT 派生 loss 的样本选择，让约束更偏向低 waveform confidence 窗口。该 probe 不改模型结构、数据 split、评价指标或已有 F0/F-A2/F-A2b 结果。

代码入口：

- 训练编排：`scripts/run_f_a2_confidence_guard_probe.py`
- 汇总入口：继续使用 `scripts/summarize_f_a_stft_loss.py`
- manifest：`runs/f_a2_confidence_guard_manifest.csv`

实现口径：

- `ResearchV2WindowDataset` 将 index 中已有的 `waveform_confidence_score` 和 `waveform_confidence_level` 暴露到 batch `meta`。
- `WeakSyncLoss` 新增默认关闭的 `loss.stft_sample_weight_mode=none`；仅当 runner 显式覆盖时，才从 meta 生成 `sample_weight`，并且只作用到 target STFT 派生 loss，不改变 base waveform/spectrum loss。
- 两个新模式分别是 `waveform_confidence_score_inverse` 与 `waveform_confidence_level_medlow`。前者使用 `1 - waveform_confidence_score`，并保留 `stft_sample_weight_min=0.05` 的高置信兜底；后者对 low/medium 赋权 1.0，对 high 赋权 0.0。

矩阵：

| label | 训练方式 | `stft_dist_weight` | `stft_band_energy_weight` | `stft_sample_weight_mode` | 目的 |
|---|---|---:|---:|---|---|
| `F0_native_stft_pre_mixer` | 复用 `runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual` | 0.0 | 0.0 | `none` | 同 seed anchor |
| `F-A2_dist_bandE` | 复用 `runs/f_a_stft_loss/f_a2_dist_bande/dual` | 0.02 | 0.01 | `none` | 原 F-A2 对照 |
| `F-A2b_dist_bandE_w005` | 复用 `runs/f_a2_guard/f_a2b_dist_bande_w005/dual` | 0.02 | 0.005 | `none` | 当前最强 scalar 护栏候选 |
| `F-A2d_confScoreInv_w005` | 新训练 | 0.02 | 0.005 | `waveform_confidence_score_inverse` | 连续低置信加权，避免完全关闭高置信样本 |
| `F-A2e_confLevelMedLow_w005` | 新训练 | 0.02 | 0.005 | `waveform_confidence_level_medlow` | 对 high confidence 关闭 STFT 派生 loss，强化 low/medium |

执行范围：每个新候选 3 seed，即 6 个新训练 run；旧 F0、原 F-A2、F-A2b 只写入 manifest 供同表汇总，不重新训练。

训练后汇总路径：

- summary：`runs/f_a2_confidence_guard_summary.csv`
- paired delta：`runs/f_a2_confidence_guard_paired_delta.csv`
- strata delta：`runs/f_a2_confidence_guard_strata_delta.csv`

阶段判断：

- 只有当候选保留 baseline hard / low-spectrum 收益，且 baseline easy 不再三 seed 同向变差时，才考虑扩 seed。
- 若 score inverse 仍损伤 easy，但 level medlow 缓解，下一步优先做更细的 confidence/quality 分桶，而不是进入 F-B。
- 若两个候选都削弱 hard 收益或 easy/fast-RR 仍系统性变差，F-A2 系列应先停止，转向 hard-window 诊断或重新定义 target STFT loss 形式。

### 7.5 F-A2 confidence guard probe 结果记录（2026-06-27）

运行范围：

- manifest：`runs/f_a2_confidence_guard_manifest.csv`
- summary：`runs/f_a2_confidence_guard_summary.csv`
- paired delta：`runs/f_a2_confidence_guard_paired_delta.csv`
- strata delta：`runs/f_a2_confidence_guard_strata_delta.csv`
- 新训练 run：`F-A2d_confScoreInv_w005`、`F-A2e_confLevelMedLow_w005`，各 3 seed
- 复用 run：`F0_native_stft_pre_mixer`、`F-A2_dist_bandE`、`F-A2b_dist_bandE_w005`，各 3 seed
- 汇总状态：`summary` 为 15 行 complete，15 个 `run_dir` 唯一，每个 run 的验证窗口数均为 2675
- resolved config 核对：`F-A2d` 三个 seed 均为 `loss.stft_sample_weight_mode=waveform_confidence_score_inverse`、`loss.stft_sample_weight_min=0.05`、`loss.stft_band_energy_weight=0.005`；`F-A2e` 三个 seed 均为 `loss.stft_sample_weight_mode=waveform_confidence_level_medlow`、`loss.stft_sample_weight_min=0.0`、`loss.stft_band_energy_weight=0.005`

主结果：

- `F-A2d_confScoreInv_w005` 相对 F0 的 overall paired `rr_peak_band_abs_error_mean` 平均 `-0.0204`，略好于 `F-A2b` 的 `-0.0192`；`breath_count_zero_cross_abs_error_mean` 平均 `-0.0244`，`frac_gt_1` 平均 `-0.0080`，`frac_gt_2` 平均 `-0.0036`。但 `rr_peak_band_abs_error_median` 平均 `-0.0049`，弱于 `F-A2b` 的 `-0.0070`。
- `F-A2e_confLevelMedLow_w005` 是本轮 overall 数值最强但优势很小的候选：paired `rr_peak_band_abs_error_mean` 平均 `-0.0208`，`breath_count_zero_cross_abs_error_mean` 平均 `-0.0245`，`frac_gt_1` 平均 `-0.0081`，`frac_gt_2` 平均 `-0.0040`。相对 `F-A2b`，overall peak mean 只再降 `-0.0016`，属于小幅边际增益。
- 两个新候选保留了 hard / low-spectrum 正信号：`F-A2d` baseline hard 平均 `-0.2075`、low-spectrum 平均 `-0.0217`；`F-A2e` baseline hard 平均 `-0.2078`、low-spectrum 平均 `-0.0211`。二者都略强于 `F-A2b` 的 baseline hard `-0.1960` 和 low-spectrum `-0.0185`。
- 关键失败点仍是 easy / fast-RR 护栏：`F-A2d` baseline easy 三个 seed 全部变差，平均 `+0.0193`；fast-RR 平均 `+0.0036`，2/3 seed 变差。`F-A2e` baseline easy 也三 seed 全部变差，平均 `+0.0190`；fast-RR 平均 `+0.0045`，2/3 seed 变差。相对 `F-A2b`，easy 没有缓解，fast-RR 反而更差。
- 梯度规模显示 sample-weight 路径确实生效，但没有解决护栏：tail STFT/base 梯度比 `F-A2b` 约 `0.79%`，`F-A2d` 约 `0.86%`，`F-A2e` 约 `0.97%`；新候选并非完全关闭 STFT 分量，而是在低置信样本上提高了有效约束占比。

阶段判断：

- 不进入 F-B，也不扩 seed。confidence gating 的 pilot 结果保留了 F-A2b 的 overall / hard / low-spectrum 正信号，但没有消除 baseline easy 和 fast-RR 的系统性退化，不满足第 6 节护栏。
- 不建议继续沿着 `waveform_confidence_score/level` 做更细 scalar sweep。当前结果说明 waveform confidence 不是 easy/fast-RR 退化的有效代理，继续调 confidence 权重的高后悔成本较高。
- 若继续 F 系列，应先做诊断而不是训练新候选：定位 baseline easy / fast-RR 退化窗口的共同特征，例如 target RR、baseline spectrum similarity、预测相位/lag、band-energy 分布和 confidence 的交叉关系；只有找到可训练期使用且不引入验证泄漏的 proxy 后，再设计下一批 gating 或 loss 形式。
- 当前可保留的候选仍是 `F-A2b_dist_bandE_w005`，但仅作为“有 hard/low-spectrum 正信号、未过 easy/fast 护栏”的候选，不应作为主线升级。

### 7.6 F-A2 护栏窗口诊断记录（2026-06-27）

执行范围：

- 诊断入口：`scripts/diagnose_f_a2_guard_windows.py`
- manifest：`runs/f_a2_confidence_guard_manifest.csv`
- 输出目录：`runs/diagnostics/f_a2_guard_windows/`
- window delta：`runs/diagnostics/f_a2_guard_windows/window_delta.csv`
- bucket summary：`runs/diagnostics/f_a2_guard_windows/bucket_summary.csv`
- easy 退化清单：`runs/diagnostics/f_a2_guard_windows/top_degraded_easy.csv`
- hard 改善清单：`runs/diagnostics/f_a2_guard_windows/top_improved_hard.csv`
- 候选：`F-A2b_dist_bandE_w005`、`F-A2d_confScoreInv_w005`、`F-A2e_confLevelMedLow_w005`
- 输出规模：`window_delta` 为 24075 行，`bucket_summary` 为 96 行，两个 top list 各 50 行。
- paired 图入口：`scripts/plot_paired_f_a2_windows.py`
- top degraded paired 图：`runs/diagnostics/f_a2_guard_windows/paired_plots_top_degraded_fa2b/`，当前绘制 `F-A2b` 8 个窗口。
- dirty easy paired 图：`runs/diagnostics/f_a2_guard_windows/paired_plots_dirty_easy_fa2b/`，当前按 `delta_rr_peak_band_abs_error` 降序绘制 `F-A2b` 8 个窗口。
- STFT ratio 入口：`scripts/summarize_f_a2_stft_ratios.py`
- top degraded STFT ratio：`runs/diagnostics/f_a2_guard_windows/stft_ratios_top_degraded_fa2b.csv`，17 行。
- dirty easy STFT ratio：`runs/diagnostics/f_a2_guard_windows/stft_ratios_dirty_easy_fa2b.csv`，30 行。
- motion 审计：`runs/diagnostics/f_a2_guard_windows/motion_top_degraded_fa2b.csv` 与 `runs/diagnostics/f_a2_guard_windows/motion_dirty_easy_fa2b.csv`。
- top improved hard paired 图：`runs/diagnostics/f_a2_guard_windows/paired_plots_top_improved_hard_fa2b/`，当前绘制 `F-A2b` 8 个窗口。
- top improved hard STFT ratio：`runs/diagnostics/f_a2_guard_windows/stft_ratios_top_improved_hard_fa2b.csv`，14 行。
- top improved hard motion 审计：`runs/diagnostics/f_a2_guard_windows/motion_top_improved_hard_fa2b.csv`，14 行。

初步观察：

- baseline easy 不是同质护栏集合。`F-A2b` 的 baseline easy 平均 delta 为 `+0.0183`，但 `clean_easy_highspec` 只有 `+0.0030`；`dirty_easy_lowspec` 为 `+0.0469`。`F-A2d/e` 也呈现同样模式，说明退化主要集中在 peak-band easy 但 spectrum/count 已经不干净的窗口。
- hard / low-spectrum 正信号仍存在：`F-A2b` baseline hard 平均 `-0.1965`，low-spectrum 平均 `-0.0185`；`F-A2d/e` baseline hard 约 `-0.208`，low-spectrum 约 `-0.021`。
- fast RR 平均退化较小但仍同向：`F-A2b` 为 `+0.0019`，`F-A2d` 为 `+0.0036`，`F-A2e` 为 `+0.0045`。confidence gating 没有解决 fast-RR 护栏。
- 严重 easy 退化数量不大但需要复核：`F-A2b` 中 `easy_regressed_gt_0_5` 为 37 个窗口，`easy_regressed_gt_1` 为 9 个窗口；top degraded easy 清单显示首位窗口属于 `low|count_dirty`，且预测 RR 从约 `12.88 bpm` 下移到约 `8.98 bpm`。
- `F-A2b` top degraded 的 STFT ratio 不支持“统一高频或 harmonic ratio 抬升”解释：`delta_log_harm_over_low` 平均 `-0.0145`、中位 `+0.0104`；`delta_log_high_over_low` 平均 `-0.9344`。更明显的共同变化是 `0.1-0.7Hz` low-band energy 上升，平均 `+0.2199`。
- 按最严重 dirty easy 取 30 个窗口后，STFT ratio 仍接近中性：`delta_log_harm_over_low` 平均 `-0.0054`、中位 `+0.0148`；`delta_log_high_over_low` 平均 `-0.9806`，low-band energy 平均 `+0.2032`。
- motion 审计显示这些窗口多数不是短时极端体动主导：top degraded 中 `task_regular=16/17`，dirty easy 中 `task_regular=28/30`；只有少数进入 `dataset_label_review`，input motion dominated 为 0。
- top improved hard 也显示同类 low-band energy 上升：`delta_log_harm_over_low` 平均 `-0.0793`、中位 `-0.0174`，`delta_log_high_over_low` 平均 `-1.2197`，low-band energy 平均 `+0.2560`，harm band energy 平均 `+0.0300`。
- top improved hard 的 motion 审计更干净：`task_regular=14/14`，input/target motion dominated 均为 0。paired 图示例 `row_8595` 显示 baseline 从目标 `11.64 bpm` 误拣到约 `19.48 bpm`，F-A2b 回到约 `12.50 bpm`，说明 hard 改善常见模式是抑制/修正高 RR 误拣，而不是依赖高频谐波增强。

阶段判断：

- 不因为 baseline easy 均值退化直接否定 F-A2b 的 hard/low-spectrum 信号；但也不能把 F-A2b 升级为通过候选，因为 dirty easy 和 fast-RR 护栏仍未解决。
- 下一步若训练新候选，应先围绕 `clean_easy_highspec` 与 `dirty_easy_lowspec` 的差异设计训练期可用 proxy。禁止用验证集 paired delta 直接生成 gate，以免引入验证泄漏。
- 当前复核更像是 band-energy 约束把部分 dirty easy 窗口推向更强低频能量或更平滑的低频解，而不是普遍推高 harmonic/high band。下一步应继续人工查看 paired 图，确认是低频峰迁移、局部事件主导，还是 target 本身多峰/不稳定。
- hard 改善与 dirty easy 退化共享“low-band energy 上升 / harmonic ratio 不统一上升”这一机制，但方向取决于 baseline 错误形态：hard 窗口中它常把高 RR 误拣拉回目标主峰，dirty easy 中它可能把本已接近正确的窗口推向更平滑或偏低频的解。
- 下一轮训练候选应优先考虑 `F-A2f_peak_anchor_w005`：保留 `bandE=0.005`，但增加 peak-anchor / distribution 约束，避免 band-energy 单独改变低频能量后造成 peak-bin 迁移。若设计 gating，proxy 应关注训练期可用的 peak ambiguity 或 baseline-like high-RR mismatch 风险，而不是继续使用 waveform confidence。

### 7.7 F-A2 peak-anchor probe 准备（2026-06-27）

目的：接受 7.6 的诊断判断，不改变 STFT 分辨率、模型结构、split、评价指标或已有 run 产物；只在 `F-A2b_dist_bandE_w005` 的基础上增加一个直接约束 target STFT 主峰 bin 的小权重分项，检验它是否能减少 band-energy 带来的 peak-bin 迁移。

代码入口：

- loss 实现：`resp_train/losses/stft.py` 与 `resp_train/losses/weak.py`
- 训练编排：`scripts/run_f_a2_peak_anchor_probe.py`
- 汇总入口：继续使用 `scripts/summarize_f_a_stft_loss.py`
- manifest：`runs/f_a2_peak_anchor_manifest.csv`

实现口径：

- `TargetStftLoss` 新增默认计算的 `stft_peak_anchor` 分项：在 `stft_dist_low_hz..stft_dist_high_hz` 频带内取 target 每帧峰值 bin，用 `dist_beta * pred_logmag` 的 `log_softmax` 做 cross-entropy。
- `loss.stft_peak_anchor_sigma_bins=1.0` 时，target 峰值用约 1 个 bin 的高斯软化；按当前 `n_fft=3000, fs=100Hz`，1 bin 约 `0.033Hz` 或 `2 bpm`，避免把边界 bin 抖动当作强错误。
- `WeakSyncLoss` 新增默认关闭的 `loss.stft_peak_anchor_weight=0.0`；只有 runner 显式覆盖时才进入总 loss 和梯度诊断。旧配置、旧 run 与既有 F-A2 结论不需要重算。

矩阵：

| label | 训练方式 | `stft_dist_weight` | `stft_band_energy_weight` | `stft_peak_anchor_weight` | 目的 |
|---|---|---:|---:|---:|---|
| `F0_native_stft_pre_mixer` | 复用 `runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual` | 0.0 | 0.0 | 0.0 | 同 seed anchor |
| `F-A2_dist_bandE` | 复用 `runs/f_a_stft_loss/f_a2_dist_bande/dual` | 0.02 | 0.01 | 0.0 | 原 F-A2 对照 |
| `F-A2b_dist_bandE_w005` | 复用 `runs/f_a2_guard/f_a2b_dist_bande_w005/dual` | 0.02 | 0.005 | 0.0 | 当前最强 scalar 护栏候选 |
| `F-A2d_confScoreInv_w005` | 复用 `runs/f_a2_confidence_guard/f_a2d_confscoreinv_w005/dual` | 0.02 | 0.005 | 0.0 | confidence score 对照 |
| `F-A2e_confLevelMedLow_w005` | 复用 `runs/f_a2_confidence_guard/f_a2e_conflevelmedlow_w005/dual` | 0.02 | 0.005 | 0.0 | confidence level 对照 |
| `F-A2f_peak_anchor_w005` | 新训练 | 0.02 | 0.005 | 0.005 | 保留 hard/low-spectrum 收益，同时抑制错误 peak-bin 迁移 |

执行范围：新候选 3 seed，即 3 个新训练 run；已有 F0、原 F-A2、F-A2b、F-A2d、F-A2e 只写入 manifest 供同表汇总，不重新训练。

推荐命令：

```bash
./.venv/bin/python scripts/run_f_a2_peak_anchor_probe.py \
  --dry-run \
  --manifest runs/f_a2_peak_anchor_manifest.csv

./.venv/bin/python scripts/run_f_a2_peak_anchor_probe.py \
  --device cuda:0 \
  --max-parallel 1 \
  --manifest runs/f_a2_peak_anchor_manifest.csv

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/f_a2_peak_anchor_manifest.csv \
  --output runs/f_a2_peak_anchor_summary.csv \
  --paired-output runs/f_a2_peak_anchor_paired_delta.csv \
  --strata-output runs/f_a2_peak_anchor_strata_delta.csv
```

阶段判断：

- 通过条件不是 STFT loss 下降，而是相对 F0 保留 overall / baseline hard / low-spectrum 正信号，同时 baseline easy 不再三 seed 同向变差，fast-RR 不应比 `F-A2b` 更差。
- 若 `F-A2f` hard 收益保留但 dirty easy 仍退化，说明单纯 peak-anchor 仍不足以区分 dirty easy，需要转向训练期可用的 peak ambiguity / spectrum-quality proxy。
- 若 `F-A2f` 削弱 hard 收益或明显损伤 count / lag / envelope，F-A2 系列应停止继续加 STFT 分项，回到窗口诊断和 proxy 设计。

### 7.8 F-A2 peak-anchor probe 结果记录（2026-06-27）

运行范围：

- manifest：`runs/f_a2_peak_anchor_manifest.csv`
- summary：`runs/f_a2_peak_anchor_summary.csv`
- paired delta：`runs/f_a2_peak_anchor_paired_delta.csv`
- strata delta：`runs/f_a2_peak_anchor_strata_delta.csv`
- 新训练 run：`F-A2f_peak_anchor_w005`，3 seed
- 复用 run：`F0_native_stft_pre_mixer`、`F-A2_dist_bandE`、`F-A2b_dist_bandE_w005`、`F-A2d_confScoreInv_w005`、`F-A2e_confLevelMedLow_w005`，各 3 seed
- 汇总状态：`summary` 为 18 行 complete，18 个 `run_dir` 唯一，每个 run 的验证窗口数均为 2675。
- resolved config 核对：`F-A2f` 三个 seed 均为 `loss.stft_dist_weight=0.02`、`loss.stft_band_energy_weight=0.005`、`loss.stft_peak_anchor_weight=0.005`、`loss.stft_peak_anchor_sigma_bins=1.0`、`loss.stft_sample_weight_mode=none`。

主结果：

- `F-A2f_peak_anchor_w005` 相对 F0 的 overall paired `rr_peak_band_abs_error_mean` 三个 seed 全部改善，平均 `-0.0187`；`rr_peak_band_abs_error_median` 平均 `-0.0046`；`rr_spec_abs_error_mean` 平均 `-0.0142`；`breath_count_zero_cross_abs_error_mean` 平均 `-0.0171`；`relative_envelope_corr_mean` 平均 `+0.0032`；`band_limited_corr_mean` 平均 `+0.0012`；`frac_gt_1` 平均 `-0.0077`，`frac_gt_2` 平均 `-0.0026`。
- 相对 `F-A2b_dist_bandE_w005`，`F-A2f` 没有形成整体升级：overall peak mean 略差 `+0.0005`，peak median 差 `+0.0024`，breath-count delta 差 `+0.0037`，`frac_gt_2` 差 `+0.0011`；但 `rr_spec_abs_error_mean` 好 `-0.0031`，`frac_gt_1` 好 `-0.0019`。
- hard 正信号保留但不是决定性增强：`F-A2f` baseline hard 平均 `-0.2023`，略强于 `F-A2b` 的 `-0.1960`，但弱于 `F-A2d/e` 的约 `-0.208`。
- low-spectrum 收益保留但比 `F-A2b/d/e` 弱：`F-A2f` low-spectrum 平均 `-0.0169`，`F-A2b` 为 `-0.0185`，`F-A2d/e` 约 `-0.021`。
- 关键失败点仍是 easy / fast-RR 护栏：`F-A2f` baseline easy 三个 seed 全部变差，平均 `+0.0173`；fast-RR 平均 `+0.0057`，2/3 seed 变差。easy 只比 `F-A2b` 的 `+0.0182` 略好，不足以改变阶段判断；fast-RR 明显差于 `F-A2b` 的 `+0.0019`。
- 梯度规模显示 peak-anchor 分项确实进入训练，但没有解决选择性问题：tail `stft_total/base_total` 梯度比 `F-A2f` 约 `1.05%`，高于 `F-A2b` 约 `0.79%`；其中 `stft_peak_anchor/base_total` 约 `0.63%`，`stft_band_energy/base_total` 约 `0.75%`，`stft_dist/base_total` 约 `0.17%`。

阶段判断：

- `F-A2f_peak_anchor_w005` 不通过第 6 节护栏，不进入 F-B，也不扩 seed。它证明直接给 target STFT 主峰加小权重 anchor 可以维持 overall / hard 正信号，但不能消除 baseline easy 的系统性退化，且 fast-RR 护栏更差。
- 不建议继续在 `stft_peak_anchor_weight` 或 `sigma_bins` 上做短期 scalar sweep。当前问题仍是 STFT 派生 loss 对窗口类型选择性不足，而不是单个峰值约束过弱。
- 若继续 F 系列，下一步应优先复用现有 run 做 `F-A2f` 与 `F-A2b` 的窗口级 paired 复核，重点看 fast-RR 退化是否来自 target 多峰、低频峰迁移、计数变化或 lag/phase；训练新候选前必须先找到训练期可用、不会泄漏验证 paired delta 的 proxy。

### 7.9 F-A2f vs F-A2b 窗口级复核（2026-06-27）

执行范围：

- 诊断入口：`scripts/diagnose_f_a2_guard_windows.py`
- manifest：`runs/f_a2_peak_anchor_manifest.csv`
- 输出目录：`runs/diagnostics/f_a2_peak_anchor_windows/`
- F0-paired window delta：`runs/diagnostics/f_a2_peak_anchor_windows/window_delta.csv`，16050 行，即 `F-A2b` 与 `F-A2f` 各 3 seed × 2675 窗口。
- 直接比较表：`runs/diagnostics/f_a2_peak_anchor_windows/f2f_vs_f2b_window_delta.csv`，8025 行，以 `F-A2b` 为 baseline、`F-A2f` 为 candidate。
- 直接分层汇总：`runs/diagnostics/f_a2_peak_anchor_windows/f2f_vs_f2b_strata_summary.csv`
- top list：`top_f2f_worse_fast.csv`、`top_f2f_worse_easy.csv`、`top_f2f_worse_dirty_easy.csv`、`top_f2f_better_hard.csv`
- paired 图：`paired_plots_top_f2f_worse_fast/` 与 `paired_plots_top_f2f_better_hard/`，各 8 张。
- STFT ratio：`stft_ratios_top_f2f_worse_fast.csv`、`stft_ratios_top_f2f_worse_easy.csv`、`stft_ratios_top_f2f_better_hard.csv`，各 30 行。

直接比较主结果：

- overall 几乎中性：`F-A2f - F-A2b` 的窗口级 `rr_peak_band_abs_error` 平均 `+0.0005`，中位 `0.0`；8025 个窗口中变差率 `30.9%`，`>0.25 bpm` 变差 148 个，`>0.25 bpm` 改善 141 个。
- baseline easy 不是 `F-A2f` 相对 `F-A2b` 的新增退化来源：baseline easy 平均 `-0.0009`，clean-easy-highspec 平均 `+0.0001`，dirty-easy-lowspec 平均 `-0.0007`。这说明 `F-A2f` 没有解决 easy 护栏，但也没有比 `F-A2b` 进一步恶化 easy。
- fast-RR 是新增问题：fast-RR 平均 `+0.0039`；其中 fast-RR + baseline easy 基本中性 `-0.0000`，但 fast-RR + baseline hard 平均 `+0.0591`，`>0.25 bpm` 变差 16 个、`>0.5 bpm` 变差 9 个。第 7.8 中 fast-RR 护栏变差主要来自 fast-hard 窗口，而不是 fast-easy 窗口。
- baseline hard 总体仍受益：baseline hard 平均 `-0.0065`，但这个平均掩盖了两类相反窗口：慢 RR / 高 RR 误拣窗口被下拉后改善，fast-hard 窗口被继续下拉后恶化。

paired 图与 ratio 观察：

- top fast 退化窗口表现为预测 RR 下移。示例 `row_8448`：target 约 `18.52 bpm`，`F-A2b` 已偏低到 `15.56 bpm`，`F-A2f` 进一步下移到 `12.40 bpm`，peak error 从 `2.95` 增至 `6.12 bpm`。
- top hard 改善窗口也是同一“下拉”机制。示例 `row_8599`：target 约 `11.88 bpm`，`F-A2b` 误拣到 `19.08 bpm`，`F-A2f` 下拉到 `13.13 bpm`，peak error 从 `7.20` 降至 `1.25 bpm`。
- top fast 退化的 STFT ratio 不支持“高频或 harmonic ratio 抬升”解释：30 个窗口中 `delta_energy_low` 平均 `-0.0051`，`delta_energy_harm` 平均 `-0.0025`，`delta_log_harm_over_low` 平均 `-0.0135`，`delta_log_high_over_low` 平均 `-0.0153`。它更像是 peak 选择/主频下移问题，而不是能量整体转向高频。
- top hard 改善同样不是 harmonic ratio 明显变化：30 个窗口中 `delta_log_harm_over_low` 平均 `-0.0003`。收益主要来自把预测主峰从过高 RR 拉向低 RR 目标。

阶段判断：

- `F-A2f` 的 peak-anchor 分项放大了一个双刃机制：对慢 RR 且被 F-A2b 误拣高频的 hard 窗口有益，但对 fast-hard 且 F-A2b 已经偏低的窗口有害。
- 不建议继续扫 `stft_peak_anchor_weight` 或 `sigma_bins`。新增 fast-RR 退化说明当前 framewise target peak-anchor 可能在多峰或快呼吸窗口上强化了低频/次谐波选择。
- 下一步不应再训练 F-A2 系列小改动；应先针对 fast-hard top list 检查 target STFT 峰、全局 peak-band RR、framewise peak 与 subharmonic 的关系。如果确认 target framewise STFT 常把 fast RR 映到低频峰，后续 loss 需要改成 target-RR-aware / anti-subharmonic 的 anchor，或直接停止 F-A2 并转向非 STFT 派生路线。

### 7.10 Fast-hard target STFT peak 复核（2026-06-27）

执行范围：

- 输入：`runs/diagnostics/f_a2_peak_anchor_windows/f2f_vs_f2b_window_delta.csv`
- 检查集合：全部 `fast_rr & baseline_hard` 窗口，共 151 行；并复核 `top_f2f_worse_fast`、`top_f2f_better_hard`、`top_f2f_worse_easy` 各 top 50。
- 输出：`runs/diagnostics/f_a2_peak_anchor_windows/target_stft_peak_fast_hard_all.csv`
- 输出：`runs/diagnostics/f_a2_peak_anchor_windows/target_stft_peak_top_f2f_worse_fast.csv`
- 输出：`runs/diagnostics/f_a2_peak_anchor_windows/target_stft_peak_top_f2f_better_hard.csv`
- 输出：`runs/diagnostics/f_a2_peak_anchor_windows/target_stft_peak_top_f2f_worse_easy.csv`
- 输出：`runs/diagnostics/f_a2_peak_anchor_windows/target_stft_peak_summary.csv`

检查口径：

- 对 target waveform 使用与 F-A loss 相同的 `fs=100Hz, win_length=3000, hop_length=500, n_fft=3000, center=False` 计算 STFT。
- 在 `0.067-1.2Hz` 频带内记录每帧 target STFT 主峰 bpm，并与 `target_rr_peak_band_bpm`、`F-A2b` 预测 RR、`F-A2f` 预测 RR 比较。
- 另记录 target 平均 STFT 功率的 top peaks，用于判断窗口是否存在平均谱低频峰或多峰结构。当前检查仍是诊断，不改变训练或评价指标。

主观察：

- `top_f2f_worse_fast` 的 target STFT 明显多峰且常被更低频峰主导：50 个窗口中，framewise target peak 靠近全局 target RR 的平均比例为 `0.523`，但低于 target RR 超过 `2 bpm` 的平均比例为 `0.377`；平均 STFT top1 与 target RR 的中位距离为 `3.11 bpm`，且 `F-A2f` 比 `F-A2b` 更接近平均 STFT top1 的比例为 `70%`。
- 这不是干净的 half-subharmonic 现象：`top_f2f_worse_fast` 中 framewise peak 靠近 `target_rr/2` 的平均比例只有 `0.054`。更准确的说法是 fast-hard target STFT 里存在低频主导、多峰或 framewise 峰不稳定，而不是简单二分频错误。
- 全部 151 个 fast-hard 窗口中，`F-A2f` 相对 `F-A2b` 变差超过 `0.25 bpm` 的只有 16 个，但这些窗口的 target STFT 特征更集中：framewise peak 靠近 target RR 的平均比例 `0.357`，低于 target RR 超过 `2 bpm` 的平均比例 `0.579`，平均 STFT top1 的中位值约 `10 bpm`，与 target RR 的中位距离约 `10.17 bpm`；`F-A2f` 比 `F-A2b` 更接近这个平均 STFT top1 的比例为 `87.5%`。
- 对照的 `top_f2f_better_hard` 也有低频/多峰结构，但目标 RR 更慢，`F-A2f` 的下拉方向常把高 RR 误拣拉回目标附近：top 50 中 framewise peak 靠近 target RR 的平均比例 `0.577`，target 与平均 STFT top1 的中位距离只有 `0.89 bpm`。
- `top_f2f_worse_easy` 更接近普通 peak-bin 抖动：framewise peak 靠近 target RR 的平均比例 `0.777`，target 与平均 STFT top1 的中位距离 `0.85 bpm`，说明 easy 退化不是本轮新增 fast-hard 问题的主要来源。

阶段判断：

- `F-A2f` 的 fast-hard 退化来自 target-STFT peak-anchor 与全局 RR 目标之间的错位：在一小批 fast-hard 窗口中，target STFT 平均谱或 framewise 峰更偏低，peak-anchor 会把模型往这个低频峰推。
- 这个证据支持第 7.9 的判断：不继续调 `stft_peak_anchor_weight/sigma`。只要 anchor 仍以 framewise target STFT peak 为准，就会在这类 fast-hard 多峰窗口上有相同风险。
- 若仍想保留 STFT 派生监督，下一版必须显式引入 target-RR-aware 或 anti-subharmonic 约束，例如只在 target STFT 峰接近 `target_rr_peak_band_bpm` 时启用 peak-anchor，或对 fast-RR 窗口排除明显低于 target RR 的 framewise peak。否则应暂停 F-A2 小改动，把工作转向非 STFT 派生 loss 或更可靠的训练期 quality/ambiguity proxy。

### 7.11 F-A2g target-RR-aware / anti-subharmonic peak-anchor 准备（2026-06-27）

目的：针对 7.9-7.10 发现的 fast-hard 下拉问题，只修改 `stft_peak_anchor` 的生效帧选择，不改变 STFT 分辨率、模型结构、split、评价指标或既有 run 产物。核心假设是：F-A2f 的 framewise target peak-anchor 在 target STFT 多峰或低频主导帧上会把预测推向偏低 RR，因此下一版应屏蔽明显低于样本级 target RR proxy 的 anchor 帧。

实现口径：

- `TargetStftLoss` 新增 `loss.stft_peak_anchor_mode`，默认 `framewise`，保持 F-A2f 和旧配置行为不变。
- 新模式 `target_rr_guard` 只作用于 `stft_peak_anchor`，不改变 `stft_dist` 和 `stft_band_energy`。
- 训练期不读取验证指标 `target_rr_peak_band_bpm`，也不把 paired delta 做 gate。样本级 RR proxy 来自 target waveform 派生的 STFT framewise peak bpm 分布：默认取 `loss.stft_peak_anchor_target_quantile=0.5`，即 framewise peak 中位数。
- 对每个 STFT 帧，若 target framewise peak bpm 低于该样本 proxy 超过 `loss.stft_peak_anchor_guard_tolerance_bpm=2.0`，则该帧的 peak-anchor loss 权重置 0。这样保留接近样本主 RR proxy 的 anchor，同时减少明显次谐波/低频峰对 fast-hard 窗口的下拉。
- `WeakSyncLoss` 无需新增 meta 输入；新参数通过 `TargetStftLoss.from_config` 读取，旧 run 不需要重算。

矩阵：

| label | 训练方式 | `stft_dist_weight` | `stft_band_energy_weight` | `stft_peak_anchor_weight` | `stft_peak_anchor_mode` | 目的 |
|---|---|---:|---:|---:|---|---|
| `F0_native_stft_pre_mixer` | 复用 `runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual` | 0.0 | 0.0 | 0.0 | `framewise` | 同 seed anchor |
| `F-A2_dist_bandE` | 复用 `runs/f_a_stft_loss/f_a2_dist_bande/dual` | 0.02 | 0.01 | 0.0 | `framewise` | 原 F-A2 对照 |
| `F-A2b_dist_bandE_w005` | 复用 `runs/f_a2_guard/f_a2b_dist_bande_w005/dual` | 0.02 | 0.005 | 0.0 | `framewise` | 当前保留候选 |
| `F-A2d_confScoreInv_w005` | 复用 `runs/f_a2_confidence_guard/f_a2d_confscoreinv_w005/dual` | 0.02 | 0.005 | 0.0 | `framewise` | confidence score 对照 |
| `F-A2e_confLevelMedLow_w005` | 复用 `runs/f_a2_confidence_guard/f_a2e_conflevelmedlow_w005/dual` | 0.02 | 0.005 | 0.0 | `framewise` | confidence level 对照 |
| `F-A2f_peak_anchor_w005` | 复用 `runs/f_a2_peak_anchor/f_a2f_peak_anchor_w005/dual` | 0.02 | 0.005 | 0.005 | `framewise` | 直接 peak-anchor 对照 |
| `F-A2g_targetRRAware_w005` | 新训练 | 0.02 | 0.005 | 0.005 | `target_rr_guard` | 检查 anti-subharmonic guard 能否缓解 fast-hard 下拉 |

执行范围：新候选 3 seed，即 3 个新训练 run；已有 F0、F-A2、F-A2b、F-A2d、F-A2e、F-A2f 只写入 manifest 供同表汇总，不重新训练。

推荐命令：

```bash
./.venv/bin/python scripts/run_f_a2_target_rr_anchor_probe.py \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 4 \
  --manifest runs/f_a2_target_rr_anchor_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/f_a2_target_rr_anchor_manifest.csv \
  --output runs/f_a2_target_rr_anchor_summary.csv \
  --paired-output runs/f_a2_target_rr_anchor_paired_delta.csv \
  --strata-output runs/f_a2_target_rr_anchor_strata_delta.csv
```

阶段判断：

- 首要通过条件：相对 F0 保留 overall、baseline hard、low-spectrum 收益，同时 fast-RR 不再差于 `F-A2b_dist_bandE_w005`，且相对 `F-A2f_peak_anchor_w005` 的 fast-hard 直接比较应明显回收。
- 次要通过条件：baseline easy 不能继续三 seed 同向退化；若 easy 仍系统性变差，即使 fast-hard 回收，也只能作为机制验证，不能进入 F-B。
- 若 `F-A2g` 牺牲 hard/low-spectrum 收益，说明简单 anti-subharmonic guard 把有效 hard 帧也屏蔽掉了，下一步不应继续调 tolerance/quantile 大扫，而应回到窗口级 proxy 诊断。
- 若 `F-A2g` fast-hard 仍明显差于 F-A2b，说明仅用 target STFT framewise peak 分布无法可靠近似训练期 target RR，应暂停 F-A2 小改动，考虑 lag-aware/complex loss 或仅对 hard 窗口启用 STFT loss 的更严格 proxy。

### 7.12 F-A2g target-RR-aware / anti-subharmonic peak-anchor 结果记录（2026-06-27）

运行范围：

- manifest：`runs/f_a2_target_rr_anchor_manifest.csv`
- summary：`runs/f_a2_target_rr_anchor_summary.csv`
- paired delta：`runs/f_a2_target_rr_anchor_paired_delta.csv`
- strata delta：`runs/f_a2_target_rr_anchor_strata_delta.csv`
- 窗口诊断：`runs/diagnostics/f_a2_target_rr_anchor_windows/window_delta.csv`
- 新训练 run：`F-A2g_targetRRAware_w005`，3 seed
- 复用 run：`F0_native_stft_pre_mixer`、`F-A2_dist_bandE`、`F-A2b_dist_bandE_w005`、`F-A2d_confScoreInv_w005`、`F-A2e_confLevelMedLow_w005`、`F-A2f_peak_anchor_w005`，各 3 seed
- 汇总状态：`summary` 为 21 行 complete；每个 run 的验证窗口数均为 2675。
- resolved config 核对：`F-A2g` 三个 seed 均为 `loss.stft_dist_weight=0.02`、`loss.stft_band_energy_weight=0.005`、`loss.stft_peak_anchor_weight=0.005`、`loss.stft_peak_anchor_sigma_bins=1.0`、`loss.stft_peak_anchor_mode=target_rr_guard`、`loss.stft_peak_anchor_guard_tolerance_bpm=2.0`、`loss.stft_peak_anchor_target_quantile=0.5`、`loss.stft_sample_weight_mode=none`。

主结果：

- `F-A2g_targetRRAware_w005` 相对 F0 的 overall paired `rr_peak_band_abs_error_mean` 三 seed 全部改善，平均 `-0.0188`；绝对均值为 `0.4721`。这个量级与 `F-A2f_peak_anchor_w005` 的 `-0.0187` 基本相同，但弱于 `F-A2b_dist_bandE_w005` 的 `-0.0192` 和 `F-A2e_confLevelMedLow_w005` 的 `-0.0208`。
- 相对 `F-A2b_dist_bandE_w005`，`F-A2g` 没有形成 overall 升级：overall peak mean 差 `+0.0004`，peak median 差 `+0.0037`，breath-count delta 差 `+0.0005`；`rr_spec_abs_error_mean` 好 `-0.0015`，`frac_gt_1` 好 `-0.0007`，但不足以抵消 RR peak 护栏问题。
- 相对 `F-A2f_peak_anchor_w005`，`F-A2g` 只略微改善 overall peak mean `-0.0001` 和 breath-count `-0.0032`，但 `rr_spec_abs_error_mean` 差 `+0.0016`，`frac_gt_1` 差 `+0.0011`。因此 guard 没有把 F-A2f 的主要问题转成稳定收益。
- baseline hard 正信号保留：`F-A2g` baseline hard 平均 `-0.2033`，略强于 `F-A2b` 的 `-0.1960`，接近 `F-A2f` 的 `-0.2023`，但弱于 `F-A2d/e` 的约 `-0.208`。
- low-spectrum 收益保留但不增强：`F-A2g` low-spectrum 平均 `-0.0170`，接近 `F-A2f` 的 `-0.0169`，弱于 `F-A2b` 的 `-0.0185` 和 `F-A2d/e` 的约 `-0.021`。
- 关键失败点仍是 easy / fast-RR 护栏：`F-A2g` baseline easy 平均 `+0.0176`，仍三 seed 同向变差；fast-RR 平均 `+0.0062`，差于 `F-A2b` 的 `+0.0019`，也略差于 `F-A2f` 的 `+0.0057`。
- 梯度规模确认 guard 分项进入训练：tail 5 epoch 的 `stft_total/base_total` 梯度比 `F-A2g` 约 `1.10%`，`stft_peak_anchor/base_total` 约 `0.69%`，高于 `F-A2f` 的约 `0.63%`。这说明失败不是 peak-anchor 没有梯度，而是当前 target-RR proxy / guard 选择性仍不够。

直接窗口比较：

- `F-A2g - F-A2b` 的 overall 窗口级 `rr_peak_band_abs_error` 平均 `+0.0004`，中位 `0.0`；8025 个窗口中 `>0.25 bpm` 变差 148 个，`>0.25 bpm` 改善 142 个，整体近中性。
- `F-A2g - F-A2b` 在 fast-RR 分层平均 `+0.0043`，fast-hard 分层平均 `+0.0643`。fast-hard 共 151 个窗口，`>0.25 bpm` 变差 17 个、改善 10 个；seed `20260837` 的 fast-hard 平均差达 `+0.1930`，是主要退化来源。
- `F-A2g - F-A2f` 的 overall 平均 `-0.0001`，但 fast-hard 平均仍为 `+0.0052`，没有形成明确回收。`row_8448` 等代表性 fast-hard 下拉窗口仍存在：target 约 `18.52 bpm`，`F-A2b` 预测约 `15.56 bpm`，`F-A2f` 约 `12.40 bpm`，`F-A2g` 约 `12.45 bpm`。
- baseline easy 的直接比较基本中性：`F-A2g - F-A2b` 为 `-0.0006`，`F-A2g - F-A2f` 为 `+0.0002`。这说明 F-A2g 没有新增 easy 退化，但也没有解决相对 F0 的 easy 系统性变差。

阶段判断：

- `F-A2g_targetRRAware_w005` 不通过第 6 节护栏，不进入 F-B，也不扩 seed。它保留了 F-A2b/F-A2f 的 hard 正信号，但没有解决 easy 退化，并且 fast-RR / fast-hard 仍差于 F-A2b。
- 本次失败说明“用 target STFT framewise peak 中位数作为训练期 target-RR proxy，再屏蔽低于 proxy 2 bpm 的帧”不足以防止多峰 fast-hard 窗口下拉。尤其是部分窗口中 proxy 或保留帧仍可能偏向低频峰，guard 没有真正对齐评估口径里的 target RR。
- 不建议继续扫 `stft_peak_anchor_guard_tolerance_bpm` 或 `stft_peak_anchor_target_quantile`。当前问题不是一个 scalar threshold 没调好，而是训练期可用 proxy 仍无法可靠区分“有效低频 hard 修正”和“fast-hard 次谐波下拉”。
- F-A2 小改动阶段应收口：保留 `F-A2b_dist_bandE_w005` 作为“有 hard/low-spectrum 正信号但未过护栏”的参考候选；若继续 F 系列，应转向更可靠的 hard/ambiguity proxy、lag-aware/complex 小权重，或只对经过更严格训练期 proxy 的 hard 窗口启用 STFT loss，而不是继续在同一 framewise magnitude-anchor 上叠加小改动。

### 7.13 F-B0/F-B1 auxiliary probe 准备（2026-06-27）

目的：接受 7.12 的收口判断，不把 F-A2 系列升级到 residual 或输出空间；只准备 F-B0/F-B1
诊断入口，检查 target-STFT auxiliary supervision 是否能进入 shared latent 并传导到最终
waveform。

代码入口：

- 模型：`TimeStftDual1D` 可选 `model.fb_aux_head=enc1_min_aux`，在共享 patch token 上输出
  `aux_target_stft_logmag`。
- loss：`WeakSyncLoss` 支持模型返回 `{"waveform": ..., "aux_target_stft_logmag": ...}`；
  新增 `loss.fb_aux_weight` 与 `loss.fb_consistency_weight`，默认均为 0。
- runner：`scripts/run_f_b_aux_probe.py`
- summary：复用 `scripts/summarize_f_a_stft_loss.py`，该脚本现在同时识别 `F-A*` 与 `F-B*`。
- manifest：`runs/f_b_aux_manifest.csv`

第一批矩阵：

| label | 训练方式 | `fb_aux_weight` | `fb_consistency_weight` | `fb_consistency_start_epoch` | 目的 |
|---|---|---:|---:|---:|---|
| `F0_native_stft_pre_mixer` | 复用 `runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual` | 0.0 | 0.0 | 1 | 同 seed anchor |
| `F-B0_aux_enc1` | 新训练 | 0.01 | 0.0 | 1 | 判断 shared latent 是否包含目标 STFT 结构 |
| `F-B1_aux_consistency_detach` | 新训练 | 0.01 | 0.005 | 7 | warmup 后用 `detach(aux)` 拉 `STFT(y_hat)` |

执行范围：每个 F-B 新候选 3 seed，即 6 个新训练 run；F0 anchor 只写入 manifest 供同表汇总。
本阶段不训练 `F-B2_low_complex_residual`，也不进入 F-C。

推荐命令：

```bash
./.venv/bin/python scripts/run_f_b_aux_probe.py \
  --dry-run \
  --manifest runs/f_b_aux_manifest.csv

./.venv/bin/python scripts/run_f_b_aux_probe.py \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 2 \
  --manifest runs/f_b_aux_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/f_b_aux_manifest.csv \
  --output runs/f_b_aux_summary.csv \
  --paired-output runs/f_b_aux_paired_delta.csv \
  --strata-output runs/f_b_aux_strata_delta.csv
```

阶段判断仍沿用第 6 节：aux loss 下降本身不算通过；只有 `F-B1` 同时改善或至少不损伤
`STFT(y_hat)`、peak-band RR、计数、easy 与 fast-RR 护栏，才考虑后续 residual probe。

### 7.14 F-B0/F-B1 auxiliary probe 结果（2026-06-27）

运行范围：

- manifest：`runs/f_b_aux_manifest.csv`
- summary：`runs/f_b_aux_summary.csv`
- paired delta：`runs/f_b_aux_paired_delta.csv`
- strata delta：`runs/f_b_aux_strata_delta.csv`
- 完成情况：`F0_native_stft_pre_mixer` 3 seed 复用完成；`F-B0_aux_enc1` 与
  `F-B1_aux_consistency_detach` 各 3 seed 训练完成。

相对 `F0_native_stft_pre_mixer` 的 paired 结果：

| label | overall `rr_peak_band_abs_error_mean` delta | 改善 seed | `frac_gt_1` delta | `frac_gt_2` delta | 判断 |
|---|---:|---:|---:|---:|---|
| `F-B0_aux_enc1` | `+0.0265` | 2/3 | `+0.0070` | `+0.0077` | 不通过；seed `20260901` 明显退化 |
| `F-B1_aux_consistency_detach` | `+0.0438` | 1/3 | `+0.0130` | `+0.0086` | 不通过；consistency 没有改善 waveform RR |

关键分层：

- baseline easy 是硬护栏失败点：`F-B0` 三 seed 全部变差，平均 `+0.0298`；
  `F-B1` 三 seed 全部变差，平均 `+0.0370`。
- baseline hard 没有稳定收益：`F-B0` 平均 `+0.0197`，`F-B1` 平均 `+0.0727`；
  二者都被 seed `20260901` 的大幅退化拉坏。
- fast-RR 有局部正信号：`F-B0` 平均 `-0.0247`，`F-B1` 平均 `-0.0207`，均为 2/3 seed 改善；
  但该收益不足以抵消 overall、baseline easy 与 tail fraction 的退化。
- 频谱/相关性指标有旁路迹象：`F-B1` 的 `rr_spec_abs_error_mean` 与
  `band_limited_corr_mean` 多数改善，但没有传导到 peak-band RR 与计数主指标。
- 训练日志确认 auxiliary/consistency 分量进入训练：`F-B0` 末轮 `val_fb_aux` 约 `0.160-0.171`，
  consistency 为 0；`F-B1` 末轮 `val_fb_consistency` 约 `0.346-0.371`。

阶段判断：

- `F-B0_aux_enc1` 只能说明 shared latent/aux head 能学习目标 STFT 表征；aux loss 下降不构成通过。
- `F-B1_aux_consistency_detach` 没能把目标 STFT 监督稳定传导到最终 waveform，且 easy 护栏系统性失败。
- 本轮不进入 `F-B2_low_complex_residual`，也不进入 F-C；继续做 residual/output-space 会把问题从“监督能否传导”扩大成新 decoder 搜索。
- 若后续重启 F-B，应先解决 aux head 对 shared latent 的扰动与 seed `20260901` 的退化窗口，而不是扩 seed 或加大 consistency 权重。

### 7.15 F-B2 low-complex residual 受控推进（2026-06-27）

说明：7.14 的阶段判断仍成立，`F-B1` 没有通过护栏。此处是用户明确要求继续 F-B/F-B2 后的受控
probe，目的不是宣称 F-B1 已可升级，而是用最小 residual 结构验证“低频复数残差是否能在不接管主
decoder 的情况下修正 waveform”。

实现范围：

- 模型：`TimeStftDual1D` 新增 `model.fb_residual_head=low_complex_residual`。
- residual 形式：从 shared token 预测 `0.067-1.2Hz` 低频复数 STFT delta，`center=true` iSTFT
  到 waveform residual，再以 `model.fb_residual_scale=0.03` 加到主 waveform。
- 初始化：residual head 最后一层零初始化，初始 `waveform == base_waveform`，降低接管风险。
- 输出诊断：模型 dict 返回 `waveform`、`base_waveform`、`residual_waveform`；loss/metrics 仍只使用
  `waveform`。
- runner：`scripts/run_f_b2_residual_probe.py`
- manifest：`runs/f_b2_residual_manifest.csv`

第一批矩阵：

| label | 训练方式 | 目的 |
|---|---|---|
| `F0_native_stft_pre_mixer` | 复用 `runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual` | 同 seed anchor |
| `F-B1_aux_consistency_detach` | 复用 `runs/f_b_aux/f_b1_aux_consistency_detach/dual` | auxiliary + consistency 对照 |
| `F-B2_low_complex_residual` | 新训练 3 seed | 验证低频复数 residual 是否改善 waveform 主指标 |

命令：

```bash
./.venv/bin/python scripts/run_f_b2_residual_probe.py \
  --dry-run \
  --manifest runs/f_b2_residual_manifest.csv

./.venv/bin/python scripts/run_f_b2_residual_probe.py \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 2 \
  --manifest runs/f_b2_residual_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/f_b2_residual_manifest.csv \
  --output runs/f_b2_residual_summary.csv \
  --paired-output runs/f_b2_residual_paired_delta.csv \
  --strata-output runs/f_b2_residual_strata_delta.csv
```

通过条件：

- `F-B2_low_complex_residual` 相对 `F0_native_stft_pre_mixer` 的 overall peak-band RR 不能只靠单 seed
  偶然改善，且 `frac_gt_1/frac_gt_2` 不应变差。
- baseline easy 不允许 3/3 seed 同向退化；若重复 7.14 的 easy 失败，F-B2 立即停止。
- fast-RR 的局部正信号必须同时带来 hard/overall 收益，不能以 easy 或 tail fraction 为代价。

### 7.16 F-B2 low-complex residual 结果（2026-06-27）

运行范围：

- manifest：`runs/f_b2_residual_manifest.csv`
- summary：`runs/f_b2_residual_summary.csv`
- paired delta：`runs/f_b2_residual_paired_delta.csv`
- strata delta：`runs/f_b2_residual_strata_delta.csv`
- 完成情况：`F0_native_stft_pre_mixer` 3 seed 复用完成；`F-B1_aux_consistency_detach` 3 seed 复用完成；
  `F-B2_low_complex_residual` 3 seed 训练完成。

相对 `F0_native_stft_pre_mixer` 的 paired 结果：

| label | overall `rr_peak_band_abs_error_mean` delta | 改善 seed | `frac_gt_1` delta | `frac_gt_2` delta | 判断 |
|---|---:|---:|---:|---:|---|
| `F-B1_aux_consistency_detach` | `+0.0438` | 1/3 | `+0.0130` | `+0.0086` | 不通过 |
| `F-B2_low_complex_residual` | `+0.0072` | 2/3 | `+0.0016` | `+0.0034` | 不通过；明显优于 F-B1，但仍未过护栏 |

关键分层：

- baseline hard 出现稳定正信号：`F-B2` 的 `rr_peak_band_abs_error_mean` 三 seed 全部改善，平均
  `-0.1108`；这说明低频复数 residual 确实能修正一部分 hard 窗口。
- fast-RR 也有稳定正信号：三 seed 全部改善，平均 `-0.0152`；`rr_spec_abs_error_mean` 三 seed 全部改善，
  平均 `-0.0387`。
- baseline easy 仍是失败点：三 seed 全部变差，平均 `+0.0333`；这与 7.14 中 F-B1 的 easy 系统性退化
  同向，说明 residual 没有解决护栏问题。
- tail fraction 未通过：overall `frac_gt_2` 三 seed 全部变差，平均 `+0.0034`；`frac_gt_1` 虽 2/3 seed
  改善，但 seed `20260901` 仍变差 `+0.0157`。
- 相对 F-B1，F-B2 有明确回收：overall peak-band RR 平均 `-0.0366`，`frac_gt_1` 三 seed 全部改善，
  breath-count 三 seed 全部改善。但该回收不足以超过 F0 anchor。

阶段判断：

- `F-B2_low_complex_residual` 不通过第 7.15 的通过条件，不能扩 seed，也不能进入 F-C。
- 这轮结果给出一个有用机制信号：low-complex residual 对 baseline hard / fast-RR 有帮助；问题在于它同时损伤
  easy 与 tail fraction。
- 后续若继续 F-B2，只应做“选择性启用 residual”的诊断，例如 hard/ambiguity gate、residual energy cap、
  或只在训练期 proxy 认定为 hard/fast 的窗口上启用 residual；不建议直接加大 residual scale、扩大频带或改成更强 decoder。

### 7.17 F-B 3.5 feature extractor probe 准备（2026-06-27）

目的：沿第 3.5 节尝试特征提取器选择，但不直接进入 `Enc3_tfgrid_residual`。7.16 已说明 residual 对
hard/fast 有正信号但损伤 easy；因此本轮先只替换 auxiliary extractor，检查 band-aware 先验是否能减少
Enc1 的旁路/误伤模式。

实现范围：

- 模型：`TimeStftDual1D` 新增 `model.fb_aux_head=enc2_band_aware_aux`。
- 结构：shared temporal Conv1d 后按 `0.033-0.067 / 0.067-0.3 / 0.3-0.7 / 0.7-1.2 / 1.2-3.0Hz`
  等频带分 head，最后按频率顺序拼回 `aux_target_stft_logmag`。
- 输出契约：仍返回 `{"waveform": ..., "aux_target_stft_logmag": ...}`，loss 与评估口径不变。
- runner：`scripts/run_f_b_feature_extractor_probe.py`
- manifest：`runs/f_b_feature_extractor_manifest.csv`

矩阵：

| label | 训练方式 | 目的 |
|---|---|---|
| `F0_native_stft_pre_mixer` | 复用 `runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual` | 同 seed anchor |
| `F-B1_aux_consistency_detach` | 复用 `runs/f_b_aux/f_b1_aux_consistency_detach/dual` | Enc1 baseline |
| `F-B1b_aux_enc2_band_aware_consistency` | 新训练 3 seed | 验证 band-aware auxiliary extractor 是否改善传导与 easy 护栏 |

命令：

```bash
./.venv/bin/python scripts/run_f_b_feature_extractor_probe.py \
  --dry-run \
  --manifest runs/f_b_feature_extractor_manifest.csv

./.venv/bin/python scripts/run_f_b_feature_extractor_probe.py \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 2 \
  --manifest runs/f_b_feature_extractor_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/f_b_feature_extractor_manifest.csv \
  --output runs/f_b_feature_extractor_summary.csv \
  --paired-output runs/f_b_feature_extractor_paired_delta.csv \
  --strata-output runs/f_b_feature_extractor_strata_delta.csv
```

判断：

- 若 Enc2 相对 Enc1 只改善 aux/STFT 分量但 `STFT(y_hat)`、peak-band RR、计数无改善，仍判为旁路。
- 若 baseline easy 继续三 seed 同向退化，不进入 Enc3 或 residual 复杂化。
- 只有 Enc2 同时缓解 easy、保持 hard/fast 信号，才考虑下一步 residual gate 或 TF-Grid-lite。

### 7.18 F-B 3.5 feature extractor probe 结果（2026-06-27）

运行范围：

- manifest：`runs/f_b_feature_extractor_manifest.csv`
- summary：`runs/f_b_feature_extractor_summary.csv`
- paired delta：`runs/f_b_feature_extractor_paired_delta.csv`
- strata delta：`runs/f_b_feature_extractor_strata_delta.csv`
- 完成情况：`F0_native_stft_pre_mixer` 3 seed 复用完成；`F-B1_aux_consistency_detach` 3 seed 复用完成；
  `F-B1b_aux_enc2_band_aware_consistency` 3 seed 训练完成。

相对 `F0_native_stft_pre_mixer` 的 paired 结果：

| label | overall `rr_peak_band_abs_error_mean` delta | 改善 seed | `frac_gt_1` delta | `frac_gt_2` delta | 判断 |
|---|---:|---:|---:|---:|---|
| `F-B1_aux_consistency_detach` | `+0.0438` | 1/3 | `+0.0130` | `+0.0086` | 不通过 |
| `F-B1b_aux_enc2_band_aware_consistency` | `+0.0173` | 1/3 | `+0.0076` | `+0.0042` | 不通过；较 Enc1 有回收但仍未过护栏 |

关键分层：

- baseline hard 出现正信号：`F-B1b` 的 `rr_peak_band_abs_error_mean` 三 seed 全部改善，平均 `-0.0536`；
  这比 Enc1 的 hard 分层更稳定。
- breath-count 有回收：overall `breath_count_zero_cross_abs_error_mean` 平均 `-0.0169`，2/3 seed 改善；
  相对 Enc1，breath-count 三 seed 全部改善，平均 `-0.0516`。
- baseline easy 仍失败：`rr_peak_band_abs_error_mean` 三 seed 全部变差，平均 `+0.0293`；虽然比 Enc1 的
  `+0.0370` 略轻，但仍触发 7.17 的停止条件。
- fast-RR 信号变弱：`F-B1b` fast-RR 平均 `-0.0050`，2/3 seed 改善，但弱于 Enc1 的 `-0.0207`；且
  fast-RR `rr_spec_abs_error_mean` 平均 `+0.0038`，没有形成频谱收益。
- 相对 Enc1，Enc2 的 overall peak-band RR 平均 `-0.0265`，但只有 1/3 seed 改善；均值主要来自 seed
  `20260901` 的大幅回收，不能视为稳定通过。

阶段判断：

- `F-B1b_aux_enc2_band_aware_consistency` 不通过第 7.17 的条件，不进入 `Enc3_tfgrid_residual` 或更强
  feature extractor。
- Enc2 的有用信号是 hard 分层和 breath-count 回收，说明 band-aware 先验不是完全无效；但它仍没有解决 easy
  误伤，也没有把 fast-RR/频谱收益稳定放大。
- 下一步若继续 F 系列，优先做 hard/easy 选择性 gating 或 residual energy cap，而不是继续堆更强 auxiliary
  extractor。

### 7.19 F-B Enc3 residual 与 energy cap 准备（2026-06-27）

说明：7.18 的阶段判断仍成立，`F-B1b_aux_enc2_band_aware_consistency` 没有通过 easy 护栏；本节是用户明确要求
“试 Enc3，再来 hard/easy 选择性 gating 或 residual energy cap”后的受控 probe。为避免把失败的 auxiliary
结论误升级成“已通过”，本轮只做两件事：

1. `Enc3_tfgrid_residual`：把 F-B2 的低频复数 residual head 替换成 TF-Grid-lite complex residual head，
   仍从 shared token 输出 `0.067-1.2Hz` 复数 STFT residual，经 iSTFT 后加回 waveform。
2. `residual_energy_cap`：先选择比 hard/easy gate 更可复现的能量上限方案，限制 residual RMS 不超过 base
   waveform RMS 的 5%。hard/easy 选择性 gating 需要可靠的窗口级 hard/easy proxy；在没有新 proxy 前不把它写成
   模型内 gating，避免用训练标签或事后分层制造泄漏。

新增实现：

- residual head：`model.fb_residual_head=enc3_tfgrid_residual`。
- residual cap：`model.fb_residual_energy_cap=0.05`；`0.0` 表示关闭。
- runner：`scripts/run_f_b_enc3_residual_probe.py`。
- manifest：`runs/f_b_enc3_residual_manifest.csv`。

实验臂：

| label | 运行方式 | 作用 |
|---|---|---|
| `F0_native_stft_pre_mixer` | 复用 `runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual` | paired anchor |
| `F-B2_low_complex_residual` | 复用 `runs/f_b2_residual/f_b2_low_complex_residual/dual` | residual baseline |
| `F-B3_enc3_tfgrid_residual` | 新训练 3 seed | 验证 TF-Grid-lite residual 是否比低复杂度 residual 更会修 hard |
| `F-B3b_enc3_tfgrid_residual_cap` | 新训练 3 seed | 验证 residual energy cap 是否缓解 easy 误伤 |

命令：

```bash
./.venv/bin/python scripts/run_f_b_enc3_residual_probe.py \
  --dry-run \
  --manifest runs/f_b_enc3_residual_manifest.csv

./.venv/bin/python scripts/run_f_b_enc3_residual_probe.py \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 2 \
  --manifest runs/f_b_enc3_residual_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/f_b_enc3_residual_manifest.csv \
  --output runs/f_b_enc3_residual_summary.csv \
  --paired-output runs/f_b_enc3_residual_paired_delta.csv \
  --strata-output runs/f_b_enc3_residual_strata_delta.csv
```

通过/停止口径：

- `F-B3` 只有在 overall peak-band RR 至少 2/3 seed 改善，且 baseline easy 不再 3/3 同向变差时，才算有继续价值。
- `F-B3b` 的核心判断是相对 `F-B3` 是否降低 baseline easy 误伤；如果 hard 收益也同步消失，说明 cap 只是关掉
  residual，不是有效选择性机制。
- 若 `F-B3/F-B3b` 仍重复“hard 改善、easy 3/3 退化”，下一步不应继续堆 residual head，而应回到窗口级 proxy
  设计，再做 hard/easy selective gating。

### 7.20 F-B Enc3 residual 结果与 cap 数值修复（2026-06-27）

运行范围：

- manifest：`runs/f_b_enc3_residual_manifest.csv`
- summary：`runs/f_b_enc3_residual_summary.csv`
- paired delta：`runs/f_b_enc3_residual_paired_delta.csv`
- strata delta：`runs/f_b_enc3_residual_strata_delta.csv`
- 完成情况：`F0_native_stft_pre_mixer` 3 seed 复用完成；`F-B2_low_complex_residual` 3 seed 复用完成；
  `F-B3_enc3_tfgrid_residual` 3 seed 训练完成。
- `F-B3b_enc3_tfgrid_residual_cap` 本轮 3 seed 训练无效：三个 run 从 epoch 1 开始即 `loss=nan`，early stop
  后没有 checkpoint/metrics，因此 summary 中状态为 `missing`，不能用于结论。

相对 `F0_native_stft_pre_mixer` 的 paired 结果：

| label | overall `rr_peak_band_abs_error_mean` delta | 改善 seed | `rr_spec_abs_error_mean` delta | `frac_gt_1` delta | `frac_gt_2` delta | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `F-B2_low_complex_residual` | `+0.0072` | 2/3 | `-0.0265` | `+0.0016` | `+0.0034` | 旧结果，仍不通过 |
| `F-B3_enc3_tfgrid_residual` | `+0.0262` | 1/3 | `-0.0279` | `+0.0070` | `+0.0052` | 不通过；比 F-B2 更差 |

关键分层：

- baseline easy 仍失败：`F-B3` 的 `rr_peak_band_abs_error_mean` 三 seed 全部变差，平均 `+0.0405`；
  比 F-B2 的 `+0.0333` 更差。
- baseline hard 有正信号但弱于 F-B2：`F-B3` hard 平均 `-0.0655`、2/3 seed 改善；F-B2 是
  `-0.1108`、3/3 seed 改善。
- fast-RR 收益基本被削弱：`F-B3` fast-RR 平均 `-0.0027`、2/3 seed 改善；F-B2 是 `-0.0152`、
  3/3 seed 改善。
- 频谱指标仍有收益：`F-B3` 的 `rr_spec_abs_error_mean` overall 三 seed 全部改善，平均 `-0.0279`；
  但这没有转化成 peak-band RR 或 easy 护栏通过。
- 相对 F-B2，`F-B3` 的 overall peak-band RR 平均再差 `+0.0190`；只有 seed `20260700` 比 F-B2 好。

cap 失败与修复：

- 失败模式：`F-B3b_enc3_tfgrid_residual_cap` 三个旧 run 均在 epoch 1 出现全损失 NaN，最终没有
  `checkpoint.pt` 或 `metrics.csv`。
- 根因：`fb_residual_energy_cap` 原实现先做 `sqrt(mean(residual^2))` 再 clamp；residual head 零初始化时
  residual 为 0，反传会经过 `sqrt(0)`，导致 NaN 梯度并污染训练。
- 修复：改为先 clamp residual power，再 sqrt；新增回归测试覆盖 zero residual + cap 的 backward finite。
- 验证：`tests/test_stft_branch.py::test_fb_residual_energy_cap_keeps_zero_residual_backward_finite` 先失败后通过；
  相关测试 `165 passed`；`/tmp/f_b_cap_smoke` 的 1 epoch CPU smoke 训练和验证 loss 均为有限值。

阶段判断：

- `F-B3_enc3_tfgrid_residual` 不通过；不能扩 seed，也不应继续堆更强 TF residual head。
- `F-B3b` 旧结果无效；若要判断 residual energy cap，需要基于修复后的代码只重跑 cap 三个 seed。
- 修复后重跑 cap 时不要重跑已完成的 `F-B3`，建议：

```bash
./.venv/bin/python scripts/run_f_b_enc3_residual_probe.py \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 2 \
  --manifest runs/f_b_enc3_residual_manifest.csv \
  --start-stagger-sec 90 \
  --skip f_b3_enc3_tfgrid_residual_dual_20260700 \
  --skip f_b3_enc3_tfgrid_residual_dual_20260837 \
  --skip f_b3_enc3_tfgrid_residual_dual_20260901
```

若 cap 重跑仍不能缓解 baseline easy，则 F-B residual 路线应停止，下一步转向无泄漏的 hard/easy window proxy
与选择性 gating，而不是继续扩大 residual head。

### 7.21 F-B3b residual energy cap 重跑结果（2026-06-27）

运行范围：

- manifest：`runs/f_b_enc3_residual_manifest.csv`
- summary：`runs/f_b_enc3_residual_summary.csv`
- paired delta：`runs/f_b_enc3_residual_paired_delta.csv`
- strata delta：`runs/f_b_enc3_residual_strata_delta.csv`
- 完成情况：修复 `fb_residual_energy_cap` 数值稳定性后，`F-B3b_enc3_tfgrid_residual_cap` 3 seed 重跑完成；
  summary 当前 12 行全部为 `complete`，paired delta 9 行，strata delta 45 行。

相对 `F0_native_stft_pre_mixer` 的 paired 结果：

| label | overall `rr_peak_band_abs_error_mean` delta | 改善 seed | `rr_spec_abs_error_mean` delta | `frac_gt_1` delta | `frac_gt_2` delta | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `F-B2_low_complex_residual` | `+0.0072` | 2/3 | `-0.0265` | `+0.0016` | `+0.0034` | 旧结果，仍不通过 |
| `F-B3_enc3_tfgrid_residual` | `+0.0262` | 1/3 | `-0.0279` | `+0.0070` | `+0.0052` | 不通过 |
| `F-B3b_enc3_tfgrid_residual_cap` | `+0.0266` | 1/3 | `-0.0279` | `+0.0071` | `+0.0054` | 不通过；cap 未缓解误伤 |

关键分层：

- baseline easy 仍失败：`F-B3b` 的 `rr_peak_band_abs_error_mean` 三 seed 全部变差，平均 `+0.0406`；
  与 Enc3 的 `+0.0405` 基本相同，比 F-B2 的 `+0.0333` 更差。
- baseline hard 收益弱于 F-B2：`F-B3b` hard 平均 `-0.0630`、2/3 seed 改善；F-B2 是
  `-0.1108`、3/3 seed 改善。
- fast-RR 收益没有保住：`F-B3b` fast-RR 平均 `-0.0026`、2/3 seed 改善；F-B2 是 `-0.0152`、
  3/3 seed 改善。
- 频谱指标有改善但没有转化：`F-B3b` overall `rr_spec_abs_error_mean` 三 seed 全部改善，平均 `-0.0279`。
- 相对 Enc3，cap 对 overall peak-band RR 平均仅 `+0.00036`，三 seed 都略差；说明 5% RMS cap 在当前 residual
  幅度下基本没有形成有效选择性。

阶段判断：

- `F-B3b_enc3_tfgrid_residual_cap` 不通过，且没有缓解 baseline easy 护栏。
- F-B residual head 扩展路线停止：不扩 seed，不继续 `F-B2b_wide_residual`、更强 TF residual head 或更高 cap sweep。
- 下一步若继续 F 系列，只能转向可解释的 hard/easy 选择性机制：先定义无泄漏窗口级 proxy，再把 residual/gate 只作用在
  proxy 判定的 hard/ambiguous 窗口；不能用事后 baseline 分层标签直接训练 gating。

### 7.22 F-C0 low-complex-STFT 输出准备（2026-06-28）

背景：F-B residual 路线已停止；本节不是因为 F-B 通过而升级，而是用户明确要求后做一个受控 F-C pilot，用来判断
“低频 complex STFT 作为主输出空间”本身是否值得保留。

实现范围：

- 新模型：`time_stft_low_complex_output1d`。它复用 `TimeStftDual1D` 的 native token 生成路径，从 token
  直接预测 low-band complex STFT real/imag，再用 iSTFT 还原 `waveform`。
- 输出 STFT 口径：`win=3000`、`hop=500`、`n_fft=3000`、`center=true`、`length=18000`、`0-3Hz`。
- 输出契约：模型返回 `{"waveform": ..., "output_stft_realimag": ...}`；训练、评估和 summary 仍只用
  `waveform`。
- runner：`scripts/run_f_c_stft_output_probe.py`，只包含 `F0_native_stft_pre_mixer` 复用和
  `F-C0_low_complex_stft_output` 三 seed 新训练。
- summary：`scripts/summarize_f_a_stft_loss.py` 已扩展为识别 `F-A*`、`F-B*` 和 `F-C*`。
- checkpoint gate：`F-C0` 保留 `training.checkpoint_gate.metric=auto_direction`，但将
  `training.checkpoint_gate.max` 从 F-A/F-B 继承的 `0.5` 放宽到 `1.0`。原因是首轮三 seed 在
  `max=0.5` 下没有任何 epoch 通过 gate，无法保存 checkpoint/metrics；放宽后只用于生成 RR metrics，
  不表示通过原严格方向护栏。

本轮没有改变：

- 不改 data split、窗口采样、target 构造或核心 metrics 口径。
- 不加入 low-pass target baseline；`F-C1_lowpass_target_baseline` 需要单独确认，因为它会改变训练目标解释。
- 不做 full-band complex STFT、magnitude-only 主输出、CWT/SST 输出空间或 learned proxy/gating。

命令：

```bash
./.venv/bin/python scripts/run_f_c_stft_output_probe.py \
  --dry-run \
  --manifest runs/f_c_stft_output_manifest.csv

./.venv/bin/python scripts/run_f_c_stft_output_probe.py \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 2 \
  --manifest runs/f_c_stft_output_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/f_c_stft_output_manifest.csv \
  --output runs/f_c_stft_output_summary.csv \
  --paired-output runs/f_c_stft_output_paired_delta.csv \
  --strata-output runs/f_c_stft_output_strata_delta.csv
```

通过/停止条件：

- 至少 2/3 seed 的 overall `rr_peak_band_abs_error_mean` 优于 `F0_native_stft_pre_mixer`。
- baseline easy 不能 3/3 同向变差；fast-RR 不能明显劣于 F0。
- 需要额外报告 `train_history.csv` 中原始 `val_signed_corr` 是否低于 `0.5`；如果始终高于 `0.5`，
  即使 RR metrics 偶然改善，也只能视为高风险输出空间信号。
- 若 F-C0 只改善 `rr_spec_abs_error` 但 peak-band RR / easy 护栏失败，则判为输出空间不适合作为当前主线，不扩
  full-band 或 lowpass-target sweep。

首轮执行记录：

- 2026-06-28 首次正式运行沿用了 F-A/F-B 的 `checkpoint_gate.max=0.5`，三个 `F-C0` run 均完成
  50 epoch 训练但没有任何 epoch 通过 gate，因此没有 `checkpoint.pt` 或 `metrics.csv`，runner 抛出
  “没有 epoch 满足 checkpoint_gate”。
- 三个 run 的 `val_signed_corr` 大致从 `0.92-0.93` 降到 `0.77` 左右，始终高于 `0.5`。这说明 F-C0
  在严格方向护栏下已经失败；需要重跑只是为了补齐 RR metrics 和分层表，而不是因为已有结果通过。
- 失败 run 保留在 `runs/f_c_stft_output/f_c0_low_complex_stft_output/dual/`，summary 会忽略没有
  `metrics.csv` 的目录；使用修复后的 runner 重跑会新建 timestamp 目录。

## 8. 外部经验在本计划中的用法

- 音频 waveform generation 中常用 multi-resolution STFT / spectrogram loss，但这是外部结构经验，不应直接照搬全频强匹配到呼吸任务。
- 复合 loss 的权重最好按梯度贡献而非 raw 数值调节；否则 STFT loss 的数值尺度可能误导训练。
- iSTFT 输出路径必须显式处理 window、center、NOLA、length 和边界裁剪问题。
- TF-GridNet / Conformer / Mamba 等模型只提供结构偏置参考。F-B 的 target-STFT token 很少，第一批应优先小型 TF-CNN、band-aware head 和 TF-Grid-lite，而不是大模型替换。
- CWT/SST 的多分辨率和 ridge 能力有合理信号基础，但第一用途应是 high-CWT 心冲击调制分支、modulation features 和 SST ridge diagnostics；不适合第一阶段最终输出空间。
