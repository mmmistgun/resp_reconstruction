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

### 2.6 后续候选

| 实验 | 改动 | 进入条件 | 风险 |
|---|---|---|---|
| `F-A4_mrstft` | `F-A0/F-A1/F-A2` 的 multi-resolution STFT 版本 | F-A0/F-A1/F-A2 至少一个有 paired 正信号 | 窗长变量增加归因难度 |
| `F-A5_low_complex` | 极小权重 low-band complex STFT | magnitude/distribution 改善频谱但相位、polarity 或 lag 仍不稳 | phase 对时移和低能量 bin 敏感 |

MR-STFT 第一版只建议 `20s/30s/60s`，其中 30s 为主权重。不要把 5s/10s 作为主频 loss；如果要用短窗，它只能作为 envelope/quality 辅助，不应主导 RR 选择。

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

| 实验 | 改动 | 目的 | 是否进入第一批 F-B |
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

## 4. F-B 特征提取器选择

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
| `F-B0-Enc1` | 最小 auxiliary head | pool 到 31 帧 + 2 层 Conv1d + Linear 到频率 bin | F-B0 默认 |
| `F-B1-Enc2` | band-aware auxiliary head | pool 到 31 帧 + shared temporal Conv1d + 少量 band heads | F-B0/F-B1 有旁路疑虑或需要频带先验 |
| `F-B2-Enc3` | TF-Grid-lite complex residual | pool/interpolate 到 37 帧 + 频率局部卷积 + 时间卷积 + band global gate | F-B1 证明监督能传导到 waveform 后 |

## 5. F-C：STFT 作为主输出

F-C 是高风险路线，只有在 F-A/F-B 出现稳定正信号后才考虑。

| 实验 | 改动 | 判断 |
|---|---|---|
| `F-C0_low_complex_stft` | 输出 `0-3Hz` low-band complex STFT，经 iSTFT 回 waveform | 只能作为有限范围验证 |
| `F-C1_full_complex_stft` | 输出 full-band complex STFT，经 iSTFT 回 waveform | 维度和过拟合空间过大，暂不建议 |
| `F-Cx_mag_only` | magnitude-only STFT output | 不建议作为最终输出，只能作为 auxiliary head |

如果进入 F-C，应优先输出 real/imag，而不是 magnitude/phase angle。phase angle 有 wrapping 问题；real/imag 更适合普通实值网络优化。magnitude-only 不能决定符号、相位和时移，不适合作为最终输出路径。

F-C0 若只输出 `0-3Hz` 并把高频置零，最终 waveform 是低通结果。这会改变输出定义，必须明确两件事：

1. waveform loss 是否也对 low-pass target 计算；
2. 任务指标是否仍在完整 target 上解释。

如果仍对完整 target 解释，F-C0 不是和 F0 同任务；如果改成 low-pass target，则需要另设 baseline，例如 `F0_lowpass_waveform_decoder`。

F-C1 full-band 维度约为 `1501 freq bins * 37 frames * 2 real/imag`，远大于 18000 点 waveform，且大量高频 bin 对呼吸任务没有价值，容易增加过拟合空间。因此 F-C 不进入第一阶段。

## 6. F-D：CWT / SST 与多分辨率高频心冲击分支（不作为第一阶段输出空间）

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

### 6.1 CWT 图像分辨率口径

CWT 图像不应被视为“另一张同尺寸 STFT 图”。STFT 使用固定窗长，每个频率 bin 的时间/频率分辨率相同；CWT 的低频 scale 有更长有效窗，高频 scale 有更短有效窗。因此 CWT 可以采样成 `[scale, time]` 矩阵，但每一行的物理感受野不同。

建议在文档和配置里区分两件事：

```text
representation grid:
  high-CWT 自己的 scale/time 采样网格，例如 36 scales × 180 time steps。

fusion token grid:
  high-CWT encoder 输出的 respiratory-scale context tokens，例如 31 tokens。
```

F-D 不要求 CWT 和 STFT image-grid 对齐。正确做法是：high-CWT 保留更密时间采样和 log-scale 频率轴，经独立 encoder 压缩到 `31` 个 respiratory-scale tokens，再与 low-STFT/native tokens 融合。

### 6.2 F-D0：短窗 high-STFT anchor

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

### 6.3 F-D1：high-CWT dense map branch

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

### 6.4 Low-CWT 后置，不进入第一批

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

### 6.5 SST 参数：ridge feature extractor，而不是 dense map 第一版输入

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

### 6.6 F-D 特征提取网络

#### F-D1-Enc1：high-CWT small CWT-CNN + TCN

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

#### F-D2-Enc2：modulation-feature TCN

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

#### F-D3-Enc3：CWT map + SST ridge feature dual path

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

#### F-D5-Enc4：local attention pooling，后置

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

### 6.7 高频到低频的融合方式

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

### 6.8 F-D 实验矩阵

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

### 6.9 F-D 实现和诊断记录

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

## 7. 阶段通过规则

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

## 8. 第一阶段建议执行清单

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

### 8.1 F-A pilot 结果记录（2026-06-26）

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
- 主要风险是 easy / fast-RR 护栏：`F-A2_dist_bandE` 的 baseline easy 分层 `rr_peak_band_abs_error_mean` 三个 seed 全部变差，平均 `+0.0218`；fast-RR 分层有 2/3 seed 变差，平均 `+0.0022`。按第 7 节规则，它不能直接升级到扩 seed 或 F-B/F-C。
- 梯度规模没有过强迹象：`F-A2_dist_bandE` tail STFT/base 梯度比平均约 `3.7%`，低于原计划 5-10% 上限；效果主要来自 band-energy，dist 分量梯度约 `0.000004`，远小于 band-energy。

阶段判断：

- 不进入 F-B1/F-B2，也不扩到 6+ seed。
- 保留 `F-A2_dist_bandE` 作为正信号候选，但必须先做护栏修正：降低或调度 `stft_band_energy_weight`，或让 STFT 派生 loss 更偏向 hard/low-spectrum 窗口，目标是保留 hard/low-spectrum 收益，同时消除 easy 与 fast-RR 系统性恶化。
- `F-A0_dist` 不单独继续；若继续使用 distribution 项，应作为 `F-A2` 的小辅助，并重新评估权重或 beta，否则当前梯度贡献太小。

### 8.2 F-A2 guard probe 计划（2026-06-27）

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

### 8.3 F-A2 guard probe 结果记录（2026-06-27）

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

- 不直接进入 F-B，也不扩 seed。`F-A2b_dist_bandE_w005` 具有比原 F-A2 更稳定的 overall / hard 正信号，但 easy 和 fast-RR 仍系统性变差，不满足第 7 节护栏。
- 不建议继续只扫更小 scalar 权重。`0.003` 已显示继续降权会削弱 hard 收益；下一步应改变监督选择方式，例如 hard/low-spectrum 加权、阶段性 band-energy 调度，或对 easy/fast-RR 档降权/关闭 STFT 派生 loss。
- 若资源允许，下一批更有价值的候选是保留 `stft_band_energy_weight=0.005`，但把 STFT loss 只强化到 baseline hard / low-spectrum 或低 confidence 窗口；成功标准必须要求 baseline easy 不再三 seed 同向变差。

### 8.4 F-A2 confidence guard probe 计划（2026-06-27）

目的：接受 8.3 的判断，不继续做更小 scalar 权重 sweep；保留 `F-A2b_dist_bandE_w005` 的权重强度，但改变 STFT 派生 loss 的样本选择，让约束更偏向低 waveform confidence 窗口。该 probe 不改模型结构、数据 split、评价指标或已有 F0/F-A2/F-A2b 结果。

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

## 9. 外部经验在本计划中的用法

- 音频 waveform generation 中常用 multi-resolution STFT / spectrogram loss，但这是外部结构经验，不应直接照搬全频强匹配到呼吸任务。
- 复合 loss 的权重最好按梯度贡献而非 raw 数值调节；否则 STFT loss 的数值尺度可能误导训练。
- iSTFT 输出路径必须显式处理 window、center、NOLA、length 和边界裁剪问题。
- TF-GridNet / Conformer / Mamba 等模型只提供结构偏置参考。F-B 的 target-STFT token 很少，第一批应优先小型 TF-CNN、band-aware head 和 TF-Grid-lite，而不是大模型替换。
- CWT/SST 的多分辨率和 ridge 能力有合理信号基础，但第一用途应是 high-CWT 心冲击调制分支、modulation features 和 SST ridge diagnostics；不适合第一阶段最终输出空间。
