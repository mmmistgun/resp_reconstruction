# F 系列 STFT 约束与输出空间实验草稿

本文是 F 系列实验的初版草稿，用于把“输出一维时序后计算 STFT loss”和
“让 STFT 在网络中传播或作为输出空间”两条路线拆开。当前优先级是先验证
F-A：不改变推理输出、不改变主模型路径，只修改训练目标。

## 背景判断

- E 系列已经证明：在 `patch_mixer1d` 路线中，STFT 输入对降低 peak-band
  长尾谐波误拣有稳定正信号。
- `stft_only` magnitude 路线存在极性盲和 gate failure 风险，不应直接升级为
  STFT 主输出。
- 历史 L3 低频 STFT magnitude loss 已失败：`stft_mag=0.02/0.05` 均让
  `rr_peak_abs_error` 明显恶化，且 `rr_spec_abs_error` 没有补偿性收益。
  因此 F-A 不应重复“强匹配 STFT 图像”的路线。
- F 系列第一阶段应从“减少呼吸带主峰/谐波误拣”出发，而不是追求整体
  log-magnitude STFT 更像。

## 主线选择依据

F 系列不是回退到早期 `tho_small` 或旧 rawish L0/L1 口径，而是沿用当前
`tho_research_v2` soft-z 数据口径，并以 E3/E4 后的主线作为 anchor。

- E3 已经把问题从“STFT 是否有用”推进到“哪个前端和注入位置值得收口”：
  `conv2d fullband 8Hz` 是强简单 STFT 前端，`frequency_mlp` 和 `soft_band`
  未超过它；`native_inject pre_mixer` 是当前更稳的主线，`token_mid_mixer`
  只作为 `rr_spec` / 相关性诊断候选。
- E3-C2 的分层结论说明 STFT 收益主要来自 baseline hard 和低
  `spectrum_similarity` 窗口。因此 F-A 的 loss 设计应围绕“谐波误拣长尾”和
  hard/low-spectrum 分层，而不是平均地强匹配所有 STFT bin。
- E4 说明 SST 在少数 hard 窗口可视化分离度更清晰，但任务指标相对 STFT
  没有收益，不进入 F 系列第一主线。
- E5 的 gated / cross-attention 路线重复了“频谱小改善，但 peak-band RR
  和呼吸计数变差”的模式。因此 F 系列第一阶段先做 loss 判据，不再扩大
  可学习融合结构。

因此 F0 应明确是：`patch_mixer1d + conv2d fullband 8Hz + native_inject
pre_mixer + 当前原 loss`，并配对 `native_time_only` 解释新增 STFT loss 的收益。

## 固定对照口径

- 数据：沿用 `configs/tho_research_v2.yaml` 当前 research v2 soft-z 口径。
- 主线模型：E3/E4/E5 后收口的 patch STFT 主线，作为 F0 anchor。
- 输出：第一批实验仍输出一维波形，推理接口保持不变。
- 数据 seed 和验证集：固定当前主线口径，避免验证集变化掩盖 loss 差异。
- 方向门控：baseline 和所有候选使用同一 gate 口径。
- 第一批每个候选先跑 2-3 个 seed；只有出现一致正信号后再扩到 6 个以上 seed。

## F-A：波形输出 + STFT 派生 loss

F-A 只改变训练目标。模型仍输出 `y_hat`，再由 `STFT(y_hat)` 与 `STFT(y)`
构造额外损失。

### 第一批矩阵

| 实验 | 改动 | 目的 | 阶段判断 |
|---|---|---|---|
| F0 | 当前主线 anchor，原 loss | 固定同口径对照 | 必跑 |
| F-A0 | `L_base + L_dist` | 约束 0.05-0.7Hz 内 spectral distribution，直接打主峰/谐波误拣 | 第一优先级 |
| F-A1 | `F-A0 + L_bandE` | 约束呼吸相关频带能量轨迹，看 hard / low-spectrum 窗口是否改善 | 第一批 |
| F-A2 | `F-A1 + tiny L_logmag/sc` | 小权重补充 STFT 细节，不让 magnitude loss 主导训练 | 第一批可选 |

### Loss 设计草稿

`L_dist`：在呼吸带内计算 normalized spectral distribution。建议先用
Jensen-Shannon divergence 或 symmetric KL，不直接用不可导 peak bin，也暂不使用
soft-RR centroid。

`L_bandE`：对少量重叠频带计算每个 STFT frame 的 log energy trajectory，再做
SmoothL1。候选频带：

| 频带 | 含义 | 初始权重倾向 |
|---|---|---:|
| 0.05-0.3Hz | 慢呼吸 / 主节律低端 | 0.5 |
| 0.1-0.7Hz | 主要呼吸频带 | 1.0 |
| 0.3-1.2Hz | 呼吸谐波 / 波形尖锐度 | 0.3-0.5 |
| 1.2-3.0Hz | 输出高频污染诊断或弱 suppression | 0.0-0.05 |

`L_logmag/sc`：只作为 F-A2 的小权重辅助，不作为 F-A0 主项。STFT 总梯度贡献
第一版控制在 5-10%，避免重复旧 L3 “频谱图更像但 peak RR 变差”的失败模式。

### 后续候选

| 实验 | 改动 | 进入条件 | 风险 |
|---|---|---|---|
| F-A3 | F-A0/F-A1 的 multi-resolution STFT 版本 | F-A0 或 F-A1 有正信号 | 窗长变量会增加归因难度 |
| F-A4 | 极小权重 low-band complex STFT | magnitude/distribution 改善频谱但相位或极性仍不稳 | phase 对时移和低能量 bin 敏感 |

F-A3 的候选分辨率暂定为 20s/30s/60s，其中 30s 为主口径。不要用 5s/10s
短窗作为主频 loss；呼吸频带分辨率太粗，可能鼓励 envelope artifact。

## F-B：辅助 STFT 表征

F-B 改变训练结构，但最终推理输出仍优先保持一维波形。它只有在 F-A 至少证明
STFT 派生监督有任务收益后再进入。

F-B 不应先理解成“把输出空间从 waveform 换成 STFT”，而应按网络角色分层：
auxiliary head 只做表征监督；consistency 检查监督能否传导到 waveform；residual
才允许 STFT path 参与最终输出；direct decoder 最后考虑。

### STFT 参数口径

第一版保持 30s/5s 的单分辨率，减少新自由度：

- `fs=100Hz`
- `win_length=3000`
- `hop_length=500`
- `n_fft=3000`
- window 使用 Hann
- auxiliary log-magnitude head 使用 `center=False`，约 31 帧，便于和当前 E 系列
  STFT 时间帧口径对齐。
- iSTFT 输出路径使用 `center=True`，或显式 padding 后 crop，约 37 帧，避免
  Hann 窗在边界处产生 overlap-add / NOLA 伪影。

频带上不直接照搬输入 STFT 的 8Hz/12Hz 解释。输入 STFT 的高频价值来自 BCG 中
心冲击/扰动上下文；target/output STFT 是胸带呼吸波形，应更保守：

| 用途 | 预测或监督频带 | 主权重区域 | 说明 |
|---|---|---|---|
| auxiliary log-magnitude head | 0.033-3Hz | 0.067-1.2Hz | 可输出 k=1..90，但 k=1 低权重 |
| consistency | 同 auxiliary head | 0.067-1.2Hz | 只验证 aux 是否传导到 `STFT(y_hat)` |
| residual correction | 0.067-1.2Hz | 0.067-1.2Hz | 第一版避免高频/谐波接管计数 |
| wider residual | 0.067-3Hz | 0.067-1.2Hz 为主 | 仅在 1.2Hz residual 过度平滑时再试 |

### F-B 矩阵

| 实验 | 改动 | 目的 | 是否进入第一批 F-B |
|---|---|---|---|
| F-B0 | 主波形 head 不变，增加 aux target-STFT logmag head | 判断 shared latent 是否含有目标时频结构 | 可做，但只作诊断 |
| F-B1 | F-B0 + delayed consistency：`STFT(y_hat)` 对齐 aux head | 判断目标 STFT 监督能否传导到 waveform 输出 | 更有判断价值 |
| F-B2 | low-band complex residual：`y_hat = y0 + alpha * iSTFT(delta_S)` | 让 STFT path 低扰动修正 waveform | F-B1 有信号后再做 |
| F-B2b | residual 扩到 0.067-3Hz | 处理 1.2Hz residual 过度平滑 | F-B2 成功后再做 |

F-B0 的通过条件不能是 aux loss 下降。aux head 很可能只是旁路任务，尤其当前主线
已经有 BCG STFT 输入。F-B0 最多证明 latent 有目标时频信息；真正推进信号是
F-B1 是否改善 `STFT(y_hat)` 和任务指标。

F-B1 的 consistency 不应第一步就双向回传。建议先 warmup aux head，只开
`L_aux`；随后开小权重 consistency。第一版 consistency 用 `detach(Z_aux)` 去拉
`STFT(y_hat)`，避免当前较差的 waveform 反向污染 aux head。

F-B2 需要避免 residual branch 接管主 decoder。注意 `alpha=0` 会让 residual head
参数初始梯度也为 0，只剩 `alpha` 自己能动；更稳的做法是 `alpha=1e-3` 小非零初始化，
或给 residual head 加直接 STFT auxiliary loss。F-B2 的 loss 应保留 base waveform：

```text
L = L_base(y_hat, y)
  + beta * L_base(y0, y)
  + lambda_stft * L_stft(y_hat, y)
  + lambda_res * L_residual
```

其中 `beta` 先取 0.3-1.0，确保 `y0` 仍是可用主输出。`L_residual` 可用
`||delta_S||_1` 或 low-band residual energy ratio，防止 STFT path 过度扰动。

保留 ReZero / residual gate 的原因不是形式偏好，而是归因和稳定性要求：

- 初始模型应尽量等价 F0，避免新增 STFT path 一开始破坏 easy windows。
- E5 已出现“频谱小改善，但 peak-band RR / 计数变差”的模式，residual path
  必须先作为小扰动进入。
- residual branch 若自由接管，F-B2 就不再回答“STFT 是否修正 hard windows”，而会退化成
  新 decoder 搜索。

`alpha=0` 的梯度问题按以下优先级解决：

1. 使用小非零 gate，例如 `alpha=1e-3` 或 `alpha = alpha_max * sigmoid(a)` 且 bias
   初始化到约 `1e-3`。
2. 给 residual head 一个直接 STFT auxiliary loss，使 head 在 final waveform 通路很弱时
   仍有梯度。
3. 使用 staged training：先训练 F-B1/aux 表征，再解锁 residual；不建议一开始冻结主
   decoder 让 residual 单独学习，否则会放大接管风险。

不同部分可以使用不同 STFT 分辨率，但第一版应保持“同组内一致、跨角色分工清晰”：

- auxiliary 和 consistency 使用同一 30s/5s、`center=False`、约 31 帧口径，减少旁路监督
  与 waveform-STFT 对齐之间的网格差异。
- residual / direct iSTFT 使用 30s/5s、`center=True` 或 padding+crop、约 37 帧口径，
  这是为了可逆重建和边界稳定。
- 不建议第一版给 auxiliary、consistency、residual 分别引入 20s/30s/60s 多分辨率。
  若 F-B1 有正信号，再考虑只给 aux/consistency 增加 multi-resolution 诊断；residual
  仍先保持单分辨率。

### F-B 诊断指标

F-B 不能用 aux/STFT loss 选模。除通用任务指标外，需要额外保存：

- `metrics_y0_base.csv`：residual 前 waveform 指标。
- `metrics_yhat_final.csv`：residual 后 waveform 指标。
- `stft_head_error_by_band.csv`：aux/residual head 对 target STFT 的分频带误差。
- `waveform_stft_error_by_band.csv`：`STFT(y_hat)` 对 target STFT 的分频带误差。
- `residual_energy_by_window.csv`：`||delta_y|| / ||y0||` 或同类低频能量占比。

若 aux head 的 STFT error 下降但 `STFT(y_hat)` 不变，说明它是旁路；若 residual
energy 很大但主指标没有改善，说明 STFT path 在接管；若 `rr_spec_abs_error`
改善但 `breath_count_zero_cross_abs_error` 变差，应停止该路线。

### F-B 特征提取器选择

F-B 的 STFT 图很小：30s/5s 下 180s 窗口只有约 31 或 37 个 frame；target/output
侧第一版主要关注 0.067-1.2Hz，最多扩到 3Hz。因此只参考语音时频模型的结构思想，
不直接搬 full TF-GridNet、full U-Net、大 attention、Mamba 或大 Conformer。

优先结构偏置：

- 小型 TF-CNN：第一优先级。E3 中 `conv2d fullband 8Hz` 是最清楚的强 STFT
  参照，`frequency_mlp` 和 `soft_band` 未超过它，说明当前尺度下局部 TF 卷积足够强。
- band-aware TF-CNN：用少量生理频带 head 或 band gate 注入先验，减少逐 bin 自由度。
- TF-Grid-lite：只取 full-band / sub-band / temporal 分工思想，用频率局部卷积、
  时间卷积和全频带 gate，而不是完整 cross-frame attention 堆叠。
- dual-path TCN/GRU：可作为 residual head 的中等复杂度候选，把频率内结构和时间动态拆开。

后置或不进第一批：

- mini-Conformer：只可放在 F-B0/F-B1 auxiliary token 上试，不直接控制最终 waveform。
  E5 的 cross-attention 已暴露“spec 小改善、peak-band/计数变差”风险。
- SincNet/LEAF：更适合 raw BCG learned frontend probe，不应和 F-B STFT output head 同批改。
- Conv-TasNet/SEANet：适合另开 waveform 主干或 encoder probe，不适合替代 F-B STFT head。
- Mamba/SSM：适合 long raw/patch sequence；F-B 的 31/37 frame 太短，长序列优势发挥不出来。
- full U-Net / full TF-GridNet / 大 attention：第一批不建议，参数和结构自由度过高。

建议最多保留三个 F-B head 版本：

| 版本 | 用途 | 结构 | 进入条件 |
|---|---|---|---|
| F-B0-Enc1 | 最小 auxiliary head | pool 到 31 帧 + 2 层 Conv1d + Linear 到频率 bin | F-B0 默认 |
| F-B1-Enc2 | band-aware auxiliary head | pool 到 31 帧 + shared temporal Conv1d + 少量 band heads | F-B0/F-B1 有旁路疑虑或需要频带先验 |
| F-B2-Enc3 | TF-Grid-lite complex residual | pool/interpolate 到 37 帧 + 频率局部卷积 + 时间卷积 + band global gate | F-B1 证明监督能传导到 waveform 后 |

F-B0-Enc1 只验证 target-STFT auxiliary supervision 是否有价值；F-B1-Enc2 验证
显式分频带结构是否更稳；F-B2-Enc3 才让 STFT 参与最终 waveform。不要把 learned
frontend、aux head、residual head、cross-attention 同批叠加，否则无法归因。

## F-C：STFT 作为主输出

F-C 是高风险路线，只有在 F-A/F-B 出现稳定正信号后才考虑。

| 实验 | 改动 | 判断 |
|---|---|---|
| F-C0 | 输出 0-3Hz low-band complex STFT，经 iSTFT 回波形 | 可作为有限范围验证 |
| F-C1 | 输出 full-band complex STFT，经 iSTFT 回波形 | 维度和过拟合空间过大，暂不建议 |
| F-Cx | magnitude-only STFT output | 不建议作为最终输出，只能作为 auxiliary head |

如果进入 F-C，应优先输出 real/imag，而不是 magnitude/phase angle。phase angle
有 wrapping 问题；real/imag 更适合普通实值网络优化。

F-C0 若只输出 0-3Hz 并把高频置零，最终 waveform 是低通结果；这会改变输出定义，
必须和当前 waveform decoder 对照解释。F-C1 full-band 维度约为
`1501 freq bins * 37 frames * 2 real/imag`，远大于 18000 点 waveform，且大量高频
bin 对呼吸任务没有价值，容易增加过拟合空间。因此 F-C 不进入第一阶段。

### CWT / SST 是否进入 F-C

当前不建议把 CWT 加进 F-C 输出空间。CWT 的多分辨率直觉成立：低频呼吸需要较好
频率分辨率，高频心冲击需要更好时间分辨率；但仓库已有 E4 前置和 E4-SST 证据约束：

- E4 前置里，低频 CWT 在 hard 窗口的谐波分离不如当前固定窗 STFT，CWT 当时被砍掉。
- SST 在前置诊断中能提升 hard 窗口分离度，但 E4-SST 训练打平，说明“分离度更清晰”
  没有兑现成任务指标收益。
- CWT 是冗余表示，inverse CWT 的数值稳定性、边界 cone-of-influence、尺度归一化和
  重建一致性都比 iSTFT 更难控；SST 的反变换 / ridge reconstruction 更不适合作为第一版
  网络输出路径。

因此 CWT/SST 不进入 F-C。若后续仍想验证 CWT，应另开“表示输入/诊断”分支，而不是作为
F-C decoder：

| 候选 | 目的 | 判断 |
|---|---|---|
| F-CWT0 | 当前低频 STFT + 短窗 high-STFT，作为多分辨率 anchor | 先证明高频时间分辨率变量本身有用 |
| F-CWT1 | 当前低频 STFT + high-CWT 1-8Hz，经 gate/FiLM 影响 low/native tokens | 只验证高频 CWT 是否优于 high-STFT |
| F-CWT3 | high-CWT modulation features：能量轨迹、ridge/centroid/entropy、质量提示 | 比 dense CWT map 更可归因 |
| F-SST1 | 在 CWT 有收益后加 cardiac SST ridge features | 后置诊断 / 显式 ridge 特征，不做 dense input 第一批 |

这些候选若要做，也应放在 F-A/F-B 之后，并且必须使用 zero/small-init gate 或 FiLM，
避免重演 E5 “spec 小改善但 peak-band / 计数变差”。不要第一批做 full CWT map、
dense SST map、attention 和新 decoder 的叠加实验。

## 阶段通过规则

候选进入扩 seed 或下一阶段前，至少满足：

- `rr_peak_band_abs_error_mean/median` 不劣于 F0，优先改善。
- `frac_gt_1` 或 `frac_gt_2` 至少一个明确下降。
- `rr_spec_abs_error` 不能明显恶化。
- `breath_count_zero_cross_abs_error` 不能明显恶化。
- `relative_envelope_mae/corr` 至少不明显恶化。
- `band_limited_corr` 和 `best_lag_corr` 只作诊断，不单独决定胜负。

停止条件：

- 只改善 STFT 相关 loss 或 `rr_spec_abs_error`，但 peak-band RR / 计数明显恶化。
- 多 seed 方差大到无法区分候选收益和训练随机性。
- F-A0/F-A1 没有正信号时，不继续 F-A3/F-A4，也不升级到 F-B/F-C。

## 第一阶段建议执行清单

1. 为 F-A0/F-A1/F-A2 补充 loss 实现和单元测试。
2. 增加 runner/manifest，固定 F0、F-A0、F-A1、F-A2 的同 seed 配对。
3. 训练 2-3 seed pilot。
4. 汇总整体指标和 hard/low-spectrum 分层指标。
5. 只有 F-A0 或 F-A1 通过阶段规则，才扩 seed 或设计 F-B。
