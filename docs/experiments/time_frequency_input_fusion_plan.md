# 时频输入与多分支融合实验规划

本文档规划下一阶段从“纯时序输入”转向“时序 + 时频输入”的实验路线。它不是代码实现计划，
重点是明确研究假设、实验顺序、对照关系、指标护栏和阶段决策规则，避免在模型复杂度增加后
无法解释收益来源。

## 背景

当前 20260620 soft-z 主线已经完成多种纯时序模型候选重跑。按现有记录，
`multiscale_decomp_mixer1d`、`patch_mixer1d` 和
`patch_hann_bandlimited_output1d` 更接近当前任务主线；`polyphase` 虽然验证
loss 和部分频谱指标较好，但 `rr_peak_band_abs_error` 明显偏高，不宜作为主选。

历史 rawish state-aligned 实验还暴露了一个关键问题：模型存在稳定的低频方向/极性盆地。
正向和反向解都能给出可用的带通呼吸率，甚至反向解在部分 RR 指标上更好，但它不一定符合
state-aligned 前提下对胸带呼吸波形方向的解释。因此，下一阶段不能只追求更低 loss 或更复杂
频域表征，而要判断新增输入是否真的改善呼吸任务、低频方向和低信噪比窗口鲁棒性。

## 核心问题

1. 时频输入是否提供了纯时序模型没有稳定利用到的信息。
2. 手工分频带是否能帮助模型区分呼吸主频、谐波、心冲击和高频扰动。
3. 多输入融合是否能改善低信噪比、节律变化和局部体动窗口，而不是只提升平均 loss。
4. 新增时频分支是否会扩大模型自由度，导致指标收益不可归因或加重极性/相位歧义。
5. 高频心冲击信息是否能通过心肺耦合为呼吸重建提供增益，还是只引入个体状态、
   运动伪相关或训练捷径。

## 非目标

- 本阶段不以一次性引入 STFT、CWT、SST、分频带和 cross-attention 大模型为目标。
- 本阶段不重新定义核心评价指标。
- 本阶段不以单 run 的 best val loss 排名作为模型选择依据。
- 本阶段不把 CWT/SST 作为第一批主实验；它们应在 STFT 输入证明有价值后再进入。

## 固定实验口径

除非特别说明，后续实验应固定以下口径：

- 数据：`configs/tho_research_v2.yaml` 当前 20260620 soft-z 数据集。
- 输入主信号：`bcg_rawish_wideband_state_aligned_segment_soft_z`。
- 目标：`tho_waveform_segment_soft_z`。
- 数据窗口：正式对照使用全量 train/val。
- 数据 seed：固定 `data.train_sample_seed=20260610`、`data.val_sample_seed=20260611`。
- 训练 batch：优先沿用当前 batch128 效率口径。
- 训练 seed：每个进入正式比较的候选至少 3 个 seed。
- 指标汇总：使用 `summarize_tho_runs.py` 生成同口径汇总表。
- 方向门控：沿用当前候选脚本的 `training.checkpoint_gate.metric=auto_direction`、
  `checkpoint_gate.max=0.5`，并对 baseline 和所有时频候选使用同一设置。polarity
  比较必须在同一门控口径下进行，否则比较的是“门控 + 模型”而非模型本身。

### STFT 默认参数

STFT 的信息量主要由窗长、hop 和归一化决定，频带上限只是其中一个变量。为避免
E1 的频带消融被这些隐含参数混淆，第一轮固定以下默认值，只把频带上限作为实验变量：

- 输入采样率：`fs=100Hz`，窗长 180s（18000 样本），与训练窗口一致。
- STFT 窗长：`30s`（3000 样本）。频率分辨率约 `0.033Hz`，足以区分呼吸主频
  （约 0.1-0.5Hz）与其谐波；更短窗会把呼吸主带和谐波糊在一起，导致“STFT 无增益”
  其实是分辨率问题而非信息问题。
- STFT hop：`5s`（500 样本）。在 180s 窗内给出约 31 帧，覆盖节律变化窗口；
  更长 hop 会损失节律时间分辨率，正是分层诊断最关心的部分。
- 幅值表示：`log1p magnitude`，先做 per-window log1p 压缩动态范围，再做全局
  （跨 train 统计）每频带 z-score，保证低 SNR 窗口不被高能心冲击帧主导。
- 复数相位：第一轮默认丢弃，只用 magnitude（见 E1 关键判断与 E4）。
- 第一轮 STFT 窗长和 hop 固定，只扫频带上限；窗长/hop 作为 E1 出现稳定收益后的
  二级消融，不与频带变量同批改变。

## 模型选择指标

主护栏：

1. `rr_peak_band_abs_error`：当前模型选择主指标，不能明显恶化。
2. `rr_spec_abs_error`：频域呼吸率护栏，不能明显恶化。

任务辅助：

1. `relative_envelope_mae`：相对呼吸强弱误差。
2. `relative_envelope_corr`：相对呼吸强弱同步。
3. `spectrum_similarity`：频谱一致性辅助指标。

诊断指标：

1. `band_limited_corr`：低频方向和形态诊断。
2. `best_lag_corr`、`best_lag_sec`：小范围时移诊断。
3. raw `rr_peak_abs_error`：局部毛刺、尖峰和双峰诊断，不作为唯一主护栏。

阶段通过条件：

- 新模型的 `rr_peak_band_abs_error` 均值不能明显劣于同 seed 或同批 baseline。
- `rr_spec_abs_error` 不能以明显变差换取 peak-band RR 小幅收益。
- 至少一个任务辅助指标改善，或低信噪比/节律变化窗口的分层指标改善。
- 如果 `band_limited_corr` 长期为负，需要明确解释为可接受的任务解还是 polarity 失败。

阶段停止条件：

- 只改善 best val loss，但主护栏恶化。
- 只改善低频相关，但 `rr_peak_band_abs_error` 和 `rr_spec_abs_error` 明显恶化。
- 多 seed 方差大到无法区分模型收益和训练随机性。
- 新增输入分支只在训练集改善，验证集或跨 samp_id（leave-samp_id-out）口径无收益。

## 实验阶段

### E0：确认纯时序对照基线

目的：为后续输入端实验提供同口径 baseline。这里不要求证明某个纯时序模型是“最强”，
而是保留多个有代表性的纯时序参照，避免后续时频输入只相对单一结构得出偶然结论。

前置确认：

- 先跑 `scripts/audit_split_independence.py --config configs/tho_research_v2.yaml`，
  确认当前 20260620 数据 train/val 的 `has_samp_id_leakage=false`、`overlap_samp_id_count=0`。
  数据制作阶段已按 samp_id 分离，这里只是把它固化为可复核的前置门槛。
- 该确认是后续所有“怀疑高频/宽频带捷径就看跨 samp_id 是否无收益”护栏的事实基础。
  若审计意外报告 samp_id 泄漏，必须先修数据划分，否则 E1 的高频消融护栏失效。

候选（三个主线 + 三个反例）：

主线（RR peak-band 主护栏较好，作为收益参照下界）：

- `patch_mixer1d`
- `multiscale_decomp_mixer1d`
- `patch_hann_bandlimited_output1d`

反例（用于压力测试护栏和 polarity，不作为收益参照）：

- `polyphase_patch_hann_bandlimited1d`：val loss 最强但 `rr_peak_band_abs_error`
  明显偏高，用作“低 loss ≠ 好呼吸任务”的反例。
- `multiscale_patch_hann_bandlimited1d`：val loss 也较好但 RR peak-band 偏高，
  作为第二个 loss/RR 背离反例。
- `downsampled_ssm1d`：SSM 类结构，历史 rawish 口径下 raw peak 最低、带通 RR 也好，
  但多个 seed 稳定反向（`band_limited_corr` 约 -0.80），是 polarity 盆地最硬的样本。
  注意它尚未在 20260620 soft-z 口径重跑，E0 需要先补跑同口径 3 seed 再纳入对照。

要求：

- 固定数据 seed。
- 至少 3 个训练 seed。
- 全部候选使用统一方向门控口径（见“固定实验口径 / 方向门控”）。
- 记录整体指标、按 seed 聚合指标和失败样本诊断图。

进入下一阶段条件：

- 确认 3 个主线 baseline 的稳定指标范围，以及 3 个反例各自的失败签名
  （loss/RR 背离、polarity 反向）。
- 确认 baseline 是否仍存在 polarity、低信噪比或节律变化失败模式。

### E1：STFT 输入信息增益验证

目的：验证“增加 STFT 输入是否有信息增益”，同时把 STFT 频带宽度作为独立实验变量。
BCG 中的心冲击信息可能携带心肺耦合线索，因此不应武断排除高频；但高频也更容易混入
体动、个体心率状态和非呼吸捷径，必须通过频带宽度消融判断。

输入设计：

- 时序分支：原始 BCG 时序。
- 时频分支：STFT log magnitude（参数见“固定实验口径 / STFT 默认参数”）。
- STFT 时频图的读图器（编码器）在 E1 内作为“零号消融”先于主线确定，只在两种通用结构间
  二选一：conv1d（频率→通道、仅沿时间卷）与小 conv2d（图像式、给频率局部感受野）。
  更强的频域专用结构（frequency-token mixer、可学习频带前端）属 E3，E1 不引入，以免提前
  把 E3 的变量混入“STFT 是否有信息增益”这个主问题。
- 频率下限固定 `0.05Hz`，频率上限作为实验变量，第一轮固定扫 `3/8/12Hz` 三档。
  该取值有当前数据实测频谱支撑（对 `bcg_rawish_wideband_state_aligned_segment_soft_z`
  整晚信号做频谱估计，频带重叠，占比之和大于 100%）：
  - `0.05-3Hz`：呼吸主带 + 心冲击主峰。实测 0.7-3Hz 心冲击带约占 46%，是能量主峰；
    0.05-0.7Hz 呼吸相关约占 31%。保守且覆盖主要生理信息。
  - `0.05-8Hz`：加入 3-8Hz 的更多心冲击/高频结构，实测约占 16%。
  - `0.05-12Hz`：覆盖到 8-12Hz，但实测 8-12Hz 仅约 2.4%，主要用于探边界，
    判断更宽频带是否只是引入噪声/捷径而非生理信息。
  - `0.05-20/25Hz`：实测 12Hz 以上不足 1%，只作为后续扩展，不作为第一轮默认主实验。
- 高频分支的解释应区分“心肺耦合增益”和“运动/个体状态捷径”。

候选对照：

- E1a：纯时序 baseline（直接用现有模型，不经双分支包装器，等同 E0 代码路径）。
- E1a'：容量桩（走双分支包装器但只启用时序分支，time_only 模式）。用于隔离“融合头容量”
  与“STFT 信息”：若 E1a'≈E1a，说明融合头本身不贡献容量增益，E1b 的增益才能干净归因到 STFT。
- E1b：时序 + STFT 单图输入，简单融合（concat）。
- E1c：仅 STFT 输入，用于判断时频分支自身是否有足够任务信息。
- E1d：同一模型结构下扫描 STFT 频带上限，比较 `3Hz`、`8Hz`、`12Hz`。

模型选择：

- E1 不应只绑定一个模型结构。至少选择一个 patch/mixer 类模型和一个多尺度低频类模型。
- 每个模型结构先跑最小 STFT 频带对照；只有出现稳定收益后，再扩大频带宽度或 seed 数。
- 不同模型结构之间只比较趋势，最终结论以同模型、同 seed、同频带变量消融为主。

关键判断：

- E1b 相对 E1a 是否改善主护栏。
- E1c 如果接近 E1b，说明模型可能主要依赖频谱包络；如果明显差，说明时序相位/形态仍必要。
  注意 STFT magnitude 已丢弃相位，该对照不能反推“时序相位本身是否有用”——它只能说明
  幅值谱是否够用。时序相位/低频极性的价值仍主要由时序分支承载，相位的专门验证留到
  E4/CWT 阶段（CWT 的连续相位/瞬时频率比固定窗 STFT 相位更稳，更适合做相位实验）。
- E1d 如果随频带加宽持续改善，需要进一步证明收益来自心肺耦合或鲁棒性，而不是个体/运动捷径。
- 如果 `12Hz` 或更宽频带改善训练 loss 但不改善跨 samp_id（leave-samp_id-out）验证指标，
  应优先怀疑捷径。该判据成立的前提是 E0 已确认 train/val samp_id 隔离。
- 如果 E1b 没有收益，不进入 CWT/SST，先回查 STFT 参数、归一化和融合位置。

### E2：手工分频带输入验证

目的：验证人工频带先验是否能帮助模型分离呼吸、谐波、心冲击和高频扰动。

不建议第一版使用完全硬切频带。优先使用重叠频带：

- `0.05-0.3Hz`：慢呼吸和主节律低端。
- `0.1-0.7Hz`：主要呼吸频带。
- `0.3-1.2Hz`：呼吸谐波和较快节律。
- `0.7-3Hz`：心冲击和呼吸外低频扰动。
- `3-8Hz` 或 `3-12Hz`：高频扰动/运动质量辅助。

保留但谨慎使用：

- `0-0.1Hz`：可能包含基线漂移和个体/设备状态信息。它可以用于诊断或质量分支，
  但不应在第一轮强力参与呼吸波形输出。

候选对照：

- E2a：时序 + 单 STFT 图。
- E2b：时序 + 重叠分频带图。
- E2c：时序 + 分频带能量序列，不使用完整 2D 图。
- E2d：Soft band decomposition，用可学习或软分配频带替代硬切频带，降低边界选择的任意性。

E2 主线只比较“固定重叠频带”（E2a/b/c）和“软频带”（E2d）两类。更激进的可学习频带
前端（MoE / SincConv / LEAF / frequency-domain mixer）统一移到 E3，不在本阶段引入，
避免一次性叠加过多自由度导致收益不可归因。

专业约束：

- E2a/b/c/d 之间只比较“频带先验形式”这一个变量，其它输入/融合/loss 口径保持一致。
- 软频带（E2d）相对固定频带（E2b/c）若有收益，需确认是“软边界本身有效”，
  而不是只多了一层可训练参数；必要时与固定频带在同 seed、同融合下逐一对照。

关键判断：

- 分频带是否提升低信噪比窗口和节律变化窗口，而不是只改善整体均值。
- 高频分支是否真的帮助模型识别扰动，还是引入噪声捷径。
- `0-0.1Hz` 加入后是否出现验证指标虚高或跨 samp_id 泛化变差。

### E3：可学习频带前端验证

进入条件：

- E2 已证明“固定/软频带先验”相对单 STFT 图有稳定收益。
- 需要进一步判断“手工/软频带是否过强”，即让模型自己学滤波器组能否做得更好。

候选前端（不宜同一批全部引入，按下列顺序逐一验证）：

- Multi-band Mixture-of-Experts：不同频带交给不同 expert，再由门控网络按窗口动态加权。
- SincConv / LEAF frontend：从原始时序学习类滤波器组，适合验证“手工频带是否过强”。
- frequency-domain MLP mixer：在频率 token 上做混合，验证频带间关系而非只做局部卷积。

专业约束：

- 这些方法都会显著增加自由度，每批最多引入一种，并保留 E2 最佳频带方案作同口径参照。
- MoE 需要特别关注门控塌缩：如果模型长期只选择单个 expert，说明频带专家没有形成有效分工，
  此时 MoE 的“收益”不可信。
- SincConv / LEAF frontend 的收益必须和固定滤波器组对照，否则无法区分“可学习滤波器有效”
  还是“只是多了一层可训练参数”。

关键判断：

- 可学习前端是否在 E2 最佳频带方案之上仍有稳定收益，还是只在训练集改善。
- 收益是否集中在低信噪比/节律变化窗口，而非只抬高整体均值。
- 自由度增加是否反而加重 polarity/相位歧义或跨 samp_id 泛化变差。

#### E3 已完成进展（2026-06-24）

当前 E3 实际推进顺序已经从原始“先有 E2 最佳频带再进入 E3”的理想路线，调整为先在
`conv2d fullband 8Hz + concat_generic + deep + fuse_len600` 这个强融合基线上做机制验证。
这个调整是可接受的，因为 E3-A0/B/C0 回答的是“当前 STFT 分支是否真的被用到、可学习前端是否
优于最强简单 STFT 基线、融合位置是否值得继续投入”，而不是直接替代 E2 的完整频带先验验证。

E3-A0 的结论：

- E3-A0.0 不是最早的 E1 简单 concat 基线，而是更强的融合基线：使用 `concat_generic`
  融合头、`deep` 头部、`fuse_len=600`，并以 `conv2d fullband 8Hz` 作为 STFT 分支。
- 该基线可以作为后续 E3/E5 的默认强参照，因为它已经排除了“只是早期融合头太弱”的部分疑问。
- 后续实验应继续保留同 seed 的 `time_only` substrate，但不需要为每个 dual arm 重复训练一份
  完全相同的 time-only；manifest 中记录复用关系即可。

E3-B 的结论：

- 已比较 `B0 conv2d fullband`、`B1 frequency-domain MLP mixer`、`B2 soft_band`，每组
  `time_only/dual` 小 seed 对照。
- 结果显示 dual STFT 相对 time-only 有小幅稳定收益，但 B1/B2 没有超过 B0。
- 因此当前不扩大 B1/B2；后续主参照继续使用 B0，即
  `conv2d fullband 8Hz + concat_generic + deep + fuse_len600`。
- B1/B2 暂时只作为“可学习频带前端未证明优于强简单基线”的负结果保留，不作为 E5 的默认输入。

E3-C0 的结论：

E3-C0 对 B0 dual top1 checkpoint 做输入扰动诊断。3 个 run 均完成，每个 mode 的 val 指标
均为 2675 个窗口。

| mode | relative_envelope_corr | relative_envelope_mae | band_limited_corr | best_lag_corr | rr_peak_band_abs_error | rr_peak_band_abs_error_p95 | rr_peak_band_abs_error_frac_gt_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 0.5069 | 0.2416 | 0.7547 | 0.8032 | 0.4808 | 1.8275 | 0.1168 |
| stft_zero | 0.4635 | 0.2618 | 0.7199 | 0.7707 | 0.4748 | 1.6884 | 0.1118 |
| stft_shuffle_batch | 0.4784 | 0.2482 | 0.7494 | 0.7986 | 0.4758 | 1.8653 | 0.1169 |
| stft_shuffle_time | 0.4589 | 0.2530 | 0.7077 | 0.7547 | 0.4991 | 1.9250 | 0.1231 |
| time_zero | 0.0261 | 0.4385 | -0.0001 | 0.0377 | 12.1587 | 17.0940 | 0.9991 |

相对 normal 的平均扰动：

- `time_zero` 几乎完全崩溃，说明当前模型仍主要依赖时序分支；STFT 分支单独不能完成重建。
- `stft_zero` 使 `relative_envelope_corr` 下降约 0.043、`band_limited_corr` 下降约 0.035、
  `best_lag_corr` 下降约 0.033，说明 STFT 分支不是无效旁路，确实贡献了波形相关性。
- `stft_shuffle_time` 伤害最大，`band_limited_corr` 和 `best_lag_corr` 分别下降约 0.047
  和 0.048，说明 STFT 特征的时间对齐比 batch 内样本身份更关键。
- `stft_shuffle_batch` 对 `band_limited_corr` 和 `best_lag_corr` 影响较小，但仍降低
  `relative_envelope_corr` 约 0.029；这提示 STFT 分支可能更多提供通用频谱上下文和时间对齐线索，
  而不是强样本特异的独立 RR 估计器。
- `rr_peak_band_abs_error` 上 `stft_zero` 均值和 p95 略优于 normal，说明 STFT 对波形相关性
  有帮助，但对 peak-band RR 指标不是单调正贡献。后续不能只按单一 RR peak 指标判断 STFT
  是否有效。

当前 E3-C0 支持的判断：

- “纯时序与时频图表达没有充分对齐导致收益受限”是合理怀疑，尤其由 `stft_shuffle_time`
  明显退化支持。
- 简单 concat 已能利用 STFT，但利用方式不够选择性；它改善相关性，却未稳定改善 peak-band RR。
- E5 的 gated fusion / 条件注入有研究价值，但应先做 E3-C1/C2 确认注入位置和分层贡献，
  否则直接进入更复杂融合会把“位置问题”和“门控机制问题”混在一起。

建议后续顺序：

1. E3-C1：固定 B0 输入和训练口径，比较 STFT 注入位置（early / mid / late / post-fusion），
   重点看 `stft_shuffle_time` 与 `stft_zero` 的退化曲线是否随位置变化。
2. E3-C2：补分层诊断，至少按 baseline 成功/失败窗口、`rr_peak_valid_ratio` 和
   `band_limited_corr` 档位检查 STFT 贡献，判断 STFT 是帮助难例还是只改善容易窗口。
3. E5：在 C1/C2 确认“哪个位置需要选择性 STFT 注入”后，再做 gated fusion。第一轮不建议
   直接上 cross-attention。

### E4：CWT / SST 候选验证

进入条件：

- E1 或 E2 已证明时频输入有稳定收益。
- 当前 STFT 参数敏感性成为主要限制。
- 需要更高时间分辨率来解释节律变化窗口。

候选顺序：

1. CWT：优先验证多尺度时频表示是否优于固定窗 STFT。
2. STFT/SST：在 STFT 有收益但脊线模糊时验证同步压缩。
3. CWT/SST：只在 CWT 有收益且计算成本可接受时进入。

风险：

- CWT/SST 增加大量自由度，可能导致收益不可归因。
- SST 对噪声和实现细节敏感，不适合作为第一批输入实验。
- 如果只改善可视化清晰度，不改善任务指标，不应继续扩大。
- 注意信息爆炸：CWT/SST 的尺度数、频率范围、时间分辨率、归一化和融合方式都可能成为变量。
  每批最多改变一个主变量，并保留 STFT 同口径参照。
- 如果 CWT/SST 需要更宽频带才能表现出收益，必须同步检查训练成本、显存和分层指标，
  防止因为表示更大而非生理信息更有效。

### E5：融合结构升级

进入条件：

- 简单融合已经证明时频输入有稳定收益。
- 分层诊断显示不同窗口需要动态选择时序或时频信息。

融合路线：

1. 简单 concat / projection 融合。
2. gated fusion：让时频分支作为对时序特征的条件修正。
3. token 级 cross-attention：让时序 token 查询时频 token。
4. finetune 路线：先训练纯时序或简单融合模型，再加入更复杂融合结构做低学习率微调。

默认推荐：

- 第一轮使用 gated fusion。
- cross-attention 只在 gated fusion 有收益但仍不足时引入。

不建议：

- 第一轮直接使用双向 cross-attention。
- 同时引入 CWT/SST、分频带、复杂 attention 和新 decoder。
- 在没有多 seed 证据前按单 run 选择注意力模型。
- 把 finetune 结果直接和从头训练结果混在一起排序；二者回答的问题不同，应分表记录。

## 训练策略

### Loss 口径

当前 loss 是 envelope/spectrum/smooth/high_freq/relative_envelope/phase_alignment/
signed_corr 等多项加权复合，不是单纯波形 MSE。为保证 val loss 跨实验可比、收益可归因，
固定以下口径：

- E0–E2 全程冻结现有主 loss 权重，不随输入分支调整。新增 STFT/分频带分支不得顺手
  改主权重，否则 val loss 跨实验失去可比性。
- val loss 仅作训练健康度参考，不作选模依据；选模仍以 `rr_peak_band_abs_error` 等
  护栏为准（见“模型选择指标”）。
- 时频分支若确需额外正则，只允许加“独立的轻正则”，且必须满足：
  - 正则只作用于新增分支或融合层输出，不修改既有主 loss 项。
  - 该正则的权重作为单独开关，必须做开/关消融并单独记录，不能和“STFT 是否有增益”
    这个主问题混在一起判断。
  - 建议优先用与现有口径一致的形式，降低新自由度：时频分支频谱一致性正则可复用
    `spectrum_low_hz/high_hz` 同口径的频带；分支输出平滑可复用 `smooth_weight`
    的曲率/差分形式。除非有明确证据，不引入全新结构的正则项。
- 若某个新正则确实带来稳定收益，应在 E2 收尾后作为独立消融条目固化，再决定是否
  写入默认配置；探索期不写入长期 loss 默认值。

### 训练顺序

从头训练：

- 作为每个新结构的第一基线。
- 优点是对照干净，不依赖旧模型盆地。
- 缺点是训练成本更高。

加载纯时序 checkpoint 并低学习率微调：

- 作为第二阶段对照。
- time branch 使用较低学习率，时频分支和融合层使用正常学习率。
- 用于判断新增时频分支是否能在已有时序解上提供增益。

完全冻结时序主干：

- 不作为默认策略。
- 仅用于验证“时频分支是否能作为外挂修正器”。
- 风险是融合层无法纠正原时序模型的错误盆地。

建议顺序：

1. 每个候选先从头训练。
2. 只有候选从头训练接近或超过 baseline，再做 checkpoint 初始化对照。
3. 对有潜力的候选补充 finetune：纯时序 checkpoint 初始化 time branch，新增时频/融合层正常学习。
4. 冻结实验只作为诊断，不作为主线选模依据。

## 分层诊断

整体均值不足以判断时频输入是否有价值。后续每批正式实验应尽量补充分层分析：

- 按 `allowed_losses` 或质量标记分层。
- 按目标呼吸率分层：慢呼吸、正常呼吸、较快呼吸。
- 按低频形态分层：单峰呼吸、低频双峰/多峰呼吸、峰谷不清窗口。
- 按 input/target 运动支配程度分层。
- 按 BCG 信噪比分层：高 SNR、中等 SNR、低 SNR 或低置信 BCG 窗口。
- 按 baseline 误差分层：纯时序模型成功窗口和失败窗口分别比较。
- 按 polarity 诊断分层：正向、反向、低相关窗口分别比较。

分层定量口径（第一轮复用现有指标，不引入新依赖）：

- 频谱质量分层：用 `spectrum_similarity`，按当前 baseline 在 val 上的三分位
  （p33/p66）切高/中/低三档。三分位边界以 E0 baseline 的 val 分布为准并固定下来，
  后续 E1/E2 沿用同一边界，保证跨实验可比。
- polarity 分层：用 `band_limited_corr` 符号与幅度切档——`>0.2` 正向、`<-0.2` 反向、
  `[-0.2, 0.2]` 低相关。阈值 0.2 是诊断门槛，可在 E0 观察分布后微调一次并固定。
- 坏段/置信分层：用 `rr_peak_valid_ratio`，按 `>=0.9` / `[0.6,0.9)` / `<0.6`
  切高/中/低置信三档。
- BCG 信噪比分层：第一轮先用上面的频谱质量档和 valid_ratio 档作为 SNR 代理，
  不单独实现 SNR 估计；待 STFT 证明有增益后，再投入实现“呼吸带能量 / 带外能量”
  形式的显式 BCG-SNR，作为更精确的 SNR 分层依据。
- 分层最小样本数：每个档位至少保留约 50 个窗口才纳入分层结论，样本不足的档位只标注
  样本量、不下定量结论，避免把噪声当信号。

需要重点看：

- 时频输入是否只改善已经容易的窗口。
- 分频带是否改善低信噪比窗口。
- 对低频双峰呼吸，模型是否错误地把二次峰当成主呼吸峰，导致 `rr_peak_band_abs_error` 偏高。
- 对低信噪比 BCG，时频输入是否提升呼吸主频稳定性，还是只放大心跳/运动频带。
- 高频分支是否帮助识别体动，还是导致模型过拟合干扰。
- 新模型是否降低 polarity 失败率，而不是只在反向解中获得更稳定峰间距。

## 记录模板

每一批实验完成后，在对应实验记录文档中至少记录：

- 实验标签。
- 数据口径。
- 输入表示。
- 模型结构类别。
- 融合方式。
- 训练 seed。
- run root。
- best epoch。
- best val loss。
- `rr_peak_band_abs_error` mean / median。
- `rr_spec_abs_error` mean / median。
- `relative_envelope_mae` mean。
- `relative_envelope_corr` mean。
- `band_limited_corr` mean。
- `best_lag_corr` mean。
- 结论：通过、失败、仅诊断保留或需要补 seed。

## 推荐第一批实验

第一批只回答一个问题：STFT 输入是否有稳定信息增益。

推荐最小实验组：

| label | 输入 | 融合 | 目的 |
|---|---|---|---|
| `E1a_timeseries_baseline` | 原始时序 | 无 | 同口径纯时序对照 |
| `E1b_time_stft_simple` | 时序 + `0.05-3Hz` STFT | 简单融合 | 验证 STFT 信息增益 |
| `E1c_stft_only` | `0.05-3Hz` STFT | 无时序分支 | 判断时频输入单独可用性 |
| `E1d_time_stft_bandwidth` | 时序 + `0.05-8/12Hz` STFT | 同 E1b | 验证高频心冲击/扰动信息是否有增益 |

第一批模型不要求覆盖所有结构，但至少应包含两个代表性结构，避免 STFT 结论绑定单一模型：

- 一个 patch/mixer 类纯时序主干。
- 一个多尺度/低频分解类主干。

如果 E1b 未超过 E1a，不进入 E2 及后续阶段，先分析失败样本、STFT 参数、频带宽度和融合位置。

如果 E1b 超过 E1a，第二批进入重叠分频带 E2。

如果 E1d 相对 E1b 改善，下一步优先做频带宽度和高频诊断消融，而不是直接上 SST。

如果 E1b/E1d 只在低信噪比窗口改善但整体均值不明显，可保留为质量条件分支方向，
再设计 gated fusion 或 finetune。

## 当前建议

早期 E0/E1 的核心作用是建立纯时序参照、确认 STFT 输入有可重复信息增益。当前已经有
E3-A0/B/C0 证据表明：

- `conv2d fullband 8Hz + concat_generic + deep + fuse_len600` 是当前最强、最清楚的 STFT
  融合参照。
- 可学习前端 B1/B2 暂时没有超过 B0，不应继续扩 seed。
- STFT 分支被模型实际使用，尤其依赖时间对齐；但它对 peak-band RR 不是单调正贡献。

因此下一步不建议立刻扩大可学习前端，也不建议直接进入 E5 的复杂 gated/cross-attention。
更合理的顺序是：

1. E3-C1：做融合/注入位置消融，固定 B0 输入、训练和评价口径，只改变 STFT 特征进入主干的位置。
2. E3-C2：做分层贡献诊断，确认 STFT 到底改善哪些窗口，尤其是 baseline 失败窗口、低置信窗口
   和低频相关弱窗口。
3. E5：如果 C1/C2 显示 STFT 贡献集中在特定窗口或特定位置，再做 gated fusion，让模型按窗口
   选择是否注入 STFT；cross-attention 继续后置。

高频心冲击信息仍值得验证，尤其可能反映心肺耦合；但专业上不能默认更宽频带一定更好。
实测 8Hz 以上能量已很低，高频更容易引入个体心率、运动状态和设备噪声捷径。后续任何更宽频带、
CWT/SST 或更复杂融合，都应继续保留 B0 强基线和同 seed/time-only substrate 对照。
