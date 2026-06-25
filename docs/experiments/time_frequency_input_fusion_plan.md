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

## 当前结果总览（2026-06-24）

E 系列当前已经从“时频输入是否值得继续”推进到“哪种融合结构值得收口”。稳定结论如下：

| 阶段 | 已完成口径 | 结果摘要 | 当前结论 |
|---|---|---|---|
| E0 | 20260620 soft-z 纯时序候选，6 类结构各 3 seed | `multiscale_decomp_mixer1d`、`patch_mixer1d`、`patch_hann_bandlimited_output1d` 的 `rr_peak_band_abs_error` 均值分别约 `0.483`、`0.489`、`0.497`；`polyphase_patch_hann_bandlimited1d` val loss 最低但 peak-band RR 约 `0.652` | E0 证实“低 val loss 不等于呼吸任务好”；后续 baseline 以 patch/multiscale 主线解释，polyphase/多尺度 hann 作为 loss/RR 背离反例 |
| E1 | `patch_mixer1d + STFT`，clean wrapper、8Hz、补到 9 个成功配对，并做 3Hz/12Hz 稳健性复核 | 8Hz dual 相对 time-only：peak-band mean 9/9 改善，平均 `-0.120`；median 9/9 改善，平均 `-0.033`；`frac_gt_1` 9/9 改善，平均 `-2.34pp`；`frac_gt_2` 8/9 改善，平均 `-1.87pp` | patch 路线 STFT 信息增益成立，主要是降低长尾谐波误拣；`stft_only` 不成立，multiscale wrapper 仍受优化多稳态限制 |
| E1-D | 轻 polarity warmup，6 个代表性 seed | dual 相对 time-only 的 peak-band mean 平均 `-0.117`、median 平均 `-0.050`、`frac_gt_1` 平均 `-2.23pp`、`frac_gt_2` 平均 `-1.38pp` | 轻 warmup 修复极性稳定性时不伤 STFT 净增益，可作为 patch STFT 默认训练配方 |
| E2b | 重叠分频带图 vs B2 强 STFT 基线，6 seed 配对 | peak-band mean 平均 `-0.0265`，`frac_gt_1` 平均 `-0.60pp`，但仅 4/6 seed 同向改善 | 分频带图有小幅正信号，但不足以升为默认主线 |
| E2c | 分频带能量序列 vs B2 强 STFT 基线，6 seed 配对 | 前 4 个 seed 多数改善，但后 2 个 seed 明显恶化；总体 peak-band mean 平均 `+0.0167`、`frac_gt_1` 平均 `+0.50pp` | band-energy 不是稳定主线，暂作负/混合结果 |
| E3 | B0/B1/B2 前端、C0/C1/C2 注入位置和分层诊断 | B0 `conv2d fullband 8Hz + concat_generic + deep + fuse_len600` 是强简单基线；B1 frequency MLP 和 B2 soft band 未超过 B0；C1B `token_pre_mixer` peak-band RR 最稳；C2 显示 STFT 收益主要来自 baseline hard 和低 `spectrum_similarity` 窗口 | 当前主线转为 `native_inject pre_mixer`；`token_mid_mixer` 仅作为 rr_spec/相关性诊断候选 |
| E4 | SST 8Hz vs B2，4 seed 配对，另做少量 time-frequency separation 诊断 | SST 相对 B2 的 peak-band mean 平均 `+0.0004`，median `-0.0002`，p95 `+0.0074`；相关性 delta 约 `1e-4` 量级 | SST 可视化/分离度在少数 hard 窗口更清晰，但任务指标无收益；不进入主线 |
| E5-A0 | gated native pre-mixer，3 seed | gated 相对 ungated：`rr_peak_band_abs_error` 从 `0.4907` 到 `0.5119`，`rr_spec_abs_error` 从 `0.5679` 到 `0.5589`，呼吸次数误差从 `1.0652` 到 `1.1087` | gate 只带来很小 spec/相关性收益，peak-band 和计数变差，不升默认 |
| E5-A1/A2 | cross-attention 从头训练与 warm-start，3 seed | A1.1 相对 A1.0：peak-band `+0.1600`、计数 `+0.1441`，仅 spec `-0.0108`；A2 warm-start 后 peak-band 仍 `+0.1433`、计数 `+0.1713` | cross-attention 重复“频谱小改善、peak-band/计数变差”的失败模式；不建议继续扩 seed 或做 freeze/unfreeze |

当前可收口的默认研究主线是：`patch_mixer1d + conv2d fullband 8Hz + native_inject pre_mixer`，
配对 `native_time_only` 解释收益；`concat_generic + deep + fuse_len600` 保留为强简单 STFT 参照。
后续若继续投入，应优先围绕 hard/low-spectrum 窗口做更有约束的条件注入或损失/判据研究，
而不是继续扩大可学习频带、SST 或 cross-attention。

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

#### E0 已完成结果（2026-06-20 soft-z）

E0 的 20260620 soft-z 候选重跑已经完成 6 类模型、每类 3 seed。汇总文件为
`runs/tho_research_v2_20260620_softz_model_candidates_summary.csv`。

| model | n | `rr_peak_band_abs_error` | `rr_spec_abs_error` | `relative_envelope_mae` | `band_limited_corr` | `best_lag_corr` | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `multiscale_decomp_mixer1d` | 3 | 0.4834 | 0.5449 | 0.2406 | 0.7859 | 0.8395 | 主线 baseline，RR peak-band 稳定 |
| `patch_mixer1d` | 3 | 0.4892 | 0.5885 | 0.2311 | 0.7913 | 0.8426 | 主线 baseline，后续 STFT patch 路线的核心参照 |
| `patch_hann_bandlimited_output1d` | 3 | 0.4969 | 0.5985 | 0.2363 | 0.7877 | 0.8393 | 可用 baseline，但不是后续时频主线 |
| `period_aware_patch_hann_bandlimited1d` | 3 | 0.5122 | 0.5628 | 0.2472 | 0.7783 | 0.8311 | 非主线，仅保留为结构参考 |
| `multiscale_patch_hann_bandlimited1d` | 3 | 0.5807 | 0.5450 | 0.2375 | 0.7803 | 0.8343 | loss/RR 背离反例 |
| `polyphase_patch_hann_bandlimited1d` | 3 | 0.6518 | 0.5279 | 0.2321 | 0.7937 | 0.8462 | val loss 和 spec 较好，但 peak-band RR 明显差，是最典型反例 |

E0 支持的阶段结论是：后续不能按 val loss 排序选模，必须继续以
`rr_peak_band_abs_error` 和长尾谐波误拣为主护栏；`patch_mixer1d` 与
`multiscale_decomp_mixer1d` 是解释 STFT 增益的主要纯时序参照。

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

#### E1 已完成结果（2026-06-22）

详细记录见 `docs/experiments/e1_stft_info_gain_20260622.md`。最终结论是：

- `patch_mixer1d` 路线 STFT 信息增益通过。clean wrapper 口径下，
  `patch_mixer1d + conv2d + N0 8Hz + deep + fuse_len=600` 在 9 个成功配对中
  peak-band mean、median、`frac_gt_1` 均 9/9 改善，`frac_gt_2` 8/9 改善。
- 8Hz 是当前甜点。3Hz/12Hz 复核仍有正信号，但 12Hz 没有优于 8Hz，支持“8Hz 以上能量低，
  更宽频带未必更好”的判断。
- `stft_only` 在当前 magnitude STFT、loss 和方向门控下不成立。它不能替代时序分支，
  STFT 的价值更像是降低谐波误拣和提供时间对齐频谱上下文。
- `multiscale_decomp_mixer1d` 路线不能把初期崩坏归咎于 STFT。E1a' time-only wrapper
  已经明显抬高误拣长尾，问题发生在 STFT 进入之前；后续只可作为 wrapper 稳定性诊断，
  不作为 STFT 无效证据。
- E1-D 的轻 polarity warmup 在 6 个代表性 seed 上保住 STFT 净增益，因此可进入后续 patch
  STFT 默认训练配方。

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

#### E2 已完成结果（2026-06-23）

E2 没有完整替代 E1/B2 主线，只完成了两类针对 B2 强 STFT 基线的 6 seed 配对探针：

| arm | 对照 | `delta_mean` | `delta_frac_gt_1` | `delta_frac_gt_2` | seed 方向 | 结论 |
|---|---|---:|---:|---:|---|---|
| E2b 重叠分频带图 | B2 native dual 8Hz | -0.0265 | -0.60pp | -0.42pp | 4/6 改善 | 有小幅正信号，但不足以升为默认主线 |
| E2c 分频带能量序列 | B2 native dual 8Hz | +0.0167 | +0.50pp | +0.49pp | 前 4 个 seed 多数改善，后 2 个 seed 明显恶化 | 不稳定，暂作负/混合结果 |

当前判断：

- 固定重叠分频带图可能略微降低长尾误拣，但收益小于 E1/E3 中确认的 STFT 主效应，
  且 seed 稳定性不足。
- 分频带能量序列丢掉了过多时频局部结构，不能替代 2D STFT 图。
- E2 不支持继续扩大手工频带或把 `0-0.1Hz` 质量/漂移分支加入主线；如需继续，
  应先明确它要解决 C2 中的 hard/low-spectrum 窗口，而不是追求整体均值小幅变化。

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

E3-C1 的结论：

E3-C1 固定 B0 输入口径（`conv2d fullband 8Hz`、`patch_mixer1d`、3 个探索 seed），
比较 `concat_generic` 后融合和 `native_inject` token 级注入位置。由于 `concat_generic`
和 `native_inject` 不是同一 substrate，C1 只按各自配对的 time-only 解释，不直接横向比较
绝对分数。

| arm | paired substrate | rr_peak_band_abs_error delta | rr_spec_abs_error delta | relative_envelope_corr delta | band_limited_corr delta | best_lag_corr delta | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| C1A concat post-fusion | C1S concat time-only | -0.1146 ± 0.1086 | -0.0122 ± 0.0184 | +0.0219 ± 0.0131 | +0.0243 ± 0.0066 | +0.0281 ± 0.0026 | STFT 明确有增益，与 B0/C0 连续 |
| C1B token pre-mixer | C1T native time-only | -0.0507 ± 0.0395 | -0.0155 ± 0.0216 | +0.0082 ± 0.0024 | +0.0017 ± 0.0003 | +0.0013 ± 0.0008 | 主候选，3/3 seed 的 peak-band RR 改善 |
| C1C token mid-mixer | C1T native time-only | -0.0037 ± 0.0603 | -0.0495 ± 0.0186 | +0.0119 ± 0.0024 | +0.0030 ± 0.0018 | +0.0036 ± 0.0013 | 相关性和 rr_spec 候选，但 peak-band RR 不稳 |
| C1D token post-mixer | C1T native time-only | +0.0027 ± 0.0558 | +0.0131 ± 0.0310 | -0.0019 ± 0.0056 | -0.0020 ± 0.0019 | -0.0006 ± 0.0013 | 暂停，不作为后续主线 |

C1 的主要发现：

- `native_inject` substrate 本身明显强于 concat substrate，因此后续所有 native 注入实验必须继续
  配对 `native_time_only`，不能只和 concat/B0 baseline 比绝对分数。
- `token_pre_mixer` 是当前最稳的注入位置：主护栏 `rr_peak_band_abs_error` 在 3 个 seed 上均改善，
  适合进入下一步机制诊断。
- `token_mid_mixer` 对 `rr_spec_abs_error`、`relative_envelope_corr`、`best_lag_corr` 更有吸引力，
  但 peak-band RR 不稳定；它应作为 C2 诊断候选，而不是立即替代 C1B。
- `token_post_mixer` 基本无收益，说明把 STFT 作为解码前的晚期 token 修正不足以稳定利用时频信息。
- C1 当前不需要先补 topK：各 run 的 top3 val loss spread 最大约 0.0045（不到 0.75%），且
  E3-B 的 topK 复核没有推翻 B0/B1/B2 的方向。topK 可作为最终收口复核，但不是进入 C2 的前置条件。

E3-C2 要回答的问题：

C2 不是继续扩 seed 或按整体均值重新选模型，而是做“分层贡献诊断”。它要判断 STFT 的收益来自哪些
窗口，以及 C1B/C1C 的差异是否对应不同失败模式。

第一轮 C2 范围：

- 必选：`C1B token_pre_mixer`、`C1C token_mid_mixer`、`C1T native_time_only`。
- 可选：`C1A concat_post_fusion`，用于和 B0/C0 保持连续性。
- 暂不纳入：`C1D token_post_mixer`，除非后续需要负对照。

C2 的两类诊断：

- 输入扰动：对 native 注入路径补 `stft_zero`、`stft_shuffle_time`、`stft_shuffle_batch`，
  在 STFT 投影后的 token delta 上做扰动，并保持原注入位置不变，检查 C1B/C1C 是否真的
  依赖 STFT，以及依赖是否仍主要来自时间对齐。`time_zero` 只作为额外负对照，不进入
  C2 主结论。
- 分层汇总：按 baseline 成功/失败窗口、`rr_peak_valid_ratio`、`band_limited_corr`、
  `spectrum_similarity` 和 target RR 档位，比较 dual - time-only 或 normal - ablated 的 delta。

C2 完成后再决定是否进入 E5 gated fusion：如果 STFT 只在特定窗口或特定失败模式中有收益，
E5 应优先做 gated/conditional fusion；如果 C2 显示收益均匀且 pre-mixer 稳定，则可以先收敛到
更简单的 native pre-mixer 主线，不急于上 cross-attention。

E3-C2 的结论：

C2 已完成 `C1B token_pre_mixer`、`C1C token_mid_mixer` 和配对
`C1T native_time_only` 的 3 seed 分层诊断。未纳入 `C1A concat_post_fusion` continuity，
因为当前问题是 native 注入路径中 STFT token 对不同窗口的贡献；C1A 只在需要把 C2
和 C0/B0 的 concat 消融曲线写成连续叙事时再补，不是进入 E5 的前置条件。

执行口径：

- 消融指标文件：`runs/e3_c1/*/metrics_e3c_{normal,stft_zero,stft_shuffle_time,stft_shuffle_batch}_best.csv`。
- 分层汇总文件：`runs/e3_c2/dual_minus_time_only_by_layer.csv` 和
  `runs/e3_c2/normal_minus_ablation_by_layer.csv`。
- `mean_delta = candidate - reference`；error 类指标为负更好，corr 类指标为正更好。
- 分层边界由同 seed 的 `C1T native_time_only` 固定，再对 dual 或 ablated metrics 做
  `dataset_row_id` 同窗口 join。

全局结果：

| arm | rr_peak_band_abs_error delta | rr_spec_abs_error delta | relative_envelope_corr delta | band_limited_corr delta | best_lag_corr delta | 判断 |
|---|---:|---:|---:|---:|---:|---|
| C1B dual - time_only | -0.0507 | -0.0155 | +0.0082 | +0.0017 | +0.0013 | 主线，peak-band RR 最稳 |
| C1C dual - time_only | -0.0037 | -0.0495 | +0.0119 | +0.0030 | +0.0036 | rr_spec 和相关性更强，但 peak-band RR 不稳 |
| C1B normal - stft_zero | +0.0083 | -0.0593 | +0.0492 | +0.0165 | +0.0116 | STFT 明确改善相关性和 rr_spec，但 peak-band 不单调 |
| C1C normal - stft_zero | +0.0411 | -0.0504 | +0.0728 | +0.0144 | +0.0108 | STFT 依赖更强，也更容易带偏 peak-band |

主要分层发现：

- dual 相对 time-only 的收益主要集中在 baseline peak-band 失败窗。C1B 在失败窗
  `rr_peak_band_abs_error` 为 `-0.4802`，成功窗反而为 `+0.0180`；C1C 在失败窗为
  `-0.2413`，成功窗为 `+0.0359`。
- STFT token 对同一 checkpoint 的 `rr_spec_abs_error`、`relative_envelope_corr`、
  `band_limited_corr` 和 `best_lag_corr` 是稳定正贡献；但对 `rr_peak_band_abs_error`
  不是单调正贡献，尤其 `stft_zero` 消融下 C1B/C1C 的 peak-band delta 分别为
  `+0.0083` 和 `+0.0411`。
- 低 `spectrum_similarity` 窗口中 STFT 贡献最大，尤其体现在 `rr_spec_abs_error`
  和相关性指标；高 `spectrum_similarity` 窗口收益很小，甚至有轻微负效应。
- target RR 的 fast 档没有稳定收益。C1B 在 fast 档 `rr_peak_band_abs_error` 为
  `+0.0092`、`rr_spec_abs_error` 为 `+0.0167`，不适合作为强注入目标。
- `band_limited_corr` 的 negative/low-corr 档样本过少，大量分层未过 50 窗口门槛，
  因此 C2 不支持对 polarity 失败率下结论。

C2 后续决策：

- 保留 `C1B token_pre_mixer` 为当前 native STFT 主线。
- 保留 `C1C token_mid_mixer` 作为 rr_spec/相关性诊断候选，但不替代 C1B。
- 下一步优先进入 E5 的 gated/conditional STFT，而不是直接上 cross-attention：
  门控应重点让 STFT 作用在 baseline hard、低 spectrum-similarity 和正常 RR 窗口，
  并抑制 easy 或 fast RR 窗口中的过强注入。

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

#### E4-SST 已完成结果（2026-06-23）

E4 首轮只验证 SST 8Hz 是否能在 B2 强 STFT 参照上带来任务收益。执行口径：

- 训练结果：`runs/e4_sst_grouped.csv`、`runs/e4_sst_vs_b2_paired_delta.csv`。
- 相关性配对：`runs/e4_sst_vs_b2_corr_paired_delta.csv`。
- 少量表示分离诊断：`runs/e4_timefreq_compare/separation_summary.csv` 和
  `runs/e4_timefreq_compare_8hz/separation_summary.csv`。

4 个同 seed 配对中，SST 相对 B2 的主指标变化极小：

| metric | 平均 delta（SST - B2） | seed 方向 | 判断 |
|---|---:|---|---|
| `rr_peak_band_abs_error` mean | +0.0004 | 1/4 改善 | 无任务收益 |
| `rr_peak_band_abs_error` median | -0.0002 | 2/4 改善 | 基本持平 |
| `rr_peak_band_abs_error` p95 | +0.0074 | 2/4 改善 | 长尾无改善 |
| `frac_gt_1` | 约 0.00pp | 2/4 改善 | 误拣率无改善 |
| `frac_gt_2` | +0.04pp | 0/4 改善 | 严重误拣略差 |

相关性指标的 delta 也只有 `1e-4` 量级，不能解释为稳定收益。少量 hard 窗口的 time-frequency
separation 诊断显示 SST 表示有时更清晰，但这种可视化优势没有转化为任务指标。

因此 E4-SST 只作为诊断结果保留，不进入当前主线；后续不建议继续扩大 SST seed 或直接进入
CWT/SST 结构搜索，除非先定义新的、能解释 hard 窗口失败的表示判据。

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

E5-A0 完成后需要修正上述默认顺序：轻量 gated native pre-mixer 对 `rr_spec_abs_error` 和相关性
有小幅收益，但 `rr_peak_band_abs_error`、呼吸次数误差和 seed 稳定性不足。因此后续不是把
gated fusion 升为默认主线，而是把它作为负/混合结果保留；如果继续验证 cross-attention，
必须拆成两个独立问题，避免把结构收益和预训练/优化策略混在一起。

#### E5-A0：gated native pre-mixer 首轮探针

E5-A0 从 E3-C2 的结论出发，只验证一个问题：STFT token delta 是否应该被逐窗口、逐通道选择性
使用。它不是 E5 的全量融合搜索，也不引入新的时频前端、loss、训练 seed 或注入位置。

固定口径：

- 数据、loss、epoch、batch、checkpoint gate、STFT 参数和 3 个探索 seed 全部沿用 E3-C1/C2。
- 输入仍为 `conv2d fullband 8Hz + n0`，主干仍为 `patch_mixer1d`。
- 注入位置固定为 `pre_mixer`，因为 C1B 是当前 peak-band RR 最稳的 native STFT 主线。
- 继续配对同 seed 的 native time-only substrate，禁止把 native substrate 本身的强度解释为 gate 收益。

候选：

| arm | branch_mode | fusion_mode | paired substrate | 目的 |
|---|---|---|---|---|
| `E5-A0T_native_time_only` | `time_only` | `native_inject` | 自身 | native substrate 参照 |
| `E5-A0.0_native_pre_mixer_ungated` | `dual` | `native_inject` | `E5-A0T_native_time_only` | 复现 C1B ungated pre-mixer |
| `E5-A0.1_gated_native_pre_mixer` | `dual` | `gated_native_inject` | `E5-A0T_native_time_only` | 只在 STFT token delta 上加轻量 gate |

实现约束：

- `gated_native_inject` 仍使用 ReZero 式 `stft_proj` 零初始化，初始输出等价 time-only。
- gate 为逐 token/channel 的 `1x1 Conv1d + sigmoid`，输入为 STFT encoder feature；
  bias 初始化为 `2.0`，即初始 gate 约 `0.88`，尽量接近 ungated C1B，同时允许训练下调。
- 第一轮不记录 gate 分布为正式结论；如果 A0.1 有收益，再补 gate 分布和 C2 同口径分层消融。

通过条件：

- A0.1 相对 A0.0 或 paired time-only 至少不恶化主护栏 `rr_peak_band_abs_error`。
- A0.1 应降低 C2 暴露的 easy/fast 窗口副作用，尤其不能继续扩大 fast RR 档误差。
- 低 `spectrum_similarity` 或 baseline hard 窗口中，相关性和 `rr_spec_abs_error` 收益不应被 gate 抹掉。

停止条件：

- A0.1 只改善相关性但明显恶化 peak-band RR。
- A0.1 与 A0.0 差异小到不能解释为选择性融合收益。
- A0.1 的收益只出现在单 seed，或分层结果与 C2 的 hard/low-spectrum 证据不一致。

运行入口：

- 编排脚本：`scripts/run_e5_a0_gated_fusion_probe.py`
- manifest：`runs/e5_a0_manifest.csv`
- run root：`runs/e5_a0/<arm>/<branch_mode>/`

结果（2026-06-24，3 seeds，2675 val windows/run）：

| arm | `rr_peak_band_abs_error` | `rr_spec_abs_error` | `breath_count_zero_cross_abs_error` | `relative_envelope_corr` | `band_limited_corr` | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `E5-A0T_native_time_only` | 0.5410 | 0.5836 | 1.0556 | 0.5175 | 0.7913 | native substrate |
| `E5-A0.0_native_pre_mixer_ungated` | 0.4907 | 0.5679 | 1.0652 | 0.5257 | 0.7929 | 复现 C1B，仍是当前主线 |
| `E5-A0.1_gated_native_pre_mixer` | 0.5119 | 0.5589 | 1.1087 | 0.5277 | 0.7936 | spec/相关性小幅改善，但 peak-band 与计数变差 |

阶段判断：

- A0.1 相对 A0.0 的 `rr_spec_abs_error` 降低约 `0.009`，相关性略升，但
  `rr_peak_band_abs_error` 增加约 `0.021`，呼吸次数误差增加约 `0.044`。
- 这与 C2 暴露的问题一致：STFT 对频谱和相关性有贡献，但选择性注入没有自动消除 peak-band
  副作用。
- E5-A0 不通过“升默认”条件；gated native pre-mixer 仅作为负/混合结果保留。

#### E5-A1：cross-attention 从头训练

E5-A1 只回答“token 级 cross-attention 结构本身是否比简单 token delta 注入更有效”。它从头训练，
不使用 time-only checkpoint warm-start，不使用不同参数组学习率，不引入新 loss 或新 STFT 前端。

候选：

| arm | branch_mode | fusion_mode | paired substrate | 目的 |
|---|---|---|---|---|
| `E5-A1T_native_time_only` | `time_only` | `native_inject` | 自身 | native substrate 参照 |
| `E5-A1.0_native_pre_mixer_ungated` | `dual` | `native_inject` | `E5-A1T_native_time_only` | C1B/E5-A0 ungated 主线复现 |
| `E5-A1.1_cross_attention_pre_mixer` | `dual` | `cross_attention_inject` | `E5-A1T_native_time_only` | time token 查询 STFT token |

结构约束：

- time token 作为 query，STFT token 作为 key/value；输出投影后作为 pre-mixer token delta。
- 第一轮只用 1 层、2 heads，避免把 A1 变成大模型容量实验。
- cross-attention 输出投影零初始化，初始输出等价 time-only/native backbone；这与 C1B 的 ReZero
  注入保持可比。
- 不做双向 cross-attention，不加 hard mask，不加显式 fast/easy 规则门控。

实现：

- 模型分支：`fusion_mode=cross_attention_inject`，配置项 `model.cross_attention_heads=2`。
- 编排脚本：`scripts/run_e5_a1_cross_attention_probe.py`
- manifest：`runs/e5_a1_manifest.csv`
- run root：`runs/e5_a1/<arm>/<branch_mode>/`

通过条件：

- A1.1 相对 A1.0 不能恶化 `rr_peak_band_abs_error` 和 `breath_count_zero_cross_abs_error`。
- A1.1 若主要改善 `rr_spec_abs_error`、相关性或 val loss，但 peak-band/呼吸次数变差，
  视为重复 E5-A0 失败模式，不进入 A2。
- 如果 A1.1 至少接近 A1.0，并在低 `spectrum_similarity` 或 baseline hard 窗口有稳定收益，
  再进入 E5-A2。

#### E5-A2：cross-attention warm-start 与分组学习率

E5-A2 只在 E5-A1 不差于 ungated 主线后进入。它回答的是“cross-attention 能否在已有时序解上
做条件修正”，不再回答 cross-attention 结构本身是否有效。

候选：

| arm | 初始化 | 学习率策略 | 目的 |
|---|---|---|---|
| `E5-A2.0_cross_attention_warm_start` | 同 seed native time-only checkpoint 初始化 time backbone | time backbone 低学习率，STFT encoder 和 cross-attention 正常学习率 | 低扰动条件修正 |
| `E5-A2.1_cross_attention_warm_start_freeze_probe` | 同上 | 短程冻结 time backbone，仅训练 STFT/cross-attention，再整体解冻 | 后续可选诊断；当前不进入首轮 runner |

实现约束：

- A2 需要支持参数组学习率和 checkpoint 部分加载；若后续追加冻结/解冻调度且会让
  `scripts/train_tho_small.py` 或通用 `ThoExperiment` 过重，应单独实现 E5-A2 训练脚本或轻量
  experiment wrapper，不强行污染当前通用训练入口。
- A2 的结果必须和 A1 从头训练分表记录；不能把 warm-start 结果直接混进从头训练排名。
- A2 不应改变数据 split、STFT 参数、loss 权重或评价指标；唯一新增自由度是初始化和优化策略。

当前实现：

- 首轮只实现 `E5-A2.0_cross_attention_warm_start`。
- 训练入口：`scripts/train_e5_a2_tho.py`，内部使用 `ThoE5A2Experiment`。
- 编排脚本：`scripts/run_e5_a2_cross_attention_warm_start_probe.py`
- 默认 warm-start root：`runs/e5_a1/e5_a1t_native_time_only/time_only`
- 分组学习率：`training.time_backbone_learning_rate=0.0001`，
  `training.learning_rate=0.001` 用于 STFT encoder 和 cross-attention。
- warm-start 只加载 `training.warm_start_prefixes=[time_backbone.]`，cross-attention 和 STFT encoder
  保持新初始化。
- manifest：`runs/e5_a2_manifest.csv`
- run root：`runs/e5_a2/e5_a2_0_cross_attention_warm_start/dual/`

结果（2026-06-24，3 seeds，2675 val windows/run）：

| arm | `rr_peak_band_abs_error` | `rr_spec_abs_error` | `breath_count_zero_cross_abs_error` | `relative_envelope_corr` | `band_limited_corr` | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `E5-A1T_native_time_only` | 0.5411 | 0.5836 | 1.0555 | 0.5175 | 0.7913 | native substrate |
| `E5-A1.0_native_pre_mixer_ungated` | 0.4905 | 0.5679 | 1.0639 | 0.5257 | 0.7929 | 当前主线对照 |
| `E5-A1.1_cross_attention_pre_mixer` | 0.6505 | 0.5571 | 1.2080 | 0.5206 | 0.7894 | peak-band 与呼吸次数明显退化 |
| `E5-A2.0_cross_attention_warm_start` | 0.6338 | 0.5821 | 1.2353 | 0.5214 | 0.7900 | warm-start/分组 LR 未修复退化 |

配对差异：

- E5-A1.1 相对 E5-A1.0：`rr_peak_band_abs_error +0.1600`，
  `rr_spec_abs_error -0.0108`，`breath_count_zero_cross_abs_error +0.1441`，
  `relative_envelope_corr -0.0052`，`band_limited_corr -0.0035`。
- E5-A2.0 相对 E5-A1.0：`rr_peak_band_abs_error +0.1433`，
  `rr_spec_abs_error +0.0142`，`breath_count_zero_cross_abs_error +0.1713`，
  `relative_envelope_corr -0.0044`，`band_limited_corr -0.0029`。
- 呼吸次数误差主要表现为偏多计：E5-A1.0 signed breath count error 为 `+0.4299`，
  E5-A1.1 为 `+0.6178`，E5-A2.0 为 `+0.6117`。

阶段判断：

- cross-attention 从头训练只轻微改善 `rr_spec_abs_error`，但显著破坏更关键的 peak-band RR
  和呼吸次数；这重复了 E5-A0 “频谱指标小改善、计数/peak-band 变差”的失败模式。
- warm-start + time backbone 低学习率没有解决问题，且 seed `20260901` 明显拉坏，
  说明问题不是单纯的随机初始化或主干被扰动过大。
- E5-A1/E5-A2 不建议继续扩 seed 或进入 freeze/unfreeze；E5 收口为
  `native_inject pre_mixer` 仍是当前最稳主线，后续若继续探索应换问题定义，而不是继续调
  cross-attention 优化细节。

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

早期 E0/E1 已经完成它们的核心任务：建立纯时序参照，并确认 patch 路线的 STFT 输入有可重复信息增益。
E2/E3/E4/E5 进一步把路线收敛到以下判断：

- `conv2d fullband 8Hz + concat_generic + deep + fuse_len600` 是当前最清楚的简单 STFT 强参照。
- `native_inject pre_mixer` 是当前最稳的 STFT 主线；所有 native 实验必须配对
  `native_time_only` 解释，不能和 concat substrate 直接比绝对分数。
- 可学习频带前端、band-energy、SST、gated fusion 和 cross-attention 都没有证明优于
  `native_inject pre_mixer`。
- STFT 的主要收益来自降低 baseline hard / low-spectrum 窗口的长尾误拣，对 peak-band RR
  不是单调正贡献；过强或过复杂的融合容易换来 spec 小改善、计数和 peak-band 退化。

因此当前不建议继续扩 B1/B2、E4-SST、E5-A0/A1/A2 的 seed，也不建议进入 cross-attention
freeze/unfreeze。下一步如果继续沿时频输入推进，应把问题收窄为：

1. 固化 `patch_mixer1d + conv2d fullband 8Hz + native_inject pre_mixer + 轻 warmup` 作为当前默认候选，
   做最终 topK/多 seed 收口复核。
2. 只围绕 C2 已确认的 hard/low-spectrum 窗口设计条件注入或损失约束，避免重新打开宽泛的结构搜索。
3. 若再验证高频心冲击信息，只比较 3/8/12Hz 同结构同 seed 差异，并继续保留 B0 强基线和
   paired time-only substrate；不要把更宽频带、SST 和复杂 attention 同时叠加。
