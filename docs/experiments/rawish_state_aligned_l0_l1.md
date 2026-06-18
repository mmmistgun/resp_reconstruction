# Rawish State-Aligned L0/L1 实验记录

本文档记录 `bcg_rawish_wideband_state_aligned -> tho_waveform_ref` 口径下的
全量 `patch_mixer1d` L0/L1 对齐实验。原始 run、checkpoint、metrics 和图片
保留在本地 `runs/`，不进入 Git。

## 实验假设

本轮先忽略由两台采集设备采样率或时钟漂移造成的相位偏差，因为该偏差不是
BCG 到胸带呼吸之间的生理相位差。训练输入固定为
`bcg_rawish_wideband_state_aligned`，`phase_lag_weight` 保持 `0.0`。

## 固定口径

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 数据窗口：全量（`data.max_train_windows=null`，`data.max_val_windows=null`）
- 模型：`patch_mixer1d`
- Patch 参数：`patch_len=256`，`patch_stride=128`，`mixer_layers=2`
- 训练：`epochs=50`，`batch_size=128`，`learning_rate=0.001`，`use_amp=false`
- 早停：`patience=8`，`min_delta=0.001`
- seed：`training.seed=20260610`，`data.train_sample_seed=20260610`，`data.val_sample_seed=20260611`
- run root：`runs/tho_research_v2_patch_mixer_rawish_eff_l0_l1`
- L0：`loss.band_waveform_weight=0.0`
- L1a：`loss.band_waveform_weight=0.05`
- L1b：`loss.band_waveform_weight=0.10`
- `loss.high_freq_weight=0.2`
- `loss.relative_envelope_weight=0.01`
- `loss.phase_lag_weight=0.0`

本批 run 使用 batch128 效率口径重跑 L0/L1，不能与历史 batch8 anchor 当作
唯一变量对照；历史 run 只保留为背景参考，不作为本页阶段判断基线。

## 模型选择口径

本项目当前仍用胸带波形作为监督信号，但实验选择不能只按波形恢复判断。
波形形态是任务证据链的一部分，尤其用于解释低频相位、形态和对齐问题；
最终是否进入下一阶段，应优先看呼吸任务指标是否稳定。

优先级如下：

1. `rr_peak_band_abs_error`：当前主护栏，先对预测和目标做呼吸频带带通，再用峰间距估计呼吸率。
2. `rr_spec_abs_error`：频域呼吸率护栏，不应明显恶化。
3. `relative_envelope_mae` / `relative_envelope_corr`：相对呼吸强弱变化的任务指标。
4. `spectrum_similarity`：频谱一致性辅助护栏。
5. `rr_peak_abs_error`：原始时域峰值呼吸率诊断，用于暴露尖峰、毛刺和局部双峰，不再作为唯一主护栏。
6. `band_limited_corr`、`best_lag_corr`、`best_lag_sec`：波形诊断指标，用于解释低频形态和时移，不能单独判定通过。

注：2026-06-18 前的 L0/L1/L2/L3/N3/M0/M1 结论仍按当时的
`rr_peak_abs_error` 主护栏记录。自 2026-06-18 正式接入
`rr_peak_band_abs_error` 后，后续实验选择应优先使用带通峰值 RR，原始 peak RR
只用于解释局部高频尖峰或模型输出不平滑。

## Split 独立性审计

- 口径：全量（`data.max_train_windows=null`，`data.max_val_windows=null`）
- 输入：`bcg_rawish_wideband_state_aligned`
- train windows：`9881`
- val windows：`2557`
- train/val `samp_id` 重叠：`0`
- train/val `segment` 重叠：`0`
- 泄漏标记：`has_samp_id_leakage=False`，`has_segment_leakage=False`
- 结论：本轮全量对齐实验可作为 leave-samp_id-out 开发指标。

## L0

| run | source | model | data windows | band waveform | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean / median | `relative_envelope_mae` mean / median | `relative_envelope_corr` mean | `spectrum_similarity` mean | `band_limited_corr` mean | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260617_173758_049730` | fresh batch128 L0 | `patch_mixer1d` | full | 0.00 | 0.618557 | 0.962277 / 0.315239 | 0.457155 / 0.000000 | 0.218097 / 0.184294 | 0.442228 | 0.975034 | -0.790021 | 本批 L1 对照基线。 |

## L1

| run | model | data windows | band waveform | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean / median | `relative_envelope_mae` mean / median | `relative_envelope_corr` mean | `spectrum_similarity` mean | `band_limited_corr` mean | 相对 L0 判断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260617_174421_369708` | `patch_mixer1d` | full | 0.05 | 0.644862 | 4.056376 / 0.839552 | 0.442833 / 0.000000 | 0.215319 / 0.180710 | 0.453016 | 0.975352 | 0.796477 | 相对包络和波形诊断改善，但 `rr_peak_abs_error` 主护栏明显恶化，不通过。 |
| `20260617_174842_762073` | `patch_mixer1d` | full | 0.10 | 0.668635 | 4.497422 / 0.903271 | 0.447989 / 0.000000 | 0.215914 / 0.181888 | 0.449149 | 0.975579 | 0.795582 | 波形诊断改善，但 peak RR 恶化更重，不通过。 |

## 阶段判断

- 波形诊断：L1a 的 `band_limited_corr` 从 `-0.790021` 上升到 `0.796477`，`best_lag_corr` 从 `0.337117` 上升到 `0.845196`，`abs(best_lag_sec)` 从 `0.999984` 变小到 `0.157962`。L1b 也有类似改善，`band_limited_corr=0.795582`，`best_lag_corr=0.845046`，`abs(best_lag_sec)=0.158901`。
- RR 任务指标：L1a 的 `rr_peak_abs_error` mean 从 `0.962277` 恶化到 `4.056376`，L1b 恶化到 `4.497422`。虽然 `rr_spec_abs_error` mean 小幅好于 L0，但 peak RR 恶化幅度过大，不能视为任务收益。
- 相对包络：L1a 的 `relative_envelope_mae` 从 `0.218097` 降到 `0.215319`，`relative_envelope_corr` 从 `0.442228` 升到 `0.453016`；L1b 也略优于 L0。说明带限波形约束确实强化了低频相对形态，但没有保护峰值呼吸率任务。
- 当前判断：L1 证明带限波形损失能显著改变低频形态和 lag-aware 波形诊断，但在当前权重下会破坏 `rr_peak_abs_error` 主护栏。L1a/L1b 不应作为通过实验进入模型结构结论；若继续 L1c，应只尝试更小权重或改成不主导 RR 峰检测的弱约束，并继续以任务指标优先选择模型。
- 工程备注：全量评价曾卡在包络/lag 指标计算，已将 moving average 和 lag correlation 改为前缀和/相关序列实现；本批 run 的最终 `metrics.csv` 均已导出。

## L2/L3 预注册

下一轮实验输出到 `runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3`。
L2 复用 `phase_lag_loss`，L3 新增低频 STFT magnitude / phase / complex loss。
候选 run 只有在 `rr_peak_abs_error` 主护栏不明显恶化时，才允许根据波形诊断继续比较。

预注册候选：

- L2a：`phase_lag_weight=0.01`，`band_waveform_weight=0.0`
- L2b：`phase_lag_weight=0.03`，`band_waveform_weight=0.0`
- L2c：`phase_lag_weight=0.01`，`band_waveform_weight=0.005`
- L3a：`stft_mag_weight=0.02`
- L3b：`stft_mag_weight=0.05`
- L3c：`stft_mag_weight=0.02`，`stft_phase_weight=0.005`
- L3d：`stft_mag_weight=0.02`，`stft_complex_weight=0.005`

## L2 Phase-Aware / Lag-Tolerant

| run | label | phase lag | band waveform | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean / median | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | `abs(best_lag_sec)` mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260617_193555_606596` | L2a | 0.01 | 0.000 | 0.622132 | 0.738865 / 0.274892 | 0.458301 / 0.000000 | 0.217473 | 0.444611 | -0.792417 | 0.336866 | 0.999992 | 通过主护栏；RR peak 优于 L0，频域 RR 基本持平，但波形诊断未改善。 |
| `20260617_193601_991523` | L2b | 0.03 | 0.000 | 0.627500 | 4.075785 / 0.827036 | 0.443979 / 0.000000 | 0.215211 | 0.453139 | 0.796337 | 0.845213 | 0.158185 | 不通过；波形诊断显著改善，但 `rr_peak_abs_error` 主护栏明显恶化。 |
| `20260617_194044_644417` | L2c | 0.01 | 0.005 | 0.630026 | 0.795458 / 0.286958 | 0.460592 / 0.000000 | 0.217264 | 0.444784 | -0.792669 | 0.336207 | 0.999992 | 通过主护栏；相对 L2a 无明显收益，弱 band waveform 没有改善波形诊断。 |

阶段判断：

- L2a 是当前最稳的 phase-aware 候选：`rr_peak_abs_error` mean 从 L0 的 `0.962277` 降到 `0.738865`，median 从 `0.315239` 降到 `0.274892`，未触发主护栏。
- L2b 说明 phase lag 权重不能简单加大；`phase_lag_weight=0.03` 会把模型推向与 L1 类似的“低频波形相关更好，但 peak RR 明显变坏”的区域。
- L2c 没有证明 `band_waveform_weight=0.005` 有额外价值；它保住了 RR，但波形诊断仍接近 L0。
- 下一步 L3 只应先跑 low-frequency STFT magnitude，phase/complex 仍需小权重和主护栏控制。

## L3 Low-Frequency STFT

| run | label | STFT mag | STFT phase | STFT complex | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean / median | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260617_194914_779721` | L3a | 0.02 | 0.000 | 0.000 | 0.631131 | 3.756982 / 0.949931 | 0.464030 / 0.000000 | 0.215847 | 0.444725 | -0.790244 | 0.338834 | 不通过；低频 STFT magnitude 降低了局部频谱损失，但 `rr_peak_abs_error` 主护栏明显恶化。 |
| `20260617_194917_548266` | L3b | 0.05 | 0.000 | 0.000 | 0.645123 | 3.810567 / 1.031564 | 0.482362 / 0.000000 | 0.213820 | 0.451230 | -0.791732 | 0.333499 | 不通过；权重加大后 peak RR 仍严重恶化，summary 未保留 model 聚合，已按 `metrics.csv` 手工聚合。 |

阶段判断：

- L3a/L3b 都没有守住 L0 主护栏：L0 的 `rr_peak_abs_error` mean / median 为 `0.962277 / 0.315239`，L3a 恶化到 `3.756982 / 0.949931`，L3b 恶化到 `3.810567 / 1.031564`。
- `rr_spec_abs_error` 没有补偿性收益：L3a `0.464030`、L3b `0.482362`，均不优于 L0 的 `0.457155`。
- 波形诊断也没有解决相位问题：`band_limited_corr` 仍为负，`best_lag_corr` 仍接近 L0。说明仅用低频 STFT magnitude 约束没有把模型推向更合理的低频相位结构。
- 按预注册规则，L3c/L3d 暂停，不继续叠加 phase/complex STFT 分支；下一步若继续 loss 线，应回到 L2a 作为候选，在极小权重下尝试 phase-aware 约束，或先改成只参与模型诊断而不参与主训练目标。

## N1-N4 Signed Phase Alignment

本轮把旧 `phase_lag_weight` 置零，单独评估 state-aligned 前提下的 signed
phase alignment。该损失由零延迟相关、小范围 soft best-lag 相关和 lag 惩罚组成，
目标是约束低频相位方向，而不是复制胸带绝对波形。

| run | label | relative envelope | phase alignment | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | `best_lag_sec` mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260617_214224_840961` | N1 | 0.03 | 0.000 | 0.622188 | 0.953758 / 0.313205 | 0.463457 | 0.217557 | 0.442286 | -0.790206 | 0.337028 | 0.231897 | 接近 L0，单独提高 relative envelope 没有明显收益。 |
| `20260617_214227_262687` | N2 | 0.03 | 0.003 | 0.627833 | 0.967694 / 0.328634 | 0.469185 | 0.216980 | 0.441607 | -0.790659 | 0.335838 | 0.224075 | 不优于 N1；phase alignment 权重过弱时只增加训练目标复杂度。 |
| `20260617_214228_595099` | N3 | 0.03 | 0.005 | 0.628607 | 0.756761 / 0.270270 | 0.465748 | 0.217560 | 0.444018 | -0.792856 | 0.334920 | 0.208443 | 通过主护栏，RR peak 接近 L2a；但低频相关仍为负，没有解决相位方向。 |
| `20260617_214228_454909` | N4 | 0.05 | 0.003 | 0.631633 | 0.970258 / 0.322040 | 0.472050 | 0.216573 | 0.441233 | -0.790920 | 0.335469 | 0.214693 | 提高 relative envelope 权重略降 MAE，但 RR 与相关指标无收益。 |

阶段判断：

- N3 是本轮唯一值得保留的候选：`rr_peak_abs_error` mean / median 为
  `0.756761 / 0.270270`，接近 L2a 的 `0.738865 / 0.274892`。
- signed phase alignment 没有修复低频相位方向：N1-N4 的 `band_limited_corr`
  仍在 `-0.79` 附近，说明弱权重相关项不足以把模型从反向/错相区域拉出来。
- `relative_envelope_weight=0.05` 不值得继续加大；N4 的 `relative_envelope_mae`
  仅小幅下降，但 RR 和相关指标没有同步改善。
- 下一步 loss 线应以 N3 为新候选，但如果目标是解决 state-aligned 后低频相位方向，
  需要更强或更直接的 signed phase 机制；继续保留旧 `phase_lag`/STFT 主线价值不高。

## 当前主线 N3 正式复跑

代码已移除旧 `band_waveform`、`phase_lag`、STFT magnitude/phase/complex
训练分支，仅保留当前 N3 主线：

- `envelope_weight=1.0`
- `spectrum_weight=0.2`
- `smooth_weight=0.10`
- `high_freq_weight=0.2`
- `relative_envelope_weight=0.03`
- `phase_alignment_weight=0.005`

本轮用全量数据、`batch_size=128`、`bcg_rawish_wideband_state_aligned`
重新跑一次当前代码路径，输出到
`runs/tho_research_v2_patch_mixer_rawish_current_loss_n3`。

| run | epochs | best epoch | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean / median | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | `best_lag_sec` mean | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260617_220245_852336` | 47 | 39 | 0.628330 | 0.752053 / 0.279525 | 0.464602 / 0.000000 | 0.217530 | 0.444300 | -0.793033 | 0.334928 | 0.206097 | 复现 N3 口径，任务 RR 指标接近 `20260617_214228_595099`；低频相关仍为负，说明当前弱 signed phase alignment 仍未解决相位方向问题。 |

补充判断：

- 当前主线 loss 的复跑与历史 N3 基本一致，可作为下一阶段模型/数据实验的
  代码基线。
- `phase_alignment` 在训练日志中长期约 `1.66`，说明该项以当前权重更多是弱正则，
  不是主导目标。
- 若下一步仍围绕 loss 推进，关键不是恢复旧 STFT/phase_lag 分支，而是重新定义
  更贴合任务指标的节律/相对努力目标，并明确低频相位方向是否必须作为主约束。

## PatchMixer 边界伪影诊断

针对当前主线 N3 run 的 `plots_rr_peak_worst/window_004_row_10196.png`
做局部诊断。该窗口表现为预测信号在下降过程中出现尖峰，且整体预测与胸带目标
低频相位方向相反。

关键证据：

- `row=10196` 的频谱呼吸率命中目标：`pred_rr_spec_bpm=13.183594`，
  `target_rr_spec_bpm=13.183594`，`rr_spec_abs_error=0.0`。
- 同一窗口的峰值呼吸率明显偏高：`pred_rr_peak_bpm=21.818182`，
  `target_rr_peak_bpm=12.931034`，`rr_peak_abs_error=8.887147`。
- 低频相关为负：`band_limited_corr=-0.843285`。重新推理后，
  `corr(pred_lowpass, target_lowpass)=-0.846348`，
  `corr(-pred_lowpass, target_lowpass)=0.846348`。
- 当前模型 `patch_stride=128`，采样率 `100Hz`，patch 边界间隔为 `1.28s`。
  `row=10196` 中预测信号前 30 个最大一阶跳变全部落在 patch boundary 附近。
- `row=10196` 的边界附近预测跳变均值是非边界的 `2.86x`；但 target 为
  `1.00x`，输入为 `1.01x`。因此尖峰不是标签或输入在相同时间点的真实结构。
- 对 `rr_peak_abs_error` 最差的 8 个窗口复核，预测边界跳变均为非边界的
  `2.86x-4.24x`；target 基本保持 `~1.0x`。说明该现象不是单个窗口偶发。
- `row=10196` 中预测的 `>0.7Hz` 能量占比约 `1.2%`，target 约 `0.02%`。
  这解释了为什么现有 `high_freq` 正则会认为高频能量不大，但局部尖峰仍足以
  干扰 peak-based RR。

判断：

- 当前“不平滑”主要有两层来源：一是低频相位方向仍未对齐；二是
  `PatchMixer1D` patch 输出折回原长度时产生边界纹理。
- 这类尖峰会污染 `rr_peak_abs_error`，即使 `rr_spec_abs_error` 已经很好。
- 下一步优先级应先放在模型输出结构，而不是继续只调 loss 权重。首选实验是把
  `PatchMixer1D._overlap_add` 从等权平均改成 Hann 或 triangular 加权 overlap-add，
  以降低 patch boundary 不连续；随后再评估是否需要二阶 smooth、带限输出或
  更强的低频相位约束。

## 非 PatchMixer 模型复核

为确认 `row=10196` 的尖峰是否是所有模型共有问题，按当前 N3 loss、全量数据、
`batch_size=128` 和 `bcg_rawish_wideband_state_aligned` 口径，额外训练
`dlinear_waveform` 与 `periodic_unet1d_tiny`。

| model | run | epochs | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `patch_mixer1d` | `20260617_220245_852336` | 47 | 0.628330 | 0.752053 / 0.279525 | 0.464602 | 0.217530 | 0.444300 | -0.793033 | 0.334928 | RR peak 最稳，但存在 patch boundary 伪影和低频反相。 |
| `dlinear_waveform` | `20260617_233643_611251` | 13 | 0.739541 | 4.062325 / 0.627799 | 0.458301 | 0.245696 | 0.353240 | -0.756941 | 0.358269 | 不存在 patch-grid 锁定，但输出更毛刺，整体性能弱。 |
| `periodic_unet1d_tiny` | `20260617_233658_899316` | 30 | 0.603853 | 4.406477 / 1.024590 | 0.436531 | 0.206196 | 0.459925 | 0.787739 | 0.849838 | 低频相位和形态明显更好，但局部毛刺导致 peak RR 严重高估。 |

RR peak worst 前 8 个窗口的跳变诊断：

| model | `rr_peak_abs_error` mean | pred `diff` p95 mean | target `diff` p95 mean | pred/target p95 ratio | 1.28s grid ratio | top30 hits on 1.28s grid | `>0.7Hz` energy mean | lowpass corr mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `patch_mixer1d` | 9.093811 | 0.082576 | 0.022784 | 3.673612 | 3.389985 | 27.000 | 0.026022 | -0.602639 |
| `dlinear_waveform` | 14.046392 | 0.230538 | 0.026084 | 8.966111 | 1.023328 | 1.750 | 0.068056 | -0.768448 |
| `periodic_unet1d_tiny` | 14.027085 | 0.228895 | 0.026334 | 8.785184 | 1.098700 | 1.625 | 0.075132 | 0.761595 |

`row=10196` 复核：

| model | `rr_peak_abs_error` | pred/target p95 ratio | 1.28s grid ratio | top30 hits on 1.28s grid | `>0.7Hz` energy | lowpass corr | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `patch_mixer1d` | 8.887147 | 3.447927 | 2.860660 | 30 | 0.012040 | -0.846348 | 尖峰严格锁定 patch boundary，且低频反相。 |
| `dlinear_waveform` | 12.709991 | 13.017561 | 1.043926 | 2 | 0.096135 | -0.827536 | 没有 1.28s 边界锁定，但输出高频毛刺更重，低频仍反相。 |
| `periodic_unet1d_tiny` | 12.820038 | 15.114666 | 1.077991 | 1 | 0.127785 | 0.823838 | 低频相位方向正确，但毛刺和额外峰让 peak RR 高估。 |

判断：

- `PatchMixer1D` 的 `row=10196` 尖峰属于模型结构边界伪影；这一点没有在
  `dlinear_waveform` 或 `periodic_unet1d_tiny` 中复现。
- 不平滑不是 PatchMixer 独有。DLinear 和 PeriodicUNet 没有 patch-grid 锁定，
  但预测一阶跳变 p95 是 target 的 `~9x`，高于 PatchMixer 的 `~3.7x`。
- `periodic_unet1d_tiny` 是重要线索：它解决了低频反相和 lag-aware 相关问题，
  但 peak RR 被局部毛刺污染。下一步模型方向可以分成两条：PatchMixer 先修
  weighted overlap-add；PeriodicUNet 先加输出带限/抗毛刺约束，再看 peak RR
  是否回落。

## M0/M1 模型结构实验

本轮冻结当前 N3 loss，只改模型结构：

- M0：`patch_mixer1d` 使用 `model.overlap_window=hann`，验证 weighted
  overlap-add 是否能消除 patch boundary 伪影。
- M1：`periodic_unet1d_tiny` 使用 `model.output_smoothing_kernel=51`，验证输出
  平滑是否能降低局部毛刺对 peak RR 的污染。

| label | run | epochs | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `patch_uniform` | `20260617_220245_852336` | 47 | 0.628330 | 0.752053 / 0.279525 | 0.464602 | 0.217530 | 0.444300 | -0.793033 | 0.334928 | 当前 N3 对照；RR peak 最稳，但低频反相和 patch boundary 伪影明显。 |
| `patch_hann` | `20260618_004519_403378` | 21 | 0.615959 | 3.588456 / 0.563851 | 0.433094 | 0.214250 | 0.459107 | 0.796677 | 0.848886 | 消除边界伪影并修正低频方向，但 peak RR 明显恶化。 |
| `periodic_base` | `20260617_233658_899316` | 30 | 0.603853 | 4.406477 / 1.024590 | 0.436531 | 0.206196 | 0.459925 | 0.787739 | 0.849838 | 低频方向好，但局部毛刺导致 peak RR 高估。 |
| `periodic_smooth51` | `20260618_004523_162304` | 22 | 0.697770 | 2.413637 / 0.534640 | 0.696617 | 0.211167 | 0.382533 | 0.707788 | 0.779738 | peak RR 有改善，但平滑过强，频谱 RR 和相对包络明显受损。 |

RR peak worst 前 8 个窗口的跳变诊断：

| label | `rr_peak_abs_error` mean | pred `diff` p95 mean | pred/target p95 ratio | max `diff` mean | 1.28s grid ratio | top30 hits on 1.28s grid | `>0.7Hz` energy mean | lowpass corr mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `patch_uniform` | 9.093811 | 0.082576 | 3.673612 | 2.088198 | 3.389985 | 27.000 | 0.026022 | -0.602639 |
| `patch_hann` | 13.407430 | 0.142145 | 5.487226 | 0.544715 | 1.030292 | 1.125 | 0.010306 | 0.851602 |
| `periodic_base` | 14.027085 | 0.228895 | 8.785184 | 1.175888 | 1.098700 | 1.625 | 0.075132 | 0.761595 |
| `periodic_smooth51` | 13.189241 | 0.072611 | 3.068617 | 0.296188 | 1.020515 | 2.000 | 0.067513 | 0.507030 |

`row=10196` 复核：

| label | `rr_peak_abs_error` | pred/target p95 ratio | 1.28s grid ratio | top30 hits on 1.28s grid | `>0.7Hz` energy | lowpass corr | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `patch_uniform` | 8.887147 | 3.447927 | 2.860660 | 30 | 0.012040 | -0.846348 | 原始尖峰严格锁定 patch boundary，且低频反相。 |
| `patch_hann` | 10.415269 | 7.010955 | 0.995784 | 0 | 0.011174 | 0.766121 | 边界锁定消失，低频方向转正，但普通局部毛刺仍会多计峰。 |
| `periodic_base` | 12.820038 | 15.114666 | 1.077991 | 1 | 0.127785 | 0.823838 | 低频方向正确，但毛刺很强。 |
| `periodic_smooth51` | 10.506466 | 5.549697 | 0.978932 | 1 | 0.063921 | 0.929730 | 毛刺下降、低频相关更高，但 peak RR 仍偏高。 |

阶段判断：

- M0 验证成立：Hann weighted overlap-add 能消除 PatchMixer 的 patch boundary
  伪影。`row=10196` 的 top30 boundary hits 从 `30` 降到 `0`，worst8 的
  1.28s grid ratio 从 `3.39` 降到 `1.03`。
- M0 同时改变了模型优化区域：低频相关从负转正，但 peak RR 变差。说明原来的
  PatchMixer 虽然有边界伪影，但它的 peak RR 稳定性可能部分来自反相/特定形态；
  不能只以边界伪影消失判定通过。
- M1 的 `output_smoothing_kernel=51` 太重：它降低了局部跳变和 peak RR 均值，
  但 `rr_spec_abs_error` 从 `0.436531` 恶化到 `0.696617`，相对包络相关也下降。
- 下一步不建议继续扩大 loss 实验。更合理的是继续模型线：
  1. PatchMixer 保留 Hann overlap-add，再尝试更温和的 peak 稳定机制或轻量输出
     平滑，目标是在保住正低频相关的同时拉回 `rr_peak_abs_error`。
  2. PeriodicUNet 把平滑核从 `51` 降到 `11/21` 做小网格，或改成只惩罚尖峰的
     二阶差分/robust anti-spike，而不是直接强平滑输出。

## PeriodicUNet 平滑与 loss 复核

用户提出当前输出没有波形约束，可能导致信号不平滑和 `row=10196` 这类下降段
中途尖峰。为避免把任务重新定义成“胸带波形复制”，本轮只在
`periodic_unet1d_tiny + output_smoothing_kernel=21` 上做小权重、单变量复核：

- `bandwave0005`：低频带限波形 SmoothL1，`band_waveform_weight=0.005`。
- `curv002`：预测二阶差分 L1，`curvature_weight=0.02`。
- `rhythm002`：低频自相关周期分布 L1，`rhythm_weight=0.02`。
- `rhythm002_curv002`：`rhythm_weight=0.02` 与 `curvature_weight=0.02` 组合。

主指标对比：

| label | run | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `smooth11` | `20260618_105617_629286` | 0.603316 | 4.252718 / 0.847369 | 0.432521 | 0.207306 | 0.454037 | 0.790170 | 0.850358 | 轻平滑，任务指标未优于 smooth21。 |
| `smooth21` | `20260618_105623_624928` | 0.607850 | 3.222728 / 0.547314 | 0.435386 | 0.208959 | 0.445360 | 0.783797 | 0.846181 | 当前 PeriodicUNet 最稳折中。 |
| `smooth51` | `20260618_004523_162304` | 0.697770 | 2.413637 / 0.534640 | 0.696617 | 0.211167 | 0.382533 | 0.707788 | 0.779738 | peak RR 改善，但过度平滑破坏频谱 RR 和相对包络。 |
| `bandwave0005` | `20260618_111249_280881` | 0.606309 | 3.577836 / 0.649636 | 0.425074 | 0.210494 | 0.453256 | 0.786309 | 0.850139 | 点对点低频波形约束未通过主护栏。 |
| `curv002` | `20260618_111248_784881` | 0.598427 | 3.773799 / 0.705576 | 0.419345 | 0.209753 | 0.457453 | 0.793522 | 0.849979 | 平滑诊断改善、val loss 最低，但全量 peak RR 变差。 |
| `rhythm002` | `20260618_112800_265846` | 0.615309 | 3.737409 / 0.687972 | 0.405023 | 0.211290 | 0.444890 | 0.789321 | 0.849146 | 自相关节律改善频谱 RR，但未改善全量 peak RR。 |
| `rhythm002_curv002` | `20260618_112800_829058` | 0.609667 | 3.758621 / 0.675747 | 0.431376 | 0.208247 | 0.454685 | 0.789542 | 0.850298 | 局部坏例有改善，全量主指标仍不通过。 |

RR peak worst 前 8 个窗口的波形诊断：

| label | `rr_peak_abs_error` mean | pred/target p95 ratio | `>0.7Hz` energy mean | lowpass corr mean | `best_lag_corr` mean | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `smooth21` | 13.070958 | 3.623321 | 0.045159 | 0.729829 | 0.832965 | 对照。 |
| `bandwave0005` | 13.263968 | 3.577995 | 0.044268 | 0.666990 | 0.853703 | 没有解决坏例，低频相关反而下降。 |
| `curv002` | 13.162373 | 3.091685 | 0.042849 | 0.786344 | 0.914636 | 能压低跳变和提高 lag-aware 相关，但主指标不够。 |
| `rhythm002` | 12.893762 | 3.248975 | 0.039176 | 0.856382 | 0.890237 | worst8 有小幅收益，但全量变差。 |
| `rhythm002_curv002` | 12.825278 | 3.522351 | 0.041834 | 0.743189 | 0.919413 | 局部 peak 错误最低，但波形平滑性不稳定。 |

`row=10196` 复核：

| label | `rr_peak_abs_error` | pred/target p95 ratio | `>0.7Hz` energy | lowpass corr | 结论 |
|---|---:|---:|---:|---:|---|
| `smooth21` | 12.279050 | 4.044834 | 0.034420 | 0.859785 | 仍有额外峰导致 peak RR 偏高。 |
| `bandwave0005` | 11.558761 | 5.182730 | 0.064950 | 0.830125 | peak 稍好但更不平滑，不是正确方向。 |
| `curv002` | 12.121158 | 4.244350 | 0.055291 | 0.843643 | 曲率权重对该窗口帮助有限。 |
| `rhythm002` | 11.608843 | 3.761059 | 0.030289 | 0.842668 | 局部跳变和高频能量下降，但 peak 错误仍大。 |
| `rhythm002_curv002` | 10.784381 | 4.845408 | 0.048428 | 0.802688 | `row=10196` peak 最好，但以全量均值变差为代价。 |

阶段判断：

- 低频带限波形 SmoothL1 不应作为主线。它会把模型推向“更像胸带”的局部形态，
  但没有保护 peak RR，且 `row=10196` 的跳变和高频能量反而更糟。
- 曲率项是有用诊断项：它能降低 worst8 的跳变 p95 ratio，并提高
  `best_lag_corr`，但单独训练时 `rr_peak_abs_error` 均值从 `3.222728`
  恶化到 `3.773799`，不能作为当前选择目标。
- 自相关节律 loss 符合任务直觉，确实让 worst8 peak mean 从 `13.070958`
  降到 `12.893762/12.825278`，但全量窗口主指标没有改善，说明当前模型的
  peak 错误不是单靠全窗口周期分布能解决的。
- 这轮支持“loss 限制不够”这个怀疑，但更准确地说是：当前 loss 缺少对局部
  peak topology 的直接约束；点对点波形、二阶平滑和全窗口自相关都只能处理
  一部分症状。下一步仍应回到模型输出结构，让模型天然产生带限、低自由度、
  不易出现双峰的呼吸轨迹，再用小权重节律/曲率项做辅助。

## 带通峰值 RR 口径更新

2026-06-18 复核 `row=10196` 和多个模型后，确认原始
`rr_peak_abs_error` 会被局部毛刺、额外峰和 patch 边界尖峰显著污染。该指标仍有
诊断价值，但不适合作为当前任务的唯一主护栏。正式接入后：

- `rr_peak_band_abs_error`：模型选择主指标，先带通到呼吸频带，再用峰间距估计 RR。
- `rr_spec_abs_error`：频域 RR 辅助护栏，用于防止模型只优化局部峰形。
- `rr_peak_abs_error`：原始输出尖峰诊断，用于定位不平滑、双峰和边界伪影。
- `band_limited_corr` / `best_lag_corr`：低频方向、形态和残余时移诊断，不能单独作为通过标准。

一次性复核输出保存在本地 `/tmp/tho_filtered_rr_summary.csv` 和
`/tmp/tho_filtered_rr_rows.csv`，不进入 Git。核心结果如下：

| label | raw peak RR mean | band peak RR mean | band peak RR median | `row=10196` band peak RR |
|---|---:|---:|---:|---:|
| `patch_uniform` | 0.752053 | 0.457180 | 0.183720 | 0.084515 |
| `patch_hann` | 3.588456 | 0.568421 | 0.197455 | 0.524183 |
| `periodic_smooth21` | 3.222728 | 0.654075 | 0.171471 | 0.286070 |
| `periodic_curv002` | 3.773799 | 0.752821 | 0.186706 | 0.374370 |
| `periodic_smooth51` | 2.413637 | 0.771716 | 0.241118 | 0.227850 |
| `periodic_base` | 4.406477 | 0.798523 | 0.184324 | 0.213375 |
| `periodic_smooth11` | 4.252718 | 0.806439 | 0.184324 | 0.315373 |
| `periodic_rhythm002_curv002` | 3.758621 | 0.883504 | 0.186706 | 0.315373 |
| `periodic_bandwave0005` | 3.577836 | 0.897629 | 0.181211 | 0.344806 |
| `periodic_rhythm002` | 3.737409 | 1.243227 | 0.187309 | 0.374370 |
| `dlinear` | 4.062325 | 1.580762 | 0.232897 | 0.539349 |

阶段判断：

- `row=10196` 的原始 peak RR 坏例主要来自额外尖峰；带通后
  `periodic_smooth21` 的该窗口错误从原始 `12.279050` 降到 `0.286070`。
- `patch_uniform` 的带通 RR 最好，但低频相关为负，仍不应单独作为更优模型结论。
- 如果同时要求 RR 稳定和低频方向正确，`patch_hann` 与 `periodic_smooth21`
  是更合理的下一轮模型候选。
- 后续正式实验的 summary 已改为以 `selection_task_rr_peak_band_abs_error_mean`
  作为主选择指标，原始 `rr_peak_abs_error` 进入 `selection_waveform_*` 诊断列。

## M2 模型候选复核

执行计划：`docs/superpowers/plans/2026-06-18-model-m2-band-rr.md`。

本轮先用正式 `rr_peak_band_abs_error` 口径重评历史候选，再围绕两个候选补 seed：

- `patch_mixer1d + overlap_window=hann`
- `periodic_unet1d_tiny + output_smoothing_kernel=21`

本地输出：

- `runs/tho_research_v2_model_m2_band_rr/`
- `runs/tho_research_v2_model_m2_band_rr_summary.csv`
- 临时对照表：`/tmp/m2_model_candidates_table.csv`

核心结果：

| label | run | best epoch | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `periodic_smooth21_seed20260620` | `20260618_132348_839544` | 13 | 0.616427 | 0.453749 / 0.172381 | 0.443979 | 0.209745 | 0.444236 | -0.792724 | 0.349528 | 2.575073 | 带通 RR 最好，但低频反相，不能作为通过模型。 |
| `patch_uniform_old` | `20260617_220245_852336` | 39 | 0.628330 | 0.457180 / 0.183720 | 0.464602 | 0.217530 | 0.444300 | -0.793033 | 0.334928 | 0.752053 | RR 上限锚点；低频反相。 |
| `patch_hann_seed20260620` | `20260618_132345_623904` | 11 | 0.619316 | 0.486101 / 0.188186 | 0.436531 | 0.216116 | 0.457626 | 0.794931 | 0.846602 | 3.853268 | 新 seed 仍保持正低频方向，带通 RR 接近最优。 |
| `patch_hann_old` | `20260618_004519_403378` | 13 | 0.615959 | 0.568421 / 0.197455 | 0.433094 | 0.214250 | 0.459107 | 0.796677 | 0.848886 | 3.588456 | 正低频方向稳定；任务指标略弱于新 seed。 |
| `periodic_smooth21_seed20260630` | `20260618_133343_665730` | 5 | 0.621051 | 0.610902 / 0.174149 | 0.454291 | 0.213112 | 0.443119 | -0.789184 | 0.357582 | 2.362895 | 第二个新 seed 仍低频反相，说明方向不稳定。 |
| `periodic_smooth21_old` | `20260618_105623_624928` | 8 | 0.607850 | 0.654075 / 0.171471 | 0.435386 | 0.208959 | 0.445360 | 0.783797 | 0.846181 | 3.222728 | 旧 seed 低频方向正确，但未被新 seed 复现。 |

阶段判断：

- `periodic_unet1d_tiny + smoothing21` 不是稳定的相位方向解决方案。三个 seed 中，
  只有旧 seed 得到正 `band_limited_corr`；两个新 seed 均回到约 `-0.79` 的反相区域。
- `patch_hann` 更稳定：两个 seed 都保持正 `band_limited_corr` 和高
  `best_lag_corr`，且新 seed 的带通 RR 均值达到 `0.486101`，接近反相
  `patch_uniform` 的 `0.457180`。
- `patch_uniform` 和 `periodic_smooth21_seed20260620` 虽然带通 RR 最好，但低频方向为负，
  只能作为“RR 可达上限/反相锚点”，不应作为通过模型。
- 下一步模型主线建议转向 `patch_mixer1d + hann`，优先解决普通局部毛刺和原始
  peak 诊断偏高；不要继续把 `periodic_unet1d_tiny` 当作已解决低频相位方向的模型。
- 若继续 PeriodicUNet，应先引入显式极性/相位方向稳定机制或输出低自由度约束，
  否则多 seed 不稳定会让模型选择结论不可复现。
