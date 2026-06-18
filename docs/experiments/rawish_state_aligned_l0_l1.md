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

## M3 Patch-Hann 抗毛刺小网格

执行计划：`docs/superpowers/plans/2026-06-18-patch-hann-m3-antispike.md`。

本轮只沿 `patch_mixer1d + overlap_window=hann` 主线推进，目标是降低普通局部毛刺
和 raw peak 偏高，同时守住带通 RR 和正低频方向。

本地输出：

- `runs/tho_research_v2_patch_hann_m3_antispike/`
- `runs/tho_research_v2_patch_hann_m3_antispike_summary.csv`
- 临时对照表：`/tmp/m3_patch_hann_table.csv`

核心结果：

| label | run | best epoch | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean / median | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `M3_smoothing5` | `20260618_140038_159638` | 11 | 0.616997 | 0.460197 / 0.185325 | 0.451426 | 0.215232 | 0.453990 | 0.786820 | 0.839476 | 3.247666 / 0.545622 | 带通 RR 最好，raw peak 明显低于 M2。 |
| `M3_smoothing11` | `20260618_140919_387087` | 15 | 0.616586 | 0.468274 / 0.183150 | 0.444552 | 0.216352 | 0.453222 | 0.790242 | 0.842341 | 3.214445 / 0.569872 | raw peak 均值最低，频谱 RR 和低频相关略优于 smoothing5。 |
| `M2_patch_hann_seed20260620` | `20260618_132345_623904` | 11 | 0.619316 | 0.486101 / 0.188186 | 0.436531 | 0.216116 | 0.457626 | 0.794931 | 0.846602 | 3.853268 / 0.685358 | M2 对照。 |
| `M3_stride64` | `20260618_140039_545507` | 11 | 0.620982 | 0.538696 / 0.176835 | 0.441687 | 0.218182 | 0.449991 | 0.787429 | 0.842882 | 4.157955 / 0.799066 | 更大 overlap 没有带来抗毛刺收益，raw peak 反而更差。 |

阶段判断：

- `patch_stride=64` 不建议继续。它没有改善 raw peak，且带通 RR 均值从 M2 的
  `0.486101` 恶化到 `0.538696`。
- 轻量输出平滑有效：`smoothing5` 与 `smoothing11` 都保持正低频方向，并把 raw
  `rr_peak_abs_error` 均值从 `3.853268` 降到约 `3.2`。
- `smoothing5` 更偏任务主指标，`rr_peak_band_abs_error_mean=0.460197`；`smoothing11`
  更偏抗毛刺和频谱护栏，raw peak 均值最低且 `rr_spec_abs_error_mean=0.444552`。
- 这一轮支持把 Patch-Hann 的普通局部毛刺问题先用轻量输出平滑处理；但
  `relative_envelope_corr` 和 `band_limited_corr` 较 M2 略降，下一步不应继续加大
  平滑核，而应对 `smoothing5/11` 做多 seed 复核。

## M4 Patch-Hann 输出平滑多 seed 复核

执行计划：`docs/superpowers/plans/2026-06-18-patch-hann-m4-smoothing-multiseed.md`。

M4 修正了 M3 的一个对照缺口：如果要判断输出平滑是否真的优于未平滑
Patch-Hann，必须在同一训练 seed 下加入 `output_smoothing_kernel=1` baseline。
本轮因此固定所有数据、loss、模型和训练参数，只比较 smoothing kernel：

- `output_smoothing_kernel=1`：同 seed 未平滑 baseline。
- `output_smoothing_kernel=5`：轻量输出平滑。
- `output_smoothing_kernel=11`：更强抗毛刺输出平滑。

本地输出：

- `runs/tho_research_v2_patch_hann_m4_smoothing_multiseed/`
- `runs/tho_research_v2_patch_hann_m4_smoothing_multiseed_summary.csv`
- 临时明细表：`/tmp/m4_patch_hann_table.csv`
- 临时聚合表：`/tmp/m4_patch_hann_agg.csv`

核心结果：

| label | run | seed | smoothing | best epoch | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean / median | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `M4_baseline_seed20260630` | `20260618_142601_305525` | 20260630 | 1 | 7 | 0.625545 | 0.442357 / 0.183446 | 0.429657 | 0.222684 | 0.452918 | 0.789606 | 0.843318 | 4.602143 / 0.912935 | 同 seed 主指标最好，但 raw peak 和相对包络 MAE 最差。 |
| `M4_smoothing5_seed20260630` | `20260618_142242_572035` | 20260630 | 5 | 12 | 0.618640 | 0.451466 / 0.183150 | 0.429084 | 0.214174 | 0.450427 | 0.792442 | 0.844588 | 3.811119 / 0.763775 | 主指标轻微劣于 baseline，但 raw peak 与相对包络 MAE 改善。 |
| `M4_smoothing11_seed20260630` | `20260618_142243_089784` | 20260630 | 11 | 12 | 0.613268 | 0.475326 / 0.187915 | 0.429657 | 0.214901 | 0.452390 | 0.793863 | 0.845206 | 3.354320 / 0.546693 | 抗毛刺最强，但主指标明显劣于 baseline。 |
| `M4_baseline_seed20260640` | `20260618_143157_819337` | 20260640 | 1 | 10 | 0.626740 | 0.418410 / 0.170943 | 0.453145 | 0.220565 | 0.449760 | 0.791463 | 0.843165 | 3.676039 / 0.652300 | 第二个 seed 仍是主指标最好。 |
| `M4_smoothing5_seed20260640` | `20260618_143155_840988` | 20260640 | 5 | 10 | 0.617850 | 0.438323 / 0.172840 | 0.457155 | 0.217167 | 0.449998 | 0.794162 | 0.843798 | 3.540723 / 0.626532 | raw peak 略改善，但主指标仍劣于 baseline。 |
| `M4_smoothing11_seed20260640` | `20260618_143155_842195` | 20260640 | 11 | 10 | 0.614522 | 0.467586 / 0.178564 | 0.449708 | 0.216640 | 0.450400 | 0.795315 | 0.844187 | 3.108679 / 0.555556 | raw peak 最低，主指标仍明显劣于 baseline。 |

M4 两个 seed 聚合：

| smoothing | n | `rr_peak_band_abs_error` mean / std | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean / median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.430384 / 0.016933 | 0.441401 | 0.221624 | 0.451339 | 0.790534 | 0.843242 | 4.139091 / 0.782617 |
| 5 | 2 | 0.444894 / 0.009293 | 0.443120 | 0.215670 | 0.450212 | 0.793302 | 0.844193 | 3.675921 / 0.695154 |
| 11 | 2 | 0.471456 / 0.005473 | 0.439682 | 0.215771 | 0.451395 | 0.794589 | 0.844696 | 3.231499 / 0.551124 |

阶段判断：

- M4 不支持把简单输出平滑设为默认主线。两个新 seed 中，未平滑
  `output_smoothing_kernel=1` 都拿到最低 `rr_peak_band_abs_error_mean`，两 seed
  平均为 `0.430384`；`smoothing5` 与 `smoothing11` 分别恶化到 `0.444894`
  和 `0.471456`。
- M3 中 `smoothing5/11` 相对 M2 的主指标收益，不能再解释为平滑机制本身稳定有效；
  更可能混入了训练 seed 或缺少同 seed baseline 的影响。
- 输出平滑仍有诊断价值：kernel 越大，raw `rr_peak_abs_error` 越低，且
  `band_limited_corr` / `best_lag_corr` 略升。这说明它确实在抑制局部毛刺，
  但代价是损伤带通 peak RR 主任务。
- `smoothing11` 可以保留为抗毛刺诊断候选；`smoothing5` 可以作为折中备选；
  但如果按当前正式选模口径，下一步不应继续放大 moving-average 平滑。
- 下一步应进入更明确的模型输出结构设计：让模型天然输出低自由度、带限、连续的
  呼吸轨迹，而不是在输出端事后移动平均。候选方向包括带限 decoder、低频 basis
  decoder、frequency bottleneck 或显式平滑/节律分支，并继续用
  `rr_peak_band_abs_error_mean` 做主护栏。

## M5 模型结构候选实验

执行计划：`docs/superpowers/plans/2026-06-18-model-m5-structure-candidates.md`。

本轮冻结当前 N3 loss、Research v2 数据口径和正式选模指标，只比较低风险模型
结构候选：

- `patch_mixer1d + overlap_window=hann + output_smoothing_kernel=1`：同 seed
  Patch-Hann baseline。
- `unet1d_tiny_noskip1`：去掉最浅层 skip 的历史候选。
- `unet1d_tiny_noskip_all`：新增完全去掉两个 decoder skip 的 U-Net。
- `patch_mixer1d_fir_frontend`：新增初始化为呼吸频带的可学习 FIR 前端，再接
  Patch-Hann。

首轮固定 `training.seed=20260650`。按预注册规则，三个新候选均未通过进入第二
seed 的条件；因此 `training.seed=20260660` 只补跑 Patch-Hann baseline，用于确认
baseline 自身稳定性。

本地输出：

- `runs/tho_research_v2_model_m5_structure_candidates/`
- `runs/tho_research_v2_model_m5_structure_candidates_summary.csv`
- 临时明细表：`/tmp/m5_model_candidates_table.csv`
- 临时含 best epoch 明细表：`/tmp/m5_model_candidates_table_with_epoch.csv`

核心结果：

| label | run | model | seed | epochs | best epoch | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean / median | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `M5_patch_hann_seed20260650` | `20260618_150349_945127` | `patch_mixer1d` | 20260650 | 21 | 13 | 0.627256 | 0.431143 / 0.189873 | 0.468613 | 0.215636 | 0.453012 | -0.791998 | 0.319607 | 2.147589 / 0.587406 | 同 seed baseline；带通 RR 尚可，但低频方向为负。 |
| `M5_fir_frontend_seed20260650` | `20260618_150346_623888` | `patch_mixer1d_fir_frontend` | 20260650 | 21 | 13 | 0.620956 | 0.424903 / 0.178564 | 0.477206 | 0.217683 | 0.452910 | -0.790535 | 0.324134 | 3.180196 / 0.811103 | 带通 RR 略优于同 seed baseline，但方向护栏失败，raw peak 更差。 |
| `M5_noskip1_seed20260650` | `20260618_150347_069377` | `unet1d_tiny_noskip1` | 20260650 | 13 | 5 | 0.842562 | 1.862619 / 0.390866 | 1.170958 | 0.215698 | 0.284147 | -0.648240 | 0.353571 | 4.729596 / 3.495199 | 方向和 RR 均不通过。 |
| `M5_noskip_all_seed20260650` | `20260618_150345_380356` | `unet1d_tiny_noskip_all` | 20260650 | 18 | 10 | 0.829679 | 0.774305 / 0.271248 | 0.731563 | 0.214936 | 0.288875 | 0.641058 | 0.753618 | 5.158663 / 1.906501 | 低频方向转正，但 RR、频谱和相对努力指标明显变差。 |
| `M5_patch_hann_seed20260660` | `20260618_151631_769707` | `patch_mixer1d` | 20260660 | 20 | 12 | 0.627622 | 0.445080 / 0.197455 | 0.431376 | 0.213248 | 0.456756 | -0.796528 | 0.329242 | 2.616823 / 0.734900 | 第二个 baseline seed 仍低频反向，说明 Patch-Hann 方向不稳定。 |

阶段判断：

- M5 不支持把 FIR 前端作为下一条主线。`patch_mixer1d_fir_frontend` 在
  seed20260650 上把带通 RR 均值从 `0.431143` 小幅降到 `0.424903`，但
  `band_limited_corr=-0.790535`，raw peak 从 `2.147589` 恶化到 `3.180196`。
  它更像是在同一个反向解附近微调频带输入，而不是解决低频方向或毛刺问题。
- M5 不支持简单去 skip 作为答案。`unet1d_tiny_noskip1` 方向和任务指标都失败；
  `unet1d_tiny_noskip_all` 虽然把低频相关推到正值 `0.641058`，但
  `rr_peak_band_abs_error_mean=0.774305`，`rr_spec_abs_error_mean=0.731563`，
  `relative_envelope_corr=0.288875`，说明完全去 skip 会显著损伤呼吸节律和相对
  努力建模能力。
- M5 反过来暴露了 Patch-Hann baseline 的 seed 方向不稳定。M2/M4 中
  seed20260620/20260630/20260640 的 Patch-Hann 都是正低频方向；本轮
  seed20260650/20260660 都回到约 `-0.79` 的低频负相关，且 `best_lag_corr`
  只有约 `0.32`，不能解释为简单小范围时延。
- 因此，M4 的“Patch-Hann 低频方向稳定”结论需要收缩：Hann overlap-add 能消除
  patch boundary 伪影，但不足以稳定选择正确的低频相位方向。
- 下一步不建议直接扩大到 Mamba、完整频域 bottleneck 或复杂 dual-stream 大模型。
  更低后悔成本的 M6 应先诊断并约束“正负方向/相位盆地”问题，例如：固定同一
  Patch-Hann 配置做多 seed 方向稳定性统计，或加入显式 signed low-frequency
  direction 选择机制。只有在方向稳定后，再设计低自由度、带限或 dual-stream
  decoder 才更有解释价值。

## Patch-Hann 多 seed 方向稳定性诊断

本轮固定 Patch-Hann baseline 口径，只改变 `training.seed`，用于诊断低频方向
翻转是否来自训练随机性。新增 seed 为 `20260670`、`20260680`、`20260690`、
`20260700`、`20260710`、`20260720`，输出到
`runs/tho_research_v2_patch_hann_direction_multiseed/`。其中 `20260680` 与
`20260720` 首次并行启动时被提权审批器拒绝，后续已单独补跑完成。

合并分析包含 M2/M4/M5 与本轮新增的 11 个可比 Patch-Hann baseline。筛选条件：

- `model.name=patch_mixer1d`
- `model.overlap_window=hann`
- `model.output_smoothing_kernel=1`
- `model.patch_stride=128`
- full data、N3 loss、固定数据 seed。

本地输出：

- `runs/tho_research_v2_patch_hann_direction_multiseed_summary.csv`
- 临时合并表：`/tmp/patch_hann_direction_multiseed_combined.csv`

核心结果：

| seed | run | epochs | best epoch | best val loss | val phase alignment | `rr_peak_band_abs_error` mean | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | direction | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 20260620 | `20260618_132345_623904` | 19 | 11 | 0.619316 | 0.229988 | 0.486101 | 0.436531 | 0.216116 | 0.457626 | 0.794931 | positive | 0.846602 | 3.853268 |
| 20260630 | `20260618_142601_305525` | 15 | 14 | 0.625545 | 0.233192 | 0.442357 | 0.429657 | 0.222684 | 0.452918 | 0.789606 | positive | 0.843318 | 4.602143 |
| 20260640 | `20260618_143157_819337` | 18 | 10 | 0.626740 | 0.232929 | 0.418410 | 0.453145 | 0.220565 | 0.449760 | 0.791463 | positive | 0.843165 | 3.676039 |
| 20260650 | `20260618_150349_945127` | 21 | 13 | 0.627256 | 1.672178 | 0.431143 | 0.468613 | 0.215636 | 0.453012 | -0.791998 | negative | 0.319607 | 2.147589 |
| 20260660 | `20260618_151631_769707` | 20 | 12 | 0.627622 | 1.673372 | 0.445080 | 0.431376 | 0.213248 | 0.456756 | -0.796528 | negative | 0.329242 | 2.616823 |
| 20260670 | `20260618_152809_698210` | 20 | 12 | 0.620683 | 0.228390 | 0.594224 | 0.445697 | 0.216391 | 0.454621 | 0.796705 | positive | 0.845614 | 4.346215 |
| 20260680 | `20260618_154745_946398` | 24 | 16 | 0.622076 | 0.228285 | 0.469799 | 0.451999 | 0.214720 | 0.450995 | 0.796917 | positive | 0.845812 | 3.483859 |
| 20260690 | `20260618_152813_981274` | 26 | 18 | 0.626557 | 1.673147 | 0.444933 | 0.437104 | 0.213617 | 0.454105 | -0.796184 | negative | 0.326488 | 2.520622 |
| 20260700 | `20260618_152811_169836` | 19 | 11 | 0.620598 | 0.230589 | 0.435927 | 0.443979 | 0.215978 | 0.453891 | 0.794646 | positive | 0.844050 | 4.289256 |
| 20260710 | `20260618_152815_350433` | 26 | 22 | 0.628415 | 1.670164 | 0.458545 | 0.444552 | 0.215785 | 0.451463 | -0.793236 | negative | 0.327301 | 2.787340 |
| 20260720 | `20260618_154747_423382` | 18 | 18 | 0.630826 | 1.672243 | 0.441465 | 0.452572 | 0.216838 | 0.453360 | -0.794948 | negative | 0.322067 | 3.062251 |

按方向聚合：

| direction | n | `rr_peak_band_abs_error` mean / std | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | val phase alignment mean | best val loss mean | raw `rr_peak_abs_error` mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positive | 6 | 0.474470 / 0.063494 | 0.443501 | 0.217742 | 0.453302 | 0.794045 | 0.230562 | 0.622493 | 4.041797 |
| negative | 5 | 0.444233 / 0.009811 | 0.446843 | 0.215025 | 0.453739 | -0.794579 | 1.672221 | 0.628135 | 2.626925 |

诊断结论：

- Patch-Hann 的低频方向存在两个稳定盆地。11 个 seed 中，6 个进入正向盆地，
  5 个进入反向盆地；方向不是单个 seed 的偶发结果。
- `val_phase_alignment` 是近乎完全分离的方向诊断特征：正向 seed 在 best epoch
  约 `0.23`，反向 seed 约 `1.67`。当前 `phase_alignment_weight=0.005` 能让
  正向 seed 的 `best_val_loss` 平均更低，但权重太弱，不能保证训练从反向盆地
  翻到正向盆地。
- 方向正确与当前主任务指标并不完全一致。反向 seed 的
  `rr_peak_band_abs_error_mean` 平均为 `0.444233`，略优于正向 seed 的
  `0.474470`；raw `rr_peak_abs_error` 也明显更低。这说明模型可能利用反向形态
  得到更稳定的峰间距，而不是恢复更合理的低频方向。
- `best_lag_corr` 在正向 seed 约 `0.84`，反向 seed 约 `0.32`，说明这不是简单
  小范围 lag 能解释的问题，而是低频极性/相位盆地问题。
- 下一步不应继续单纯扩大模型复杂度。更合理的 M6 是把 signed low-frequency
  direction 从诊断项升级为训练约束或选择约束，例如加大/分阶段使用
  `phase_alignment_weight`、加入 early direction gate，或把正向方向作为模型选择
  的硬护栏后再比较 RR 指标。

## Patch-Hann 0.8 呼吸周期 lag / polarity 诊断

用户追问当前方向翻转到底是“刚好反相”，还是模型输出存在不确定时延。本轮不重新
训练，只对上述 11 个 Patch-Hann checkpoint 重新推理验证集，并在低频带上做
动态 lag search：

- 每个窗口用目标信号频谱 RR 估计呼吸周期。
- lag search 上限设为 `0.8 * target_period_sec`，平均约 `3.10s`。
- 同时比较 `pred` 与 `-pred`：
  - `same_best_corr`：不翻转预测，在 `0.8` 周期内找最佳 lag。
  - `inv_best_corr`：翻转预测后，在 `0.8` 周期内找最佳 lag。
  - `same_abs_lag_frac` / `inv_abs_lag_frac`：最佳 lag 占目标呼吸周期的比例。

为降低计算量，诊断先对 100Hz 信号做呼吸带通，再降采样到 10Hz 做 lag search。
输出保存在本地：

- `/tmp/patch_hann_cycle_lag_08_summary.csv`
- `/tmp/patch_hann_cycle_lag_08_windows.csv`

按方向聚合：

| direction | n | zero corr mean | same best corr mean | same abs lag frac mean | inverted zero corr mean | inverted best corr mean | inverted abs lag frac mean | inverted winner rate | same halfcycle-like rate | inverted near-zero rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| positive | 6 | 0.794017 | 0.843727 | 0.040572 | -0.794017 | 0.808298 | 0.473213 | 0.316452 | 0.000130 | 0.000000 |
| negative | 5 | -0.794556 | 0.806451 | 0.472828 | 0.794556 | 0.844418 | 0.040922 | 0.684943 | 0.719280 | 0.804771 |

诊断结论：

- 反向组如果不翻转预测，`same_best_corr` 可通过约 `0.47` 个呼吸周期的 lag 提升
  到 `0.806451`。这说明从周期信号角度看，它确实接近半周期相移。
- 但反向组把预测取负后，`inv_best_corr=0.844418`，最佳 lag 只剩约 `0.041`
  个周期，几乎回到正向组的近零延迟形态。这说明对当前输出而言，更简洁的解释是
  **近零延迟的极性翻转**，而不是任意、不稳定的时延。
- 正向组相反：不翻转时最佳 lag 约 `0.041` 周期；翻转后才需要约 `0.47` 周期
  才能找到较高相关。这与反向组形成镜像关系。
- 因此，M6 的重点应是稳定低频 signed direction / polarity，而不是只扩大
  lag-tolerant search。若只做更宽 lag-tolerant loss，模型可能继续接受半周期错相
  解，反而不能保证 state-aligned 前提下的生理方向一致。

## M6：Phase alignment 权重扫描

基于上面的 polarity 诊断，本轮不改模型结构，只固定 `patch_mixer1d +
overlap_window=hann`，扫描 `phase_alignment_weight`。目标是验证 signed
low-frequency direction 能否从“诊断项”升级为稳定训练约束，同时继续用任务指标
作为护栏。

固定条件：

- 输入：`bcg_rawish_wideband_state_aligned`
- 模型：`patch_mixer1d`，`patch_len=256`，`patch_stride=128`，
  `mixer_layers=2`，`overlap_window=hann`，`output_smoothing_kernel=1`
- 数据：full train/val windows，`train_sample_seed=20260610`，
  `val_sample_seed=20260611`
- 训练：`batch_size=128`，`epochs=50`，`patience=8`，`min_delta=0.001`
- 输出：`runs/tho_research_v2_patch_hann_m6_phase_weight_sweep/`
- 汇总：`runs/tho_research_v2_patch_hann_m6_phase_weight_sweep_summary.csv`
- 临时明细：`/tmp/m6_phase_weight_sweep_table.csv`

核心结果：

| seed | `phase_alignment_weight` | run | best epoch | best val loss | val phase alignment | `rr_peak_band_abs_error` mean | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260650 | 0.02 | `20260618_170857_118784` | 13 | 0.651580 | 1.671082 | 0.432834 | 0.467467 | 0.215241 | 0.454097 | -0.791611 | 0.321759 | 2.198552 |
| 20260650 | 0.05 | `20260618_171550_291071` | 13 | 0.704754 | 1.665620 | 0.435258 | 0.461738 | 0.215353 | 0.451796 | -0.787219 | 0.326138 | 2.476685 |
| 20260650 | 0.10 | `20260618_172152_328561` | 17 | 0.643348 | 0.229674 | 0.445811 | 0.449135 | 0.213498 | 0.446414 | 0.795338 | 0.846277 | 4.061872 |
| 20260650 | 0.20 | `20260618_172155_060655` | 14 | 0.665370 | 0.227620 | 0.473473 | 0.447416 | 0.213426 | 0.451383 | 0.798214 | 0.846289 | 4.519892 |
| 20260660 | 0.02 | `20260618_170859_672497` | 12 | 0.628927 | 0.231462 | 0.458584 | 0.437104 | 0.220660 | 0.451728 | 0.792940 | 0.845263 | 3.851374 |
| 20260660 | 0.05 | `20260618_171607_902218` | 12 | 0.635268 | 0.230613 | 0.459685 | 0.436531 | 0.220208 | 0.452480 | 0.794060 | 0.845636 | 3.936681 |
| 20260690 | 0.10 | `20260618_172845_958331` | 6 | 0.646572 | 0.232608 | 0.479456 | 0.431948 | 0.223545 | 0.453749 | 0.792177 | 0.845448 | 3.451273 |
| 20260680 | 0.10 | `20260618_172848_311726` | 16 | 0.645000 | 0.227770 | 0.477737 | 0.454291 | 0.213962 | 0.449543 | 0.796776 | 0.845828 | 3.513082 |

同 seed 对照：

| seed | baseline direction | baseline `rr_peak_band_abs_error` | M6 best candidate | M6 direction | M6 `rr_peak_band_abs_error` | 结论 |
|---:|---|---:|---|---|---:|---|
| 20260650 | negative | 0.431143 | weight 0.10 | positive | 0.445811 | `0.02/0.05` 拉不动；`0.10` 能转正且带通 RR 只小幅变差。 |
| 20260660 | negative | 0.445080 | weight 0.02 | positive | 0.458584 | 较小权重已经转正，但 RR 轻微变差。 |
| 20260690 | negative | 0.444933 | weight 0.10 | positive | 0.479456 | `0.10` 可把负向 seed 转正，但 RR 代价更明显。 |
| 20260680 | positive | 0.469799 | weight 0.10 | positive | 0.477737 | 对本来正向的 seed 不会翻坏，但 RR 略退。 |

阶段结论：

- `phase_alignment_weight=0.02/0.05` 不是足够强的通用解法。它们可以让
  seed20260660 转正，但 seed20260650 仍停留在 `val_phase_alignment≈1.67` 的
  反向盆地。
- `phase_alignment_weight=0.10` 能把顽固负向 seed20260650 转正，并在
  seed20260690 上复现；同时保持 seed20260680 的正方向。因此当前证据支持：
  反转问题主要是训练目标对 signed direction 约束太弱，而不是输入无法提供相位方向。
- `phase_alignment_weight=0.20` 不是更优选择。它同样转正，但 best val loss 与
  `rr_peak_band_abs_error` 均差于 `0.10`，说明强约束开始明显挤压任务指标。
- `0.10` 的主要代价是 raw `rr_peak_abs_error` 上升。这更像输出局部毛刺/尖峰诊断
  变差，而不是带通 RR 主任务失败；后续仍应把 raw peak 作为诊断，不应压过
  `rr_peak_band_abs_error`、`rr_spec_abs_error` 和相对努力指标。

建议：

- 下一轮主线可把 `phase_alignment_weight=0.10` 作为 M6 候选默认，而不是直接使用
  `0.20`。
- 不建议继续加宽 lag-tolerant loss 作为主线，因为宽 lag 会容忍半周期错相；
  当前更需要 signed low-frequency direction。
- 若 `0.10` 在更多 seed 上持续带来 raw peak 毛刺，可以再考虑输出平滑、方向
  warm-up/curriculum 或 checkpoint selection gate，而不是继续盲目加大权重。

## M7：替代 phase loss 的 signed direction loss 候选

本轮目标是去掉 `phase_alignment` 训练项，用更直接的 signed direction anchor
替代它的功能。为避免高频 BCG 结构主导，所有新项都作用在呼吸带限信号或其带符号
低频派生量上。首轮只用顽固负向 seed `20260650` 做 single-seed 压力测试。

固定条件：

- 输入：`bcg_rawish_wideband_state_aligned`
- 模型：`patch_mixer1d`，`patch_len=256`，`patch_stride=128`，
  `mixer_layers=2`，`overlap_window=hann`，`output_smoothing_kernel=1`
- 数据：full train/val windows，`train_sample_seed=20260610`，
  `val_sample_seed=20260611`
- 旧相位项：`phase_alignment_weight=0.0`
- 候选项权重：各候选先用 `0.1`
- 输出：`runs/tho_research_v2_patch_hann_m7_direction_loss_candidates/`
- 汇总：`runs/tho_research_v2_patch_hann_m7_direction_loss_candidates_summary.csv`
- 临时明细：`/tmp/m7_direction_loss_candidates_table.csv`

候选定义：

- `signed_cosine`：`1 - cosine(LPF(pred), LPF(target))`
- `signed_corr`：`1 - corr(LPF(pred), LPF(target))`
- `signed_rms_envelope`：`RMS(LPF(x)) * tanh(zscore(LPF(x)) / temperature)` 后做相关损失；
  使用 soft sign 是为了保留梯度，避免 hard sign 阻断方向修正。
- `signed_mean`：对 `LPF(x)` 做 2 秒窗口均值后做相关损失。
- `si_sdr`：带限 SI-SDR ratio loss，保持 scale-invariant 投影性质。
- `signed_corr + si_sdr`：同时打开 `signed_corr_weight=0.1` 和 `si_sdr_weight=0.1`。

核心结果：

| label | run | best epoch | best val loss | active val component | `rr_peak_band_abs_error` mean | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `signed_cosine` | `20260618_190452_269933` | 17 | 0.642742 | 0.206898 | 0.440934 | 0.454291 | 0.215898 | 0.449247 | 0.795196 | 0.844561 | 4.193630 |
| `signed_corr` | `20260618_190454_720126` | 17 | 0.642674 | 0.207026 | 0.444317 | 0.454291 | 0.215842 | 0.449395 | 0.795174 | 0.844580 | 4.233409 |
| `signed_rms_envelope` | `20260618_191019_767435` | 17 | 0.646991 | 0.269968 | 0.450692 | 0.454291 | 0.214703 | 0.450719 | 0.796385 | 0.845385 | 4.190817 |
| `signed_mean` | `20260618_191019_646156` | 17 | 0.647465 | 0.262790 | 0.448465 | 0.447989 | 0.213418 | 0.445362 | 0.795043 | 0.846075 | 4.003601 |
| `si_sdr` | `20260618_191558_872790` | 14 | 0.665652 | 0.487022 | 0.451273 | 0.448562 | 0.214069 | 0.456016 | -0.799176 | 0.317593 | 2.941830 |
| `signed_corr+si_sdr` | `20260618_191559_193237` | 14 | 0.845285 | signed_corr=1.796819 | 0.443881 | 0.450853 | 0.214486 | 0.455875 | -0.798436 | 0.317561 | 2.595499 |

阶段结论：

- 直接 signed anchor 最符合替代目标。`signed_cosine` 与 `signed_corr` 都把
  seed20260650 从历史 `band_limited_corr=-0.791998` 拉到约 `+0.795`，
  同时 `rr_peak_band_abs_error` 分别为 `0.440934` / `0.444317`。这与 M6
  `phase_alignment_weight=0.10` 的 `0.445811` 相当，甚至 `signed_cosine`
  略好。
- `signed_rms_envelope` 与 `signed_mean` 也能修正方向，但 active component
  停在约 `0.26-0.27`，方向锚定弱于直接 cosine/corr。它们更像相对努力辅助项，
  不像最直接的 phase replacement。
- `si_sdr` 单独不能替代 phase loss。它的 `band_limited_corr=-0.799176`，
  说明标准 SI-SDR 对整体取负近似不敏感，会保留反向盆地。
- `si_sdr + signed_corr` 在当前同权重 `0.1 + 0.1` 下也失败，best epoch 的
  `signed_corr=1.796819`，`band_limited_corr=-0.798436`。这说明 SI-SDR 会
  强烈奖励“形态相似但极性相反”的解；如果后续要使用 SI-SDR，必须显著提高 signed
  anchor 权重或把 SI-SDR 放到方向稳定后的次级阶段。
- 所有成功修正方向的候选都带来 raw `rr_peak_abs_error` 上升，和 M6 一致。当前仍应
  把 raw peak 作为毛刺诊断，而不是主选择指标。

建议：

- 下一轮优先多 seed 复核 `signed_cosine_weight=0.1` 与
  `signed_corr_weight=0.1`，并与 M6 `phase_alignment_weight=0.1` 对照。
- 暂不把 `si_sdr` 作为 phase replacement 主线；它可以作为后续形态重构辅助项，
  但必须在 signed direction 已稳定的前提下使用。

### M7b：`signed_cosine_weight=0.1` 多 seed 展开

沿着 M7 single-seed 最优候选继续展开，本轮只保留 `signed_cosine_weight=0.1`，
旧 `phase_alignment_weight=0.0`。新增 6 个 seed，其中 `20260660`、`20260690`、
`20260710`、`20260720` 是历史反向困难 seed，`20260680`、`20260700` 是正向
guard seed。合并 M7 single-seed 的 `20260650` 后，共 7 个可比 seed。

本地输出：

- `runs/tho_research_v2_patch_hann_m7_signed_cosine_multiseed/`
- `runs/tho_research_v2_patch_hann_m7_signed_cosine_multiseed_summary.csv`
- 临时同 seed 对照：`/tmp/m7_signed_cosine_multiseed_compare.csv`

同 seed 对照结果：

| seed | baseline run | signed cosine run | baseline `rr_peak_band_abs_error` | signed `rr_peak_band_abs_error` | delta | baseline `band_limited_corr` | signed `band_limited_corr` | baseline `best_lag_corr` | signed `best_lag_corr` | baseline raw peak error | signed raw peak error |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260650 | `20260618_150349_945127` | `20260618_190452_269933` | 0.431143 | 0.440934 | +0.009792 | -0.791998 | 0.795196 | 0.319607 | 0.844561 | 2.147589 | 4.193630 |
| 20260660 | `20260618_151631_769707` | `20260618_194634_942860` | 0.445080 | 0.466617 | +0.021537 | -0.796528 | 0.795744 | 0.329242 | 0.844921 | 2.616823 | 4.574583 |
| 20260680 | `20260618_154745_946398` | `20260618_195758_251790` | 0.469799 | 0.461971 | -0.007828 | 0.796917 | 0.796310 | 0.845812 | 0.845570 | 3.483859 | 3.659973 |
| 20260690 | `20260618_152813_981274` | `20260618_194636_773281` | 0.444933 | 0.481485 | +0.036552 | -0.796184 | 0.792269 | 0.326488 | 0.845492 | 2.520622 | 3.439409 |
| 20260700 | `20260618_152811_169836` | `20260618_195800_340726` | 0.435927 | 0.543405 | +0.107478 | 0.794646 | 0.798198 | 0.844050 | 0.846204 | 4.289256 | 4.616028 |
| 20260710 | `20260618_152815_350433` | `20260618_195217_741698` | 0.458545 | 0.543998 | +0.085454 | -0.793236 | 0.792835 | 0.327301 | 0.844445 | 2.787340 | 4.760163 |
| 20260720 | `20260618_154747_423382` | `20260618_195219_459060` | 0.441465 | 0.477011 | +0.035545 | -0.794948 | 0.796995 | 0.322067 | 0.846355 | 3.062251 | 4.435717 |

聚合结果：

- `signed_cosine_weight=0.1` 在 7/7 seed 上保持正低频方向，
  `band_limited_corr` 均值为 `0.795364`，`best_lag_corr` 均值为 `0.845364`。
- 相对同 seed baseline，`rr_peak_band_abs_error` 平均变差 `+0.041219`，
  `best_val_loss` 平均变差 `+0.015338`，raw `rr_peak_abs_error` 平均上升
  `+1.253109`。
- 方向收益非常稳定，但任务指标存在明确代价；尤其 `20260700` 和 `20260710`
  的带通 RR 退化较明显。

阶段判断：

- `signed_cosine_weight=0.1` 是当前证据下最稳定的 phase replacement：它能把
  历史反向 seed 统一拉回正向，且不会破坏已有正向 seed 的方向。
- 但 `0.1` 不是无代价解法。它把方向问题从“随机反相”变成“稳定正向但局部峰值/RR
  更容易受扰”的问题。后续若把它作为默认 loss，需要配套降低 RR 代价。

### M7c：`signed_cosine_weight` 下探

为了寻找更小的 Pareto 权重，本轮只在两个困难 seed 上扫描
`signed_cosine_weight=0.03/0.05/0.075`，并与 `0.1` 对照。

本地输出：

- `runs/tho_research_v2_patch_hann_m7_signed_cosine_weight_sweep/`
- `runs/tho_research_v2_patch_hann_m7_signed_cosine_weight_sweep_summary.csv`
- 临时明细：`/tmp/m7_signed_cosine_weight_sweep_table.csv`

核心结果：

| seed | weight | run | best epoch | best val loss | val signed cosine | `rr_peak_band_abs_error` mean | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `band_limited_corr` mean | direction | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 20260650 | 0.030 | `20260618_200509_853563` | 13 | 0.672067 | 1.791328 | 0.431226 | 0.466894 | 0.215431 | -0.791843 | negative | 0.320492 | 2.177113 |
| 20260650 | 0.050 | `20260618_201119_580806` | 13 | 0.708081 | 1.789947 | 0.430133 | 0.461738 | 0.215327 | -0.790350 | negative | 0.322911 | 2.195995 |
| 20260650 | 0.075 | `20260618_201722_861121` | 11 | 0.634719 | 0.204593 | 0.456162 | 0.451426 | 0.215472 | 0.797383 | positive | 0.847677 | 3.989047 |
| 20260650 | 0.100 | `20260618_190452_269933` | 17 | 0.642742 | 0.206898 | 0.440934 | 0.454291 | 0.215898 | 0.795196 | positive | 0.844561 | 4.193630 |
| 20260710 | 0.030 | `20260618_200510_696319` | 22 | 0.674709 | 1.790830 | 0.449470 | 0.454863 | 0.217223 | -0.792529 | negative | 0.326330 | 2.731036 |
| 20260710 | 0.050 | `20260618_201121_835893` | 17 | 0.711716 | 1.790779 | 0.446561 | 0.447989 | 0.217680 | -0.792655 | negative | 0.322688 | 2.730329 |
| 20260710 | 0.075 | `20260618_201720_525927` | 18 | 0.758007 | 1.787141 | 0.442052 | 0.460592 | 0.217878 | -0.790711 | negative | 0.325890 | 3.071408 |
| 20260710 | 0.100 | `20260618_195217_741698` | 15 | 0.640375 | 0.207371 | 0.543998 | 0.451426 | 0.215190 | 0.792835 | positive | 0.844445 | 4.760163 |

阶段判断：

- `0.03` 与 `0.05` 都不是有效权重：两个困难 seed 均保持反向，
  `val_signed_cosine≈1.79`。
- `0.075` 是 seed-dependent：能修正 `20260650`，但不能修正更难的
  `20260710`。因此它不能作为通用默认权重。
- 当前最小已证实稳定权重仍是 `0.1`。如果继续下探，优先只在
  `0.085/0.09/0.095` 之间细扫 `20260710`，而不是再尝试更低权重。

后续建议：

- 若追求稳定方向，当前默认候选应是 `signed_cosine_weight=0.1`。
- 若追求 Pareto 最优，可以继续做窄区间权重扫描：`0.085/0.09/0.095` on
  `20260710`，然后用 `20260700` 做正向 guard。
- 方向约束已基本回答“能否替代 phase loss”；下一步重点不是继续发明新 loss，
  而是降低 `0.1` 带来的 RR/raw peak 代价，例如 warm-up/curriculum、checkpoint
  selection gate，或后续模型结构里降低局部尖峰自由度。

## M8：signed cosine curriculum 与 checkpoint direction gate

基于 M7c 的结论，直接从低权重 ramp-up 风险较高，因为 `0.03/0.05/0.075`
在困难 seed 上会先进入反向盆地且难以自行翻出。因此 M8 采用反向的课程策略：
**先用 `0.1` 做 direction bootstrap，再线性衰减到较低权重**。同时启用
checkpoint direction gate：

- `training.checkpoint_gate.metric=val_signed_cosine`
- `training.checkpoint_gate.max=0.5`

该 gate 的含义是：只有验证集 signed direction 已经进入正向区间的 epoch 才允许
成为最终 checkpoint；如果训练全程没有通过 gate，则回退到普通 best val loss
checkpoint，避免没有 checkpoint 可评。

本地输出：

- `runs/tho_research_v2_patch_hann_m8_signed_cosine_curriculum/`
- `runs/tho_research_v2_patch_hann_m8_signed_cosine_curriculum_summary.csv`
- 临时对照表：`/tmp/m8_curriculum_compare.csv`

核心结果：

| seed | label | run | best epoch | best val loss | gate passed | val signed cosine | `rr_peak_band_abs_error` mean | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20260700 | `baseline_phase0005` | `20260618_152811_169836` | 11 | 0.620598 | none | NaN | 0.435927 | 0.443979 | 0.215978 | 0.794646 | 0.844050 | 4.289256 |
| 20260700 | `constant_0.1` | `20260618_195800_340726` | 14 | 0.635835 | none | 0.202372 | 0.543405 | 0.436531 | 0.213555 | 0.798198 | 0.846204 | 4.616028 |
| 20260700 | `linear_0.1_to_0.05_e8` | `20260618_203725_395386` | 14 | 0.626382 | True | 0.203238 | 0.543607 | 0.435959 | 0.213965 | 0.797349 | 0.845676 | 4.566183 |
| 20260710 | `baseline_phase0005` | `20260618_152815_350433` | 22 | 0.628415 | none | NaN | 0.458545 | 0.444552 | 0.215785 | -0.793236 | 0.327301 | 2.787340 |
| 20260710 | `constant_0.03` | `20260618_200510_696319` | 22 | 0.674709 | none | 1.790830 | 0.449470 | 0.454863 | 0.217223 | -0.792529 | 0.326330 | 2.731036 |
| 20260710 | `constant_0.05` | `20260618_201121_835893` | 17 | 0.711716 | none | 1.790779 | 0.446561 | 0.447989 | 0.217680 | -0.792655 | 0.322688 | 2.730329 |
| 20260710 | `constant_0.075` | `20260618_201720_525927` | 18 | 0.758007 | none | 1.787141 | 0.442052 | 0.460592 | 0.217878 | -0.790711 | 0.325890 | 3.071408 |
| 20260710 | `constant_0.1` | `20260618_195217_741698` | 15 | 0.640375 | none | 0.207371 | 0.543998 | 0.451426 | 0.215190 | 0.792835 | 0.844445 | 4.760163 |
| 20260710 | `linear_0.1_to_0.05_e8` | `20260618_203118_513376` | 15 | 0.630009 | True | 0.207636 | 0.539610 | 0.452572 | 0.215247 | 0.792714 | 0.844334 | 4.775488 |
| 20260710 | `linear_0.1_to_0.075_e8` | `20260618_203115_098423` | 15 | 0.635060 | True | 0.207153 | 0.549920 | 0.449708 | 0.215049 | 0.793337 | 0.844964 | 4.722806 |

阶段判断：

- Direction bootstrap 成功：`0.1 -> 0.05` 和 `0.1 -> 0.075` 在困难 seed
  `20260710` 上都保持正向，说明早期强 signed cosine 可以把模型推入正确 basin，
  后续降权不会立刻掉回反向。
- 但 curriculum 没有实质降低 RR 代价。对 `20260710`，`0.1 -> 0.05`
  仅把 `rr_peak_band_abs_error` 从常量 `0.1` 的 `0.543998` 小幅降到
  `0.539610`；`0.1 -> 0.075` 反而到 `0.549920`。对正向 guard seed
  `20260700`，`0.1 -> 0.05` 的 `rr_peak_band_abs_error=0.543607`，几乎
  等同于常量 `0.1` 的 `0.543405`。
- checkpoint direction gate 在本轮主要是安全护栏，不是独立解法。低权重失败
  run 的 best epoch `val_signed_cosine≈1.79`，没有可选的正向 checkpoint；
  curriculum run 的 best epoch 本身已经通过 gate，因此 gate 没有改变最终
  checkpoint，只保证不会保存反向 epoch。

结论：

- `signed_cosine_weight=0.1` 仍是当前最稳定方向约束；课程式降权可以保方向，
  但不能明显缓解 `rr_peak_band_abs_error` 和 raw peak 代价。
- checkpoint direction gate 应保留为后续训练的保护机制，尤其用于防止“验证 loss
  选中反向 checkpoint”；但它不能替代训练目标本身。
- 下一步不建议继续在 warm-up/curriculum 上做大网格。更有价值的方向是降低正向
  解中的局部峰值自由度，例如模型输出带限/低自由度 decoder、轻量 anti-spike
  正则，或把 checkpoint 选择从 `val_loss` 转向 `direction gate + task RR`。

## Loss 阶段收口

基于 L0/L1、phase replacement、signed cosine 多 seed、权重下探和 M8
curriculum/gate 结果，当前 loss 探索可以阶段性冻结，不再继续做大规模 loss 网格。

阶段性默认候选：

- `high_freq_weight=0.2`
- `relative_env_weight=0.03`
- `phase_alignment_weight=0.0`
- `signed_corr_weight=0.1`
- `signed_cosine_schedule.mode=none`
- `training.checkpoint_gate.metric=val_signed_cosine`
- `training.checkpoint_gate.max=0.5`

保留这个组合的理由：

- `signed_corr_weight=0.1` 是当前证据下唯一在多 seed 困难样本上稳定修正方向的
  phase replacement。
- 低权重 `0.03/0.05/0.075` 会在困难 seed 上保持反向，不能作为默认候选。
- `0.1 -> 0.05/0.075` curriculum 能保住方向，但没有明显降低
  `rr_peak_band_abs_error` 或 raw peak 代价，因此不作为下一阶段主线默认值。
- checkpoint direction gate 有必要保留为安全护栏，避免验证 loss 选中反向
  checkpoint；但它不能替代训练目标，也不能单独解决 RR 代价。

当前瓶颈已经从“loss 是否能约束方向”转为“方向正确后，模型输出仍有过高局部峰值
自由度”。因此下一阶段主线应转向模型与输出结构：

- 低自由度或带限输出 decoder。
- 降低局部尖峰自由度的模型结构，而不是继续主要依赖 loss 惩罚。
- dual-stream / low-frequency branch 等带明确低频归纳偏置的结构。
- 后续可以评估 `direction gate + task RR` 的 checkpoint 选择策略，但这属于
  selection protocol，不应继续混同为新 loss 方案。

例外条件：

- 如果新模型仍然稳定出现明显局部尖峰，才回到 loss 侧补充轻量 anti-spike 或
  band-limited consistency 正则。
- 如果新输入或新 split 改变了方向统计，再重新验证 `signed_corr_weight=0.1`
  是否仍是有效默认值。

## M9 低自由度模型结构实验

执行计划：`docs/superpowers/plans/2026-06-18-model-m9-lowfreq-structure.md`。

本轮固定 loss 收口结论，只新增模型结构。任务 1-8 已完成并通过两阶段审查：

- 新增 `basis_decoder1d`
- 新增 `multiscale_decomp_mixer1d`
- 新增 `timesnet_lite1d`
- 新增 `frequency_bottleneck1d`
- 新增 `downsampled_ssm1d`

代码阶段提交：

- `045fce3 feat: 添加 M9 低频结构模型候选`
- `c9e0a9d fix: 修正 M9 TimesNet 周期选择`

### M9 smoke

GPU smoke 使用小样本配置：`max_train_windows=32`、`max_val_windows=16`、
`epochs=1`、`batch_size=8`、`use_amp=false`，并启用
`checkpoint_gate.metric=val_signed_cosine` 与 `checkpoint_gate.max=0.5`。

本地输出：

- `runs/tho_research_v2_model_m9_lowfreq_smoke/`
- 临时汇总：`/tmp/m9_lowfreq_smoke_summary.csv`

smoke 结果：

| model | run | best val loss | `rr_peak_band_abs_error` mean | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `band_limited_corr` mean | raw `rr_peak_abs_error` mean |
|---|---|---:|---:|---:|---:|---:|---:|
| `basis_decoder1d` | `20260618_214035_084385` | 1.467416 | 2.829800 | 10.070801 | 0.567285 | -0.015386 | 2.846315 |
| `multiscale_decomp_mixer1d` | `20260618_214035_168052` | 1.002929 | 3.590507 | 1.007080 | 0.472961 | -0.616648 | 5.463063 |
| `timesnet_lite1d` | `20260618_214112_479922` | 1.094991 | 2.297592 | 2.655029 | 0.426429 | -0.203160 | 3.165232 |
| `frequency_bottleneck1d` | `20260618_214115_100615` | 1.404325 | 2.966085 | 13.458252 | 0.476259 | 0.000147 | 2.858006 |
| `downsampled_ssm1d` | `20260618_214141_977332` | 1.033291 | 3.277764 | 6.408691 | 0.292141 | 0.063458 | 4.093035 |

阶段判断：

- 5 个模型均能在 CUDA 上完成最小训练、保存 checkpoint 并生成 `metrics.csv`，
  说明 registry、forward/backward、loss、checkpoint gate 和评价链路可用。
- smoke 只用于工程可运行性验证，不用于选模。小样本 1 epoch 下的 RR 和方向指标
  不解释为科研结论。
- 可以进入 M9 首轮正式训练：Patch-Hann signed baseline + 5 个新模型，
  每个模型先跑困难 seed `20260710` 与 guard seed `20260700`。

### M9 正式首轮：Patch-Hann signed baseline

固定 `patch_mixer1d + overlap_window=hann + output_smoothing_kernel=1`，
使用 loss 收口组合 `signed_corr_weight=0.1`，先跑 M9 同口径 baseline。

| label | run | model | seed | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_patch_hann_seed20260700` | `20260618_214430_858665` | `patch_mixer1d` | 20260700 | 0.635817 | 0.547664 / 0.194709 | 0.436531 | 0.213462 | 0.457882 | 0.798281 | 0.846291 | 4.619530 |
| `baseline_patch_hann_seed20260710` | `20260618_214431_129136` | `patch_mixer1d` | 20260710 | 0.640246 | 0.547049 / 0.198383 | 0.446270 | 0.214987 | 0.452004 | 0.792969 | 0.844906 | 4.790879 |

阶段判断：

- 两个 baseline seed 均通过方向护栏，`band_limited_corr` 约 `0.79`，
  `best_lag_corr` 约 `0.845`。
- 主指标 `rr_peak_band_abs_error_mean` 约 `0.547`，作为 M9 新结构首轮比较的同口径
  baseline。
- raw peak 均值仍约 `4.6-4.8`，保留为局部尖峰/输出自由度诊断问题。

### M9 正式首轮：`basis_decoder1d`

`basis_decoder1d` 用少量低频 Fourier-like basis 直接重建整段 180s 输出，是本轮
自由度最低的候选。

| label | run | model | seed | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `basis_decoder_seed20260700` | `20260618_215117_832352` | `basis_decoder1d` | 20260700 | 1.349573 | 3.218592 / 2.670940 | 3.163994 | 0.268893 | 0.002997 | -0.003746 | 0.084924 | 3.199126 |
| `basis_decoder_seed20260710` | `20260618_215110_312709` | `basis_decoder1d` | 20260710 | 1.351367 | 3.365857 / 2.778772 | 3.164567 | 0.265992 | 0.003646 | 0.001347 | 0.086415 | 3.349572 |

阶段判断：

- `basis_decoder1d` 不通过。两个 seed 的 `rr_peak_band_abs_error_mean` 均超过
  `3.2`，远差于 Patch-Hann signed baseline 的约 `0.547`。
- `relative_envelope_corr` 接近 `0`，`best_lag_corr` 也只有约 `0.085`，说明模型
  基本没有恢复可用的呼吸节律/相对努力结构。
- raw peak 低于 baseline，但这是过低自由度/近似弱输出带来的副作用，不能视为
  抗毛刺成功。
- 结论：单一全局 basis decoder 过强地压低自由度，不进入扩 seed；后续若复用
  basis 思路，需要改成局部/分段 basis 或与多尺度 encoder 结合。

### M9 正式首轮：`multiscale_decomp_mixer1d`

`multiscale_decomp_mixer1d` 使用低通分解后的多尺度分支做 token mixing，目标是保留
Patch-Hann 的节律能力，同时降低普通局部输出尖峰自由度。

| label | run | model | seed | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `multiscale_decomp_mixer1d_seed20260700` | `20260618_215715_456449` | `multiscale_decomp_mixer1d` | 20260700 | 0.772281 | 0.397505 / 0.170943 | 0.445697 | 0.215300 | 0.456124 | -0.778031 | 0.371425 | 1.143390 |
| `multiscale_decomp_mixer1d_seed20260710` | `20260618_215659_191953` | `multiscale_decomp_mixer1d` | 20260710 | 0.618128 | 0.459622 / 0.174155 | 0.462311 | 0.216110 | 0.449103 | 0.795152 | 0.850356 | 1.805841 |

阶段判断：

- `20260710` 困难 seed 通过方向判断，`band_limited_corr=0.795152`、
  `best_lag_corr=0.850356`，与 Patch-Hann signed baseline 同级；同时 raw
  `rr_peak_abs_error_mean` 从 baseline 的约 `4.79` 降到 `1.81`，说明多尺度低频结构
  确实能削弱局部尖峰自由度。
- `20260700` guard seed 的 `rr_peak_band_abs_error_mean=0.397505` 低于 baseline，
  但 `band_limited_corr=-0.778031`，属于方向失败；这一行不能作为胜出证据。
- 结论：`multiscale_decomp_mixer1d` 是有潜力但方向不稳定的候选。下一步若继续，应优先
  研究方向稳定机制或 checkpoint 选择口径，而不是只看 RR 数字扩 seed。

### M9 正式首轮：`timesnet_lite1d`

`timesnet_lite1d` 从低通输入中估计主要周期，再用简化 TimesBlock 做周期折叠建模；
它用于测试“显式周期 inductive bias”是否能改善呼吸节律恢复。

| label | run | model | seed | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `timesnet_lite1d_seed20260700` | `20260618_220553_387526` | `timesnet_lite1d` | 20260700 | 1.072838 | 1.629966 / 0.366087 | 1.893355 | 0.334103 | 0.155524 | 0.246110 | 0.432915 | 4.000236 |
| `timesnet_lite1d_seed20260710` | `20260618_220545_156722` | `timesnet_lite1d` | 20260710 | 1.070854 | 1.589163 / 0.384504 | 1.921426 | 0.345660 | 0.153686 | 0.245518 | 0.430302 | 3.753427 |

阶段判断：

- `timesnet_lite1d` 不通过。两个 seed 的 `rr_peak_band_abs_error_mean` 均约
  `1.6`，明显差于 Patch-Hann signed baseline 的约 `0.547`。
- `band_limited_corr` 只有约 `0.246`，`best_lag_corr` 约 `0.43`，说明当前简化
  TimesBlock 没有学到足够稳定的低频相位/方向结构。
- raw `rr_peak_abs_error_mean` 仍在 `3.75-4.00`，也没有解决普通局部峰值问题。
- 结论：当前轻量 TimesNet 版本不进入扩 seed。若后续重启该方向，应先改周期选择与
  周期 folding 的表达能力，而不是直接增加训练轮数。

### M9 正式首轮：`frequency_bottleneck1d`

`frequency_bottleneck1d` 将输出限制在低频频域系数上，再重建整段波形；它用于测试
“频域瓶颈”能否直接抑制非呼吸高频结构。

| label | run | model | seed | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `frequency_bottleneck1d_seed20260700` | `20260618_221401_396344` | `frequency_bottleneck1d` | 20260700 | 1.292045 | 2.935836 / 2.791221 | 3.164567 | 0.244966 | 0.006218 | 0.001196 | 0.104741 | 2.971419 |
| `frequency_bottleneck1d_seed20260710` | `20260618_221400_071143` | `frequency_bottleneck1d` | 20260710 | 1.293896 | 3.256301 / 3.161118 | 3.709372 | 0.245573 | 0.005933 | -0.002514 | 0.102683 | 3.141821 |

阶段判断：

- `frequency_bottleneck1d` 不通过。两个 seed 的 `rr_peak_band_abs_error_mean`
  分别约 `2.94` 与 `3.26`，远差于 Patch-Hann signed baseline。
- `relative_envelope_corr` 与 `band_limited_corr` 都接近 `0`，说明该结构虽然把
  high-frequency penalty 压到近似 `0`，但同时也压掉了有效呼吸节律和相对努力变化。
- raw peak 误差低于 Patch-Hann，但这是输出自由度过低/弱输出导致的副作用，不能作为
  任务收益。
- 结论：当前直接频域 bottleneck 过硬，不进入扩 seed。若重启 M4 方向，需要改为
  “time encoder + 可学习低频 mask / 局部频域块”，而不是全局低频系数直接解码。

### M9 正式首轮：`downsampled_ssm1d`

`downsampled_ssm1d` 先把 100 Hz 输入压到低采样 latent，再用轻量状态卷积/残差块建模；
它用于测试低时间自由度的 state-style decoder 是否能减少局部尖峰。

| label | run | model | seed | best val loss | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `downsampled_ssm1d_seed20260700` | `20260618_221948_278851` | `downsampled_ssm1d` | 20260700 | 0.777833 | 0.489537 / 0.196713 | 0.457728 | 0.213351 | 0.458868 | -0.803800 | 0.322486 | 0.840981 |
| `downsampled_ssm1d_seed20260710` | `20260618_221949_477007` | `downsampled_ssm1d` | 20260710 | 0.764959 | 0.470890 / 0.199652 | 0.443979 | 0.207432 | 0.469428 | -0.801594 | 0.331623 | 0.816449 |

阶段判断：

- `downsampled_ssm1d` 不能按当前形态进入扩 seed。两个 seed 的
  `rr_peak_band_abs_error_mean` 均低于 Patch-Hann signed baseline，并且 raw
  `rr_peak_abs_error_mean` 从 baseline 的约 `4.6-4.8` 降到约 `0.82-0.84`，
  说明低采样 state-style decoder 明显降低了局部尖峰自由度。
- 但两个 seed 的 `band_limited_corr` 都约 `-0.80`，`best_lag_corr` 只有约
  `0.32-0.33`，属于稳定反向输出；这不是可接受的方向/相位结果。
- 这一组证明：仅靠 `best_val_loss` 或带通 RR 选模型会误选反向模型。后续模型选择必须
  把 `band_limited_corr > 0` 或等价方向护栏作为硬约束，再比较 RR 与 raw peak。
- 结论：`downsampled_ssm1d` 的结构方向值得保留，但需要加入显式 polarity/direction
  修正机制或改 checkpoint gate；当前 run 不作为胜出模型。

### M9 首轮汇总判断（旧 gate 口径）

聚合口径：每个模型 2 个 seed，`usable_direction` 使用
`band_limited_corr_mean >= 0.5` 作为可用方向门槛。这个阈值不是论文指标，只用于避免
把接近 0 的弱相关输出误判为方向通过。

注意：本节表格是在旧 `checkpoint_gate.metric=val_signed_cosine` 口径下生成的历史汇总。
下方“checkpoint gate 修正”已更新有效证据范围；后续选择以修正后的判断为准。

| model | usable direction | best val loss | `rr_peak_band_abs_error` mean | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `patch_mixer1d` | 2/2 | 0.638032 | 0.547356 | 0.441401 | 0.214224 | 0.454943 | 0.795625 | 0.845599 | 4.705204 |
| `multiscale_decomp_mixer1d` | 1/2 | 0.695204 | 0.428564 | 0.454004 | 0.215705 | 0.452614 | 0.008560 | 0.610891 | 1.474615 |
| `downsampled_ssm1d` | 0/2 | 0.771396 | 0.480214 | 0.450853 | 0.210391 | 0.464148 | -0.802697 | 0.327055 | 0.828715 |
| `timesnet_lite1d` | 0/2 | 1.071846 | 1.609564 | 1.907390 | 0.339881 | 0.154605 | 0.245814 | 0.431609 | 3.876832 |
| `frequency_bottleneck1d` | 0/2 | 1.292970 | 3.096068 | 3.436969 | 0.245270 | 0.006076 | -0.000659 | 0.103712 | 3.056620 |
| `basis_decoder1d` | 0/2 | 1.350470 | 3.292225 | 3.164280 | 0.267442 | 0.003322 | -0.001199 | 0.085669 | 3.274349 |

首轮结论：

- 当前可作为稳定 baseline 的仍然是 `patch_mixer1d + Hann overlap + signed_corr`：
  两个 seed 均方向可用，RR 指标稳定，但 raw peak 仍高。
- `multiscale_decomp_mixer1d` 是最值得继续的结构候选：方向通过的 seed 同时保持
  `band_limited_corr` 与 `best_lag_corr`，并显著降低 raw peak；但另一个 seed 反向，
  所以不能直接扩 seed。
- `downsampled_ssm1d` 是强结构信号：raw peak 最低、带通 RR 也好，但两个 seed 都反向。
  它说明低采样 state-style decoder 可以解决局部尖峰自由度，却必须先解决方向选择。
- `basis_decoder1d`、`frequency_bottleneck1d` 和当前 `timesnet_lite1d` 不进入下一轮。

方法学修正：

- 本轮暴露出当前 `checkpoint_gate.metric=val_signed_cosine` 与实际使用的
  `signed_corr_weight=0.1` 不对齐；在 `signed_cosine_weight=0.0` 时，这个 gate 不能
  阻止 SSM 这种低 loss 但反向的 checkpoint。
- 下一轮模型实验应先把 checkpoint/selection 方向护栏改为与 `signed_corr` 或
  `band_limited_corr` 对齐，再比较 `multiscale_decomp_mixer1d` 与
  `downsampled_ssm1d` 的修正版。

### M9 checkpoint gate 修正

问题根因：

- M9 首轮训练使用 `signed_corr_weight=0.1`、`signed_cosine_weight=0.0`。
- 但 checkpoint gate 配置为 `checkpoint_gate.metric=val_signed_cosine`。
- `signed_cosine_weight=0.0` 时，训练记录里的 `val_signed_cosine` 恒为 `0.0`，
  gate 等价于总是通过，无法阻止反向 checkpoint。
- 训练循环还存在一个 fallback：配置了 gate 但没有任何 epoch 通过时，会退回保存
  ungated best checkpoint。这会让反向模型继续产生 `metrics.csv`。

代码修正：

- `checkpoint_gate.metric=auto_direction` 会优先选择当前启用的 signed 方向损失：
  `signed_corr_weight>0` 时使用 `val_signed_corr`，否则使用有效的
  `val_signed_cosine`。
- 显式配置到未启用的 signed 分项时直接报错，避免恒为 0 的假通过。
- 配置了 checkpoint gate 后，若没有任何 epoch 通过 gate，训练会在写出
  `train_history.csv` 后报错，不再保存 ungated checkpoint。

对 M9 首轮历史 run 的影响：

| run | model | seed | old best epoch | old `val_signed_corr` | 新 gate 影响 |
|---|---|---:|---:|---:|---|
| `20260618_214430_858665` | `patch_mixer1d` | 20260700 | 14 | 0.202323 | 同一 checkpoint 仍有效 |
| `20260618_214431_129136` | `patch_mixer1d` | 20260710 | 15 | 0.207389 | 同一 checkpoint 仍有效 |
| `20260618_215659_191953` | `multiscale_decomp_mixer1d` | 20260710 | 11 | 0.208220 | 同一 checkpoint 仍有效 |
| `20260618_215715_456449` | `multiscale_decomp_mixer1d` | 20260700 | 15 | 1.773924 | 无有效 checkpoint，应删除/重跑修正版 |
| `20260618_221948_278851` | `downsampled_ssm1d` | 20260700 | 10 | 1.800903 | 无有效 checkpoint，应删除/重跑修正版 |
| `20260618_221949_477007` | `downsampled_ssm1d` | 20260710 | 14 | 1.798271 | 无有效 checkpoint，应删除/重跑修正版 |
| 其余 `basis_decoder1d` / `frequency_bottleneck1d` / `timesnet_lite1d` | 多个 | - | - | `0.76-1.00` | 无有效 checkpoint，不进入下一轮 |

处理决定：

- M9 首轮旧汇总中，只有 Patch-Hann 两个 seed 与
  `multiscale_decomp_mixer1d_seed20260710` 可继续作为有效证据。
- 其余受错误 gate 污染的 run 不再作为可用模型结果；如果保留本地目录，会继续污染
  `summarize_tho_runs.py` 汇总，因此应从 `runs/tho_research_v2_model_m9_lowfreq_structure/`
  删除。
- 仅修 gate 不需要重跑所有模型；需要重跑的是下一轮“修正版模型/selection”实验，而不是
  这些已知没有有效 checkpoint 的旧配置。

本地清理结果：

- 已删除 9 个新 gate 下无有效 checkpoint 的正式 run：
  `basis_decoder1d` 2 个、`frequency_bottleneck1d` 2 个、`timesnet_lite1d` 2 个、
  `downsampled_ssm1d` 2 个、`multiscale_decomp_mixer1d_seed20260700` 1 个。
- 已重新生成
  `runs/tho_research_v2_model_m9_lowfreq_structure_summary.csv`，当前只包含 3 行有效
  formal 证据：Patch-Hann 两个 seed 与 `multiscale_decomp_mixer1d_seed20260710`。

### M9 direction-fix probe：提高 `signed_corr_weight`

目的：验证 M9 首轮中 `multiscale_decomp_mixer1d` 与 `downsampled_ssm1d` 的方向失败，
究竟是结构天然反向，还是 `signed_corr_weight=0.1` 对低自由度结构不够强。

固定口径：

- 数据与训练参数同 M9 formal。
- `checkpoint_gate.metric=auto_direction`，`checkpoint_gate.max=0.5`。
- 只测试 `multiscale_decomp_mixer1d` 与 `downsampled_ssm1d`。
- 输出目录：`runs/tho_research_v2_model_m9_direction_fix/`。

结果：

| model | seed | `signed_corr_weight` | run | metrics | best val loss | min `val_signed_corr` | `rr_peak_band_abs_error` mean / median | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean / median |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `multiscale_decomp_mixer1d` | 20260700 | 0.15 | `20260619_002233_751520` | yes | 0.624541 | 0.201369 | 0.933992 / 0.185510 | 0.451999 | 0.211132 | 0.452247 | 0.800133 | 0.852742 | 1.323754 / 0.333362 |
| `multiscale_decomp_mixer1d` | 20260700 | 0.20 | `20260619_000936_322874` | yes | 0.637581 | 0.200824 | 0.859055 / 0.182623 | 0.438250 | 0.211379 | 0.450604 | 0.801535 | 0.852442 | 2.171832 / 0.415690 |
| `multiscale_decomp_mixer1d` | 20260710 | 0.20 | `20260619_001618_933963` | yes | 0.637323 | 0.204924 | 0.466144 / 0.172117 | 0.459446 | 0.215463 | 0.450001 | 0.797208 | 0.851231 | 1.987368 / 0.395299 |
| `downsampled_ssm1d` | 20260710 | 0.15 | `20260619_002231_993204` | no | 0.859249 | 1.772933 | - | - | - | - | - | - | - |
| `downsampled_ssm1d` | 20260700 | 0.20 | `20260619_000946_522782` | yes | 0.621465 | 0.189964 | 0.637134 / 0.196784 | 0.460592 | 0.203242 | 0.470012 | 0.812418 | 0.857446 | 1.949801 / 0.480769 |
| `downsampled_ssm1d` | 20260710 | 0.20 | `20260619_001619_133684` | yes | 0.604559 | 0.190954 | 1.248217 / 0.209415 | 0.436531 | 0.201328 | 0.491670 | 0.811895 | 0.856985 | 2.467473 / 0.443262 |

聚合判断：

| model | `signed_corr_weight` | valid n | `rr_peak_band_abs_error` mean | `rr_spec_abs_error` mean | `relative_envelope_mae` mean | `relative_envelope_corr` mean | `band_limited_corr` mean | `best_lag_corr` mean | raw `rr_peak_abs_error` mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `patch_mixer1d` baseline | 0.10 | 2 | 0.547356 | 0.441401 | 0.214224 | 0.454943 | 0.795625 | 0.845599 | 4.705204 |
| `multiscale_decomp_mixer1d` | 0.20 | 2 | 0.662600 | 0.448848 | 0.213421 | 0.450303 | 0.799371 | 0.851837 | 2.079600 |
| `downsampled_ssm1d` | 0.20 | 2 | 0.942675 | 0.448562 | 0.202285 | 0.480841 | 0.812156 | 0.857215 | 2.208637 |

阶段结论：

- 提高到 `signed_corr_weight=0.2` 可以把两个结构都拉回正向盆地，
  `band_limited_corr` 与 `best_lag_corr` 均略高于 Patch-Hann baseline。
- 但 `0.2` 不是直接胜出：`multiscale_decomp_mixer1d` 的带通 RR 均值为
  `0.662600`，差于 Patch-Hann baseline 的 `0.547356`；`downsampled_ssm1d`
  的 RR 代价更高，均值 `0.942675`。
- `0.15` 对 SSM 不够，困难 seed 全程未通过 gate；对 multiscale 能过 gate，
  但 guard seed 的 RR 均值 `0.933992`，比 `0.2` 更差。
- 两个结构都显著降低 raw peak，说明低自由度结构确实能控制局部尖峰；
  但当前 loss/selection 下，这个收益会以 RR 代价换来。

下一步建议：

- 不继续单纯加大或下探 `signed_corr_weight`。
- 优先保留 `multiscale_decomp_mixer1d` 作为结构候选，但需要新的 checkpoint selection
  或轻量 anti-spike 策略，而不是继续只调方向权重。
- `downsampled_ssm1d` 暂不扩 seed；它证明 state-style decoder 有降尖峰价值，但当前
  RR 代价太高。
