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

1. `rr_peak_abs_error`：主护栏，均值和中位数不应明显恶化。
2. `rr_spec_abs_error`：频域呼吸率护栏，不应明显恶化。
3. `relative_envelope_mae` / `relative_envelope_corr`：相对呼吸强弱变化的任务指标。
4. `spectrum_similarity`：频谱一致性辅助护栏。
5. `band_limited_corr`、`best_lag_corr`、`best_lag_sec`：波形诊断指标，用于解释低频形态和时移，不能单独判定通过。

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
