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
