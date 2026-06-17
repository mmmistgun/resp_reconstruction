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
- 训练：`epochs=50`，`batch_size=8`，`learning_rate=0.001`
- 早停：`patience=8`，`min_delta=0.001`
- seed：`training.seed=20260610`，`data.train_sample_seed=20260610`，`data.val_sample_seed=20260611`
- L0 anchor：`runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320`
- L0：历史 anchor，`loss.band_waveform_weight=0.0`
- L1a：`loss.band_waveform_weight=0.05`
- L1b：`loss.band_waveform_weight=0.10`
- `loss.high_freq_weight=0.2`
- `loss.relative_envelope_weight=0.01`
- `loss.phase_lag_weight=0.0`

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

| run | source | model | data windows | band waveform | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean / median | `relative_envelope_mae` mean / median | `spectrum_similarity` mean | `band_limited_corr` mean | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `20260616_005516_616320` | history anchor | `patch_mixer1d` | full | 0.00 | 0.624477 | 0.651706 / 0.278480 | 0.443979 / 0.000000 | 0.219451 / 0.184972 | 0.975401 | -0.790210 | L0 anchor，作为 L1 任务指标对照。 |

## L1

| run | model | data windows | band waveform | best val loss | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean / median | `relative_envelope_mae` mean / median | `spectrum_similarity` mean | `band_limited_corr` mean | 相对 L0 判断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `20260617_153802_524411` | `patch_mixer1d` | full | 0.05 | 0.650255 | 2.060745 / 0.445540 | 0.430803 / 0.000000 | 0.222524 / 0.190867 | 0.974725 | 0.789400 | 波形诊断改善，但 `rr_peak_abs_error` 明显恶化，不通过。 |
| `20260617_155532_175711` | `patch_mixer1d` | full | 0.10 | 0.665458 | 0.932913 / 0.260365 | 0.471477 / 0.000000 | 0.222588 / 0.190387 | 0.974869 | 0.794815 | 波形诊断改善，但 RR 与相对包络整体不优于 L0，不通过。 |

## 阶段判断

- 波形诊断：L1a 的 `band_limited_corr` 从 `-0.790210` 上升到 `0.789400`，`best_lag_corr` 从 `0.331171` 上升到 `0.841062`，`abs(best_lag_sec)` 从 `1.000000` 变小到 `0.165209`。L1b 也有类似改善，`band_limited_corr=0.794815`，`best_lag_corr=0.844943`，`abs(best_lag_sec)=0.160113`。
- RR 任务指标：L1a 的 `rr_peak_abs_error` mean 从 `0.651706` 恶化到 `2.060745`，L1b 恶化到 `0.932913`。虽然 L1a 的 `rr_spec_abs_error` mean 小幅下降到 `0.430803`，但 peak RR 恶化幅度过大，不能视为任务收益。
- 相对包络：L1a 的 `relative_envelope_corr` 从 `0.440967` 上升到 `0.444807`，但 `relative_envelope_mae` 从 `0.219451` 变大到 `0.222524`；L1b 的 `relative_envelope_corr` 上升到 `0.445248`，但 `relative_envelope_mae` 变大到 `0.222588`。这说明相关性略有改善，但误差没有同步改善。
- 诊断图结论：尚未生成诊断图，证据不足。
- 当前判断：L1 证明带限波形损失能显著改变低频形态，但没有证明它改善了当前任务目标。L1a/L1b 都不应作为通过实验进入模型结构结论；后续若继续做 L1c，应把 `rr_peak_abs_error` 设为主护栏，且只尝试更小权重或改成不主导 RR 峰检测的弱约束。
- 训练效率：后续若使用 `batch_size=128` 提升 GPU 利用率，应标记为新的效率口径或新 baseline，不应和本页 `batch_size=8` 历史 L0/L1 当作唯一变量对照。
