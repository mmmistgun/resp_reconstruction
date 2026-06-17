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

| run | source | model | data windows | band waveform | phase lag | rel env | high freq | best val loss | band_limited_corr mean | best_lag_corr mean | abs(best_lag_sec) mean | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260616_005516_616320` | history anchor | `patch_mixer1d` | full | 0.00 | 0.00 | 0.01 | 0.20 | 0.624477 | -0.790210 | 0.331171 | 1.000000 | L0 anchor，作为 L1 对照。 |

## L1

| run | model | data windows | band waveform | phase lag | rel env | high freq | best val loss | band_limited_corr mean | best_lag_corr mean | abs(best_lag_sec) mean | 相对 L0 判断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260617_153802_524411` | `patch_mixer1d` | full | 0.05 | 0.00 | 0.01 | 0.20 | 0.650255 | 0.789400 | 0.841062 | 0.165209 | 待任务 5 汇总判断。 |
| `20260617_155532_175711` | `patch_mixer1d` | full | 0.10 | 0.00 | 0.01 | 0.20 | 0.665458 | 0.794815 | 0.844943 | 0.160113 | 待任务 5 汇总判断。 |

## 阶段判断

尚未执行。
