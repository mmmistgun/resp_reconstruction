# Rawish State-Aligned L0/L1 实验记录

本文档记录 `bcg_rawish_wideband_state_aligned -> tho_waveform_ref` 口径下的
L0/L1 结果。原始 run、checkpoint、metrics 和图片保留在本地 `runs/`，不进入
Git。

## 实验假设

本轮先忽略由两台采集设备采样率或时钟漂移造成的相位偏差，因为该偏差不是
BCG 到胸带呼吸之间的生理相位差。训练输入固定为
`bcg_rawish_wideband_state_aligned`，`phase_lag_weight` 保持 `0.0`。

## 固定口径

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 训练窗口：`4096`
- 验证窗口：`1024`
- 训练 seed：`20260610`
- 验证 seed：`20260611`
- L0：`loss.band_waveform_weight=0.0`
- L1a：`loss.band_waveform_weight=0.05`
- L1b：`loss.band_waveform_weight=0.10`
- `loss.phase_lag_weight=0.0`

## Split 独立性审计

- 口径：`4096/1024`
- 输入：`bcg_rawish_wideband_state_aligned`
- train/val `samp_id` 重叠：`0`
- train/val `segment` 重叠：`0`
- 结论：本轮 pilot 可作为 leave-samp_id-out 开发指标。

## L0

尚未执行。

## L1

尚未执行。

## 阶段判断

尚未执行。
