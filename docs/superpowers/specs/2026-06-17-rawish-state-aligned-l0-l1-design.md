# Rawish State-Aligned L0/L1 实验设计

## 背景

2026-06-17 的 L0/L1/L2 pilot 已作废。旧 run 的 `data.bcg_input_key`
实际走到了 `bcg_input_aligned_key`，在 research v2 索引中解析为
`bcg_resp_band_state_aligned`。这相当于使用呼吸频段且 state-aligned 的 BCG
表征，不是 rawish wideband BCG 主输入。

本轮重新从 L0/L1 开始，使用 `bcg_rawish_wideband_state_aligned`。选择这个
输入是为了先忽略由两台采集设备采样率或时钟漂移造成的相位偏差。该偏差不是
BCG 到胸带呼吸之间的生理相位差，不应由 L0/L1 模型或 phase-tolerant loss
承担。

## 目标

建立一个干净的 L0/L1 实验口径，回答：

- 在 state-aligned rawish wideband BCG 输入下，当前模型和默认 loss 的基线
  表现如何。
- 加入 band-limited waveform loss 后，低频呼吸波形形态是否改善。
- 残余 lag 是否仍然显著，是否值得进入后续 L2 phase-aware loss。

## 非目标

- 不在本轮训练中开启 `phase_lag_weight`。
- 不改模型结构。
- 不重新生成数据集。
- 不把 `runs/` 产物提交到 Git。
- 不用 `bcg_resp_band_state_aligned` 解释 rawish BCG 主线效果。

## 实验口径

公共配置：

- 数据集：`research_v2_waveform`
- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 训练窗口：`4096`
- 验证窗口：`1024`
- 训练 seed：`20260610`
- 验证 seed：`20260611`
- 模型：沿用当前 `configs/tho_research_v2.yaml` 默认模型
- `loss.phase_lag_weight=0.0`
- `outputs.run_root=runs/tho_research_v2`

L0：

- `loss.band_waveform_weight=0.0`
- 目的：建立 state-aligned rawish 输入下的重新基线。

L1a：

- `loss.band_waveform_weight=0.05`
- 目的：低风险测试带限波形约束是否有收益。

L1b：

- `loss.band_waveform_weight=0.10`
- 目的：观察收益是否随权重增强。

L1c 可选：

- `loss.band_waveform_weight=0.20`
- 触发条件：L1a 或 L1b 在 RR、相对包络和频谱一致性上没有明显恶化，同时波形诊断显示低频形态有收益。

## 指标与判定

主指标：

- `rr_peak_abs_error`：主护栏，均值和中位数不应明显恶化。
- `rr_spec_abs_error`：频域呼吸率护栏，不应明显恶化。
- `relative_envelope_mae` / `relative_envelope_corr`：用于判断相对呼吸强弱变化。
- `spectrum_similarity`：用于判断频谱一致性。

辅助指标：

- `envelope_corr`
- `band_limited_corr`
- `best_lag_corr`
- `abs(best_lag_sec)`
- 诊断图中的低频波形形态

通过标准：

- L1 相比 L0 的 `rr_peak_abs_error` 和 `rr_spec_abs_error` 不明显恶化。
- `relative_envelope_mae` / `relative_envelope_corr` 至少不显示整体退化。
- `spectrum_similarity` 保持稳定。
- 诊断图和波形诊断指标支持低频形态解释，而不是只靠训练 loss 下降。

失败标准：

- `rr_peak_abs_error` 明显恶化，即使 `band_limited_corr` 提升也不通过。
- `rr_spec_abs_error`、相对包络或频谱一致性出现系统性退化。
- 波形诊断改善无法转化为任务指标收益。
- 可视化显示预测仍主要保留 BCG 尖峰结构，或只变平滑而没有呼吸任务收益。

## 独立性与记录

每轮实验前运行 split 独立性审计。若 train/val 存在 `samp_id` 或 segment 重叠，
结果只作为 within-subject 开发指标，不作为跨个体泛化结论。

每个完成的实验或紧密对照组需要提交一次仓库状态。`runs/` 只保留本地，不提交
到 Git；需要提交的是配置、计划和小型人工实验台账。

## 后续分支

若 L1a/L1b 有稳定收益，再考虑：

- L1c：`band_waveform_weight=0.20`
- 更小 spectrum 权重的组合实验
- L2：phase-aware loss，但仅在 state-aligned 后仍有稳定残余 lag 时进入
