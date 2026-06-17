# L2 Phase-Aware Physiological Loss 实验

## 实验问题

L1 低频波形损失没有整体改善低频相关或最佳 lag 相关。本实验验证 phase-invariant alignment loss 是否能在允许小范围时移的前提下，提高胸带呼吸波形的相位连续性和局部形态。

## 对照 Run

| 实验 | run | phase_lag_weight | band_waveform_weight | best epoch | best val loss | band_limited_corr mean | best_lag_corr mean | abs_best_lag_sec mean | rr_peak_abs_error mean | spectrum_similarity mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L0 | `runs/tho_research_v2/20260617_120905_055953` | 0.0 | 0.0 | 3 | 0.611827 | 0.783169 | 0.843794 | 0.173818 | NaN | NaN | L2 对照基线 |
| L2a | `runs/tho_research_v2_l2_phase_lag_020/20260617_134612_266433` | 0.2 | 0.0 | 3 | 0.662768 | 0.761349 | 0.839422 | 0.205918 | 2.231080 | 0.965749 | 未通过，不进入 L2b |
| L2b | 未运行 | 0.2 | 0.05 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | 仅 L2a 通过后运行 |

## 判定规则

- L2a 需要提升 `best_lag_corr`，或在不降低 `band_limited_corr` 的前提下降低 `abs_best_lag_sec`。
- 若 L2a 只降低 train/val loss，但 `best_lag_corr`、`band_limited_corr` 和诊断图无改善，不进入 L2b。
- 若 L2a 改善 lag-aware 指标但 RR 或 spectrum 明显恶化，先调低 `phase_lag_weight`，不直接进入 L3。
- L3 cycle consistency 只在 L2 有明确收益后规划。

## L2a 结果记录

L2a 使用 `phase_lag_weight=0.2`、`phase_lag_max_sec=1.0`、`phase_lag_step_sec=0.1`、`phase_lag_temperature=0.05`，训练 3 个 epoch。run 目录：`runs/tho_research_v2_l2_phase_lag_020/20260617_134612_266433`。

与 L0 对照相比，L2a 的 `band_limited_corr` 从 `0.783169` 降到 `0.761349`，`best_lag_corr` 从 `0.843794` 降到 `0.839422`，`abs_best_lag_sec` 从 `0.173818` 升到 `0.205918`。这说明当前 soft-lag phase loss 虽然降低了训练侧 phase-lag 分量，但没有改善验证集的低频形态或 lag-aware 指标。

当前判定：

- 不进入 L2b，即暂不叠加 `band_waveform_weight=0.05`。
- 不进入 L3 cycle consistency。
- 后续若继续探索 phase-aware loss，应优先降低 `phase_lag_weight` 或调整 lag 聚合温度，而不是直接加更多 waveform 约束。
