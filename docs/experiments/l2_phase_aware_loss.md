# L2 Phase-Aware Physiological Loss 实验

## 实验问题

L1 低频波形损失没有整体改善低频相关或最佳 lag 相关。本实验验证 phase-invariant alignment loss 是否能在允许小范围时移的前提下，提高胸带呼吸波形的相位连续性和局部形态。

## 对照 Run

| 实验 | run | phase_lag_weight | band_waveform_weight | best epoch | best val loss | band_limited_corr mean | best_lag_corr mean | abs_best_lag_sec mean | rr_peak_abs_error mean | spectrum_similarity mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L0 | `runs/tho_research_v2/20260617_120905_055953` | 0.0 | 0.0 | 3 | 0.611827 | 0.783169 | 0.843794 | 0.173818 | NaN | NaN | L2 对照基线 |
| L2a | 未运行 | 0.2 | 0.0 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | 未运行 |
| L2b | 未运行 | 0.2 | 0.05 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | 仅 L2a 通过后运行 |

## 判定规则

- L2a 需要提升 `best_lag_corr`，或在不降低 `band_limited_corr` 的前提下降低 `abs_best_lag_sec`。
- 若 L2a 只降低 train/val loss，但 `best_lag_corr`、`band_limited_corr` 和诊断图无改善，不进入 L2b。
- 若 L2a 改善 lag-aware 指标但 RR 或 spectrum 明显恶化，先调低 `phase_lag_weight`，不直接进入 L3。
- L3 cycle consistency 只在 L2 有明确收益后规划。
