# 稳定发现

本文件只记录跨阶段稳定结论；详细证据仍放在 `docs/experiments/*.md`。

## THO research v2 soft-z

- 当前横向比较默认只看 soft-z research v2 主线；旧 rawish state-aligned 结果只作结构或失败模式参考。
- 当前默认 STFT anchor 倾向 `G3_C_wide_8p0`。
- `G3_C_bandenergy` 只保留为选择性修正候选，暂不替代宽频 STFT anchor。
- 旧 E/F 系列路线不再作为默认扩展方向；下一步优先围绕难恢复窗口和选择性修正设计。

## 指标与选择

- 当前排序优先看 `rr_peak_band_robust_abs_error` 与 `breath_count_zero_cross_abs_error`，其他频域、形态、包络和 lag-aware 指标主要用于解释。
- 局部呼吸率后续围绕新版局部 RR 指标展开；旧局部 RR 算法已淘汰，不进入当前 summary、证据账本或汇报正文。
- 训练 / 验证损失只作训练诊断；模型选择必须结合任务指标、checkpoint 选择口径和 held-out test 复评。

## 重评边界

- 指标、mask、target feature cache 或 checkpoint 选择口径变化后，旧 run 若要参与当前讨论，应按当前口径重评或重训。
- 被淘汰的指标和路线只保留一处归档记录，不在当前文档体系中反复解释。
