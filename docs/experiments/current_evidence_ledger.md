# 当前证据账本

日期：2026-07-08

## 定位

本文只记录当前 THO research v2 soft-z 主线仍影响下一步判断的证据。它不是完整实验
库存，也不为旧结果维护横向可比性。指标和重评边界见
`docs/experiments/metric_schema.md`。

状态含义：

- `active_current`：当前问题仍需要它，且已有证据满足当前整理口径。
- `needs_reeval_for_active`：当前问题仍需要它，但还要按当前指标或 checkpoint 选择口径重评。
- `needs_retrain_for_active`：当前问题仍需要它，旧 checkpoint 不足以回答当前问题。
- `reference_only`：只作为结构想法、失败模式或旧结论解释。
- `discard_for_current_question`：不再为当前问题投入整理、重评或重训成本。

## 当前主问题

当前最该回答的问题不是“哪个历史实验最好”，而是：

1. STFT 输入默认 anchor 应收口到哪个时间/频带参数。
2. 高频或分频带信息是否值得作为选择性修正，而不是无约束主分支。
3. 哪些路线已经可以停止，避免继续堆复杂结构。

## Active Shortlist

| 对象 | 状态 | 当前判断 | 证据路径 | 下一步 |
|---|---|---|---|---|
| `G3_C_wide_8p0` | `active_current` | 当前最稳默认 STFT 输入：`win2000/hop250 + 0.05-8Hz + conv2d + native_inject pre_mixer`。普通 checkpoint、validation top-k、held-out robust test 和 2026-07-09 local RR 复评都支持它优于 time-only / F0 anchor。 | `docs/experiments/g_series_stft_input_resolution_band_plan.md`；`runs/g_series_stft_input_g3_summary.csv`；`runs/g_series_stft_input_g3_paired_delta.csv`；`runs/g_series_stft_input_topk_best_by_rr.csv`；`runs/test_eval_g_series_20260705_robust_v1/g_series_robust_v1_label_summary.csv`；`runs/test_eval_g_series_20260709_local_rr_canonical/g_series_local_rr_canonical_label_summary.csv`；`runs/test_eval_g_series_20260709_local_rr_canonical/g_series_local_rr_canonical_delta_summary.csv` | 作为当前默认 anchor；后续若继续模型结构研究，应以它为 baseline。 |
| `G3_C_bandenergy` | `active_current` | 唯一仍有讨论价值的分频带/低维编码候选。2026-07-09 local RR 复评中 breath count 和 local RR 略优于 `G3_C_wide_8p0`，但 robust RR 与 lag-aware 形态不如 wide anchor，不能升级为默认。 | `docs/experiments/g_series_stft_input_resolution_band_plan.md`；`runs/g_series_stft_input_g3_summary.csv`；`runs/g_series_stft_input_g3_paired_delta.csv`；`runs/g_series_stft_input_topk_best_by_rr.csv`；`runs/test_eval_g_series_20260705_robust_v1/g_series_robust_v1_label_summary.csv`；`runs/test_eval_g_series_20260709_local_rr_canonical/g_series_local_rr_canonical_label_summary.csv`；`runs/test_eval_g_series_20260709_local_rr_canonical/g_series_local_rr_canonical_delta_summary.csv` | 只在要验证“选择性/条件注入”时保留；不扩大全矩阵，不直接作为 H 系列 anchor。 |
| `F-D0_high_stft_anchor` / `F-D2_high_cwt_modulation` | `reference_only` | 高频上下文对 hard/low-spectrum 窗口有局部信号，但 easy/fast-RR 护栏失败；不能升级主线。若后续研究选择性 gating，可作为局部信号来源参考。 | `docs/experiments/f_series_stft_loss_plan.md`；`runs/f_d_highfreq_summary.csv`；`runs/f_d_highfreq_paired_delta.csv`；`runs/f_d_highfreq_topk_best_by_rr.csv`；`runs/f_d_feature_extractor_summary.csv` | 不扩 seed，不继续 CWT/SST dense path。只有明确做无泄漏 hard/easy proxy 时再引用。 |

## 2026-07-09 local RR 口径复评

本次只复评当前 active G 系列代表 checkpoint，不做全历史重评，也不重训。范围来自
`configs/eval_specs/g_series_test_eval_20260705.csv`：`g0_time_only`、
`g0_f0_native_stft_pre_mixer`、`g3_c_wide_8p0`、`g3_c_bandenergy` 各 3 个 seed，
输出在 `runs/test_eval_g_series_20260709_local_rr_canonical/`。

完整性检查通过：12 个 `metrics/summary/manifest` 全部存在，逐窗口 metrics 包含
`rr_peak_band_robust_abs_error`、4s lag-aware envelope/corr 和 2026-07-09 后当前
`local_rr_*` 列。

| 对象 | robust RR mean | count error mean | lag4 envelope corr mean | lag4 corr mean | local RR MAE mean | local RR corr mean | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| `g0_time_only` | 0.773193 | 2.334776 | 0.596490 | 0.865265 | 0.786678 | 0.663619 | 当前参照。 |
| `g0_f0_native_stft_pre_mixer` | 0.733281 | 2.350216 | 0.611588 | 0.868751 | 0.791792 | 0.660398 | RR/形态优于 time-only，但 count 和 local RR 不改善。 |
| `g3_c_wide_8p0` | 0.712186 | 2.150216 | 0.613269 | 0.870774 | 0.780002 | 0.664304 | robust RR 与 lag-aware 形态最稳，继续作为默认 anchor。 |
| `g3_c_bandenergy` | 0.729792 | 2.054401 | 0.607602 | 0.867238 | 0.773299 | 0.665137 | count 和 local RR 略好，但整体稳定性不如 wide anchor。 |

相对 `g0_time_only`，`g3_c_wide_8p0` 的 robust RR mean 改善 -0.061007，
count error 改善 -0.184560，local RR MAE 改善 -0.006675；`g3_c_bandenergy`
的 count error 改善更大（-0.280375），local RR 也略好，但 robust RR 与
lag-aware 形态收益较弱。

## 已停止路线

| 路线 | 状态 | 停止原因 | 证据路径 |
|---|---|---|---|
| E4 SST 输入 | `discard_for_current_question` | 任务指标无收益；只保留为少数 hard 窗口可视化/分离度线索。 | `docs/experiments/time_frequency_input_fusion_plan.md`；`runs/e4_sst_vs_b2_paired_delta.csv` |
| E5 gated / cross-attention 融合 | `discard_for_current_question` | 重复“频谱小改善、peak-band/count 变差”模式；不继续扩 seed 或做 freeze/unfreeze。 | `docs/experiments/time_frequency_input_fusion_plan.md`；`runs/e5_a0_manifest.csv`；`runs/e5_a1_manifest.csv`；`runs/e5_a2_manifest.csv` |
| F-A target STFT loss 系列 | `reference_only` | hard/low-spectrum 有局部正信号，但 easy/fast 护栏系统性失败；不作为默认训练目标。 | `docs/experiments/f_series_stft_loss_plan.md`；`runs/f_a_stft_loss_summary.csv`；`runs/f_a2_*_summary.csv` |
| F-B auxiliary / residual 扩展 | `discard_for_current_question` | residual / Enc3 / energy cap 未缓解 baseline easy 误伤，且更强 head 继续放大退化。 | `docs/experiments/f_series_stft_loss_plan.md`；`runs/f_b2_residual_summary.csv`；`runs/f_b_enc3_residual_summary.csv` |
| F-C low-complex STFT 主输出 | `discard_for_current_question` | 低频 complex-STFT 主输出大幅退化；方向 gate 放宽后也只补齐 metrics，不代表通过。 | `docs/experiments/f_series_stft_loss_plan.md`；`runs/f_c_stft_output_summary.csv` |
| F-D CWT / modulation 特征提取网络扩展 | `discard_for_current_question` | 更强 dense CWT encoder 和 modulation residual TCN 没有缓解 easy/fast 误伤；不继续加更强 CWT-CNN、attention pooling 或 SST dense path。 | `docs/experiments/f_series_stft_loss_plan.md`；`runs/f_d_feature_extractor_summary.csv` |

## 旧结果处理

- 2026-06-20 soft-z E0 纯时序候选作为 `reference_only`：它证明低 val loss 不等于任务好，并提供 patch/multiscale baseline 语境；不再按当前指标批量重评，除非某个模型被重新拉回当前主线。
- 旧 rawish state-aligned 分支默认不进入当前证据账本：只在需要解释极性盆地、结构想法或失败模式时按需读取。
- 早期 `tho_small` 与 2026-06-17/18 rawish 实验默认不整理；不为它们补当前指标或 summary。

## 当前建议动作

1. 暂不全量重训，也不全量重评历史 run。
2. 当前默认 anchor 收口为 `G3_C_wide_8p0`。
3. 暂不继续 validation top-k 全面复评；只有当 held-out test 结论与 validation 选择口径发生冲突，或准备把新排序链写入 selection/summarizer 时，再做旁路 top-k 复评。
4. 若要开新方向，应围绕“无泄漏 hard/ambiguous window proxy + 选择性修正”设计，不再扩大 STFT 频带/编码、CWT/SST dense path 或 cross-attention。
