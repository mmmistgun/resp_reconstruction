# 呼吸重建评价指标设计

## 背景

当前 THO 呼吸重建任务的目标不是逐点复制胸带波形，而是恢复有生理意义的呼吸率、低频形态和相对呼吸强弱变化。已有 `rr_spec_abs_error`、`rr_peak_abs_error`、`rr_peak_band_abs_error` 和 `rr_peak_band_robust_abs_error` 主要是窗口级 dominant rhythm 指标，会把局部呼吸变化、逐呼吸对齐和形态细节压缩成整窗主周期。

新评价口径应避免静默改变旧结果含义。旧指标继续保留用于追溯和兼容，新增指标先作为 test/diagnostic 扩展汇总；只有在复核波形图和分层结果后，才把新增指标升为正式选模主指标。

## 任务目标

评价体系优先回答四个问题：

1. 呼吸率是否一致，且不只是一整窗主频一致。
2. 低频形态是否一致，并能区分形态错误和固定时延。
3. 相对呼吸强弱变化是否一致，包括上沿、下沿和幅度带宽变化。
4. 低质量信号下是否恢复了合理呼吸次数和逐呼吸事件。

## 指标分层

### 核心呼吸率护栏

保留 `rr_peak_band_robust_abs_error` 作为整窗全局 RR 护栏。它抗伪峰能力强，适合防止模型偏离基本呼吸节律，但不能单独代表细节恢复。

保留 `rr_peak_band_abs_error` 用于历史口径对接。它不应长期作为唯一主指标，因为普通峰检测更容易受伪峰影响，而 robust 版本又会主动平滑局部细节。

`rr_spec_abs_error` 降级为频域安全检查。它可以确认频域主呼吸率没有明显恶化，但功率谱最大峰会偏向最强或最常见呼吸节律，不适合作为核心排序指标。

### 低频形态与时延

保留 `band_limited_corr`，作为 zero-lag 低频形态一致性指标。它回答“模型在原时间轴上是否同步恢复了目标形态”。

新增或显式输出 `best_lag_corr_4s` 和 `best_lag_sec_4s`，搜索范围为 `+-4s`。`best_lag_corr_4s` 是核心辅助指标，回答“允许合理时延后形态是否恢复”；`best_lag_sec_4s` 用于解释延迟方向和大小，不单独作为越大越好的指标。

zero-lag 和 lag-aligned 指标必须同时保留。`band_limited_corr` 低但 `best_lag_corr_4s` 高，表示形态可恢复但存在时延；二者都低才支持“形态没有恢复”的判断。

### 相对变化与包络

保留现有 `relative_envelope_mae` 和 `relative_envelope_corr`，但明确其局限：当前 RMS envelope 更接近强度包络，不区分上包络、下包络和中线变化。

新增双边包络指标：

- `upper_envelope_corr` / `upper_envelope_mae`
- `lower_envelope_corr` / `lower_envelope_mae`
- `envelope_band_corr` / `envelope_band_mae`，其中 band 表示 `upper - lower`
- `envelope_midline_corr` / `envelope_midline_mae`，其中 midline 表示 `(upper + lower) / 2`

其中 `envelope_band_*` 最贴近相对呼吸强弱变化，应优先进入主表；`envelope_midline_*` 更偏基线漂移诊断。

形态和包络指标应提供 zero-lag 与 lag-aligned 两类口径。主表保留 zero-lag 结果；lag-aligned 结果使用 `best_lag_sec_4s` 对齐后计算，并加 `_lag4s` 后缀，用于判断形态本身与时间延迟的贡献。

### 呼吸次数与逐呼吸事件

保留 `breath_count_zero_cross_abs_error` 作为轻量护栏。它能体现低质量信号下是否恢复出合理周期数量，但不评价每次呼吸是否对齐。

优化 zero-cross 计数输出，除现有 cycle count 外，增加：

- `pred_breath_count_zero_cross_up`
- `target_breath_count_zero_cross_up`
- `pred_breath_count_zero_cross_down`
- `target_breath_count_zero_cross_down`

新增逐呼吸事件指标：

- `breath_event_precision`
- `breath_event_recall`
- `breath_event_f1`
- `breath_timing_mae_sec`
- `cycle_rr_mae`
- `cycle_rr_corr`

事件匹配在带通信号上进行，默认以 target 事件为参照，使用与目标周期相关的容忍窗口。该组指标用于补上“整窗 RR 正确但局部呼吸变化没跟上”的盲点。

## 旧指标处理

`rr_peak_abs_error` 保留为 raw peak 诊断，只用于检查局部尖峰、坏段 mask 和原始峰检测污染，不进入主排序。

`rr_peak_unmasked_abs_error` 只用于复核坏段或 mask 影响，默认不进入主汇总表。

`spectrum_similarity` 保留为频谱分布诊断和分层变量，不作为主排序指标。

`best_lag_corr` / `best_lag_sec` 的现有 `+-1s` 口径保留用于旧结果兼容；新口径以显式命名的 `best_lag_corr_4s` / `best_lag_sec_4s` 输出，避免和旧表混淆。

## 推荐主表列

下一版 test summary 的主表建议包含：

- `rr_peak_band_robust_abs_error_mean`
- `rr_peak_band_robust_abs_error_median`
- `breath_count_zero_cross_abs_error_mean`
- `relative_envelope_mae_mean`
- `relative_envelope_corr_mean`
- `envelope_band_mae_mean`
- `envelope_band_corr_mean`
- `band_limited_corr_mean`
- `best_lag_corr_4s_mean`
- `best_lag_sec_4s_median`
- `local_rr_mae_mean`
- `local_rr_corr_mean`
- `breath_event_f1_mean`
- `breath_timing_mae_sec_median`
- `cycle_rr_mae_mean`

旧指标进入 diagnostic 表，避免主结论被 dominant rhythm 或历史兼容指标冲淡。

## 验证策略

实现新增指标后，先对已有 `runs/test_eval_g_series_20260705_robust_v1` 生成旁路扩展表，不覆盖原 CSV。需要确认：

1. 新增指标不会改变旧指标数值。
2. `best_lag_corr_4s` 与 `best_lag_sec_4s` 能把形态差和时延差分开。
3. 双边包络指标在人工复核窗口中能反映上沿、下沿和幅度带宽变化。
4. 逐呼吸事件指标能识别漏呼吸、多呼吸和局部 RR 漂移。
5. 新指标分层结论不被少数坏 target 或低质量窗口主导；必要时按 `rr_peak_valid_ratio`、质量标签和 baseline hard/easy 分层报告。
