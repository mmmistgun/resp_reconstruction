# 呼吸重建评价指标设计

## 背景

当前 THO 呼吸重建任务的目标不是逐点复制胸带波形，而是恢复有生理意义的呼吸率、低频形态和相对呼吸强弱变化。已有 `rr_spec_abs_error`、`rr_peak_abs_error`、`rr_peak_band_abs_error` 和 `rr_peak_band_robust_abs_error` 主要是窗口级 dominant rhythm 指标，会把局部呼吸变化、逐呼吸对齐和形态细节压缩成整窗主周期。

新评价口径应避免静默改变旧结果含义。旧指标继续保留用于追溯和兼容，新增指标先作为 test/diagnostic 扩展汇总；只有在复核波形图和分层结果后，才把新增指标升为正式选模主指标。

## 任务目标

评价体系优先回答四个问题：

1. 呼吸率是否一致，且不只是一整窗主频一致。
2. 低频形态是否一致，并能区分形态错误和持续时延。
3. 相对呼吸强弱变化是否一致，优先关注幅度强弱而不是绝对上下边界形状。
4. 低质量信号下是否恢复了合理呼吸次数；逐呼吸事件先作为探索性复核。

## 判断依据与风险

### 关于时延与 zero-lag 相关

不把原时间轴同步恢复作为主要目标是合理的。BCG 和 THO 来自不同生理链路，中间可能叠加传导、姿态、个体差异、接触状态和信号处理窗口带来的持续且非线性时延。若强行把 zero-lag 相关作为主指标，模型可能被迫学习个体化时延，而不是学习稳定的呼吸形态和相对变化。

因此 `best_lag_corr_4s` 比 `band_limited_corr` 更贴近当前任务目标。它允许把“形态是否恢复”与“是否刚好同步”拆开看。`band_limited_corr` 仍有价值，但主要用于诊断：如果它也高，说明模型在原时间轴上同步性好；如果它低而 `best_lag_corr_4s` 高，说明模型恢复了低频形态但存在时延。

这个判断也有边界。呼吸是周期信号，lag 搜索范围扩大到 `+-4s` 后，规整周期波可能更容易找到高相关位置，尤其在较快呼吸或形态单调的窗口中。因此 `best_lag_corr_4s` 不能单独作为通过标准，必须同时看 `best_lag_sec_4s`、局部 RR、相对包络和坏例图。

### 关于 RMS relative envelope

RMS envelope 不是只看上包络。它对信号平方后做局部均方根，正负振幅都会贡献，因此更接近局部能量或幅度强度包络。对于“相对呼吸强弱变化”这个目标，RMS relative envelope 是合理指标，因为它天然弱化了绝对幅值标定和符号方向问题。

它的缺点不是忽略下包络，而是丢失上沿、下沿和中线的符号化形态。如果任务未来关注吸呼不对称、基线漂移或胸带波形上下边界形态，才需要进一步拆 `upper`、`lower`、`midline`。在当前阶段，一次性引入整套双边包络指标容易增加噪声和解释成本，且可能和 RMS relative envelope 高度冗余。

因此第一轮只探索 `envelope_band_*` 作为幅度带宽候选，用它和现有 RMS relative envelope 做互补性检查。若人工坏例和模型排序都没有显示额外收益，就不把它升入主表。

### 关于呼吸次数与逐呼吸事件

`breath_count_zero_cross_abs_error` 仍值得保留。它粗糙，但能回答一个很实际的问题：低质量信号下模型是否恢复出合理数量的呼吸周期。这个指标不依赖整窗主频，也比逐呼吸事件匹配更稳，适合作为轻量护栏。

逐呼吸事件指标值得尝试，但不适合第一轮直接进入主排序。它能识别漏呼吸、多呼吸、局部 RR 漂移和事件级时间偏差，是对 dominant rhythm 指标的重要补充；但它高度依赖 target 局部质量、峰/谷定义、容忍窗口、双峰呼吸和带通参数。若没有人工复核，很容易把检测规则误差当成模型误差。

因此第一轮更稳的路径是先加 `local_rr_mae` / `local_rr_corr`，作为整窗 RR 和逐呼吸事件之间的中间层；同时优化 zero-cross 计数输出。逐呼吸事件指标先在小样本坏例复核中验证稳定性，再决定是否扩大使用。

## 指标分层

### 核心呼吸率护栏

保留 `rr_peak_band_robust_abs_error` 作为整窗全局 RR 护栏。它抗伪峰能力强，适合防止模型偏离基本呼吸节律，但不能单独代表细节恢复。

保留 `rr_peak_band_abs_error` 用于历史口径对接。它不应长期作为唯一主指标，因为普通峰检测更容易受伪峰影响，而 robust 版本又会主动平滑局部细节。

`rr_spec_abs_error` 降级为频域安全检查。它可以确认频域主呼吸率没有明显恶化，但功率谱最大峰会偏向最强或最常见呼吸节律，不适合作为核心排序指标。

### 低频形态与时延

保留 `band_limited_corr`，作为 zero-lag 低频形态一致性指标，但不把它解释为主要任务目标。BCG 和 THO 之间可能存在由多种耦合因素导致的持续、个体化且非线性的时延；让模型学习这个时延本身意义有限。因此 zero-lag 指标只用于判断“原时间轴上是否恰好同步”，不能过度惩罚可解释的生理/耦合延迟。

新增或显式输出 `best_lag_corr_4s` 和 `best_lag_sec_4s`，搜索范围为 `+-4s`。`best_lag_corr_4s` 是核心辅助指标，回答“允许合理时延后形态是否恢复”；`best_lag_sec_4s` 用于解释延迟方向和大小，不单独作为越大越好的指标。

zero-lag 和 lag-aligned 指标必须同时保留，但权重不同。`best_lag_corr_4s` 更贴近“低频形态是否恢复”，应作为主辅助指标；`band_limited_corr` 和 `best_lag_sec_4s` 主要解释时延。`band_limited_corr` 低但 `best_lag_corr_4s` 高，表示形态可恢复但存在时延；二者都低才支持“形态没有恢复”的判断。

### 相对变化与包络

保留现有 `relative_envelope_mae` 和 `relative_envelope_corr`。当前 RMS envelope 是局部能量/幅度强度包络，正负振幅都会进入平方后求均方根，因此不是单纯只看上包络；它的局限在于不保留上沿、下沿和中线的符号化形态差异。

包络扩展先保持克制。优先探索一个幅度带宽候选：

- `envelope_band_corr` / `envelope_band_mae`，其中 band 表示 `upper - lower`

`envelope_band_*` 最贴近相对呼吸强弱变化，可与现有 RMS relative envelope 对照。如果它和人工复核、模型排序没有提供额外信息，就不进入主表。

`upper_envelope_*`、`lower_envelope_*` 和 `envelope_midline_*` 暂不进入第一轮实现，只在发现 RMS/band 包络无法解释上下不对称或基线漂移坏例时再补。

形态和包络指标优先提供 lag-aligned 口径。zero-lag 结果可保留为诊断；lag-aligned 结果使用 `best_lag_sec_4s` 对齐后计算，并加 `_lag4s` 后缀，用于判断形态本身与时间延迟的贡献。

### 呼吸次数与逐呼吸事件

保留 `breath_count_zero_cross_abs_error` 作为轻量护栏。它能体现低质量信号下是否恢复出合理周期数量，但不评价每次呼吸是否对齐。

优化 zero-cross 计数输出，除现有 cycle count 外，增加：

- `pred_breath_count_zero_cross_up`
- `target_breath_count_zero_cross_up`
- `pred_breath_count_zero_cross_down`
- `target_breath_count_zero_cross_down`

逐呼吸事件指标先作为探索性候选，不直接进入主排序：

- `breath_event_precision`
- `breath_event_recall`
- `breath_event_f1`
- `breath_timing_mae_sec`
- `cycle_rr_mae`
- `cycle_rr_corr`

事件匹配在带通信号上进行，默认以 target 事件为参照，使用与目标周期相关的容忍窗口。该组指标用于补上“整窗 RR 正确但局部呼吸变化没跟上”的盲点，但它对 target 局部质量、峰/谷选择、容忍窗口和信号双峰非常敏感。第一轮更适合先实现 `local_rr_mae` / `local_rr_corr` 和优化后的 zero-cross 计数；逐呼吸事件指标在小样本人工复核后再决定是否扩大使用。

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
- `relative_envelope_mae_lag4s_mean`
- `relative_envelope_corr_lag4s_mean`
- `band_limited_corr_mean`
- `best_lag_corr_4s_mean`
- `best_lag_sec_4s_median`
- `local_rr_mae_mean`
- `local_rr_corr_mean`

第一轮候选表可额外包含 `envelope_band_mae_lag4s_mean`、`envelope_band_corr_lag4s_mean`、`breath_event_f1_mean`、`breath_timing_mae_sec_median` 和 `cycle_rr_mae_mean`，但这些列需通过人工复核后再决定是否进入主表。

旧指标进入 diagnostic 表，避免主结论被 dominant rhythm 或历史兼容指标冲淡。

## 验证策略

实现新增指标后，先对已有 `runs/test_eval_g_series_20260705_robust_v1` 生成旁路扩展表，不覆盖原 CSV。需要确认：

1. 新增指标不会改变旧指标数值。
2. `best_lag_corr_4s` 与 `best_lag_sec_4s` 能把形态差和时延差分开。
3. RMS relative envelope 与 `envelope_band_*` 是否提供互补信息；若二者高度冗余，优先保留更稳定、更容易解释的一项。
4. 逐呼吸事件指标能否稳定识别漏呼吸、多呼吸和局部 RR 漂移；若对峰检测参数过敏，则仅保留为诊断脚本输出。
5. 新指标分层结论不被少数坏 target 或低质量窗口主导；必要时按 `rr_peak_valid_ratio`、质量标签和 baseline hard/easy 分层报告。
