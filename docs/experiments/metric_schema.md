# THO research v2 指标口径

首次冻结：2026-07-08

代码对齐：2026-07-17

## 定位

本文冻结当前 THO research v2 实验整理使用的评价口径，并定义新实验的自动 checkpoint
选择口径。2026-07-17 已对齐训练期选择、top-k 复评和通用汇总脚本；历史 run 产物不改写。

本仓库是科研实验仓库，当前口径优先，不为旧结果维持兼容成本。整理实验记录的目标是
服务下一步可信判断，不是维护完整历史排名。旧结果默认不进入当前横向排序；只有当它能直接支撑
当前假设、模型选择或停止判断时，才进入当前证据账本。

后续整理实验记录时，优先用本文判断一个结果属于：

- `active_current`：当前问题仍需要它，且证据已满足当前口径。
- `needs_reeval_for_active`：当前问题仍需要它，但 checkpoint 或 top-k 需要按当前指标重评。
- `needs_retrain_for_active`：当前问题仍需要它，且旧 checkpoint 不足以回答当前问题。
- `reference_only`：只作为结构想法、失败模式或旧结论解释，不进入当前排序。
- `discard_for_current_question`：对当前问题没有继续价值，不再投入整理、重评或重训成本。

## 当前适用范围

默认适用于 `configs/tho_research_v2.yaml` 当前 soft-z research v2 主线：

- 数据集：2026-06-20 soft-z research v2 口径。
- 主任务：BCG window 到 THO waveform / 呼吸状态相关评价。
- 正式比较：固定 split、验证窗口、数据 seed、训练 seed 配对关系和 checkpoint
  选择口径。
- 旧 rawish state-aligned run 默认只作为模型结构和失败模式线索；除非被明确拉回当前
  soft-z 问题，否则不重评、不重训、不参与当前横向比较。

如果后续改变 split、subject/session 隔离、标签定义、mask 逻辑、窗口构造或目标
波形定义，需要另写新指标口径或在本文增补版本号。

## 指标分组

### 主任务护栏

- `rr_peak_band_robust_abs_error`：当前最优先看的整窗带通 RR 护栏，降低峰值误拣和
  坏段影响。正式记录应报告 mean、median、p95 和长尾比例；若脚本暂未输出这些汇总，
  至少保留逐窗口列以便补算。
- `breath_count_zero_cross_abs_error`：低质量或低频形态不稳定窗口中，是否恢复合理周期
  数量的轻量护栏。
- `breath_count_zero_cross_bpm_error`：`breath_count_zero_cross_abs_error / (window_seconds / 60)`。
  当前 180 秒窗口下等于原始周期数误差除以 3，只作为汇报和分层表中的平均周期率误差展示口径；
  原始 count-error 分层仍用 `breath_count_zero_cross_abs_error` 判断。

### 辅助诊断与旧结果解释

- `rr_peak_band_abs_error`：旧主指标，保留用于解释既有 E/F/G 记录和辅助定位误拣，
  但不应单独主导当前排序。
- `rr_spec_abs_error`：频域 RR 护栏。若它改善但 `rr_peak_band_robust_abs_error`、
  `rr_peak_band_abs_error` 或 count 明显变差，应解释为频谱收益不足以通过任务护栏。
- `rr_peak_abs_error`：当前代码中是按 THO/BCG 共同好段 mask 后的 raw peak RR，不再是
  未遮罩整窗 raw peak。
- `rr_peak_unmasked_abs_error`：旧式未遮罩 raw peak 诊断，只用于尖峰、坏段和历史解释。

### 低频形态与时延诊断

- `band_limited_corr`：zero-lag 低频形态诊断。BCG 与 THO 可能存在持续时延，因此不能
  单独作为通过或失败标准。
- `best_lag_corr_4s`、`best_lag_sec_4s`：当前解释低频形态时优先看的 lag-aware 指标。
  需要同时看相关性大小和最佳 lag 是否落在可解释范围内。
- `best_lag_corr`、`best_lag_sec`：小范围 lag 诊断，保留用于和旧记录解释同一失败现象。

### 相对强弱与局部 RR

- `relative_envelope_mae_lag4s`、`relative_envelope_corr_lag4s`：允许持续时延后的相对
  呼吸强弱诊断，优先于 zero-lag 相对包络列解释波形强弱是否恢复。
- `relative_envelope_mae`、`relative_envelope_corr`：zero-lag 相对包络诊断，保留用于
  旧记录解释和不考虑持续时延的坏例定位。
- `local_rr_mae`、`local_rr_corr`、`local_rr_valid_frac`：局部 RR
  曲线指标。2026-07-09 起采用原 v2 口径：默认 40 秒窗口、10 秒步长，并使用更严格的
  spectral-guided peak 间距，降低同一呼吸周期双峰被算成两个周期的概率。它不是逐呼吸
  事件匹配指标，应结合 `valid_frac`、目标侧局部 RR 跳变和波形复核解释。
- legacy local RR：旧 `local_rr_*` 20 秒窗口、5 秒步长口径已退场；旧 CSV 中的
  `local_rr_*` 若生成于 2026-07-09 转正前，只能作为历史反例筛查列。
- local RR v3：40 秒/10 秒过零周期率探针已退场；它只保留为 2026-07-09 探索记录，
  不进入当前默认评价输出、summary 或排序链。

## 当前排序建议

当前整理记录时使用以下解释顺序。自动选择只固化前两个主护栏及旧 RR 辅助平局项；
lag-aware 形态和 local RR 仍需人工结合波形解释，不折叠成单一总分。

1. 先确认同数据口径、同 split、同验证窗口、同 seed 配对和同 checkpoint 选择口径。
2. 看 `rr_peak_band_robust_abs_error` 与 `breath_count_zero_cross_abs_error` 是否守住主护栏。
3. 用 `rr_peak_band_abs_error`、`rr_spec_abs_error` 和长尾比例辅助解释误拣来源。
4. 用 `best_lag_corr_4s`、`best_lag_sec_4s`、`relative_envelope_*_lag4s` 和当前
   `local_rr_*` 解释低频形态、强弱和局部节律；局部 RR 仍需结合目标侧跳变和波形复核，
   不用 legacy/v3 探针参与当前排序。
5. `val_loss`、`band_limited_corr`、`spectrum_similarity` 或单个频谱收益不能单独决定模型
   通过。

## 2026-07-09 局部 RR 转正记录

本次只对 `g3_c_wide_8p0_20260837` 的 held-out test checkpoint 做一次探索性复评，
输出保存在 `runs/local_rr_v2_v3_probe_20260709/`：

- `g3_c_wide_8p0_20260837_v2_v3_test_metrics.csv`
- `g3_c_wide_8p0_20260837_v2_v3_test_summary.csv`
- `g3_c_wide_8p0_20260837_v2_v3_test_manifest.csv`
- `g3_c_wide_8p0_20260837_v2_v3_case_eval.csv`

结果显示：当时的 current/legacy `local_rr_mae` mean/median 为 `1.029/0.470`；v2 降到
`0.790/0.297`；v3 为 `0.864/0.400`。在 `rr_peak_band_robust_abs_error < 0.5`
但局部 RR 误差大于 `2 bpm` 的窗口数上，当时的 current/legacy 为 `120`，v2 降到 `49`，
v3 为 `101`。关键样本 row 3718 从当时的 current/legacy `5.413` 降到 v2 `0.563`，支持“20 秒
寻峰把标签双峰化”的判断；v3 对该样本仍为 `6.0`，说明过零对局部振荡形态也敏感。

稳健性补查：v2 的中位数、90 分位和 95 分位优于当时的 current/legacy，但极端尾部仍未消失；
`local_rr_v2_mae > 5 bpm` 的窗口数为 `67/2310`，高于当时 current/legacy 的 `48/2310`。
target 侧 v2 局部曲线自身有 `80/2310` 个 180 秒窗口出现超过 `15 bpm` 的窗内
范围跳变，且 v2 高误差窗口中 `48/67` 个伴随这类 target 侧局部 RR 跳变。说明
40 秒局部频谱周期约束能修复一部分同周期双峰，但仍会受谱峰硬切换、谐波抢峰和
低质量局部窗影响。

2026-07-09 人工坏例复核后接受原 v2 作为 `local_rr_*` 口径。legacy 20s/5s
寻峰口径和 v3 过零口径同时退入历史，不再作为默认评价列输出。旧结果若要进入当前
local RR 讨论，必须按转正后的 `local_rr_*` 口径重评 checkpoint；只重算旧 summary 不足以
改变该指标语义。

## 重算、重评与重训边界

### 只需重算 summary

仅当结果已属于 `active_current` 或 `needs_reeval_for_active` 候选，且满足全部条件时，
才重算 summary：

- 已有 `metrics.csv` 或 `metrics_top*.csv` 包含本文要求的逐窗口指标列。
- 指标公式、mask 逻辑和评价配置未发生变化。
- 只需要新增聚合列、改排序表、改文字结论或补分层统计。

### 需要重评 checkpoint

当前问题仍需要该模型或候选，且满足任一条件时，归为 `needs_reeval_for_active`：

- 逐窗口指标缺少 `rr_peak_band_robust_abs_error`、`best_lag_corr_4s`、
  `relative_envelope_*_lag4s` 或 2026-07-09 后的 `local_rr_*` 等当前列。
- `rr_peak_abs_error` 生成于旧口径，未区分 masked 与 unmasked raw peak。
- target-side cache、评价 mask、lag 搜索、local RR 或 robust peak 估计逻辑变化后，
  旧 CSV 无法证明满足当前口径。
- 需要比较 `checkpoint_top1/2/3.pt`，但当前只有旧 `metrics_top*.csv` 或没有 top-k
  指标。

重评应优先写旁路文件，例如 `metrics_v20260708.csv`、
`metrics_top1_v20260708.csv`、`summary_v20260708.csv`，避免覆盖历史 `metrics.csv`。
每次重评必须保存 manifest，记录 checkpoint、config、split、sample seed、评价参数、
输出路径和代码版本。

不属于当前问题的旧 run，即使缺少新指标，也不自动重评。

### 2026-07-08 已完成的当前口径复评

已按当前口径复评 G 系列当前 active 代表 checkpoint，输出在
`runs/test_eval_g_series_20260708_current_metrics/`。范围来自
`configs/eval_specs/g_series_test_eval_20260705.csv`，包括 `g0_time_only`、
`g0_f0_native_stft_pre_mixer`、`g3_c_wide_8p0`、`g3_c_bandenergy` 各 3 个 seed。

本次复评确认逐窗口 metrics 已包含 `rr_peak_band_robust_abs_error`、
`best_lag_corr_4s`、`relative_envelope_*_lag4s` 和 legacy `local_rr_*` 旧列；
2026-07-09 该局部 RR 旧列已退场，因此这批旧 local RR 数值只作为历史参考；
local RR 讨论使用下一节的复评结果。派生汇总见：

- `runs/test_eval_g_series_20260708_current_metrics/g_series_current_metrics_seed_summary.csv`
- `runs/test_eval_g_series_20260708_current_metrics/g_series_current_metrics_label_summary.csv`
- `runs/test_eval_g_series_20260708_current_metrics/g_series_current_metrics_delta_vs_time_label.csv`

### 2026-07-09 local RR 口径已完成复评

已按转正后的 `local_rr_*` 口径重评同一批 G 系列 active checkpoint，输出在
`runs/test_eval_g_series_20260709_local_rr_canonical/`：

- `g_series_local_rr_canonical_seed_summary.csv`
- `g_series_local_rr_canonical_label_summary.csv`
- `g_series_local_rr_canonical_delta_summary.csv`
- `g_series_test_eval_manifest.csv`

当前解释：`G3_C_wide_8p0` 仍是实验比较 anchor；`G3_C_bandenergy` 在 breath count 和
local RR 上略优，但 robust RR 与 lag-aware 形态不如 wide anchor，只作为
选择性/条件注入线索保留。本次结果不触发全历史重评，也不触发重训。

### 需要重评 top-k，而非直接重训

如果新口径改变了“哪个 checkpoint 最好”的解释，但 run 目录保留了
`checkpoint_top1/2/3.pt` 或其他候选 checkpoint，应先按当前指标重评这些 checkpoint，
再用当前任务排序链选择代表 checkpoint。

这种情况不等价于训练失效；它只说明旧 `checkpoint.pt` 或 `checkpoint_best_task.pt`
可能不是当前口径下最合适的代表。

### 需要重训

当前问题仍需要该模型或候选，且满足任一条件时，才归为 `needs_retrain_for_active`：

- 训练期 checkpoint gate、epoch metrics、early stopping 或 `final_checkpoint=best_task`
  的选择逻辑已经改变，而旧 run 没有足够 checkpoint 可以重排。
- 训练 loss、数据构造、split、标签、mask 或目标定义改变。
- 旧 run 缺少可复评 checkpoint，且该模型仍要进入当前正式对照。
- 需要验证新训练期指标是否影响优化轨迹，而不仅是重新解释同一个 checkpoint。

## 2026-07-17 自动选择对齐记录

新代码统一使用以下 lexicographic 排序链：

1. `rr_peak_band_robust_abs_error_mean`；
2. `breath_count_zero_cross_abs_error_mean`；
3. `rr_peak_band_abs_error_mean`；
4. `rr_spec_abs_error_mean`。

前两项是当前主护栏；后两项仅在主护栏完全持平时辅助打破平局。训练期
`checkpoint_best_rr.pt` 从本次变更起按 robust RR 保存，`checkpoint_best_task.pt`、
`eval_topk_checkpoints.py` 和 `summarize_tho_runs.py` 使用同一主护栏定义。top-k 的
`--select-only` 若发现旧 `metrics_topN.csv` 缺少 robust RR 或 breath count，会明确失败，
不静默退回旧排序。

该变化不改写历史 checkpoint、CSV 或既有 canonical test 结论：

- 旧 `checkpoint_best_rr.pt` / `checkpoint_best_task.pt` 的语义以生成时代码和 run 配置为准；
- 当前 active G 系列已有现行指标的 canonical test 证据，不因此批量重训或覆盖历史结果；
- 旧 run 若重新进入当前比较，优先旁路重评保留的 top-k；没有可复评 checkpoint 时再判断是否重训。

## 后续整理动作

1. `docs/experiments/current_evidence_ledger.md` 只记录当前 active shortlist、必要证据
   路径、当前状态和下一步动作。
2. 旧 rawish、早期失败分支和临时 probe 默认不进入证据账本；仅在它们直接影响当前
   模型选择或停止判断时，作为 `reference_only` 引用。
3. 对后续新进入当前比较的 run，若 checkpoint 选择口径不清，先做旁路 top-k 复评，
   再决定是否需要重训；不要默认从历史 `metrics_top*.csv` 推断当前最终排序。
4. 把可复用稳定结论和停止判断写入 `findings.md`；专题细节继续留在
   `docs/experiments/*.md`。
