# RTM-v1 validation summary 协议

日期：2026-08-21

状态：**15/15 formal receipt已完成只读完整性验收；本附件只冻结validation汇总实现，正式summary尚未运行。Research-test继续关闭。**

## 1. 证据与访问边界

本summary只能形成`validation-development evidence; research-history-informed; not untouched held-out or unbiased generalization evidence`。输入固定为双GPU v2矩阵的15份`formal_receipt.json / lifecycle.json / artifact_manifest.json / validation_metrics.csv / validation_metrics_summary.csv`，以及已冻结GPU engineering v2的`candidate_engineering.csv / access_receipt.json / artifact_manifest.json`。

允许读取既有validation逐sample metrics和summary；禁止模型推理、训练、GPU、dataset/index/signal/target array、checkpoint内容、research-test或其他旧run。Checkpoint只通过formal manifest中的既有记录参与完整性链，不打开checkpoint文件。不得根据summary结果增加指标、改变tolerance、删除seed、重选epoch、重评checkpoint或重训。

Formal完整性固定为五项candidate×三个seed、15个receipt/lifecycle、每项80 epochs/6400 updates、2675 validation rows、同一执行commit `24c54a88ea8b17ad183b48976bd1f690e3867706`、`gpu_0/0` 8项与`gpu_1/1` 7项。逐项receipt SHA-256由冻结summary config逐一列出；任一identity/hash/count/access/finite/artifact/row identity不闭合即fail closed。

实现前为确认CSV schema，只读显示过一份T0 stored summary行及TCN一个seed的前两条sample rows；未计算candidate或family比较。下述metric、tolerance、聚合、Pareto和停止规则全部直接来自既有RTM-v1第9节，不根据所见数值调整。

## 2. Seed级重算与候选聚合

每份`validation_metrics.csv`必须满足：

- 恰好2675行，`evaluation_split=validation`、`split=val`、`method=<candidate_id>`；
- `dataset_row_id`有序SHA-256固定为`b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a`；
- 15份文件的row identity、target modulation/stratum及全部target-eligibility字段逐行相同；
- 五项primary每行finite；冻结eligibility允许secondary存在null，但不得存在Inf；
- 使用`resp_train.metrics.task.summarize_task_metrics`从逐sample metrics重算，并以`rtol=atol=1e-12`复现stored summary。

五项primary及方向为：

1. `whole_rr_abs_error_bpm_mean`，minimize；
2. `local_rr_mae_bpm_mean`，minimize；
3. `envelope_trajectory_mae_mean`，minimize；
4. `global_envelope_modulation_error_mean`，minimize；
5. `lag_aware_signed_pcc_mean`，maximize。

先保存15行seed summary。再按candidate与固定seed `20260811/12/13`计算arithmetic mean与sample SD（`ddof=1`）；不pool逐sample、不做subject balancing、不计算确认性p-value或置信区间。所有可聚合seed级描述量均保存mean/SD，但只有五项primary进入质量Pareto。

## 3. Paired-seed方向

对五项candidate的10个无序pair、五项primary分别生成paired-seed方向，共50行。每行保存三个同seed差值、left/right raw-better count、within-tolerance count及left/right materially-better count。方向只作描述，不进行显著性检验，也不把paired计数追加为Pareto gate。

## 4. Tolerance-aware Pareto

候选级mean上固定：

- error primary相对tolerance=`0.5%`；
- signed PCC绝对tolerance=`0.002`；
- standardized throughput与peak allocated相对tolerance=`5%`；
- trainable parameters无tolerance，按精确整数比较。

对于minimize维度，A相对B“不差”定义为`A <= B*(1+tolerance)`，“实质更优”定义为`A < B*(1-tolerance)`；对于signed PCC，分别为`A >= B-0.002`与`A > B+0.002`；对于maximize资源维度使用对应相对反向不等式。A只有在全部维度不差且至少一项实质更优时才dominate B。

必须分别计算：

1. **quality Pareto**：只用五项primary；
2. **quality-efficiency Pareto**：五项primary加`trainable_parameters`（minimize）、`update_throughput_windows_per_second`（maximize）、`update_peak_allocated_mib`（minimize）。

资源来自冻结GPU engineering v2统一128×1 update benchmark。`batch1_inference_p50/p90_ms`、`update_peak_reserved_mib`、IBI/coverage、Low/Medium/High envelope Spearman及degeneracy只进入supplement，不改变quality Pareto或科学解释。两个Pareto必须独立输出；不加权、不总分排序、不根据secondary强制唯一赢家。

T0保持trunk-attribution control。对四个trunk candidate分别记录相对T0是否至少一项primary materially improved；若四项全部为false，停止线为`all_trunks_no_material_primary_improvement_stop`，不得追加容量搜索。否则为`at_least_one_trunk_material_primary_improvement`。无论结果如何，当前15-run协议在汇总后关闭，不追加宽度、LR、epoch、dilation、hidden、depth或多尺度分支。

## 5. 输出、receipt与执行

固定输出目录：

```text
runs/resp_temporal_v1/formal_validation_summary_v1/
```

目录必须预先不存在，禁止覆盖。至少原子发布：

- `resolved_config.json`
- `seed_summary.csv`
- `candidate_summary.csv`
- `paired_seed_directions.csv`
- `dominance_audit.csv`
- `quality_pareto.csv`
- `quality_efficiency_pareto.csv`
- `supplementary_metrics.csv`
- `decision.json`
- `artifact_manifest.json`
- `access_receipt.json`
- `access_receipt.sha256`

Receipt必须记录15项input hashes、2675×15 metrics rows、finite/null计数、formal/GPU provenance、validation-only access flags及全部输出artifact hash。Summary输出不得包含prediction waveform或target waveform。

唯一命令由用户手动执行：

```bash
./.venv/bin/python scripts/summarize_resp_temporal_v1.py --config configs/resp_temporal_v1/validation_summary_v1.yaml
```

Codex只实现与运行CPU定向测试，不执行正式summary。用户返回summary receipt后再解释quality Pareto、quality-efficiency Pareto、T0 attribution与停止线；research-test不自动开放。
