# 脚本索引

本目录暂时保持脚本平铺，不移动文件。当前数量还少，平铺能减少命令路径变化；等腹带、消融、整夜推理和更多诊断脚本加入后，再考虑按子目录分类。

当前脚本以 THO 小规模实验为主，默认配置来自 `configs/tho_small.yaml`：

- 默认输入集：`mixed_zscore`
- 默认训练抽样：`data.train_sample_strategy=stratified_random`
- 默认验证抽样：`data.val_sample_strategy=stratified_random`
- 默认固定验证 seed：`data.val_sample_seed=20260602`
- `head` 抽样只建议临时 debug，不作为正式实验口径

所有支持 `--set` 的脚本都使用 OmegaConf dotlist 覆盖配置，例如：

```bash
--set data.train_sample_seed=1 --set loss.smooth_weight=0.001
```

训练入口会检查必要依赖；如需新增第三方库，先确认再安装。

## 推荐工作流

1. 先审计本次配置对应的数据子集，确认抽样数量和质量分层。
2. 跑平凡基线，给模型指标一个参照。
3. 跑训练脚本，保存完整 run 目录。
4. 必要时从 checkpoint 重新评价，或导出不同数量的诊断预测。
5. 绘制预测图，检查主频、峰谷、包络同步和失败样本。
6. 汇总多个 run，比较 loss、seed 和采样策略变化。

## 数据检查

### `audit_tho_dataset.py`

生成胸带小规模训练数据审计表，检查 split、input_set、残差质量分层和可用窗口数量。脚本复用训练数据工厂的过滤和抽样口径，避免审计结果与训练入口不一致。

```bash
./.venv/bin/python scripts/audit_tho_dataset.py \
  --config configs/tho_small.yaml \
  --output /tmp/tho_audit.csv
```

常用：固定验证 seed、改变训练 seed 时，审计命令也应传入同样覆盖项，确保审计和训练口径一致。

## Split 独立性审计

在解释 L1 或模型实验前，先检查 train/val 是否共享 `samp_id` 或 `(samp_id, segment_id)`：

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config configs/tho_research_v2.yaml \
  --output-dir runs/audits/split_independence_research_v2
```

脚本会输出 `summary.csv`、重叠明细和分布对照表。若 `summary.csv` 中 `overlap_samp_id_count` 或
`overlap_segment_count` 大于 0，当前结果只能作为 within-subject / within-dataset 开发指标，不应作为跨个体泛化结论。

## 训练与评价

### `train_tho_small.py`

训练胸带参考小规模模型，并输出 run 目录。该脚本是薄入口，负责解析命令行并委托 `ThoExperiment` 执行实验流程。默认使用 `configs/tho_small.yaml`，可用 `--set` 覆盖配置。

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.train_sample_seed=1 \
  --set data.val_sample_seed=2 \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=4
```

正式小规模对照建议固定验证 seed，只改变训练 seed 或 loss 参数：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.train_sample_seed=1 \
  --set data.val_sample_seed=20260602 \
  --set training.epochs=10
```

### 当前 N3 signed phase alignment 主线

当前阶段先不改模型结构，使用 N3 候选 loss 验证“呼吸节律 + 相对努力变化 +
state-aligned signed phase alignment”是否带来可用的呼吸任务收益。波形是重要证据，
但不是唯一选择目标。执行顺序建议固定为：

1. 先运行 split 独立性审计，确认当前结果应解释为 within-subject 开发指标，还是可以支撑跨 `samp_id` 泛化结论。
2. 使用同一个验证 seed 和全量窗口，避免验证集变化掩盖 loss 差异。
3. 每完成一次正式实验 run 后提交一次仓库状态，提交信息写明实验口径和关键权重。
4. 先按任务指标筛选：`rr_peak_abs_error` 和 `rr_spec_abs_error` 不应明显恶化；再看 `relative_envelope_mae`、`relative_envelope_corr` 和 `spectrum_similarity`。
5. `band_limited_corr`、`best_lag_corr`、`best_lag_sec` 用于解释低频波形形态和小范围时移，不能单独作为通过标准。

效率口径建议：正式对照实验仍应显式记录 `training.batch_size`。当前
`patch_mixer1d + WeakSyncLoss` 在 RTX 4070 Ti SUPER 上建议先用
`training.batch_size=128`；它比历史 `batch_size=8` 明显减少小 step 调度开销，
同时比 `256` 保留更多每 epoch 更新步。`training.use_amp=true` 已支持 18000 点
非 2 次幂窗口，但当前模型在 `batch_size=128` 下 AMP 吞吐收益不明显，可作为显存
余量不足或更大 batch 的备用选项。

注意：2026-06-17 早期 L0/L1/L2/L3 loss 分支已经完成历史评估，不再作为当前
训练入口。当前主线使用 `bcg_rawish_wideband_state_aligned`，默认 loss 收窄为
`envelope + spectrum + smooth + high_freq + relative_envelope + signed phase_alignment`。
其中 `phase_alignment` 是 state-aligned 前提下的低频 signed correlation 约束，
带小范围 lag 容忍和 lag 惩罚；它不是胸带波形复制 loss。

正式训练使用全量窗口和 batch128 效率口径：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set loss.phase_alignment_max_sec=0.5 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phasealign_n1_n4
```

历史 L0/L1/L2/L3 与 N1-N4 结果保留在
`docs/experiments/rawish_state_aligned_l0_l1.md`。模型选择仍以
`rr_peak_abs_error` 为主护栏，`band_limited_corr`、`best_lag_corr` 和
`best_lag_sec` 只作为低频形态和残余时移诊断指标。

每个 run 输出到配置中的 `outputs.run_root/<timestamp>/`；`configs/tho_small.yaml` 默认是
`runs/tho_small`，`configs/tho_research_v2.yaml` 默认是 `runs/tho_research_v2`。常见产物包括：

- `config.yaml`：本次 resolved config 快照。
- `audit.csv`：训练数据工厂生成的数据审计摘要。
- `baseline_metrics.csv`：val 子集平凡基线指标。
- `train_history.csv`：每轮训练和验证损失。
- `metrics.csv`：best checkpoint 在 val 子集上的逐窗口指标。
- `checkpoint.pt`：验证损失最优 checkpoint。
- `train.log`：训练日志。

### `eval_tho_small.py`

从 `checkpoint.pt` 重新生成指标。脚本委托 checkpoint 评价函数执行加载、配置一致性校验和指标计算。默认读取 checkpoint 同目录的 `config.yaml`。

```bash
./.venv/bin/python scripts/eval_tho_small.py \
  --checkpoint runs/tho_small/<timestamp>/checkpoint.pt \
  --metrics-output /tmp/tho_metrics.csv
```

注意：显式传入 `--config` 或 `--set` 时，会校验模型结构、验证集定义、窗口参数和评价频带等关键字段，避免用不一致配置误评 checkpoint。

## 诊断分析

### `baseline_tho_hilbert.py`

在 val 子集运行平凡基线，输出逐窗口 RR/主频、峰谷、包络相关和频谱相似度指标。脚本复用训练数据工厂口径，保证 baseline 与训练验证子集一致。

```bash
./.venv/bin/python scripts/baseline_tho_hilbert.py \
  --config configs/tho_small.yaml \
  --output /tmp/baseline_metrics.csv
```

BCG 频谱主峰法和峰谷检测法只是诊断参照，不是训练目标。质量好的片段可能能靠规则法读出合理结果，质量差的片段规则法失败也不等同于模型路线失败。

### `plot_tho_predictions.py`

读取一个 run 的 `metrics.csv`，按指标选出待查看窗口，再用 `checkpoint.pt` 重新推理这些窗口并生成预测/参考波形诊断图。默认输出到 `<run-dir>/plots/`。

```bash
./.venv/bin/python scripts/plot_tho_predictions.py \
  --run-dir runs/tho_small/<timestamp> \
  --max-plots 8
```

默认按 `rr_peak_abs_error` 从大到小优先绘制，便于先看峰谷形态最差的窗口。也可以改为：

```bash
./.venv/bin/python scripts/plot_tho_predictions.py \
  --run-dir runs/tho_small/<timestamp> \
  --sort-by spectrum_similarity \
  --max-plots 8
```

### `summarize_tho_runs.py`

汇总 `runs/tho_small/*` 下各 run 的训练损失、模型指标、平凡基线指标和审计数量。
输出中的 `selection_task_*` 列用于模型选择，`selection_waveform_*` 列用于波形诊断；`best_val_loss` 只作为训练代理指标，不作为最终选择依据。

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_small \
  --output /tmp/tho_runs_summary.csv
```

汇总表适合先筛查趋势，但不能替代诊断图。尤其是当前数据量小、窗口间相关性强，单个 run 的数值差异需要配合固定验证集和多训练 seed 才能形成更稳的判断。

## 分类建议

当脚本数量继续增加时，推荐迁移到以下结构：

```text
scripts/
  data/
    audit_tho_dataset.py
  train/
    train_tho_small.py
    eval_tho_small.py
  diagnostics/
    baseline_tho_hilbert.py
    plot_tho_predictions.py
    summarize_tho_runs.py
```

迁移前需要同步更新文档、测试和常用命令。
