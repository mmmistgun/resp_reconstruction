# Rawish State-Aligned L0/L1 对齐实验计划

> **面向 AI 代理的工作者：** 使用 `executing-plans` 或 `subagent-driven-development` 执行。GPU 命令需要在沙盒外运行。实验产物留在本地 `runs/`，不进入 Git。

## 目标

在 `bcg_rawish_wideband_state_aligned -> tho_waveform_ref` 口径下，重新生成
batch128 效率口径的 fresh L0/L1，对齐评估带限波形损失是否改善当前呼吸任务指标。

波形恢复是证据链的一部分，但不是唯一选择目标。最终判断优先使用 RR、相对包络和频谱一致性；`band_limited_corr`、`best_lag_corr` 和 `best_lag_sec` 只作为波形诊断指标。

## 固定口径

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 数据窗口：全量（`data.max_train_windows=null`，`data.max_val_windows=null`）
- 模型：`patch_mixer1d`
- Patch 参数：`patch_len=256`，`patch_stride=128`，`mixer_layers=2`
- 损失固定项：`high_freq_weight=0.2`，`relative_envelope_weight=0.01`，`phase_lag_weight=0.0`
- 训练：`epochs=50`，`batch_size=128`，`learning_rate=0.001`，`use_amp=false`
- 早停：`patience=8`，`min_delta=0.001`
- seed：`training.seed=20260610`，`data.train_sample_seed=20260610`，`data.val_sample_seed=20260611`
- 输出：`runs/tho_research_v2_patch_mixer_rawish_eff_l0_l1`

历史 `runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320`
只作为背景参考。由于本轮使用 batch128 效率口径，必须 fresh 跑 L0，不能把历史
batch8 run 当作唯一变量基线。

## 模型选择口径

1. 主护栏：`rr_peak_abs_error`，均值和中位数不应明显恶化。
2. 次护栏：`rr_spec_abs_error`，用于确认频域呼吸率没有被牺牲。
3. 任务辅助：`relative_envelope_mae` / `relative_envelope_corr`，用于判断相对呼吸强弱变化。
4. 频谱辅助：`spectrum_similarity`，用于判断频谱一致性。
5. 波形诊断：`band_limited_corr`、`best_lag_corr`、`best_lag_sec`，用于解释低频形态和时移，但不能单独作为通过标准。
6. `best_val_loss` 只作为训练代理指标，不作为最终模型选择依据。

## 执行步骤

### 1. 清理旧 L1 产物

删除旧的不可比 L1 目录和 summary，只保留历史背景 run：

```bash
rm -rf runs/tho_research_v2_patch_mixer_rawish_bandwave_l1
rm -f runs/tho_research_v2_patch_mixer_rawish_bandwave_l1_summary.csv
```

确认 `runs/` 已被 `.gitignore` 忽略，实验产物不进入仓库。

### 2. 运行 fresh L0

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
  --set loss.relative_envelope_weight=0.01 \
  --set loss.band_waveform_weight=0.0 \
  --set loss.phase_lag_weight=0.0 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_eff_l0_l1
```

### 3. 运行 L1 网格

只改变 `loss.band_waveform_weight`：

```bash
for w in 0.05 0.10; do
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
    --set loss.relative_envelope_weight=0.01 \
    --set loss.band_waveform_weight="$w" \
    --set loss.phase_lag_weight=0.0 \
    --set training.epochs=50 \
    --set training.batch_size=128 \
    --set training.patience=8 \
    --set training.min_delta=0.001 \
    --set training.use_amp=false \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_eff_l0_l1
done
```

### 4. 汇总结果

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_patch_mixer_rawish_eff_l0_l1 \
  --output runs/tho_research_v2_patch_mixer_rawish_eff_l0_l1_summary.csv
```

把 fresh L0、L1a、L1b 写入：

```text
docs/experiments/rawish_state_aligned_l0_l1.md
```

表格至少包含：

- run id
- `band_waveform_weight`
- `best_val_loss`
- `rr_peak_abs_error` mean / median
- `rr_spec_abs_error` mean / median
- `relative_envelope_mae` mean / median
- `relative_envelope_corr` mean
- `spectrum_similarity` mean
- `band_limited_corr` mean
- `best_lag_corr` mean
- `abs(best_lag_sec)` mean

## 当前执行记录

- L0：`20260617_173758_049730`，`band_waveform_weight=0.0`
- L1a：`20260617_174421_369708`，`band_waveform_weight=0.05`
- L1b：`20260617_174842_762073`，`band_waveform_weight=0.10`

当前阶段判断写入 `docs/experiments/rawish_state_aligned_l0_l1.md`。本批结果显示
L1 显著改善低频波形诊断，但明显破坏 `rr_peak_abs_error` 主护栏，因此 L1a/L1b
都不作为通过实验进入模型结构结论。

## 验收清单

- [ ] L0/L1 都使用 `bcg_rawish_wideband_state_aligned`。
- [ ] L0/L1 都使用全量数据、batch128、同一 seed、同一模型参数。
- [ ] L1 只相对 L0 改变 `band_waveform_weight`。
- [ ] 三个 run 都生成 `config.yaml`、`train_history.csv`、`metrics.csv`、`checkpoint.pt`。
- [ ] summary 文件生成在 `runs/tho_research_v2_patch_mixer_rawish_eff_l0_l1_summary.csv`。
- [ ] 阶段判断优先检查 `rr_peak_abs_error`、`rr_spec_abs_error`、相对包络和频谱一致性。
- [ ] `runs/` 产物不进入 Git。
- [ ] 完成后运行相关测试，并提交代码和文档。
