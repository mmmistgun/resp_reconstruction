# M2 模型候选实验计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `rr_peak_band_abs_error` 正式接入后，复核并小规模扩展当前最有希望的 THO 模型候选。

**架构：** 先重评已有 `patch_hann`、`periodic_smooth21` 和 `patch_uniform` checkpoint，补齐正式带通峰值 RR 指标；再只围绕 `patch_mixer1d + hann` 与 `periodic_unet1d_tiny + output_smoothing_kernel=21` 做少量 seed 复核。DLinear、过强平滑和点对点低频波形 loss 不作为本轮主线。

**技术栈：** Python、PyTorch、OmegaConf、pandas、pytest、现有 `resp_train` 训练与评价脚本。

---

## 背景与固定口径

本轮沿用 rawish state-aligned 主线：

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 数据窗口：全量，`data.max_train_windows=null`，`data.max_val_windows=null`
- split：当前全量审计为 `train/val samp_id` 和 `segment` 无重叠，可作为 leave-samp_id-out 开发指标
- 训练：`epochs=50`，`batch_size=128`，`patience=8`，`min_delta=0.001`，`use_amp=false`
- 基础 loss：`high_freq_weight=0.2`，`relative_envelope_weight=0.03`，`phase_alignment_weight=0.005`
- 主选模：`rr_peak_band_abs_error_mean`
- 次护栏：`rr_spec_abs_error_mean`
- 任务辅助：`relative_envelope_mae_mean` / `relative_envelope_corr_mean`
- 诊断：`rr_peak_abs_error_mean`、`band_limited_corr_mean`、`best_lag_corr_mean`

已有一次性复核表：

```text
/tmp/tho_filtered_rr_summary.csv
```

阶段事实：

- `patch_uniform`：带通 RR 最好，但低频相关为负，作为 RR 上限和反例锚点。
- `patch_hann`：带通 RR 接近最好，低频方向转正，patch boundary 伪影消失。
- `periodic_smooth21`：带通 RR 稳定，低频方向正确，相对包络更好。
- `periodic_smooth51`：过度平滑，频谱 RR 和相对包络受损。
- `dlinear_waveform`：高频毛刺重，低频仍反相，不进入本轮主线。

## 文件结构

- 读取：`runs/*/<timestamp>/config.yaml`
  - 职责：复用每个 run 的训练配置快照，避免误评 checkpoint。
- 本地覆盖：`runs/*/<timestamp>/metrics.csv`
  - 职责：用当前评价代码重算指标，补齐 `rr_peak_band_abs_error` 等列；`runs/` 已忽略，不进入 Git。
- 本地生成：`runs/tho_research_v2_model_m2_band_rr/`
  - 职责：保存 M2 新训练候选；目录不进入 Git。
- 本地生成：`runs/tho_research_v2_model_m2_band_rr_summary.csv`
  - 职责：保存 M2 汇总表；文件不进入 Git。
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`
  - 职责：追加 M2 执行结果、关键表格和阶段判断。

## 任务 1：重评已有模型候选

**文件：**
- 本地覆盖：`runs/tho_research_v2_patch_mixer_rawish_current_loss_n3/20260617_220245_852336/metrics.csv`
- 本地覆盖：`runs/tho_research_v2_patch_mixer_hann_rawish_current_loss_n3/20260618_004519_403378/metrics.csv`
- 本地覆盖：`runs/tho_research_v2_periodic_unet_smooth_grid_rawish_current_loss_n3/20260618_105623_624928/metrics.csv`

- [ ] **步骤 1：重评 `patch_uniform`**

```bash
./.venv/bin/python scripts/eval_tho_small.py \
  --checkpoint runs/tho_research_v2_patch_mixer_rawish_current_loss_n3/20260617_220245_852336/checkpoint.pt \
  --metrics-output runs/tho_research_v2_patch_mixer_rawish_current_loss_n3/20260617_220245_852336/metrics.csv \
  --set training.device=cuda:0
```

预期：输出 `写出指标: .../metrics.csv`，且 CSV 包含 `rr_peak_band_abs_error`。

- [ ] **步骤 2：重评 `patch_hann`**

```bash
./.venv/bin/python scripts/eval_tho_small.py \
  --checkpoint runs/tho_research_v2_patch_mixer_hann_rawish_current_loss_n3/20260618_004519_403378/checkpoint.pt \
  --metrics-output runs/tho_research_v2_patch_mixer_hann_rawish_current_loss_n3/20260618_004519_403378/metrics.csv \
  --set training.device=cuda:0
```

预期：输出 `写出指标: .../metrics.csv`，且 CSV 包含 `rr_peak_band_abs_error`。

- [ ] **步骤 3：重评 `periodic_smooth21`**

```bash
./.venv/bin/python scripts/eval_tho_small.py \
  --checkpoint runs/tho_research_v2_periodic_unet_smooth_grid_rawish_current_loss_n3/20260618_105623_624928/checkpoint.pt \
  --metrics-output runs/tho_research_v2_periodic_unet_smooth_grid_rawish_current_loss_n3/20260618_105623_624928/metrics.csv \
  --set training.device=cuda:0
```

预期：输出 `写出指标: .../metrics.csv`，且 CSV 包含 `rr_peak_band_abs_error`。

- [ ] **步骤 4：核查重评指标**

```bash
./.venv/bin/python -c "import pandas as pd; paths={'patch_uniform':'runs/tho_research_v2_patch_mixer_rawish_current_loss_n3/20260617_220245_852336/metrics.csv','patch_hann':'runs/tho_research_v2_patch_mixer_hann_rawish_current_loss_n3/20260618_004519_403378/metrics.csv','periodic_smooth21':'runs/tho_research_v2_periodic_unet_smooth_grid_rawish_current_loss_n3/20260618_105623_624928/metrics.csv'}; rows=[]; [rows.append({'label': k, 'rr_peak_band_abs_error_mean': pd.read_csv(v)['rr_peak_band_abs_error'].mean(), 'rr_spec_abs_error_mean': pd.read_csv(v)['rr_spec_abs_error'].mean(), 'relative_envelope_mae_mean': pd.read_csv(v)['relative_envelope_mae'].mean(), 'band_limited_corr_mean': pd.read_csv(v)['band_limited_corr'].mean(), 'rr_peak_abs_error_mean': pd.read_csv(v)['rr_peak_abs_error'].mean()}) for k,v in paths.items()]; print(pd.DataFrame(rows).to_string(index=False))"
```

预期：三行都有有限的 `rr_peak_band_abs_error_mean`。

## 任务 2：运行 M2 最小新训练

**文件：**
- 本地生成：`runs/tho_research_v2_model_m2_band_rr/`

- [ ] **步骤 1：运行 `M2a_patch_hann_seed20260620`**

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
  --set model.overlap_window=hann \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260620 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_model_m2_band_rr
```

预期：run 目录包含 `config.yaml`、`train_history.csv`、`metrics.csv`、`checkpoint.pt`。

- [ ] **步骤 2：运行 `M2b_periodic_smooth21_seed20260620`**

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=periodic_unet1d_tiny \
  --set model.output_smoothing_kernel=21 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260620 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_model_m2_band_rr
```

预期：run 目录包含 `config.yaml`、`train_history.csv`、`metrics.csv`、`checkpoint.pt`。

- [ ] **步骤 3：生成 M2 汇总**

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_model_m2_band_rr \
  --output runs/tho_research_v2_model_m2_band_rr_summary.csv
```

预期：summary 包含 `selection_task_rr_peak_band_abs_error_mean`。

## 任务 3：阶段判断与收尾

**文件：**
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：追加 M2 表格**

表格列固定为：

```text
label | run | seed | best val loss | rr_peak_band_abs_error mean / median | rr_spec_abs_error mean | relative_envelope_mae mean | relative_envelope_corr mean | band_limited_corr mean | best_lag_corr mean | rr_peak_abs_error mean | 结论
```

- [ ] **步骤 2：写阶段判断**

判断规则：

- `patch_uniform` 只作为 RR 上限，不能因低频反相被选为通过模型。
- 若 `patch_hann` 与 `periodic_smooth21` 的带通 RR 接近，优先选择低频方向正确且相对包络更好的模型。
- 若新 seed 与旧 seed 排名相反，继续补第二个 seed；若排名一致，进入下一阶段结构设计。
- 若两条线都明显不稳定，暂停训练，先做样本分桶和坏例集中性分析。

- [ ] **步骤 3：运行验证**

```bash
./.venv/bin/python -m pytest tests/test_signal_metrics.py tests/test_eval_metrics.py tests/test_diagnostics_scripts.py -v
./.venv/bin/python -m py_compile resp_train/metrics/signal.py resp_train/metrics/evaluate.py scripts/summarize_tho_runs.py scripts/plot_tho_predictions.py
git diff --check
```

预期：pytest 全部通过；py_compile 无输出；`git diff --check` 无输出。

- [ ] **步骤 4：Commit**

```bash
git add docs/superpowers/plans/2026-06-18-model-m2-band-rr.md docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 M2 模型候选实验"
```

预期：只提交计划和实验台账；`runs/` 产物仍留在本地。

## 验收清单

- [ ] 重评后的三个历史候选都有 `rr_peak_band_abs_error` 列。
- [ ] 新训练只改模型候选和 `training.seed`，不混入新的 loss 变量。
- [ ] 选模主指标使用 `rr_peak_band_abs_error_mean`。
- [ ] 原始 `rr_peak_abs_error` 只作为尖峰诊断。
- [ ] 每次正式实验阶段完成后提交仓库状态。
