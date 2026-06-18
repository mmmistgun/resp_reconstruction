# M4 Patch-Hann 输出平滑多 seed 复核实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 复核 `patch_mixer1d + hann + output_smoothing_kernel=5/11` 是否在同 seed 下稳定优于未平滑的 Patch-Hann。

**架构：** 不改训练代码，只跑全量数据、多 seed 对照实验。先跑 `training.seed=20260630` 的 baseline、smoothing5 与 smoothing11；若主指标或护栏排序摇摆，再补 `training.seed=20260640`。

**技术栈：** Python、PyTorch、OmegaConf、pandas、现有 `resp_train` 训练与 `summarize_tho_runs.py` 汇总脚本。

---

## 背景与固定口径

M3 已完成单 seed 小网格：

- `patch_stride=64` 带通 RR 明显变差，不继续。
- `output_smoothing_kernel=5` 主指标最好：`rr_peak_band_abs_error_mean=0.460197`。
- `output_smoothing_kernel=11` raw peak 与 `rr_spec_abs_error` 略好，但带通 RR 略弱：`rr_peak_band_abs_error_mean=0.468274`。
- 两个平滑候选都保持正 `band_limited_corr`，说明没有回到反相区域。

M4 固定：

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 数据窗口：全量（`data.max_train_windows=null`，`data.max_val_windows=null`）
- 模型：`patch_mixer1d`
- Patch 参数：`patch_len=256`，`patch_stride=128`，`mixer_layers=2`
- Overlap：`model.overlap_window=hann`
- Loss：N3 主线，`high_freq_weight=0.2`，`relative_envelope_weight=0.03`，`phase_alignment_weight=0.005`
- 训练：`epochs=50`，`batch_size=128`，`patience=8`，`min_delta=0.001`，`use_amp=false`
- 数据 seed：`data.train_sample_seed=20260610`，`data.val_sample_seed=20260611`
- 首轮候选：`output_smoothing_kernel=1/5/11`
- 首轮训练 seed：`training.seed=20260630`
- 条件补跑 seed：`training.seed=20260640`
- 输出：`runs/tho_research_v2_patch_hann_m4_smoothing_multiseed`
- 汇总：`runs/tho_research_v2_patch_hann_m4_smoothing_multiseed_summary.csv`

选择标准：

1. 主指标：优先比较 `rr_peak_band_abs_error_mean`。
2. 频域护栏：`rr_spec_abs_error_mean` 不应明显恶化。
3. 方向护栏：`band_limited_corr_mean` 必须保持为正。
4. 相对努力：`relative_envelope_mae_mean` 与 `relative_envelope_corr_mean` 不应明显退化。
5. 毛刺诊断：raw `rr_peak_abs_error_mean/median` 仅用于判断输出平滑收益，不作为单独通过标准。

补第二组 seed 的触发条件：

- `smoothing5` 或 `smoothing11` 没有在同 seed 下优于 `output_smoothing_kernel=1` baseline。
- `smoothing5` 与 `smoothing11` 在 `training.seed=20260630` 上主指标差距小于 `0.03 bpm`。
- 任一候选出现 `band_limited_corr_mean <= 0`。
- `smoothing5` 主指标更好但 raw peak 明显恶化，或 `smoothing11` raw peak 更好但主指标明显恶化，导致结论不稳定。

## 文件结构

- 创建：`docs/superpowers/plans/2026-06-18-patch-hann-m4-smoothing-multiseed.md`
  - 职责：记录 M4 实验设计、命令、判断规则和验收清单。
- 本地生成：`runs/tho_research_v2_patch_hann_m4_smoothing_multiseed/`
  - 职责：保存 M4 训练 run，不进入 Git。
- 本地生成：`runs/tho_research_v2_patch_hann_m4_smoothing_multiseed_summary.csv`
  - 职责：保存 M4 汇总表，不进入 Git。
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`
  - 职责：追加 M4 结果和阶段判断。

## 任务 1：提交 M4 计划

**文件：**
- 创建：`docs/superpowers/plans/2026-06-18-patch-hann-m4-smoothing-multiseed.md`

- [ ] **步骤 1：检查计划格式**

运行：

```bash
git diff --check
```

预期：无输出。

- [ ] **步骤 2：提交计划**

运行：

```bash
git add docs/superpowers/plans/2026-06-18-patch-hann-m4-smoothing-multiseed.md
git commit -m "docs: 添加 M4 输出平滑多 seed 计划"
```

预期：生成一个只包含计划文档的提交。

## 任务 2：运行首轮 baseline/smoothing5/smoothing11

**文件：**
- 本地生成：`runs/tho_research_v2_patch_hann_m4_smoothing_multiseed/`

- [ ] **步骤 1：运行 M4 baseline seed20260630**

运行：

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
  --set model.output_smoothing_kernel=1 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260630 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_hann_m4_smoothing_multiseed
```

预期：写出未平滑 Patch-Hann baseline run，包含 `metrics.csv` 和 `config.yaml`。

- [ ] **步骤 2：运行 M4 smoothing5 seed20260630**

运行：

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
  --set model.output_smoothing_kernel=5 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260630 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_hann_m4_smoothing_multiseed
```

预期：写出 `output_smoothing_kernel=5` 的 run，包含 `metrics.csv` 和 `config.yaml`。

- [ ] **步骤 3：运行 M4 smoothing11 seed20260630**

运行：

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
  --set model.output_smoothing_kernel=11 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260630 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_hann_m4_smoothing_multiseed
```

预期：写出 `output_smoothing_kernel=11` 的 run，包含 `metrics.csv` 和 `config.yaml`。

## 任务 3：汇总并判断是否补 seed20260640

**文件：**
- 本地生成：`runs/tho_research_v2_patch_hann_m4_smoothing_multiseed_summary.csv`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：生成 M4 summary**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_patch_hann_m4_smoothing_multiseed \
  --output runs/tho_research_v2_patch_hann_m4_smoothing_multiseed_summary.csv
```

预期：summary 至少包含三行，并且有 `selection_task_rr_peak_band_abs_error_mean`。

- [ ] **步骤 2：生成 M3/M4 对照表**

运行：

```bash
./.venv/bin/python -c "import pandas as pd; paths=['runs/tho_research_v2_patch_hann_m3_antispike_summary.csv','runs/tho_research_v2_patch_hann_m4_smoothing_multiseed_summary.csv']; df=pd.concat([pd.read_csv(p).assign(source=p) for p in paths], ignore_index=True); cols=['source','run_id','best_val_loss','selection_task_rr_peak_band_abs_error_mean','selection_task_rr_peak_band_abs_error_median','selection_task_rr_spec_abs_error_mean','selection_task_relative_envelope_mae_mean','selection_task_relative_envelope_corr_mean','selection_waveform_band_limited_corr_mean','selection_waveform_best_lag_corr_mean','selection_waveform_rr_peak_abs_error_mean','selection_waveform_rr_peak_abs_error_median']; print(df[cols].to_string(index=False))"
```

预期：M3 的 seed20260620、M4 baseline 和 M4 smoothing 候选均可直接比较。

- [ ] **步骤 3：按触发条件决定是否补跑**

判断：

- 若 `smoothing5` 或 `smoothing11` 在同 seed 下优于 baseline，且 `band_limited_corr_mean > 0`，可保留为通过候选。
- 若 `smoothing11` 主指标接近但 raw peak 更稳，记录为抗毛刺备选。
- 若 seed20260630 与 seed20260620 排名矛盾，进入任务 4。

## 任务 4：条件补跑 seed20260640

**文件：**
- 本地生成：`runs/tho_research_v2_patch_hann_m4_smoothing_multiseed/`

- [ ] **步骤 1：补跑 baseline seed20260640**

运行任务 2 步骤 1 的命令，仅把：

```bash
--set training.seed=20260630
```

改为：

```bash
--set training.seed=20260640
```

预期：新增一个 `output_smoothing_kernel=1` 的 baseline run。

- [ ] **步骤 2：补跑 smoothing5 seed20260640**

运行任务 2 步骤 2 的命令，仅把：

```bash
--set training.seed=20260630
```

改为：

```bash
--set training.seed=20260640
```

预期：新增一个 `output_smoothing_kernel=5` 的 run。

- [ ] **步骤 3：补跑 smoothing11 seed20260640**

运行任务 2 步骤 3 的命令，仅把：

```bash
--set training.seed=20260630
```

改为：

```bash
--set training.seed=20260640
```

预期：新增一个 `output_smoothing_kernel=11` 的 run。

- [ ] **步骤 4：重新生成 summary**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_patch_hann_m4_smoothing_multiseed \
  --output runs/tho_research_v2_patch_hann_m4_smoothing_multiseed_summary.csv
```

预期：summary 至少包含六行。

## 任务 5：记录、验证与提交

**文件：**
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：追加 M4 表格**

表格列：

```text
label | run | seed | smoothing | best epoch | best val loss | rr_peak_band_abs_error mean / median | rr_spec_abs_error mean | relative_envelope_mae mean | relative_envelope_corr mean | band_limited_corr mean | best_lag_corr mean | raw rr_peak_abs_error mean / median | 结论
```

- [ ] **步骤 2：写阶段判断**

判断必须覆盖：

- baseline、`smoothing5` 与 `smoothing11` 在主指标上的稳定性。
- 低频方向是否保持为正。
- raw peak 诊断是否支持抗毛刺收益。
- 下一步是否进入低自由度/带限输出结构设计。

- [ ] **步骤 3：运行验证**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py tests/test_signal_metrics.py tests/test_eval_metrics.py tests/test_diagnostics_scripts.py -v
./.venv/bin/python -m py_compile resp_train/models/timeseries.py resp_train/models/registry.py scripts/summarize_tho_runs.py
git diff --check
```

预期：pytest 全部通过；py_compile 退出码为 0；diff check 无输出。

- [ ] **步骤 4：提交实验记录**

运行：

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 M4 输出平滑多 seed 复核"
```

预期：生成一个只包含实验记录更新的提交。

## 验收清单

- [ ] 首轮至少包含 `output_smoothing_kernel=1`、`5` 与 `11` 的同 seed 对照。
- [ ] 所有正式比较均使用全量数据和相同数据 seed。
- [ ] 主选模仍使用 `rr_peak_band_abs_error_mean`。
- [ ] `band_limited_corr_mean` 为负的候选不得作为通过模型。
- [ ] `runs/` 产物不进入 Git。
- [ ] 阶段完成后提交 Git 状态。
