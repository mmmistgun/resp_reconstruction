# Rawish State-Aligned L0/L1 对齐实验计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在与历史 `patch_mixer_rawish_relenv001` 完全可比的口径下，评估 L1 带限波形损失是否改善胸带呼吸波形恢复。

**架构：** L0 使用既有历史 run 作为 anchor，不重复制造一个不同参数的基线。L1 继承历史 run 的全量数据、`patch_mixer1d`、相对包络损失、高频惩罚和训练策略，只改变 `loss.band_waveform_weight`，并保持 `loss.phase_lag_weight=0.0`。实验产物留在本地 `runs/`，只提交计划和人工台账。

**技术栈：** Python、PyTorch、OmegaConf、pandas、pytest、现有 `resp_train` 训练与评价框架。

---

## 对齐基准

历史可比基线为：

```text
runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320
```

该 run 的关键配置必须作为本轮 L1 的固定对照项：

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`target_waveform_key`
- 数据：`data.max_train_windows=null`，`data.max_val_windows=null`
- 模型：`patch_mixer1d`
- Patch 参数：`patch_len=256`，`patch_stride=128`，`mixer_layers=2`
- 损失：`envelope_weight=1.0`，`spectrum_weight=0.2`，`smooth_weight=0.1`
- 高频惩罚：`loss.high_freq_weight=0.2`
- 相对包络：`loss.relative_envelope_weight=0.01`
- 训练：`epochs=50`，`batch_size=8`，`learning_rate=0.001`
- 早停：`patience=8`，`min_delta=0.001`
- seed：`training.seed=20260610`，`data.train_sample_seed=20260610`，`data.val_sample_seed=20260611`
- 设备：`training.device=cuda:0`
- 进度：`training.show_progress=false`

本轮不使用 4096/1024 pilot 结果做结论；该口径与历史全量 run 不可直接比较。

## 科研假设

本轮继续使用 `bcg_rawish_wideband_state_aligned`，因为当前主要相位偏差来自两台采集设备采样率或时钟漂移，不是 BCG 到胸带呼吸之间的生理相位差。L1 只测试带限波形重建约束，不引入 phase-tolerant training loss。

## 文件结构

- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`
  - 职责：记录全量 `patch_mixer1d` 对齐协议、历史 L0 anchor、L1 结果和阶段判断。
- 本地生成：`runs/audits/split_independence_rawish_state_aligned_full/`
  - 职责：保存全量 split 独立性审计 CSV；该目录被 `.gitignore` 忽略，不进入 Git。
- 本地读取：`runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/`
  - 职责：作为历史 L0 anchor；不复制、不移动原始 run。
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/metrics_lagaware.csv`
  - 职责：用现有 checkpoint 重新评价历史 L0，补齐 `band_limited_corr`、`best_lag_corr` 和 `best_lag_sec`。
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_bandwave_l1/` 下的时间戳 run 目录
  - 职责：保存 L1a/L1b 训练产物；该目录被 `.gitignore` 忽略，不进入 Git。

---

### 任务 1：清理旧 pilot 记录并确认历史 L0 anchor

**文件：**
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`
- 本地删除：`runs/tho_research_v2/`
- 本地删除：`runs/audits/split_independence_rawish_state_aligned_4096_1024/`

- [ ] **步骤 1：确认旧 pilot 产物已不存在**

运行：

```bash
test ! -e runs/tho_research_v2
test ! -e runs/audits/split_independence_rawish_state_aligned_4096_1024
```

预期：两个命令都无输出且退出码为 `0`。

- [ ] **步骤 2：确认历史 L0 anchor 存在**

运行：

```bash
test -f runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/config.yaml
test -f runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/train_history.csv
test -f runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/metrics.csv
```

预期：三个命令都无输出且退出码为 `0`。

- [ ] **步骤 3：核验历史 L0 配置**

运行：

```bash
rg -n "bcg_input_key|max_train_windows|max_val_windows|name: patch_mixer1d|relative_envelope_weight|high_freq_weight|epochs|patience|min_delta|device" \
  runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/config.yaml
```

预期包含：

```text
bcg_input_key: bcg_rawish_wideband_state_aligned
max_train_windows: null
max_val_windows: null
name: patch_mixer1d
relative_envelope_weight: 0.01
high_freq_weight: 0.2
epochs: 50
patience: 8
min_delta: 0.001
device: cuda:0
```

- [ ] **步骤 4：把实验台账改成全量对齐协议**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 写入固定口径：

```markdown
- 数据窗口：全量（`data.max_train_windows=null`，`data.max_val_windows=null`）
- 模型：`patch_mixer1d`
- 历史 L0 anchor：`runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320`
- `loss.high_freq_weight=0.2`
- `loss.relative_envelope_weight=0.01`
- `loss.phase_lag_weight=0.0`
```

删除或改写所有 `4096/1024` pilot 作为本轮 L0/L1 结论的表述。

- [ ] **步骤 5：提交清理与协议修订**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md docs/superpowers/plans/2026-06-17-rawish-state-aligned-l0-l1.md
git commit -m "docs: 对齐 rawish L0 L1 全量实验计划"
```

---

### 任务 2：全量 split 独立性审计

**文件：**
- 本地生成：`runs/audits/split_independence_rawish_state_aligned_full/`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：运行全量审计**

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --output-dir runs/audits/split_independence_rawish_state_aligned_full
```

预期生成：

```text
runs/audits/split_independence_rawish_state_aligned_full/summary.csv
```

- [ ] **步骤 2：读取审计摘要**

```bash
cat runs/audits/split_independence_rawish_state_aligned_full/summary.csv
```

必须记录：

- `train_windows`
- `val_windows`
- `overlap_samp_id_count`
- `overlap_segment_count`
- `has_samp_id_leakage`
- `has_segment_leakage`

- [ ] **步骤 3：记录审计结论**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 写入：

```markdown
## Split 独立性审计

- 口径：全量（`data.max_train_windows=null`，`data.max_val_windows=null`）
- 输入：`bcg_rawish_wideband_state_aligned`
- train windows：填入 `summary.csv` 的 `train_windows`
- val windows：填入 `summary.csv` 的 `val_windows`
- train/val `samp_id` 重叠：填入 `summary.csv` 的 `overlap_samp_id_count`
- train/val `segment` 重叠：填入 `summary.csv` 的 `overlap_segment_count`
- 结论：根据重叠情况写明是 leave-samp_id-out 开发指标，还是只能作为 within-subject 开发指标。
```

- [ ] **步骤 4：提交全量审计记录**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 rawish 全量 split 审计"
```

---

### 任务 3：重评并记录历史 L0 anchor

**文件：**
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/metrics_lagaware.csv`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：用现有 checkpoint 重新生成 lag-aware 指标**

GPU 命令需要在沙盒外执行：

```bash
./.venv/bin/python scripts/eval_tho_small.py \
  --checkpoint runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/checkpoint.pt \
  --metrics-output runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/metrics_lagaware.csv
```

预期生成：

```text
runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/metrics_lagaware.csv
```

- [ ] **步骤 2：确认新指标列存在**

```bash
head -1 runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/metrics_lagaware.csv
```

预期表头包含：

```text
band_limited_corr,best_lag_corr,best_lag_sec
```

- [ ] **步骤 3：计算历史 L0 指标**

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

run_dir = Path("runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320")
history = pd.read_csv(run_dir / "train_history.csv")
metrics = pd.read_csv(run_dir / "metrics_lagaware.csv")
print("run", run_dir.name)
print("epochs", int(history["epoch"].max()))
print("best_val_loss", float(history["val_loss"].min()))
print("band_limited_corr_mean", float(metrics["band_limited_corr"].mean()))
print("best_lag_corr_mean", float(metrics["best_lag_corr"].mean()))
print("abs_best_lag_sec_mean", float(metrics["best_lag_sec"].abs().mean()))
print("relative_envelope_corr_mean", float(metrics["relative_envelope_corr"].mean()))
print("relative_envelope_mae_mean", float(metrics["relative_envelope_mae"].mean()))
PY
```

- [ ] **步骤 4：记录 L0 anchor**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 写入 L0 表格。表格字段必须
来自步骤 1 的打印结果：

- `best val loss` 使用 `best_val_loss`
- `band_limited_corr mean` 使用 `band_limited_corr_mean`
- `best_lag_corr mean` 使用 `best_lag_corr_mean`
- `abs(best_lag_sec) mean` 使用 `abs_best_lag_sec_mean`

```markdown
## L0

| run | source | model | data windows | band waveform | phase lag | rel env | high freq | best val loss | band_limited_corr mean | best_lag_corr mean | abs(best_lag_sec) mean | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
```

- [ ] **步骤 5：提交 L0 anchor 记录**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 rawish 历史 L0 anchor"
```

---

### 任务 4：运行 L1a 和 L1b

**文件：**
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_bandwave_l1/` 下的时间戳 run 目录
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：运行 L1a**

GPU 命令需要在沙盒外执行：

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
  --set loss.band_waveform_weight=0.05 \
  --set loss.phase_lag_weight=0.0 \
  --set training.epochs=50 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_bandwave_l1
```

- [ ] **步骤 2：运行 L1b**

GPU 命令需要在沙盒外执行：

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
  --set loss.band_waveform_weight=0.10 \
  --set loss.phase_lag_weight=0.0 \
  --set training.epochs=50 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_bandwave_l1
```

- [ ] **步骤 3：确认 L1 配置快照**

分别检查两个新 run 的 `config.yaml`：

```bash
for run_dir in $(find runs/tho_research_v2_patch_mixer_rawish_bandwave_l1 -mindepth 1 -maxdepth 1 -type d | sort); do
  echo "== ${run_dir} =="
  rg -n "bcg_input_key|max_train_windows|max_val_windows|name: patch_mixer1d|relative_envelope_weight|high_freq_weight|band_waveform_weight|phase_lag_weight|epochs|patience|min_delta|device" "$run_dir/config.yaml"
done
```

预期两个 run 都包含：

```text
bcg_input_key: bcg_rawish_wideband_state_aligned
max_train_windows: null
max_val_windows: null
name: patch_mixer1d
relative_envelope_weight: 0.01
high_freq_weight: 0.2
phase_lag_weight: 0.0
epochs: 50
patience: 8
min_delta: 0.001
device: cuda:0
```

且 L1a 为：

```text
band_waveform_weight: 0.05
```

L1b 为：

```text
band_waveform_weight: 0.1
```

- [ ] **步骤 4：记录 L1 结果**

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf

for run_dir in sorted(p for p in Path("runs/tho_research_v2_patch_mixer_rawish_bandwave_l1").iterdir() if p.is_dir()):
    cfg = OmegaConf.load(run_dir / "config.yaml")
    history = pd.read_csv(run_dir / "train_history.csv")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    print(
        run_dir.name,
        float(cfg.loss.band_waveform_weight),
        float(history["val_loss"].min()),
        float(metrics["band_limited_corr"].mean()),
        float(metrics["best_lag_corr"].mean()),
        float(metrics["best_lag_sec"].abs().mean()),
        float(metrics["relative_envelope_corr"].mean()),
        float(metrics["relative_envelope_mae"].mean()),
    )
PY
```

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 写入：

```markdown
## L1

| run | model | data windows | band waveform | phase lag | rel env | high freq | best val loss | band_limited_corr mean | best_lag_corr mean | abs(best_lag_sec) mean | 相对 L0 判断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
```

- [ ] **步骤 5：提交 L1 记录**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 rawish 对齐 L1"
```

---

### 任务 5：汇总对比并决定是否进入 L1c

**文件：**
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_bandwave_l1_summary.csv`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：汇总 L1 runs**

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_patch_mixer_rawish_bandwave_l1 \
  --output runs/tho_research_v2_patch_mixer_rawish_bandwave_l1_summary.csv
```

预期：

```text
runs/tho_research_v2_patch_mixer_rawish_bandwave_l1_summary.csv
```

- [ ] **步骤 2：打印 L0/L1 对比表**

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

l0_dir = Path("runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320")
l0_history = pd.read_csv(l0_dir / "train_history.csv")
l0_metrics = pd.read_csv(l0_dir / "metrics_lagaware.csv")
l0 = {
    "label": "L0-history",
    "run_id": l0_dir.name,
    "best_val_loss": float(l0_history["val_loss"].min()),
    "model_band_limited_corr_mean": float(l0_metrics["band_limited_corr"].mean()),
    "model_best_lag_corr_mean": float(l0_metrics["best_lag_corr"].mean()),
    "model_best_lag_sec_abs_mean": float(l0_metrics["best_lag_sec"].abs().mean()),
    "model_relative_envelope_corr_mean": float(l0_metrics["relative_envelope_corr"].mean()),
    "model_relative_envelope_mae_mean": float(l0_metrics["relative_envelope_mae"].mean()),
}
l1 = pd.read_csv("runs/tho_research_v2_patch_mixer_rawish_bandwave_l1_summary.csv")
l1_records = []
for run_dir in sorted(p for p in Path("runs/tho_research_v2_patch_mixer_rawish_bandwave_l1").iterdir() if p.is_dir()):
    history = pd.read_csv(run_dir / "train_history.csv")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    summary_row = l1[l1["run_id"] == run_dir.name].iloc[0].to_dict()
    summary_row.update(
        {
            "label": "L1",
            "best_val_loss": float(history["val_loss"].min()),
            "model_best_lag_sec_abs_mean": float(metrics["best_lag_sec"].abs().mean()),
        }
    )
    l1_records.append(summary_row)
frame = pd.concat([pd.DataFrame([l0]), pd.DataFrame(l1_records)], ignore_index=True, sort=False)
cols = [
    "label",
    "run_id",
    "best_val_loss",
    "model_band_limited_corr_mean",
    "model_best_lag_corr_mean",
    "model_best_lag_sec_abs_mean",
    "model_relative_envelope_corr_mean",
    "model_relative_envelope_mae_mean",
]
print(frame[cols].to_string(index=False))
PY
```

- [ ] **步骤 3：写入阶段结论**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 写入：

```markdown
## 阶段判断

- L1a 相对 L0 的 `band_limited_corr`：写明升降方向和差值。
- L1b 相对 L0 的 `band_limited_corr`：写明升降方向和差值。
- L1a/L1b 相对 L0 的 `best_lag_corr`：写明是否下降和差值。
- L1a/L1b 相对 L0 的 `abs(best_lag_sec)`：写明是否变大和差值。
- L1a/L1b 相对 L0 的 `relative_envelope_corr` 和 `relative_envelope_mae`：写明是否牺牲相对包络。
- 诊断图结论：写明预测低频形态更像胸带、仍像 BCG 尖峰，或证据不足。
- 下一步：写明进入 L1c、调整带限权重，或暂停进入模型结构实验。
```

- [ ] **步骤 4：提交汇总结论**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "docs: 总结 rawish 对齐 L0 L1"
```

---

## 自检清单

- [ ] 不引用 4096/1024 pilot 作为本轮 L0/L1 结论。
- [ ] L0 anchor 和 L1 使用同一个输入：`bcg_rawish_wideband_state_aligned`。
- [ ] L0 anchor 和 L1 使用同一个模型：`patch_mixer1d`。
- [ ] L0 anchor 和 L1 使用全量数据：`max_train_windows=null`，`max_val_windows=null`。
- [ ] L0 anchor 和 L1 都使用 `relative_envelope_weight=0.01` 和 `high_freq_weight=0.2`。
- [ ] L0/L1 都保持 `phase_lag_weight=0.0`。
- [ ] L1 只相对 L0 改变 `band_waveform_weight`。
- [ ] `runs/` 没有进入 Git。
- [ ] 每个完成的实验或紧密对照组都有中文 commit。
