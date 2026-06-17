# Rawish State-Aligned L0/L1 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 从 L0/L1 重新建立 THO 波形恢复实验，主输入固定为 `bcg_rawish_wideband_state_aligned`。

**架构：** 只修改配置、文档和实验执行口径，不改模型结构。L0 关闭带限波形损失，L1a/L1b 只改变 `loss.band_waveform_weight`，`phase_lag_weight` 始终保持 `0.0`。实验产物留在本地 `runs/`，只提交配置、计划和人工台账。

**技术栈：** Python、PyTorch、OmegaConf、pytest、现有 `resp_train` 训练与评价框架。

---

## 科研假设

本轮使用 `bcg_rawish_wideband_state_aligned`，因为当前主要相位偏差来自两台采集
设备采样率或时钟漂移，不是 BCG 到胸带呼吸之间的生理相位差。L0/L1 先隔离
宽频 BCG 到胸带波形恢复问题，不使用 phase-tolerant training loss。

## 文件结构

- 修改：`configs/tho_research_v2.yaml`
  - 职责：将默认 research v2 输入设为 `bcg_rawish_wideband_state_aligned`。
- 修改：`tests/test_config.py`
  - 职责：防止默认输入口径回退到 `to_tho_timebase` 或 resp-band。
- 修改：`docs/tho_small_training.md`
  - 职责：记录旧 L0/L1 作废原因和新 L0/L1 科研口径。
- 修改：`scripts/README.md`
  - 职责：给出 L0/L1 运行命令和判断规则。
- 创建：`docs/superpowers/specs/2026-06-17-rawish-state-aligned-l0-l1-design.md`
  - 职责：保存设计依据、指标和进入下一阶段条件。
- 创建：`docs/experiments/rawish_state_aligned_l0_l1.md`
  - 职责：记录后续 L0/L1 运行结果；该文件可以进入 Git。
- 本地生成：`runs/tho_research_v2/` 下的时间戳 run 目录
  - 职责：保存训练产物；该目录被 `.gitignore` 忽略，不进入 Git。

---

### 任务 1：配置口径和测试保护

**文件：**
- 修改：`configs/tho_research_v2.yaml`
- 修改：`tests/test_config.py`

- [ ] **步骤 1：确认默认输入配置**

检查 `configs/tho_research_v2.yaml`：

```yaml
data:
  bcg_input_key: bcg_rawish_wideband_state_aligned
```

- [ ] **步骤 2：确认配置测试断言**

检查 `tests/test_config.py::test_load_research_v2_config_has_expected_format`：

```python
assert cfg.data.bcg_input_key == "bcg_rawish_wideband_state_aligned"
```

- [ ] **步骤 3：运行配置测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_config.py tests/test_research_v2_data.py -q
```

预期：

```text
11 passed
```

- [ ] **步骤 4：提交配置口径**

```bash
git add configs/tho_research_v2.yaml tests/test_config.py
git commit -m "fix: 使用 state-aligned rawish 输入"
```

---

### 任务 2：split 独立性审计

**文件：**
- 本地生成：`runs/audits/split_independence_rawish_state_aligned_4096_1024/`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：运行审计**

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --output-dir runs/audits/split_independence_rawish_state_aligned_4096_1024
```

预期：

```text
runs/audits/split_independence_rawish_state_aligned_4096_1024/summary.csv
```

- [ ] **步骤 2：读取审计摘要**

```bash
cat runs/audits/split_independence_rawish_state_aligned_4096_1024/summary.csv
```

预期检查：

- `overlap_samp_id_count=0`
- `overlap_segment_count=0`

- [ ] **步骤 3：记录审计结论**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 写入：

```markdown
## Split 独立性审计

- 口径：`4096/1024`
- 输入：`bcg_rawish_wideband_state_aligned`
- train/val `samp_id` 重叠：0
- train/val segment 重叠：0
- 结论：本轮 pilot 可作为 leave-samp_id-out 开发指标。
```

如果重叠不为 0，把最后一行改为：

```markdown
- 结论：本轮 pilot 只能作为 within-subject 开发指标。
```

- [ ] **步骤 4：提交审计记录**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 rawish state-aligned split 审计"
```

---

### 任务 3：运行 L0 基线

**文件：**
- 本地生成：`runs/tho_research_v2/` 下的时间戳 run 目录
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：运行 L0**

GPU 命令需要在沙盒外执行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set loss.band_waveform_weight=0.0 \
  --set loss.phase_lag_weight=0.0 \
  --set training.epochs=3 \
  --set training.device=cuda:0 \
  --set outputs.run_root=runs/tho_research_v2
```

预期：

```text
runs/tho_research_v2/`date-like timestamp`
```

- [ ] **步骤 2：确认 L0 配置快照**

```bash
RUN_DIR=$(find runs/tho_research_v2 -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
rg -n "bcg_input_key|band_waveform_weight|phase_lag_weight" "$RUN_DIR/config.yaml"
```

预期包含：

```text
bcg_input_key: bcg_rawish_wideband_state_aligned
band_waveform_weight: 0.0
phase_lag_weight: 0.0
```

- [ ] **步骤 3：记录 L0 run**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 写入 L0 表格。表格中的
`run` 使用 `basename "$RUN_DIR"`，其余数值从 `train_history.csv` 和
`metrics.csv` 统计得到：

```markdown
## L0

| run | input | band waveform | phase lag | best val loss | band_limited_corr mean | best_lag_corr mean | abs(best_lag_sec) mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
```

数值计算命令：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

run_dir = Path(sorted(p for p in Path("runs/tho_research_v2").iterdir() if p.is_dir())[-1])
history = pd.read_csv(run_dir / "train_history.csv")
metrics = pd.read_csv(run_dir / "metrics.csv")
print("run", run_dir.name)
print("best_val_loss", float(history["val_loss"].min()))
print("band_limited_corr_mean", float(metrics["band_limited_corr"].mean()))
print("best_lag_corr_mean", float(metrics["best_lag_corr"].mean()))
print("abs_best_lag_sec_mean", float(metrics["best_lag_sec"].abs().mean()))
PY
```

- [ ] **步骤 4：提交 L0 记录**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 rawish state-aligned L0"
```

---

### 任务 4：运行 L1a 和 L1b

**文件：**
- 本地生成：`runs/tho_research_v2/` 下的时间戳 run 目录
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：运行 L1a**

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set loss.band_waveform_weight=0.05 \
  --set loss.phase_lag_weight=0.0 \
  --set training.epochs=3 \
  --set training.device=cuda:0 \
  --set outputs.run_root=runs/tho_research_v2
```

- [ ] **步骤 2：运行 L1b**

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set loss.band_waveform_weight=0.10 \
  --set loss.phase_lag_weight=0.0 \
  --set training.epochs=3 \
  --set training.device=cuda:0 \
  --set outputs.run_root=runs/tho_research_v2
```

- [ ] **步骤 3：确认两个 run 的配置快照**

分别运行：

```bash
RUN_DIR=$(find runs/tho_research_v2 -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
rg -n "bcg_input_key|band_waveform_weight|phase_lag_weight" "$RUN_DIR/config.yaml"
```

预期两个 run 都包含：

```text
bcg_input_key: bcg_rawish_wideband_state_aligned
phase_lag_weight: 0.0
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

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 追加 L1 表格。每行的 `run`
使用对应 run 目录名，指标字段使用下方命令统计：

```markdown
## L1

| run | input | band waveform | phase lag | best val loss | band_limited_corr mean | best_lag_corr mean | abs(best_lag_sec) mean | 相对 L0 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
```

数值计算命令：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf

for run_dir in sorted(p for p in Path("runs/tho_research_v2").iterdir() if p.is_dir())[-2:]:
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
    )
PY
```

- [ ] **步骤 5：提交 L1 记录**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 rawish state-aligned L1"
```

---

### 任务 5：汇总与是否进入 L1c

**文件：**
- 本地生成：`runs/tho_research_v2_rawish_state_aligned_l0_l1_summary.csv`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：汇总 runs**

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2 \
  --output runs/tho_research_v2_rawish_state_aligned_l0_l1_summary.csv
```

预期：

```text
runs/tho_research_v2_rawish_state_aligned_l0_l1_summary.csv
```

- [ ] **步骤 2：筛选本轮 run**

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv("runs/tho_research_v2_rawish_state_aligned_l0_l1_summary.csv")
cols = [
    "run_id",
    "best_val_loss",
    "model_band_limited_corr_mean",
    "model_best_lag_corr_mean",
    "model_best_lag_sec_mean",
    "model_envelope_corr_mean",
    "model_spectrum_similarity_mean",
]
print(summary[cols].tail(10).to_string(index=False))
PY
```

- [ ] **步骤 3：写入阶段结论**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 追加阶段判断。每一项必须
根据 L0/L1 的实际统计结果写成明确结论：

```markdown
## 阶段判断

- L1a 是否提升 `band_limited_corr`：是或否，并写明差值。
- L1b 是否提升 `band_limited_corr`：是或否，并写明差值。
- `best_lag_corr` 是否下降：是或否，并写明差值。
- `abs(best_lag_sec)` 是否变大：是或否，并写明差值。
- 诊断图结论：写明预测低频形态更像胸带、仍像 BCG 尖峰，或证据不足。
- 下一步：写明进入 L1c、调整 loss 权重，或暂停进入模型结构实验。
```

- [ ] **步骤 4：提交汇总结论**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "docs: 总结 rawish state-aligned L0 L1"
```

---

## 自检清单

- [ ] 默认配置使用 `bcg_rawish_wideband_state_aligned`。
- [ ] 所有 L0/L1 run 都显式设置 `loss.phase_lag_weight=0.0`。
- [ ] `runs/` 没有进入 Git。
- [ ] 每个完成的实验或紧密对照组都有中文 commit。
- [ ] 旧 resp-band L0/L1/L2 结果没有被当作本轮结论引用。
