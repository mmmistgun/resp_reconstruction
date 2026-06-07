# 脚本索引

本目录暂时保持脚本平铺，不移动文件。当前数量还少，平铺能减少命令路径变化；等腹带、消融、整夜推理和更多诊断脚本加入后，再考虑按子目录分类。

## 数据检查

### `audit_tho_dataset.py`

生成胸带小规模训练数据审计表，检查 split、input_set、残差质量分层和可用窗口数量。

```bash
./.venv/bin/python scripts/audit_tho_dataset.py \
  --config configs/tho_small.yaml \
  --output /tmp/tho_audit.csv
```

## 训练与评价

### `train_tho_small.py`

训练胸带参考小规模模型，并输出 run 目录。默认使用 `configs/tho_small.yaml`，可用 `--set` 覆盖配置。

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=8
```

### `eval_tho_small.py`

从 `checkpoint.pt` 重新生成诊断预测和可选指标。默认读取 checkpoint 同目录的 `config.yaml`，并校验关键配置一致性。

```bash
./.venv/bin/python scripts/eval_tho_small.py \
  --checkpoint runs/tho_small/<timestamp>/checkpoint.pt \
  --output /tmp/tho_predictions.npz \
  --metrics-output /tmp/tho_metrics.csv
```

## 诊断分析

### `baseline_tho_hilbert.py`

在 val 子集运行平凡基线，输出逐窗口 RR/主频、峰谷、包络相关和频谱相似度指标。

```bash
./.venv/bin/python scripts/baseline_tho_hilbert.py \
  --config configs/tho_small.yaml \
  --output /tmp/baseline_metrics.csv
```

### `plot_tho_predictions.py`

读取一个 run 的 `predictions.npz` 和 `metrics.csv`，生成预测/参考波形诊断图。默认输出到 `<run-dir>/plots/`。

```bash
./.venv/bin/python scripts/plot_tho_predictions.py \
  --run-dir runs/tho_small/<timestamp> \
  --max-plots 8
```

### `summarize_tho_runs.py`

汇总 `runs/tho_small/*` 下各 run 的训练损失、模型指标、平凡基线指标和审计数量。

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_small \
  --output /tmp/tho_runs_summary.csv
```

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
