# 胸带小规模训练使用说明

## 范围

首版只训练 `r_tho_hat(t)` 波形，不训练 RR 头，也不把 RR 作为损失项。RR/主频、峰谷读数、频谱相似度和 envelope corr 仅作为评价指标。

当前代码只覆盖胸带输出和评价；腹带、腹带数据集与后续实验代码保留扩展空间，暂不纳入本轮训练入口。

## 默认数据

- 数据根目录：`/mnt/disk_code/marques/dataset/RespPairs/20260530_tho_ramp5_stage2_1`
- 索引：`training/dataset_index.csv`
- 默认输入集：`mixed_zscore`
- 默认窗口：180 秒，100 Hz，18000 点
- 默认模型：`unet1d_tiny`
- 训练抽样：`data.train_sample_strategy=stratified_random`
- 验证抽样：`data.val_sample_strategy=stratified_random`

`head` 抽样只用于 debug，不作为当前默认或正式实验口径。训练集和验证集的抽样 seed 相互独立：`data.train_sample_seed` 只影响训练窗口抽样，`data.val_sample_seed` 只影响验证窗口抽样。做多 run 对照时应固定 `data.val_sample_seed`，避免验证集变化掩盖模型或 loss 差异。

## 实验骨架

`BaseExperiment` 接收已经加载好的配置，负责通用实验生命周期，包括输出目录、日志、随机 seed、配置快照、训练/验证循环、best checkpoint、early stopping 和运行产物写出等与具体任务无关的流程。

`ThoExperiment` 负责胸带小规模训练的任务逻辑，包括构建数据集和 DataLoader、生成 THO 平凡基线、构建模型与 loss，并在最佳 checkpoint 上导出指标和诊断预测。

## 快速 Smoke Test

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

命令成功后会打印本次 run 目录，例如 `runs/tho_small/20260602_000725_068311`。

## 主要产物

每次运行输出到 `runs/tho_small/<timestamp>/`：

- `config.yaml`：本次运行的完整配置快照，包含命令行 `--set` 覆盖后的 resolved config。
- `audit.csv`：按 split、input_set、residual_quality_class 分组的数据可用性摘要。
- `baseline_metrics.csv`：val 子集上的平凡基线指标。
- `train_history.csv`：每轮训练和验证损失。
- `metrics.csv`：best checkpoint 对应的完整 val 子集逐窗口评价指标。
- `checkpoint.pt`：验证损失最优 checkpoint，内含模型、优化器、epoch、metrics 和训练配置快照。
- `predictions.npz`：少量诊断窗口预测，不作为完整预测归档。
- `train.log`：训练日志。

## 独立评价

训练完成后可以用 checkpoint 重新生成诊断预测和指标：

```bash
./.venv/bin/python scripts/eval_tho_small.py \
  --checkpoint runs/tho_small/<timestamp>/checkpoint.pt \
  --output /tmp/tho_predictions.npz \
  --metrics-output /tmp/tho_metrics.csv
```

默认不传 `--config` 时，脚本会读取 checkpoint 同目录的 `config.yaml`。如果显式传入 `--config` 或 `--set`，脚本会校验模型结构、验证集定义、窗口参数和损失评价频带等关键字段是否与 checkpoint 内保存的训练配置一致；不一致时直接报错。

`--metrics-output` 会覆盖完整 val 子集，`--output` 只保存 `outputs.max_prediction_windows` 个诊断窗口。

## 评价口径

BCG 原始信号的频谱主峰法和峰谷检测法只用于诊断。质量较好的片段可能能达到这些规则要求，质量较差的片段不应因为规则法失败就被直接判为研究失败；本研究的起点正是希望模型从 BCG 中学到规则法不总能稳定提取的信息。

## 常用检查命令

```bash
./.venv/bin/python -m pytest tests -v
./.venv/bin/python scripts/audit_tho_dataset.py --config configs/tho_small.yaml --output /tmp/tho_audit.csv
./.venv/bin/python scripts/baseline_tho_hilbert.py --config configs/tho_small.yaml --output /tmp/baseline_metrics.csv
```
