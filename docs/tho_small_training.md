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

## 2026-06-09 阶段结论：先优化训练目标，不急于改模型

本阶段对 `unet1d_tiny` 和 `unet1d_tiny_noskip1` 做了小规模 pilot，并额外跑了完整验证口径的 `loss.smooth_weight=0.10` 原模型实验：

- 原模型、完整训练、`smooth_weight=0.01`：`runs/tho_small/20260609_215224_229600`
- 原模型、完整训练、`smooth_weight=0.10`：`runs/tho_small/20260609_231017_850384`
- 去掉浅层 skip 的 pilot：`runs/tho_small/20260609_222153_788899`
- 去掉浅层 skip 且 `smooth_weight=0.03` 的 pilot：`runs/tho_small/20260609_222340_244989`

完整训练结果显示，增大 smooth 权重可以明显压低预测的高频粗糙度，但没有解决“预测波形和 THO 目标形状离得远”的核心问题。`smooth_weight=0.10` 相比 `0.01` 的主要变化是：

- `pred_hf_ratio_mean` 从约 `3.60` 降到约 `0.68`
- `roughness_ratio_mean` 从约 `19.31` 降到约 `3.83`
- `best_val_loss` 从约 `0.7995` 变差到约 `0.8167`
- `model_env_corr_mean` 基本没有改善，约 `0.28`

可视化复核中，部分窗口已经能匹配呼吸主频或频谱分布，但红色预测仍明显带有 BCG 式尖峰，和蓝色 THO 平滑呼吸波形差距较大。这说明当前模型更像是在学习节律/频谱，而不是学习目标波形本身。

当前 `WeakSyncLoss` 主要包含 envelope correlation、频谱分布和一阶平滑项，没有直接约束低频呼吸波形的时域形状。因此，继续单纯增加 smooth 权重或优先改模型结构，收益预计有限。下一步优先方向应是：

1. 新增呼吸频段的 waveform loss，例如对预测和目标做 `0.05-0.7Hz` band-limited 处理后计算相关性损失或 SmoothL1。
2. 考虑加入小范围 lag tolerance，例如 `±0.5s`，避免轻微时移导致可学习波形被过度惩罚。
3. 降低频谱 loss 的主导性，让它作为节律辅助项，而不是主要训练目标。
4. 在 loss 目标修正后，再判断是否需要继续调整模型结构，例如去掉浅层 skip、加低通前端或改用更强的时序模型。

建议下一步先做最小验证实验：不改模型，新增 band-limited waveform loss 和对应 waveform/best-lag-corr 诊断指标，先跑 `4096/1024` 规模 pilot；只有当可视图上低频形状明显靠近目标后，再进入完整训练。

## 2026-06-17 L1 实验口径：低频波形约束与 lag-aware 评价

L1 阶段已经补齐三类实验基础设施：

- split 独立性审计：检查 train/val 是否共享 `samp_id` 或 `(samp_id, segment_id)`；若存在重叠，实验只能作为 within-subject 开发指标。
- lag-aware 评价：`metrics.csv` 增加 `band_limited_corr`、`best_lag_corr`、`best_lag_sec`，用于区分低频形态相似度和小范围时移。
- 带限波形损失：`loss.band_waveform_weight` 默认 `0.0`，开启后对预测和目标做 `0.05-0.7Hz` torch FFT 带限重建，z-score 后计算 L1，优先约束低频相位/形态而不是绝对幅值。

旧的 2026-06-17 L0/L1/L2 pilot 已作废：这些 run 的 `data.bcg_input_key`
实际指向 `bcg_input_aligned_key`，在 research v2 索引中会解析到
`bcg_resp_band_state_aligned`，即呼吸频段且 state-aligned 的 BCG 表征，
不是 rawish wideband 主输入。后续重新建立 L0/L1 前，需要先明确主线输入
信号；当前 `configs/tho_research_v2.yaml` 已暂定为
`bcg_rawish_wideband_to_tho_timebase`。

## 常用检查命令

```bash
./.venv/bin/python -m pytest tests -v
./.venv/bin/python scripts/audit_tho_dataset.py --config configs/tho_small.yaml --output /tmp/tho_audit.csv
./.venv/bin/python scripts/baseline_tho_hilbert.py --config configs/tho_small.yaml --output /tmp/baseline_metrics.csv
```
