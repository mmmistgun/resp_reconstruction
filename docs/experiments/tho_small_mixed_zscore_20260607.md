# THO mixed_zscore 小规模实验记录（2026-06-07）

## 目标

确认当前 `unet1d_tiny + WeakSyncLoss` 在 `mixed_zscore` 输入上的学习行为：

- 是否快速过拟合。
- 是否能学到主频。
- 是否保留峰谷形态。
- loss 权重变化是否改善输出形态。

## 固定设置

- 数据集：`/mnt/disk_code/marques/dataset/RespPairs/20260530_tho_ramp5_stage2_1`
- `input_set`：`mixed_zscore`
- train windows：512
- val windows：256
- model：`unet1d_tiny`
- `base_channels`：16
- batch size：8
- val 质量分层：`near_zero_residual=208`，`stable_nonzero_residual=48`

## Run 记录

| run_id | 变量 | best_epoch | best_val_loss | model rr_spec MAE | model rr_peak MAE | model corr | model spec_sim | 初步结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `20260607_102608_834359` | lr=0.001，epoch=50，默认 loss | 3 | 0.6906 | 1.3332 | 9.5589 | 0.5077 | 0.8396 | 第 3 轮后明显过拟合；主频好于 baseline，但峰谷和频谱形态差 |
| `20260607_103206_584999` | lr=0.0003，epoch=20，默认 loss | 2 | 0.6918 | 1.3905 | 9.7040 | 0.5089 | 0.8385 | 降低学习率没有明显改善，best 更早出现 |
| `20260607_103511_604663` | envelope=2.0，spectrum=0.1，smooth=0.01 | 2 | 1.1181 | 1.4019 | 9.2428 | 0.4919 | 0.8303 | 加强 envelope 后 loss 尺度变大，整体指标没有改善 |
| `20260607_103617_574202` | envelope=1.0，spectrum=0.2，smooth=0.001 | 6 | 0.6704 | 1.4763 | 9.7444 | 0.5065 | 0.9003 | 降低 smooth 后 val loss 和频谱相似度改善，但峰谷仍差 |

baseline（同一 val 子集）：

| rr_spec MAE | rr_peak MAE | envelope corr | spectrum similarity |
| ---: | ---: | ---: | ---: |
| 3.2387 | 1.8141 | 0.4867 | 0.9370 |

## 采样口径修正

这 4 个 run 使用首版前缀采样口径。真实索引检查显示，前缀窗口会集中到少数 `samp_id` 和片段，
因此这些结果只作为 smoke test 和现象记录，不作为正式消融结论。

后续消融应基于新实验骨架的 `stratified_random` 默认策略，并固定 `val_sample_seed`，再做多 seed 比较。

## 当前判断

1. 当前模型能学到主频线索。所有模型 run 的 `rr_spec_abs_error` 均明显低于平凡基线。
2. 峰谷形态仍不稳定。`rr_peak_abs_error` 长期在 9 bpm 左右，明显差于平凡基线。
3. 默认 smooth 权重可能过强。`smooth_weight=0.001` 的 run 将 `spectrum_similarity` 从约 0.84 提升到 0.90，并得到当前最低 `best_val_loss=0.6704`。
4. 单纯降低学习率不是关键瓶颈。`lr=0.0003` 与 `lr=0.001` 的结果非常接近。
5. 加强 envelope 权重没有带来收益，反而使验证 loss 变差。

## 下一步

先重跑基于 `stratified_random` 的分层随机对照，不继续扩大旧前缀采样结论：

- 固定 `data.val_sample_strategy=stratified_random` 和 `data.val_sample_seed`。
- 保持相同 train/val 窗口规模，先重跑默认 loss 与 `smooth_weight=0.001` 两组。
- 每组再做多 `data.train_sample_seed` 比较，确认现象是否稳定。
- 保持 `training.learning_rate=0.001`。
- 保持 `training.epochs=10`。
- 重点看 `spectrum_similarity`、`rr_peak_abs_error` 和诊断图中的峰谷形态。

暂不加入 RR 头或 RR 派生损失。当前问题更像是波形形态和峰谷结构没有保住，应先确认 smooth 与频谱/包络 loss 的权衡。
