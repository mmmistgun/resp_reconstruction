# BCG–CWT patch Mamba v1 训练接口与证据边界

协议 ID：`bcg-cwt-patch-mamba-v1-20261014`。

工程状态：模型实现、合成数据工程检查和官方 Mamba2 CUDA 合成检查已完成。正式 train/val 运行记录位于 `runs/bcg_cwt_patch_mamba_v1/`，各运行的启动、完成与失败状态以目录内回执为准。模型、命令及实际检查记录见[使用说明](../models/bcg_cwt_patch_mamba.md)。

## 模型与输入

结构默认值来自 `configs/bcg_cwt_patch_mamba_v1/model.yaml`。窗口为 100 Hz、18000 点，输入 BCG `[B,1,18000]` 与完整 W-CWT `[B,97,360]`，输出为 float32 呼吸波形 `[B,1,18000]`。每 100 点形成一个 token，与两个连续 CWT 时间列对齐；六层双向 Mamba2 在融合后处理 180 个 token。

实际频率来自当前 cache 的 `w_frequencies_hz.npy`，保存于 checkpoint。缓存 manifest、频率文件和 split 行身份复用现有冻结校验。模型独立配置允许 100 点 hop；旧 `PatchTFConfig` 的 50 点 hop、41 行 H-CWT 定义继续用于其原有模型。

## 数据、损失和选择规则

训练配置复用 `configs/crd_tf_v1/crd_tf101_m_smoke.yaml` 的 data、window、loss、evaluation 和优化器字段，由本模型 `train.yaml` 指定完整 train/val 与新运行身份。数据使用 `research_v2` 索引中原有 train/val 划分；BCG 使用 `bcg_rawish_segment_soft_z_key`，目标使用 `target_waveform_segment_soft_z_key`。采样 seed 沿用 train=20260610、val=20260611。启动时保存实际行表，检查受试者和样本跨 split 隔离。

任务损失复用 `RespirationTaskLoss`，频带为 0.05–0.70 Hz，sync 权重 1、effort 权重 0.25、最大 lag 0.30 秒、包络窗口/步长 10/5 秒。目标资格、统计单位和指标实现保持现有定义。checkpoint 采用 validation Local RR MAE 严格最小值，平局保留更早 epoch；逐样本指标使用 `evaluate_task_predictions(include_test_only=False)`。

本模型使用独立架构和运行目录，历史结果保持各自的实验身份。数据定义、任务损失、指标、统计单位和 Local RR 选点口径沿用既有实现。每轮完整指标只计算一次，Local RR selector 复用逐样本表中的 target-eligible 窗口均值；eligible 非有限值显式失败。

2026-10-14 增量修订：默认开启基于 validation Local RR MAE 的 early stopping，参数为 min_epoch=30、patience=15、min_delta=0.0。从首轮累计未改善次数；指标严格低于早停参考值减去 min_delta 时更新参考值并清零。epoch 达到 min_epoch 且等待次数达到 patience 时，保存完整本轮产物后停止。RR 最优 checkpoint 始终按严格最小值选择；自定义非零 min_delta 只影响早停参考。每轮 record/checkpoint 保存参考值、等待次数与停止状态，完成回执保存 `early_stopped`。

另行记录 validation 总 loss 的严格最小值，平局保留较早 epoch。每轮 record/checkpoint 保存 `best_val_loss`、`best_loss_epoch`；正常结束（含触发早停）时，将所选完整 checkpoint 复制为运行目录中的 `best_loss.pt`，在 `completed.json` 中记录路径、来源 epoch 和 SHA-256。`best_checkpoint` 继续由 Local RR MAE 选择。最低 loss 的选择与保存不参与早停判断，也不重置其等待次数。此修订用于后续新运行，历史产物不补写。

## 运行与记录

开发者使用 `scripts/train_bcg_cwt_patch_mamba.py` 启动。默认最多训练 80 epoch、微批量 2、梯度累积 64 次、有效 batch 128、AdamW、bfloat16 AMP，构造 seed 和训练 seed 均为 20261013；完整解析值由 `--print-config` 输出。按 10,141 个训练窗口计算，每 epoch 为 80 次更新，最大预算为 6,400 次，warmup 为 320 次。学习率计划按最大预算计算，早停结束后保留已完成的训练前缀。此训练预算修订用于新运行，不能在恢复时更改已有运行的 batch、累积或最大 epoch。

新运行使用全新 `outputs.run_root`。训练保存解析配置、源码快照与文件哈希、Git commit、环境版本、实际频率、数据访问记录、split 行表、优化器分组、完整 epoch history、每 epoch validation metrics 与 checkpoint。`--config <运行目录>/resolved_config.yaml --resume` 从最后一个通过回执校验的完整 epoch 恢复；完整 checkpoint 保存模型、优化器、全部使用到的 RNG、更新预算及计数、RR 最优与早停状态、loss 最优状态。

恢复核对配置、源码、环境、数据 index/cache 与实际行顺序。失败 epoch 保留，重试使用独立目录；恢复尝试记录在 `attempts/` 下。已触发早停或到达预算时只补齐完成产物，已经完成的运行校验后直接返回。回执作为完整 epoch 的提交标记，内容包含 checkpoint、record 和 metrics 的哈希；损坏或不连续的完整记录显式失败。恢复开关与终端进度显示不属于训练身份，其余配置保持一致。

训练只访问 train/val。本协议不开放 research-test；未来独立测试集评价需要匹配的专项协议及当次授权。真实数据训练默认由开发者执行。

## 工程验收与结论范围

本轮自动检查仅使用人工 tensor、临时 NPZ/cache 和官方内核上的少量合成样本。CPU 依赖注入用于接口与外围算子验证；官方 CUDA 测试验证默认长度前后向、任务损失、参数更新及 checkpoint 往返。

这些检查支持实现和接入正确性。真实波形重建质量、输出片段边界质量、泛化与运行吞吐由后续训练和评价确定。
