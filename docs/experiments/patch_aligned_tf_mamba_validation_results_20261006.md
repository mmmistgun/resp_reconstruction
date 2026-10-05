# Patch-aligned TF-Mamba 时长敏感性 validation 结果

2026-10-06：1/2/4 秒 patch × 三 seed 的 9-cell train/validation 已完成，训练及本轮 validation 汇总关闭。当前状态以本记录为准；运行时训练协议及源代码快照保留原位。独立测试集不在本轮范围内。

## 结果

各 cell 使用完整 2675 个 validation 窗口、7 名受试者。checkpoint 按每 epoch 完整 validation Local RR MAE 的严格最小值选择，相等时保留最早 epoch。下表为三个 seed 的均值 ± 样本标准差；RR 单位为 bpm。

| Patch 时长 | Whole RR MAE ↓ | Local RR MAE ↓ | 包络轨迹 MAE ↓ | 全局包络调制误差 ↓ | Lag-aware signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| 1 秒 | 0.479238 ± 0.011546 | 0.518260 ± 0.013716 | 0.151721 ± 0.004169 | 0.182342 ± 0.004796 | 0.867415 ± 0.002442 |
| 2 秒 | 0.473621 ± 0.013995 | 0.523665 ± 0.002629 | 0.152230 ± 0.006105 | 0.181983 ± 0.001970 | 0.866825 ± 0.000424 |
| 4 秒 | 0.460585 ± 0.022594 | 0.524450 ± 0.008530 | 0.151555 ± 0.005850 | 0.198298 ± 0.021413 | 0.851686 ± 0.001824 |

1 秒在 Local RR 与 PCC 的三 seed 均值上领先；Local RR 比 2 秒低 0.005405 bpm，差距较小，本轮仅包含三个 seed，未进行显著性检验。2 秒的 Local RR、全局调制误差和 PCC 跨 seed 波动更小。4 秒的 Whole RR 均值最低，包络轨迹误差近似，但 PCC 更低且全局调制误差更高。

本轮显示时长影响存在指标间取舍，没有同时支配五项指标的候选。结果为同一 validation 集上的结构敏感性证据，不用于声称独立测试集泛化改善。

## 训练与选点

| Patch | Seed | Best epoch | 完成 epoch | Optimizer updates | Best Local RR |
|---|---:|---:|---:|---:|---:|
| 1 秒 | 20260811 | 5 | 30 | 2400 | 0.511628 |
| 1 秒 | 20260812 | 12 | 30 | 2400 | 0.534031 |
| 1 秒 | 20260813 | 9 | 30 | 2400 | 0.509120 |
| 2 秒 | 20260811 | 5 | 30 | 2400 | 0.521801 |
| 2 秒 | 20260812 | 6 | 30 | 2400 | 0.522524 |
| 2 秒 | 20260813 | 8 | 30 | 2400 | 0.526672 |
| 4 秒 | 20260811 | 33 | 48 | 3840 | 0.519965 |
| 4 秒 | 20260812 | 4 | 30 | 2400 | 0.534288 |
| 4 秒 | 20260813 | 3 | 30 | 2400 | 0.519098 |

全部 cell 按既定 min_epoch=30、patience=15 早停；其中八组在 epoch 30 结束，一组在 epoch 48 结束。最优 checkpoint 早于 min_epoch 合法：min_epoch 限制停止时点，不限制候选 checkpoint 的 epoch。

## 来源与完成核对

本轮完成核对只读取既有报告、checkpoint 字节和指标 CSV，未运行训练、模型推理或独立测试集评价。9 组完成回执所列 best/final checkpoint、配置、history 与 validation 指标文件全部通过 SHA-256 校验；各 cell 的 validation 分母与受试者数量一致。汇总入口同时核对跨 cell 样本顺序及 target 资格。

- 训练协议：`docs/experiments/patch_aligned_tf_mamba_training_v1_20261005.md`。
- Session：`runs/patch_aligned_tf_mamba/formal_v1_20261005/`。
- Session SHA-256：`ad34f3a03680cadd88000ee7bcf3c2a7053fd48d501cdb9153b2e4062f07ae3c`。
- 源码快照 SHA-256：`e5913ea46fdf56561b4143df6ac7b0ee2940fe51c40417884a9f4ac6f3ef0046`。
- 汇总：`summary/8549bd9021c54ac485c8461da36f61f1/`，位于上述 session 内。
- 汇总 receipt SHA-256：`5394a9187db44588dbc1649f0c592982f742318b428c3d5a438607473996fd7f`。
- 汇总文件：`per_seed.csv`、`across_seed.csv`、`receipt.json`。

运行产物不进入 Git。已完成的训练、validation 与汇总不自行重跑；新增实验或独立评价使用相应的新协议及授权。
