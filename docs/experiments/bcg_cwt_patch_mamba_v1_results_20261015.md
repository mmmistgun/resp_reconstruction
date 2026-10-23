# BCG–CWT patch Mamba v1 训练与 research-test 结果

截至 2026-10-15，seed=20261013 的 run02 train/validation 与两个固定 checkpoint 的 research-test 均已完成并关闭。主结果采用 validation Local RR 最优的 epoch 6；epoch 3 为 validation loss 最优的辅助结果。执行定义见[训练协议](bcg_cwt_patch_mamba_v1_protocol_20261014.md)与[评价协议](bcg_cwt_patch_mamba_research_test_v1_20261015.md)。

本轮主模型的 test Local RR MAE 为 **0.780100 bpm**，包络轨迹 MAE 为 **0.150562**，lag-aware signed PCC 为 **0.868536**。相对历史 CWT-APOR A0、H-only、原始 M4 和 NoTF 的三 seed 均值，这三项指标均较差；当前结果不支持将该配置作为性能升级。该比较为单 seed 对历史三 seed 均值的描述性比较，架构变化的具体贡献尚未分离。

## 训练与选点

模型参数量为 1,728,572。输入为 100 Hz、180 秒 BCG 与完整 97×360 CWT，连续 CNN 编码、片段内交叉注意力、六层双向 Mamba2 和共享输出头形成 180 个一秒片段。任务损失、预处理、split 和主指标定义沿用既有实现。

训练包含 10,141 窗口/32 人，validation 包含 2,675 窗口/7 人；微批量 64、累积 2 次、有效 batch 128。最多 80 epoch，学习率按 6,400 次更新规划。本轮在 epoch 30 触发早停，完成 2,400 次更新。

| 选点角色 | Epoch | Validation Local RR MAE (bpm) | Validation loss |
| --- | ---: | ---: | ---: |
| 主结果：best_rr | 6 | 0.554725 | 0.257840 |
| 辅助结果：best_loss | 3 | 0.571687 | 0.254048 |

两个 checkpoint 均在 test 访问前由完整 validation 轨迹确定，严格最小值并列时选更早 epoch。主结果身份保持为 epoch 6。

## Research-test 指标

完整 test 为 2,310 窗口/8 人，sample seed=20260612。该人群已用于项目研究开发，本轮属于 reused research/development evidence。以下为窗口均值；主指标有效分母均为 2,310。

| 指标 | best_rr：epoch 6 | best_loss：epoch 3 |
| --- | ---: | ---: |
| Whole RR MAE ↓ (bpm) | 0.755879 | 0.778519 |
| Local RR MAE ↓ (bpm) | 0.780100 | 0.845953 |
| Envelope trajectory MAE ↓ | 0.150562 | 0.141306 |
| Global envelope modulation error ↓ | 0.166674 | 0.160774 |
| Lag-aware signed PCC ↑ | 0.868536 | 0.872846 |
| IBI MedAE ↓ (s) | 0.108348 | 0.106217 |
| IBI coverage ↑ | 0.788853 | 0.789946 |
| Respiratory-band coherence ↑ | 0.522004 | 0.543699 |
| Constrained nDTW ↓ | 0.215953 | 0.214774 |
| Low envelope Spearman ↑ | 0.353784 | 0.415379 |
| Medium envelope Spearman ↑ | 0.513814 | 0.534571 |
| High envelope Spearman ↑ | 0.722483 | 0.744270 |

IBI MedAE 的可解释窗口分母分别为 1,447 和 1,418，不能把其差值视为相同窗口集合上的配对效应；coverage 分母为全部 2,310 窗口。Low/Medium/High 分层分母分别为 791/957/562。Local RR prediction-valid fraction 均为 1，joint 与各包络分层的 prediction-degenerate fraction 均为 0。

两个模型的绝对 lag 中位数分别为 0.12/0.14 秒，P95 均为 0.30 秒，搜索边界比例分别为 0.149351/0.183550。PCC 按允许 ±0.30 秒校正的既有定义解释。

epoch 6 的两项 RR 误差更低，epoch 3 的包络与波形形态指标更好。主模型 Local RR 从 validation 的 0.554725 增至 test 的 0.780100 bpm。

## 历史模型参照

以下均采用完整相同 test 人群、窗口平均与 validation Local RR 选点。历史值为三 seed 均值，新模型为单 seed；训练实现和模型结构存在差异，因此该表用于描述表现，不构成单组件的配对消融。

| 模型 | Local RR MAE ↓ (bpm) | Envelope MAE ↓ | PCC ↑ |
| --- | ---: | ---: | ---: |
| CWT-APOR H-only | 0.60766 | 0.14339 | 0.88159 |
| CWT-APOR A0 | 0.65015 | 0.14222 | 0.88182 |
| 原始 M4 | 0.640376 | 0.145531 | 0.871763 |
| M4 NoTF | 0.622098 | 0.142161 | 0.873386 |
| BCG–CWT patch Mamba v1 | 0.780100 | 0.150562 | 0.868536 |

历史来源为 [CWT-APOR v2 汇总](cwt_apor_v2_results_summary_20261002.md)和 [M4 research-test 汇总](m4_components_research_test_results_20261009.md)。主模型 Local RR 相对 H-only 增加约 28.4%，相对原始 M4 增加约 21.8%。其 global envelope error 略低于原始 M4（0.166674 对 0.167532），因此整体落后不等同于每项指标均变差。前端、融合、时间几何与解码同时变化，现有结果尚不能确定退步来自哪个部分。

## 来源、验证与完成边界

以下路径相对于项目实体目录 `/mnt/disk_code/marques/resp_reconstruction`：

- 训练：`runs/bcg_cwt_patch_mamba_v1/seed_20261013_run02/`；训练源码提交 `0fe6e47f87d892c70ca0b93187b56cb1164a00b0`。
- 评价：`runs/bcg_cwt_patch_mamba_v1/research_test/seed_20261013_run02_v1_20261015/`；`evaluation_manifest.json`、源码快照和协议副本记录实际执行身份。
- 汇总：评价目录中的 `comparison.csv`，SHA-256 为 `eff02a8071a8f50c9812a3fac220f7d8e50e38e5646cb055e4cd1b0aa1641cfd`。
- 主 checkpoint：`epoch_0006/checkpoint.pt`，SHA-256 为 `65604a69734292e2dab5cf92faeb7727a92ab3f130aed5a41866205dc1be922c`。
- 辅助 checkpoint：`epoch_0003/checkpoint.pt`（与 `best_loss.pt` 相同），SHA-256 为 `061c635c2f82d7bcd0b39d75dc9082dfcb97d4a5cfe69282258fcdb79fb354da`。

评价进程以退出码 0 结束，完成回执覆盖两项评价；整理时核对了 comparison、两个 receipt 及其引用的 predictions/metrics/summary 文件哈希。当前评价脚本与运行源码快照一致。评价入口的 4 项 synthetic CPU 定向测试已通过，覆盖授权门禁、完整轨迹选点、频率身份、完整指标及行顺序检查。

本记录整理已保存结果。该运行的训练与固定 checkpoint 评价已结束；新增实验应另设实验身份与执行授权。源码、协议、测试和本结果记录进入 Git，预测、checkpoint、日志及逐窗口产物保留在运行目录。
