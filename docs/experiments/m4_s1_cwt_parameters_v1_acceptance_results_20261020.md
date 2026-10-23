# M4-S1 CWT 参数与模块消融：数据、缓存及 GPU 验收

日期：2026-10-20。设计 ID：`m4-s1-cwt-parameters-v1-20261020`。

状态：**正式 train/validation 数据核验、20份历史缓存兼容性核验、28臂 GPU 合成验收全部通过，阶段关闭。84个正式训练 cell 均未启动。** 用户本次授权数据与缓存核验及 GPU 合成验收，明确暂不启动正式训练；research-test 未开放。

科学合同和操作入口见[设计协议](m4_s1_cwt_parameters_v1_design_20261020.md)。该协议、源码与配置已随 session 冻结；本文件记录后续阶段状态，不修改冻结内容。

## 1. 数据与缓存

| 项目 | 验收结果 |
|---|---|
| train | 10,141窗，32名受试者 |
| validation | 2,675窗，7名受试者 |
| 受试者隔离 | train/validation 无交集 |
| 数据来源 | 78份源文件，7.9251 GiB；准备前后哈希一致 |
| validation 参考目标 | 全部2,675窗生成公共 float32 参考，shape、row身份和有限性检查通过 |
| CWT 数学校准 | 20种合成表示通过；B与原变换最大绝对差0；频带子集与时间池化关系通过 |
| 缓存兼容性 | 20/20份通过，全部只读引用历史 W0 时频缓存 |
| 缓存数组体积 | 34,612,432,640 bytes（32.2353 GiB），仅计40个train/val `_w.npy` |

正式数据核验沿用冻结index、样本资格、顺序、输入和target口径。缓存逐份检查旧session receipt、源码、依赖、表示参数、数据源哈希、row集合与顺序、完成manifest以及全部缓存字节。未重建缓存。每份本轮 `cache_record.json` 保存源session与cache manifest身份。

预处理源码只有 `resp_train/data/research_v2.py` 存在已审阅差异：新增Patch-aligned TF-Mamba专用H-cache分支。当前M4-S1原始数据配置不会进入该分支；旧/新精确SHA对、配置条件及旧源码快照均受检查，其他漂移继续拒绝。相关定向CPU验证为 **2 passed，58.82秒**，覆盖临时样本上新旧输入/target/meta等价、错误配置拒绝及历史cache只读引用。这一兼容处理不改变波形切片、split、target或CWT数学定义。

缓存来源：

`/mnt/disk_code/marques/resp_reconstruction/runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/cwt-time-frequency-v1/runs/cwt_time_frequency_v1/session_20260930T184031Z_812b424a1c92`

其 `session.json` SHA256：`f750d5c03cd19cd15d5c54b0260d649be046691e4f23110204381ed7c7395dca`。

## 2. GPU 合成验收

在物理GPU0单卡顺序执行全部28臂。设备为 **NVIDIA GeForce RTX 4070 Ti SUPER**；PyTorch `2.12.0+cu130`、CUDA `13.0`、`mamba-ssm 2.3.2.post1`、`causal-conv1d 1.6.2.post1`。执行时清除 `LD_LIBRARY_PATH` 与 `LD_PRELOAD`。含Mamba的臂使用官方Mamba2 CUDA实现，NoMamba按合同使用Identity。

| 检查 | 结果 |
|---|---|
| 覆盖 | 28/28臂，失败0 |
| 输入与精度 | 100 Hz、180秒；微批32，梯度累积4，BF16 |
| 更新 | 每臂2次，共56次合成optimizer更新 |
| 有限性 | task loss、validation loss、模型/optimizer状态及全部活动参数梯度通过 |
| 有效学习路径 | 两次更新后受检条件上游或时域参数梯度非零、参数发生更新 |
| B与原生L1 | 三seed × FP32/BF16共6项，非零条件输出下前向与梯度通过；前向最大绝对差均0 |
| 最大 allocated | 7,876,534,784 bytes（7.3356 GiB），出现在 `Q_mu6_v24` |
| 最大 reserved | 11,037,310,976 bytes（10.2793 GiB），占PyTorch报告容量66.03%，低于95%门槛 |
| 收尾 | GPU进程已退出，两张卡均恢复空闲 |

reserved包含同进程顺序验收时分配器保留的缓存，因此不能将各臂reserved直接解释为该臂独立进程的最低显存需求。逐臂allocated、reserved、参数量、耗时与更新数保存在 `acceptance_report/gpu_acceptance.csv`。耗时包含初始内核开销，仅用于本次工程记录，不作为正式训练吞吐benchmark。

本阶段验证的是执行兼容性、梯度和显存可行性；尚无本轮正式训练结果或重建性能结论。数据、loss、指标和checkpoint选择合同保持原定义。

## 3. 产物与冻结来源

本轮session：

`/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/acceptance_v1_20261020`

| 文件（相对session） | SHA256 |
|---|---|
| `session.json` | `87b201abb30dea105ca8c9ada5c2f48a33e1f5ebd47b78af11276f67e93740cc` |
| `data/attempt_20261020T143426Z_7f2afe488b73/manifest.json` | `e1b8e70e794d2cc321860b39f960ad4c7fb5559a06fa9732ec4c6d6a06b79931` |
| `acceptance_report/manifest.json` | `0fca697f72af06ddffb61aa19adfe512edb77c1b847af897b2be47d29c3a3380` |

`acceptance_report/summary.json` 汇集50个成功阶段的manifest身份（1校准、1数据、20缓存、28GPU）、缓存引用、环境、B等价结果和执行边界。收尾已复核这些阶段的receipt及其产物、session源码快照与84份resolved config；不存在失败回执或 `cells/` 训练目录。

执行日志及收尾脚本位于：

`/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/execution_logs_20261020/`

依次为 `01_prepare.log`、`02_calibrate.log`、`03_prepare_data.log`、`04_import_cache.log`、`05_gpu_smoke.log`、`06_collect_acceptance.py` 与 `06_collect_acceptance.log`。各阶段回执另存实际命令与时间；收尾报告使用排他创建。

完整84-cell train/validation矩阵保留待后续授权。正式训练及research-test均未执行，已完成的本阶段产物只读复用。
