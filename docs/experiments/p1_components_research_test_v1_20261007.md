# P1 组件消融固定 checkpoint research-test 协议

协议 ID：`p1-components-research-test-v1-20261007`。当前范围为实现与来源准备，正式 research-test 由用户执行带 `--confirm-research-test` 的入口。完整 30-cell train/validation 已关闭，结果与来源见 [validation 结果记录](p1_components_v1_validation_results_20261007.md)。

## 固定矩阵与选点

固定评价 M0–M6、S1–S3 × seeds 20260811/20260812/20260813，共 30 个 checkpoint。模型定义、共有初始化及 GroupNorm representation-variant 合同沿用 [训练协议](p1_components_v1_protocol_20261006.md)。每组只取完整 validation Local RR MAE 严格最小、并列最早的 epoch。

| Arm | 20260811 | 20260812 | 20260813 |
|---|---:|---:|---:|
| M0 | 5 | 5 | 8 |
| M1 | 5 | 11 | 8 |
| M2 | 4 | 5 | 8 |
| M3 | 5 | 5 | 8 |
| M4 | 25 | 5 | 8 |
| M5 | 4 | 5 | 8 |
| M6 | 5 | 4 | 6 |
| S1 | 4 | 5 | 8 |
| S2 | 4 | 19 | 8 |
| S3 | 14 | 6 | 9 |

训练 session 为 `runs/p1_components_v1/formal_v1_20261006`，`session.json` SHA-256 为 `34d0c62cb64adae16fca4db512687d28bb96e74e89857c049a506d66c4e71fb6`；validation summary receipt SHA-256 为 `de7ee8375fc29a0ef2958893750c34dd51b0c1e3d5f5204e8cbc65d795f0fdc1`。准备时核对全部 completion、产物哈希、完整 validation history、选点、模型 state、配置及 development 行身份。

该 test 已用于既有研究开发，证据角色为 **reused research/development evidence**。本轮用于固定矩阵跨 split 复核。不得根据中间 test 结果改变剩余矩阵、checkpoint、阈值、指标或模型结构；固定模型的频带替换、时间位移和 attention 分布研究由独立机制协议定义。

## 数据与推理

固定 test=2310 窗口、8 名受试者，180 秒/100 Hz，sample seed=20260612。有序 row ID 字节 SHA-256 为 `184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。dataset index 采用训练来源记录的路径与 SHA-256；运行时校验 test 与 train/validation 受试者集合零交集。

条件臂复用冻结 test W cache：identity=`40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`，manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。原生 reader 校验哈希、shape、dtype 与 row identity；完整 97-scale 频率与 W-full checkpoint 逐值一致，H 按 `(0.8,8] Hz` 选择 41 scales，L 按 `≤0.8 Hz` 选择 56 scales，并与对应 checkpoint 逐值核对。M1 NoTF 直接读取波形与目标，不构造 CWT reader 或条件模块。

统一 batch=32、BF16、eval 模式、尾批完整保留。正式模型 strict 加载，参数量与训练 completion 一致。评价配置副本只调整设备及进度显示。逐 batch 核对 row 顺序、浮点类型、波形/目标 `(B,1,18000)` 与各臂条件 `(B,F,360)`，NoTF 要求无条件张量。非有限 input、target、prediction、checkpoint 或关键指标显式失败；prediction degeneracy 单独记录并保留分母。训练侧 reader 继续仅接受 train/validation。

W-full/H/L 使用各自支持域上的 GroupNorm。频带比较及下述交互均按 representation variant 解释，支持域同时影响归一化统计与边界邻域。

## 指标与完整汇总

五项主指标沿用 validation：Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。原生 test 辅助指标包括 IBI/coverage、coherence、constrained nDTW、分层 envelope Spearman；保留资格、分母与退化标记。

全部 30-cell 完成后生成逐 seed、三 seed mean/sample SD、逐受试者与 subject-macro、Local RR tail、资格分母、参数量及原生辅助汇总。配对比较沿用训练协议：M1–M6、S1、S3 对 M0；S2 对 M2。`delta=对照臂−参照臂`，误差下降和 PCC 上升记正 improvement。

W-full/H × attention/mean 交互采用同 seed 的 `attention_gain_W_full − attention_gain_H`：误差 gain 为 mean 减 attention，PCC gain 为 attention 减 mean。对应 M0/M3 与 M2/S2；正交互表示 W-full 下 attention 改善更大。跨 cell 的样本顺序及 target 资格须一致；未完成矩阵不生成最终汇总。

## 生命周期与执行

实现位于 `scripts/eval_p1_components.py`；冻结训练模型、代码及原协议保持原身份。`prepare` 只读取 development 产物和源码，不读取 test index、manifest、波形、目标或 cache 数组。allowlist 固定 30 个 checkpoint、配置、环境、实际频率及本 test 合同，并保存源码快照。

成功准备及评价产物通过哈希校验后复用。每个 cell 使用互斥锁；失败 attempt 保留，检查失败原因后使用 `--retry-failed` 创建独立重试。两张 GPU 各自串行执行 15 个 cell，失败即停止当前分片。test 访问前必须显式确认，确认不外推至其他实验。

在 worktree 准备一次：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_p1_components.py prepare --training-session runs/p1_components_v1/formal_v1_20261006 --output runs/p1_components_v1/research_test_v1_20261007
```

两个终端分别执行（每个终端先进入上述目录）：

```bash
# GPU 0
bash scripts/run_p1_components_test_shard.sh runs/p1_components_v1/research_test_v1_20261007 0 0
```

```bash
# GPU 1
bash scripts/run_p1_components_test_shard.sh runs/p1_components_v1/research_test_v1_20261007 1 1
```

脚本包含 `env -u LD_LIBRARY_PATH -u LD_PRELOAD` 和 `--confirm-research-test`，执行脚本即确认该固定分片的 test 访问；失败重试在末尾加 `--retry-failed`。也可使用 CLI `evaluate --allowlist ... --shard-index 0 --shard-count 2 --device cuda:0 --confirm-research-test`。

状态与最终汇总：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_p1_components.py status --allowlist runs/p1_components_v1/research_test_v1_20261007
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_p1_components.py summarize --allowlist runs/p1_components_v1/research_test_v1_20261007 --confirm-research-test
```

验收标准：30 个 evaluation 成功 receipt，全部评价各 2310 窗口/8 受试者；checkpoint、代码与环境来源一致；完整汇总通过样本/target 资格一致性及指标有限性检查。退化数量按实际结果报告，不据此静默过滤样本。

## 实现验证

2026-10-07：38 项 CPU 定向测试通过，覆盖确认门禁、W-full/H/L 频带读取及频率漂移拒绝、NoTF 条件排除、batch 身份/shape/finite/完整性、原生 test 辅助指标、完整 30-cell 配对及交互汇总、最早并列选点、失败保留/重试/成功复用、allowlist 与产物身份漂移拒绝。shell 入口通过 `bash -n`。

CPU 定向测试仅使用 synthetic/disposable fixtures；正式 GPU 推理与真实 test 数据访问由用户执行。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q tests/test_p1_components_research_test.py
bash -n scripts/run_p1_components_test_shard.sh
```
