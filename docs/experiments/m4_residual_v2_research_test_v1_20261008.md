# M4-v2 固定 checkpoint research-test 协议

协议 ID：`m4-residual-v2-research-test-v1-20261008`。当前范围为实现与来源准备；正式 GPU research-test 由用户执行显式确认入口。train/validation 已完成并关闭，结果及训练协议入口见 [validation 结果](m4_residual_v2_validation_results_20261008.md)。

## 固定范围与来源

评价 A0/A1/A2/A3/S1/S2 × seeds 20260811/20260812/20260813，共 18 个固定 checkpoint。每组只取完整 validation Local RR MAE 严格最小、并列最早的 epoch。

| Arm | 20260811 | 20260812 | 20260813 |
|---|---:|---:|---:|
| A0 M4-v2 | 4 | 6 | 4 |
| A1 NoTF | 9 | 6 | 10 |
| A2 MeanTF | 4 | 6 | 4 |
| A3 NoMamba | 22 | 5 | 4 |
| S1 Query RMSNorm | 4 | 6 | 4 |
| S2 ContentAttention | 14 | 6 | 4 |

训练 session：`runs/m4_residual_v2/formal_v1_20261007`；`session.json` SHA-256=`eeb7d56625a3a8a15a9d83745e30204de936ca939d8f280ee2df371aba3ecb5d`。validation summary receipt SHA-256=`2454699771d45d779dbd4f4582796a374e744771e0e339884f831321de810eb1`。准备 allowlist 时重新核对 completion、全部产物哈希、完整 validation 选点轨迹、checkpoint 配置与有限性、频率坐标及 development 行身份。

该 test 已用于研究开发，并影响本轮设计，证据角色为 **reused research/development evidence**。本轮用于固定矩阵跨 split 复核。不得根据 partial test 修改剩余候选、checkpoint、指标、阈值或模型结构。

## 数据与推理

固定 test=2310 窗口/8 人、180 秒/100 Hz、sample seed=20260612。有序 row ID 字节 SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。dataset index 使用 development 来源记录的路径与 SHA；检查 test 与 train/validation 受试者集合零交集。

条件臂全部使用冻结 97-scale W test cache：identity=`40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`，manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。实际频率与 allowlist/checkpoint 逐值一致；原生 reader 检查 cache 哈希、shape、dtype 和 row identity。A1 NoTF 直接读取波形/target，不构造条件模块或打开 CWT reader。

统一 batch=32、BF16、eval 模式，完整保留尾批。模型 strict 加载并核对参数量；只在评价配置副本中调整设备与进度显示。逐 batch 核对行顺序、浮点类型、波形/目标 `(B,1,18000)`，条件臂要求 `(B,97,360)`。非有限 input、target、prediction、checkpoint 或关键指标显式失败；预测退化另行记录并保持分母。训练 reader 继续限定 train/validation。

## 指标与预设比较

五项主指标沿用 validation：Whole RR AE、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。开启原生 test 辅助指标：IBI/coverage、respiratory-band coherence、constrained nDTW、target 分层 envelope Spearman；保留各自资格、分母及退化标记。

全部 18-cell 后汇总逐 seed、三 seed mean/sample SD、逐受试者与 subject-macro、Local RR tail、资格分母、参数量和原生辅助汇总。主配对 A1/A2/A3/S1/S2 各对 A0；递增链为 A2−A1（TF information）、S2−A2（content attention）、A0−S2（physical coordinates）。同 seed 配对，`delta=候选−参照`，误差下降/PCC 上升为正 improvement。跨 cell 样本顺序及 target 资格必须一致。

综合阅读所有指标与聚合口径，不用单一最小值替代结构判断。S2 保留相同物理窗口，坐标对照只检验显式 bias；A3 仍含 CWT 支持及整窗 GroupNorm，检验显式 BiMamba 增量。以上解释边界与训练协议一致。

## 生命周期与命令

实现位于 `scripts/eval_m4_residual.py`。`prepare` 只读取冻结 development 产物和源码，不访问 test index、manifest 或数据数组；allowlist 固定 18 个 checkpoint、配置、环境、频率、比较合同及源码快照。成功准备/评价经哈希校验后复用。每个 cell 使用互斥锁；失败 attempt 保留，检查原因后以 `--retry-failed` 创建独立重试。两张 GPU 每卡串行 9 组，失败即停止当前分片，完整矩阵完成后才生成汇总。

准备一次：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_m4_residual.py prepare --training-session runs/m4_residual_v2/formal_v1_20261007 --output runs/m4_residual_v2/research_test_v1_20261008
```

两个终端分别执行（各终端先进入上述目录）：

```bash
# GPU 0
bash scripts/run_m4_residual_test_shard.sh runs/m4_residual_v2/research_test_v1_20261008 0 0
```

```bash
# GPU 1
bash scripts/run_m4_residual_test_shard.sh runs/m4_residual_v2/research_test_v1_20261008 1 1
```

脚本包含 `env -u LD_LIBRARY_PATH -u LD_PRELOAD` 和 `--confirm-research-test`，执行即确认该分片的固定 test 访问。手工入口为 `evaluate --allowlist ... --shard-index 0 --shard-count 2 --device cuda:0 --confirm-research-test`。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_m4_residual.py status --allowlist runs/m4_residual_v2/research_test_v1_20261008
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_m4_residual.py summarize --allowlist runs/m4_residual_v2/research_test_v1_20261008 --confirm-research-test
```

验收标准：18 个成功评价 receipt；每组 2310 窗口/8 人，checkpoint、代码及环境身份一致；完整汇总通过样本与资格一致性、关键指标有限性检查。退化计数按实际报告。

## 实现验证

2026-10-08：38 项合成 CPU 定向测试通过，覆盖确认门禁、完整 W 频率及 checkpoint 身份、NoTF 条件排除、batch 身份/shape/finite/完整性、原生 test 辅助指标、完整 18-cell 主配对及递增比较、最早并列选点、失败保留/重试/成功复用、allowlist 与产物身份漂移拒绝。shell 入口通过 `bash -n`。

CPU 测试只使用 synthetic/disposable fixtures；正式 GPU 推理及真实 test 访问由用户执行。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q tests/test_m4_residual_research_test.py
bash -n scripts/run_m4_residual_test_shard.sh
```
