# M4 池化评分/值位置路径固定 checkpoint research-test

协议 ID：`m4-pool-position-paths-research-test-v1-20261017`。

2026-10-17 用户已授权 ABCD 全部十二个固定最佳 checkpoint 在空闲 GPU 后台评价，完成后自动汇总评分路径、值路径和交互效应。训练与 validation 已关闭，来源见 [validation 结果](m4_pool_position_paths_v1_validation_results_20261016.md)。本协议独立开放本轮 research-test。

## 固定范围

| Arm | 评分位置 | 值位置 | seed 20260811 epoch | 20260812 | 20260813 |
|---|---|---|---:|---:|---:|
| A | 开 | 开 | 25 | 5 | 8 |
| B | 关 | 关 | 4 | 5 | 8 |
| C | 开 | 关 | 4 | 5 | 3 |
| D | 关 | 开 | 4 | 5 | 8 |

沿用完整 validation Local RR 严格最小、并列最早的 checkpoint。A/B 来源为 `runs/m4_pool_position_v1/formal_r3_20261015`，C/D 来源为 `runs/m4_pool_position_paths_v1/formal_v1_20261016`。第二阶段 session SHA-256 为 `cb8a536afe0162891b947c0a645f406eefc44105a2e56e1e314a589aba8dd3d1`，完整 validation summary receipt 为 `b518b8df71041984acd7231f7b5fac7dd1b5e84c2ac595159377a00451b82045`。A/B 六项也在本轮按各自冻结来源评价。

prepare 在任何 test 数据访问前核验完整矩阵：训练源码/来源 session、全部 completion 产物哈希、配置、完整 history 选点、best checkpoint 有限性与频率、development 行身份和受试者隔离。保存十二项 checkpoint/config/completion SHA、训练环境、原配置、源码快照及不可覆盖 allowlist。原训练与 validation 产物保持冻结。

## 数据、推理与证据口径

test 使用既有冻结 2310 窗口/8 人、180 秒/100 Hz、sample seed=20260612；有序 row ID SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。dataset index 身份继承 development 来源；评价时核验与 train/validation 的受试者隔离。

全部臂使用冻结 W-full 97-scale test CWT，cache identity=`40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`，manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。频率逐值匹配 checkpoint，保留既有 GroupNorm、模型结构、数据和信号处理口径。

batch=32、BF16、eval 模式、保留尾批，软件版本和 GPU 型号与训练一致。直接构造冻结 PoolPositionModel，strict 加载并核对参数量。A/B 读取第一阶段原配置，C/D 读取第二阶段原配置；评价副本仅改变设备和进度显示。逐 batch 核验顺序、输入与 target `(B,1,18000)`、条件 `(B,97,360)`；非有限输入、target、checkpoint、prediction 或关键指标显式失败，预测退化保留原分母。

该 test 已参与研究开发，证据角色为 **reused research/development evidence**。本轮用于固定矩阵的跨 split 复核。三 seed 共用同一 test 人群；不将 seed 当作独立受试者队列，不依据部分结果更改模型、指标、选点、候选或阈值。

## 汇总

完整十二项成功后才汇总。保留五项主指标（Local RR MAE、Whole RR AE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC），以及原生 test 辅助指标、资格、退化、Local RR tail 和参数量。跨 cell 样本顺序与 target 资格必须相同。

窗口加权和受试者宏平均均输出逐 seed、三 seed mean/sample SD 与方向数；逐受试者另存。固定效应为：评分条件效应 C−B、A−D；值条件效应 D−B、A−C；评分主效应 `[(C−B)+(A−D)]/2`；值主效应 `[(D−B)+(A−C)]/2`；交互 `A−C−D+B`。delta 保留原方向，误差下降/PCC 上升为正 improvement；交互的正负表示偏离可加性，须结合指标方向及条件效应解读。

自动保存 CSV、两种聚合的 PNG 交互图、中文 `report.md` 及全来源 receipt。图中误差棒为三 seed 样本 SD。

## 执行与验收

所有命令在仓库 `/mnt/disk_code/marques/resp_reconstruction` 执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_m4_pool_position_paths_research_test.py
bash -n scripts/run_m4_pool_position_paths_test_shard.sh
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/eval_m4_pool_position_paths.py prepare --training-session runs/m4_pool_position_paths_v1/formal_v1_20261016 --output runs/m4_pool_position_paths_v1/research_test_v1_20261017
bash scripts/run_m4_pool_position_paths_test_shard.sh runs/m4_pool_position_paths_v1/research_test_v1_20261017 1
```

后台 shell 顺序执行全部评价及汇总，失败立即停下并保留 attempt/traceback；成功 receipt 经哈希验证后复用。入口提供 `status`，评价与汇总必须显式 `--confirm-research-test`；失败修复后仅以 `--retry-failed` 创建独立 attempt。验收要求十二项成功 receipt、每项固定 2310 窗口/8 人、完整效应汇总和来源校验。运行进度与最终结果由产物 receipt 和后续结果文档记录。
