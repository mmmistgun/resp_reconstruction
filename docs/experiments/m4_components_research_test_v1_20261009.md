# 原始 P1-M4 固定 checkpoint research-test 协议

协议 ID：`m4-components-research-test-v1-20261009`。当前范围为实现、合成 CPU 验证及来源准备；正式 GPU research-test 需要当次执行授权或由用户运行显式确认入口。train/validation 已完成并关闭，来源与结果见 [validation 结果](m4_components_v1_validation_results_20261009.md)，原训练合同见 [train/validation 协议](m4_components_v1_protocol_20261008.md)。

## 固定范围与选点

完整 B0–B7 × seeds 20260811/20260812/20260813，共 24 个固定 checkpoint 身份。B2–B7 新增 18 项评价；B0/B1 只读引用原 P1 M4/M1 的六项已完成 test 结果，不重新评价。每个 checkpoint 仍取完整 validation Local RR 严格最小、并列最早的 epoch，评价候选覆盖完整矩阵，不根据 validation 排名删减。

| Arm | 20260811 | 20260812 | 20260813 | 本轮执行 |
|---|---:|---:|---:|---|
| B0 原始 M4 | 25 | 5 | 8 | 引用 P1 M4 |
| B1 NoTF | 5 | 11 | 8 | 引用 P1 M1 |
| B2 MeanTF | 5 | 5 | 8 | 新评价 |
| B3 ContentAttention | 5 | 5 | 8 | 新评价 |
| B4 H | 4 | 5 | 8 | 新评价 |
| B5 L | 4 | 5 | 8 | 新评价 |
| B6 NoMamba | 4 | 4 | 6 | 新评价 |
| B7 UniformPatchPool | 5 | 6 | 8 | 新评价 |

训练 session：`runs/m4_components_v1/formal_v1_20261008/`。

- session.json SHA-256：`9f5f43120f2664f4b2eb214e73890e431c18320944c0f88aa70ad4c10b64f833`。
- validation summary receipt SHA-256：`7f0dc351f5d33c60e90d8d07f53442597ab2584c9658bd731677ae0bb994f926`。
- 历史 test 根目录：`runs/p1_components_v1/research_test_v1_20261007/`。
- 历史 test allowlist SHA-256：`8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47`。
- 历史 test summary receipt SHA-256：`bae9f68d1834483dc9cd1b734f01d98ea5826186b9f1c41d8c423af7d962d29e`。

准备阶段核验全部 development completion、必需产物哈希、完整 checkpoint history 选点、checkpoint 配置及有限性、频率坐标、train/validation 行身份与受试者隔离。B0/B1 必须与历史 test 中的 checkpoint/config/completion 路径和 SHA、选点、参数量一致；历史推理合同、环境和成功 receipt 必须匹配。历史来源不符时显式失败，不自动重评或替换参照。

该 test 已参与研究开发并影响本轮设计，证据角色为 **reused research/development evidence**；用于固定完整矩阵的跨 split 复核。不得根据 partial test 改变候选、选点、指标、阈值或模型结构。

## 数据与推理合同

test 固定为 2310 窗口/8 人、180 秒/100 Hz、sample seed=20260612。有序 row ID 字节 SHA-256：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。dataset index 沿用 development 来源的路径与 SHA；运行时检查 test 与 train/validation 受试者集合无交集。

条件臂使用既有冻结 97-scale W test cache：identity=`40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`，manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。Full 使用全部 97 scales，B4 使用 (0.8,8] Hz 的 41 scales，B5 使用 ≤0.8 Hz 的 56 scales。实际频率与 allowlist/checkpoint 逐值一致；频带裁剪前检查完整源 CWT 的非有限值。沿用原 GroupNorm，频带对照包含统计域和卷积边界变化。

统一 batch=32、BF16、eval 模式并完整保留尾批。新模型按原始 M4 双投影结构 strict 加载，核对参数量和选点；仅在评价配置副本调整设备与进度显示。推理软件版本及 GPU 型号与训练一致。逐 batch 检查样本顺序、浮点类型、波形/target `(B,1,18000)`，条件 `(B,F,360)`；非有限输入、目标、checkpoint、预测或关键指标显式失败，预测退化标记保持原分母。

NoTF 的历史结果原本不读取 CWT；新入口禁止对 B0/B1 再次推理。训练 reader 和训练入口继续限定 train/validation。`prepare` 读取已保存的 development 产物及历史 test provenance/产物哈希，不读取 test index、cache manifest、原始数组，也不解析历史 test 指标值或执行模型推理。

## 完整指标与比较

五项主指标：Whole RR AE、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。保留原生 test 辅助指标：IBI MedAE/coverage、respiratory-band coherence、constrained nDTW、目标分层 envelope Spearman、最优 lag 诊断以及各指标资格/退化分母。

全部 18 项新评价成功并核验六项历史参照后，才生成完整 24-cell 汇总：逐 seed、三 seed mean/sample SD、主配对与递增比较、配对均值/SD/seed 方向数、逐受试者及 subject-macro、Local RR tail、参数量、原生辅助汇总和来源表。读取历史 metrics 后只在内存把 M4/M1 映射为 B0/B1，历史文件保持原身份。

主配对为 B1–B7 各对 B0；递增比较为 B2−B1（均匀 TF）、B3−B2（内容读取及其 Q/K/V 参数化）、B0−B3（显式坐标 bias）。delta=候选−参照，误差下降/PCC 上升为正 improvement。跨 cell 必须保持相同样本顺序与 target 资格。

综合五指标、受试者等权和尾部判断，保留 validation 中已观察到的聚合权重差异。B3 保留物理对齐读取；B6 保留 CWT 时间支持、整窗 GroupNorm 及连续 CNN 邻域。三 seed 共用同一 test 人群，不视为三个独立受试者队列。

## 生命周期与命令

入口：`scripts/eval_m4_components.py`，提供 `prepare / evaluate / status / summarize`。固定 allowlist 保存 24 个 checkpoint 身份、18 项新评价角色、六项历史参照、配置、development 受试者、频率、比较合同和源码快照。每个新 cell 互斥运行；失败 attempt 原地保留，核对原因后以 `--retry-failed` 创建独立重试；成功产物经哈希校验后复用。

准备一次（不执行 test 推理）：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_m4_components.py prepare --training-session runs/m4_components_v1/formal_v1_20261008 --output runs/m4_components_v1/research_test_v1_20261009
```

用户确认执行后，两个终端分别运行，每卡串行 9 个新 checkpoint：

```bash
bash scripts/run_m4_components_test_shard.sh runs/m4_components_v1/research_test_v1_20261009 0 0
```

```bash
bash scripts/run_m4_components_test_shard.sh runs/m4_components_v1/research_test_v1_20261009 1 1
```

shell 入口含 `--confirm-research-test`，用户执行即确认相应固定分片访问；Codex 代跑仍需当次执行授权。手工入口支持 `evaluate --allowlist ... --shard-index 0 --shard-count 2 --device cuda:0 --confirm-research-test`。缺少确认时，评价和汇总在读取 allowlist 前失败。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_m4_components.py status --allowlist runs/m4_components_v1/research_test_v1_20261009
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_m4_components.py summarize --allowlist runs/m4_components_v1/research_test_v1_20261009 --confirm-research-test
```

验收：18 个成功新评价 receipt + 六个持续通过身份校验的历史参照；每个 cell 为 2310 窗口/8 人，checkpoint/环境/代码固定；完整主指标、辅助指标、配对和来源汇总通过资格及有限性校验；不重选 checkpoint、不重训模型或改写历史结果。

## 合成验证

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_m4_components_research_test.py
bash -n scripts/run_m4_components_test_shard.sh
```

测试仅使用 synthetic/disposable fixtures，覆盖确认先于来源访问、Full/H/L 频率与非有限检查、NoTF 条件排除、历史参照禁止重评、原生 test 辅助指标、完整 24-cell 配对/递增汇总、参照只读复用、最早并列选点、失败保留/重试、allowlist/历史来源漂移拒绝及 prepare 不读取 test 数据。

2026-10-09：54 项合成 CPU 定向测试通过，shell 与 Python 语法检查通过。六个历史 test 参照的只读 provenance 校验通过，checkpoint/config/completion、推理合同、环境和结果 receipt 均匹配；未解析历史 test 指标值、访问 test 原始数据或执行模型推理。
