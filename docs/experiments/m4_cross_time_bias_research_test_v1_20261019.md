# M4 交叉注意力 Δt 时间 bias 固定 checkpoint research-test

协议 ID：`m4-cross-time-bias-research-test-v1-20261019`。

2026-10-19 用户授权三个 NoCrossTimeBias 固定最佳 checkpoint 在空闲 GPU 后台进行 test 评价，完成后与原始 M4 对比，并结合 validation 按 0.01 阈值判断时间 bias 的保留价值。[Train/validation 已完成并关闭](m4_cross_time_bias_v1_validation_results_20261019.md)，本协议独立开放本轮 research-test。

## 固定矩阵与来源

| Arm | 含义 | seeds | 最佳 epoch（11/12/13） | test 来源 |
|---|---|---|---|---|
| A0 | 原始 M4，频率与时间 bias 均保留 | 20260811/20260812/20260813 | 25/5/8 | 只读复用历史 P1-M4 三项 test |
| A1 | 仅移除交叉注意力 Δt 时间 bias | 同上 | 25/5/8 | 本轮新增三项评价 |

固定完整六个来源身份，新增评价范围为 A1 三个 cell。所有 checkpoint 由完整 validation Local RR 严格最小、并列最早规则确定，保持各自训练配置、原始指标和数据口径。

- 训练 session：`runs/m4_cross_time_bias_v1/formal_r2_20261018/`；session SHA-256=`96975f061d9e1e0e51cbd75d8f0154f383a07389535d6423c0b9853d68c5ee43`。
- validation summary：`summary/10d3d10d991d4a18a57386c755e4d3fe/`；receipt SHA-256=`e3f6206ef63157aaf9ebde106d6c9ffb10f5e5d2c350107e047866c44e1f0458`。
- 历史 P1 test 位于迁存根 `runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/model-architecture-review/` 下的 `runs/p1_components_v1/research_test_v1_20261007/`。
- 历史 test allowlist SHA-256=`8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47`；summary receipt SHA-256=`bae9f68d1834483dc9cd1b734f01d98ea5826186b9f1c41d8c423af7d962d29e`。

prepare 验证训练 session/源码、六项 completion 全部产物、完整 history 选点、best/final 配置及有限性、W 频率坐标、development 样本身份和受试者隔离。A0 的 checkpoint/config/completion 与历史 test 必须为同一文件身份；历史兼容链接按 resolve 后等价路径核对，文件 SHA 与原字符串记录保持冻结。

历史 test 的 allowlist、源码快照、清单内各源码、完整 summary 和对应三个 cell 的 receipt/产物均作哈希核验，推理合同及环境必须匹配。新 allowlist 保存完整六项来源、三项新评价角色、三个旧结果参照、development 身份、validation summary、比较规则和当前完整源码快照。prepare 读取已保存的元数据和产物哈希，不读取 test index、cache manifest、数组或历史 test 指标值。

## 数据与推理

test 固定 2310 窗口/8 人、180 秒/100 Hz、sample seed=20260612；有序 row ID SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。继承冻结 dataset index，运行时核对与 development 的受试者隔离。

使用既有 W-full 97×360 test cache：identity=`40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`，manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。实际频率坐标逐值匹配 checkpoint/allowlist，既有输入预处理、target、归一化和指标保持不变。

A1 按冻结 CrossTimeBiasModel strict 加载，参数量 1,056,257，频率 bias 保留、时间 bias 无参数；物理对齐、时域注意力池化及其位置编码、双投影残差、BiMamba2 保持训练定义。batch=32、BF16、eval 模式、保留尾批；环境软件版本、关键包和 GPU 型号与训练相同。仅在评价配置副本调整设备与进度显示。

逐 batch 核验样本顺序、输入与 target `(B,1,18000)`、CWT `(B,97,360)`。非有限输入、target、checkpoint、prediction 或关键指标显式失败；预测退化保留原分母和诊断。

## 完整对照与判断

三个新评价全部成功后，才读取全部新旧逐窗口结果进行六 cell 汇总。A0 的历史 M4 标签只在内存映射为 A0。跨 cell 样本身份、顺序和 target 资格必须一致。

五项主指标为 Local RR MAE、Whole RR AE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC；保留原生 test 辅助指标、分母、预测退化、Local RR tail、逐受试者与参数量。输出每 seed、跨 seed mean/sample SD、配对差、原始改善方向及达 0.01 阈值的方向数，分别报告窗口加权和受试者宏平均。

delta=A1−A0。误差负值/PCC 正值表示移除后改善；绝对均值差 **<0.01** 记为均值无明显差异，达到阈值按方向分类。按已冻结 validation 表与本轮 test 表生成跨 split 对照。若任一 split、任一种聚合的主指标出现达到阈值的均值退步，则结论标为“存在超阈值取舍，暂不支持直接移除”；若均未出现，则标为“按均值阈值支持移除”。同时完整呈现各 seed 的阈值方向，避免均值掩盖波动。

该规则用于描述性结构选择，不是统计等效检验。validation 参与选点；该 test 已参与研究开发，证据角色为 **reused research/development evidence**。不同 split 的人群不同，三个 seed 共用各自 split 人群，不作独立人群重复解释。不根据部分结果改变剩余评价、候选、阈值、checkpoint 或结构。

自动保存 CSV、两张跨 split PNG、中文 `report.md`、`decision.json` 及全部来源 receipt。原训练、validation、历史 test 与源码快照保持冻结身份。

## 执行与验收

默认新输出 identity：`runs/m4_cross_time_bias_v1/research_test_v1_20261019/`。评价与汇总入口必须显式确认；用户本次已授权 Codex 在空闲 GPU 后台执行。

```bash
cd /mnt/disk_code/marques/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_m4_cross_time_bias_research_test.py
bash -n scripts/run_m4_cross_time_bias_test.sh
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/eval_m4_cross_time_bias.py prepare --training-session runs/m4_cross_time_bias_v1/formal_r2_20261018 --output runs/m4_cross_time_bias_v1/research_test_v1_20261019
bash scripts/run_m4_cross_time_bias_test.sh runs/m4_cross_time_bias_v1/research_test_v1_20261019 1
```

入口提供 prepare/evaluate/status/summarize，shell 单卡串行三个新 checkpoint，成功后自动汇总。每项独立互斥；失败 attempt 保存 traceback，修复原因后可用 `--retry-failed` 创建新 attempt；成功结果哈希验证后复用。验收为三个新成功 receipt、三个持续通过来源验证的历史参照、每项固定 2310 窗口/8 人，以及完整六 cell 和跨 split 阈值报告。
