# M4 linear fusion 固定 checkpoint research-test

协议 ID：`m4-linear-fusion-research-test-v1-20261019`。

2026-10-19 用户授权本轮单投影配置开展 test 评价。三个 seed 的 [train/validation 已完成并关闭](m4_linear_fusion_v1_validation_results_20261019.md)；本协议开放三个固定 checkpoint 的 research-test、与冻结双投影参照的完整六 cell 汇总及 validation/test 比较。完成状态以评价和汇总 receipt 为准；prepare 后本协议与入口保持冻结。

## 固定矩阵与来源

| Arm | 配置 | seeds | 最佳 epoch（11/12/13） | test 角色 |
|---|---|---|---|---|
| L0 | NoCrossTimeBias 双投影 | 20260811/20260812/20260813 | 25/5/8 | 只读复用历史 A1 test |
| L1 | NoCrossTimeBias 单投影 | 同上 | 4/5/8 | 新增三项评价 |

checkpoint 均由完整 validation Local RR 严格最小、并列最早规则确定。单投影保持 `attention.output` 并直接残差相加，参数量 1,046,945；其预定零初始化位置与模型定义见 [训练协议](m4_linear_fusion_v1_protocol_20261019.md)。

- 训练 session：`runs/m4_linear_fusion_v1/formal_v1_20261019/`，SHA-256=`f84949530ceb877a9a1e9e58460dca533f2231b10cccae29ace2e1f76fb0bb7c`。
- validation summary：`summary/0b3bdd6476c14144a9e5c6dd86bef39e/`，receipt SHA-256=`836b554c5796920c838a0e8060797000aad2a641ff3fc8854c733aa647359ac6`。
- 历史双投影 test：`runs/m4_cross_time_bias_v1/research_test_v1_20261019/` 的 A1 三 seed。
- 历史 allowlist SHA-256=`b85d4521c65eaba63114d5e38449a514886f2e2b5ba69761d3e5e6fd848d6a07`；summary receipt SHA-256=`662e194c73329c5c32c40846c388e293e7a8b5b294270b36deec5d36d34987e5`。

prepare 核验训练 session、源码快照、配置、completion 全部产物、best/final checkpoint 有限性与完整 history 选点、W 频率坐标、development 行身份与受试者隔离。L0 的 checkpoint/config/completion 必须匹配历史 test 的同一文件身份。历史 allowlist、源码快照、清单内源码、summary 及对应三项评价产物均核验哈希与推理环境。

新 allowlist 固定六个来源、三项评价角色、validation 汇总和当前源码快照。prepare 只读取保存的元数据、checkpoint 与产物哈希，不读取 test index、cache manifest、数组或历史 test 指标值。代码身份包含前序冻结评价依赖及本轮入口、测试、shell、协议；文件实体均保存于项目内。

## 数据与推理

test 固定 2,310 窗口/8 人，180 秒/100 Hz，sample seed=20260612；有序 row ID SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。沿用冻结 dataset index，核对与 train/validation 的受试者隔离。

既有 W-full 97×360 test cache identity=`40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`，manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。频率坐标逐值匹配 checkpoint 和 allowlist。

L1 使用冻结 LinearFusionModel strict 加载，核验参数量；batch=32、BF16、eval 模式，保留尾批。软件版本、关键包及 GPU 型号与训练匹配。仅调整评价配置副本的设备和进度显示。逐 batch 检查样本顺序、输入/target `(B,1,18000)` 与 CWT `(B,97,360)`；非有限输入、target、checkpoint、prediction 或关键指标显式失败，预测退化保留原分母与诊断。

## 完整比较与判断

三个新评价全部成功后，汇总三个 L1 与三个冻结 L0；旧 A1 标签仅在汇总内存中映射为 L0。六个 cell 样本身份、顺序及 target 资格必须一致。

报告五项主指标：Local RR MAE、Whole RR AE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC；保留 test 原生辅助指标、分母、预测退化、Local RR tail、逐受试者、参数与来源。保存各 seed、跨 seed mean/sample SD、同 seed 配对及原始方向和达阈值方向数。

delta=L1−L0，误差负值/PCC 正值为改善。沿用训练协议的**窗口加权选择口径，受试者宏平均作为诊断**。绝对均值差 <0.01 为均值无明显差异，达到阈值按方向分类。若 validation 或 test 任一窗口加权主指标出现达阈值均值退步，标记存在超阈值取舍；否则按均值阈值支持单投影。同时完整呈现宏平均与单 seed 差异。

0.01 是描述性判据，不是统计等效检验。validation 参与选点；test 已用于研究开发，证据角色为 **reused research/development evidence**。不同 split 人群不同，各 seed 共用各自 split 人群。不根据部分结果改变模型、checkpoint、剩余矩阵、指标或阈值。

## 执行与验收

新输出 identity：`runs/m4_linear_fusion_v1/research_test_v1_20261019/`。GPU 0 执行 seed 20260811、20260813，GPU 1 执行 seed 20260812，启动前确认空闲。shell 原子排他创建分片日志，等待两组成功后自动汇总；任一失败保留非零状态和失败产物。评价入口须显式 `--confirm-research-test`；本次用户授权覆盖 Codex 后台执行。

```bash
cd /mnt/disk_code/marques/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_m4_linear_fusion_research_test.py
bash -n scripts/run_m4_linear_fusion_test.sh
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/eval_m4_linear_fusion.py prepare --training-session runs/m4_linear_fusion_v1/formal_v1_20261019 --output runs/m4_linear_fusion_v1/research_test_v1_20261019
bash scripts/run_m4_linear_fusion_test.sh runs/m4_linear_fusion_v1/research_test_v1_20261019 0 1
```

prepare/evaluate/status/summarize 复用不可覆盖生命周期；失败原因修复后通过 `--retry-failed` 创建新 attempt，成功产物哈希验证后复用。验收要求三个新增成功 receipt、三个持续通过来源验证的历史参照、每项固定 2,310 窗口/8 人，以及完整六 cell、跨 split CSV、两张 PNG、中文报告、decision 和来源 receipt。历史训练、validation 与 test 产物保持冻结。
