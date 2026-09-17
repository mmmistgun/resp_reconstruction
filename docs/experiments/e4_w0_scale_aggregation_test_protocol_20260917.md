# E4：四区域尺度聚合的 test 评价附件

协议 ID：`e4-w0-scale-aggregation-test-v1-20260917`；日期：2026-09-17。

状态：**独立 test 评价、三 seed 汇总入口与专项锁已准备；20 项 synthetic CPU 定向测试通过。真实 test 推理由用户执行，尚未运行。**

## 1. 科学问题与当次范围

承接 [E4 主协议](e4_w0_scale_aggregation_protocol_20260917.md)第 9 节，检验固定四区域残差聚合相对 W0 的五属性变化在既有、受试者隔离的 test split 上如何表现。validation 已完成：Whole RR 三 seed 改善，Local RR 和包络指标存在取舍，signed PCC 三 seed 下降。本附件完整评价三个预选 checkpoint，报告五指标连续效应量与方向；不依据部分 test 结果修改剩余矩阵、结构、超参数或 selector。

用户本次指令“完成test相关的代码”授权独立 test 合同、代码、身份锁及 synthetic CPU 测试。本次准备允许只读核验已冻结 checkpoint 的字节身份、配置/history/manifest、W0 已有 test 指标及 cache manifest；真实 test 输入/target 和 cache 数组留到用户执行 `evaluate` 时访问。Codex 不代跑真实 test 推理，不新建或重建 cache。

该 test 有研究过程中的重复访问历史，证据定位为 **固定 validation-selected checkpoint 在复用研究测试集上的比较**。结果分别绑定 test 与 validation，不合并为新 selector 或确认性独立泛化证据。

## 2. 固定模型与来源

| Seed | E4 selected epoch | W0 selected epoch |
|---|---:|---:|
| 20260811 | 10 | 13 |
| 20260812 | 5 | 15 |
| 20260813 | 17 | 14 |

E4 arm：`w0_mr4_residual`，四区槽位数 25/24/24/24，`384→96` 残差投影、fill=65，全模型 1,256,714 参数。模型从各自完成的 80 epochs / 6400 updates 中按完整 validation Local RR 最早严格最小值预选；本附件锁定 `checkpoint_best_local_rr.pt`，不重选或微调。

E4 validation 来源：

```text
runs/e4_w0_scale_aggregation_v1/summary/summary_464e073dbd57_20260917T110009Z_b816da172253/
```

- summary manifest SHA-256：`eccc4b2b0a725d19900073b6fbcadf5e00b3fc1a93f13cf5b79a6b7e77ba869b`。
- 训练实现锁 SHA-256：`464e073dbd5707a30575d606dec2a84dcd89a161945e537c463b214a15c2b493`。
- 训练执行 commit：`d4c27a89758d7d314173abf33fa0986a8c787d0f`。
- 三个正式 attempt 从冻结 summary receipt 获取，锁保存它们的完成身份、config、history、checkpoint、train/val samp_id。历史训练代码、协议、实现锁及产物保持原字节；test 使用新文件和新锁。

W0 对照复用同 seed 原 run 的 `research_test_metrics.csv`、`research_test_metrics_summary.csv` 与评价 manifest。来源闭合到 `runs/crd_tf_v1/research_test_summary/access_audit.csv`，该文件 SHA-256 为 `eb75cce11e1827b983f6540719e772c1e0288e8ffda06b7f3ae917bc2920d377`。其中 checkpoint/selected epoch 必须匹配 E4 原训练锁的 W0 anchor。只对 E4 新推理三次，共生成 **6930 条**新逐窗口记录。

## 3. Test 数据及推理合同

- 完整 test：**2310 窗口、8 个 samp_id**；test sample seed=20260612。沿用 180 s、100 Hz、原 BCG/THO 载体、admission、任务投影、target eligibility 和 sample-direct mean。
- 原冻结 cache 根：`runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839/`。
- cache manifest SHA-256：`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。
- test row 顺序 SHA-256：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。
- W shape=`[97,360]`；频率文件 SHA-256=`15cc722c38e5572b3284c92137cfdeec7cbc4153b4588c2c1e086f26ae3238b3`，与原 E4 训练网格完全一致。
- batch=128，BF16 AMP，eval mode，shuffle=false，完整尾 batch（18×128+6）。原 E4 eval forward 保持六层主干、相同 FiLM 和 decoder。

评价前先登记 access_started；核验全部来源、test cache 文件及 dataset index 字节身份，再严格加载 E4 checkpoint。checkpoint 完整训练配置须与 allowlist 相同；独立 data config 仅追加 research-test cache 路径，并允许设备/进度显示变化。旧训练配置单独保留，不用含 cache override 的 data config 冒充训练身份。

读取真实 test 波形前，核验完整 test rows 与 W0 的 row/samp/split 顺序，以及 test 与该 seed 的 train/validation samp_id 无交叉。数据加载继续使用原生 `TfV1ResearchTestCacheReader`。逐 batch 核对 identity、顺序、W shape、input/target finite，禁止静默过滤或缩小样本集合。模型通过 `build_e4_model` 构造并 `strict=True` 加载，普通 W0 模型不能代替 E4。

## 4. 指标、质量与配对统计

五主指标：Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。调用原生 `evaluate_task_predictions(..., include_test_only=False)`，同时保存原生资格与诊断字段；本附件不追加 coherence/nDTW。

逐窗口 `TARGET_COLUMNS`（Whole/Local RR、Local RR eligible windows、joint、envelope Spearman、IBI）须与 W0 同窗口一致；五主指标的 sample 分母均为 2310。非有限 input/target/prediction 或应有资格的关键指标显式失败。有限退化预测按原指标定义计入结果，保留退化数并标记 `quality_acceptance_passed=false`，不据此删样本、换 checkpoint 或按效果重跑。质量失败与工程失败分开记录。

四项 error 逐 seed 原始差为 `E4−W0`，相对差为 `100*(E4−W0)/W0`；PCC 为绝对下降 `W0−E4`。正 delta 统一表示 E4 较差。完整报告三 seed 原值、各 arm mean/sample SD（ddof=1）、配对差 mean/sample SD、改善/相等/恶化方向数，以及均值之比。基线误差为零时，相对差明确 NA，保留原始差与全部 seed。沿用 E4 validation 的统计公式，所有 test 表显式标记 `split=test`，不得混用 validation 表或其分母。

没有效果合格线或五项加权总分。解释结合效应量、seed 方向、属性代价和已冻结的容量/效率信息。三 seed SD 描述训练实例差异，重叠窗口不作为独立显著性检验样本。test 结果不会自动替换论文冻结主模型。

## 5. 实现、身份锁与生命周期

```text
resp_train/paper_evidence/e4_scale_aggregation_test.py
scripts/eval_e4_w0_scale_aggregation_test.py
tests/test_e4_scale_aggregation_test.py
docs/experiments/e4_w0_scale_aggregation_test_lock_20260917.json
runs/e4_w0_scale_aggregation_test_v1/
```

准备锁排他创建，保存新协议/代码 SHA、训练锁/validation summary identity、三枚 checkpoint allowlist、原训练配置及开发 samp_id、W0 test 来源、cache/index/row/frequency identity。准备过程读取 cache manifest 中的预期字节身份，不加载或解码真实 test 数组。运行时才核验 cache 实体文件并开放受控读取。

每 seed 输出到 `evaluation/seed_<seed>/evaluation_<lock前缀>_<UTC>_<UUID>/`，汇总输出到独立 `summary/`。同 seed/phase/lock 使用进程互斥锁；时间戳不是唯一身份。每次保存 started lifecycle，工程错误保留 failed lifecycle 和现场；完成写 manifest 与 freeze receipt。同实现身份的完成 seed/summary 拒绝重跑，工程失败在排除故障后以新 attempt 继续，已完成 seed 保留。影响实现或科学合同的修订需新的身份和可比性说明。

评价交付：`metrics.csv`、`metrics_summary.csv`、`test_rows.csv`、`resolved_config.yaml`、实现锁快照、环境、access_started/access_receipt、evaluation_receipt 与生命周期。回执包含 checkpoint SHA、selected epoch、完整分母、target identity 与质量状态。

汇总仅接受三个不同、完成且同 test lock 的 seed，重新从其逐窗口 CSV 核对 summary、资格与 W0 参照。输出 `seed_metrics.csv`（6 行）、`paired_seed_delta.csv`（15 行）、`three_seed_comparison.csv`（5 行）和来源回执；保存两侧 SHA、selected epoch、分母和明确 test 标签。

## 6. 验证与用户执行

synthetic CPU 验证仅使用临时生成的 checkpoint、metadata、cache 字节 fixture、波形和 tiny 模型，覆盖完整准备→三 seed 评价→配对汇总、尾 batch、错误模型/epoch/config、cache/source 漂移、身份顺序、target eligibility、非有限值、退化计数、受试者隔离、并发互斥和不可覆盖生命周期。另在 CPU 构造原生 E4 检查类型、参数数及严格拒绝 W0 state，不运行原生 GPU 推理。

2026-09-17 验证结果：`20 passed in 27.23s`。fixture 的 test 集合为 4 个合成窗口、2 个合成 samp_id、batch=3+1；正式合同仍为 2310 窗口、8 个 samp_id、batch=128。锁准备 fixture 使用不可解码为 npy 的临时字节，确保准备阶段不会提前加载 test 数组。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python -m pytest tests/test_e4_scale_aggregation_test.py -q
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/eval_e4_w0_scale_aggregation_test.py prepare-lock
```

已准备的锁直接复用。提交本轮 test 实现、附件及锁，保持工作树干净后，由用户执行：

```bash
for E4_TEST_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    ./.venv/bin/python scripts/eval_e4_w0_scale_aggregation_test.py evaluate \
    --device cuda:0 --seed "$E4_TEST_SEED" || break
done
```

工程错误时停止并保留现场；继续时只执行尚未完成的 seed。完整三个 test attempt 后：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/eval_e4_w0_scale_aggregation_test.py summarize --runs \
  '/seed_20260811的完成test_attempt目录' \
  '/seed_20260812的完成test_attempt目录' \
  '/seed_20260813的完成test_attempt目录'
```

完成验收要求三个预选 checkpoint、每 seed 2310 个唯一窗口和 8 个 samp_id、完整五指标与资格、质量标志、6/15/5 行汇总及来源/冻结回执均可核对。退化计数为零才标记质量验收通过；质量失败结果仍完整披露。
