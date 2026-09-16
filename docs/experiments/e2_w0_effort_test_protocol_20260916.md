# E2：相对努力损失消融的 test 评价附件

协议 ID：`e2-w0-effort-test-v1-20260916`；日期：2026-09-16。

状态：**独立 test 入口已实现，正式 test 推理待用户执行。**

## 1. 科学问题与范围

本附件承接 E2 已完成的三 seed train/validation 消融，评价去除相对努力项后在既有 test split 上的包络轨迹、包络范围、Whole/Local RR 和 signed PCC 变化。五项指标共同报告，结果解释结合属性间取舍。

E2 validation 已完成 3×80 epochs / 6400 updates，checkpoint 按各自完整 validation 的最早最小 Local RR 固定。该选择在本次 test 评价前已经完成。本附件沿用既有 subject/session 隔离的 test：2310 个窗口、8 个 samp_id，test sample seed=20260612。既有 test 具有重复访问历史，证据按复用研究测试集上的固定 checkpoint 比较解释。

用户已要求实现 E2 test 评价代码。当前工作包括专项合同、代码、身份锁和 synthetic CPU 验证；正式 GPU test 推理由用户执行下面的独立入口。

## 2. 冻结对象与对照

| Seed | E2 selected epoch | W0 FULL selected epoch | 比较角色 |
|---|---:|---:|---|
| 20260811 | 31 | 13 | 同 seed loss 消融 |
| 20260812 | 36 | 15 | 同 seed loss 消融 |
| 20260813 | 33 | 14 | 同 seed loss 消融 |

E2 来源为：

```text
runs/e2_w0_effort_ablation_v1/summary/summary_29ff147334a3_20260916T064752Z_180c97e12740/
```

其 manifest SHA-256 为 `ab0165f4846a2b74e83a7f0bf70bc9bd09f80cee53fecf76a322e9d8f54489ad`，训练实现锁 SHA-256 为 `29ff147334a358be0b807a5e068e280891522f7ceef831b4ed16105988589edf`。三个具体训练 attempt 从该冻结 summary receipt 读取，锁定各自 `checkpoint_best_local_rr.pt`、resolved config、history 和完成回执。

W0 FULL 直接复用同 seed 的冻结 `research_test_metrics.csv` 与 `research_test_metrics_summary.csv`。来源通过 `runs/crd_tf_v1/research_test_summary/access_audit.csv` 的 checkpoint、epoch、metrics、summary 和 evaluation manifest 哈希闭合。该 audit SHA-256 为 `eb75cce11e1827b983f6540719e772c1e0288e8ffda06b7f3ae917bc2920d377`。

所有新评价均为已固定的 E2 模型，新推理数为 3 次、逐窗口新记录共 6930 条。完整 W0 的原始 test 结果保留为比较锚点。

## 3. 数据、缓存和推理路径

使用既有冻结 test W cache：

```text
runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839/
```

manifest SHA-256 为 `5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`；test row 顺序哈希为 `184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。

- 继续使用 180 s、100 Hz、原 BCG/THO 载体、admission、任务投影和冻结五主指标。
- 单窗口 W shape=`[97,360]`；batch=128，BF16 AMP，完整尾 batch；模型使用原生 eval forward。
- 独立 data config 指向冻结 test cache，使用原生 `TfV1ResearchTestCacheReader` 只读校验并按 row ID 加载；训练 config 独立保留用于 checkpoint 配置比较。
- 正式读取波形前校验 dataset index、完整 test rows、与冻结 W0 的 row/samp/split 顺序，以及 test 与该训练 run 的 train/validation samp_id 隔离。
- 对每个 batch 核对身份、W shape 和 input/target finite。prediction、五主指标、target eligibility 与预测退化使用冻结实现校验。五主指标分母均为 2310。
- 本附件报告五主指标及原生资格/诊断字段，调用 `evaluate_task_predictions(..., include_test_only=False)`；与冻结 W0 配对使用相同的五主指标定义。

## 4. 实现、锁与产物

独立入口：

```text
scripts/eval_e2_w0_effort_test.py
resp_train/paper_evidence/e2_effort_test.py
docs/experiments/e2_w0_effort_test_lock_20260916.json
```

准备阶段核验 E2 完成的训练与 validation 产物、W0 已冻结 test metrics 和缓存 manifest。准备过程只读取来源配置、日志表、manifest、checkpoint 字节身份和已有结果记录；test 输入/target 波形及缓存数组在正式评价阶段读取。

新锁记录全部 checkpoint allowlist、E2 原始训练配置、开发集 samp_id、W0 test 来源、test cache 文件预期 hash/size、dataset index、row 顺序及本轮源码/协议身份。原训练协议、实现锁和已完成产物作为只读来源。

输出根为 `runs/e2_w0_effort_test_v1/`，按 `evaluation/seed_<seed>/<attempt>` 与 `summary/<attempt>` 隔离。每个 attempt 排他创建，以实现锁标识、阶段和独立 ID 命名。

每 seed 交付 `metrics.csv`、`metrics_summary.csv`、`test_rows.csv`、`resolved_config.yaml`、实现锁快照、checkpoint/访问/环境/评价回执和 lifecycle。完成标识为 `manifest.json` 与 `freeze_receipt.json`。异常显式失败并保留已有文件；同身份成功 seed 直接复用。代码修订应另立身份并明确与既有产物的兼容关系。

## 5. 配对汇总

只有三个完成 seed 齐全、全部对应本附件实现锁及固定 checkpoint 时才生成汇总。逐窗口身份、target-only 资格与同 seed W0 对齐后，按既有 sample-direct mean 聚合。

四项 error 的逐 seed delta 为 `100×(sync_only−W0_FULL)/W0_FULL`，PCC 为 `W0_FULL−sync_only`。正值表示完整目标更好。输出两臂的原始 seed 指标、各自三 seed mean/sample SD、逐 seed delta、delta mean/sample SD、方向数和均值层面的变化。样本 SD 使用 ddof=1。

最终产物为 `seed_metrics.csv`（2 arms×3 seeds）、`paired_seed_delta.csv`（15 行）、`three_seed_comparison.csv`（5 行）及来源回执。test 表独立于既有 validation 表，解释分别绑定各自 split。

## 6. 验证与运行命令

准备与 synthetic CPU 验证：

```bash
./.venv/bin/python -m pytest tests/test_e2_effort_test.py -q
./.venv/bin/python scripts/eval_e2_w0_effort_test.py prepare-lock
```

2026-09-16 synthetic CPU 定向测试结果：`15 passed`。验证覆盖临时数据上的完整锁准备、原生预测收集/指标计算、尾 batch、三 seed 配对汇总、源文件保持、checkpoint/cache 漂移、split/row 顺序、非有限/退化/eligibility、预选 epoch 与开发受试者隔离，以及不可覆盖的成功与失败 lifecycle。正式原生模型的 GPU 数值路径沿用已完成 E2 validation；本轮 test 推理尚待用户运行。

提交本轮实现、协议与锁，保持工作树干净后执行：

```bash
for E2_TEST_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    ./.venv/bin/python scripts/eval_e2_w0_effort_test.py evaluate \
    --device cuda:0 --seed "$E2_TEST_SEED" || break
done
```

每条命令输出对应的完成 attempt 目录。工程错误时停止并保留现场；继续时执行尚未完成的 seed。完整三 seed 后汇总：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/eval_e2_w0_effort_test.py summarize --runs \
  '/seed_20260811的完成test_attempt目录' \
  '/seed_20260812的完成test_attempt目录' \
  '/seed_20260813的完成test_attempt目录'
```

验收标准：每 seed 2310 个唯一窗口、8 个 samp_id；五主指标 finite 且分母完整，prediction degeneracy 为零；checkpoint 为 31/36/33，固定 W0 对照身份一致；三 seed 汇总和完整 manifest/冻结回执齐全。
