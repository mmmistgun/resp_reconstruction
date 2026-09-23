# E5 时域前端：固定 checkpoint 的 research-test 评价附件

日期：2026-09-23。协议 ID：`e5-temporal-frontend-test-v1-20260923`。

状态：**T0 独立合同、实现和 synthetic CPU 验证已完成；尚未读取真实 test input/target/cache 数组，专项锁、真实评价与汇总尚未执行。**

## 1. 目的与证据边界

E5 validation 已完成并判定 `e5_tfe101_aa10_res_w0` 为 net negative，不替换 W0。用户随后指令“开始test相关代码”，据此实现完整三 seed research-test 路径。Test 不承担挽救候选、重选 checkpoint、修改 early stopping、追加训练或改变 validation 决定的作用；无论 test 结果如何，validation 结论保持原身份。

现有 test split 已在多轮 CRD、CRD-TF、E4 和论文证据工作中使用。本附件只能形成 **固定 validation-selected checkpoint 在复用 research-test 上的开发性描述**，不得称为首次 held-out、独立确认或无偏泛化证据。

本轮代码阶段允许只读核验协议、validation summary、训练 manifest/checkpoint metadata、既有 W0 test 指标和 test cache manifest 的接口设计；synthetic 测试全部使用临时 fixture。真实 test 波形、target、cache 数组和模型推理只在用户以后明确授权 `evaluate` 后开放。

## 2. 固定 checkpoint 矩阵

Validation summary：

```text
runs/e5_temporal_frontend/summary/summary_d7cf90ccb432_20260923T022609Z_937e34d10c67
```

Summary manifest SHA-256：`153b581b0ec1ca0ba5b286d990bdf248bb8e7a35949e79d36331b5cf6740d7f8`；训练 implementation lock SHA-256：`d7cf90ccb432f2bda4563f6719385eb98564c1594c43c8b75c10f10374dce7f5`。

| Seed | E5 selected epoch | Completed epoch | W0 selected epoch |
|---|---:|---:|---:|
| 20260811 | 9 | 30 | 13 |
| 20260812 | 13 | 30 | 15 |
| 20260813 | 5 | 30 | 14 |

三个 E5 checkpoint 必须由 summary receipt 指向的完成 formal attempt 获取，严格锁定 `checkpoint_best_local_rr.pt` 的 path、SHA-256、完整训练 config、selected epoch、history 和 manifest。模型通过 `build_e5_model` 构造并 `strict=True` 加载，不允许用普通 W0 state、final checkpoint 或其他 epoch 替代。

W0 对照复用相同 seed 的既有 `research_test_metrics.csv`、`research_test_metrics_summary.csv` 和评价 manifest；其 checkpoint/epoch 必须与 E5 训练锁中的 W0 anchor 一致。只新增三次 E5 推理，共 6,930 条逐窗口指标。

## 3. Test 数据和推理合同

- split=`test`，2,310 窗口、8 个 `samp_id`、sample seed=`20260612`；不得抽样或过滤。
- 复用冻结 input-only cache：`runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839/`。
- Cache manifest SHA-256：`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`；test row-order SHA-256：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。
- E5 只读取 `test_w.npy [2310,97,360]`、`test_row_ids.npy` 和相同 frequency identity；不重建 cache，不在线重算 W。
- Batch=128、BF16 AMP、eval mode、shuffle=false，保留 6-window 尾 batch。
- 沿用原 180 s/100 Hz input/target、admission、任务投影、eligibility、五指标和 sample-direct mean。

每次评价先写 `access_started.json`，随后核验 test lock、checkpoint、cache manifest/所需数组、dataset index 和 row identity。读取波形前严格加载模型并验证 checkpoint config/epoch/state finite。逐 batch 核对 row/samp/split 顺序、W shape 和 input/target/W finite；非有限值显式失败，禁止缩小集合。

## 4. 指标与汇总

五主指标为 Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error 和 lag-aware signed PCC。调用冻结 `evaluate_task_predictions(..., include_test_only=False)`，同时保存 eligibility、IBI/coverage、分层 envelope Spearman 和退化诊断；不新增 coherence/nDTW 或新 selector。

逐窗口 target eligibility 必须与同 seed W0 test 完全一致。四项 error 的配对变化为 `E5−W0`，PCC 为 `W0−E5`；正值统一表示 E5 更差。报告两臂三 seed mean/sample SD、同 seed 配对差与方向、均值之比、逐 seed×samp_id 配对差和受试者等权宏平均。Test 不构造总分、Pareto、重新 qualification 或 checkpoint 排名。

有限 prediction degeneration 按指标定义保留并使 `quality_acceptance_passed=false`，不删样本、不换 checkpoint、不按效果重跑。非有限 prediction 或关键指标缺失属于工程失败。

## 5. 实现与不可覆盖生命周期

```text
resp_train/paper_evidence/e5_temporal_frontend_test.py
scripts/eval_e5_temporal_frontend_test.py
tests/test_e5_temporal_frontend_test.py
docs/experiments/e5_temporal_frontend_test_lock_20260923.json
runs/e5_temporal_frontend_test/
```

`prepare-lock` 排他生成专项锁，绑定本附件/源码 SHA、训练 implementation lock、validation summary、三枚 checkpoint allowlist、开发 split `samp_id`、W0 test 来源、cache manifest/file metadata 和 row/frequency identity。准备阶段只读 manifest 和既有指标，不加载 test cache 数组或原始波形；工作树必须干净。

每 seed 的评价输出到 `evaluation/seed_<seed>/evaluation_<lock前缀>_<UTC>_<UUID>/`；汇总输出到独立 `summary/`。每阶段先写 started lifecycle，失败保留现场，成功写 manifest/freeze receipt；相同 test lock 的完成 seed 和 summary 拒绝重跑。影响科学合同或实现的修订必须生成新锁，不能覆盖。

评价至少保存 resolved config、test rows、metrics/summary、环境、实现锁快照、access started/receipt、evaluation receipt。汇总保存 6 行 seed metrics、15 行 paired-seed delta、5 行 three-seed comparison、120 行 paired seed-subject delta、15 行 subject macro、来源与完成回执。

## 6. 阶段和授权

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| T0 | 协议、代码、synthetic CPU 测试 | 已完成；专项与相邻回归共 70 tests passed |
| T1 | 只读 prepare-lock | 待代码完成与用户后续指令 |
| T2 | 三 seed GPU test evaluation | 未授权 |
| T3 | 一次性 test 汇总 | 随完整 T2 开放，不得用部分结果汇总 |

T0 测试使用 synthetic checkpoint、metadata、不可解码 cache 字节 fixture、波形和 tiny 模型，覆盖准备→三 seed 评价→汇总、严格 checkpoint/config/epoch、尾 batch、row/subject 隔离、eligibility、finite/degeneration、cache/source 漂移、并发和不可覆盖语义。当前不得运行真实 `prepare-lock`、`evaluate` 或 `summarize`。

未来命令仅作入口设计；实际执行须遵守上述阶段授权：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e5_temporal_frontend_test.py -q

./.venv/bin/python scripts/eval_e5_temporal_frontend_test.py prepare-lock

for E5_TEST_SEED in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/eval_e5_temporal_frontend_test.py evaluate \
    --seed "$E5_TEST_SEED" --device cuda:0 || break
done

./.venv/bin/python scripts/eval_e5_temporal_frontend_test.py summarize --runs \
  '/seed_20260811完成attempt' '/seed_20260812完成attempt' '/seed_20260813完成attempt'
```
