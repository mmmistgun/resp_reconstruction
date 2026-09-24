# E6：固定 checkpoint 的 research-test 评价附件

日期：2026-09-24。协议 ID：`e6-temporal-frontend-test-v1-20260924`。

状态：**T0 独立合同、实现和 synthetic CPU 验证已完成；没有读取真实 test input、target 或 cache 数组，也没有执行 GPU test 推理。专项锁、真实评价与汇总尚未执行。**

## 1. 目的与证据边界

E6 validation 已完成。`e6_tfe201_aa20_demod10_w0` 的 pooled trajectory 优于 W0，但 Whole RR、Local RR、global envelope 和 PCC 的三 seed 均值退化，subject-macro 未复现 trajectory 收益，因此冻结决定为属性权衡且不替换 W0。完整记录见 `docs/experiments/e6_temporal_frontend_results_20260924.md`。

用户随后开放 E6 test 相关代码。本附件只评价 validation 已固定的三个 epoch-4 checkpoint。Test 不承担挽救候选、重选 checkpoint、修改 early stopping、追加训练或改变 validation 决定的作用。

现有 test split 已被 CRD、CRD-TF、RTM、E4、E5 和论文证据工作重复使用。本附件结果只能称为固定 checkpoint 在复用 research-test 上的开发性描述，不能称为首次 held-out、独立确认或无偏泛化证据。

## 2. 固定 checkpoint 矩阵

Validation summary：

```text
runs/e6_temporal_frontend/summary/summary_d139aca0a26b_20260923T143709Z_4382db584e0d
```

Summary manifest SHA-256：`66b93fee3e1a66d7807528e80a4c7236895501a0d38848653944063861d1b0c4`；训练 implementation lock SHA-256：`d139aca0a26bd7fb1715e7f1fbd802e113e2e9cbc347ed17b4d2133ff613aea3`。

| Seed | E6 selected epoch | Completed epoch | W0 selected epoch |
|---|---:|---:|---:|
| 20260811 | 4 | 30 | 13 |
| 20260812 | 4 | 30 | 15 |
| 20260813 | 4 | 30 | 14 |

三个 E6 checkpoint 必须由 summary receipt 指向的完成 formal attempt 获取，严格锁定 `checkpoint_best_local_rr.pt` 的 path、SHA-256、完整训练 config、selected epoch、history 和 manifest。模型通过 `build_e6_model` 构造并 `strict=True` 加载；不接受 final checkpoint、其他 epoch、W0 state 或失败 attempt。

W0 对照复用相同 seed 的既有 `research_test_metrics.csv`、`research_test_metrics_summary.csv` 和评价 manifest；其 checkpoint/epoch 必须与 E6 训练锁中的 W0 anchor 一致。只新增三次 E6 推理，共 6,930 条逐窗口指标。

## 3. Test 数据与推理合同

- split=`test`，2,310 窗口、8 个 `samp_id`、sample seed=`20260612`；不得抽样或过滤。
- 复用冻结 input-only cache：`runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839/`。
- Cache manifest SHA-256：`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`；test row-order SHA-256：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。
- E6 读取 `test_w.npy [2310,97,360]`、`test_row_ids.npy` 和相同 frequency identity；不重建 cache，不在线重算 W。
- Batch=128、BF16 AMP、eval mode、shuffle=false，保留 6-window 尾 batch。
- 沿用原 180 s/100 Hz input/target、admission、任务投影、eligibility、五指标和 sample-direct mean。

每次评价先写 `access_started.json`，随后核验 test lock、checkpoint、cache manifest/数组、dataset index 和 row identity。读取波形前严格加载模型并验证 checkpoint config、epoch 和 finite state。逐 batch 核对 row/samp/split 顺序、W shape 与 input/target/W finite；任何非有限值显式失败，禁止缩小样本集合。

## 4. 指标与汇总

五主指标为 Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error 和 lag-aware signed PCC。调用冻结 `evaluate_task_predictions(..., include_test_only=False)`，同时保存 eligibility、IBI/coverage、分层 envelope Spearman 和退化诊断；不新增 selector 或综合分数。

逐窗口 target eligibility 必须与同 seed W0 test 完全一致。四项 error 的配对变化为 `E6−W0`，PCC 为 `W0−E6`；正值统一表示 E6 更差。报告两臂三 seed mean/sample SD、同 seed 配对差与方向、三 seed 均值之比、逐 seed×samp_id 配对差和受试者等权宏平均。

有限 prediction degeneration 按指标定义保留并使 `quality_acceptance_passed=false`，不删除样本、不换 checkpoint、不按效果重跑。非有限 prediction 或关键指标缺失属于工程失败。

## 5. 实现与不可覆盖生命周期

```text
resp_train/paper_evidence/e6_temporal_frontend_test.py
scripts/eval_e6_temporal_frontend_test.py
tests/test_e6_temporal_frontend_test.py
docs/experiments/e6_temporal_frontend_test_lock_20260924.json
runs/e6_temporal_frontend_test/
```

`prepare-lock` 排他生成专项锁，绑定本附件/源码 SHA、训练 implementation lock、validation summary、三枚 checkpoint allowlist、开发 split `samp_id`、W0 test 来源、cache manifest/file metadata 和 row/frequency identity。准备阶段只读 manifest、既有指标和 cache metadata，不加载 test cache 数组或原始波形；工作树必须干净。

每 seed 评价输出到 `evaluation/seed_<seed>/evaluation_<lock前缀>_<UTC>_<UUID>/`；汇总输出到独立 `summary/`。每阶段先写 started lifecycle，失败保留现场，成功写 manifest/freeze receipt；相同 test lock 的完成 seed 和 summary 拒绝重跑。

评价至少保存 resolved config、test rows、metrics/summary、环境、实现锁快照、access started/receipt 和 evaluation receipt。汇总保存 6 行 seed metrics、15 行 paired-seed delta、5 行 three-seed comparison、120 行 paired seed-subject delta、15 行 subject macro、来源与完成回执。

## 6. 阶段与授权

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| T0 | 协议、代码、synthetic CPU 测试 | 已完成；18 tests passed |
| T1 | 只读 prepare-lock | 待代码完成、提交和后续指令 |
| T2 | 三 seed GPU test evaluation | 未授权 |
| T3 | 一次性 test 汇总 | 随完整 T2 开放，不得用部分结果汇总 |

T0 fixture 使用临时 checkpoint、metadata、不可解码 cache 字节、解析波形和 tiny 模型，覆盖准备→三 seed 评价→汇总、严格 checkpoint/config/epoch、尾 batch、row/subject 隔离、eligibility、finite/degeneration、cache/source 漂移、并发和不可覆盖语义。

未来入口如下；当前只执行第一条：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e6_temporal_frontend_test.py -q

./.venv/bin/python scripts/eval_e6_temporal_frontend_test.py prepare-lock

for E6_TEST_SEED in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/eval_e6_temporal_frontend_test.py evaluate \
    --seed "$E6_TEST_SEED" --device cuda:0 || break
done

./.venv/bin/python scripts/eval_e6_temporal_frontend_test.py summarize --runs \
  '/seed_20260811完成attempt' '/seed_20260812完成attempt' '/seed_20260813完成attempt'
```
