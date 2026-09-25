# E7：固定 18-checkpoint 的 research-test 评价附件

日期：2026-09-26。协议 ID：`e7-scale-encoding-aggregation-research-test-v1-20260926`。

状态：**T0 专项合同、入口与 synthetic CPU 验收已完成（10 tests passed）；尚未建立正式 test lock，未读取真实 test input、target 或 cache 数组，未执行 GPU test 推理。**

## 1. 目的与证据边界

E7 已完成六臂×三 seed formal train/validation、selected-checkpoint 诊断和 P5 冻结。P5 的 validation 决定为：测试的模型族与容量范围内，没有稳定的五指标 tolerance-aware Pareto 改善；deep-local 的 envelope 收益伴随 RR 权衡，axis-spanning 没有稳定增量收益，attention 被模型实际使用但仍表现为属性权衡。

本附件只评价 P5 之前已经冻结的 18 个 `checkpoint_best_local_rr.pt`。Research-test 不用于重选 checkpoint、修改 early stopping、追加训练、筛除 arm/seed、改变 tolerance 或改写 validation 决定。

当前 test split 已被多个科研阶段使用。本附件结果只作为固定 checkpoint 在复用 research-test 上的开发性描述，不表述为首次 held-out、独立确认或无偏泛化证据。

## 2. 固定矩阵与来源

- P4 execution lock SHA-256：`068ba8ec6c5866ebf1b17d560b4fb5f5448e7e921e0dc8b15dda66c2f907c193`。
- P5 closeout：`docs/experiments/e7_scale_encoding_aggregation_p5_closeout_20260926.json`，SHA-256：`7e9bc50516020c2193cacae9c3295dd28669906d4a74f80512d6f35e3fe91eee`。
- P5 final manifest SHA-256：`aae2d23fecd116db73b239ea028b4e394e72300b09e7077d54bc8e4235ad9837`。
- 固定 arms：`s0_shallow__mean`、`s0_shallow__frequency_attention`、`s1_deep_local__mean`、`s1_deep_local__frequency_attention`、`s2_axis_spanning__mean`、`s2_axis_spanning__frequency_attention`。
- 固定 seeds：`20260811/20260812/20260813`。

专项锁必须从唯一完成的 18 个 P4 formal attempt 构造 allowlist。每项绑定 arm、seed、训练 attempt/manifest、完整 config、history、selected/completed epoch、`checkpoint_best_local_rr.pt` identity 及开发 split `samp_id`。必须回放 earliest strict-minimum Local RR selector，并严格加载 checkpoint state。

同 seed W0 只复用现有 `research_test_metrics.csv`、summary 与 manifest，作为固定外部参照；不新增 W0 推理。E7 新评价共 `18 × 2310 = 41,580` 条逐窗口指标。

## 3. 数据与推理合同

- split=`test`，2,310 窗口、8 个 `samp_id`、sample seed=`20260612`，完整读取，不抽样。
- 复用冻结 input-only W cache：`runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839/`。
- Cache manifest SHA-256：`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`；row-order SHA-256：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。
- 核验 `test_w.npy [2310,97,360]`、`test_row_ids.npy`、frequency file 与 dataset index identity；不重建 cache。
- batch=128、BF16 AMP、eval mode、shuffle=false，保留 6-window 尾 batch。
- 保持 `L_sync + 0.25 L_effort` 对应的训练配置、原始模型结构、任务投影、eligibility 与 `include_test_only=false` 指标口径。

每个评价 attempt 先写 `access_started.json`，再核验专项锁、全部冻结来源、checkpoint、cache 文件、dataset index 和 row identity。逐 batch 核对 row/samp/split 顺序、W shape 以及 input/target/W 有限性。非有限数据、state、prediction 或主指标显式失败；有限 prediction degeneration 保留完整行并写入质量标志。

## 4. 汇总与解释

一次性汇总只接受专项锁下六臂×三 seed 的 18 个唯一成功 attempt。产物包括：

- 18 行 per-seed 指标、30 行 arm×metric across-seed 表；
- 105/35 行 simple-effects per-seed/across-seed；
- 75/25 行 factorial-contrasts per-seed/across-seed；
- 105/35 行 tolerance-aware materiality per-seed/across-seed；
- 810 行 subject-macro（18 cell × 5 metric × 8 subjects 加 macro）；
- 90/30 行相对同 seed W0 的 per-seed/across-seed 描述。

四项 error 使用相对 utility 改变量，PCC 使用绝对 utility 改变量，正值表示 candidate 改善。核心结论仍按五指标 tolerance-aware Pareto、三 seed 方向和 subject-macro 解释，不构造总分。Test 汇总与 validation 结论分开报告，并明确 `validation_decision_changed=false`。

## 5. 生命周期与执行锁

实现入口：

```text
resp_train/paper_evidence/e7_scale_encoding_aggregation_test.py
scripts/eval_e7_scale_encoding_aggregation_test.py
tests/test_e7_scale_encoding_aggregation_test.py
docs/experiments/e7_scale_encoding_aggregation_test_lock_20260926.json
runs/e7_scale_encoding_aggregation/research_test/
```

`prepare-lock` 要求干净工作树，只读取 P4/P5 冻结产物、现有 W0 指标及 cache manifest 元数据；不会加载 test cache 数组或原始波形。专项锁建立后先提交，再进入真实评价。每个 cell 与 summary 使用文件锁和不可覆盖 attempt；失败现场保留，相同 test lock 下已经完成的 cell/summary 拒绝重跑。

## 6. 阶段门控

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| T0 | 协议、实现、synthetic CPU tests | 已完成；10 tests passed |
| T1 | metadata-only `prepare-lock`、核验并提交专项锁 | 待执行 |
| T2 | 18-cell GPU research-test evaluation | 由用户执行 |
| T3 | 完整矩阵一次性汇总 | 仅在 T2 全部完成后执行 |

入口命令：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python -m pytest tests/test_e7_scale_encoding_aggregation_test.py -q

./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py prepare-lock
./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py check-lock

for E7_TEST_SEED in 20260811 20260812 20260813; do
  for E7_TEST_ARM in \
    s0_shallow__mean \
    s0_shallow__frequency_attention \
    s1_deep_local__mean \
    s1_deep_local__frequency_attention \
    s2_axis_spanning__mean \
    s2_axis_spanning__frequency_attention; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      ./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py evaluate \
      --arm "$E7_TEST_ARM" --seed "$E7_TEST_SEED" --device cuda:0 || break 2
  done
done

./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py check-completed
./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py summarize --completed
```
