# E8 P5：完整 validation 汇总协议

日期：2026-09-26。协议 ID：`e8-film-decoder-redesign-factorial-v1-20260925`。

状态：**36/36 个 formal train/validation cell 与一次性 P5 汇总均已完成并冻结；当前结果入口为 `e8_film_decoder_redesign_v1_validation_results_20260926.md`。Research-test 未开放。**

## 1. 输入门控

汇总只接受以下固定身份：

- Formal execution lock：`40b30fc6ecdcc9b40750c393be8fd2faf4e223da0566432df739aacb513713c5`。
- Runtime amendment：`517bdf281a0db0902e0c10481b0ea210a7c514d627eb6397e9e982d95ba270e3`。
- 12 arms × 3 seeds 的 36 个唯一成功 attempt；每个 cell 必须恰有一个 freeze receipt。
- 每个 attempt 为 2,675 行 validation metrics，总计 96,300 行。
- Amendment 前 11 个成功 cell 必须属于其冻结 allowlist；后续成功 cell 必须登记 amendment SHA。

`e8_res192_pointwise / seed_20260811` 的一次 GPU 容量门控失败发生在数据访问和训练之前，失败生命周期保留；同一 cell 后续有唯一成功 attempt。失败 attempt 不进入指标汇总。

## 2. 汇总内容

一次性生成：

1. 36 行逐 arm/seed 五主指标与 12×5 的跨 seed mean/sample SD；
2. 11 arms 相对同轮 W0 参照 `e8_fill65_pointwise` 的逐 seed 方向与材料性比较；
3. 条件末端计划对比：`direct-fill65`、`res96-direct`、`res192-res96`，分别报告三个 decoder 下的简单效应和跨 decoder 等权边际效应；
4. Decoder 计划对比：`single-pointwise`、`temporal-pointwise`、`temporal-single`，分别报告四个条件末端下的简单效应和跨条件末端等权边际效应；
5. 以 `fill65×pointwise` 为参照编码的 3×2=6 个 interaction difference-in-differences；
6. Local RR median/P90/P95、`>2/>5 bpm`，逐 subject 指标与 7-subject macro；
7. 指标分母、完成 epoch/update、选中 epoch、formal wall-time、显存、参数、局部 MAC 与 P2 benchmark；
8. 五指标 tolerance-aware Pareto 集合、推荐结构 `e8_direct_temporal` 相对参照的判断及边际对比分类。

材料性阈值固定为四项 error 相对 0.5%、PCC 绝对 0.002。三个 seed 只描述优化随机性；不构造窗口级或 seed-level 人群显著性，不生成跨指标总分。

## 3. 产物

输出根：

```text
runs/e8_film_decoder_redesign_v1/summary/{attempt}
```

至少包含：

```text
seed_primary_metrics.csv
arm_primary_summary.csv
arm_reference_comparison.csv
planned_contrasts_by_seed.csv
planned_contrasts_across_seed.csv
local_rr_tail_summary.csv
local_rr_tail_across_seed.csv
subject_stratified_metrics.csv
subject_macro_by_seed.csv
subject_macro_across_seed.csv
metric_denominators.csv
parameter_compute_memory.csv
decision.json
source_manifest.json
summary_receipt.json
access_receipt.json
source_code.json
manifest.json
freeze_receipt.json
```

汇总 lifecycle 排他执行；成功后同一 lock/amendment 身份拒绝重复汇总，失败现场保留。

## 4. 用户命令

该阶段只读取已冻结 validation 产物并在 CPU 回放合同，不训练、不推理、不读取 research-test：

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/e8_film_decoder_redesign_v1

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PYTHONPATH=. \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/summarize_e8_film_decoder_redesign_v1.py \
  --confirm-p5-summary
```

汇总会回载 72 个 best/final checkpoint 的配置、state 与 optimizer 合同，因此预计需要数分钟；不需要设置 `CUDA_VISIBLE_DEVICES`。

## 5. 后续边界

P5 成功冻结后，先依据 validation 预注册对比形成结论并固定全部 36 个 validation-selected checkpoints。Research-test 必须另立专项协议与 allowlist，并获得当次用户明确授权；test 不参与结构选择、阈值修改或 checkpoint 选择。
