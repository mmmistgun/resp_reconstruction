# Center-30 上下文效应独立测试集协议

日期：2026-09-04

协议 ID：`paper-center30-context-research-test-v1-20260904`

状态：**用户已授权完整 8 arms × 3 seeds 独立测试集评价；validation 结论与 24 个 checkpoint 已先行冻结。当前只开放 P4-ST1 test input-only W cache 构建，等待用户手动执行；评价入口需在 cache manifest SHA-256 冻结后实现。**

## 1. 目的与证据边界

本附件只检验 center-30 validation 结论能否在独立测试集上复现：对固定中心 30 s 输出，30→45 s 是否稳定带来
RR 收益，以及 45→60/90 s 是否存在稳定追加收益。它继续是游离于论文主模型实验之外的窗口长度辅助任务，不回写
W0/W3/C201/RTM 的既有选择、Pareto 或论文主模型定位。

独立测试集与本任务 train/validation 在 subject/session 上严格隔离，且没有参与 center-30 模型、epoch、checkpoint、
长度或结论分支选择。该数据集曾用于仓库内其他已关闭研究问题，因此这里的“独立”指相对本 center-30 任务的数据隔离，
不表示整个研究历史中的首次访问。

Test 只能确认或否定 validation 结论的外推；不得据 test：

- 重选 checkpoint、模型、长度、seed 或指标；
- 修改 `Pi_30`、loss、RR selector、材料性阈值或 W scale mapping；
- 只保留 test 表现较好的臂；
- 启动补训、微调或新增候选。

## 2. 冻结前置证据

Validation 矩阵已在任何 center-30 test signal/target/prediction 访问前闭合：

- 8 arms × 3 seeds，共 24/24 runs；
- checkpoint 固定为每项完整 validation 2675 rows 上由 `center30_rr_mae_bpm` 严格更小选择的
  `checkpoint_best_center30_rr.pt`；
- 三 seed summary receipt SHA-256=
  `2c533117c735e5bedb65f31ca77fa9dd22663433c0a284b4ee7bd9dc0beee2ef`；
- 三 seed artifact manifest SHA-256=
  `b18bbed8ab826b8a6e415866c4ad1b9d5171fd4eebc0d5c4090544428fc19bba`；
- 冻结 validation 结论为：45 s 是相对 30 s 在两种表征、3/3 seeds 均有材料性 RR 改善的最短输入；
  60/90 s 没有稳定追加 RR 收益。

Test 矩阵固定为以下 24 个 validation-selected checkpoints，不得缩减：

| 模型 | 输入 | seeds | P4-S2/P4-S3 checkpoint 来源 |
|---|---:|---|---|
| C201-center30 | 30/45/60/90 s | 20260811/20260812/20260813 | 8 项长度×seed |
| W-reduced-center30 | 30/45/60/90 s | 20260811/20260812/20260813 | 8 项长度×seed |

其中 seed `20260811` 来自 `runs/paper_evidence_v1/center30_context/p4s_single_seed/`，另外两个 seed 来自
`runs/paper_evidence_v1/center30_context/p4s_additional_seeds/`。评价前必须逐项复核 lifecycle、formal commit、resolved
config、selected epoch、checkpoint SHA-256 与 validation summary 输入链。

## 3. Test 数据身份

固定沿用 research-v2 admitted test rows：

- dataset index SHA-256=`f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f`；
- split=`test`，2310 windows，8 个 `samp_id`；
- `dataset_row_id` SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`；
- sample strategy=`stratified_random`，sample seed=`20260612`；完整 split 下该 seed 只作身份记录；
- input key=`bcg_rawish_segment_soft_z_key`；target key=`target_waveform_segment_soft_z_key`；
- 四种输入仍从同一 180 s parent 裁 `[7500,10500) / [6750,11250) / [6000,12000) /
  [4500,13500)`；共同 target 固定为 `[7500,10500)`。

Test 评价必须证明 2310 rows 在 24 项之间逐项相同、无重复、无缺失且与非 test row ID 零交集。不得按 test
质量、有限性、IBI eligibility 或模型输出过滤 row；prediction 非有限使对应评价失败。

## 4. P4-ST1：test input-only W cache

W-reduced 的四个输入长度需要与训练完全相同的 49-scale length-specific transform。为避免对 24 个 checkpoint 重复
计算 CWT，先构建一份四长度联合 test cache；C201 不读取该 cache。

固定合同：

- 只读取 2310 个 test BCG input，不读取 target array；
- 不读取 train/validation signal、target 或 prediction；
- 每个 parent input 只加载一次，再分别裁 30/45/60/90 s 并独立执行 direct scale mapping；
- `ssqueezepy==0.6.6`、Morlet `mu=13.4`、原 12-voice 网格偶数 indices、49 scales、reflect padding、
  `log1p(abs(CWT))`、每 50 点平均均与训练一致；
- feature shape 固定为 `[2310,49,60/90/120/180]`、float32、finite；
- 四个长度分别记录实际 mapped centers、duplicate count、transform spec/hash 与文件 SHA-256；
- lifecycle 和输出目录禁止覆盖；失败 identity 保留。

固定输出目录为：

```text
runs/paper_evidence_v1/center30_context_research_test_w_cache/
  6221ac4d2f9f4b0840bfe0ba1be9576b43794547792350c8cee0b6c695f38aaa/
```

由用户从本阶段实现提交后的干净 commit 手动执行：

```bash
./.venv/bin/python scripts/build_paper_center30_context_research_test_w_cache_v1.py \
  --confirm-research-test-cache-build
```

该步骤是一次完整 test cache 构建，预计执行 2310×4 次 CWT。它不产生模型结果，也不读取 test target。完成后必须先
只读验收 cache manifest、全部文件 hash、row identity、shape/finite 与 access flags，再冻结 manifest SHA-256。

## 5. P4-ST2：完整 24-checkpoint test inference

P4-ST2 仅在 P4-ST1 cache 验收并冻结 manifest SHA-256 后开放。评价实现必须：

- C201 直接读取 test BCG parent 并裁对应输入；W-reduced 额外读取冻结 test W cache；
- 每个 checkpoint 恢复其原模型、seed 与 validation-selected epoch，不读取 final checkpoint；
- 只进行 inference，不训练、不更新权重、不重选 checkpoint；
- 完整输出 2310 rows 的五项 `center30_*` primary metrics、IBI coverage、interpretable fraction 与身份列；
- 使用同一 `Pi_30`、±0.30 s lag、2 bpm RR bin、5-point envelope、退化值与 finite 规则；
- 每个 model×length×seed 使用不可覆盖的独立 evaluation identity，允许分配到两张 GPU，但进程内均为逻辑
  `cuda:0`；
- 24/24 全部完成前不得汇总或据中间结果修改队列。

预计计算成本为 24×2310 次模型推理。具体双 GPU 队列在 evaluator、cache path/hash 和 checkpoint allowlist 全部冻结后
给出；在此之前不访问 test target。

## 6. P4-ST3：冻结汇总与结论

汇总固定逐 seed 报告，再按三 seed计算 arithmetic mean ± sample SD (`ddof=1`)；按 seed 配对报告 30→45、
30→60、30→90、45→60、60→90 的相对变化、方向计数与材料性计数。PCC 同时报告绝对差；IBI 同时报告 coverage、
interpretable fraction 与 eligible count。不构造加权总分、Pareto、p-value 或唯一模型排名。

Test 结论相对 validation 先验解释：

- 若 30→45 在两个模型均保持稳定材料性 RR 改善，且 45→60/90 无稳定追加收益，则确认 45 s 是当前矩阵内
  RR 优先的合理输入下限；
- 若 30→45 不能跨模型、seed 复现，则报告 validation 窗口结论未外推，不用 test 结果回头修改 validation 选择；
- 若更长输入出现稳定新增收益，则报告 test 与 validation 的长度响应差异，不能事后改称 validation 已支持更长窗口；
- 其他指标只作多轴描述，不改变 RR 优先的预注册判据。

任何结果分支都不开放新增训练或新的独立测试集搜索。
