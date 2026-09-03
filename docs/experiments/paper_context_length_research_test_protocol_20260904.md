# Center-30/Center-60 上下文效应独立测试集协议

日期：2026-09-04

协议 ID：`paper-context-length-research-test-v1-20260904`

状态：**用户已授权 center-30 与 center-60 两种输出任务共 42 个 validation-selected checkpoints 的完整独立测试集评价。P4-T1 联合五长度 input-only W cache 已完成并冻结；P4-T2 evaluator、42-checkpoint allowlist 与定向 CPU 测试已实现，等待用户从统一干净 commit 手动执行。尚未执行模型 inference 或读取 test target。**

## 1. 科学问题与证据边界

本附件检验两个已经冻结的 validation 上下文效应能否外推到独立测试集：

1. center-30：固定输出中心 30 s，输入为 30/45/60/90 s；重点复核 30→45 s 的 RR 收益，以及继续延长是否有稳定收益；
2. center-60：固定输出中心 60 s，输入为 60/90/180 s；重点复核更长输入没有稳定 RR 收益的 validation 描述。

两种任务的输出长度、FFT 分辨率、包络轨迹点数、loss 与指标命名不同，必须作为两个独立 panel 汇总。只允许比较各任务
内部的 paired-seed 相对变化；不得把 center-30 与 center-60 的绝对指标合并成一条窗口曲线、总分或统一排名。

独立测试集与两个任务的 train/validation 在 subject/session 上严格隔离，且未参与模型、epoch、checkpoint、长度矩阵或
validation 结论选择。该 split 曾用于仓库内其他关闭任务，因此“独立”只指相对本次两项上下文任务的数据隔离。

Test 不得用于重选 checkpoint、模型、长度、seed、指标或阈值，也不得触发补训、微调、删臂或新增候选。

## 2. 冻结矩阵：42 checkpoints

| 任务 | 模型 | 输入 | seeds | checkpoints |
|---|---|---|---|---:|
| center-30 | C201-center30 | 30/45/60/90 s | 20260811/12/13 | 12 |
| center-30 | W-reduced-center30 | 30/45/60/90 s | 20260811/12/13 | 12 |
| center-60 | C201-center60 | 60/90/180 s | 20260811/12/13 | 9 |
| center-60 | W-reduced-center60 | 60/90/180 s | 20260811/12/13 | 9 |

共 42 个 validation-selected checkpoints，全部使用各原任务冻结的 RR selector：

- center-30 validation receipt/manifest SHA-256=
  `2c533117c735e5bedb65f31ca77fa9dd22663433c0a284b4ee7bd9dc0beee2ef / b18bbed8ab826b8a6e415866c4ad1b9d5171fd4eebc0d5c4090544428fc19bba`；
- center-60 validation receipt/manifest SHA-256=
  `7a8e5a3118e5059fc46ff287f4553952f2fea44653501c7ddedfd09145da9cb9 / c0806893d1b4350da46fb9bc02d2056a8b6c8e29d368205709be00aae9bacb2a`。

评价实现必须逐项冻结 checkpoint path、SHA-256、formal commit、seed、variant、input/output length 与 selected epoch。
42/42 完成前不得汇总或根据中间 test 结果修改队列。

## 3. Test 数据身份

- dataset index SHA-256=`f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f`；
- split=`test`，2310 windows，8 个 `samp_id`；
- row-ID SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`；
- sample strategy=`stratified_random`、sample seed=`20260612`，完整 split 下仅作身份记录；
- input key=`bcg_rawish_segment_soft_z_key`，center-30 target slice=`[7500,10500)`，center-60 target
  slice=`[6000,12000)`。

四十二项评价必须使用相同 2310 个 row IDs，无重复、无缺失，并与 train/validation row IDs 零交集。不得按 test 质量、
IBI eligibility、有限性或模型输出过滤样本；prediction 非有限使对应评价失败。

## 4. P4-T0/T1：联合五长度 input-only W cache

W-reduced 两项任务需要输入长度 30/45/60/90/180 s 的 49-scale W 表征。60/90 s 在两项任务中的 BCG slice 与
transform spec 逐字段相同，因此只计算和保存一份；C201 不读取 W cache。

P4-T1 固定：

- 每个 test parent BCG 只加载一次，依次裁 30/45/60/90/180 s；
- 每个长度独立执行 direct scale mapping、Morlet `mu=13.4`、49 scales、reflect padding、
  `log1p(abs(CWT))` 与每 50 点平均；
- shape=`[2310,49,60/90/120/180/360]`，float32、finite；
- 共执行 `2310×5=11,550` 次 CWT，受管 feature 约 `366,735,600 bytes`（约 350 MiB）；
- 只读取 test BCG input，不读取 test target，不读取 train/validation signal，不执行模型 inference；
- 输出目录、lifecycle 与文件均禁止覆盖，失败 identity 保留。

旧四长度联合实现存在运行成本错误：底层 scale mapping 的 LRU 只能容纳 3 个长度，会在逐 row 的 4/5 长度循环中反复
执行 scale search。P4-T0 将该纯性能缓存容量从 3 扩为 5；transform 输入、scales、frequencies 与 feature 数值不变，
并由定向测试固定共享 60/90 s 的 slice/spec identity。

固定输出目录：

```text
runs/paper_evidence_v1/context_length_research_test_w_cache/
  cf89c6e678bb243801c0ca577ec14d69d724603e51b3335c4e1f474e7f999d5c/
```

用户从 P4-T0 实现提交后的干净 commit 手动执行：

```bash
./.venv/bin/python scripts/build_paper_context_length_research_test_w_cache_v1.py \
  --confirm-research-test-cache-build
```

完整 cache 完成后先只读复核 manifest、全部文件 hash、row identity、shape/dtype/finite、frequency identity 与 access
flags，并冻结 manifest SHA-256；P4-T2 evaluator 在此之前不实现、不访问 test target。

P4-T1 已由用户从干净 commit `f87efd927a87c018888a06b7927cf50df3ada676` 完成，耗时 `03:49`。冻结目录为：

```text
runs/paper_evidence_v1/context_length_research_test_w_cache/
  cf89c6e678bb243801c0ca577ec14d69d724603e51b3335c4e1f474e7f999d5c/
```

`cache_manifest.json` SHA-256=`9b475926258d129851fb7b9c10d2b342ac2e833b121842b18437585059bf1b98`。
只读验收确认 lifecycle=`complete`、2310 rows、8 samp_id、row-ID SHA-256 固定、非 test row overlap=0；五组
features 的 shape 分别为 `[2310,49,60/90/120/180/360]`，均为 float32/finite。全部 11 个受管 `.npy`
文件的 size/SHA-256、五组 frequency identity、duplicate count 与 nominal error 均与 manifest 一致。Access flags 确认
只读取 test BCG input，未读取 test target、train/validation signal，未执行训练或 inference。

## 5. P4-T2：完整 42-checkpoint inference

Cache 已冻结；下一步实现一个受控 evaluator：

- C201 从 test parent 裁对应输入；W-reduced 读取同一冻结联合 cache；
- center-30 与 center-60 分别恢复原模型和指标 evaluator；
- 只加载 `checkpoint_best_*rr.pt`，不加载 final checkpoint；
- 每个 checkpoint 输出 2310 rows 的原任务五项 primary、IBI coverage/interpretable fraction 与身份列；
- 只进行 inference，不训练、不更新权重、不重选 checkpoint；
- 每项使用不可覆盖的独立 evaluation identity；允许分配到两张 GPU，进程内均使用逻辑 `cuda:0`。

预计成本为 42×2310 次模型推理。具体双 GPU 队列只在 cache path/hash、checkpoint allowlist、输出 schema 与 evaluator
定向测试全部冻结后给出。

P4-T2 固定入口为：

```text
scripts/eval_paper_context_length_research_test_v1.py
```

每次调用必须显式给出 `task / model / input-sec / seed / --confirm-research-test`。入口只接受第 2 节的 42 项 identity，
从两份冻结 validation summary 的 inputs 链到每项 lifecycle、artifact manifest、resolved config、selected epoch 与
checkpoint SHA-256；最终 checkpoint 文件在 inference 前再次核验 size/hash。进程内 device 固定为逻辑 `cuda:0`，物理卡
只通过 `CUDA_VISIBLE_DEVICES` 选择，并将该环境变量、逻辑设备、可见设备数和 GPU 型号写入 runtime identity。

每项输出到独立且不可覆盖的：

```text
runs/paper_evidence_v1/context_length_research_test/<task>/<experiment_id>/seed_<seed>/
```

输出包含 resolved evaluation config、checkpoint/data/runtime identity、2310-row metrics、metrics summary、evaluation
receipt、artifact manifest 与 lifecycle。C201 不打开 W feature；W-reduced reader 固定 cache path 与 manifest SHA-256。
任何失败保留 lifecycle 并使相应 shell `&&` 队列停止。

P4-T2 新增 5 项定向 CPU 测试，覆盖 42 项 allowlist 完整性、两种 target crop、W row-ID 注入、checkpoint payload
selected-epoch identity 与 CLI confirmation gate；测试只读 validation provenance/checkpoint 文件身份，不读取 checkpoint
内容、test signal/target，不执行模型 inference。

## 6. P4-T3：分任务冻结汇总

每项任务逐 seed 报告，再计算 arithmetic mean ± sample SD (`ddof=1`) 和 paired-seed 相对变化。Center-30 固定
30→45、30→60、30→90、45→60、60→90；center-60 固定 60→90、90→180、60→180。Error 材料阈值为相对
`0.5%`，PCC 为绝对 `0.002`；IBI 同时报告 coverage、interpretable fraction 与 eligible count。

Center-30 test 用于确认或否定“45 s 是 RR 优先合理下限”；center-60 test 用于确认或否定“延长输入没有稳定 RR
收益”。其他指标作多轴描述，不覆盖 RR 优先判据。不构造跨任务绝对比较、加权总分、p-value、Pareto 或唯一模型排名。

任何结果分支均不开放新增训练或新的 test 搜索。
