# Center-30/Center-60 上下文效应独立测试集协议

日期：2026-09-04

协议 ID：`paper-context-length-research-test-v1-20260904`

状态：**P4-T1–T3 均已完成并冻结。42/42 个 validation-selected checkpoints 已完成独立测试集评价，两个任务分别汇总且未作跨任务绝对指标比较。Center-60 test 支持“延长输入没有稳定 RR 收益”；center-30 test 未复现 validation 中 30→45 的 3/3 seed 稳定性，因此不能把 45 s 冻结为 test-confirmed 下限。该附件关闭，不据 test 重选或追加训练。**

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

## 7. P4-T2 执行与只读验收

用户从统一干净 commit `52d723891e34ad5051f18bdf93f2b0fbe4eb9d7f` 将 42 项分配到两张同型号
NVIDIA GeForce RTX 4070 Ti SUPER，进程内均为逻辑 `cuda:0`、可见设备数 1，物理卡仅由
`CUDA_VISIBLE_DEVICES=0/1` 区分。Center-30 24/24、center-60 18/18 lifecycle 均为 `complete`，共生成
`42×2310=97,020` 条逐 sample test metrics。

只读验收逐项确认：

- 42 项 identity 与两份冻结 validation summary 锁定的 checkpoint path/SHA-256、selected epoch、variant、seed、
  input/output length 完全一致，只加载 best-RR checkpoint，未读取 final checkpoint；
- 每项 artifact manifest 与 receipt 登记的全部文件 size/SHA-256 均匹配，evaluation commit、resolved evaluation
  config、依赖、runtime、命令与 access flags 无漂移；
- 每项均为相同 2310 个 test row IDs、8 个 `samp_id`、无重复且 non-test overlap=0；逐 sample summary 可由 metrics
  重算复现，prediction failure 没有被静默过滤，IBI 不可解释样本继续由 coverage/eligible-count 口径显式登记；
- C201 不读取 W cache，W-reduced 只读取 P4-T1 冻结 cache；执行中不训练、不重选 checkpoint。

## 8. P4-T3 冻结结果

只读汇总入口 `scripts/summarize_paper_context_length_research_test_v1.py` 从干净 commit
`241d9ba96373dfd8fa3ff1d71782f7a8f9d7e926` 执行，固定输出为：

```text
runs/paper_evidence_v1/context_length_research_test_summary/
```

`summary_receipt.json` SHA-256=`84ac2153f7e30ece4838dfb37645ca5b261a788fbbb6a09f8524b6119100f937`，
`artifact_manifest.json` SHA-256=`d69d47a80e514a9556000296938dff642d2642656c61ecff82d57c1a1baeeb66`。
汇总只读取已生成的 test metrics/summary 与 provenance JSON/YAML，不读取 test signal/target array、checkpoint 内容或
dataset/index，不执行 inference/GPU。统计单位为 seed；表中为三个 seed 的算术均值 ± sample SD。

### 8.1 Center-30 panel

| 模型 | 输入 | RR MAE bpm ↓ | IBI MedAE s ↓ | trajectory MAE ↓ | global error ↓ | signed PCC ↑ | IBI coverage | interpretable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C201-center30 | 30 | 0.82176±0.04912 | 0.13109±0.00226 | 0.10550±0.00240 | 0.14108±0.00257 | 0.87066±0.00078 | 0.78574±0.00754 | 0.62554±0.01061 |
| C201-center30 | 45 | 0.79427±0.04264 | 0.13101±0.00316 | 0.10886±0.00351 | 0.14361±0.00400 | 0.87544±0.00480 | 0.79598±0.00768 | 0.64228±0.01318 |
| C201-center30 | 60 | 0.84524±0.05814 | 0.12674±0.00365 | 0.10222±0.00326 | 0.13850±0.00318 | 0.87482±0.00345 | 0.78731±0.01563 | 0.63319±0.01877 |
| C201-center30 | 90 | 0.75694±0.04787 | 0.12542±0.00281 | 0.10299±0.00521 | 0.13767±0.00469 | 0.87797±0.00120 | 0.80116±0.00549 | 0.64675±0.00510 |
| W-reduced-center30 | 30 | 0.79881±0.01938 | 0.13069±0.00416 | 0.10120±0.00279 | 0.14137±0.00265 | 0.87357±0.00218 | 0.79406±0.00879 | 0.64286±0.01652 |
| W-reduced-center30 | 45 | 0.77627±0.01461 | 0.12907±0.00342 | 0.10110±0.00143 | 0.13804±0.00278 | 0.87781±0.00258 | 0.79655±0.00462 | 0.64416±0.00766 |
| W-reduced-center30 | 60 | 0.76916±0.03502 | 0.12726±0.00546 | 0.09856±0.00223 | 0.13660±0.00229 | 0.87722±0.00258 | 0.79220±0.01140 | 0.63694±0.01342 |
| W-reduced-center30 | 90 | 0.74862±0.01543 | 0.12397±0.00117 | 0.09828±0.00121 | 0.13817±0.00282 | 0.87993±0.00101 | 0.79480±0.00236 | 0.64214±0.00968 |

RR paired-seed 有向相对变化如下，正值表示改善；括号为达到 `0.5%` 材料改善的 seed 数：

| 模型 | 30→45 | 30→60 | 30→90 | 45→60 | 60→90 |
|---|---:|---:|---:|---:|---:|
| C201-center30 | +3.0408% (2/3) | −2.8201% (0/3) | +7.7433% (3/3) | −6.7407% (1/3) | +10.2245% (3/3) |
| W-reduced-center30 | +2.7733% (2/3) | +3.6817% (2/3) | +6.2179% (3/3) | +0.9411% (1/3) | +2.5473% (2/3) |

30→90 的五主指标变化为：C201 的 RR/IBI/trajectory/global/ΔPCC 为
`+7.7433% / +4.3255% / +2.3252% / +2.3942% / +0.007312`；W-reduced 为
`+6.2179% / +5.0908% / +2.8342% / +2.2611% / +0.006356`。但 RR 的 30→45 仅在两种模型中各
2/3 seeds 达到材料改善，30→60 又出现明显表征分歧。故 test 支持“30 s 相对 90 s 不足”，但没有复现 validation
中使 45 s 成为稳定最短下限的证据，也不能改为宣称 60 s 是下限；当前矩阵只说明 center-30 的最短合理边界仍未由
独立测试集确认。90 s 在两个模型的 test 三-seed RR 均值最低只是描述性结果，不用于事后长度重选。

### 8.2 Center-60 panel

| 模型 | 输入 | RR MAE bpm ↓ | IBI MedAE s ↓ | trajectory MAE ↓ | global error ↓ | signed PCC ↑ | IBI coverage | interpretable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C201-center60 | 60 | 0.68373±0.04035 | 0.11185±0.00273 | 0.13022±0.00396 | 0.17183±0.00484 | 0.87349±0.00555 | 0.79396±0.00684 | 0.62929±0.00806 |
| C201-center60 | 90 | 0.70554±0.05123 | 0.10891±0.00295 | 0.13594±0.00245 | 0.18382±0.00394 | 0.87647±0.00303 | 0.80079±0.00680 | 0.64156±0.01541 |
| C201-center60 | 180 | 0.76029±0.01187 | 0.11188±0.00111 | 0.13083±0.00293 | 0.17186±0.00309 | 0.87584±0.00298 | 0.78934±0.00766 | 0.63434±0.01054 |
| W-reduced-center60 | 60 | 0.67267±0.04451 | 0.11441±0.00737 | 0.12637±0.00445 | 0.17587±0.00403 | 0.87380±0.00389 | 0.79594±0.00495 | 0.62915±0.00656 |
| W-reduced-center60 | 90 | 0.70507±0.01718 | 0.10940±0.00056 | 0.12520±0.00210 | 0.17726±0.00088 | 0.87690±0.00216 | 0.80229±0.00312 | 0.64055±0.00565 |
| W-reduced-center60 | 180 | 0.74842±0.04107 | 0.11112±0.00205 | 0.12751±0.00404 | 0.17858±0.00423 | 0.87283±0.00621 | 0.78922±0.01479 | 0.62165±0.02576 |

| 模型 | 60→90 RR | 90→180 RR | 60→180 RR |
|---|---:|---:|---:|
| C201-center60 | −3.3010% (2/3 材料改善) | −8.0812% (0/3) | −11.4546% (0/3) |
| W-reduced-center60 | −5.2009% (1/3 材料改善) | −6.1549% (0/3) | −11.7991% (1/3) |

60→180 的五主指标变化为：C201 的 RR/IBI/trajectory/global/ΔPCC 为
`−11.4546% / −0.0760% / −0.5163% / −0.0509% / +0.002354`；W-reduced 为
`−11.7991% / +2.6813% / −1.0340% / −1.5549% / −0.000968`。C201 的 60→180 RR 为
3/3 seeds 方向和材料性恶化；W-reduced 为 2/3 恶化、1/3 改善。90→180 RR 在两个模型中均 3/3 方向恶化。
因此独立 test 支持 validation 的核心描述：延长输入没有稳定 RR 收益；在固定中心 60 s 输出和当前两个模型内，60 s
是 RR 优先的合理选择。该判断不表示五项指标逐项最优，也不外推为所有输出任务或论文主模型的统一窗口结论。

## 9. 关闭状态

P4-T 现已关闭：不依据 test 结果重选 checkpoint、seed、模型或长度，不追加训练或新的 test 搜索。Center-30 与
center-60 的绝对指标继续保持分 panel；`180→180` 历史结果只作既有外部参照，不纳入本附件统一统计。
