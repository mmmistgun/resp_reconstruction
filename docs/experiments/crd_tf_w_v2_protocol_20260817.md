# CRD-TF-W v2 机制、频带与效率实验协议（修订版）

日期：2026-08-17

状态：**P0、P−1 与 FiLM statistics correction 均已完成并冻结，P2 关闭。P1 三臂与 P3 D4 的全部 stress/formal 均已完成并核验；W1/W2/D4 不通过严格候选门槛，W3 通过质量门槛但不通过效率门槛。P4 专用 summarizer、artifact/hash 审计、严格 eligibility 与描述性 trade-off Pareto 已实现并通过真实 15-run 只读预审；当前唯一下一步是从干净 commit 一次性生成 P4 summary，P5 与 research-test 仍关闭。**

## 1. 权威性、替代关系与证据边界

本文是 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 49 节引用的规范性附件；若发生冲突，以主协议为准。

本文取代 `docs/experiments/crd_tf_w_v2_protocol_20260816.md`。旧附件和 `docs/temp/实验计划20260817.md` 只保留为设计演化记录，不得作为执行依据。

本阶段继续属于 **research-test-informed development**。既有 research-test 已被多次访问并影响研究方向；任何后续结果只能称为 reused research/development evidence，不得表述为未触碰、独立或无偏 held-out 证据。

本修订版明确暂不进行 `samp_id` 分析：

- 不计算逐 `samp_id` candidate-minus-anchor delta；
- 不设置 `5/7 samp_id` 同方向门槛；
- 不执行 leave-one-`samp_id`-out；
- `samp_id` 只保留在逐 sample 产物中用于身份追溯，不参与汇总、晋级或研究结论。

正式数据集汇总继续沿用冻结的逐 sample direct mean；三个训练 seed 用于描述优化随机性和 paired-seed 方向，不构造加权总分或确认性统计推断。

## 2. 本阶段只回答的问题

当前统一 anchor 为冻结的 `crd_tf102_w`，记为 `W0_FULL_12V_FILM_D6`。本阶段只回答四个问题：

1. 冻结 W checkpoint 在推理时对 FiLM gamma、beta、频带组成和时间对齐的功能依赖是什么；
2. 重新训练时，W 的收益主要来自名义目标 `0.03125–0.80 Hz`、`(0.80,8.00] Hz`，还是二者互补；实际 cache mapped centers 按 P0 锁定的 `≤0.80` / `>0.80 Hz` 分割；
3. 将 12 voices/octave 降为 6 voices/octave 能否在基本保持质量的同时产生可测效率收益；
4. 在完全相同的 full-band 12V FiLM W 设置下，4 层 Local BiMamba2 能否作为现有 6 层结构的效率候选。

以下内容不属于本轮标准范围：

- Morlet effective-Q、mother wavelet、24/48 voices、pooling 或边界搜索；
- residual concat、matched temporal concat control、gate、attention 或 local cross-attention；
- D8、其他深度、D6 capacity-fill 或深度×表示组合；
- 将 RESP/CARRIER/6V、ADD/SCALE 与 D4 组合成一个未经训练验证的复合模型；
- TCN+decoder、其他 decoder 或 100-Hz placement；
- `samp_id`/subject-balanced 汇总、bootstrap、p-value 或显著性检验。

这些方向不会因某个标准 arm 数值不理想而自动开放；若未来需要，必须另立协议。

## 3. 冻结证据与禁止重复项

### 3.1 直接复用

- C201：`crd_c201_decoder_10hz_cap × seeds 20260811/12/13`；
- W0：`crd_tf102_w × seeds 20260811/12/13` 的 validation-selected checkpoints；
- P5 的 C201、M/W/MS、CTRL1、capacity、interaction 与 Pareto 结果；
- P6a 的 MWS additive/gate 负结果；
- C1 的六层 Local BiMamba2 与 parameter-matched TCN 结果；
- C2 的 decoder capacity/placement 结果；
- 现有 Morlet calibration、train/validation W cache 与 reused research-test cache/summary。

W0 三个 selected epoch 为 `13 / 15 / 14`。在 `patience=30` 上回放不会改变三个 Local-RR-selected checkpoint，因此 W0/D6 anchor 不重训。

### 3.2 不重复或不旁路

- 不重复 W0 vs C201、W0 vs CTRL1 或原 M/W/S/MS 矩阵；
- 不重复 P6a gate、MWS additive 或旧 pair/triple；
- 不重新打开 TCN、100-Hz placement、其他 decoder 或已关闭 CRD 队列；
- 不调用普通 `eval_crd.py` 读取 test；它继续保持 validation-only；
- 不覆盖现有 cache、checkpoint、metrics、summary、manifest 或 research-test 产物。

## 4. 公共训练、评价与工程口径

除本协议明确改变的单一因素外，所有新训练 arm 冻结为：

```text
dataset / admission / split / target = 现有 CRD-TF v1
input = BCG 100 Hz × 180 s
loss = L_sync + 0.25 L_effort
five primary metrics = 现有冻结定义
dataset aggregation = sample direct mean
checkpoint selector = full-validation Local RR minimum
optimizer = AdamW
max_lr / min_lr = 3e-4 / 3e-5
effective batch = 128
physical batch = 128, accumulation = 1
max_epochs = 80
early stopping = disabled；固定完成 80 epochs
formal seeds = 20260811 / 20260812 / 20260813
AMP = bf16
```

一旦某个 arm 进入 formal 队列，必须完成三个 seed。seed `20260811` 的普通 validation 数值不得用于取消另两个 seed。只有 OOM、非有限、checkpoint/lifecycle 不完整、prediction collapse 或 config/cache identity 错误可以中止；这些属于工程/协议失败，不是效果筛选。

### 4.1 质量候选门槛

四个 error primary 为 Whole RR、Local RR、envelope trajectory 和 global envelope modulation；均越低越好。对任一 error $E$，相对 anchor 的变化定义为：

$$
\Delta_E=\frac{E_{candidate}-E_{anchor}}{E_{anchor}}\times100\%.
$$

Signed PCC 使用绝对差：

$$
\Delta_{PCC}=PCC_{candidate}-PCC_{anchor}.
$$

质量候选必须同时满足：

1. 至少一项 primary 达到实质改善：某个 error 的 $\Delta_E\leq-0.5\%$，或 $\Delta_{PCC}\geq0.002$；
2. 被声明改善的 primary 至少 `2/3` paired seeds 同方向；
3. Local RR 的 $\Delta_E\leq0.5\%$；
4. Whole RR、trajectory、global envelope 各自的 $\Delta_E\leq1.5\%$；
5. $\Delta_{PCC}\geq-0.003$；
6. 生命周期完整、全部正式数值 finite、prediction degeneracy 为 0。

### 4.2 效率候选门槛

效率候选必须同时满足：

1. 四个 error primary 各自的 $\Delta_E\leq1.0\%$；
2. $\Delta_{PCC}\geq-0.003$；
3. 具有预注册的结构缩减：W3 active scale elements 至少减少 40%，或 D4 trainable parameters 至少减少 20%；
4. 相对相同测量环境的 W0，warm-up 后稳态 throughput 提高至少 10%，或 peak allocated 降低至少 15%；
5. 生命周期完整、全部正式数值 finite、prediction degeneracy 为 0。

### 4.3 Catastrophic failure 线

`3% / 0.005` 只作明显失败线：任一 error primary 的三-seed mean 恶化超过 3%，或 PCC 下降超过 0.005，候选不得进入质量或效率池。它不能替代第 4.1/4.2 节更严格的最终门槛。

所有比较均报告三个 seed、三-seed mean ± sample SD 和 paired-seed 方向。不构造加权总分；若多个候选互不支配，保留 tolerance-aware Pareto set。

## 5. P0：candidate lock 与实现前冻结

P0 不训练、不推理、不构建 cache，只完成只读身份审计和执行契约冻结。必须登记：

- C201、W0 和 CTRL1 三 seed checkpoint/config/manifest/validation summary 的路径、大小和 SHA-256；
- 当前 W cache manifest、row identity、shape、dtype、频率数组和 SHA-256；
- 当前代码中 FiLM raw projection、`0.5*tanh`、gamma/beta 拆分顺序和融合位置；
- W1/W2 mask、W3 scale index、P−1 干预的精确张量位置；
- W0/D6 与 D4 的 module seed、共享 state 和准确参数数；
- 新协议 ID、不可覆盖输出目录、summary schema、正式命令和 clean-commit preflight；
- 本协议各阶段的配置 allowlist 与禁止字段。

P0 不允许依据现有 validation/research-test 数值重新修改第 4 节门槛。若静态审计发现实际代码、cache 频率或 checkpoint 身份与本文假设不符，应停止并修订协议，不得凭猜测实现。

P0 已于 2026-08-17 完成，固定 candidate lock 为：

```text
docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json
size = 24,316 bytes
SHA-256 = 6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6
```

静态审计重新计算并通过 C201/W0/CTRL1 共 9 个 checkpoint 的 36 项 checkpoint/config/manifest/validation-summary size/hash；三个 seed selected epoch 分别为 C201 `10/11/13`、W0 `13/15/14`、CTRL1 `13/18/12`。当前代码对三个代表 checkpoint 均 strict-load，无 missing/unexpected key；同 seed W0/CTRL1 的 base 初始 state 与 C201 逐 tensor 相同。

P0 写回以下离散频率事实：原 `0.03125–8.00 Hz` 是 97 点 Morlet **名义目标网格**，现有 `w_frequencies_hz.npy` 的实际 mapped centers 为 `0.03662109375–7.99560546875 Hz`，有一个重复 mapped center。`0.80 Hz` 分割后 RESP 为 indices `0..55`、56 scales、实际 `0.03662109375–0.7965087890625 Hz`；CARRIER 为 indices `56..96`、41 scales、实际 `0.84228515625–7.99560546875 Hz`。W3 固定使用 indices `0,2,…,96`；12V 偶数目标频率与理论 6V 网格最大绝对差为 `1.7763568394002505e-15 Hz`，49 个 mapped centers 无重复。

参数静态审计固定 W0/C201/W-branch/CTRL1 为 `1,219,850 / 1,069,802 / 150,048 / 1,219,754`；每个 Local BiMamba2 block 为 `158,564`，D4 准确总参数为 `902,722`，相对 W0 减少 `317,128 = 25.9973%`。当前代码仍硬编码 W shape `[97,360]`、Local block count=6，且未实现 ADD/SCALE 与新 variant registry；这些是后续实现项，不允许被 P0 静态审计误写成已可运行。

P0 审计时 repository HEAD 为 `00037cd84704bc9132752125c97bbd820346af69`，dirty 内容仅为本轮协议文档，runtime code 无 dirty；未来实现验收与 formal 仍必须来自新的干净 commit。P0 全程未推理、未构建/修改 cache、未训练、未访问 research-test。

## 6. P−1：冻结 checkpoint 的无训练功能审计

P−1 只在 P0 锁定的三个 W0 checkpoint 上运行完整 validation；不训练、不修改 checkpoint、不读取 research-test。所有干预发生在冻结 W cache 载入后、进入 W encoder 或 FiLM 融合的规范位置。

### 6.1 FiLM 功能干预

P0 已确认当前代码与下式一致。设 projection 输出为 $\gamma_{raw},\beta_{raw}$：

$$
g=0.5\tanh(\gamma_{raw}),\qquad
b=0.5\tanh(\beta_{raw}),\qquad
z'=z(1+g)+b.
$$

每个 checkpoint 固定评价：

| ID | 干预 | 解释边界 |
|---|---|---|
| `FULL` | 原始 $g+b$ | 冻结基准 |
| `BETA_ONLY` | 有效 $g=0$ | 已联合训练模型对加性路径的功能依赖 |
| `GAMMA_ONLY` | 有效 $b=0$ | 已联合训练模型对尺度路径的功能依赖 |
| `CONDITION_OFF` | $g=b=0$ | 已联合训练主体对 W 条件的依赖；不等于 C201 |

逐 sample 只保存紧凑统计，不保存完整 latent：`mean/median |g|`、`mean/median |b|`、相邻时间帧 mean absolute difference，以及 `|g|≥0.49` / `|b|≥0.49` 的元素比例。统计只在 FULL 路径记录；BETA_ONLY/GAMMA_ONLY/CONDITION_OFF 复用同一 full-W raw gamma/beta，不重复保存。数据集仍按 sample direct mean 汇总，不做 `samp_id` 分析。

FiLM 干预属于冻结联合模型的功能敏感性证据，不是重新训练 ADD/SCALE 的因果效果。它只按第 6.4 节决定是否分配三个 P2 runs。

### 6.2 W 频率遮挡

对现有 `[97,360]` W cache 创建只读视图，shape 和 encoder 均不变：

```text
FULL       = 97 mapped centers, 0.03662109375–7.99560546875 Hz
RESP       = indices 0..55, mapped_frequency <= 0.80 Hz
CARRIER    = indices 56..96, mapped_frequency > 0.80 Hz
CARRIER_L  = indices 56..71, 0.80 < mapped_frequency <= 2.00 Hz  # 仅诊断
CARRIER_H  = indices 72..96, mapped_frequency > 2.00 Hz          # 仅诊断
```

`0.80 Hz` 固定归入 RESP。被遮挡 scale 在任何 W encoder normalization 之前置零；不得改变 cache、本体频率数组、encoder bias、通道或参数。

### 6.3 时间信息负对照

在 full-band W 上额外评价：

- `TIME_MEAN`：每个 scale 沿 360 帧取均值，再重复到 360 帧；
- `TIME_SHIFT_30S`：沿时间轴固定循环平移 60 帧，即 30 秒。

两者只用于判断冻结模型是否依赖局部时频演化与时间对齐。循环平移保留边界 wrap-around，不能解释成生理上真实的固定延迟，也不参与 P1/P2 自动选臂。

### 6.4 P2 唯一预算分配规则

对 `BETA_ONLY` 和 `GAMMA_ONLY`，分别相对同 checkpoint 的 FULL 计算第 4 节五项 primary。某项干预被称为 `quality-near`，必须同时满足：

- 三-seed mean 上四个 error primary 均不恶化超过 1.5%；
- 三-seed mean PCC 下降不超过 0.003；
- 至少 `2/3` checkpoints 分别满足同一组门槛；
- 全部输出 finite、无 prediction degeneracy。

固定分支为：

- `BETA_ONLY` near、`GAMMA_ONLY` 非 near：P2 只训练 ADD；
- `GAMMA_ONLY` near、`BETA_ONLY` 非 near：P2 只训练 SCALE；
- 两者都 near：P2 只训练机制更简单的 ADD；
- 两者都非 near：P2 不训练新融合，保留 FiLM。

不得在看到 P−1 数值后改变 near 门槛、增加第三个融合 arm，或把 P−1 结果表述成重新训练机制的优胜结论。

### 6.5 实现验收（2026-08-17）

P−1 已通过新增的独立包装器实现，不修改 P0 锁定的 `tf_v1_model.py`、既有 checkpoint 或 cache reader：

- `resp_train/crd/tf_w_v2_audit.py`：SHA-256=`42d3ebc7536979763cc306d5b7e71345f168388633af73bb9e455f5ec9ea6db5`；
- `scripts/eval_crd_tf_w_v2_functional_audit.py`：SHA-256=`62cee8677595f5351a3a771d422116273f64d4a759a2cbcff997a6625e139dd3`；
- `scripts/eval_crd_tf_w_v2_film_statistics_correction.py`：SHA-256=`6bd6de9770d597327c22818829283f1555f26aaae87fee010e81a608578328e8`；
- `tests/test_crd_tf_w_v2_audit.py`：SHA-256=`13d7d3db7a8040b1ca87f92cbf2c63a063deeb420f85af2a78e00ed516ef0515`。

入口强制 candidate-lock path/hash、36 项 anchor artifact identity、锁定 runtime identity、validation W cache/frequency/view identity、干净 Git、CUDA、`split=val`、三个 W0 checkpoint 与 10 项固定 intervention；不提供 test、max-windows、任意 checkpoint 或输出覆盖入口。所有 W 干预只作用于 `TfV1CacheReader.get` 已复制的 batch tensor，不写原 memmap。失败保留独立 `.incomplete_*` 目录和 `failure.json`，成功才原子重命名到固定 P−1 目录。

FULL 每个 seed 必须以绝对容差 `1e-6` 复现锁定 validation 五项 primary 与 degeneracy summary，否则整个 audit 失败。成功产物严格为 candidate lock 中的五项文件；预计逐 sample metric 行数为 `3×10×2675=80,250`，FULL FiLM statistic 行数为 `3×2675=8,025`。P2 decision 由第 6.4 节纯函数生成，不读取 `samp_id`、secondary 排名或 research-test。

定向命令 `./.venv/bin/python -m pytest -q tests/test_crd_tf_w_v2_audit.py tests/test_crd_tf_v1_model.py tests/test_crd_tf_v1_data.py tests/test_crd_experiment.py` 已通过，结果为 `42 passed`；同时完成 correction 相关 Python 文件的 `py_compile`、CLI help、candidate-lock/source-audit input verification。测试覆盖真实 W0 结构在替换 Local Mamba 为 Identity 后的 active-FiLM FULL wrapper 逐 tensor identity、相邻帧 mean absolute difference 的数值语义，以及 correction 对身份与六项非目标统计的保持；没有由 Codex 运行 GPU correction。

第一次完整 audit 从干净 commit `82926b2` 启动，在 seed `20260811` 的 FULL 锚点失败并按约定保留 `p_minus_1_validation_audit.incomplete_20260817T070706_900712Z/failure.json`；观测到锁定六项 summary 最大绝对差 `2.6775125796740795e-05`。随后用原生 `eval_crd.py` 对同一 checkpoint、cache 和 validation rows 复评，六项与冻结 summary 的差均严格为 `0.0`，排除历史 summary 或当前 CUDA 环境漂移。根因是旧包装器在 decoder 前执行 FiLM 统计算子，改变了原生 CUDA 执行/内存路径。修订后 FULL 直接调用未修改的 W0 原生 forward，只用只读 forward hook 捕获 gamma/beta，并在原生输出完成后计算统计；`1e-6` 锚点门槛不放宽。该失败不形成 P−1 科研结果，修订代码仍须从新的干净 commit 重跑完整 audit。

第二次完整 audit 从干净 commit `d91db6e2177fa8e4fea3ac27237cff2cdb8b9c35` 完成 3 checkpoints × 10 interventions。固定 source manifest SHA-256=`249c761799b1f8020a77ed51875985776718d9cf1e9701900ce5dae783492f3f`；80,250 行逐 sample metrics、8,025 行 FiLM statistics 与 40 行 seed/aggregate summary 的身份、行数、finite 和文件 hash 均通过。FULL 三 seed 的最大锚点差分别为 `2.7755575615628914e-17 / 8.326672684688674e-17 / 5.551115123125783e-17`。BETA_ONLY 与 GAMMA_ONLY 均非 quality-near，冻结 P2 decision=`retain_film_no_p2_training`、decision SHA-256=`6aae72dd32b09f0ac5c5a5151f78510ec292d7b2c8942feddd0b751bc0787f4a`；不训练 ADD/SCALE。

独立复核随后发现 source `film_statistics.csv`（SHA-256=`a785b2df38c5012e342e2400d2f817bc113421b9becdfac5f6783bbf0b04127a`）把协议要求的 `mean(abs(diff(g))) / mean(abs(diff(b)))` 错写为 `mean(diff(abs(g))) / mean(diff(abs(b)))`，导致 gamma/beta 两列分别有 `1,974 / 1,887` 个负值。该错误只影响两项描述性 FiLM 时间统计；统计在原生输出完成后计算，因此不影响 prediction、primary、频带/时间负对照或 P2 decision。原 source audit 目录保持不可改写。

修订后的独立 correction 入口固定校验 source manifest/film/decision hash，只运行三个 FULL validation checkpoints，并将 `film_statistics_corrected.csv` 与 correction manifest 原子写入 `runs/crd_tf_w_v2/p_minus_1_film_statistics_correction/`。它必须重验 FULL 锚点、8,025 行身份、全部 corrected 值 finite/nonnegative，以及六项非目标 FiLM 统计相对 source 的最大绝对差不超过 `1e-12`。

correction 随后从干净 commit `4eb9b3ce937792d151393a40c0f95b5cab0e7c9b` 完成，manifest SHA-256=`1c6f7218c280b2a1579b2169a2b4752d7c10416d89de67ed5e9a1cc58d104463`，`film_statistics_corrected.csv` SHA-256=`2b7a4edef8e9828356a336c3c1ec7880235914207b00d164bf016ce1cd7a5203`。FULL 三 seed 锚点差与 source audit 一致，8,025 行 identity 唯一且 finite，gamma/beta 时间差负值数均为 0；六项非目标统计最大绝对差不超过 `9.98e-17`。原 source audit 的 manifest/film/decision hash 复核未变，correction 未训练、未修改 checkpoint/cache/source、未访问 research-test、未做 `samp_id` 分析。

修正后的逐 sample direct mean 为：`mean|g|=0.26685546`、`median|g|=0.28009504`、`mean|b|=0.24302296`、`median|b|=0.24550465`、`mean|Δ_t g|=0.02261490`、`mean|Δ_t b|=0.02396674`、gamma/beta saturation fraction=`0.02633766 / 0.02032499`。这些是三个已联合训练 checkpoint 的 validation 描述性功能统计，不是重新训练机制的因果效应。

P−1 的冻结功能证据为：BETA_ONLY 仅 `1/3` seed quality-near，三-seed mean global-envelope error 恶化 `9.31%`；GAMMA_ONLY 为 `0/3`，global-envelope error 恶化 `2.53%`，因此不分配 P2。CONDITION_OFF 的 global-envelope error 恶化 `20.40%`、PCC 下降 `0.00822`。TIME_MEAN 的 global-envelope error 恶化 `25.97%`、PCC 下降 `0.01210`；TIME_SHIFT_30S 的四项 error 分别变化 `+3.99% / +6.60% / +23.05% / +24.10%`，PCC 下降 `0.02382`，且四项 error 和 PCC 在 `3/3` seed 方向一致，支持冻结 W0 对局部时频演化与时间对齐存在功能依赖。RESP/CARRIER 遮挡呈指标间权衡，只作为 P1 两个固定重训练臂的机制动机，不解释为频带因果优胜。

P−1 至此关闭并只保留 provenance；完整功能 audit 与 correction 均不得重复运行。P2 已关闭，剩余训练预算固定为 P1 九 runs 加 P3 三 runs，共 12 runs。

## 7. P1：W 信息来源与 6-voice 效率

P1 固定 current W0 的 FiLM、6 层 Local BiMamba2、refinement/head/decoder、数据与训练协议，只改变 W cache 的可见 scale 集合：

| ID | W 输入 | 新 formal runs | 科学角色 |
|---|---|---:|---|
| `W0_FULL_12V_FILM_D6` | 97 scales；名义 `0.03125–8.00 Hz`，实际 mapped `0.036621–7.995605 Hz` | 0，复用 | anchor |
| `W1_RESP_12V_FILM_D6` | 97-shape view，仅 mapped indices `0..55` 非零 | 3 | 直接低频呼吸来源 |
| `W2_CARRIER_12V_FILM_D6` | 97-shape view，仅 mapped indices `56..96` 非零 | 3 | 高频载波/调制来源 |
| `W3_FULL_6V_FILM_D6` | full-band 6 voices/octave，49 scales | 3 | representation efficiency package |

W1/W2 必须保持 `[97,360]` shape、W encoder、FiLM、parameter-fill、Mamba、decoder 和参数数不变；只在 W encoder 前应用静态 input-only mask，不为不同频带修改 normalization、LR 或通道。

W3 固定从锁定的 12V cache 使用 scale index `0,2,…,96` 得到 `[49,360]` 只读 view。P0 已证明 12V 偶数位置的名义目标频率与理论 6 voices/octave 网格一致；实际输入频率则严格采用同一位置的 mapped-center 子集。实现测试必须重验 source frequency file、index hash 与 49 个 mapped centers；任一不一致即停止。W3 identity 必须包含 source-cache hash 与 index hash。

W3 不声称现有磁盘 cache 自动减半。必须分别报告 source cache 体积、实际读取/物化策略、active tensor elements、branch activation、throughput 与 peak allocated/reserved。其结论名称固定为 `6-voice efficiency package`，不能解释为纯 voices 因果。

W1/W2/W3 一旦开放 formal，均直接完成三个 seed；P−1 普通效果不取消任何 P1 arm。

### 7.1 P1 实现锁（2026-08-17）

P1 实现锁固定为 `docs/experiments/crd_tf_w_v2_p1_implementation_lock_20260817.json`，SHA-256=`fb822ca7f8607e45443e07a94d91150bcc108217fb25c47f0b4504fbdf644f58`。实现只注册 candidate lock 的三个 P1 variant；P2 ADD/SCALE 与 P3 D4 未注册。

W1/W2 在 batch W tensor 已由只读 memmap copy 后、`CwtBranch.conv_in` 前应用 `zeros_like` input-only mask，保持 `[97,360]`、W encoder state、FiLM、D6 Mamba、decoder 和 `1,219,850` 参数不变。W3 用 `[:,::2,:]` 实现固定 indices `0,2,…,96` 的只读 strided view，模型输入为 `[49,360]`；source 仍从完整 `[97,360]` cache 读取，单 sample model-input elements 从 `34,920` 降至 `17,640`，不声称磁盘或 host→device 传输自动减半。W3 CWT branch 参数与初始化 state 仍和 W0 逐 tensor 相同。

运行时 source preflight 固定复核 candidate lock、cache manifest、train/val W metadata、frequency file/content、RESP/CARRIER/6V index hash，以及 49 个 mapped centers SHA-256=`2eda45200cc329c4989a516fa3611f0862c3c546bca26bb124025a2bb8146b29`。每个 run manifest/checkpoint 记录 variant view、source hashes、input/active elements、application point 与 materialization strategy；不写 source cache。

三份 stress config 固定 5 epochs、physical batch `128×1`、完整 train/validation、seed `20260811`，即 `5×ceil(10141/128)=400` optimizer updates 和 5 次完整 validation。stress 结束后专用 receipt 额外验证 history/checkpoint/primary finite、degeneracy=0、batch-1 bf16 input/全部 parameter gradients、warm train samples/s、forward 与 forward+backward latency、独立 validation peak；`<85% / 85%–90% / >90%或OOM` 显存规则不变。任一 stress run 目录存在即拒绝静默重复；formal preflight 要求三个 variant 各自唯一且 passed 的 `p1_stress_receipt.json`、全部与 formal 使用同一 commit，并拒绝重复 seed run。

定向 `py_compile` 与 `tests/test_crd_tf_w_v2_p1.py`、P−1 audit、v1 model/data/formal-config、CRD config/experiment/training 回归通过，结果为 `112 passed`。测试覆盖 allowlist/config path、mask/view/no-mutation、frequency/index/source identity、参数数、W0/C201 同 seed 初始化 identity、W3 `[49,360]` branch、variant forward application、stress/formal schema 和 receipt/formal gate；实现锁建立时未运行 CPU/GPU smoke、GPU stress、formal 或 research-test。

P1 实现阶段至此关闭并只保留 provenance；后续 stress/formal 完成登记见下一节。

### 7.2 P1 stress 与 formal 完成登记（2026-08-18）

三个 isolation stress 均从干净 commit `1d1b22edc6f1b9b96459c1b05eddf39208041162` 串行完成，均为 `400` optimizer updates、5 次完整 validation、history/checkpoint/primary finite、prediction degeneracy=0、batch-1 bf16 input/parameter gradients finite，且 source cache 未修改、`target_read=false`、`research_test_used=false`：

| Arm | Run | Receipt SHA-256 | warm samples/s | peak reserved |
|---|---|---|---:|---:|
| W1 | `runs/crd_tf_w_v2/engineering/crd_tfw_v2_w1_resp_12v_film_d6/20260817_162641_480661` | `489b49cd4765f13ddd30908931eed1604862ad7395080b1c3c07b89045c05a6c` | 219.4886 | 81.6497% |
| W2 | `runs/crd_tf_w_v2/engineering/crd_tfw_v2_w2_carrier_12v_film_d6/20260817_164531_290929` | `c1db21673ec58e25b16c345944e31c9fda8adb2ada64ad429bd4ae18a772fce8` | 221.9951 | 81.6497% |
| W3 | `runs/crd_tf_w_v2/engineering/crd_tfw_v2_w3_full_6v_film_d6/20260817_165715_535534` | `b1bc8a1dcb14b4213a84cf6a9f20fd701515ecb0af9288e2ffaf19cd309f8414` | 221.7101 | 71.2960% |

随后 9/9 formal 仍从同一干净 commit 完成。每个 run 均完整执行 80 epochs / 6,400 optimizer updates；三 seed、初始化 seed、config/cache/view identity、best/final checkpoint、逐 epoch history 与 validation summary 均通过复核，全部 finite 且 prediction degeneracy=0。选中 epoch 与唯一 run 为：

- W1：`9 / 18 / 29`；对应目录时间戳为 `20260817_171024_127509 / 20260817_183439_496437 / 20260817_195932_042352`；
- W2：`10 / 9 / 15`；对应目录时间戳为 `20260817_222958_552354 / 20260817_235416_840867 / 20260818_011907_164696`；
- W3：`9 / 12 / 12`；对应目录时间戳为 `20260818_142005_581105 / 20260818_154254_103140 / 20260818_142041_877481`。

三-seed validation 的 mean ± sample SD 及相对 W0 的冻结变化为：

| Arm | Whole RR | Local RR | trajectory | global envelope | signed PCC |
|---|---:|---:|---:|---:|---:|
| W0 anchor | 0.499298 ± 0.018320 | 0.551309 ± 0.012064 | 0.152729 ± 0.001423 | 0.191051 ± 0.003022 | 0.865300 ± 0.001491 |
| W1 | 0.505571 ± 0.024581 (`+1.2564%`) | 0.558017 ± 0.005791 (`+1.2168%`) | 0.153386 ± 0.006058 (`+0.4302%`) | 0.186611 ± 0.004759 (`−2.3242%`) | 0.861362 ± 0.003875 (`−0.003938`) |
| W2 | 0.465755 ± 0.003527 (`−6.7179%`) | 0.537615 ± 0.014087 (`−2.4838%`) | 0.153743 ± 0.003382 (`+0.6640%`) | 0.194306 ± 0.008800 (`+1.7038%`) | 0.862142 ± 0.003169 (`−0.003158`) |
| W3 | 0.480697 ± 0.017247 (`−3.7254%`) | 0.546916 ± 0.013429 (`−0.7968%`) | 0.149029 ± 0.002601 (`−2.4228%`) | 0.190644 ± 0.005097 (`−0.2133%`) | 0.863237 ± 0.001579 (`−0.002064`) |

按第 4 节预注册门槛，W1 因 Local RR 恶化超过 `0.5%` 且 PCC 下降超过 `0.003`，不进入质量池；W2 虽在 Whole/Local RR 上均为 `3/3` paired seeds 改善，但 global envelope 恶化 `1.7038%` 且 PCC 下降 `0.003158`，也不进入质量池。二者均未触发 catastrophic failure 线。

W3 的 Whole/Local/trajectory 改善方向分别为 `3/3、2/3、3/3`，四个 error guardrail 与 PCC guardrail 全部通过，因此进入质量候选池。W3 active model-input elements 减少 `49.4845%`，但 formal warm train throughput 仅由 W0 的 `223.2250` 增至 `232.7809 samples/s`（`+4.2809%`），peak allocated 由 `8773.3438` 变为 `8773.8242 MiB`，未达到 `+10%` throughput 或 `−15%` peak-allocated 门槛，因此不进入效率候选池。最终候选池仍须等待 P3 后由 P4 一次性冻结。

实现测试和六份 P1 stress/formal resolved config 从一开始均固定 `early_stopping_enabled=false`，W0 anchor 也完整运行 80 epochs；本协议第 3 节先前写成 `patience 30` 是文档错误。九个 P1 run 均按冻结 executable config 完成，所有选中 epoch 均不晚于 29，之后没有更低 Local RR；因此多跑的尾部 epoch 不改变 validation-selected checkpoint。现将文字纠正为固定 80 epochs，不把它隐匿成运行后改变 selector。P1 stress 与 formal 至此关闭，不得重复运行。

## 8. P2：有证据才开放的单一融合 arm

P2 只允许第 6.4 节自动选出的 ADD 或 SCALE 之一，最多三个 formal runs。它始终使用 `W0_FULL_12V`、6 层 Local BiMamba2 和当前融合位置；不切换到 P1 中表现较好的频带/6V 设置，避免把融合与 W 表示同时改变。

为保持 encoder 和 projection state 形状不变，ADD/SCALE 继续使用当前 `96→192` projection：

- ADD：训练和推理均固定有效 $g=0$，只让 $b$ 参与输出；
- SCALE：训练和推理均固定有效 $b=0$，只让 $g$ 参与输出。

被屏蔽的 projection 半部不得通过 parameter-fill 或扩宽其他模块补偿；其梯度应为零，并同时报告总参数数与 active parameter/path 口径。该比较优先保持机制干净，不伪装成严格 active-parameter-matched control。

P2 不开放 concat、gate、attention 或第二个融合 arm。P2 当前关闭，等待 P−1 固定 decision 和工程验收。

## 9. P3：D4 独立深度效率候选

P3 固定使用 `W0_FULL_12V_FILM`，只把 Local BiMamba2 从 6 blocks 改为 4 blocks：

| ID | blocks | 预计总参数 | 新 formal runs |
|---|---:|---:|---:|
| `D6_W0` | 6 | `1,219,850` | 0，复用 W0 |
| `D4_W0` | 4 | 约 `902,722` | 3 |

P0 已将 D4 准确参数数登记为 `902,722`；实现测试必须复核。D4 必须复用同 seed 的 W0 frontend、W encoder、FiLM、refinement/head/decoder 初始化语义，只减少两个完整 Local BiMamba2 blocks。

D4 只回答当前 W0 设置下的质量—效率问题。不得把 D4 与 RESP/CARRIER/6V/ADD/SCALE 未经训练地组合，也不得把 D4 结果外推为某个“最终 W 设置”下的深度结论。

D8 保持关闭。D4 失败本身不构成 D8 的开放证据。

P3 实现阶段已按用户授权完成；GPU stress 与 formal 状态见下一节。

### 9.1 P3 实现锁（2026-08-18）

P3 实现锁固定为 `docs/experiments/crd_tf_w_v2_p3_implementation_lock_20260818.json`，SHA-256=`0aa2f520a52a667640ea6550a6d26c48b0f65dd088de6b4616938705a7e40da7`。唯一注册 variant 为 `crd_tfw_v2_d4_full_12v_film`；P2、D8 和任何频带/6V/深度复合 variant 均未注册。

D4 继续读取未修改的 full-12V `[97,360]` W input，W branch、FiLM、frontend、refinement/head/decoder 与同 seed W0 保持逐 tensor 相同；只从 C201 local trunk 尾部移除 blocks 4/5。三个固定 seed 的共享 state identity 均已测试，参数数固定为 `902,722`，相对 W0 减少 `317,128 = 25.9973%`；每个移除 block 为 `158,564` 参数。四层 override 只允许 D4 内部复用 C201，不作为通用任意深度接口，D8 会被拒绝。

新增 `_p3_base.yaml`、唯一 stress/formal config、`tf_w_v2_p3_contract` manifest/checkpoint provenance、不可覆盖 `p3_stress_receipt.json` 与 formal preflight。stress 固定 5 epochs、`128×1`、完整 train/validation、seed `20260811`，即 400 updates 与 5 次完整 validation；receipt 复核 history/checkpoint/optimizer/primary finite、degeneracy=0、batch-1 bf16 input/parameter gradients、latency/throughput，并以 training/validation 两者最大值计算 peak allocated/reserved。formal 只接受唯一 passed receipt，且必须与 stress 使用同一 clean commit。

定向 `py_compile` 与 P1/P3、CRD config/model/experiment/training、v1 model/data/formal-config 回归通过，结果为 `123 passed`。实现期间未运行 GPU、stress、formal 或 research-test，未读写 target test、未修改 cache/checkpoint。P3 implementation 至此关闭并只保留 provenance。

### 9.2 P3 stress 与 formal 完成登记（2026-08-19）

D4 isolation stress 从干净 commit `6c4f6229eda6eb72c82e4cd17571bdc73bd97d54` 完成，固定 run 为 `runs/crd_tf_w_v2/engineering/crd_tfw_v2_d4_full_12v_film/20260818_224221_844176`，`p3_stress_receipt.json` SHA-256=`431604d505f10733082ad5a380a75c2544ff45649f86fc3452301d7d0bb61a8c`。receipt 为 passed：400 updates、5 次完整 validation、history/checkpoint/optimizer/primary 与 batch-1 bf16 input/parameter gradients 全部 finite、degeneracy=0；warm throughput=`280.4775 samples/s`，peak allocated=`6797.5527 MiB`，training/validation 最大 peak reserved=`11012 MiB = 69.0998%`。

随后 3/3 formal 仍从同一干净 commit 完成，每个 run 均完整执行 80 epochs / 6,400 updates；seed/config/cache/full-12V/D4/checkpoint identity、best/final checkpoint、history/summary finite 与 degeneracy=0 全部通过，source cache 未修改、`target_read=false`、`research_test_used=false`。selected epoch 与唯一时间戳为：

- seed `20260811`：epoch 13，`20260818_230002_842674`；
- seed `20260812`：epoch 26，`20260819_001222_827515`；
- seed `20260813`：epoch 14，`20260819_012343_109819`。

D4 三-seed validation mean ± sample SD 为：Whole RR `0.500556 ± 0.009903`、Local RR `0.561402 ± 0.023087`、trajectory `0.154691 ± 0.002532`、global envelope `0.190442 ± 0.005560`、signed PCC `0.862363 ± 0.001562`。相对 W0 的变化依次为 `+0.2519% / +1.8308% / +1.2847% / −0.3187% / −0.002938`，paired 改善方向依次为 `1/3 / 1/3 / 0/3 / 1/3 / 0/3`。

D4 没有任何 primary 达到第 4.1 节实质改善要求，且 Local RR 恶化超过质量候选的 `0.5%`，因此不进入质量池。其参数减少 `25.9973%`、formal warm throughput 相对 W0 提升 `24.8925%`、peak allocated 降低 `22.5204%`，三项结构/资源条件均通过；但 Local RR 与 trajectory 分别恶化 `1.8308% / 1.2847%`，超过效率候选对每个 error 的 `1.0%` 保护线，因此也不进入效率池。D4 未触发 catastrophic failure 线；结论固定为“显著工程效率收益，但质量交换超过预注册容忍度”，不开放 D8 或任何组合补跑。

P3 stress 与 formal 至此关闭，不得重复运行。P4 尚未生成；在其一次性汇总完成前，W3 只能称为按单臂门槛通过的 provisional quality candidate，不能提前写成最终 research-test allowlist。

## 10. 工程验收与显存停止线

每个新 variant 在 formal 前必须完成：

1. 严格 config/schema、参数数、共享 state、mask/index 和初始化 identity 测试；
2. CUDA bf16 batch-1 model/core-loss/backward，input/全部 parameter gradients finite；
3. 与 formal 相同 `128×1`、AMP、optimizer、checkpoint 和 validation 路径的独立长程 stress；
4. stress 至少覆盖 400 optimizer updates 和两次完整 validation；所有 history/checkpoint/optimizer/primary finite，prediction degeneracy 为 0；
5. 记录 warm-up 后 samples/s、forward latency、forward+backward latency、validation peak、peak allocated/reserved 及其占设备总显存比例。

Stress 产物必须进入独立、不可覆盖的 engineering 目录，不计入三 formal seeds，不复用其 checkpoint，不形成科研效果证据。

显存规则固定为：

- peak reserved `<85%`：工程通过；
- `85%–90%`（端点包含）：带 warning 通过，formal 报告必须保留该风险；
- `>90%` 或任意 OOM：关闭该候选。

不得为单一 arm 改为 `64×2`、`32×4`、改变 LR 或临时引入新的 activation-checkpoint 策略。若现有实现无法满足，则停止并修订协议，不旁路资源线。

所有长程 GPU stress、formal training 和 research-test 默认由用户执行；Codex 只准备实现、命令、产物契约和验收口径，除非用户在对应阶段明确授权代跑。

## 11. P4：冻结汇总与候选池

P1–P3 完成后生成一次性、不可覆盖的 validation summary：

- 重审每个 run 的 config、seed、commit、cache/view identity、checkpoint epoch、finite、degeneracy 与资源记录；
- 应用第 4 节质量/效率门槛；
- 报告三 seed、mean ± sample SD、paired-seed 方向、相对 W0 delta 和 tolerance-aware Pareto；
- P−1 只列为功能敏感性证据，不与 formal arm 混成模型排名；
- RESP/CARRIER 即使不晋级，也保留为机制结果；
- 不进行 `samp_id` 分析，不构造总分，不强行宣称唯一科学赢家。

若质量 Pareto 有多个候选，P5 的单一质量主候选按三-seed mean Local RR 最低者选择；精确相同时依次取 Whole RR 更低、PCC 更高者。若效率 Pareto 有多个候选，P5 的单一效率候选按 measured throughput gain 最大者选择；精确相同时依次取 peak allocated 更低、结构缩减比例更大者。该规则只限制 research-test allowlist，不把其他 Pareto 候选改写为失败。

P4 不新增训练，也不补组合回退。

### 11.1 P4 实现与 D4 双层报告口径（2026-08-19）

专用 `resp_train/crd/tf_w_v2_p4.py` 与 `scripts/summarize_crd_tf_w_v2.py` 已实现。summarizer 从 candidate/P1/P3/source-cache 四项冻结 hash 开始，逐一复核 W0/W1/W2/W3/D4 共 15 个 seed-level run 的 config、manifest、commit、cache/view/depth contract、80 epochs / 6,400 updates、checkpoint epoch/hash/finite、2,675 行逐 sample direct mean、degeneracy 与资源记录；输出严格限制为 candidate lock 预注册的六个文件，并通过临时目录完成后原子落盘，目标目录存在即拒绝覆盖。

P4 明确分开两层结论：

1. `strict_quality_pool / strict_efficiency_pool` 严格应用第 4 节硬门槛，并唯一控制未来 P5 allowlist；
2. `descriptive_quality_efficiency_pareto` 只描述非灾难性的资源—质量交换，不得推翻硬门槛。

因此 D4 若同时满足结构/资源收益和 non-catastrophic，但因轻度质量损失超过 `1%` 保护线而不合格，仍会以 `descriptive_noncatastrophic_efficiency_tradeoff=true` 保留。它不会被简化成“无价值失败”，也不会被偷渡为 strict efficiency candidate。W1/W2 继续作为机制结果保留，不因未晋级而删除。

定向 `py_compile` 与 P1–P4、CRD config/model/experiment/training、v1 model/data/formal-config 回归为 `127 passed`。对真实 15-run matrix 的只读预审已通过：严格质量池预期为 W3，严格效率池为空，描述性质量—效率 Pareto 预期为 W3/D4；预审未创建 summary、未读取 research-test、未做 `samp_id` 分析。P4 尚未生成，以上均不替代一次性冻结产物。

## 12. P5：最小 reused research-test

P5 只有在 P4 完成、allowlist 冻结且用户再次明确授权后才开放。现有 C201/W0 research-test 结果直接复用；新增 evaluation 最多包含：

- 一个 validation-selected 质量主候选的三个 checkpoints；
- 最多一个 validation-selected 效率候选的三个 checkpoints。

若没有 qualified 新候选，不为完成流程而新增 research-test。任何 checkpoint 均不得重选 epoch、重训或删 seed。

W1/W2/W3/D4/ADD/SCALE 都只使用现有 W transform；可复用锁定的 test input-only W cache，但必须登记 source-cache hash、mask/index/view identity 和 `target_read=false`。不得覆盖既有 test cache，也不得调用普通 `eval_crd.py`、已关闭的 S1C/P5/P6a 入口或旧 research-test 队列。必须建立本协议专用 checkpoint allowlist 与隔离输出。

证据名称固定为：

```text
reused research/development evidence; research-test-informed; not untouched independent test
```

P5 当前关闭。

## 13. 固定预算

训练预算为：

| 阶段 | 内容 | 新 training runs |
|---|---|---:|
| P−1 | FiLM、频带、时间负对照功能审计 | 0 |
| P1 | RESP、CARRIER、full-band 6V | 9 |
| P2 | 冻结决策为 retain FiLM；ADD/SCALE 均关闭 | 0 |
| P3 | W0 设置下 D4 | 3 |
| P4 | validation summary | 0 |
| P5 | 最终 research-test | 0 |

原执行预算固定为 **12 个新 training runs**，原条件硬上限 15 已因 P2 不触发而关闭；P1 九 runs 与 P3 三 runs 现已全部完成，剩余新 training run 预算为 **0**。Engineering stress 和 checkpoint evaluation 不计为 training runs，但其 GPU 成本必须单独记录。

禁止执行融合×频带×voices×深度全因子矩阵，也不为形成论文线性故事追加复合候选。

## 14. 当前唯一下一步

P0/P−1/P1/P3 已关闭，P2 不触发，训练预算已用完。P4 implementation 与真实 matrix 只读预审已完成。当前唯一下一步是从新的干净 commit 一次性生成冻结 summary：

```bash
git status --short
./.venv/bin/python scripts/summarize_crd_tf_w_v2.py \
  --candidate-lock docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json
```

首行必须无输出。命令只读既有 validation artifacts，不训练、不重选 checkpoint、不读取 research-test，也不做 `samp_id` 分析；固定输出目录存在即拒绝覆盖。P5 仍须在 P4 allowlist 冻结后另行获得用户明确授权。
