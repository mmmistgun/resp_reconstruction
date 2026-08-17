# CRD-TF-W v2 机制、频带与效率实验协议（修订版）

日期：2026-08-17

状态：**用户已确认本修订版的实验范围、执行顺序、晋级门槛、资源线与 GPU 预算。P0 candidate lock 已完成；P−1 validation-only 功能审计入口与定向测试已实现，完整三-checkpoint audit 尚未运行。P1–P5 的实现、stress、formal training 与 research-test 仍关闭。**

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
early stopping = patience 30, min_delta 0
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

- `resp_train/crd/tf_w_v2_audit.py`：SHA-256=`96307c9ee28e8070651868113313378fe21bb561808c80b5ddfb6b8a6c79b21e`；
- `scripts/eval_crd_tf_w_v2_functional_audit.py`：SHA-256=`62cee8677595f5351a3a771d422116273f64d4a759a2cbcff997a6625e139dd3`；
- `tests/test_crd_tf_w_v2_audit.py`：SHA-256=`b6b02b7433341e4fb6554bea492c3899b71ca121f8d723981d1c24a930a4fe3e`。

入口强制 candidate-lock path/hash、36 项 anchor artifact identity、锁定 runtime identity、validation W cache/frequency/view identity、干净 Git、CUDA、`split=val`、三个 W0 checkpoint 与 10 项固定 intervention；不提供 test、max-windows、任意 checkpoint 或输出覆盖入口。所有 W 干预只作用于 `TfV1CacheReader.get` 已复制的 batch tensor，不写原 memmap。失败保留独立 `.incomplete_*` 目录和 `failure.json`，成功才原子重命名到固定 P−1 目录。

FULL 每个 seed 必须以绝对容差 `1e-6` 复现锁定 validation 五项 primary 与 degeneracy summary，否则整个 audit 失败。成功产物严格为 candidate lock 中的五项文件；预计逐 sample metric 行数为 `3×10×2675=80,250`，FULL FiLM statistic 行数为 `3×2675=8,025`。P2 decision 由第 6.4 节纯函数生成，不读取 `samp_id`、secondary 排名或 research-test。

定向命令 `./.venv/bin/python -m pytest -q tests/test_crd_tf_w_v2_audit.py tests/test_crd_tf_v1_model.py tests/test_crd_tf_v1_data.py tests/test_crd_experiment.py` 已通过，结果为 `40 passed`；同时完成三个新增文件的 `py_compile` 和 candidate-lock input verification。测试覆盖真实 W0 结构在替换 Local Mamba 为 Identity 后的 active-FiLM FULL wrapper 逐 tensor identity，但没有加载冻结 W0 checkpoint、运行原生 Mamba 或完整 validation。

第一次完整 audit 从干净 commit `82926b2` 启动，在 seed `20260811` 的 FULL 锚点失败并按约定保留 `p_minus_1_validation_audit.incomplete_20260817T070706_900712Z/failure.json`；观测到锁定六项 summary 最大绝对差 `2.6775125796740795e-05`。随后用原生 `eval_crd.py` 对同一 checkpoint、cache 和 validation rows 复评，六项与冻结 summary 的差均严格为 `0.0`，排除历史 summary 或当前 CUDA 环境漂移。根因是旧包装器在 decoder 前执行 FiLM 统计算子，改变了原生 CUDA 执行/内存路径。修订后 FULL 直接调用未修改的 W0 原生 forward，只用只读 forward hook 捕获 gamma/beta，并在原生输出完成后计算统计；`1e-6` 锚点门槛不放宽。该失败不形成 P−1 科研结果，修订代码仍须从新的干净 commit 重跑完整 audit。

P−1 实现已完成，但完整 audit 尚未运行。它必须在本轮代码与协议提交、工作树干净后由用户执行；运行结果返回并登记前不实现 P1/P2/P3。

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

P1 当前关闭；P0 已完成，仍等待后续明确开放实现、工程验收和 GPU stress。

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

P3 当前关闭；P0 已完成，仍等待后续明确开放实现、工程验收和 GPU stress。

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

P4 不新增训练，也不补组合回退。P4 当前关闭。

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
| P2 | ADD 或 SCALE，按 P−1 固定规则最多一个 | 0–3 |
| P3 | W0 设置下 D4 | 3 |
| P4 | validation summary | 0 |
| P5 | 最终 research-test | 0 |

因此标准预算为 **12 个新 training runs**，硬上限为 **15 个**。Engineering stress 和 checkpoint evaluation 不计为 training runs，但其 GPU 成本必须单独记录。

禁止执行融合×频带×voices×深度全因子矩阵，也不为形成论文线性故事追加复合候选。

## 14. 当前唯一下一步

P0 与 P−1 实现验收已经完成。下一步不是继续写 P1/P2/P3，而是先提交本轮代码和协议、确认 `git status --short` 无输出，再由用户运行一次完整 P−1：

```bash
./.venv/bin/python scripts/eval_crd_tf_w_v2_functional_audit.py \
  --candidate-lock docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json \
  --split val \
  --device cuda:0
```

该命令将依次评价 3 checkpoints × 10 interventions，可能长时间占用 GPU，Codex 不代跑。固定成功输出为 `runs/crd_tf_w_v2/p_minus_1_validation_audit/`；目录已存在时拒绝覆盖。若留下 `.incomplete_*`，必须先审计 `failure.json`，不得删除后静默重跑。

P−1 结果决定 P2 的 ADD/SCALE/retain-FiLM 分支，但不改变固定 P1 三臂。P−1 结果登记前不实现 P1/P2/P3，不运行 stress、formal 或 research-test。P5 仍须在 P4 allowlist 冻结后另行获得用户明确授权。
