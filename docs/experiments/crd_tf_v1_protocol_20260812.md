# CRD-TF v1 固定时频表示与交互实验协议

日期：2026-08-12

状态：**P0–P3 已完成并冻结；最终 physical batch 固定为 `128×1`。P4 正式训练等待用户明确确认 45-run 成本，P5 结果汇总与 P6 Fusion 仍未开放**

协议标识：`crd-tf-v1-research-informed-20260812`

## 1. 权威性、证据属性与当前边界

本文由 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 46 节纳入当前唯一实验协议；发生冲突时以主协议为准。设计来源为 `docs/temp/时频融合_2026_08_12__0022.md`，但 temp 文件继续只作讨论来源，不是可运行协议。

本阶段建立在以下已冻结结果之上：

- 第一阶段 B0/T2/T4 research-test 已被观察，并影响了后续 CRD 研究方向；
- CRD-v1.1 S0/S1/S1C/S1F/S2A/S2B-R、CRD_102 failure/matched-observability diagnostics 与 C0/C1/C2 控制线均已完成并关闭；
- C2 选择 `CRD_102` Mamba backbone + `crd_c201_decoder_10hz_cap`，但该结果只属于 validation-development decoder evidence；
- C201 相对 CRD_102 的 Local RR seed-mean 改善为 `0.6978%`、paired seed 为 `2/3`，signed PCC 下降 `0.001155`、trajectory 恶化 `0.7648%`。该收益较小，不能表述为已经获得独立确认的新最终模型。

因此本文全部结果只能称为 **research-test-informed development/validation evidence**。P0–P5 只允许读取既有 train/validation；普通 CRD research-test、S1C 入口、既有 research-test 产物和 test target 均不得读取。若 P5 形成候选，也不能自动访问 research-test；强泛化证据需要新的锁定 cohort、外部数据或 prospective holdout，并另立协议。

本阶段不改变：

- 数据集、admission、train/validation split、subject/session 隔离与 sample seed；
- 输入 `bcg_rawish_segment_soft_z_key`、target、100 Hz/180 秒任务定义；
- 正式输出算子 `Pi`、`L_sync + 0.25 L_effort`；
- 五项 primary、IBI/coverage、分层包络指标及 Local-RR checkpoint selector；
- 非有限 prediction、target-only eligibility 与逐 sample direct-mean 语义。

## 2. 当前开放开关

| 阶段 | 内容 | 当前状态 | 用户介入 |
|---|---|---|---|
| P0 | C201 candidate lock、精确数学/决策冻结 | 已完成并冻结 | 已完成 |
| P1 | synthetic/input-only calibration、固定表示缓存实现与审计 | 已完成并冻结 | 已完成 |
| P2 | 模型、配置、汇总器与单测 | 已完成并冻结 | 已完成 |
| P3 | CUDA synthetic、最大臂 physical-batch acceptance | 已完成并冻结 | 已完成 |
| P4 | 正式三 seed 全矩阵 | 关闭 | 必须明确确认规模与运行顺序 |
| P5 | 冻结 validation 汇总与候选选择 | 关闭 | 汇总前不需要主观选模型 |
| P6 | gated residual / local cross-attention Fusion | 关闭 | P5 后另立协议并确认 |

任何 single 的 validation 结果都不得关闭尚未完成的 pair/triple。工程 OOM、非有限、缓存身份错误或实现契约失败可以阻塞对应正式队列，但不能用模型效果结果删减组合。

## 3. 第一性问题拆分

### 3.1 可观察量

从同一 BCG 输入只构造四类信息：

1. `M`：固定 multi-resolution STFT 的局部谱形与谱演化；
2. `W`：固定 analytic Morlet CWT 的自适应多尺度幅度；
3. `L`：可学习 analytic carrier filterbank 的低频幅度调制；
4. `S`：固定 WSST 上提取的稀疏 ridge frequency/amplitude/confidence。

M/W/S 最终特征完全由输入和冻结算子决定，可预计算；L 的 filter center/Q 可学习，不能预计算最终 envelope，但可预计算输入 BCG 的固定 complex spectrum。

### 3.2 可控制量

- 表示算子、频带、时间分辨率、边界与数值精度；
- 每支 encoder 的参数预算；
- 唯一 Stage-1 FiLM 位置与 zero-init；
- single/pair/triple 的组合；
- 参数匹配的 temporal-only capacity controls；
- batch、梯度累计、LR/update schedule、seed 和训练预算。

### 3.3 必须保证

- cache 不读取 target，不跨 split 拟合统计量，不包含 research-test；
- 所有包含相同模块的 variant 在相同 seed 下共享模块初始 state 逐 tensor 一致；
- zero-init 时各 TF variant 的初始 waveform 与相同 seed C201 逐点一致；
- 容量对照覆盖 single、pair、triple 三种基数；
- 交互按同 seed 配对计算，不把五项 primary 合成加权总分；
- 全部正式 arm 在查看任何 P4 结果前冻结，single 结果不承担组合 gate。

## 4. C201 锚点、candidate lock 与训练身份

P0 已对以下三个既有 `checkpoint_best_local_rr.pt` 建立新的只读 candidate lock，记录 checkpoint/config/manifest/metrics-summary 的路径、文件大小与 SHA-256、训练 commit、seed、selected epoch 和协议：

```text
runs/crd_v1/crd_c201_decoder_10hz_cap/seed_20260811/20260811_162853_343330
runs/crd_v1/crd_c201_decoder_10hz_cap/seed_20260812/20260811_173454_537310
runs/crd_v1/crd_c201_decoder_10hz_cap/seed_20260813/20260811_184113_013380
```

锁定文件为 `docs/experiments/crd_tf_v1_candidate_lock_20260812.json`，SHA-256 为 `c8d4823500e6096fcacb8d2e8787f7b3422160813eabe31adf01b1f1f75cc139`，锁定名称为 `TF000_C201_ANCHOR`。文件身份已逐项复核。既有三个 run 不重跑、不覆盖，也不把 C202 或 CRD_102 混入 TF000 数值。

P4 的新 TF variant 固定采用 **from-scratch training**：

- 不 warm-start 已训练 C201 checkpoint；
- 不冻结 C201 backbone 或 decoder；
- 复用 C201 的 frontend、六个 Local BiMamba2、refinement、coarse head 和 10-Hz nonlinear decoder；
- 相同 seed 下上述共享模块必须通过独立 module seed 与 TF000 初始 state 逐 tensor 相同；
- 新表示 encoder 正常初始化，最终 FiLM projection 的 weight/bias 全零。

因此“初始模型等于 C201”只表示同 seed、同初始化时 waveform identity，不表示从已训练 C201 继续优化。

## 5. 四类表示的冻结候选

P0/P1 允许用 synthetic/input-only calibration 冻结本节仍标记为 calibration parameter 的数值；不得读取真实 validation target 调参。一旦任何 P4 run 启动，本节全部参数锁死。

### 5.1 M：Multi-resolution STFT

输入为同一个 100-Hz、18000 点 BCG。两支均先逐 frame 去均值，乘 symmetric Hann（`periodic=False`），`center=False`、不 padding、`rfft norm="backward"`，使用 float32/complex64。

| branch | window / `n_fft` | hop | band | bins | frames | cache shape |
|---|---:|---:|---:|---:|---:|---:|
| slow | 30 s / 3000 | 1.5 s / 150 | 0.03–1.20 Hz | `k=1…36`，36 | 101 | `[36,101]` |
| fast | 6 s / 600 | 0.5 s / 50 | 0.70–8.00 Hz | `k=5…48`，44 | 349 | `[44,349]` |

固定特征为 `log1p(abs(STFT))`，不先做 band average。Slow/fast 使用各自的 frequency/time anisotropic encoder，先在各自时间轴编码，再组合成一个 M context；不得在 fixed cache 中执行可学习编码。

### 5.2 W：固定 analytic Morlet CWT

- 实现依赖冻结为 `ssqueezepy==0.6.6`；wavelet 为 analytic Morlet，`mu=13.4`，dtype float32；
- 目标中心频率为 `f_j = 8 * 2^(-j/12) Hz, j=0…96`，排序后覆盖 `0.03125–8 Hz`，共 97 scales；
- 边界固定为 reflect padding 后裁回原 18000 点，不把结果解释为 causal/streaming；
- fixed feature 为 `log1p(abs(CWT))`；随后对每 50 个原始时间点做不重叠算术平均，得到 `[97,360]`；
- 不依据 validation target 删除 cone-of-influence 区域；P1 必须单独报告首末 15 秒与中部的 synthetic localization 误差。

Scale 到频率的离散映射、排序和 97-scale identity 必须固化为回归测试；若库 API 无法精确实现上述频率网格，必须在 P1 文档修订中登记实际 scale/frequency 数组后再继续，不能静默使用库默认 scales。

### 5.3 L：Learnable analytic carrier-modulation filterbank

L 是对已失败 `CRD_203 Analytic-AM` package 的 result-informed follow-up，不表述为从未测试的新信息源。与 CRD_203 的主要差异固定为：8→24 filters、modulus→modulus-square、10→2 Hz context，以及 static residual→统一 FiLM。

24 个初始中心频率为：

```text
c_k = 0.7 * (8 / 0.7) ** (k / 23),  k=0…23
```

相邻初值的几何中点定义各中心的有界区间，首尾边界固定为 0.70/8.00 Hz；sigmoid logits 保证中心不越过相邻区间。相对带宽 `rho=bandwidth/center` 固定在 `[0.08,0.30]`，初始化为 `0.15`。`center_logits` 与 `bandwidth_logits` 继续进入 no-decay 参数组，不设置独立 LR。

前端数学固定为：

```text
input one-sided rFFT spectrum `[9001]`（可缓存；按固定规则构造 analytic full spectrum）
→ positive-frequency bounded Gaussian analytic masks
→ complex IFFT
→ log1p(|z_k(t)|²)
→ whole-window zero-phase FFT bandpass 0.03–0.80 Hz
→ exact decimation ::50 to 2 Hz
→ [24,360]
```

先 log-compress 再 bandpass，避免对可能因零相位带通产生负值的序列调用 `log1p`。由于带通已严格删除 0.80 Hz 以上频率，降到 2 Hz 不再增加另一抗混叠滤波器。该算子继承完整 180 秒循环 FFT 边界语义，不声称流式能力。

最终 L feature 不能缓存；只允许缓存输入的固定 complex64 spectrum，并在训练时对 mask/center/rho 保留梯度。P1 必须检查中心有序、bounds、finite gradient、filter collapse 和边界命中率。

### 5.4 S：固定 WSST ridge

- 实现依赖冻结为 `ssqueezepy==0.6.6`；analytic Morlet `mu=13.4`；
- 目标频率网格为 `f_j = 8 * 2^(-j/48) Hz` 中不低于 0.03 Hz 的 387 个频率；
- WSST 离线确定性计算，不进入反向传播；
- 每 0.5 秒输出 respiratory `0.03–0.80 Hz` 与 carrier `0.80–8.00 Hz` 各两条 ridge 的 frequency/amplitude/confidence，共 `[12,360]`；0.80 Hz 的归属固定为 respiratory 上端，carrier 使用 `f>0.80 Hz`。

P1 synthetic calibration 必须在不读取真实 target 的条件下冻结并写回：

- ridge 动态规划的 log-frequency smoothness penalty；
- 第二 ridge 的 suppression radius；
- amplitude 与 confidence 的归一化公式；
- ridge 缺失、断裂、crossing、并列与边缘帧规则；
- harmonic switching 的诊断定义。

上述参数写回前，S 只允许原型实现，不允许 P3/P4。De-shape SST 不属于本矩阵；只有 P5 冻结结果显示普通 S 存在预定义 harmonic-switching failure 后，才可另立协议。

## 6. 统一 Stage-1 encoder 与 FiLM

四种 representation branch 的固定接口为：

```text
native fixed/learnable representation
→ branch-specific encoder
→ deterministic interpolation to [B,96,1800]
→ Conv1d(96,192,1) zero-init
→ split gamma/beta, each [B,96,1800]
```

M/W 允许 frequency/time anisotropic Conv2D；L/S 使用小型 temporal Conv/TCN。所有新 encoder 禁止 BatchNorm 和 batch-dependent normalization，第一阶段 dropout 固定为 0；允许 GroupNorm/LayerNorm。这样 physical microbatch 改变时不会引入额外 batch-statistic 变量。

FiLM 唯一位置固定为 **六个 Local BiMamba2 之后、两个 refinement blocks 之前**：

```text
z' = z * (1 + Σ 0.5*tanh(gamma_r)) + Σ 0.5*tanh(beta_r)
```

求和只遍历当前 variant 存在的 branch。不得在 P4 同时比较 pre-Mamba、decoder-feature、gated 或 attention 融合。

每个 representation branch（包含 L 的 48 个 filter logits）总 trainable increment 必须落在共同目标预算 `P_branch=150,000` 的 ±2% 内。P2 冻结的精确 trainable increment 为：M `149,984`、W `150,048`、L `149,904`、S `150,048`、每个 temporal-only control stack `149,952`。所有 parameter-fill 模块均参与 forward，不允许 inert 参数凑数。

## 7. 固定表示缓存

### 7.1 缓存范围

只为 admission 后的完整 train/validation 建缓存：`10141 + 2675 = 12816` 个窗口。不生成或读取 research-test cache。缓存只消费输入 BCG 与 `dataset_row_id`，不得读取 target waveform、target-derived eligibility 或 validation metric。

固定缓存项：

| cache | dtype | 单 sample shape | 是否跨 variant 复用 |
|---|---|---|---|
| M slow | float32 | `[36,101]` | 所有 M-containing |
| M fast | float32 | `[44,349]` | 所有 M-containing |
| W magnitude | float32 | `[97,360]` | 所有 W-containing |
| S ridge | float32 | `[12,360]` | 所有 S-containing |
| L input spectrum | complex64 | `[9001]` one-sided rFFT | 所有 L-containing |

按当前 shape 粗估总缓存约 4 GiB；实际值由 P1 manifest 登记。缓存根目录建议为：

```text
runs/crd_tf_v1/cache/<transform_sha256>/
```

该目录不进入 Git，已存在的完整 cache 不覆盖；配置或实现 hash 改变必须生成新目录。

### 7.2 存储与身份

固定 shape 使用按 split 分离的 contiguous `.npy`/memory-map 数组，另存有序 `dataset_row_id`；避免创建数万小文件。Manifest 至少保存：

- protocol、生成 commit、`git_dirty`、命令和时间；
- Python/PyTorch/NumPy/SciPy/PyWavelets/ssqueezepy 版本；
- dataset index path/hash、source key、sample seed 和有序 row-id hash；
- 完整数学参数、frequency/scale 数组、shape、dtype、端点与边界语义；
- 每个数组文件大小、SHA-256、finite/min/max/mean/std；
- train/validation 数量和 `research_test_used=false`；
- synthetic calibration 产物 hash。

训练加载器必须按 `dataset_row_id` 一对一索引 cache；缺失、重复、顺序错位、hash 不符或非有限时立即失败。不得回退为在线临时计算后继续 formal。

### 7.3 缓存等价验收

每种 fixed cache 至少抽取首/中/末和固定随机 row，比较在线算子与缓存，要求 shape/dtype 一致、float32 最大绝对差不超过预先登记的数值容差。L spectrum cache 还必须与直接从 raw input 计算的 L output/parameter gradient 做等价测试。

## 8. Synthetic/input-only calibration

固定信号集：

- `0.05 / 0.10 / 0.20 / 0.40 / 0.70 Hz` sinusoid；
- `1 / 2 / 4 / 6 Hz` carrier；
- `2 Hz × 0.2 Hz AM`、`4 Hz × 0.4 Hz AM`；
- respiration chirp `0.08→0.45 Hz`；
- carrier + harmonics + fixed-SNR noise；
- 两个 respiratory components crossing；
- 常量、impulse、首末边界事件和 NaN/Inf rejection。

最低通过条件：

- M peak localization 不超过各自半个原生 FFT bin；
- W 中部 ridge localization 不超过一个 12-voice step，首末 15 秒误差单独报告；
- L 的 AM peak 与真值差不超过 `0.01 Hz`，center/Q 梯度 finite 且至少一个非零；
- S 中部 ridge median localization 不超过一个 48-voice step，crossing/harmonic/tie-break 可重复；
- 所有输出 shape、dtype、finite、缓存复算与确定性 hash 通过。

Calibration 只允许选择 CWT scale mapping、S ridge smoothness/suppression/confidence、L 数值 epsilon 和 cache 数值容差；不得选择真实模型宽度、训练 LR、metric、频带或候选组合。

## 9. Stage-1 正式矩阵与容量对照

### 9.1 Representation matrix

| ID | M | W | L | S | 角色 |
|---|:---:|:---:|:---:|:---:|---|
| `TF000` |  |  |  |  | 锁定既有 C201 anchor，不重训 |
| `TF101` | ✓ |  |  |  | M marginal |
| `TF102` |  | ✓ |  |  | W marginal |
| `TF103` |  |  | ✓ |  | L marginal |
| `TF104` |  |  |  | ✓ | S marginal |
| `TF201` | ✓ | ✓ |  |  | M+W |
| `TF202` | ✓ |  | ✓ |  | M+L |
| `TF203` | ✓ |  |  | ✓ | M+S |
| `TF204` |  | ✓ | ✓ |  | W+L |
| `TF205` |  | ✓ |  | ✓ | W+S |
| `TF206` |  |  | ✓ | ✓ | L+S |
| `TF301` | ✓ |  | ✓ | ✓ | M+L+S |
| `TF302` |  | ✓ | ✓ | ✓ | W+L+S |

这是完整六个二阶 pair 加两个预注册 triple，不称为完整四因素 factorial；`M+W+L`、`M+W+S` 和四路组合均不运行。

### 9.2 Universal capacity controls

| ID | 输入 | 参数增量 | 对照对象 |
|---|---|---:|---|
| `CTRL1` | 只读 C201 temporal latent | `1×P_branch ±2%` | TF101–104 |
| `CTRL2` | temporal-only 双 stack | `2×P_branch ±2%` | TF201–206 |
| `CTRL3` | temporal-only 三 stack | `3×P_branch ±2%` | TF301–302 |

Control 从与 FiLM 相同位置的 temporal latent 产生同规格 gamma/beta，不读取任何 M/W/L/S cache。不得用未参与 forward 的 inert 参数凑数。

第一阶段因此包含 **15 个新 variant × 3 seeds = 45 个新 formal runs**；TF000 复用三个锁定 C201 run。若 batch fallback 触发，见第 11.3 节，需再增加三个统一 batch 的 TF000 control。

## 10. 交互、容量归因与 P5 选择

先把五项 primary 转成越大越好的 utility：

```text
uWhole = -Whole RR MAE
uLocal = -Local RR MAE
uTraj  = -Envelope trajectory MAE
uEnv   = -Global envelope modulation error
uPCC   = +signed PCC
```

二阶和三阶 interaction 沿用 temp 草案公式，但必须对相同 training seed 分别计算，再报告三 seed mean ± sample SD：

```text
I(A,B)   = u(AB)-u(A)-u(B)+u(BASE)
I(A,B,C) = u(ABC)-u(AB)-u(AC)-u(BC)+u(A)+u(B)+u(C)-u(BASE)
```

不对五项 interaction 加权或平均。某 metric 的 descriptive positive interaction 定义为 seed mean `>0` 且至少 `2/3` seed 的 interaction `>0`；它不等于统计显著性。

进入未来 Fusion 候选池还必须满足：

1. lifecycle、checkpoint、cache identity、finite 与 prediction-degeneracy 审计通过；
2. 相对 TF000：Local RR 恶化不超过 `1.5%`、signed PCC drop 不超过 `0.005`、trajectory 恶化不超过 `1.5%`；
3. 相对匹配的 CTRL1/2/3，至少一个 primary 达到实质改善且无上述 guardrail 失败；
4. 实质改善定义：Whole/Local RR、trajectory、global envelope 至少相对改善 `0.5%`，或 signed PCC 绝对增加至少 `0.002`，并至少 `2/3` paired seeds 同方向。

候选来源保留三类：

- absolute：通过上述资格且位于五项 primary 的 tolerance-aware Pareto set；
- interaction：通过资格且至少一个预注册 interaction metric 为 positive；
- single：通过资格的 single Pareto set，不强行指定唯一赢家。

Tolerance-aware dominance 使用与“实质改善”相同的 `0.5% / 0.002` 容差：A 在所有轴不劣于 B 超过容差，且至少一轴优于 B 超过容差，才称为支配。无唯一候选时保留 Pareto set，不构造总分、不靠 secondary 指标打破平局。

P5 只冻结 Stage-1 结果和候选集合。Gated residual、local cross-attention、fixed-vs-learnable、PCEN、de-shape、phase、mother-wavelet/window sensitivity 全部留到新的 P6 协议；P5 结果不自动开放它们。

## 11. 训练预算、early stopping、batch 与学习率

### 11.1 Early-stopping 回顾审计

对 `runs/crd_v1/**/train_history.csv` 中 54 个完整 80-epoch formal histories 只读回顾：

```text
selected epoch: min=4, Q1=11, median=13, Q3=19, max=72
<=20: 40/54
<=25: 43/54
<=30: 47/54
<=40: 49/54
```

“大多数在 25 epoch 前达到最佳”成立，但仍有 `11/54` 晚于 25、`5/54` 晚于 40。用 Local RR、`min_delta=0` 回放：patience 20 会提前错过 4 个历史全局最佳，patience 30 仍错过 2 个，其中一个晚期改善约 `0.00656 bpm`；这些晚恢复来自结构响应较慢的 global-stage 模型。

新矩阵同时引入 zero-init FiLM、缓存表示和 pair/triple，收敛速度可能改变。为避免把 early stopping 与 representation 作为两个同时变化的科学变量，P4 固定：

```text
max_epochs = 80
early_stopping = false
checkpoint selector = full-validation Local RR minimum
```

训练仍只永久保存 `checkpoint_best_local_rr` 与 `checkpoint_final`。P5 完成后可用本矩阵的 45 条新学习曲线为 P6 单独冻结 early stopping；不得在 P4 中看到若干早期曲线后临时启用。

### 11.2 Optimizer 与学习率

数据、有效 batch 和 C201 主体不变，新增 branch 目标参数约为模型的有限增量，当前没有证据支持线性放大或缩小 LR。P4 继续冻结：

```text
optimizer = AdamW
max_lr = 3e-4
min_lr = 3e-5
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 1e-4
warmup_fraction = 0.05
schedule = step-exact warmup + cosine over 6400 planned updates
grad_clip_norm = 1.0
AMP = bf16
```

L 的 center/bandwidth logits 使用同一 LR、no-decay；不增加 parameter-group LR 搜索。P5 只报告中心边界命中、Q 分布和梯度 finite 作为诊断，不能据此事后重训。

### 11.3 Physical batch 与梯度累计

首选维持既有：

```text
physical_batch = 128
gradient_accumulation_steps = 1
effective_batch = 128
updates_per_epoch = 80
```

固定 cache 能减少在线 M/W/S 计算，但不能保证最大 triple 或 24-filter L 的 activation 显存可行。P3 先使用 chunking/activation checkpoint，在不改变数学输出的条件下争取 `128×1`。目标 GPU 的 peak reserved fraction 必须不超过 `80%`，并完成一次 128-train/32-validation、1-update lifecycle。

若任一必须运行的最大臂无法通过 `128×1`，不得只为该 variant 临时减 batch。P4 启动前统一选择一个 fallback：

```text
64×2  或  32×4
```

现有 eligible-aware accumulation 会在 accumulation group 的总 eligible count 上精确归一化；完整数据下两种 fallback 仍形成 80 optimizer updates/epoch，因此 effective batch、6400-update LR schedule 与 LR 数值不变。新 branch 禁止 BatchNorm/dropout，减少 physical microbatch 对模型语义的影响。

但 physical batch 改变仍可能带来 kernel 数值与随机轨迹差异，不能把既有 `128×1` C201 当作严格同训练 substrate。若触发 fallback，必须新增 `TF000_BATCH_CONTROL` 三 seed，在相同 fallback 下重训 C201，并让全部 15 个新 variant 使用它作为 BASE；正式新增规模由 45 增至 48 runs。不能通过梯度累计绕过这一对照。

## 12. 工程验收与正式启动条件

P4 只有在以下全部完成后才可由主协议明确开放：

- [x] 新 C201 candidate lock 生成且身份审计通过；
- [x] M/W/L/S 数学与 S calibration parameters 全部冻结；
- [x] train/validation fixed cache 完整、不可覆盖、hash/row-id/finite 审计通过；
- [x] 四个 single、六个 pair、两个 triple、CTRL1/2/3 参数与 forward 契约通过；
- [x] 相同 seed shared C201 state 与 zero-init waveform identity 通过；
- [x] online-vs-cache 与 L cached-spectrum gradient 等价测试通过；
- [x] CPU 结构/配置/汇总器定向测试通过；
- [x] CUDA synthetic 与统一 physical-batch acceptance 通过；
- [x] 精确 trainable params、显存、吞吐和最终 `128×1` 决策写回；
- [x] 15 个 variant × 3 seeds 的固定命令、输出根目录与运行顺序冻结；
- [ ] 用户明确授权正式长时间 GPU 队列。

任何 P3 acceptance 数值都只作工程证据，不形成模型效果结论。正式 run 必须来自包含最终协议、缓存身份和实现的统一干净 commit；中断 run 不凭已有 best checkpoint 纳入比较。

## 13. 建议执行顺序

1. 生成 C201 candidate lock。
2. 实现 synthetic calibration；冻结 S ridge 与 W scale 映射。
3. 建立 M/W/S 和 L-spectrum train/validation cache，完成 cache audit。
4. 实现统一 branch interface、FiLM、CTRL1/2/3、严格 config schema 和参数匹配。
5. 运行定向测试与轻量 CPU smoke。
6. 用户运行全 15-arm CUDA synthetic，以及最大 single/pair/triple `TF102-W / TF204-WL / TF302-WLS` acceptance。W 的二维中间激活大于 M，L 引入 differentiable analytic-envelope 链，因此固定用 W、W+L、W+L+S 覆盖一至三支的保守显存边界；不得看 validation 效果改验收臂。
7. 冻结 batch、显存、吞吐、准确参数数和全部正式命令。
8. 用户确认 45-run（或 fallback 后 48-run）成本后，按固定顺序完成全部 P4；不得按中间结果删臂。
9. 一次性生成 interaction/capacity/Pareto 冻结 summary。
10. 若存在候选，再另立 P6 Fusion 与独立证据协议。

## 14. 需要用户介入的节点

以下事项不能由 Codex 静默决定或启动：

1. **缓存位置与空间**：确认 `runs/crd_tf_v1/cache/` 所在磁盘可接受约 4 GiB 起步的不可覆盖缓存；若希望放到数据盘，需给出路径。
2. **长时间预计算**：全量 CWT/WSST cache 可能耗时较长，默认由用户执行或明确授权代跑。
3. **GPU acceptance**：P2 完成后，用户执行或明确授权目标 GPU 上的 batch acceptance。
4. **batch fallback**：若 `128×1` 失败，用户确认统一采用 `64×2` 还是 `32×4`；选择依据为显存、吞吐与完整 lifecycle，而不是 validation 效果。
5. **正式矩阵成本**：用户明确确认 45 个新 formal runs；若 fallback 触发，则确认包含 TF000 batch control 的 48 个 runs。
6. **独立证据来源**：P5 后由用户决定是否能提供外部数据、新锁定 cohort 或 prospective holdout；若没有，只能接受 development evidence，不包装成强泛化结论。
7. **Fusion 阶段**：P5 候选冻结后，由用户确认是否另开 gated/cross-attention 协议；当前不得提前实现或训练。

## 15. 当前明确不做

- 不读取或重复运行既有 research-test/S1C；
- 不修改 loss、metrics、selector、数据 admission 或 split；
- 不 warm-start、冻结 C201 或根据 validation 单独调每支 LR；
- 不用 single 结果关闭 pair/triple；
- 不增加四路组合、其余 triples、STFT-only、target-STFT loss、auxiliary head 或新 decoder；
- 不在 Stage 1 比较 gated/cross-attention、PCEN、de-shape、complex phase、mother wavelet 或窗口敏感性；
- 不因 OOM 静默改变单个 variant 的 batch、accumulation、dtype 或缓存精度。

## 16. P0/P1 实现与冻结结果（2026-08-12）

用户已接受默认 cache 根目录 `runs/crd_tf_v1/cache/`、约 4 GiB 起步的存储预算，以及新增 `CTRL3` 后 45 个新 formal runs 的矩阵设计；这只确认协议与未来预算，不等于授权现在启动全量预计算、GPU 或 P4。

P0 candidate lock 已完成。P1 当前实现包括：

- `resp_train/crd/tf_v1_features.py`：M 固定双尺度 STFT、W Morlet CWT、S WSST + O(F×T) deterministic ridge、L 24-filter learnable analytic modulation 与 one-sided spectrum cache；
- `resp_train/crd/tf_v1_calibration.py`、`scripts/calibrate_crd_tf_v1.py`：synthetic-only calibration、S penalty/radius 固定候选网格、不可覆盖 JSON；
- `resp_train/crd/tf_v1_cache.py`、`scripts/build_crd_tf_v1_cache.py`：只读 input BCG 的 train/validation memory-map cache、candidate-lock/calibration/index/row-id/hash/finite 审计和不可覆盖提交；
- `tests/test_crd_tf_v1_features.py`、`tests/test_crd_tf_v1_cache.py`：固定 shape/frequency、L cached-spectrum/gradient、双 ridge、非有限拒绝、calibration/cache identity 与 array inventory。

定向测试为 `10 passed`；M/L 的轻量 synthetic 定向检查均通过。实现随后提交为干净 commit `6d16976010324c73b4c3b7aa9e313fd7f0d358c4`。

完整 synthetic calibration 已由用户从该干净 commit 执行并通过，固定产物为：

```text
runs/crd_tf_v1/calibration/7e29795edc13fe8dc2e12ada8d619c22d0ae19fa8d13feb729d2f9b261fd5535/calibration.json
SHA-256: 044494e6c6966a98eb5dc00fb8bee1dcb68539910aeb9d750eb647781da7e3c0
```

Calibration 固定为 `complete=true / passed=true / synthetic_only=true`，`research_test_used=false / validation_target_used=false / real_waveform_used=false`。M/W/L/S 四项均通过；S 冻结选择为：

```text
smoothness_penalty = 2.0
suppression_radius_bins = 2
```

全量 fixed cache 随后由用户从同一干净 commit 执行完成，固定产物为：

```text
runs/crd_tf_v1/cache/bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0/cache_manifest.json
manifest SHA-256: 6fb44aad2689d9426ad78dc1f054db5aaac698792af5818bc01a54563cb9f0b8
transform SHA-256: bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0
```

Cache 审计结果：

- 完整 train/validation 为 `10141 / 2675` windows、`32 / 7` 个 `samp_id`；row-id SHA-256 分别为 `f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e / b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a`，集合交集为 0；
- 共 14 个 `.npy`/频率文件，逐文件 size、SHA-256、shape、dtype 与 finite 审计全部通过；总字节 `3,908,167,968`，即 `3.6398 GiB`，磁盘占用约 `3.7G`；
- `research_test_used=false / test_cache_created=false / target_read=false / model_inference_used=false`；
- config source selector 为 `bcg_rawish_segment_soft_z_key`，索引实际解析到唯一源 key `bcg_rawish_wideband_state_aligned_segment_soft_z`，train/validation 一致；
- calibration、candidate lock、dataset index、commit 与依赖 identity 均写入 manifest，candidate lock hash 保持不变。

运行环境的 distribution metadata 报告 `PyWavelets=1.9.0`，而 `pywt.__version__` 为 `1.8.0`；本阶段 W/S 实际使用并冻结的是 `ssqueezepy==0.6.6`，没有直接导入 `pywt`，因此该环境元数据差异不改变本 cache 数值，但作为复现环境已知异常保留，不把 PyWavelets 版本解释为变换 identity。

P1 至此关闭，calibration/cache 不得覆盖或重复生成。

## 17. P2 实现与冻结结果（2026-08-12）

P2 已实现并冻结以下工程契约：

- `TfV1CacheReader` 只允许 train/validation，通过冻结 manifest SHA-256、transform identity、row-id hash/order、文件 size/shape/dtype/finite receipt 读取 memory-map；只打开当前 variant 所需表示，CTRL1/2/3 不读取 cache；
- M/W/L/S 四个 encoder、统一 Stage-1 additive FiLM，以及只读 C201 temporal latent 的 CTRL1/2/3 已接入；各 branch 最终 gamma/beta projection 为严格 zero-init；
- C201 拆分为 `encode_local`/`decode_local`，相同 seed 的 CRD-TF 内部 base 与独立 C201 state 逐 tensor 完全一致，隔离原生 Mamba 后的 zero-init waveform 逐 tensor 完全一致；
- 15 个 variant 的 representation/control 映射、严格配置 schema、固定 cache path 与 `p2_cpu_only` execution gate 已实现；P2 配置显式 `early_stopping_enabled=false`；
- P5 汇总逻辑已预先实现为纯函数：formal matrix 完整性闸门、逐 paired-seed 二阶/三阶 interaction、mean/sample-SD/方向计数、base guardrail、实质改善和 tolerance-aware Pareto；当前不得对未产生的 P4 结果执行选择；
- 定向回归共 `60 passed`，新增 P2 data/model/selection 测试单独为 `15 passed`；Python 编译检查通过。CPU 测试只证明结构、身份和数据契约，不构成训练效果或目标 GPU 可运行性的证据。

P2 至此关闭；其后的 P3 实现、失败修订与最终结果见下文。P4 在用户明确确认正式矩阵成本前保持关闭。

## 18. P3 工程验收入口（2026-08-12）

P3 入口与准入规则如下：

- `scripts/check_crd_tf_v1_cuda.py` 在同一干净 commit 下依次检查全部 15 个 variant 的新增 encoder dropout=0、bf16 batch-1 forward/core-loss/backward、input/全部 parameter gradient finite、每个 active branch 最终 projection gradient 非零、精确 trainable increment 与 CUDA peak memory，并生成不可覆盖 receipt；
- `scripts/run_crd_tf_v1_acceptance.py` 只允许 `TF102-W / TF204-WL / TF302-WLS`，严格解析为 `1 epoch / 128 train / 32 validation / physical batch 128 / accumulation 1 / one update / p3_cuda_acceptance`；
- CRD-TF 的 `train_history.csv` 额外记录同步后的 train elapsed time 与 samples/s；旧 CRD 产物 schema 不变；
- `scripts/audit_crd_tf_v1_p3.py` 要求 synthetic 与三个 acceptance 来自同一干净 commit，审计 resolved config/cache identity、完整 lifecycle、两个 finite checkpoint、32 条 validation、五项 primary finite、prediction nondegenerate、吞吐和 `peak_reserved_fraction≤0.80`，成功后生成不可覆盖的 P3 receipt。
- P1–P3 相关非 GPU 定向回归在 branch checkpoint 修订后为 `73 passed`，三个入口的编译与 `--help` 检查通过。

P3 的运行命令冻结于 `scripts/README.md`。若任一 `128×1` acceptance OOM、非有限、生命周期失败或显存比例超过 80%，停止并由用户在 `64×2` 与 `32×4` 中确认统一 fallback；不得自行只调整失败 variant。P3 结果只作工程证据。

首次 CUDA synthetic receipt `3353c538b0ac_20260812_130914_034001` 的 15 项计算均 finite，但在结果准入审计时发现新增 TF `_TemporalMixer` 继承了通用 `ResidualDWBlock` 的 `Dropout(0.10)`，违反第 6 节“新增 encoder dropout=0”的冻结定义。该 receipt 明确作废，不进入 P3 工程证据，也不据此启动 acceptance；修订只把新增 TF mixer 的 dropout 固定为 0，不改变 C201 主干、参数量、表示、FiLM、数据或训练口径，修订提交后必须从新干净 commit 重跑全部 15 项 synthetic。

Dropout 修订后的 synthetic receipt `6b24125aef52_20260812_131401_247126` 在 RTX 4070 Ti SUPER 上 15/15 通过，dropout/finite/projection-gradient/参数增量契约均合格。随后最大 single `TF102-W` 的首轮 `128×1` acceptance 完成一次 update 与完整 validation/checkpoint lifecycle，但 peak allocated/reserved 为 `13,318.60/14,488 MiB`，reserved fraction `90.91%`，超过 80% 安全线，因此 run `20260812_131604_764333` 判定为工程失败，pair/triple 未启动，validation 数值不作解释。

该失败仍处于第 11.3 节预注册的“先用 chunking/activation checkpoint 争取 128×1”路径。为避免 W/L 等 branch 的 parameter-fill、temporal mixer 和 encoder 在 batch 128 下保留完整中间 activation，所有新增 representation/control branch 在训练态按 physical-batch 固定 `chunk=8`，使用 non-reentrant activation checkpoint 重算；eval 不分块。分块按 sample 轴，branch 内无 BatchNorm/dropout，数学输出与梯度的 CPU 等价测试通过，不改变 C201 主干、参数量、effective batch、update/LR 或表示。修订后必须在新干净 commit 重跑 15-arm synthetic 与 TF102 acceptance；在结果通过前仍不启动 pair/triple，也不触发梯度累计 fallback。

## 19. P3 冻结结果与 P4 命令计划（2026-08-12）

最终 P3 全部来自干净 commit `56cabf1d37fa01104902b6aeaef3d256bee6b2a1`。15-arm synthetic receipt 为：

```text
runs/crd_tf_v1/p3_cuda_synthetic/56cabf1d37fa_20260812_132151_006146/synthetic_receipt.json
SHA-256 = 071dc0143084093b58944c076b1336f7a324a9bb769fb63ae5ad52f36ba1992a
```

15/15 arm 的 dropout=0、bf16 output/input/parameter-gradient finite、active projection gradient nonzero、waveform shape 和参数增量全部通过。最终三个 `128×1` acceptance 均完成一次 update、32 条 validation、best/final finite checkpoint、五项 primary finite和 prediction nondegenerate：

| arm | trainable increment | total trainable | peak allocated MiB | peak reserved MiB | reserved fraction | lifecycle throughput samples/s |
|---|---:|---:|---:|---:|---:|---:|
| TF102-W | 150,048 | 1,219,850 | 8,752.67 | 10,112 | 63.45% | 7.543 |
| TF204-WL | 299,952 | 1,369,754 | 8,847.58 | 10,222 | 64.14% | 7.421 |
| TF302-WLS | 450,000 | 1,519,802 | 8,934.71 | 10,320 | 64.76% | 7.361 |

这里的 throughput 是独立单-update lifecycle 的冷启动工程值，不外推为 80-epoch 稳态速度。最大 reserved fraction 为 `64.76%`，因此 batch 决策冻结为 `physical_batch=128 / accumulation=1 / effective_batch=128`，不触发 fallback，也不新增 TF000 batch control。统一审计为：

```text
runs/crd_tf_v1/p3_acceptance_audit/0b4af9bd0c1c6441460702ad893cc5713f8ce3578cc055e51e86e60ad285234e/p3_acceptance.json
SHA-256 = d68790e5db45ce65f60a17badab535196a913466176b7ac52a5cac4910f49f14
status=passed, complete=true, research_test_used=false
```

P4 固定按 seed 外层、arm 内层运行；任何中间效果不得删减后续 arm：

| 顺序 | variant | reps | matched group |
|---:|---|---|---|
| 1 | `crd_tf_ctrl1` | none | single capacity |
| 2–5 | `crd_tf101_m`, `crd_tf102_w`, `crd_tf103_l`, `crd_tf104_s` | M, W, L, S | singles |
| 6 | `crd_tf_ctrl2` | none | pair capacity |
| 7–12 | `crd_tf201_mw`, `crd_tf202_ml`, `crd_tf203_ms`, `crd_tf204_wl`, `crd_tf205_ws`, `crd_tf206_ls` | MW, ML, MS, WL, WS, LS | pairs |
| 13 | `crd_tf_ctrl3` | none | triple capacity |
| 14–15 | `crd_tf301_mls`, `crd_tf302_wls` | MLS, WLS | triples |

Seed 顺序固定为 `20260811 → 20260812 → 20260813`，共 45 runs。未来每个 resolved command 固定为 `80 epochs / 128×1 / early_stopping=false / cuda:0`，输出根目录固定为：

```text
runs/crd_tf_v1/formal/<variant>/seed_<seed>/
```

为防止误启动，当前 config gate 仍故意拒绝 `p4_formal`。只有用户明确确认 45-run 长时间成本后，才允许单独提交 gate-only P4 runner；该 runner 必须由上述表生成 variant/reps，不允许手填漂移，并要求显式 `--confirm-45-run-matrix`。P3 至此关闭。
