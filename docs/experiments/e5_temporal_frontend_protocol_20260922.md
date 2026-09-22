# E5：W0 时域前端替换专项方案

日期：2026-09-22。协议 ID：`e5-temporal-frontend-v1-20260922`。

状态：**来源与设计审计完成；经结构合理性复核，首轮候选已从 RTM `TemporalStem` 直复用修订为 W0 专用的抗混叠 10-Hz 局部残差前端。用户已于 2026-09-23 授权 P1 代码实现；独立模型、严格 spec、训练/汇总控制器、GPU/benchmark 入口和 synthetic CPU 测试已完成。实现锁仍等待干净提交后的真实 W cache 字节复核；GPU 验收、正式训练、真实数据 smoke、benchmark 和 test 均未授权执行。**

## 1. 科学问题、范围与独立身份

本方案承接 [E5 规划交接](e5_temporal_frontend_handoff_20260922.md)。唯一主问题是：

> 在固定原 W0 的 97-scale W 条件分支、六层 BiMamba2、FiLM、refinement/decoder、数据、损失、训练预算和 validation selector 时，用物理网格明确、保留有符号位移和局部形态的抗混叠时域前端替换旧 patch-token bridge，能否改善多属性呼吸重建？

首轮只比较一个新候选 `e5_tfe101_aa10_res_w0`，复用三个冻结 W0 对照 `e5_tfe000_patch_w0`，新增三个 seed 的从头训练。候选不是 RTM-v1 的补跑：只复用其已审计的固定 FIR/polyphase 算子和 train-only 信号证据，不复用面向无 W 条件模型的完整 `TemporalStem`；主干、W 分支和 decoder 均保持 W0。

当前授权覆盖独立 E5 模型、严格配置、控制器、CLI 和 synthetic CPU 验收；实现锁须在提交并保持工作树干净后另行准备。GPU 工程验收、效率测量和正式训练仍须用户另行授权。首轮正式科学范围为 train/validation。不得读取 test target、test prediction 或 test 指标来实现、调试、选择候选或解释 validation。

## 2. 来源核对与冻结基线

机器可读记录见 [E5 来源与设计审计](e5_temporal_frontend_source_audit_20260922.json)。本次审计起点为 commit `f83f2c0ef2ec2a1b6e05f2ae392de398077abeab`；开始时工作树只有用户提供的 E5 handoff 未跟踪，未覆盖或修改任何历史产物。

三个冻结 W0 对照为：

| Seed | Selected epoch | 冻结 run |
|---|---:|---|
| 20260811 | 13 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861` |
| 20260812 | 15 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130` |
| 20260813 | 14 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006` |

本次重新核验了三份 best checkpoint/config/manifest/validation metrics 的 SHA-256，均与 E4 来源审计登记值一致。三个 checkpoint 在 CPU 上对当前 `crd_tf102_w` 严格加载，194 个 state tensor 无缺失/多余，浮点/复数 state 全有限；epoch 为 `13/15/14`，全模型参数为 `1,219,850`，W 分支为 `150,048`，active fill 为 `96→65→96`。

W0 训练来自干净 commit `68b3b85df2f18b3dc8ec59e02ac0780f2be42122`。从该提交至当前 HEAD，W0 路径的变化只增加其他候选、可选深度和可配置 FiLM 系数；W0 仍取六层、97 scale、gamma/beta 系数 `0.5/0.5`。`frontends.py`、CRD blocks/初始化/training/loss 未改变 W0 数学路径；metrics 的后续变化是新增导出函数，原五主指标实现未改。E4 来源审计此前还验证过三个 seed 的历史/当前 W0 初始 state 逐 tensor 一致。

候选来源锁为：

- `resp_temporal_v1_signal_substrate_lock_20260820.json`，SHA-256 `11bfcad00f4532d4bdfe1413a375b5f06f46eb8ac67dfcd475701872322fee69`；
- `resp_temporal_v1_candidate_lock_20260820.json`，SHA-256 `b4a2c83310fa2ce9519e3ca25814aea0b179458ab52d6380a932545c99c25f9b`；
- `resp_temporal_v1_cpu_implementation_receipt_20260820.json`，SHA-256 `6ef3ca0d48e1ac819ff55bab4247d542c8bae0adfddb6951c8a026e81fea7872`。

当前 `resp_train/temporal/blocks.py`、`model.py`、`config.py` 和 `initialization.py` 的字节哈希仍与 exact CPU receipt 一致。该 receipt 已冻结通用 `FixedPolyphaseDecimator1D` 的边界实现、固定滤波器不可训练和梯度可穿透；RTM final stem 的 exact tests 覆盖 `100→20` 与 `20→10`。E5 设计审计另以 synthetic float64 核验了 `100→10 / 511 taps / 4.5 Hz` 实例，对同一 SciPy audit operator 的 max-abs error 为 `4.44e-16`，满足 `atol=rtol=5e-12`。E5 不重跑已关闭的 train-only signal audit，也不把 RTM-v1 的模型质量结果当作 E5 收益证据。

## 3. 原 W0 前端的真实语义

当前 `PatchTokenFrontend` 的参数量是 **8,784**，不是完整 B0 `PatchMixer1D` 的 11,408。后一个数字包含未被 W0 bridge 注册的 waveform patch head；W0 实际只注册 7,056 参数的 embedding/mixer encoder，再加 1,536 参数的 `16→96` adapter 和 192 个 GroupNorm 参数。

真实流为：

```text
B×1×18000 at 100 Hz
→ 右侧补 48 个零，长度 18048
→ unfold: patch_len=256, stride=128，140 个重叠 patch
→ Linear(256,16)
→ 2×PatchMixerBlock
→ B×16×140
→ linear interpolate(size=1800, align_corners=False)
→ Conv1d(16,96,k1,bias=False)
→ GroupNorm(12,96,eps=1e-5) → SiLU
→ B×96×1800
```

第 `j` 个 patch 覆盖原始索引 `[128j, 128j+255]`，名义中心为 `1.275+1.28j` 秒；末 patch 含 48 个右侧零。两层 `k3` token mixer 的局部可学习内容传播覆盖最多 5 个相邻 token，对应 7.68 秒跨度。每个 mixer 的两个 `GroupNorm(1,16)` 以及 adapter 后 `GroupNorm(12,96)` 都对时间维参与统计，因此前端还存在全窗统计依赖，但这不是可学习的远距离内容交换。

`align_corners=False` 的未截断 source coordinate 为

```text
u(i) = (i + 0.5) × 140 / 1800 − 0.5,  i=0…1799
```

前后各 6 个输出位置被边界截到首/末 token。若仅把 token 放在 patch 中心，内部位置对应约 `0.684778 + 0.0995556i` 秒，并非严格 `0.1i` 秒；而下游把数组索引当作 10-Hz latent 使用。因此旧前端是长度兼容的插值 bridge，不能表述为严格物理对齐的 10-Hz 重采样。

## 4. 唯一首轮候选 `e5_tfe101_aa10_res_w0`

### 4.1 为什么不直接采用 RTM `TemporalStem`

RTM stem 是为了让多个**没有 W 条件分支**的 temporal family 共享同一 substrate；它不是为 W0 的双路径分工设计。直接复用存在三个结构问题：

1. train-only audit 把 displacement 判为 primary observable mechanism；carrier 可观察，但相对 displacement 的互补性未被证明。串行 stem 让两者从一开始就在同一学习路径中纠缠。
2. W0 的 W 分支已从原始 100-Hz 输入计算覆盖约 `0.03–8 Hz` 的 CWT 对数幅度，并已有 R3 时间结构被当前模型利用的证据。时域前端再次建立完整 20-Hz carrier encoder 会与 W 路径角色重叠，而不是形成清楚互补。
3. `TemporalStem` 在每个时间点连续使用三次 channel-only LayerNorm。对局部近似成比例的响应，这会压缩幅度尺度；但 effort/envelope 是当前任务的显式目标。它在 RTM package 中可能可用，不等于在已有 W-FiLM 的 W0 中是最合理分工。

因此 E5 不把“已有实现”当作候选充分理由。新设计令时域前端负责有符号低频位移、局部相位/形态和精确 10-Hz 网格；高频 carrier 的幅度—时间结构继续由冻结 W 分支承担；长时间依赖继续由六层 BiMamba2 承担。

### 4.2 精确定义

候选使用一个固定降采样器和一个最小局部残差编码器：

```text
x: B×1×18000 at 100 Hz
→ fixed polyphase decimator: down=10, 511 taps, cutoff=4.5 Hz
x10: B×1×1800 at 10 Hz
→ Conv1d(1,96,k11,p5,bias=False)
z: B×96×1800
→ z + DWConv1d(96,k5,p2,bias=False)(SiLU(z))
→ B×96×1800
```

固定低通为 Kaiser `beta=8.6`、奇数对称 FIR、`padtype=line` 等价实现，从原始第 0 个样本开始每 10 点抽取；输出索引严格对应 `t=0.1i` 秒。`k11` 在 10 Hz 上覆盖 1.1 秒，depthwise residual 把最大可学习局部跨度扩展到 15 个点、即 1.5 秒，约为正式最高呼吸率 0.70 Hz 的一个周期。更长依赖不在前端堆叠，由后续六层 BiMamba2 建模。

`Conv1d(1,96,k11)` 将单通道局部波形映射到接口所需的 96 维滤波器响应；depthwise residual 只做每个局部响应的非线性形态修正，下一层 BiMamba2 已有完整 channel mixing，因此不再增加一个意义重复的前端 pointwise mixer。主卷积 Kaiming-normal 初始化；depthwise residual 权重严格 zero-init，使初始前端是有符号线性局部表示，同时 residual 第一轮即可获得非零梯度。输出端不加激活。

前端不设 GroupNorm、LayerNorm 或 BatchNorm。输入已是冻结的 180-s segment soft-z 表示；更重要的是，当前任务需要保留窗口内相对幅度起伏。六层 BiMamba2 自带 per-token RMS pre-norm 和 residual，refinement/head 也有既有归一化，不需要在输入端再通过时间或通道归一化消除幅度自由度。

固定 decimator 在 autocast 外执行：float64 输入保持 float64，其余输入用 float32 滤波；随后两个学习卷积进入原 BF16 autocast，最终 frontend 输出跟随卷积 dtype。FIR taps 以 float64 persistent buffer 保存，共 511 个非训练值。学习卷积使用 same-length zero padding；固定 FIR 为全局端点直线延拓，两种边界语义分别登记。

对远离边界的输出，511-tap FIR 半径为 255 个 100-Hz sample，两个学习卷积合计再增加 7 个 10-Hz token、即 70 个原始 sample，总半径 325，理论支持跨度 651 点、即 6.51 秒。首尾受 `line` 延拓影响的点还通过端点斜率依赖窗口两端，整个候选与 W0 主干仍是离线模型。

### 4.3 参数、初始化和独立构造

| 组件 | W0 patch 前端 | E5 anti-aliased 前端 |
|---|---:|---:|
| trainable parameters | 8,784 | 1,536 |
| fixed persistent values | 0 | 511 float64 FIR taps |
| 全模型参数 | 1,219,850 | 1,212,602 |
| 相对 W0 全模型变化 | — | −7,248（−0.5942%） |

E5 独立模型先按原顺序构造完整 `crd_tf102_w`，再在独立 `module_seed(seed, 'e5_temporal_frontend_aa10_residual')` 上用 E5 前端替换 `base.frontend`。固定 decimator 直接复用 `FixedPolyphaseDecimator1D`，学习编码器在 E5 独立文件中实现；不得向旧 CRD variant 注册表塞入新臂，也不得改变 RTM-v1 的类或锁。当前设计原型按这一构造模拟三个 seed，候选与 W0 除 `base.frontend.*` 外的 **165 个 state tensor 全部逐 tensor 相等**，每个候选参数量均为 1,212,602。

实现时须把这项原型检查升级为正式测试，同时核对 CPU/CUDA RNG 前后状态、optimizer 参数名及 decay/no-decay 分组。主卷积进入原生 weight-decay 组；zero-init depthwise residual 同样参与 forward、训练和 weight decay；固定 taps 不进 optimizer。候选从头训练，不加载已训练 W0 checkpoint；“公共 state 相等”只描述同 seed 初始化，不表示 warm-start。

### 4.4 选择理由和归因边界

选择该候选的理由不是参数接近，而是角色最少且可解释：固定 FIR 建立明确物理网格；线性 `k11` filterbank 保留有符号局部波形；单个 zero-init depthwise residual 提供有限的非线性形态修正；W 分支负责原始高频幅度结构；Mamba 负责长时依赖。宽度 96 由既有接口和主干维度决定，`k11/k5` 由 10-Hz 网格与最高呼吸率的局部周期决定，没有按 W0 参数量反推隐藏宽度。候选数值不是由 E5 validation/test 选择。

首轮仍只能回答“这个完整前端 package 相对旧 patch bridge 是否有收益”。它同时改变：

- patch embedding/mixer 与 10-Hz 局部卷积编码；
- 140-token 线性插值与显式 100→10 FIR decimation；
- GroupNorm 的跨时间统计与不设前端 normalization；
- 右侧零补/插值边界与 line-FIR + conv-zero-padding 边界；
- 前端参数容量、局部支持和 AMP 内部精度。

因此正结果不能单独归因于抗混叠、线性插值、去除前端归一化或局部编码；负结果也不能证明显式抗混叠或 raw-signal frontend 无效。它尤其不检验“时域前端独立保留 4.5–8 Hz carrier 是否有益”，因为该角色被明确交给固定 W 分支。140 个多通道 token 也不能仅凭 token 数量被判定为已经丢失呼吸信息。若首轮出现值得解释的收益，后续机制对照应由具体结果触发，不预先堆叠多个采样、归一化和容量变体。

## 5. 控制变量

除完整时域前端外，以下全部固定：

1. **W 条件路径**：同一 `[97,360]` float32 cache、CWT mean-scale aggregation、`CwtBranch`、三层 temporal mixer、active fill `96→65→96`、150,048 参数、chunk=8、zero-init final projection。
2. **主模型**：六层 D=96 BiMamba2、FiLM 位置、`0.5*tanh(gamma/beta)`、refinement、coarse head、1,057 参数 nonlinear decoder residual、10→100-Hz Fourier interpolation。
3. **数据**：research_v2，输入/target key、admission、subject/session 隔离、train/validation split、row order；train/validation 为 10,141/2,675 窗口、32/7 个 `samp_id`。
4. **seed**：training/model seed 均为 `20260811/20260812/20260813`；sample seed 固定 train=`20260610`、validation=`20260611`。
5. **目标与损失**：正式 `0.05–0.70 Hz` 投影、`L_sync + 0.25 L_effort`、eligibility/finite/penalty 语义不变。
6. **训练预算**：AdamW，最多 80 epochs / 6,400 planned updates；max/min LR `3e-4/3e-5`、5% planned-update exact warm-up cosine、betas `(0.9,0.999)`、eps `1e-8`、weight decay `1e-4`、grad clip 1.0、BF16、resume=false。Early stopping 固定监控完整 validation Local RR，`min_epoch=30 / patience=15 / min_delta=0`；等待计数从 epoch 1 累计，但只在 `epoch≥30` 时允许触发，因此最早停止于 epoch 30。LR 仍按 6,400 planned updates 计算，不因提前停止重标定。
7. **batch**：physical/effective batch 均为 128、accumulation=1、drop_last=false；shuffle、末 batch和随机流沿用原 W0。
8. **selector**：每 epoch 完整 validation 的 Local RR MAE 严格 `<` 最小，平局保留更早 epoch；永久保存 best/final，不按其他指标或部分 seed 改选。
9. **评价**：原五主指标、IBI/coverage 和 envelope strata 的实现、分母及 sample-direct-mean 聚合不变。

E4 已清理的 15 份聚合前 `FULL/x.npy` 与 E5 无关。E5 只需要原 CRD-TF W train/validation cache；实现锁准备时必须重新核验其 manifest、train/val row-ID 和实际所需文件字节身份，但不得重建或覆盖 cache。

## 6. 计算与效率口径

按每窗口、一次乘加记一个 MAC，仅统计明确列出的线性算子：

| 前端 | covered MAC/window |
|---|---:|
| W0 patch embedding + 2 mixers + adapter | 3,710,080 |
| E5 511-tap fixed FIR + k11 embedding + k5 depthwise residual | 3,684,600 |

E5 在这一受限口径约为 W0 的 0.993 倍；接近只是结构结果，不是设计约束或公平性门槛。该数字不含激活、插值、内存传输、Mamba、W 分支、FFT decoder 或 optimizer，不能称作完整模型 FLOPs。E5 参数更少也不自动代表质量或完整运行更高效，固定 FIR 的 float32 路径和 kernel 实现仍须实测。

未来 GPU benchmark 必须在同硬件、依赖、设备和 dtype 下，以独立进程串行比较原生 W0 与 E5；固定三组交替顺序，每项 warm-up 5 次、记录 20 次，CUDA synchronize 并重置峰值。分别测 batch-1 eval forward 与 batch-128 原生 training update，保存逐次延迟、median/IQR、samples/s、peak allocated/reserved、设备总显存和失败日志。W cache I/O 不计入模型内耗时；完整 FLOPs 在算子覆盖不足时保持未报告。

## 7. 实验矩阵与统计设计

| Arm | Seeds | 训练 | validation |
|---|---|---|---|
| `e5_tfe000_patch_w0` | 三个固定 seed | 复用冻结 W0 run | 复用对应 selected-checkpoint metrics |
| `e5_tfe101_aa10_res_w0` | 同三个 seed | 从头训练；每 seed 30–80 epochs、2,400–6,400 updates，三 seed planned 上限 19,200 updates | 每个 selected checkpoint 完整 2,675 窗口 |

五主指标并列报告：Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。Local RR 只承担 selector，不是唯一科学终点；不得构造加权总分或用 secondary 指标打破主指标取舍。

对四项 error，每个 seed 保存候选减 W0 的原始差和相对差；PCC 保存原值，并以 `W0−候选` 表示下降。统一令正 delta 表示候选更差。报告：

- 两臂三 seed 原值、arithmetic mean、sample SD（ddof=1）；
- 同 seed 配对 delta 的 mean、sample SD、改善/相等/恶化方向数；
- error 的逐 seed 相对变化均值及“三 seed 均值之比”，两者分栏；基线为零时相对差为 NA，不加 epsilon；
- 每个 seed×每个 validation `samp_id` 的配对差、7 个受试者等权宏平均及方向数；
- 参数、covered MAC、实测吞吐、延迟和显存代价。

受试者宏平均是稳健性描述，不替换冻结的 sample-direct-mean 主口径。重叠窗口、同一受试者的多个窗口和三个模型 seed 都不是新增独立受试者；不做窗口级确认性检验，也不把三 seed p-value 当作确认结论。

为预先约束“改善”措辞，沿用项目 tolerance：四项 error 相对 0.5%，PCC 绝对 0.002。候选相对 W0 的 validation candidate-mean：

- 五项均不劣于 tolerance、至少一项实质更优，且该改善至少 `2/3` paired seeds 同向：称为 **tolerance-aware net improvement**；
- 同时存在实质改善和实质退化：称为 **attribute trade-off**；
- 五项均在 tolerance 内：称为 **materially indistinguishable at this resolution**；
- 至少一项实质退化且无实质改善：称为 **net negative for this candidate**。

这是描述性决策规则，不是统计显著性。工程验收通过不等于科学假设成立；validation 正结果也不自动替换论文冻结主模型或开放 test。

计划汇总至少包含 `seed_metrics.csv`、`paired_seed_delta.csv`、`three_seed_comparison.csv`、`paired_seed_subject_delta.csv`、`subject_macro_by_seed.csv`、`parameter_compute_report.json`、`source_receipt.json` 和完成 manifest。每行绑定 arm、seed、split、selected epoch、来源 SHA 和指标分母。

## 8. 分阶段验收与门槛

### 8.1 P0：来源与设计（本次已完成）

- 三个冻结 W0 checkpoint 当前严格加载、finite、epoch/参数/哈希一致；
- RTM 科学锁、exact CPU receipt 与当前 fixed-decimator 源码哈希一致；E5 的 `100→10` 实例已完成 design-time float64 reference 核验；
- 原/候选真实张量流、时间网格、归一化、边界、参数和 covered MAC 已登记；
- 三 seed 的原型构造中 165 个非前端 state tensor 逐 tensor 相等；
- standalone synthetic 原型输出 `B×96×1800` 且 finite，input/主卷积/depthwise gradient 全 finite，zero-init depthwise 的 480 个权重首轮梯度全部非零；
- 未读取真实波形、test metrics 或 test cache，未运行训练、GPU 或 benchmark。

### 8.2 P1：synthetic CPU 实现验收

实现必须在不访问真实数据的前提下至少通过：

1. 独立 E5 config 严格 schema、未知字段/variant 拒绝，旧 CRD/RTM 构造与测试不变；
2. 输入、100→10、k11 embedding、zero-init k5 residual 和最终 shape/dtype/finite 合同；前端不得注册任何 normalization 或 pointwise mixer，参数精确为 `1,536/1,212,602`，511 个 fixed taps 不可训练且 persistent；
3. float64 `100→10` decimator 对冻结 reference operator 满足 `atol=rtol=5e-12`，float32/BF16 路径误差另行锁定实际值；固定滤波器 input gradient finite/nonzero；
4. zero-init 时前端输出须与 `k11_embedding(fixed_decimator(x))` 逐 tensor 相等，学习感受野静态核对为 15 个 10-Hz token；三 seed 的 165 个公共 state、W 分支、六层 trunk、FiLM/refinement/decoder 逐 tensor 相等，RNG 隔离和 optimizer 分组一致；
5. candidate state_dict 严格 round-trip，W0 checkpoint 不得用 `strict=False` 载入候选；旧 W0 checkpoint 对旧模型继续严格加载；
6. 常量、脉冲、首末边界和随机输入可重复；错误 shape、NaN/Inf、错误 cache keys 和非有限中间量显式失败；
7. synthetic W feature 下完整 candidate forward/loss/backward，input/全部 trainable gradient finite；首次 update 的 zero-init depthwise weight gradient必须非零，三次 update 内权重实际离零且主卷积持续有梯度；
8. tiny synthetic 原生 trainer fixture 验证末 batch、update/LR、严格最早 selector、best/final/history、失败 lifecycle、重复运行拒绝、三 seed 汇总完整性。

2026-09-23 的 P1 实现已通过 `tests/test_e5_temporal_frontend.py` synthetic CPU 定向测试，并与 W0/RTM 相关回归共同通过；Python 编译和独立 spec 检查通过。测试覆盖上述结构/数值/初始化/梯度/配置/early-stop 门槛/生命周期/汇总合同；未读取真实波形、W cache、test 或既有 prediction。精确测试计数和实现身份见 `docs/experiments/e5_temporal_frontend_p1_implementation_receipt_20260923.json`。实现锁尚未生成，因为它要求先提交本轮实现、保持工作树干净并重新核验真实 W train/validation cache 字节身份。CPU 验收不构成目标 GPU 可运行或模型质量证据。

### 8.3 P2：用户执行的 GPU 工程验收

GPU 验收须来自 P1 同一干净 commit，并至少包括：

- 三 seed batch-1 FP32/BF16 完整 forward，对前端、Mamba 输入、FiLM 输出和 waveform 记录 shape/dtype/finite；前端不同，**不要求**初始 waveform 与 W0 相等；
- 固定 seed 20260811、batch=128、synthetic input/W/target 的至少 3 次原生 loss/backward/clip/AdamW update，全部参数、梯度、optimizer state 和 prediction finite，主卷积与 depthwise residual 实际更新；
- 完整 `128×1` lifecycle acceptance：至少 128 train/32 validation、一次 update、best/final checkpoint、五指标 finite、prediction nondegenerate；
- peak reserved/device total 不超过 80%，并完成第 6 节匹配 benchmark；
- receipt 绑定实现锁、commit、Python/PyTorch/Mamba/SciPy、GPU/CUDA/cuDNN、AMP 与 benchmark 原始记录。

若 `128×1` OOM、非有限、lifecycle 失败或超过显存线，保留失败现场并暂停，不得只给候选静默改 batch、accumulation、dtype、FIR 精度或结构。若未来要用 `64×2` 或 `32×4`，必须修订协议并加入相同 substrate 的三 seed W0 batch control；否则不能与冻结 `128×1` W0 作严格训练对照。

### 8.4 P3：正式三 seed 训练与完成验收

只有 P2 成功且用户明确授权长时间 GPU 后才开放。任一 seed 开始后必须完成全部三个 seed；不得根据首个/部分 validation 结果取消、调参或改阈值。允许暂停队列的只有 OOM、非有限、identity、checkpoint/lifecycle 或 prediction-collapse 工程失败。

每个 attempt 先排他创建新目录并写 `lifecycle_started.json`。成功后保存 resolved config、命令、干净代码/来源/实现锁、数据/row/cache identity、seed、环境、访问记录、optimizer 分组、30–80 行连续 history、best/final checkpoint、完整 2,675 行逐窗口 metrics、summary、manifest 和 freeze receipt。工程失败写 `lifecycle_failed.json` 并保留现场；同合同排除执行故障后只能用新 attempt 从头运行失败 seed。完成 seed 禁止同 identity 重跑。

验收 history 从 epoch 1 连续到 completed epoch，completed epoch 必须在 `[30,80]`；若小于 80，末行必须满足 `epoch≥30` 且连续 15 个 epoch 无严格改善。核对 completed updates=`80×completed_epoch`、LR 仍对应 6,400 planned updates、最早严格 Local-RR minimum、final epoch/update、checkpoint config/epoch/history/optimizer step 和 early-stop state 一致；validation 必须恰有 2,675 个冻结 row、7 个 `samp_id`、无缺失/重复，五主指标 finite 且 eligibility/分母一致。Prediction degeneration 不删样本、不换 checkpoint；标记质量失败并如实披露。

三个成功 attempt 齐全后一次性生成汇总，显式排除并列出所有失败/中断 attempt，不静默吞掉。汇总只读冻结 W0 validation 来源和三个 E5 attempt，不打开 test 资源。

## 9. 独立实现与生命周期

当前实现使用以下独立路径，不修改旧实验身份：

| 职责 | 规划路径 |
|---|---|
| 候选模型 | `resp_train/paper_evidence/e5_temporal_frontend_model.py` |
| 配置、来源、训练与汇总控制 | `resp_train/paper_evidence/e5_temporal_frontend.py` |
| GPU 验收和 benchmark | `resp_train/paper_evidence/e5_temporal_frontend_engineering.py` |
| CLI | `scripts/run_e5_temporal_frontend.py` |
| strict config | `configs/e5_temporal_frontend/e5_tfe101_aa10_res_w0.yaml` |
| 定向测试 | `tests/test_e5_temporal_frontend.py` |

输出根建议为 `runs/e5_temporal_frontend/`，按 `implementation / engineering / formal / summary` 分 phase，每次 attempt 使用 lock 前缀、UTC 和 UUID；timestamp 不作为唯一 identity。目标已存在时 fail closed，所有产物不可覆盖。真实数据访问先写 scoped access intent，完成后在 receipt 中登记 train/validation/test/checkpoint/cache 的实际访问布尔值。

普通 `build_crd_model`、RTM builder 和旧配置校验器均未放宽。E5 的 `TemporalFrontendExperiment` 只覆盖 `_build_model()` 和 validation identity 检查，复用原生 CRD trainer/loss/metrics/scheduler/selector；独立 spec 派生完整 resolved config，普通 CRD loader 会拒绝 E5 专属字段。

## 10. Test 与结论边界

本专项当前没有 test 协议、入口或执行授权。三 seed train/validation 完成并锁定后，如仍需要跨 split 描述，须另立 E5 test 附件，固定三枚 validation-selected checkpoint、三个 W0 对照、2,310 窗口/8 个 `samp_id`、sample seed `20260612`、row/cache identity、五主指标、不可覆盖输出和当次用户授权；不得用 test 重选 checkpoint、调结构或只评价有利 seed。

既有 W0 test 和仓库过去多轮 research-test 已参与开发背景，未来 E5 test 只能称 reused research/development evidence，不是首次无偏外部确认。更强泛化结论需要未参与模型开发的新 cohort 或 prospective holdout。

本轮可以支持的最高表述是：在固定 W0 其余结构和训练合同、由 W 分支承担高频幅度条件的前提下，这个 10-Hz 显式抗混叠局部残差前端在冻结 validation 口径上呈现何种多属性收益、取舍及代价。不得外推为无 W 条件模型、某一种归一化、抗混叠或时域前端家族的普遍结论。

## 11. 当前交付与下一停止点

本阶段已完成：代码与冻结来源核对、候选重设计、独立模型、严格 spec、来源/训练/汇总控制器、GPU 工程入口、CLI、参数/计算报告和 synthetic CPU 定向测试。三 seed tiny fixture 验证了原生 trainer 的完整 history、严格最早 Local-RR selector、best/final checkpoint、optimizer step、失败 lifecycle、不可覆盖完成身份以及窗口/受试者配对汇总。

尚未创建 implementation lock，未读取真实波形/W cache/test，未运行 GPU、benchmark 或正式训练。下一步是先提交本轮实现，在干净工作树上由 `prepare-lock` 重新核验冻结 W0 与 train/validation W cache；该只读锁准备完成后，P2 GPU 验收仍需用户另行授权。
