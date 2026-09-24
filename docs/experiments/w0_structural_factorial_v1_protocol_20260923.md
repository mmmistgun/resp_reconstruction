# W0 波形前端 × CWT 后聚合时间块 × FiLM 后细化：结构因子对照 v1

日期：2026-09-23。协议 ID：`w0-structural-factorial-v1-es30p15-20260923`。

状态：**P1 模型、严格 spec、训练配置/实验适配器、冻结分析函数、排他生命周期、CLI 与 synthetic CPU 验收已完成；P2/P3 尚未开放。** 正式科学矩阵固定为 `2×2×2×3 seeds=24` 个全新 train/validation run。实现锁须在本轮代码提交并保持工作树干净后生成；GPU 工程验收、benchmark、真实数据 smoke 和正式训练在后续授权阶段开展。

输出 identity 固定为：

```text
runs/w0_structural_factorial_v1_es30p15/
```

未来实现锁固定使用新文件名：

```text
docs/experiments/w0_structural_factorial_v1_implementation_lock_20260923.json
```

本轮使用独立 worktree `/mnt/disk_code/marques/resp_reconstruction/.worktrees/w0_structural_factorial_v1`，分支为 `codex/w0-structural-factorial-v1`，同步基线为 `main` merge commit `9e86f5ad05064f2605ee9668c6afe154432263d2`。协议与后续实现均在该 worktree 中演化；冻结 checkpoint、cache 和产物从主仓库绝对路径只读引用，factorial 的实现锁、运行和结论保持独立身份。

## 1. 研究问题与证据属性

本轮以原始 W0 为结构中心，分别检验：

1. 用明确定义的卷积前端替换 W0 Patch 前端后，节律与形态重建如何变化；
2. CWT 分支在二维编码和尺度聚合之后的三层一维时间块是否提供独立价值；
3. FiLM 之后、统一读出之前的两层时间细化是否提供独立价值；
4. 三个结构因素之间是否存在互补、冗余或条件依赖；
5. 是否存在参数/计算更少、同时维持 RR、PCC 与包络质量的结构。

这是既有开发证据知情的 train/validation 实验。核心归因来自本轮统一合同下的完整 `2×2×2` 矩阵。未来如另行评价 test，证据属性固定为 reused research/development evidence。

## 2. 证据核对与当前状态

### 2.1 适用来源与冻结状态

本设计核对了：

- `crd_v1_protocol_20260808.md`；
- `crd_v1_controls_protocol_20260811.md`；
- `crd_tf_v1_protocol_20260812.md`；
- `crd_tf_v1_research_test_protocol_20260816.md`；
- `crd_tf_w_v2_protocol_20260817.md`；
- `e4_closeout_20260922.md`；
- 当前 W0/CRD-TF 源码、冻结配置、candidate lock；
- `resp_train.temporal.blocks.TemporalStem` 及其固定降采样实现。

本协议以已冻结 W0 为结构锚点。E4 的当前结论保留 W0 为论文主模型；CRD/CRD-TF 的已完成阶段提供数据、任务、训练、CWT、FiLM 与读出的只读来源。24-run 汇总的模型证据严格来自本协议 implementation lock 下生成的完整矩阵。

### 2.2 W0 精确身份、checkpoint 与历史训练合同

W0 的精确身份为：

```text
variant = crd_tf102_w
representations = [w]
base = crd_c201_decoder_10hz_cap
full-band 97-scale W branch
six Local BiMamba2 blocks
post-trunk FiLM
two refinement blocks
C201 10-Hz nonlinear decoder residual
```

正式配置模板为 `configs/crd_tf_v1/crd_tf102_w_formal.yaml`，candidate lock 中记录的模板 SHA-256 为 `4f7fb910871c4c6a33fa5e59919eff6e836b347c257830345ac9a036e73c071e`。当前 W0 来源锁为 `docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json`，SHA-256=`6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6`。W0 训练 commit 为 `68b3b85df2f18b3dc8ec59e02ac0780f2be42122`，全模型 trainable parameters 为 `1,219,850`，W 分支为 `150,048`。

三个冻结历史 checkpoint 为：

| Seed | Selected epoch | Run | Resolved config SHA-256 | Checkpoint SHA-256 |
|---:|---:|---|---|---|
| 20260811 | 13 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861` | `675232d48b7aadb2f0116e43a9cb737a5ac1737013dd54c4544e4f49cc1b2843` | `18425476831628d9614925dc6835de97ab466aac8cce157f7d95335f504c70f5` |
| 20260812 | 15 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130` | `497f6dfab8e350ee145a568deb6d72df631a81ada5dd2c00847b500323f74eeb` | `f6a174f88067bd88f0739faee6ce3c7cada79e8b43171a6a3d2e12a9798a5a61` |
| 20260813 | 14 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006` | `572711e26097b089e2e190d7e583ca59da0f162dfd5fb0421331cc7932981cc9` | `efafbb653507fbee782f5946a07660121737bfd5339ff2b13b81b4b1e3a0fe99` |

历史 W0 使用完整 `80 epochs / 6,400 optimizer updates`，不启用 early stopping；physical batch `128`、accumulation `1`，每 epoch `80` updates，按完整 validation Local RR 严格最小值选点，并列取最早。历史 validation 三 seed mean ± sample SD 为：

| Whole RR | Local RR | Trajectory | Global modulation | Signed PCC |
|---:|---:|---:|---:|---:|
| 0.499298 ± 0.018320 | 0.551309 ± 0.012064 | 0.152729 ± 0.001423 | 0.191051 ± 0.003022 | 0.865300 ± 0.001491 |

历史 reused research-test 对应为 `0.617234 / 0.609566 / 0.139546 / 0.173418 / 0.876577`。这些数字不进入本轮 selector 或 factorial contrast。

本轮完整 W0 结构必须作为 `PATCH/TM3/REF2` 单元重新训练三 seed，不能用历史 checkpoint 填格。原因是本轮统一采用新的 `30/15/0/80` 停止合同，核心比较需要同一实现锁、运行环境和生命周期。历史回放表明该规则能保留三个原 W0 选点并约在 epoch 30 停止，但新训练是否逐点复现不预设为事实。新 run 使用独立 identity，不覆盖历史 W0。

## 3. W0 真实计算图

输入为 `x∈R^(B×1×18000)`，100 Hz、180 s。模型输出 raw waveform，不在模型内部重复执行正式任务算子。外置任务算子继续为 `Pi=S(B_0.05–0.70Hz(.))`。

### 3.1 Patch 波形前端

W0 的 `PatchTokenFrontend` 由 `PatchMixer1D` 的 token encoder 与 CRD bridge 组成，参数量为 `8,784`：

```text
x [B,1,18000]
→ 右侧补 48 个零至 18048
→ unfold(patch_len=256, stride=128): 140 个重叠 patch
→ Linear(256,16)
→ 2 × PatchMixerBlock
→ [B,16,140]
→ linear interpolate 140→1800, align_corners=false
→ Conv1d(16,96,k1,bias=false)
→ GroupNorm(12,96,eps=1e-5,affine=true)
→ SiLU
→ [B,96,1800]
```

每个 `PatchMixerBlock` 为：

```text
x + Conv1d-dw(k3)→Conv1d(k1)→GELU(GroupNorm(1,16)(x))
→ residual + Conv1d(16,32,k1)→GELU→Conv1d(32,16,k1)(GroupNorm(1,16)(.))
```

这里的 GroupNorm 对通道和时间共同取组内统计。实际前端计算止于 token encoder 与 bridge；140→1800 使用长度兼容插值，其时间坐标沿用该插值网格。

### 3.2 六层双向 Mamba 主干

波形前端输出进入六层 `D=96` 的 `BidirectionalMamba2Block`。每层包含外部 RMSNorm、独立前向/反向 Mamba2、方向拼接后的 `Linear(192,96)`、dropout `0.10` 与残差。Mamba2 固定：

```text
d_state=64, d_conv=4, expand=2, headdim=32,
ngroups=1, chunk_size=256, rmsnorm=true,
bias=false, conv_bias=true, use_mem_eff_path=true
```

主干宽度、深度、方向合并、dropout 和 Mamba 依赖在八组中固定。

### 3.3 CWT 表示、二维编码与后续时间处理

W0 的 W 输入来自冻结 CRD-TF cache：`ssqueezepy==0.6.6`、analytic Morlet `mu=13.4`、`log1p(abs(CWT))`，原始 100-Hz 整窗 reflect 边界，目标为 12 voices/octave 的 97 scales，并对每 50 个原时间点做不重叠算术均值得到 `[97,360]`。实际 mapped centers 为 `0.03662109375–7.99560546875 Hz`，含一个重复中心；实际 scale identity、顺序、缓存、scale mean、频带、voices、mother wavelet 和边界在八组中固定。

W 编码真实流为：

```text
w [B,97,360]
→ Conv2d(1,48,k=(5,3),p=(2,1),bias=false)
→ GroupNorm(8,48) → SiLU
→ depthwise Conv2d(48,k=(3,3),p=1,groups=48,bias=false)
→ Conv2d(48,96,k=1,bias=true) → SiLU
→ 对 scale 维算术平均
→ linear interpolate 360→1800, align_corners=false
→ 3 × ResidualDWBlock(96,dilation=1/2/4,dropout=0)
→ active fill: residual Conv1d(96,65,k1,bias=false)→SiLU→Conv1d(65,96,k1,bias=true)
→ zero-init Conv1d(96,192,k1,bias=true)
→ split gamma_raw,beta_raw
```

B 因子的估计对象严格限定为尺度聚合后的三层一维 temporal mixer。B=TM0 时，两个二维卷积的时间核、360→1800 插值、active fill 和条件投影保持固定。

### 3.4 FiLM、post-FiLM refinement 与统一读出

令六层主干输出为 `z`，则：

```text
g = 0.5*tanh(gamma_raw)
b = 0.5*tanh(beta_raw)
z_film = z*(1+g)+b
```

FiLM 始终位于六层主干之后、refinement/decoder 之前。本轮固定 gamma/beta 系数为 W0 的 `0.5/0.5`。

W0 的 post-FiLM refinement 为两层 `ResidualDWBlock(96,dilation=1/2,dropout=0.10)`。之后的统一读出固定为：

```text
GroupNorm(12,96)
→ Conv1d(96,64,k5,p2,bias=false) → SiLU
→ depthwise Conv1d(64,k5,p2,groups=64,bias=false) → SiLU
→ Conv1d(64,32,k1,bias=false) → SiLU
→ base Conv1d(32,1,k1,bias=true)
+ C201 residual Conv1d(32,32,k1,bias=false)→SiLU→zero-init Conv1d(32,1,k1,bias=true)
→ waveform_10hz
→ Fourier interpolation 1800→18000
→ raw waveform
```

C 因子的估计对象严格限定为 FiLM 后的两层 refinement；统一读出中的 `k5` 时间卷积保持固定。

## 4. 现有证据与本轮用途

1. W 表示在 validation 与 reused research-test 上呈现 RR/trajectory 价值，因此本轮把 W 数学表示、scale identity 和 FiLM 位置作为固定条件源。
2. 冻结 checkpoint 干预与 E4 R3/GN 控制显示，W0 会利用局部时频演化和时间对齐。本轮通过 B 因子进一步识别尺度聚合后三层一维 mixer 的增量作用。
3. C0 的 10-Hz/Fourier round-trip 审计支持统一使用 10-Hz readout；本轮固定 coarse head、decoder residual 与 Fourier interpolation。
4. 既有结构对照反复呈现 RR、trajectory、global modulation、PCC 与效率之间的属性交换，因此本轮并列报告五主指标和资源，不构造单指标赢家。

## 5. 三个因素的冻结定义

### 5.1 A：波形前端 package

`A=0 PATCH` 为第 3.1 节原 W0 `PatchTokenFrontend`。

`A=1 CONV20` 固定为 `resp_train.temporal.blocks.TemporalStem` 的 exact contract：

```text
x [B,1,18000] at 100 Hz
→ fixed Kaiser line-boundary FIR: down=5, 255 taps, cutoff=9.0 Hz
→ [B,1,3600] at 20 Hz
→ Conv1d(1,48,k21,p10,bias=false)
→ channel-only LayerNorm per time step → SiLU
→ depthwise Conv1d(48,k5,p2,groups=48,bias=false)
→ channel-only LayerNorm per time step → SiLU
→ Conv1d(48,96,k5,p2,bias=false)
→ channel-only LayerNorm per time step → SiLU
→ fixed Kaiser line-boundary FIR: down=2, 127 taps, cutoff=4.5 Hz
→ [B,96,1800] at 10 Hz
```

两个固定 decimator 在 autocast 外执行；学习卷积按原 BF16 路径执行。输出第 `i` 个 token 对应 `0.1i s` 网格。Conv 前端 trainable parameters 为 `24,672`；`255+127` 个固定 FIR values 作为不可训练 buffers 登记。

A 比较的是完整 frontend package：采样路径、时间网格、卷积、SiLU、channel-only LayerNorm、边界、参数量和计算量共同构成 `CONV20 package vs PATCH package` 的效应。

### 5.2 B：CWT 尺度聚合后的一维时间 mixer

- `B=1 TM3`：保留三层 `ResidualDWBlock(96,d=1/2/4)`，其 dropout 已固定为 `0`。
- `B=0 TM0`：把 `branches.w.temporal` 替换为严格 `Identity`。

B=0 时保持二维 CWT encoder、scale mean、插值、`96→65→96` active fill、zero-init final projection、FiLM 系数和位置完全相同；TM3 参数从模型、optimizer 和 state dict 中移除。

### 5.3 C：FiLM 后 refinement

- `C=1 REF2`：保留两层 `ResidualDWBlock(96,d=1/2,dropout=0.10)`。
- `C=0 REF0`：把 `base.refinement` 替换为严格 `Identity`。

C=0 时保持 FiLM、完整 coarse head、C201 decoder residual 与 Fourier interpolation不变，不添加补偿层，也不保留未使用的 REF2 参数。

B 与 C 是两个不同的 module package：B 为三层、dilation `1/2/4`、dropout `0`；C 为两层、dilation `1/2`、dropout `0.10`。B×C 用于描述两者的互补或冗余，位置比较不属于本矩阵的估计目标。

## 6. 完整 `2×2×2` 矩阵

| Arm ID | A 前端 | B CWT-TM | C refinement | Trainable params | 角色 |
|---|---|---|---|---:|---|
| `sfv1_patch_tm3_ref2` | PATCH | TM3 | REF2 | 1,219,850 | 同轮完整 W0 |
| `sfv1_patch_tm3_ref0` | PATCH | TM3 | REF0 | 1,144,586 | Patch 下移除 C |
| `sfv1_patch_tm0_ref2` | PATCH | TM0 | REF2 | 1,106,954 | Patch 下移除 B |
| `sfv1_patch_tm0_ref0` | PATCH | TM0 | REF0 | 1,031,690 | Patch 下同时移除 B/C |
| `sfv1_conv20_tm3_ref2` | CONV20 | TM3 | REF2 | 1,235,738 | Conv 完整结构 |
| `sfv1_conv20_tm3_ref0` | CONV20 | TM3 | REF0 | 1,160,474 | Conv 下移除 C |
| `sfv1_conv20_tm0_ref2` | CONV20 | TM0 | REF2 | 1,122,842 | Conv 下移除 B |
| `sfv1_conv20_tm0_ref0` | CONV20 | TM0 | REF0 | 1,047,578 | Conv 下同时移除 B/C |

每组固定 seeds `20260811 / 20260812 / 20260813`，共 24 个新 formal training runs。矩阵在启动前整体冻结，执行与汇总均要求八组 × 三 seed 完整覆盖。

参数差来自自然结构：

- CONV20 相对 PATCH：`+15,888`；
- TM3：`112,896`；
- REF2：`75,264`。

各 arm 使用自然参数量。W0 原有、参与 forward 的 `96→65→96` active fill 在全部八组保持不变。

## 7. 初始化与配对合同

### 7.1 构造顺序

每个 arm 必须先用同一 `training.seed=model.initialization_seed` 构造完整原生 W0，再做以下后置替换：

1. A=CONV20 时，在独立命名子 seed `sfv1_frontend_conv20` 下替换 `base.frontend`；A=PATCH 保留原 `patch_frontend`。
2. B=TM0 时，在完整 W branch 构造完成后把 `branches.w.temporal` 替换为 `Identity`。
3. C=REF0 时，在完整 base 构造完成后把 `base.refinement` 替换为 `Identity`。

这样可以避免因删模块改变后续 RNG 消耗。相同 seed 下：

- 六层 Mamba、二维 W encoder、active fill、FiLM projection、head 和 decoder 在八组逐 tensor 相同；
- 同 A 的 frontend 在四个 B/C 组合中逐 tensor 相同；
- 同 B=TM3 的 temporal mixer 在四个 A/C 组合中逐 tensor 相同；
- 同 C=REF2 的 refinement 在四个 A/B 组合中逐 tensor 相同。

每个 run 保存逐 state tensor hash、optimizer parameter names/groups 和 shared-state comparison。TM0/REF0 对应模块不注册参数；固定 FIR buffers 单独登记且保持不可训练。

### 7.2 初始函数与梯度启动

`ResidualDWBlock` 的末层 `project` weight/bias 为零，因此 TM3 和 REF2 初始化时都是 identity。W branch 的最终 `96→192` projection 也为零，因此新模型初始 `gamma_raw=beta_raw=0`。

所以对同一个 A：四种 B/C 组合的初始 waveform 应逐 tensor 相同；A=PATCH 与 A=CONV20 不要求初始 waveform 相同。

零初始化带来明确的优化启动延迟：

- 第一次反向时，W final projection 可获得梯度，但其上游二维 encoder、TM3 和 active fill 的梯度为零；projection 更新后梯度才可进入上游。
- 梯度首次进入 TM3 时，各 block 的 zero-init project 可先更新，而 block 内 norm/depthwise/expand 的有效学习还会再晚一步。
- REF2 的 project 可在首次反向更新，但其内部 norm/depthwise/expand 也存在一步启动延迟。

AdamW 对已经生成零梯度 tensor 的非零 decay 参数仍可能施加 decoupled weight decay，故“早期上游梯度为零”不等于参数状态绝对冻结。实现验收必须记录连续至少三次 synthetic update 的逐模块 gradient/update norm；不另加非零初始化、warm-start、独立 LR 或初始化搜索。

## 8. 固定数据、训练与评价合同

### 8.1 数据与任务

- 数据格式、dataset root、index、input/target key、admission、subject/session 隔离均沿用 W0 `research_v2`。
- 完整 train/validation 为 `10,141 / 2,675` windows、`32 / 7` 个 `samp_id`；本协议的数据访问范围固定为 train/validation。
- sample seed 固定 train=`20260610`、validation=`20260611`；正式 training seeds 固定三项。
- 原 W train/validation cache identity固定为 transform SHA-256 `bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0`，manifest SHA-256 `6fb44aad2689d9426ad78dc1f054db5aaac698792af5818bc01a54563cb9f0b8`。实现锁逐项复核 manifest、row IDs、W/frequency 文件字节，并以只读方式使用现有 cache。
- loss 固定为 `L_sync + 0.25 L_effort`，外置 `Pi`、target-only eligibility、finite 与 penalty 语义不变。

### 8.2 Optimizer、batch 与学习率

八组统一：

```text
optimizer = AdamW
physical batch = 128
gradient accumulation = 1
effective batch = 128
drop_last = false
updates_per_epoch = ceil(10141/128) = 80
max_lr = 3e-4
min_lr = 3e-5
betas = (0.9,0.999)
eps = 1e-8
weight_decay = 1e-4
grad_clip_norm = 1.0
AMP = bf16
resume = false
```

no-decay 继续覆盖 bias、GroupNorm/LayerNorm/RMSNorm scale 和 Mamba `A_log/D/dt_bias`；其余参数按原 W0 分组。全部参数组使用相同 LR，不按 frontend、TM3 或 REF2 设置独立 LR。

LR 固定按完整 `80×80=6,400` planned updates 生成：前 5% 即 320 updates 线性 warm-up，之后 cosine 到 `3e-5`。提前停止只减少实际执行的尾部 updates，不压缩或重标定 LR 曲线。

### 8.3 Early stopping 与 checkpoint selector

统一停止合同：

```text
enabled = true
min_epochs = 30
patience = 15
min_delta = 0
max_epochs = 80
monitor = full-validation Local RR MAE
mode = min
```

每次完整 validation 后，只有严格降低才更新 best 并清零 wait；并列不算改善并保留更早 checkpoint。wait 从 epoch 1 累计，但只有 `epoch≥30` 才允许停止。若 epoch 30 刷新 best，最早在 epoch 45 停止。达到 epoch 80 时记 `max_epochs`。

终止时保存真实 final checkpoint，再重载 best checkpoint完成完整 validation 评价。manifest 必须记录实际 epochs/updates、planned updates、停止原因、best epoch/value、wait count 和 LR 位置。

相对历史 W0，唯一训练口径变化是统一 early stopping；optimizer、batch、LR 轨迹、loss、数据与 selector 不变。所有八组同时采用这一变化，因此本轮因子归因不把停止策略作为第四因素；本轮结果与历史 W0 的绝对差仍只能作历史参照。

## 9. 参数、计算与显存报告

参数量使用原生实例逐模块实测并与第 6 节静态数相符。不得把少参数自动解释为高效，也不得用参数量抵消质量退化。

预注册的线性算子 covered-MAC 口径为每窗口、一次乘加记 1 MAC，不含 norm、激活、插值、tanh、残差加法、内存传输或未列出的公共模块：

- PATCH frontend：`3,710,080`；
- CONV20 frontend：`110,300,400`，含两次固定 FIR 与三层 learned Conv；
- 每个 96-channel `ResidualDWBlock`：`67,219,200`；
- TM3：`201,657,600`；
- REF2：`134,438,400`。

因此仅三个变化因素的 covered-MAC 为：

| Arm | Factor covered-MAC/window |
|---|---:|
| PATCH/TM3/REF2 | 339,806,080 |
| PATCH/TM3/REF0 | 205,367,680 |
| PATCH/TM0/REF2 | 138,148,480 |
| PATCH/TM0/REF0 | 3,710,080 |
| CONV20/TM3/REF2 | 446,396,400 |
| CONV20/TM3/REF0 | 311,958,000 |
| CONV20/TM0/REF2 | 244,738,800 |
| CONV20/TM0/REF0 | 110,300,400 |

这些不是完整模型 FLOPs。未来实现必须另行报告：完整解析算子覆盖清单、batch-1 eval latency、batch-128 原生 train update latency、warm throughput、peak allocated/reserved、设备总显存及实际 epochs/总 wall time。benchmark 在同硬件/依赖/进程隔离下交替运行八组，不包含 cache I/O；正式训练资源另行记录，二者不混表。

## 10. 预先冻结的分析方案

### 10.1 五主指标与统计单位

五主指标继续为：

1. Whole RR absolute error，越低越好；
2. Local RR MAE，越低越好；
3. envelope trajectory MAE，越低越好；
4. global envelope modulation error，越低越好；
5. lag-aware signed PCC，越高越好。

每个 seed 先对完整 validation eligible samples 做 direct mean；八组分别报告三 seed arithmetic mean 与 sample SD (`ddof=1`)。三个 seed描述训练随机性，不作为三个独立人群；重叠窗口不作为独立样本做确认性推断，不计算 seed-level 或 window-level p-value。

### 10.2 因子效应与交互

记 `Y_abc` 为同一 seed、同一指标的 sample-direct mean，`a=1` 为 CONV20、`b=1` 为 TM3、`c=1` 为 REF2。对每个 seed 先计算，再跨三 seed 求 mean/sample SD：

```text
A main = mean_b,c(Y_1bc - Y_0bc)
B main = mean_a,c(Y_a1c - Y_a0c)
C main = mean_a,b(Y_ab1 - Y_ab0)

AB = mean_c[(Y_11c-Y_10c) - (Y_01c-Y_00c)]
AC = mean_b[(Y_1b1-Y_1b0) - (Y_0b1-Y_0b0)]
BC = mean_a[(Y_a11-Y_a10) - (Y_a01-Y_a00)]

ABC = Y_111-Y_110-Y_101-Y_011+Y_100+Y_010+Y_001-Y_000
```

同时完整报告每个因素在另两个因素四种组合下的 conditional simple effect。四项 error 的负差表示因子从 0→1 改善，PCC 的正差表示改善。另以同 seed `PATCH/TM3/REF2` 为参照，把 error 差除以参照值、PCC 保留绝对差，生成方向统一的辅助表；factorial 主结论仍保留原始单位，不构造跨指标总分。

交互的符号只表示一个因素效应随另一因素状态变化，不自动等于总体收益。三阶交互只回答 B/C 的联合依赖是否随 A 改变，不作生理因果解释。

### 10.3 Local RR 尾部、受试者与分母

对每个 arm×seed，在 `local_rr_target_eligible=true` 的窗口级 `local_rr_mae_bpm` 上报告：

- eligible 分母；
- mean、median、P90、P95、max；
- `>2 bpm` 与 `>5 bpm` 的计数和比例。

这里是每个 180-s 样本内多个 60-s、15-s step 频谱主峰误差的窗口内汇总，再看 2,675 个 validation 样本的分布；不能解释为逐呼吸相位跟踪或独立事件率。

完整受试者分层固定为每个 `arm×seed×samp_id`：样本数、五主指标 eligibility 分母、五指标 mean/median、Local RR P90/P95 与 `>2/>5` 比例。另先在 `seed×samp_id` 内聚合，再对 7 个 `samp_id` 等权形成 subject-macro 描述；不得用窗口多的主体获得更高权重。factor simple/main/interaction 也生成逐 `samp_id` 描述表，但不据 7 个主体报告确认性显著性。

每个 arm 必须报告：总 rows、唯一 row IDs、唯一 samp IDs、`whole_rr_target_eligible`、`local_rr_target_eligible`、`joint_target_eligible`、IBI target eligible/interpretable、各 envelope rank 指标有效分母，以及 prediction-degeneracy 计数。非有限值只能与冻结 eligibility 语义一致；不得删行、填补或缩小分母。

## 11. 预先冻结的解释与决策规则

材料性容差沿用近期结构实验：四项 error 相对同轮 W0 `0.5%`，PCC 绝对 `0.002`。容差用于描述和简化判断，不是统计显著性阈值。

### 11.1 单因素模块判断

对 A、B、C 分别使用 equal-weight main effect、W0 邻域 simple effect和 paired-seed方向：

- **支持保留/采用**：至少一项主指标达到材料性改善且至少 `2/3` seeds 同方向；其余四项均未超过不利容差；W0 邻域 simple effect不出现材料性反向损害。
- **支持简化/移除**：关闭该模块的 W0 邻域 arm 相对同轮完整 W0 在五项上均处于容差内；equal-weight main effect没有显示该模块的材料性收益；关闭后存在真实参数缩减，并报告实测资源变化。
- **结构依赖、继续研究**：出现材料性交互、不同上下文效应反号、改善与退化并存、seed方向不稳定，或 main effect掩盖某个 W0 邻域的明显损害。
- **当前候选净负面**：至少一项材料性退化、无任何材料性改善，且退化至少 `2/3` seeds 同方向。

若因素的 main effect 与 W0 邻域 simple effect结论冲突，以“context-dependent”报告，不用主效应覆盖条件效应。

### 11.2 整体简化结构

任一参数更少的 arm 只有同时满足下列条件，才称为 `quality-preserving structural simplification`：

1. 相对同轮 `sfv1_patch_tm3_ref2`，四项 error 的三-seed mean 恶化均不超过 `0.5%`，PCC 下降不超过 `0.002`；
2. Local RR、trajectory、PCC 各有至少 `2/3` paired seeds 处于相同容差内；
3. 生命周期、finite、分母和 prediction-degeneracy 验收通过；
4. trainable parameters 确实减少。

若同时达到 steady-state throughput `≥10%` 提升或 peak allocated `≥15%` 降低，可附加称为 `measured efficiency simplification`；否则只称结构/参数简化。资源收益不能挽救质量门槛失败。

多个 arm 均通过时，保留五主指标 tolerance-aware Pareto set；仅在质量处于同一容差层时，依次以参数更少、实测 throughput 更高、peak allocated 更低作为工程排序，不强制唯一科学赢家。

RR、PCC、trajectory 与 global modulation 发生材料性交换时，明确标记 `attribute trade-off`，不按单项均值挑选赢家。已有 research-test 已参与研究背景，任何未来复用都不升级证据独立性。

## 12. 工程与正式阶段门槛

### P0：本协议

- [x] 核对 W0 variant/config/checkpoint/training contract；
- [x] 还原 Patch、W encoder、BiMamba、FiLM、refinement 与读出真实计算图；
- [x] 核对适用上游协议、冻结来源与当前源码；
- [x] 冻结八组前向、参数、初始化、训练和分析合同；
- [x] P0 证据范围限定为文档、源码、manifest、summary 与文件身份元数据。

### P1：实现与 synthetic CPU（已完成）

实现使用本协议独立的模型、配置、控制器和输出 namespace，并保持既有 builder/state identities 兼容。已实现并验证：八组 strict schema、参数数、optimizer 只含 active parameters、forward shape/finite、同 A 的 B/C 初始 waveform identity、共享 state、三步梯度启动、optimizer 分组、early-stop 回放、factorial contrast、Local RR tails、subject/denominator 汇总和排他 lifecycle。

实现路径为：

```text
configs/w0_structural_factorial_v1/experiment.yaml
resp_train/paper_evidence/w0_structural_factorial_v1_model.py
resp_train/paper_evidence/w0_structural_factorial_v1.py
scripts/run_w0_structural_factorial_v1.py
tests/test_w0_structural_factorial_v1.py
```

专项测试与相邻 W0/TemporalStem 回归共 `49 passed`；Python 编译、`check-config` 和 `describe` 均通过。P1 receipt 为 `docs/experiments/w0_structural_factorial_v1_p1_implementation_receipt_20260924.json`。这些结果使用 synthetic CPU fixture，只形成结构、配置、初始化、分析和生命周期工程证据。

### P2：未来 GPU 工程验收

八组均做 batch-1 BF16 forward/loss/backward finite 与梯度检查；最大资源臂 `CONV20/TM3/REF2` 做 physical-batch-128 原生至少三次 update和完整 128-train/32-validation lifecycle。记录所有 arm 的参数、covered MAC、延迟、吞吐和显存。任一必须运行 arm OOM、非有限或 peak reserved 超过设备 80% 时暂停整个 formal 队列；不得只给该 arm 改 batch、accumulation、dtype、checkpointing 或结构。若统一 fallback，须在任何 formal 前修订协议并应用八组。

### P3：未来 24-run formal

只有 P1/P2 完成、实现锁冻结且用户再次明确授权长时间 GPU 后开放。输出固定为：

```text
runs/w0_structural_factorial_v1_es30p15/formal/<arm>/seed_<seed>/<lock-prefix>_<UTC>_<UUID>/
```

每个 attempt 先排他写 started receipt；失败保留现场。成功保存 resolved config、命令、Git/源码/依赖/环境、数据与 cache identity、初始化 hashes、optimizer groups、完整 history、best/final checkpoint、逐 sample metrics、summary、资源与完成回执。相同 lock/arm/seed 的完成 run 拒绝重复。

### P4：一次性冻结汇总

汇总输入严格限定为当前 implementation lock 产生的 24 个唯一完整 run。逐 run 回放 earliest legal stop、最早 Local-RR minimum、实际 updates/LR、checkpoint、2,675 row identity/order、指标/eligibility/denominator和finite；八组 × 三 seed 全部通过后生成一次性结果。

预期最小汇总产物：

```text
seed_primary_metrics.csv
arm_primary_summary.csv
conditional_effects_by_seed.csv
factorial_effects_by_seed.csv
factorial_effects_across_seed.csv
local_rr_tail_summary.csv
subject_stratified_metrics.csv
subject_factorial_effects.csv
metric_denominators.csv
parameter_compute_memory.csv
decision.json
source_manifest.json
freeze_receipt.json
```

## 13. 风险与归因边界

1. A 是完整前端 package，对采样、归一化、激活、边界、参数和计算同时变化；不能作单算子因果归因。
2. B 只移除 CWT scale-mean 后的一维 mixer；二维 CWT 时间卷积和后续 active fill 仍在。
3. C 只移除 FiLM 后 refinement；读出仍有时间卷积。
4. B/C 虽初始化为 identity，但参数化、梯度启动、dropout 和 weight decay 使优化轨迹不同；初始输出相同不等于严格容量无关。
5. 自然参数/计算差异是设计的一部分。质量差异可能同时包含表示、容量和优化效应；资源须并列报告。
6. 三 seed 足以描述固定随机种子的优化波动和交互方向，不足以作人群显著性推断。7 个 validation `samp_id` 的分层也只是有限开发样本上的异质性描述。
7. Local RR 是窗口级频谱主峰误差，不是逐呼吸相位跟踪；trajectory/PCC/global modulation 必须共同解释。
8. 统一 early stopping减少平均成本，但新组合可能晚恢复；max 80 仍是截断边界。实际停止 epoch作为优化行为报告，不用于事后延长单组。
9. 当前 test 已被多轮开发使用。未来 test 阶段须另立 checkpoint allowlist 和专项授权，并保持 validation-selected checkpoint 与完整三 seed 集合。

## 14. 适用边界与后续问题

本节记录当前矩阵的解释范围、固定处理和预先安排的后续问题。所有条目在 24-run 启动前冻结。

| ID | 当前固定项 | 解释范围 | v1 合同 | 后续触发 |
|---|---|---|---|---|
| `C01_W_ACTIVE_FILL` | W 分支固定 `96→65→96` active fill，共 `12,576` 个参数；宽度 65 继承 CRD-TF v1 的 branch 参数预算。 | B 估计 TM3 在该 active fill 之上的增量价值。 | 八组使用同一个 fill state、宽度和初始化。 | 若 B 主效应接近零、B×C 明显，或 TM0 成为简化候选，另立 `fill × TM3` 小矩阵。 |
| `C02_FRONTEND_PACKAGE` | CONV20 使用 exact `TemporalStem` contract：固定宽度、卷积核、三次 channel-only LayerNorm 和两级降采样。 | A 效应覆盖采样、网格、边界、归一化、激活、局部编码、参数和计算的完整 package。 | 以 `CONV20 package vs PATCH package` 报告质量与资源。 | 若 CONV20 呈现稳定收益或明确质量—计算交换，另立单因素前端机制对照。 |
| `C03_BC_MODULE_SCOPE` | B 为三层、dilation `1/2/4`、dropout `0`；C 为两层、dilation `1/2`、dropout `0.10`。 | B×C 描述两个真实 package 的互补或冗余。 | B、C 分别报告条件效应和主效应。 | 若结果提示位置机制，另立完全匹配 block count/dilation/dropout/初始化的 pre/post placement 对照。 |
| `C04_ZERO_INIT_STARTUP` | W final projection、TM3 residual project 和 REF2 residual project 为级联零初始化。 | 不同路径进入深层参数学习的 optimizer step 可能不同。 | 保留 W0 初始化；synthetic 连续三步记录逐模块 gradient/update norm。 | 若正式 history 在停止边界附近仍改善，或模块梯度长期接近零，先做只读优化诊断；初始化实验使用新协议和 identity。 |
| `C05_SHARED_TRAINING_CONTRACT` | 八组共用 AdamW、LR、weight decay、batch 与 `30/15/0/80`。 | 本轮估计统一训练合同下的结构表现；实际 epoch、wall-time 和优化轨迹分别报告。 | 全部 arm 使用同一训练合同，并保存停止 epoch、updates、LR 位置、wall-time 和梯度诊断。 | 工程条件无法满足共同合同时，formal 整体暂停并在全部 seed 启动前统一修订。 |
| `C06_C201_DECODER_SCOPE` | 统一读出固定 `1,057` 参数的 zero-init nonlinear decoder residual。 | C 效应条件于当前 decoder capacity。 | 八组固定 C201 residual 和读出。 | 若 REF0 成为质量保持简化候选，另立 `REF2 × decoder residual` 对照。 |

后续处理遵循两条规则：

1. 24-run 启动前，静态实现审计如发现影响比较有效性的条件，统一修订协议 ID、实现锁和全部八组。
2. 24-run 启动后按冻结矩阵完成；后续问题在完整结果冻结后使用新的专项协议和输出 identity。

## 15. 协议冻结结论

完整 `2×2×2` 的三个因素定义为：

1. A：`PATCH` 与冻结的 `CONV20` 完整前端 package；
2. B：CWT scale-mean 后的三层一维 mixer `TM3/TM0`；
3. C：FiLM 后两层 refinement `REF2/REF0`，统一读出保持固定。

八组具有明确前向、配对初始化、统一训练合同和预注册的主效应、条件效应及二阶/三阶交互。后续实现继续使用当前独立 worktree和本协议输出 identity。
