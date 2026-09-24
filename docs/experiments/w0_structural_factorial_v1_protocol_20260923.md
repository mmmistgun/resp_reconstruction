# W0 波形前端 × CWT 后聚合时间块 × FiLM 后细化：结构因子对照 v1

日期：2026-09-23。协议 ID：`w0-structural-factorial-v1-es30p15-20260923`。

状态：**证据核对与实验设计已完成；当前只形成协议，不实现模型、不读取原始波形或 cache 数组、不运行 smoke/GPU/训练/test。** 正式科学矩阵固定为 `2×2×2×3 seeds=24` 个全新 train/validation run。任何实现、工程验收和正式训练均须在后续单独授权后进行。2026-09-24 已将分支基线快进至包含 E6 结项的 `main`；该同步只更新历史来源状态，不改变本协议矩阵。

输出 identity 固定为：

```text
runs/w0_structural_factorial_v1_es30p15/
```

未来实现锁固定使用新文件名：

```text
docs/experiments/w0_structural_factorial_v1_implementation_lock_20260923.json
```

本轮使用独立 worktree `/mnt/disk_code/marques/resp_reconstruction/.worktrees/w0_structural_factorial_v1`，分支为 `codex/w0-structural-factorial-v1`，同步基线为包含 E6 结项的 `main` merge commit `9e86f5ad05064f2605ee9668c6afe154432263d2`。协议与后续实现均在该 worktree 中演化；历史 checkpoint、cache 和产物继续从主仓库绝对路径只读引用，factorial 的实现锁、运行和结论保持独立身份。

## 1. 研究问题与证据属性

本轮以原始 W0 为结构中心，分别检验：

1. 用明确定义的卷积前端替换 W0 Patch 前端后，节律与形态重建如何变化；
2. CWT 分支在二维编码和尺度聚合之后的三层一维时间块是否提供独立价值；
3. FiLM 之后、统一读出之前的两层时间细化是否提供独立价值；
4. 三个结构因素之间是否存在互补、冗余或条件依赖；
5. 是否存在参数/计算更少、同时维持 RR、PCC 与包络质量的结构。

这是既有 validation 与重复使用 research-test 结果知情的开发性实验。核心归因只来自本轮统一合同下的完整 `2×2×2` train/validation 矩阵；历史 W0、E4、E5、E6、CRD-TF-W v2 和 research-test 只作来源、设计动机或历史参照。即使未来另行评价 test，也只能称为 reused research/development evidence，不能称为新的独立确认。

## 2. 证据核对与当前状态

### 2.1 权威来源与状态优先级

本设计核对了：

- `crd_v1_protocol_20260808.md`；
- `crd_v1_controls_protocol_20260811.md`；
- `crd_tf_v1_protocol_20260812.md`；
- `crd_tf_v1_research_test_protocol_20260816.md`；
- `crd_tf_w_v2_protocol_20260817.md`；
- `e4_closeout_20260922.md`；
- `w0_film_gamma_training_results_20260919.md`；
- 当前 W0/CRD-TF 源码、冻结配置、candidate lock；
- E5/E6 前端协议、源码与已有产物索引。

状态判断按“当前结项/产物 → 新协议 → 早期协议快照”排序，不能把早期正文中的“未执行”当作当前事实：

- CRD-v1、C0/C1/C2、CRD-TF v1、CRD-TF-W v2、E4 均已完成相应冻结阶段；关闭入口不得重跑。
- E4 当前结论仍是保留 W0 为论文主模型；五个尺度聚合候选没有跨 validation/test、跨属性的稳定整体收益。
- E5 的 `100→10 Hz` 抗混叠极简卷积前端已完成三 seed validation 和 reused research-test，均支持 `net negative for this candidate`。这只否定 E5 完整 package，不证明所有卷积前端无效。
- E6 已完成实现锁、GPU acceptance、benchmark、三 seed formal/validation、固定 checkpoint reused research-test 和结项。结项记录为 `docs/experiments/e6_temporal_frontend_closeout_20260924.md`；最终决定是不替换 W0。

E6 全部产物作为排除于本矩阵的历史开发证据保留，不纳入 24-run 汇总。本轮卷积前端的选择依据是 E6 formal 结果产生前已经冻结的结构理由和 exact `TemporalStem`；后续看到的 E6 validation/test 结果不修改本协议的矩阵、阈值、初始化或停止规则。

### 2.2 W0 精确身份、checkpoint 与历史训练合同

W0 不是泛称，而是：

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

W0 的 `PatchTokenFrontend` 不是完整 `PatchMixer1D` waveform 模型，也不注册 patch waveform head；其参数量为 `8,784`：

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

这里的 GroupNorm 对通道和时间共同取组内统计。W0 只复用 token encoder，`hann` overlap-add 权重和 patch head 不进入该前端。140→1800 是长度兼容插值，不是严格物理对齐的 10-Hz 重采样。

### 3.2 六层双向 Mamba 主干

波形前端输出进入六层 `D=96` 的 `BidirectionalMamba2Block`。每层包含外部 RMSNorm、独立前向/反向 Mamba2、方向拼接后的 `Linear(192,96)`、dropout `0.10` 与残差。Mamba2 固定：

```text
d_state=64, d_conv=4, expand=2, headdim=32,
ngroups=1, chunk_size=256, rmsnorm=true,
bias=false, conv_bias=true, use_mem_eff_path=true
```

本轮不改变主干宽度、深度、方向合并、dropout 或 Mamba 依赖。

### 3.3 CWT 表示、二维编码与后续时间处理

W0 的 W 输入来自冻结 CRD-TF cache：`ssqueezepy==0.6.6`、analytic Morlet `mu=13.4`、`log1p(abs(CWT))`，原始 100-Hz 整窗 reflect 边界，目标为 12 voices/octave 的 97 scales，并对每 50 个原时间点做不重叠均值得到 `[97,360]`。实际 mapped centers 为 `0.03662109375–7.99560546875 Hz`，含一个重复中心；本轮保持实际 scale identity、顺序、缓存和 scale mean，不修改频带、voices、mother wavelet或边界。

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

因此，B 因子只能解释为“尺度聚合后的三层一维 temporal mixer”，不能写成“移除 CWT 的全部时间处理”。即使 B 关闭，两个二维卷积的时间核、360→1800 插值、active fill 和条件投影仍保留。

### 3.4 FiLM、post-FiLM refinement 与统一读出

令六层主干输出为 `z`，则：

```text
g = 0.5*tanh(gamma_raw)
b = 0.5*tanh(beta_raw)
z_film = z*(1+g)+b
```

FiLM 始终位于六层主干之后、refinement/decoder 之前。本轮固定 gamma/beta 系数为原 W0 的 `0.5/0.5`；历史 `gamma=0.4` 的 validation 候选只作为“结论依赖 FiLM 强度”的风险提示，不在本矩阵中加入第四个因素。

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

读出本身仍含 `k5` 时间卷积。C 因子只改变 FiLM 后的两层 refinement；不能写成“移除了 FiLM 后所有时间卷积”或“无时间读出”。

## 4. 已有证据能支持什么

1. CRD-TF v1 的 W arm 在 validation 候选池与 reused research-test 上均有 RR/trajectory 价值；这支持保留 W 表示作为本轮固定条件源，不证明 W 分支内部每个子模块都必要。
2. CRD-TF-W v2 的 `CONDITION_OFF`、`TIME_MEAN` 与 `TIME_SHIFT_30S` 干预显示，冻结 W0 对条件路径、局部时频演化和时间对齐有功能依赖；E4 R3/GN 控制也显示破坏 CWT 高频区域的时间对应会稳定损害 trajectory/PCC。这些证据不能区分二维时间卷积、B 因子的三层一维 mixer、FiLM 或后续主干的贡献。
3. `BETA_ONLY` 与 `GAMMA_ONLY` 均未达到 quality-near，说明已联合训练 W0 同时依赖两条 FiLM 路径；本轮因此固定 FiLM 形式，不把 B/C 结果解释为 add-only 或 scale-only 结论。
4. gamma 系数重训显示 `0.4` 在 validation 上优于原 `0.5` 的若干轴，但 W0 主模型身份仍是 `0.5/0.5`。本轮回答原 W0 条件下的结构问题，不声称得到与所有 FiLM 强度无关的结论。
5. E5 的直接 `100→10 Hz` 极简卷积 package 在 validation/test 均整体退化；因此本轮不重复 E5，而采用在 20 Hz 完成学习非线性后再降至 10 Hz 的冻结 Conv package。
6. C0 已证明最终正式呼吸带在 10-Hz/Fourier round-trip 上近似无损；这支持固定统一 10-Hz readout，但不证明原始 100-Hz BCG 可在任何学习变换前直接降到 10 Hz。
7. C1、D4、W3、E4 聚合等已有对照揭示了 RR、trajectory、global modulation、PCC 与效率之间反复出现的交换。因此本轮不以单一 Local RR 均值选赢家。

## 5. 三个因素的冻结定义

### 5.1 A：波形前端 package

`A=0 PATCH` 为第 3.1 节原 W0 `PatchTokenFrontend`。

`A=1 CONV20` 固定为已经在 E6 实现锁中定义的 exact `TemporalStem`：

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

两个固定 decimator 在 autocast 外执行；学习卷积按原 BF16 路径执行。输出第 `i` 个 token 对应 `0.1i s` 网格。Conv 前端 trainable parameters 为 `24,672`，固定 FIR values 为 `255+127`，不进入 optimizer。

A 比较的是完整 frontend package：采样路径、时间网格、卷积、SiLU、channel-only LayerNorm、边界、参数量和计算量都随之改变。结论只能写成 `CONV20 package vs PATCH package`；不得把差异单独归因于卷积、20 Hz、抗混叠、归一化或 carrier 解调。

### 5.2 B：CWT 尺度聚合后的一维时间 mixer

- `B=1 TM3`：保留三层 `ResidualDWBlock(96,d=1/2/4)`，其 dropout 已固定为 `0`。
- `B=0 TM0`：把 `branches.w.temporal` 替换为严格 `Identity`。

B=0 时必须保持二维 CWT encoder、scale mean、插值、`96→65→96` active fill、zero-init final projection、FiLM 系数和位置完全相同。不重新扩大 active fill，不添加 pointwise/normalization/minimal TCN，不保留未使用的 TM3 参数。

### 5.3 C：FiLM 后 refinement

- `C=1 REF2`：保留两层 `ResidualDWBlock(96,d=1/2,dropout=0.10)`。
- `C=0 REF0`：把 `base.refinement` 替换为严格 `Identity`。

C=0 时保持 FiLM、完整 coarse head、C201 decoder residual 与 Fourier interpolation不变，不添加补偿层，也不保留未使用的 REF2 参数。

B 与 C 使用相同 block 家族，但并非“同一模块只换位置”：B 为三层、dilation `1/2/4`、dropout `0`；C 为两层、dilation `1/2`、dropout `0.10`。因此不能把 B/C 效应大小之差直接解释为纯位置效应。

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

每组固定 seeds `20260811 / 20260812 / 20260813`，共 24 个新 formal training runs。不得根据首个 seed、部分 arm、中间 validation、历史 E6 结果或任何 test 结果取消剩余格子。

参数差来自自然结构：

- CONV20 相对 PATCH：`+15,888`；
- TM3：`112,896`；
- REF2：`75,264`。

不使用 inert/unused parameter fill 做跨 arm 参数对齐。W0 原有、参与 forward 的 `96→65→96` active fill 在全部八组保持不变；它不是为本轮新加的参数补齐。

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

每个 run 保存逐 state tensor hash、optimizer parameter names/groups 和 shared-state comparison。关闭的模块不得注册、优化或进入 state dict。固定 FIR buffers 单独登记，不进入 optimizer。

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
- 完整 train/validation 为 `10,141 / 2,675` windows、`32 / 7` 个 `samp_id`；不读取 test。
- sample seed 固定 train=`20260610`、validation=`20260611`；正式 training seeds 固定三项。
- 原 W train/validation cache identity固定为 transform SHA-256 `bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0`，manifest SHA-256 `6fb44aad2689d9426ad78dc1f054db5aaac698792af5818bc01a54563cb9f0b8`。实现锁只读复核实际需要的 manifest、row IDs、W/frequency 文件字节；不得重建或覆盖 cache。
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
- [x] 核对已有证据、阶段关闭状态与 E6 结项身份；
- [x] 冻结八组前向、参数、初始化、训练和分析合同；
- [x] 未读取原始波形或 cache arrays，未执行本协议模型或实验；E6 聚合结果仅登记为协议冻结后的历史背景。

### P1：未来实现与 synthetic CPU

必须新建独立模型/配置/控制器，不修改旧 W0/E5/E6 variant identity。至少验证：八组 strict schema、参数数、无 unused 参数、forward shape/finite、同 A 的 B/C 初始 waveform identity、共享 state hash、三步梯度启动、optimizer 分组、early-stop 回放、factorial contrast、Local RR tails、subject/denominator 汇总、不可覆盖 lifecycle。

### P2：未来 GPU 工程验收

八组均做 batch-1 BF16 forward/loss/backward finite 与梯度检查；最大资源臂 `CONV20/TM3/REF2` 做 physical-batch-128 原生至少三次 update和完整 128-train/32-validation lifecycle。记录所有 arm 的参数、covered MAC、延迟、吞吐和显存。任一必须运行 arm OOM、非有限或 peak reserved 超过设备 80% 时暂停整个 formal 队列；不得只给该 arm 改 batch、accumulation、dtype、checkpointing 或结构。若统一 fallback，须在任何 formal 前修订协议并应用八组。

### P3：未来 24-run formal

只有 P1/P2 完成、实现锁冻结且用户再次明确授权长时间 GPU 后开放。输出固定为：

```text
runs/w0_structural_factorial_v1_es30p15/formal/<arm>/seed_<seed>/<lock-prefix>_<UTC>_<UUID>/
```

每个 attempt 先排他写 started receipt；失败保留现场。成功保存 resolved config、命令、Git/源码/依赖/环境、数据与 cache identity、初始化 hashes、optimizer groups、完整 history、best/final checkpoint、逐 sample metrics、summary、资源与完成回执。相同 lock/arm/seed 的完成 run 拒绝重复。

### P4：一次性冻结汇总

只接受 24 个唯一完整 run。逐 run 回放 earliest legal stop、最早 Local-RR minimum、实际 updates/LR、checkpoint、2,675 row identity/order、指标/eligibility/denominator和finite。缺任何格、混入历史 W0/E6 run、使用部分 seed或已有 summary 目录时拒绝生成结果。

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
9. E6 已独立结项；其聚合结果只登记为本协议冻结后的历史外部背景，不能修改矩阵、阈值、初始化或停止规则。
10. 当前 test 已被多轮开发使用。若本轮以后需要 test，必须另立 checkpoint allowlist 和专项授权；不得用 test重选 checkpoint、删除 seed或追加结构。

## 14. 本次交付结论

完整 `2×2×2` 可实施，但须采用上述三项最小修正：

1. A 不是泛化的“Conv”，而是冻结的 `CONV20` 完整前端 package；不重复已经失败的 E5 `100→10 Hz` package。
2. B 明确限定为 CWT scale-mean 后的三层一维 mixer；不声称移除全部 CWT 时间处理。
3. C 明确限定为 FiLM 后两层 refinement；后续统一读出的时间卷积固定。

在这三个边界下，八组具有清楚的前向定义、配对初始化和可解释的二阶/三阶交互。当前独立 worktree 已建立；后续实现继续使用该 worktree。E6 已按自身协议结项，factorial 不复用其 run、checkpoint 或生命周期。
