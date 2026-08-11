# CRD-Net v1.1 S0/S1/S2 实现与实验协议（2026-08-08）

## 1. 权威性、范围与科学边界

本文是 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 35–44 节引用的规范性附件；发生冲突时以主协议为准。当前状态为：

- CRD-S0/S1/S1C/S1F：已完成并冻结，S2 BASE 为 candidate-lock 中的 CRD_102；
- CRD-S2A：202/203/204 九个 formal runs、prototype 描述与冻结 summary 已完成，三个单分支关闭；
- CRD-S2B-R：12 个 formal runs 与冻结汇总已完成，两个组合均不 eligible，保留 CRD_102；S3 关闭。
- CRD_102 failure diagnostic：指标分层与结果知情 metadata follow-up 均已完成，主要失败区域已定位但不作因果主张；当前无自动激活的新模型阶段。
- CRD_102 matched observability：21 个 exact-state primary pairs 与 28 个 same-samp sensitivity pairs 已完成，结果为 mixed observability/model-tracking；当前无自动激活的新实验。
- CRD_102 C0/C1/C2 控制线已由主协议第 45 节另立于 `docs/experiments/crd_v1_controls_protocol_20260811.md`；C0 已冻结为 `roundtrip_negligible=true`，C1 TCN 因 trajectory 护栏失败而保留 CRD_102并关闭，当前只激活 C2 decoder controls 的实现与工程验收。

`docs/temp/` 中的讨论稿只保留设计历史，不是运行依据。唯一激活过的 AM/Morphology 定义来自第 21 节，唯一激活过的双因素组合与 capacity control 来自第 22 节；gate、auxiliary、TCN control、S3 和最终消融仍未激活。

本阶段由第一轮 research-test 启发，但不回头修改旧 checkpoint 或旧结论。数据、split、target、正式算子 `Pi=S(B(.))`、core loss、评价指标和 Local-RR checkpoint selector 全部沿用主协议。S0/S1/S2 训练只读 train/validation；除第 19 节已经完成并关闭的 S1C 队列外，CRD research-test 仍禁止读取。

论文语言只允许称各分支为 mechanism-inspired representation；单通道 BCG 不支持“三个独立生理源已被识别”的可识别性主张。

## 2. 固定环境与失败语义

CRD 原生依赖固定为：

```text
mamba-ssm==2.3.2.post1
causal-conv1d==1.6.2.post1
```

同一最小 pin 文件保存在 `configs/crd_v1/requirements.txt`；训练入口仍会在运行时核对实际安装版本。

Mamba2 固定使用官方 `from mamba_ssm import Mamba2` 和 `use_mem_eff_path=True`。依赖缺失、版本不等或目标 GPU 无法执行 forward/backward 时，CRD Mamba 路线阻塞；不得自动替换为仓库中的卷积式 “SSM-like” 模块，也不得静默关闭 fast path。

v1.1 不支持 resume。训练入口不接受 resume checkpoint，checkpoint 明确写入 `resume_supported=false`。若运行中断，从同一 resolved config、seed 和代码版本重新开始；不得把未保存 RNG/sampler 状态的续跑包装成等价正式 run。

## 3. 公共输入、输出和数值边界

- 输入：`x: [B,1,18000]`，100 Hz，180 s。
- CRD latent：`[B,96,1800]`，10 Hz。
- global latent：`[B,128,180]`，1 Hz。
- coarse head：raw `waveform_10hz: [B,1,1800]`。
- 模型正式输出：raw `waveform: [B,1,18000]`，不加 `tanh`，不在模型内部重复执行正式 `Pi`。
- S1 模型返回 `{"waveform", "waveform_10hz"}`；S0 wrapper 至少返回 `{"waveform"}`。loss/metrics 只消费 `waveform`。

Conv、Linear、Mamba 等神经层在 CUDA 训练时允许 bf16 autocast。以下协议数学必须关闭 autocast 并使用 float32：Direct FFT/filterbank、analytic IFFT、10→1 Hz hard LPF、Fourier interpolation、T4 STFT/log-magnitude、正式 `Pi`、`L_sync` 和 `L_effort` 的 FFT/RMS/log/correlation。CPU smoke 不启用 AMP，且不形成科研证据。

## 4. 确定性独立子 seed

每个模型的 `model.initialization_seed` 必须等于 `training.seed`。模块子 seed 定义为：

```text
SHA256("crd-v1.1:{base_seed}:{module_name}")
→ 取前 8 bytes（big-endian unsigned）
→ mod (2^63-1)
```

模块在隔离的 CPU RNG context 中构造。固定 module name 为：

```text
patch_frontend
direct_frontend
local_trunk
global_trunk
refinement
coarse_head
```

因此同一 seed 下，CRD_001、CRD_002 和 CRD_101/102 共有的 patch embedding 与两个 mixer 参数逐 tensor 相同；101/102 不注册、不优化未使用的原生 patch head。CRD_102/103/104 的 local trunk、refinement/head 也不受可选 frontend/global 模块是否存在影响。

## 5. 统一 ResidualDWBlock

所有 S1 一维 residual convolution 固定为：

```text
x: [B,C,T]
u = GroupNorm(G,C,eps=1e-5,affine=True)(x)
u = DepthwiseConv1d(C→C,k=5,dilation=d,padding=2d,groups=C,bias=False)(u)
u = SiLU(u)
u = Conv1d(C→2C,k=1,bias=False)(u)
u = SiLU(u)
u = Dropout(0.10)(u)
u = Conv1d(2C→C,k=1,bias=True)(u)
out = x + u
```

`C=96` 时 `G=12`，`C=128` 时 `G=16`。depthwise 和首个 pointwise weight 使用 Kaiming-normal；末层 weight/bias 为 0；GroupNorm weight=1、bias=0。每个 block 初始化时严格为 identity。

## 6. Legacy Patch frontend 与 decoder bridge

B0 固定为现有 `PatchMixer1D`：

```text
patch_len=256
patch_stride=128
base_channels=16
mixer_layers=2
overlap_window=hann
output_smoothing_kernel=1
```

18000 点按现有右侧 padding 规则产生 140 个 post-mixer tokens，shape 为 `[B,16,140]`。

CRD_101/102 bridge 固定执行：

```text
post-mixer tokens
→ linear interpolate 140→1800, align_corners=False
→ Conv1d 16→96,k=1,bias=False
→ GroupNorm(12,96,eps=1e-5)
→ SiLU
```

adapter Conv 使用 Kaiming-normal，GroupNorm weight=1/bias=0。

CRD_001 仍走原生 patch-head overlap-add。故 101 vs 001 的唯一主变化是 decoder/output representation package。

## 7. Direct analytic frontend

使用完整 18000 点 complex FFT 和六个 Gaussian analytic masks：

```text
G_k(f)=exp(-0.5*((f-c_k)/sigma_k)^2) * 1[0.03<=f<=0.70]
```

| k | center interval | init center | sigma interval | init sigma |
|---|---|---:|---|---:|
| 1 | [0.05,0.09] | 0.07 | [0.008,0.030] | 0.015 |
| 2 | [0.09,0.14] | 0.11 | [0.010,0.040] | 0.020 |
| 3 | [0.14,0.22] | 0.18 | [0.015,0.060] | 0.030 |
| 4 | [0.22,0.34] | 0.28 | [0.020,0.080] | 0.040 |
| 5 | [0.34,0.50] | 0.43 | [0.030,0.120] | 0.060 |
| 6 | [0.50,0.69] | 0.62 | [0.040,0.150] | 0.080 |

中心和带宽分别使用各自区间内的 bounded sigmoid raw parameter；raw parameter 用 inverse-logit 初始化以恢复表中数值，不加排序 penalty。对于正频率，`A_k[f]=2 X[f] G_k(f)`；DC、Nyquist 和所有负频率置零，complex IFFT 后按 `[real_1,imag_1,...,real_6,imag_6]` 得到 12 channels。随后直接 `[...,::10]` 降为 10 Hz。

神经 encoder 固定为：

```text
Conv1d 12→48,k=7,p=3,bias=False
GroupNorm(6,48); SiLU
Conv1d 48→96,k=1,bias=False
GroupNorm(12,96); SiLU
ResidualDWBlock(96,d=1)
ResidualDWBlock(96,d=2)
```

两层 Conv 使用 Kaiming-normal，norm weight=1/bias=0。输出为 `[B,96,1800]`。

## 8. Local / global Mamba 与 FiLM

外部 RMSNorm 唯一定义为：

```text
w * x / sqrt(mean(x^2,dim=-1,keepdim=True)+1e-5)
```

每个 BiMamba2 block 输入 `[B,L,D]`：

```text
u = external RMSNorm(x)
y_f = independent_forward_Mamba2(u)
y_b = flip(independent_backward_Mamba2(flip(u,time)),time)
y = Linear(concat(y_f,y_b),2D→D,bias=True)
y = Dropout(0.10)(y)
out = x + y
```

每个方向参数独立；不得用 sum/mean 合并。Local 固定 6 blocks、`D=96`；global 固定 4 blocks、`D=128`。两者 Mamba2 参数均为：

```text
d_state=64, d_conv=4, expand=2, headdim=32, ngroups=1,
chunk_size=256, rmsnorm=True, bias=False, conv_bias=True,
use_mem_eff_path=True
```

Mamba 内部初始化沿用固定版本官方实现；外部 merge Linear 显式使用 PyTorch `Linear.reset_parameters` 等价的 Kaiming-uniform (`a=sqrt(5)`) 与 `±1/sqrt(fan_in)` uniform bias。未在上述调用中覆盖的 Mamba2 constructor 参数使用固定版本 `2.3.2.post1` 的默认值。

global stage：

```text
local [B,96,1800] at 10 Hz
→ full-window FFT hard LPF 0–0.45 Hz
→ [...,::10] = [B,96,180]
→ Conv1d 96→128,k=1,bias=False
→ 4 × global BiMamba2
→ Fourier interpolation 180→1800
→ Conv1d 128→192,k=1,bias=True
→ split gamma,beta
```

96→128 projection 使用 Kaiming-normal；FiLM Conv weight/bias 为 0，反馈为：

```text
z_F = z_L * (1 + 0.5*tanh(gamma)) + 0.5*tanh(beta)
```

无 global stage 时 `z_F=z_L`。两类模型之后都经过完全相同的 `ResidualDWBlock(96,d=1)`、`ResidualDWBlock(96,d=2)` refinement。

## 9. Fourier interpolation 与 coarse head

周期 Fourier interpolation 固定使用：源 `rfft(norm="forward")`，目标频谱初始化为 0，复制源正频率 bins；偶数源长度的 Nyquist bin 不复制；目标 `irfft(norm="forward")`。用于 180→1800 和 1800→18000，始终 float32。

10-Hz head 固定为：

```text
GroupNorm(12,96,eps=1e-5)
Conv1d 96→64,k=5,p=2,bias=False; SiLU
DepthwiseConv1d 64→64,k=5,p=2,groups=64,bias=False; SiLU
Conv1d 64→32,k=1,bias=False; SiLU
Conv1d 32→1,k=1,bias=True
```

前三层 Kaiming-normal；末层 `Normal(0,0.02)`、bias=0；无 output activation。1800 点 raw head 经上述 Fourier interpolation 得到 18000 点 raw waveform，正式限带与规范化仍由外部 `Pi` 完成。

## 10. S0/S1 唯一模型表

| Variant | Trainable params | 唯一结构变化 |
|---|---:|---|
| `crd_001_b0_retrain` | 11,408 | 原 B0 PatchMixer；仅换新训练协议 |
| `crd_002_t4_retrain` | 12,752 | 原 T4：B0 + `STFT 2000/250, 0.05–8 Hz, bandenergy 16ch, native pre-mixer inject`；仅换新训练协议 |
| `crd_101_b0_coarse` | 117,361 | B0 post-mixer tokens + bridge + refinement + 10-Hz head |
| `crd_102_b0_local_mamba` | 1,068,745 | 101 在 bridge/refinement 之间增加 6 local BiMamba2 |
| `crd_103_direct_local_mamba` | 1,144,165 | 102 只把 B0 bridge frontend 换为 Direct analytic frontend |
| `crd_104_direct_hier_mamba` | 2,256,613 | 103 只增加 1-Hz global BiMamba2 + FiLM |
| `crd_105_direct_coarse` | 192,781 | 结果后诊断：Direct frontend + 101 refinement/head，不含 Mamba |

参数数目以固定依赖版本和本节构造为契约；单元测试必须在结构漂移时失败。

## 11. 训练逐 update 语义

正式 run 固定：

```text
epochs=80
physical batch=128
gradient accumulation=1
nominal effective batch=128
drop_last=False
optimizer=AdamW
max_lr=3e-4; min_lr=3e-5
betas=(0.9,0.999); eps=1e-8
weight_decay=1e-4
grad_clip_norm=1.0
CUDA autocast dtype=bfloat16
early stopping=False
resume=False
```

最后不足 128 个 samples 的 physical batch 仍执行一次 update。每个 accumulation group（正式配置下为一个 microbatch）在 forward 前用 target-only eligibility 计算 `N_sync` 与 `N_effort`。每个 microbatch 分别 backward：

```text
sync_weight * differentiable_sync_sum_micro / N_sync
+ effort_weight * differentiable_effort_sum_micro / N_effort
```

若某 component 的 group count 为 0，其 group contribution 是 graph-connected zero。禁止简单 `microbatch_mean/4`。训练日志的 epoch component mean 由全 epoch detached numerator/count 汇总；它不改变逐 group 优化目标。

AdamW 默认所有参数 decay。以下明确 no-decay：所有 bias；GroupNorm/LayerNorm/RMSNorm scale；Gaussian center/bandwidth logits；未来 prototype `P`；Mamba2 的 `A_log`、`D`、`dt_bias`。所有 parameter groups 共用同一 LR。

设完整 train dataset 大小为 `N`：

```text
M = ceil(N/128)                        # microbatches/epoch
S = M                                  # optimizer updates/epoch
U = 80*S
W = floor(0.05*U), formal 下 W>=1
u = 0,...,U-1
```

当前完整 train admission 为 `N=10141`，因此 `M=S=80`、`U=6400`、`W=320`。这与原 `32×4` 的 `ceil(ceil(10141/32)/4)=80` 完全相同，batch-scaling 修订不改变 optimizer update 数或 LR 序列。

LR 在 `optimizer.step()` 前设置：

```text
u < W:
  lr = max_lr*(u+1)/W
u >= W:
  p = (u-W)/(U-W-1)
  lr = min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*p))
```

每组顺序严格为 `zero_grad → all micro forward/backward → clip → set_lr(u) → step → u+=1`，不用 PyTorch scheduler object。正式 checkpoint selector 仍是 validation Local RR MAE，严格 `<` 更新，因此完全相同时保留更早 epoch；80 epochs 全部执行。

## 12. 配置角色与正式 seed

正式 seed 固定为 `20260811 / 20260812 / 20260813`，通过 `training.seed` 覆盖；`model.initialization_seed` 解析为同一值。六个原 S0/S1 配置、一个结果后诊断配置和一个 S1F 配置位于 `configs/crd_v1/`。

- `run_role=formal`：必须 80/128/1，所有 `max_*_windows=null`，结果可进入 S0/S1 validation 证据。
- `run_role=acceptance`：固定 1 epoch、batch 128、accumulation 1、train/val windows 128/32、test null；只验收正式物理 batch、显存和生命周期，不形成科研结论。
- `run_role=smoke`：最多 2 epochs、batch≤32、accumulation≤4，train/val windows 均限制在 1..32，test null；只作工程检查。

如果正式 batch 128 在后续候选或目标硬件上 OOM，不得在命令行临时降 batch。应先按相同工程口径复验 runner-up `64×2`，修订本协议，并在该阶段任何正式 seed 启动前统一应用。

### 12.1 正式运行前的 batch-scaling 工程 benchmark

在任何正式 run 启动前，允许对 CRD_103 使用同一批 128 个 training windows 比较 `32×4 / 64×2 / 128×1`。这三种方案的 effective batch 都严格为 128，每次 repeat 都从相同初始化和样本顺序开始，只执行一次 optimizer update，不运行 validation/test，结果不构成模型效果证据。

固定执行 3 repeats：repeat 1 标为 cold，只用于触发 shape/runtime warmup；repeat 2/3 的 samples/s 中位数作为稳态吞吐。计时范围包含 DataLoader collate、host-to-device、target-only eligibility、全部 forward/backward、gradient clipping、AdamW 首次 state 建立与 optimizer step，不包含数据预载、模型构建或 validation。显存同时记录 PyTorch `max_memory_allocated` 与 `max_memory_reserved`，并取全部 repeats 的最大值。

工程选择规则在运行前固定：候选必须全部 repeats finite/无 OOM，peak reserved 不超过设备总显存的 80%，且相对 `32×4` 的稳态吞吐至少提升 10%；满足者中选择稳态 samples/s 最高者，否则保留 `32×4`。任何变更都必须在正式 seed 之前同步本文、全部激活配置、loader 校验和测试；effective batch、optimizer update 数和 LR 公式不变。

benchmark 已于 2026-08-08 在 RTX 4070 Ti SUPER 16 GiB、CRD_103、固定 128 个 training windows 上完成。三种方案各 3/3 repeats finite、每次恰好一个 update、`N_sync=N_effort=128`：

| 方案 | 稳态耗时中位数（s/update） | 稳态吞吐（samples/s） | Peak allocated（MiB） | Peak reserved（MiB） | Reserved/总显存 | 相对 32×4 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| `32×4` | 0.592876 | 215.899 | 2490.71 | 3114 | 19.53% | 基准 |
| `64×2` | 0.512504 | 249.812 | 4744.46 | 5448 | 34.17% | +15.71% |
| `128×1` | 0.454950 | 281.438 | 9407.40 | 10868 | 68.17% | +30.36% |

首个 `32×4` cold repeat 为 15.696 s，证实此前约 80 s acceptance 主要混入首次 kernel 编译、数据准备和 validation，不能代表稳态 step。三种 batch 边界下 dropout 随机流不同，因此单步 loss 有轻微差异；这些 loss 不用于模型选择。按预注册工程规则，`128×1` 同时满足最高吞吐、超过 10% 增益和低于 80% reserved 安全线，故从本节起正式协议改为 physical batch 128、accumulation 1；effective batch、每 epoch update 数与 LR 序列不变。原 `32×4` acceptance 只保留为工程历史。

修订后的完整生命周期 acceptance 随后在 `/tmp/crd_103_batch128_acceptance/20260808_035213_501764` 通过：128 个 train windows 形成一次 update，32 个 validation windows 完成 Local-RR selector、best/final checkpoint 和逐 sample metrics；checkpoint 配置为 `128×1` 且所有 model tensors finite。该结果仍只属于工程验收。

## 13. S1 顺序决策表

所有比较使用三 seed validation 的逐 sample direct-mean 摘要，不构造加权总分，不在开发 validation 上报告 Holm/p-value。

1. 先完成 CRD_001 与 CRD_002，建立新训练协议下的旧结构基准。
2. 比较 101 vs 001。若 101 的 Local RR seed mean 相对恶化严格大于 3%，或 signed PCC seed mean 下降严格大于 0.01，coarse-head 路线停止，102–104 不运行。
3. 101 通过后比较 102 vs 101。保留需同时满足：Local RR seed mean 相对改善至少 0.5%；至少 2/3 配对 seed 方向改善；signed PCC 下降不大于 0.005；trajectory MAE 相对恶化不大于 1.5%。若失败，102 不保留，103/104 不运行。
4. 102 通过后比较 103 vs 102，使用同一保留条件；此外 103 不得触发 101 vs 001 的 coarse-head 停止线。Direct 103 必须通过，否则 CRD-v1 primary route 停止。
5. 103 通过后才可运行 104。104 vs 103 使用同一 0.5%/2-of-3/PCC/trajectory 条件；通过则 `D_TRUNK=104`，否则 `D_TRUNK=103`。

相对变化以 comparator 为分母；误差指标改善定义为 `(baseline-candidate)/baseline`，误差恶化定义为 `(candidate-baseline)/baseline`。门槛等号视为通过。任何未定义的边缘情形先修订协议，不按结果临时解释。

## 14. 产物与验收契约

每个 run 必须保存 resolved `config.yaml`、`run_manifest.json`（命令、Git commit/dirty、协议、依赖版本、run role）、`audit.csv`、`optimizer_parameter_groups.json`、逐 epoch `train_history.csv`、best/final checkpoint、逐 sample validation `metrics.csv` 与摘要。checkpoint 保存 update index/total updates 并标记不支持 resume。独立复评默认写 `validation_reeval_metrics.csv`，不覆盖训练生命周期生成的 `metrics.csv`。

必须通过：

```bash
./.venv/bin/python -m pytest \
  tests/test_respiration_protocol.py \
  tests/test_respiration_metrics.py \
  tests/test_tho_protocol_config.py \
  tests/test_tho_current_experiment.py \
  tests/test_time_stft_fusion.py \
  tests/test_tho_time_frequency_candidates.py \
  tests/test_crd_spectral_ops.py \
  tests/test_crd_models.py \
  tests/test_crd_config.py \
  tests/test_crd_training.py \
  tests/test_crd_experiment.py \
  tests/test_crd_batch_scaling.py
```

正式训练前还必须在目标 GPU 通过：固定依赖导入/版本；`scripts/check_crd_mamba.py` 的 Mamba2 `use_mem_eff_path=True` bf16 forward/backward finite；`scripts/check_crd_variant.py` 的所选 variant 单 microbatch forward/loss/backward finite；acceptance 的 physical batch 128 不 OOM。smoke/acceptance 结果不得写入正式比较表。

## 15. CRD_101 结果与原始 gate 执行记录（结果后登记）

本节登记发生在 CRD_101 三 seed validation 结果产生之后，不改写第 13 节的原始停止规则。两个因服务器故障中断的 run 保留为工程追溯，但不进入 seed mean 或任何 gate：

| Seed | Run | 完成状态 |
|---:|---|---|
| 20260811 | `runs/crd_v1/crd_101_b0_coarse/seed_20260811/20260808_152029_499890` | 中断于 epoch 25 / update 2000；无 final checkpoint 与 metrics |
| 20260812 | `runs/crd_v1/crd_101_b0_coarse/seed_20260812/20260808_152045_952796` | 中断于 epoch 24 / update 1920；无 final checkpoint 与 metrics |

正式比较只使用以下三个完整 run：

| Seed | Run | Local-RR best epoch |
|---:|---|---:|
| 20260811 | `runs/crd_v1/crd_101_b0_coarse/seed_20260811/20260808_154855_237029` | 72 |
| 20260812 | `runs/crd_v1/crd_101_b0_coarse/seed_20260812/20260808_155229_535580` | 30 |
| 20260813 | `runs/crd_v1/crd_101_b0_coarse/seed_20260813/20260808_161853_510867` | 58 |

三者均为 commit `6f58f36f4839904014031970e5f69262aa6e96f8`、`git_dirty=false`、formal `80×128×1`、6400 updates、2675 个 validation samples；history、best/final checkpoint model/optimizer tensors 和逐 sample metrics finite，best checkpoint 与 history 的严格最低 Local RR epoch 一致。

| 三 seed mean ± sample SD | CRD_001 | CRD_101 | 101 相对变化 |
|---|---:|---:|---:|
| Local RR MAE | 0.632468 ± 0.004054 | 0.624387 ± 0.003498 | 改善 1.2777% |
| lag-aware signed PCC | 0.840287 ± 0.000860 | 0.787829 ± 0.001328 | 下降 0.052458 |

Local RR 未触发 3% 恶化线，但 signed PCC 下降严格大于 0.01。依第 13 节原始规则，CRD_101 判定失败，原 `101→102→103→104` 队列关闭；`CRD_102/103/104` 不得按原顺序直接启动。

## 16. S1D 结果后诊断修订（2026-08-08）

本节是在已观察 CRD_101 失败之后制定的 development/validation 诊断协议，证据属性必须明确标为 post-result diagnostic。它不改变数据、split、target、`Pi`、loss、metrics、Local-RR checkpoint selector、formal seeds 或 research-test 禁令，也不把任何诊断结果表述为预注册确认性证据。

### 16.1 D0：CRD_001/101 paired final-checkpoint 复评

对三个完整 CRD_001 和三个完整 CRD_101 的 `checkpoint_final.pt` 复评相同 validation；同时比较各模型 final vs Local-RR-selected checkpoint，以及 CRD_101-final vs CRD_001-final，判断 PCC 下降是否可能主要来自 checkpoint selector。输出必须写入独立的 `runs/crd_v1/crd_final_checkpoint_diagnostic/`，不得覆盖原 run 的 `metrics.csv`、`metrics_summary.csv` 或任何已有复评产物。

D0 只作归因：原正式结论仍由 `checkpoint_best_local_rr.pt` 决定；即使 final checkpoint 跨过停止线，也不能据此事后重选 CRD_101 或自动开放 CRD_102。若要改变 selector，必须另行冻结新协议，并在同一新 selector 下成对重建 CRD_001/101 三 seed 证据。

D0 已在 commit `baeb7c51f85c83cdd28fa6a69285e36461d33e7e` 的干净工作树下完成。六份 manifest 均指向对应 epoch-80 `checkpoint_final.pt`；每份 metrics 均为 2675 个 validation samples、无 Inf、`joint_prediction_degenerate_fraction=0`。结果为：

| 三 seed mean ± sample SD | CRD_001 selected | CRD_001 final | CRD_101 selected | CRD_101 final |
|---|---:|---:|---:|---:|
| Local RR MAE | 0.632468 ± 0.004054 | 0.661868 ± 0.010022 | 0.624387 ± 0.003498 | 0.640469 ± 0.010190 |
| lag-aware signed PCC | 0.840287 ± 0.000860 | 0.841586 ± 0.000585 | 0.787829 ± 0.001328 | 0.788269 ± 0.000799 |

在相同 fixed-final selector 下，CRD_101 相对 CRD_001 的 Local RR 改善 3.2332%，但 signed PCC 仍下降 0.053317；三个配对 seed 的 PCC 分别下降 0.054800、0.052177、0.052975。CRD_101 从 selected 到 final 的 PCC 仅增加 0.000440，而 CRD_001 增加 0.001299，fixed-final 的模型间 PCC 缺口反而比原 selected comparison 扩大约 0.000859。因此 checkpoint selector 不是 CRD_101 PCC 退化的主要解释，原 gate 失败结论保持不变，CRD_102 继续关闭；D1 CRD_105 可以按本修订启动。

### 16.2 D1：CRD_105 Direct-Coarse

新增唯一诊断候选 `crd_105_direct_coarse`：

```text
DirectAnalyticFrontend [B,96,1800]
→ ResidualDWBlock(96,d=1)
→ ResidualDWBlock(96,d=2)
→ 与 CRD_101 完全相同的 10-Hz coarse head
→ 与 CRD_101 完全相同的 Fourier interpolation 1800→18000
```

CRD_105 不注册 local/global Mamba 或 FiLM，trainable params 固定为 192,781。同一 seed 下，它的 Direct frontend 与 CRD_103/104 逐 tensor 同初始化，refinement/head 与 CRD_101–104 逐 tensor 同初始化。配置固定为 `configs/crd_v1/crd_105_direct_coarse.yaml`，protocol manifest/checkpoint 标识固定为 `crd-v1.1-s1d-20260808`，仍使用 formal `80 epochs / physical batch 128 / accumulation 1` 和三个原 formal seeds。正式三 seed 前必须完成该 variant 的 finite synthetic 检查与独立 physical-batch-128 acceptance；smoke/acceptance 不形成效果证据。

CRD_105 工程验收已在 commit `be21ba03573ac1a8fe54d94d410a1a33fd397572`、`git_dirty=false` 下完成。CUDA synthetic batch-1 的 waveform shape 为 `[1,1,18000]`，loss 0.483973，sync/effort eligibility 均为 1，output/input/parameter gradients 全部 finite，peak allocated 30.34 MiB。随后 `/tmp/crd_105_batch128_acceptance/20260808_204245_249478` 使用 128 train windows 形成严格一次 update，并对 32 validation windows 完成 Local-RR selector、best/final checkpoint 和逐 sample metrics；resolved config 为 acceptance `1 epoch / physical batch 128 / accumulation 1`、`max_train/max_val/max_test=128/32/null`，checkpoint model/optimizer tensors finite、update index/total 均为 1、无 prediction degeneracy。该结果只解除 formal 三 seed 的工程阻塞；其中单 epoch Local RR/PCC 等数值不得进入效果比较。

D1 的主 gate 仍以 CRD_001 三 seed mean 为 comparator：Local RR 相对恶化严格大于 3%，或 signed PCC 下降严格大于 0.01，即判定 Direct-Coarse 失败。另报告 CRD_105 vs CRD_101 的全部指标差异用于 bridge 归因，但不另设事后阈值。

D1 三 formal seeds 已在 commit `2bee3e504c91db8d63cf7c4f6424ca501e3c82a6`、`git_dirty=false` 下完成：

| Seed | Run | Local-RR best epoch | Local RR MAE | signed PCC |
|---:|---|---:|---:|---:|
| 20260811 | `runs/crd_v1/crd_105_direct_coarse/seed_20260811/20260808_204605_323232` | 28 | 0.569175 | 0.846919 |
| 20260812 | `runs/crd_v1/crd_105_direct_coarse/seed_20260812/20260808_211725_152070` | 44 | 0.599122 | 0.843825 |
| 20260813 | `runs/crd_v1/crd_105_direct_coarse/seed_20260813/20260808_214719_482054` | 14 | 0.575407 | 0.848245 |

三个 run 均为 formal `80×128×1`、6400 updates、2675 个 validation samples；history、best/final checkpoint model/optimizer tensors 与逐 sample metrics finite，无 prediction degeneracy，best checkpoint 与严格最低 Local RR epoch 一致。三 seed mean ± sample SD 为：

| 指标 | CRD_001 | CRD_105 | 105 相对 001 |
|---|---:|---:|---:|
| Whole RR MAE | 0.534701 ± 0.012776 | 0.520911 ± 0.020652 | 改善 2.5789% |
| Local RR MAE | 0.632468 ± 0.004054 | 0.581234 ± 0.015801 | 改善 8.1006% |
| trajectory MAE | 0.157245 ± 0.001236 | 0.149718 ± 0.000264 | 改善 4.7867% |
| global envelope error | 0.232488 ± 0.009831 | 0.236005 ± 0.012477 | 恶化 1.5130% |
| lag-aware signed PCC | 0.840287 ± 0.000860 | 0.846330 ± 0.002268 | 增加 0.006043 |
| IBI MedAE | 0.084340 ± 0.001021 | 0.082121 ± 0.002772 | 改善 2.6308% |
| IBI coverage | 0.848257 ± 0.001741 | 0.826697 ± 0.003483 | 下降 0.021560 |

CRD_105 的 Local RR 未恶化而是改善 8.1006%，signed PCC 未下降而是增加 0.006043，故明确通过 CRD_001 coarse gate；三个配对 seed 在 Local RR 与 PCC 上均同方向优于 CRD_001。相对 CRD_101，CRD_105 的 Local RR 改善 6.9112%、signed PCC 增加 0.058501、IBI coverage 增加 0.056450，但 trajectory MAE 恶化 3.1077%。由于 101/105 共享 refinement/head 且都不含 Mamba，该结果把原 PCC 退化定位到 PatchTokenFrontend 与 DirectAnalyticFrontend 的 frontend package 差异，而不是共享 coarse head；它不能进一步把收益拆分为 patch bridge 缺陷或 analytic filterbank 增益。

### 16.3 冻结的后续分支

1. 若 CRD_105 未通过 CRD_001 coarse gate，coarse decoder/output representation 路线停止，CRD_102/103/104 均不运行。
2. 若 CRD_105 通过，说明 Direct-Coarse package 可进入下一阶段，而原 patch-bridge 路线仍保持关闭；跳过 CRD_102，按 `105→103→104` 推进。
3. `103 vs 105` 使用第 13 节原 `102 vs 101` 的保留条件：Local RR seed mean 改善至少 0.5%、至少 2/3 配对 seed 方向改善、signed PCC 下降不大于 0.005、trajectory MAE 相对恶化不大于 1.5%。失败则 103 不保留且 104 不运行。
4. 103 通过后，`104 vs 103` 继续使用同一条件；通过则保留 104，否则保留 103。
5. 原 CRD_102 只有在未来协议预先冻结新的训练/selector 修订、并由对应的修订版 CRD_101 三 seed 重新通过 CRD_001 gate 后才可重新开放；D0 或单 seed 探索不能满足该条件。本修订不授权启动 CRD_102。

D1 已执行上述第 2 条分支：CRD_105 保留，CRD_102 继续关闭，现开放 `CRD_103 vs CRD_105` formal 三 seed 比较；CRD_104 仍等待 103 gate。自该条件分支开放起，尚未正式运行的 CRD_103/104 与 CRD_105 一样固定使用 `crd-v1.1-s1d-20260808` protocol manifest/checkpoint 标识；这只修正结果后路线的 provenance，不改变两者已冻结的模型、训练或 gate。

任何上述正式诊断 run 都只能读取 train/validation。不得因本修订读取 research-test、改变原 CRD_001/101 产物，或复用中断 run 的 best checkpoint。

### 16.4 D2：CRD_103 Local Mamba 结果

CRD_103 三 formal seeds 已在 commit `ed9d68edd7a4b6693155b2129767e7ac7b604a26`、`git_dirty=false` 下完成：

| Seed | Run | Local-RR best epoch | Local RR MAE | signed PCC | trajectory MAE |
|---:|---|---:|---:|---:|---:|
| 20260811 | `runs/crd_v1/crd_103_direct_local_mamba/seed_20260811/20260808_223610_091525` | 36 | 0.570364 | 0.847587 | 0.165441 |
| 20260812 | `runs/crd_v1/crd_103_direct_local_mamba/seed_20260812/20260808_234525_453913` | 9 | 0.609111 | 0.854979 | 0.151075 |
| 20260813 | `runs/crd_v1/crd_103_direct_local_mamba/seed_20260813/20260809_005357_322787` | 10 | 0.576268 | 0.855807 | 0.156159 |

三个 run 均为 S1D formal `80×128×1`、6400 updates、2675 个 validation samples；history、best/final checkpoint model/optimizer tensors 与逐 sample metrics finite，无 prediction degeneracy，best checkpoint 与严格最低 Local RR epoch 一致。三 seed mean ± sample SD 及冻结 gate 为：

| 指标 | CRD_105 | CRD_103 | 103 相对 105 | Gate |
|---|---:|---:|---:|---|
| Whole RR MAE | 0.520911 ± 0.020652 | 0.531336 ± 0.010433 | 恶化 2.0012% | 非 gate |
| Local RR MAE | 0.581234 ± 0.015801 | 0.585248 ± 0.020876 | 恶化 0.6905% | 需改善至少 0.5%，失败 |
| 配对 Local RR | — | — | 0/3 seeds 改善 | 需至少 2/3，失败 |
| trajectory MAE | 0.149718 ± 0.000264 | 0.157558 ± 0.007285 | 恶化 5.2364% | 最多恶化 1.5%，失败 |
| global envelope error | 0.236005 ± 0.012477 | 0.199587 ± 0.014352 | 改善 15.4310% | 非 gate |
| lag-aware signed PCC | 0.846330 ± 0.002268 | 0.852791 ± 0.004525 | 增加 0.006461 | 最多下降 0.005，通过 |
| IBI MedAE | 0.082121 ± 0.002772 | 0.078243 ± 0.002867 | 改善 4.7216% | 非 gate |
| IBI coverage | 0.826697 ± 0.003483 | 0.836222 ± 0.002067 | 增加 0.009526 | 非 gate |

CRD_103 虽改善 signed PCC、global envelope、IBI MedAE/coverage 与 lag-boundary fraction，但同时使 Whole/Local RR、trajectory MAE 以及 Low/Medium/High 三层 envelope Spearman 退化。它未满足 Local RR mean、2-of-3 paired direction 和 trajectory 三项必要条件，故 D2 判定失败：CRD_103 不保留，CRD_104 不运行，当前 S1D 保留模型为 CRD_105。上述次级收益只能作为未来独立问题的背景证据，不得回头放宽本轮 gate。

## 17. S1E 结果后探索性补全（2026-08-09）

本节在已观察 CRD_101、105、103 全部结果且 S1D 已正式停止之后制定。为补全结构响应信息，额外运行 CRD_102 与 CRD_104，但二者证据固定标记为 post-result exploratory completion；它们不重新开放第 13/16 节决策链、不参与本轮保留模型重选，也不得把结果表述为预注册确认性证据。当前 S1D 保留模型 CRD_105 在 S1E 运行前即已冻结。

### 17.1 冻结问题与比较口径

1. `CRD_102 vs CRD_101` 只回答：在同一失败的 PatchTokenFrontend/coarse package 上，Local BiMamba2 是否产生补偿或新的任务权衡。
2. `CRD_104 vs CRD_103` 只回答：在已不保留的 Direct + Local Mamba package 上，1-Hz Global BiMamba2 + FiLM 的边际作用。
3. 对两项比较均报告三个固定 seed 的逐 sample direct-mean、seed mean ± sample SD、逐 seed 配对方向，以及 Whole/Local RR、trajectory、global envelope、signed PCC、IBI MedAE/coverage、三层 envelope Spearman、interpretable fraction 与 lag-boundary fraction。
4. 第 13/16 节的 `0.5% / 2-of-3 / PCC 0.005 / trajectory 1.5%` 条件仅作为描述性参照，命中与否不改变 CRD_105 的已冻结保留状态。
5. 另将 CRD_102/104 分别与 CRD_001 和 CRD_105 做全指标描述性比较；这些跨越多个结构因素的比较不能解释为单因素因果效应。
6. 不构造加权总分、不在 validation 上报告 p-value、不修改 checkpoint selector，也不读取 research-test。若结果提示新的候选路线，必须另建未来协议；不得事后为本批结果制定选择门槛。

### 17.2 运行身份与工程门槛

CRD_102/104 的模型、初始化、loss、metrics、训练 seed、80 epochs、physical batch 128、accumulation 1、6400 updates 和 Local-RR checkpoint selector 均保持原冻结定义。由于现有 runner 的 `formal` role 表示完整预算/完整数据生命周期，两配置仍使用 `run_role=formal`；其探索性证据身份由独立 protocol `crd-v1.1-s1e-20260809` 与本节决定。manifest/checkpoint 必须记录该 protocol 和当次干净 Git commit。

两个 variant 各自在三 seed 前必须通过当前 commit 下的 CUDA synthetic batch-1 finite forward/loss/backward 与独立 physical-batch-128 acceptance。Acceptance 固定 `1 epoch / 128 train / 32 validation / max_test=null`，只解除工程阻塞，不形成效果证据。两项 acceptance 均通过后，CRD_102 与 CRD_104 的三个固定 seed 可以顺序或在独立 GPU 上并行执行，输出分别写入：

两项工程验收已在 commit `d60b050498a9d21bfaa2d7f5a624a7a5d6b09903`、`git_dirty=false` 下完成。CUDA synthetic batch-1 的 102/104 loss 分别为 1.245569/0.407518，output/input/parameter gradients 全部 finite，peak allocated 分别为 349.16/357.28 MiB。对应 acceptance 为 `/tmp/crd_102_b0_local_mamba_batch128_acceptance/20260809_021443_770877` 与 `/tmp/crd_104_direct_hier_mamba_batch128_acceptance/20260809_021601_879119`；两者均严格使用 128 train windows 形成一次 update，并完成 32 条 validation metrics、best/final checkpoint 与 optimizer state，所有 tensors/metrics finite、无 prediction degeneracy。单 epoch Local RR/PCC 等数值不进入效果解释；两项工程门槛均已解除。

```text
runs/crd_v1/crd_102_b0_local_mamba/seed_<seed>/
runs/crd_v1/crd_104_direct_hier_mamba/seed_<seed>/
```

所有 S1E run 仍严格禁止读取 research-test；任何中断或重复 run 必须按完整 lifecycle 审计后再决定是否纳入，不能仅凭存在 best checkpoint 进入汇总。

### 17.3 S1E 完整结果

CRD_102/104 各三个完整 run 均在 commit `f8fa658443b2bc2a2414f142b2a78c746000daca`、`git_dirty=false` 下完成。六个 run 均为 S1E full-budget `80×128×1`、6400 updates、2675 个 validation samples；history、best/final checkpoint model/optimizer tensors 与逐 sample metrics finite，无 prediction degeneracy，best checkpoint 与严格最低 Local RR epoch 一致。

| Variant | Seed | Run | Best epoch | Local RR | signed PCC | trajectory |
|---|---:|---|---:|---:|---:|---:|
| 102 | 20260811 | `runs/crd_v1/crd_102_b0_local_mamba/seed_20260811/20260809_022128_909818` | 13 | 0.552821 | 0.865598 | 0.149786 |
| 102 | 20260812 | `runs/crd_v1/crd_102_b0_local_mamba/seed_20260812/20260809_032737_110644` | 11 | 0.566412 | 0.863090 | 0.149075 |
| 102 | 20260813 | `runs/crd_v1/crd_102_b0_local_mamba/seed_20260813/20260809_043403_554605` | 13 | 0.549251 | 0.865328 | 0.151588 |
| 104 | 20260811 | `runs/crd_v1/crd_104_direct_hier_mamba/seed_20260811/20260809_054023_270736` | 4 | 0.548550 | 0.854263 | 0.150472 |
| 104 | 20260812 | `runs/crd_v1/crd_104_direct_hier_mamba/seed_20260812/20260809_065353_484711` | 72 | 0.554025 | 0.849170 | 0.156327 |
| 104 | 20260813 | `runs/crd_v1/crd_104_direct_hier_mamba/seed_20260813/20260809_080732_234579` | 58 | 0.561292 | 0.851580 | 0.155999 |

三 seed mean ± sample SD：

| 指标 | CRD_102 | CRD_104 | 冻结 S1D CRD_105 |
|---|---:|---:|---:|
| Whole RR MAE | 0.515775 ± 0.020060 | 0.466760 ± 0.015423 | 0.520911 ± 0.020652 |
| Local RR MAE | 0.556161 ± 0.009055 | 0.554622 ± 0.006392 | 0.581234 ± 0.015801 |
| trajectory MAE | 0.150150 ± 0.001296 | 0.154266 ± 0.003290 | 0.149718 ± 0.000264 |
| global envelope error | 0.187512 ± 0.002438 | 0.207204 ± 0.020511 | 0.236005 ± 0.012477 |
| lag-aware signed PCC | 0.864672 ± 0.001376 | 0.851671 ± 0.002548 | 0.846330 ± 0.002268 |
| IBI MedAE | 0.080181 ± 0.001384 | 0.078182 ± 0.002194 | 0.082121 ± 0.002772 |
| IBI coverage | 0.839027 ± 0.006859 | 0.843683 ± 0.003255 | 0.826697 ± 0.003483 |
| Low envelope Spearman | 0.400923 ± 0.030012 | 0.363911 ± 0.051695 | 0.400853 ± 0.029150 |
| Medium envelope Spearman | 0.568066 ± 0.019583 | 0.525940 ± 0.008351 | 0.542727 ± 0.022539 |
| High envelope Spearman | 0.742683 ± 0.006310 | 0.718028 ± 0.010620 | 0.718366 ± 0.006680 |

冻结问题的描述性结论：

1. `102 vs 101`：Whole/Local RR 分别改善 9.4864%/10.9269%，Local RR 为 3/3 paired 改善，global envelope 改善 10.1719%，signed PCC 增加 0.076843，IBI MedAE/coverage 与 Low/Medium Spearman 也改善；但 trajectory MAE 恶化 3.4048%，故若机械套用旧条件，会因 trajectory 单项失败。这说明 Local Mamba 对失败 Patch frontend package 存在强补偿，而不是简单延续 101 的退化。
2. `104 vs 103`：Whole/Local RR 改善 12.1535%/5.2329%，Local RR 为 3/3 paired 改善，trajectory 改善 2.0895%，signed PCC 仅下降 0.001120，四项旧条件均通过；global envelope 恶化 3.8164%，但 IBI coverage、三层 Spearman 与 interpretable fraction 改善。Global Mamba + FiLM 对 103 存在明显补偿/交互效应。
3. 相对 CRD_001，102 的 Whole/Local/trajectory/global-envelope 分别改善 3.5395%/12.0650%/4.5124%/19.3456%，PCC 增加 0.024385；104 分别改善 12.7063%/12.3083%/1.8946%/10.8753%，PCC 增加 0.011384。两者 IBI coverage 仍分别低 0.009229/0.004574。
4. 相对冻结的 CRD_105，102 的 Whole/Local/global-envelope 分别改善 0.9860%/4.3138%/20.5477%，PCC 增加 0.018342，trajectory 仅恶化 0.2882%，且 Local RR 为 3/3 paired 改善；它描述性满足四项旧条件。104 的 Whole/Local/global-envelope 分别改善 10.3955%/4.5785%/12.2036%，PCC 增加 0.005342，但 trajectory 恶化 3.0375%，描述性不满足旧 trajectory 条件。

S1E 暴露了明显的非单调结构交互：101 或 103 单步失败并不意味着其后继 102 或 104 也必然失败。尽管 102 尤其形成强探索性候选，本节在观察结果前已冻结为不重选模型，因此 CRD_105 仍是 S1D 的正式保留结果；102/104 只能用于设计下一版预先冻结的多候选确认协议，不能在本节事后升级为保留模型，也不能据此读取 research-test。

## 18. 102/104/105 candidate lock 与未来选择规则（2026-08-09）

本节在 S1E validation 结果全部观察之后制定，属于 post-result candidate lock，不把既有探索结果升级为确认性证据。机器可读锁文件固定为 `docs/experiments/crd_v1_candidate_lock_20260809.json`；其中锁定 CRD_102/104/105 各三个 Local-RR-selected checkpoints，并将 CRD_001 三 checkpoint 作为只读 reference。每项记录精确相对路径、SHA-256、文件大小、seed、selected epoch、训练 commit/protocol，以及 resolved config、run manifest、validation metrics summary 的 SHA-256。任何文件缺失、大小或 hash 不一致都必须停止；不得用同 variant 的其他 timestamp、final checkpoint 或重训结果替代。

Candidate lock 提交后禁止：重训候选、改变 checkpoint selector、替换 seed/epoch、修改模型或超参数、根据未来结果调整门槛。若确需变更，必须创建新 lock ID 并保留本锁，不得原地重写其科学语义。当前只冻结候选与规则，**没有激活任何独立确认数据或 research-test 入口**。

### 18.1 资格门槛

CRD_105 是稳定锚点并自动具备资格。未来确认阶段中，CRD_102/104 各自相对 CRD_105 必须同时满足：

1. Local RR seed mean 相对改善至少 0.5%；
2. 至少 2/3 同 seed 配对方向改善；
3. lag-aware signed PCC seed mean 下降不大于 0.005；
4. trajectory MAE seed mean 相对恶化不大于 1.5%。

相对变化仍按 comparator 为分母，误差改善为 `(anchor-candidate)/anchor`、trajectory 恶化为 `(candidate-anchor)/anchor`，等号视为通过。四项必须同时满足；未通过者不能进入 Pareto 选择。

### 18.2 五项 primary Pareto 规则

对通过资格门槛的候选与自动合格的 CRD_105，使用三个固定 seed 的逐 sample direct mean 再做 seed arithmetic mean。Primary 固定为：Whole RR、Local RR、trajectory MAE、global envelope error 最小化，lag-aware signed PCC 最大化。

若模型 A 在全部五项上不差于 B，且至少一项严格优于 B，则 A Pareto-dominate B。最终只保留非支配集合：若恰有一个非支配候选，可选择该候选；若有多个，则保留 Pareto set，不构造加权总分、不临时增加权重或强制唯一赢家。

IBI MedAE/coverage、Low/Medium/High envelope Spearman、IBI interpretable fraction 与 lag-boundary fraction固定为 secondary，只解释非支配候选的任务权衡，不能覆盖资格门槛或 primary Pareto 结果。CRD_001 只提供参考背景，不进入候选选择。

### 18.3 独立确认阶段尚未激活

未来若决定建立独立确认阶段，必须在读取任何确认数据前另行冻结：数据来源与 admission、是否沿用现有 split、sample/subject 统计单位、一次性评价命令、输出目录、缺失/非有限处理及结果表述边界。所有候选必须在相同样本、相同指标、相同顺序下评价一次，禁止根据中间结果停止、替换 checkpoint 或再训练。本节本身不授权读取 research-test，也不改变 CRD_105 作为当前 S1D 正式保留结果的状态。

## 19. S1C 现有 research-test 确认阶段（2026-08-09）

本节在第 18 节 candidate lock 提交 `05571bd` 后、读取任何 CRD test 结果前建立，取代第 18.3 节“尚未激活”的当前状态。S1C 协议标识固定为 `crd-v1.1-s1c-research-20260809`。确认数据选择为现有 research-test；该 split 已在旧 B0/T2/T4 等阶段被观察并允许影响后续研究，因此本阶段只能称为 **development/research confirmation evidence**，不能称为独立、全新或无偏 held-out 证据。

### 19.1 数据与独立性边界

S1C 沿用各冻结 resolved config 的 research v2 数据、admission、input/target、`test` split、test sampling seed 与全部 loss/metric 定义，不修改任何数据或指标口径。2026-08-09 在不读取波形值的索引级审计中，全量 admitted train/validation/research-test 分别为 `10141 / 2675 / 2310` 个窗口、`32 / 7 / 8` 个 `samp_id`；train–validation、train–test、validation–test 的 `samp_id` overlap 与 segment overlap 均为 0。复现命令固定为：

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config-kind crd \
  --config configs/crd_v1/crd_105_direct_coarse.yaml \
  --output-dir runs/crd_v1/crd_s1c_split_audit_20260809
```

该隔离只证明当前 split 的 subject/session 边界，不消除 research-test 已被历史观察造成的研究选择偏倚。

### 19.2 冻结评价集合与顺序

评价集合恰为 `docs/experiments/crd_v1_candidate_lock_20260809.json` 中按文件顺序排列的 12 个 Local-RR-selected checkpoints：CRD_102、104、105、001 各三个固定 seed。Lock 的 SHA-256 固定为 `9a14db8be8af22e1ce1c5a332b4912ab5c13c7fb03cdf1894fc5c6ed6ff7f8cc`；001 仍只作 reference，105 仍是资格锚点。任何 checkpoint/config/run manifest/validation summary 缺失、大小或 hash 不一致均停止，不允许替换 timestamp、seed、epoch、final checkpoint 或重训结果。

所有 12 项必须使用同一完整 2310-window、8-`samp_id` test 集合并按 lock 顺序评价；不得根据中间结果提前停止或改变顺序。运行时 Git 工作树必须干净，commit 与依赖版本写入 receipt/manifest。普通 `scripts/eval_crd.py` 继续保持 validation-only；research-test 只可由 `scripts/eval_crd_s1c.py` 访问，且必须显式传入 `--confirm-research-test`。公共入口不提供 split、config、抽样上限或输出目录覆盖。

### 19.3 指标、失败与产物契约

每个 checkpoint 一次性生成逐 sample 五项 primary、IBI-MedAE/coverage、三层 envelope Spearman、interpretable/lag diagnostics，以及 research-test-only coherence/nDTW；seed 内仍为逐 sample direct mean，跨 seed 才做 arithmetic mean。Prediction/target 非有限、任何数值列出现 Inf、eligible primary/test-only 值缺失或非有限、行数/`samp_id` 数量/split/method 不符均立即失败；协议既有 target-ineligible NaN 继续按 eligibility 口径保留，不能删行或填补。

固定输出目录为：

```text
runs/crd_v1/crd_s1c_research_confirmation/<variant>/seed_<seed>/
```

每项保存 `research_confirmation_access_receipt.json`、逐 sample metrics、summary 和 execution manifest。Access receipt 在首次读取该 checkpoint 的 test 数据前排他创建；任一正式文件或 receipt 已存在时拒绝覆盖和重复评价。若进程在 receipt 后中断，保留 receipt 并暂停，先审计失败原因和已暴露结果，再由显式协议修订决定是否技术性重跑，不得自行删除 receipt。

### 19.4 冻结决策规则

只有在 12 项全部完成、产物完整且无异常后，才由预先实现的 `scripts/summarize_crd_s1c.py` 重算全部逐 sample summary、核对 receipt/manifest，并应用第 18.1–18.2 节已经冻结的“相对 CRD_105 四项资格门槛 + 五项 primary Pareto”规则；缺任一项或已有汇总产物时拒绝执行。不得用部分结果作选择，也不得让 coherence、nDTW、IBI 或其他 secondary 覆盖资格门槛与 Pareto 结果；多个非支配候选时保留 Pareto set。Research-test 结果不得用于重选同一 run 的 epoch/checkpoint、删 seed、修改本阶段模型/超参数或追加模型；若结果形成新假设，必须建立明确标记 `research-test informed` 的后续阶段。

### 19.5 完整结果与冻结选择

S1C 于 2026-08-09 在 commit `3b280013d898287613709c4dd5648f8b94e14c9d`、`git_dirty=false` 下完成。12 个 checkpoint 均有 access receipt、逐 sample metrics、summary 和 manifest；每份 metrics 为 2310 行，总计 27720 行。冻结汇总器重新计算全部 summary 后通过一致性检查；12 份 manifest 的 protocol、candidate-lock SHA-256、checkpoint 身份、split 和代码 commit 一致，数值列无 Inf。

五项 primary 的三 seed arithmetic mean ± sample SD 为：

| 模型 | Whole RR ↓ | Local RR ↓ | Trajectory ↓ | Global envelope ↓ | Signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| CRD_102 | 0.703666 ± 0.021614 | 0.651590 ± 0.014033 | 0.141174 ± 0.001419 | 0.171800 ± 0.006681 | 0.878253 ± 0.001661 |
| CRD_104 | 0.723770 ± 0.021364 | 0.697125 ± 0.014385 | 0.151971 ± 0.005300 | 0.175258 ± 0.008463 | 0.865519 ± 0.005781 |
| CRD_105 | 0.749168 ± 0.023183 | 0.699180 ± 0.013363 | 0.142869 ± 0.000391 | 0.193158 ± 0.005621 | 0.865394 ± 0.002593 |
| CRD_001 reference | 0.691119 ± 0.003755 | 0.654473 ± 0.002706 | 0.144311 ± 0.000693 | 0.175587 ± 0.002389 | 0.857280 ± 0.000250 |

主要 secondary 的三 seed mean ± sample SD 如下；Low/Medium/High Spearman 一列为避免过宽，仅依次列出三层 seed mean，逐 seed 数值保留在 `s1c_seed_summary.csv`：

| 模型 | IBI MedAE ↓ | IBI coverage ↑ | Low/Medium/High Spearman ↑ | IBI interpretable ↑ | Coherence ↑ | nDTW ↓ |
|---|---:|---:|---:|---:|---:|---:|
| CRD_102 | 0.098255 ± 0.002480 | 0.803502 ± 0.006224 | 0.428866 / 0.527289 / 0.735967 | 0.636508 ± 0.014803 | 0.552947 ± 0.011565 | 0.203844 ± 0.003749 |
| CRD_104 | 0.100048 ± 0.001611 | 0.785600 ± 0.010875 | 0.419412 / 0.493308 / 0.678855 | 0.593651 ± 0.011699 | 0.548638 ± 0.003149 | 0.217054 ± 0.007018 |
| CRD_105 | 0.113665 ± 0.001112 | 0.762800 ± 0.016781 | 0.437779 / 0.497358 / 0.712633 | 0.562626 ± 0.017231 | 0.542735 ± 0.011560 | 0.218460 ± 0.001143 |
| CRD_001 reference | 0.094312 ± 0.000729 | 0.794697 ± 0.002100 | 0.400021 / 0.500339 / 0.725040 | 0.609812 ± 0.004379 | 0.589185 ± 0.002878 | 0.240900 ± 0.000896 |

冻结规则得到：

1. `CRD_102 vs CRD_105`：Local RR 相对改善 `6.8065%` 且 3/3 paired seed 改善；signed PCC 增加 `0.012860`；trajectory 改善 `1.1863%`。四项 eligibility 全部通过。Whole RR 与 global-envelope 另改善 `6.0737% / 11.0571%`，因此 102 在五项 primary 上严格支配 105。
2. `CRD_104 vs CRD_105`：Local RR 仅改善 `0.2940%`，虽有 2/3 paired seed 改善且 PCC 增加 `0.000126`，但未达到 `0.5%`；trajectory 恶化 `6.3707%`，超过 `1.5%`。104 不具资格，secondary 不得推翻。
3. Eligible set 为 `{CRD_102, CRD_105}`；唯一 non-dominated candidate 为 `CRD_102`。它成为后续 research-test-informed 阶段的当前结构锚点，S1D 保留 105 的旧结论不追溯改写。
4. CRD_001 只作 reference 且不进入选择。102 相对 001 的 Whole RR 恶化 `1.8155%`，Local RR/trajectory/global-envelope 改善 `0.4404% / 2.1736% / 2.1565%`，PCC 增加 `0.020974`；001 的 IBI-MedAE 与 coherence 也更好。因此“102 被选中”不等价于所有任务轴全面优于纯时域 reference。

完整机器结果为 `s1c_seed_summary.csv`、`s1c_variant_summary.csv`、`s1c_selection.json` 与 `s1c_selection_manifest.json`。本结果属于现有 research-test 上的 development/research confirmation evidence，不是无偏 held-out 结论；S1C 访问队列现已关闭，不得重复运行或用 test 结果重选 checkpoint。后续若以 CRD_102 继续设计 S2 或其他模块，必须另建明确标记 `research-test informed` 的协议。

## 20. S1F：B0 frontend 下的 global-stage 缺失格（2026-08-09）

本节在 S1C 已选择 CRD_102 后制定，protocol 固定为 `crd-v1.1-s1f-research-test-informed-20260809`。它属于由既有 research-test 结果触发的 development/validation closure，不是原 S1 的预注册确认，也不重新开放 S1C 或任何 CRD research-test。S2 仍未激活。

### 20.1 唯一问题与必要性

S1 已有的 frontend × global-stage 结构格为：

| Frontend | Local only | Local + Global |
|---|---|---|
| B0/PatchMixer | CRD_102 | 缺失 |
| Direct analytic | CRD_103 | CRD_104 |

由于 S1/S1C 已观察到明显的非单调模块交互，不能用 `104 vs 103` 代替 global stage 在已选中 B0 frontend 上的效应。本节只补齐缺失格 `CRD_106_B0_HIER_MAMBA`，回答：

> 在 CRD_102 的 B0/PatchMixer frontend 与 local Mamba trunk 不变时，增加同一 1-Hz global Mamba + FiLM 是否改善 validation Local RR，且不破坏 PCC/trajectory？

不得在本节新增其他 frontend、宽度、block、decoder、loss、seed 或超参数实验。

### 20.2 CRD_106 唯一结构与初始化

配置固定为 `configs/crd_v1/crd_106_b0_hier_mamba.yaml`。CRD_106 与 CRD_102 完全共享：

- `PatchTokenFrontend` 及 `patch_frontend` 子 seed；
- 6 个 local BiMamba2 block 及 `local_trunk` 子 seed；
- 两个 local refinement block、10-Hz coarse waveform head 与 Fourier interpolation；
- 数据、split、target、`Pi`、core loss、metrics、Local-RR checkpoint selector、80 epochs、physical batch 128、optimizer/update/LR 语义和三个正式 seed。

唯一增加项是在 local blocks 与 refinement 之间插入与 CRD_104 完全相同的 `GlobalContextStage`：`10 Hz → 0.45-Hz hard LPF → ::10 → Conv 96→128 → 4×BiMamba2(128) → Fourier 180→1800 → zero-init FiLM`，并复用 `global_trunk` 确定性子 seed。FiLM weight/bias 为零，因此初始化时 106 的 global residual 对任意 local latent 严格为 identity。冻结 trainable parameter count 为 `2,181,193`；除该 global stage 外，102/106 对应模块的初始 state 必须逐 tensor 相同，104/106 global-stage 初始 state 也必须逐 tensor 相同。

### 20.3 工程门槛与正式队列

正式训练前必须在同一提交的干净工作树完成：

1. `check_crd_variant.py` 的 CUDA batch-1 model/core-loss/backward finite；
2. 独立 physical-batch-128 acceptance：128 train windows、32 validation windows、1 epoch、1 optimizer update；
3. output/gradient/checkpoint/metrics finite，参数数量、zero-init identity 和共享 state 测试通过。

工程门槛只解除三个 formal seeds 的运行阻塞，不形成效果证据。Formal 输出固定隔离到 `runs/crd_v1/crd_106_b0_hier_mamba/seed_<seed>/`；任何中断 run 不凭已有 best checkpoint 纳入比较。

上述工程门槛已在干净 commit `8fa56f8ef7d8f4e40a644d9feb22e7cc80857e6b` 下通过。CUDA synthetic 的 model/core-loss/backward 全 finite，peak allocated `347.766 MiB`；独立 acceptance 位于 `/tmp/crd_106_b0_hier_mamba_batch128_acceptance/20260809_142221_810744`，严格使用 128/32 个 train/validation windows、1 epoch、1 optimizer update。两个 checkpoint 的 241 个 model-state tensors 全 finite，32 条 validation metrics 无 Inf、无 prediction degeneracy，manifest 记录固定依赖与 `git_dirty=false`。单 epoch loss/metrics 只用于完整生命周期验收，不进入效果比较；现解除 CRD_106 三 formal seeds 的工程阻塞。

### 20.4 冻结保留规则

只比较 CRD_106 与冻结的 CRD_102 三个 validation-selected Local-RR checkpoints，按同 seed 配对。CRD_106 必须同时满足：

1. Local RR seed mean 相对改善至少 `0.5%`；
2. 至少 `2/3` paired seeds 的 Local RR 改善；
3. lag-aware signed PCC seed mean 下降不超过 `0.005`；
4. envelope trajectory MAE seed mean 相对恶化不超过 `1.5%`。

相对变化、等号和 direct-mean/seed-mean 语义沿用第 18 节。四项全部通过则未来 S2 的 `BASE=CRD_106`，否则 `BASE=CRD_102`。IBI、coverage、三层 Spearman、global-envelope 与 lag diagnostics 只作 secondary，不能覆盖四项规则。结果不触发 research-test，不回改 S1C 选择，也不允许追加第二个 S1F variant；S1F 完成后才决定是否另立 S2 协议。

### 20.5 S1F 完整结果与关闭决定

CRD_106 三个 formal runs 已在 commit `80e635027062864a6db310d64ea07f223af5c443`、`git_dirty=false` 下完成：

```text
runs/crd_v1/crd_106_b0_hier_mamba/seed_20260811/20260809_142511_139526
runs/crd_v1/crd_106_b0_hier_mamba/seed_20260812/20260809_154036_791769
runs/crd_v1/crd_106_b0_hier_mamba/seed_20260813/20260809_153828_227713
```

三个 run 均为唯一完整 80-epoch lifecycle、6400 optimizer updates、完整 2675-row validation metrics；Local-RR-selected epochs 为 `40 / 5 / 26`。所有 train history、两个 checkpoint 的 241 个 model-state tensors 和数值 metrics 均无 Inf，prediction degeneracy 为 0，逐 sample metrics 重算 summary 完全一致。冻结的 CRD_102 candidate-lock 及其 48 个关联产物也重新通过 SHA-256 验证。

三 seed arithmetic mean ± sample SD：

| 模型 | Whole RR ↓ | Local RR ↓ | Trajectory ↓ | Global envelope ↓ | Signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| CRD_102 | 0.515775 ± 0.020060 | 0.556161 ± 0.009055 | 0.150150 ± 0.001296 | 0.187512 ± 0.002438 | 0.864672 ± 0.001376 |
| CRD_106 | 0.471658 ± 0.017571 | 0.543858 ± 0.002918 | 0.157710 ± 0.007283 | 0.189358 ± 0.004906 | 0.861494 ± 0.000997 |

冻结规则逐项结果：

1. Local RR seed mean 改善 `2.2123%`，通过 `≥0.5%`；
2. 三个 paired seeds 的 Local RR 差值均有利，`3/3`，通过；
3. signed PCC 下降 `0.003178`，未超过 `0.005`，通过；
4. trajectory MAE 恶化 `5.0351%`，超过 `1.5%`，失败。

因此 106 **不满足四项全通过条件**，不保留为未来 BASE；按第 20.4 节冻结分支，`S2 BASE=CRD_102`。106 的 Whole RR 改善 `8.5536%`、Local RR 改善以及 coverage 增加属于任务交换背景，不能覆盖 trajectory guardrail；global-envelope 也恶化 `0.9844%`，Low/Medium/High Spearman 与 signed PCC 均下降。该结果说明 global stage 在 B0 frontend 下仍形成 RR–包络/PCC 权衡，而非无条件增益。

S1F 至此关闭：不追加第二个 S1F variant、不读取 research-test、不回改 S1C；CRD_106 只保留为 validation development evidence。S2 仍需以 CRD_102 为冻结 BASE 另立 research-test-informed 协议后才能激活。

## 21. S2：以 CRD_102 为 BASE 的表征分支实验（2026-08-09）

本节在 S1F 关闭并冻结 `S2 BASE=CRD_102` 后、运行任何 S2 模型前制定。S2 属于 **research-test-informed development/validation** 阶段，不读取 research-test，不把当前 validation 上的选择表述为确认性统计推断。S2A protocol 固定为 `crd-v1.1-s2a-research-test-informed-20260809`；S2B protocol 固定为 `crd-v1.1-s2b-research-test-informed-20260809`。当前只激活 S2A 的实现与工程验收，S2B 必须等待 S2A 三分支结果按本节规则触发。

### 21.1 BASE 身份与公共 trunk

`CRD_201_BASE` 不是新配置或新训练，而是第 18 节 candidate lock 中 CRD_102 三个 checkpoint 的只读别名；checkpoint path/hash、seed、selected epoch、训练 commit/config/manifest/validation summary 均保持不变。不得重训 201 或用 CRD_102 的其他 timestamp/final checkpoint 替代。

所有 S2 模型保留 CRD_102 的：

```text
PatchTokenFrontend → 6×Local BiMamba2 → 2×refinement
→ 10-Hz coarse waveform head → Fourier ↑100 Hz → external Pi
```

新增 representation 必须在 `PatchTokenFrontend` 输出 `z_B ∈ R^(B×96×1800)` 后、进入 local Mamba 前静态注入。公共融合形式为：

```text
z0 = z_B + W_R z_R
```

组合分支为 `z0=z_B+W_X z_X+W_M z_M`。每个 `W` 都是独立 `1×1 Conv1d`，weight/bias 全零初始化；因此所有候选初始化时严格退化为 CRD_102 的 BASE latent。Patch frontend、local Mamba、refinement 和 head 使用与 102 相同的确定性子 seed，初始 state 必须逐 tensor 相同。S2 不启用 Direct frontend、global stage、gate、envelope/phase auxiliary head，也不修改 core loss、metrics 或 checkpoint selector。

### 21.2 三种候选 representation 的唯一结构

#### Legacy energy `E`

Legacy E 是在 CRD trunk 上重建的固定 T4 bandenergy representation，不复用旧 T4 run 的可训练权重：

```text
STFT: n_fft=win_length=2000, hop=250, center=true,
      pad_mode=reflect, Hann periodic=true,
      normalized=false, onesided=true
float32 log1p(abs(STFT))
bands: [0.05,0.30], [0.10,0.70], [0.30,1.20],
       [0.70,3.00], [3.00,8.00] Hz
band bins direct mean → B×5×73
Conv1d 5→16 k3 p1, GroupNorm(1,16), SiLU
Conv1d 16→16 k3 p1, SiLU
linear interpolate 73→1800, align_corners=false
zero-init W_E: Conv1d 16→96 k1
```

两层 encoder Conv 固定 `bias=true`、Kaiming-normal weight/zero bias；GroupNorm weight/bias 为 1/0。STFT 与幅值数学强制关闭 autocast、使用 float32。该分支只称 legacy energy representation，使用 `legacy_energy_encoder` 与 `legacy_energy_projection` 子 seed。

#### Analytic AM `A`

AM 使用 8 个 Gaussian analytic filters，外部支持严格为 `0.70<f≤8.0 Hz`。中心 interval/init 固定为：`[0.75,1.10]/0.90`、`[1.10,1.50]/1.30`、`[1.50,2.10]/1.80`、`[2.10,2.80]/2.40`、`[2.80,3.70]/3.20`、`[3.70,4.80]/4.20`、`[4.80,6.30]/5.60`、`[6.30,8.00]/7.20`。带宽为 `sigma_k=c_k*rho_k`，`rho∈[0.08,0.30]`、初始化 `0.15`，center/rho 均用 bounded sigmoid 参数化。

```text
analytic magnitude log1p(absolute value)
fixed float32 FFT projection 0.03–0.80 Hz → ::10
Conv1d 8→48 k5 p2, GroupNorm(6,48), SiLU
Conv1d 48→96 k1, GroupNorm(12,96), SiLU
ResidualDWBlock(96,d=1/2/4)
zero-init W_A: Conv1d 96→96 k1
```

FFT/filterbank/magnitude/LPF 强制 float32；其余神经编码允许 bf16 autocast。

Analytic 计算严格沿用 Direct filterbank 的正频率定义：正频率为 `2*X*G`，DC/Nyquist/负频率为零，再 complex IFFT。两层 encoder Conv 固定 `bias=false`、Kaiming-normal，GroupNorm 为 1/0；ResidualDWBlock 沿用第 1 节定义。子 seed 固定为 `analytic_am_filterbank / analytic_am_encoder / analytic_am_projection`。

#### Amplitude-normalized morphology `M`

Morphology 输入使用固定整窗 zero-phase FFT bandpass `0.70<f≤8.0 Hz`。Reflect padding 左右各 75；`window=151`、`hop=10`，得到 1800 个 token。每个窗口先减 mean，再除以 `sqrt(mean(x^2)+1e-6)`，有意删除局部绝对幅度。为避免 physical batch 128 下物化全部中间特征，token 轴固定按 128 个窗口分块编码；分块不改变 token 顺序或数学定义。训练态对每个 chunk 使用 `use_reentrant=false / preserve_rng_state=false` 的 activation checkpoint，validation/evaluation 直接前向；Morphology encoder 内没有 dropout 或 batch-dependent normalization，因此该工程策略只在 backward 重算相同前向，不改变模型、loss、token 顺序或 optimizer update 口径。

```text
Conv1d 1→32 k9 valid, GroupNorm(4,32), SiLU
Conv1d 32→32 k7 stride2 valid, GroupNorm(4,32), SiLU
DepthwiseConv1d 32 k5 p2, SiLU
Conv1d 32→64 k1, SiLU, AdaptiveAvgPool1d(1)
Linear 64→96, LayerNorm(96)
16 orthogonal-initialized prototypes, cosine softmax T=0.10
concat 96-d embedding + 16 scores
Linear 112→96, LayerNorm(96)
zero-init W_M: Conv1d 96→96 k1
```

前四个 Conv 固定 `bias=false`、Kaiming-normal；GroupNorm/LayerNorm 为 1/0；两个 Linear 使用 PyTorch Kaiming-uniform/zero bias；prototype 用全局 seed 下的 orthogonal initialization。固定 bandpass、window normalization 和 prototype cosine/softmax 强制 float32。Prototype forward 时逐行 L2 normalize；`L_proto=sum_(i!=j)(p_i^T p_j)^2/[K(K-1)]`。Morphology 模型使用 `L_core + 1e-3*r(u)*L_proto`，其中 `r(u)` 沿用 optimizer update `5S→15S` 的冻结 ramp。子 seed 固定为 `morphology_encoder / morphology_prototypes / morphology_projection`。该候选测试的是 amplitude-normalized morphology + prototype package，不声称识别真实独立生理源。

### 21.3 S2A：三个单分支候选

S2A 只实现并在分别通过 synthetic/physical-batch-128 acceptance 后运行：

| ID | 结构 | 新增正式 runs |
|---|---|---:|
| `CRD_202_BASE_LEGACY_ENERGY` | BASE + E | 3 |
| `CRD_203_BASE_ANALYTIC_AM` | BASE + A | 3 |
| `CRD_204_BASE_MORPHOLOGY` | BASE + M + prototype regularizer | 3 |

全部使用 80 epochs、physical batch 128、三个固定 seed、完整 train/validation、Local-RR checkpoint selector；不得读取 test。三个 variant 必须分别做工程 acceptance，尤其 204 必须验证 token chunking 的 output/gradient finite、峰值显存和完整 checkpoint lifecycle。任何一个分支的工程失败只阻塞该分支，不允许静默简化结构或缩小 formal batch。

实现登记：三个配置分别为 `crd_202_base_legacy_energy.yaml / crd_203_base_analytic_am.yaml / crd_204_base_morphology.yaml`，trainable parameter 数固定为 `1,071,449 / 1,197,785 / 1,106,857`。实现定向协议测试与全量 CPU 回归已通过；102 shared trunk 逐 tensor 相同、三个 projection 全零、频谱 float32、morphology 128-token chunking、prototype no-decay 与 `5S→15S` ramp 均有测试覆盖。CUDA synthetic 与三个独立 physical-batch-128 acceptance 尚待执行，因此 formal 队列仍未开放。

首轮 CUDA 工程结果随后显示：202/203 的 synthetic 与 physical-batch-128 acceptance 均通过，peak reserved 分别为 `9,910/11,832 MiB`，占 15,936 MiB 设备的 `62.18%/74.25%`；204 synthetic 通过，但 batch-128 训练在 morphology encoder 处 OOM，失败时 PyTorch 已分配约 `15.12 GiB`、仅 `55 MiB` 空闲，不能归因于明显的其他进程占用。由于 forward chunking 仍会为 backward 保留所有 chunk 的内部激活，正式实验前注册上述 per-chunk activation checkpoint 工程修订；它不减少 physical/effective batch，不改变科学比较因素。当前只允许重跑 204 synthetic 与独立 acceptance，204 通过且 peak reserved fraction `≤80%` 前 formal 队列继续关闭。

204 checkpointed 重验随后在干净 commit `a149913` 下通过：synthetic output/gradient finite；acceptance 严格完成 128/32 个 train/validation windows、1 optimizer update、两个 finite checkpoint、32 条 primary-finite validation metrics且 joint prediction degeneracy 为 0。Peak allocated/reserved 为 `8,636.81/10,406 MiB`，reserved fraction `65.30%`，低于 `80%` 工程线。IBI-MedAE 因该单 update 模型的 32 个样本均 `ibi_interpretable=false` 而按既有 eligibility 契约为空，不是被静默吞掉的非有限 prediction，也不作为单 epoch 工程阻塞。202/203 acceptance 来自干净 commit `258c1f3`，其后到 `a149913` 唯一运行时代码差异只在 morphology checkpoint 路径，不影响 202/203。至此三个 S2A variant 的工程验收均完成，允许在新的统一干净 commit 上启动九个 formal runs；仍不得读取 research-test 或开放 S2B。

九个 S2A formal runs 随后在统一干净 commit `41ed41d` 下完成。每个 run 均为 80 epochs/6400 updates、2675 条 validation metrics、两个 finite checkpoint且 joint prediction degeneracy 为 0；selected epochs 为 202=`10/18/12`、203=`22/11/12`、204=`8/11/12`，三结构 peak reserved fraction 分别为 `63.85%/75.91%/65.33%`。冻结门槛已确定：E 的 trajectory 相对改善为 `-1.6232%` 且 paired seed `0/3`，故不 eligible；A 的 trajectory 相对改善 `-6.9786%`、paired seed `0/3`、PCC absolute drop `0.007549`，故不 eligible，得到 `X=none`。M 的 PCC absolute increase 为 `-0.006737` 且 paired seed `0/3`、Local RR 相对恶化 `2.6374%`、IBI coverage absolute drop `0.010307`，故不 eligible。按 21.6–21.7，S2B/S3 不开放并保留 CRD_102。该结论不等待也不允许被 prototype 描述覆盖；S2A 只剩按 21.5 补齐 204 三 seed 的 validation prototype usage/entropy 与 samp 分布，再生成不可覆盖的完整 summary。

204 prototype 描述随后在干净 commit `3c3598c` 下对三个 validation-selected checkpoint 完成，每项均覆盖 2675 windows/7 samp IDs，checkpoint/hash/identity 与逐 window、逐 samp 重算审计通过。三个 seed 的 hard-usage normalized entropy 为 `0.5131/0.5731/0.6254`，soft-usage entropy 为 `0.9021/0.8809/0.9540`，mean token entropy 为 `0.8044/0.7778/0.8652`；全局 dominant hard fraction 为 `37.26%/47.50%/34.06%`，逐 samp 最大 dominant fraction 为 `67.92%/68.97%/64.15%`。因此没有跨全部 validation 的单 prototype 坍缩，但 hard assignment 明显稀疏且 seed 间 prototype ID 不可直接对齐；这种“有使用、无任务收益”只解释 M 失败，不能重开门槛。冻结产物位于 `runs/crd_v1/crd_s2a_validation_summary/`，manifest 记录干净 commit `3c3598c`、9 个完整 checkpoint、`X=none / M ineligible / S2B=false / retain CRD_102`。至此 S2A 正式关闭，相关入口禁止重复运行。

## 22. S2B-R：结果知情的双因素交互补救（2026-08-10）

本节在观察并冻结 S2A 的 `X=none / M ineligible` 后，由研究者明确要求检验“单因素失败但多因素非线性补偿成功”的可能性。因此它**不是**第 21.6 节原条件自然触发的 S2B，而是新增的 result-informed exploratory stage，协议名固定为 `crd-v1.1-s2br-result-informed-20260810`。既有 S2A decision 不改写；本阶段只使用 train/validation，不读取 research-test，不计算确认性 p-value，也不把成功结果表述为预注册确认性证据。

### 22.1 四个唯一候选

由于 S2A 未选出 X，不能结果后只挑 E 或 A；S2B-R 同时运行两组组合及各自的 capacity control：

| ID | 结构 | Trainable params |
|---|---|---:|
| `CRD_205_BASE_EM_STATIC` | BASE + exact E + exact M | 1,109,561 |
| `CRD_206_BASE_AM_STATIC` | BASE + exact A + exact M | 1,235,897 |
| `CRD_207_BASE_CAP_EM` | BASE + EM-matched capacity stack | 1,109,257 |
| `CRD_208_BASE_CAP_AM` | BASE + AM-matched capacity stack | 1,235,785 |

205/206 在 PatchTokenFrontend 后、local Mamba 前分别执行 `z0=z_B+W_E z_E+W_M z_M` 与 `z0=z_B+W_A z_A+W_M z_M`。E/A/M 的结构、参数、子 seed、zero-init projection 与 S2A 202/203/204 逐 tensor 相同；BASE trunk 也与 102 相同。205/206 沿用 `L_core+1e-3*r(u)*L_proto`，207/208 只使用 core loss。206 训练态对完整 A 分支使用 `use_reentrant=false / preserve_rng_state=true` activation checkpoint，以保持 A 内 dropout 重算一致；M 继续使用第 21.2 节的 per-chunk checkpoint。两项都只改变 backward 存储/重算，不改变模型数学定义。

Capacity control 不读取 E/A/M。每个 block 固定为 `GN(12,96)→Conv1x1 96→H(bias=false)→SiLU→Dropout(0.10)→Conv1x1 H→96(bias=true)→residual`；GN 为 1/0，首层 Kaiming-normal，末层 weight/bias 全零。穷举 `N∈[1,8]`、`H∈{32,40,...,2048}`，依次最小化完整模型相对参数差、MACs、N、H，并要求参数差 `≤2%`。冻结结果为：207 使用 `N=2,H=104`，比 205 少 304 参数（`0.027398%`）；208 使用 `N=4,H=216`，比 206 少 112 参数（`0.009062%`）。所有候选初始化时严格退化为 BASE latent。

### 22.2 工程与正式训练

四项分别完成 synthetic forward/backward 与独立 physical-batch-128 acceptance；每项要求完整 checkpoint lifecycle、eligible primary finite、joint prediction nondegenerate、peak reserved fraction `≤80%`。工程失败只阻塞该 variant，不允许临时改小 formal batch或改结构。通过后每项使用 80 epochs、physical batch 128、accumulation 1、三个固定 seed、完整 train/validation、Local-RR checkpoint selector；不得修改 loss、metrics、seed、预算或访问 research-test。

四项工程验收随后在统一干净 commit `0e541f8` 下通过。205/206/207/208 的 synthetic output/gradient 均 finite；每个 acceptance 均完成 128/32 个 train/validation windows、1 optimizer update、两个 finite checkpoint、32 条 eligible-primary-finite validation metrics且 joint prediction degeneracy 为 0。Peak allocated 为 `8,645.69/8,637.92/8,935.42/9,911.20 MiB`，peak reserved fraction 为 `67.37%/67.15%/65.41%/71.46%`，均低于 `80%`；205/206 正确记录 prototype regularizer provenance/P no-decay，206 的 filter logits 也处于 no-decay。至此允许四项在新的统一干净 commit 上各运行三个 formal seeds；单 epoch acceptance 数值不作效果比较。

### 22.3 组合资格与交互证据

每个组合必须同时通过：

1. 相对 CRD_102 BASE，Local RR seed mean 改善 `≥0.5%` 且 `≥2/3` paired seeds 改善；
2. 相对自己的 capacity control，Local RR 改善 `≥0.25%` 且 `≥2/3` paired seeds 改善；
3. 相对 BASE 与两个 constituent singles 中 Local RR 更低者，Local RR 改善 `≥0.25%` 且 `≥2/3` paired seeds 改善。冻结 comparator 为 205 对 CRD_202，206 对 CRD_102；
4. 相对 BASE 的 signed PCC drop `≤0.003`、trajectory worsening `≤1.5%`、IBI coverage drop `≤0.01`。

此外必须按 seed、paired window 与 paired samp ID 报告 `combo − energy − morphology + BASE` factorial interaction contrast；该描述用于判断补偿方向，但不能覆盖四项全通过门槛。Whole RR、IBI-MedAE、global envelope、三层 Spearman、参数、MAC/VRAM/latency 只作 secondary。

若仅一个组合通过，选择该组合并允许另立 S3 协议；若两者都通过，A+M 只有在相对 E+M 同时达到 Local RR 改善 `≥0.25%`、`≥2/3` paired seeds 改善以及相同 PCC/trajectory/coverage 护栏时才取代参数更少的 E+M。若都失败，保留 CRD_102，S3 继续关闭。S3 的 gate 结构当前仍未定义、未实现，不能与本阶段代码混入。

### 22.4 冻结结果与停止决定（2026-08-11）

12 个 formal runs 已在统一干净 training commit `c2bcfb0` 下完成。每项均为 80 epochs/6400 updates、2675 条 validation metrics、两个 finite checkpoint，eligible primary 全部 finite 且 joint prediction degeneracy 为 0；205/206/207/208 的 selected epochs 分别为 `8/11/12`、`19/11/9`、`13/11/12`、`10/18/15`，formal peak reserved fraction 分别为 `67.44%/67.20%/67.04%/71.47%`。冻结汇总在干净 commit `7eb26e6` 下生成，manifest 固定 12 个完整 checkpoint 与 training commit，未读取 research-test。

两个组合均不 eligible。205 相对 BASE、control 207、最佳 constituent 202 的 Local RR 相对改善分别为 `-1.0387%/-0.9260%/-2.5769%`，paired seed 改善数为 `1/3、1/3、0/3`；相对 BASE 的 PCC drop 为 `0.005295`、trajectory worsening 为 `-0.3834%`、coverage drop 为 `0.010579`，仅 trajectory 护栏通过。206 相对 BASE、control 208、最佳 constituent BASE 的 Local RR 相对改善分别为 `-1.0295%/-1.4515%/-1.0295%`，三项 paired seed 改善数均为 `1/3`；PCC drop 为 `0.011947`、trajectory worsening 为 `4.7290%`、coverage drop 为 `0.007237`，仅 coverage 护栏通过。

Factorial interaction 的 seed-mean contrast 表明非线性补偿方向确实存在：205 的 Local RR/trajectory/PCC/coverage interaction 为 `-0.000552/-0.003398/+0.001168/+0.000236`，206 为 `-0.010117/-0.003762/+0.002339/+0.012108`；但 Local RR 方向并不跨三个 seed 稳定，且两组合的绝对表现仍落后 BASE 与各自 capacity control。该证据支持“存在模块补偿效应”，不支持“组合已形成可选模型”。最终 decision 固定为 `passing_combinations=[] / retain CRD_102 / S3=false`；产物位于 `runs/crd_v1/crd_s2br_validation_summary/`，S2B-R 至此关闭，汇总入口不得重复用于重选。

## 23. CRD_102 validation 误差分层与失败模式诊断（2026-08-11）

本节在 S2B-R 关闭并保留 CRD_102 后建立，协议名固定为 `crd-v1.1-crd102-failure-diagnostic-20260811`。输入只能是 candidate-lock 固定的 CRD_102 三个 `checkpoint_best_local_rr.pt` 所属 run 的既有 validation `metrics.csv`；逐 seed 必须为 2675 windows/7 samp IDs，checkpoint、summary、逐行 identity 与 target/static 字段必须通过审计。该诊断不重新推理、训练或选择 checkpoint，不访问 research-test，不改变数据、split、target、loss、metrics 或聚合口径，也不计算确认性 p-value。

诊断同时保留 window、seed 和 samp/coupling 分层，不能把 2675 个相关 windows 当作 2675 个独立受试者。固定分层轴为 `samp_id`、`coupling_state_id`、冻结的 `envelope_target_stratum`、三 seed IBI interpretable 一致性以及 lag-boundary 一致性。连续 `target_envelope_modulation` 只用于描述与 Spearman association，不根据结果重新切阈值。每个分层报告 window 数/占比、覆盖的 samp 数、八项任务指标的 mean/median、持续失败率、IBI interpretable 与 `|best_lag|=0.30 s` 比例；另保留逐 seed 分层结果，防止 seed-ensemble mean 掩盖不稳定性。

固定失败判据如下：对 whole/local RR、trajectory、global modulation、signed PCC、IBI-MedAE、IBI coverage 与 envelope Spearman，在每个 seed 的 eligible windows 内分别取不利方向 worst decile（minimize 指标 `≥q90`，maximize 指标 `≤q10`，边界 ties 全保留）；同一 window 在至少 `2/3` seeds 命中才称该指标 persistent failure。Core failure count 只计 whole/local RR、trajectory、global modulation 与 signed PCC，避免缺失 IBI eligibility 改变分母。签名只描述 `rate/envelope/alignment/rank/beat` 指标共现：alignment 还包括至少 `2/3` seeds 到达 `|best_lag|=0.30 s`，beat 还包括至多 `1/3` seeds IBI interpretable。该签名不是生理或模型内部原因的因果识别。

另固定输出：三 seed 指标 Spearman/绝对差/worst-decile Jaccard、error-aligned 指标间 Spearman（无 p-value）、Local-RR 跨 seed SD 的 top decile、全部 persistent failure windows、signature 汇总、threshold/audit/manifest。输出目录为 `runs/crd_v1/crd_102_failure_diagnostic/`，已存在时禁止覆盖；结果只用于确定下一轮应优先调查的数据/任务失效区域，不能重开 S2B-R、S3 或覆盖 CRD_102 选择。

第一层结果生成后若同时出现明显 target-modulation 关联与相邻 row 聚集，允许一次 result-informed metadata follow-up，协议名固定为 `crd-v1.1-crd102-failure-metadata-20260811`。该层只把第一层冻结 consensus 逐 `dataset_row_id` 一对一连接到三个 CRD_102 resolved config 共同指向的冻结 `dataset_index.csv`；必须校验 index hash、`split=val`、samp/coupling identity 和 180 s 窗长。连续元数据固定为 valid/motion/reliable ratios、六类 confidence score、alignment lag/drift 与 finite ratio；分类元数据固定为六类 confidence level、alignment method/reference-assisted 和 allowed losses，不结果后增删字段。

Metadata follow-up 只报告元数据与 error-aligned outcome 的 Spearman、persistent Local-RR/multimetric-core 两种失败组的 mean/median contrast、原有分类 level 的分层结果，以及连续失败片段。连续片段固定为同一 samp 内相邻失败窗口 `window_start_s` 间隔不超过冻结 30 s step；分别对 persistent Local-RR 与 core failure count `≥2` 汇总 episode 数、singleton 比例、最长窗口数/跨度。重叠窗口不能当独立事件，所有 association/contrast/episode 仍是探索性描述，无 p-value、无因果主张。单独输出到 `runs/crd_v1/crd_102_failure_metadata_diagnostic/`，不得改写第一层 bundle。

### 23.1 冻结结果与失败模式判断

第一层从干净 commit `81fab55` 生成，三个 checkpoint/hash 与三份 `metrics.csv` hash、2675 validation windows/7 samp IDs/17 coupling states、跨 seed identity/target 字段均审计通过。Local-RR 三 seed 排名 Spearman 平均 `0.9668`、worst-decile Jaccard 平均 `0.7710`，所以主要难例具有很强 seed 重复性，并非某一个初始化偶发失效；不过 Local-RR error 与跨 seed SD 的 Spearman 为 `0.7881`，说明难例的误差幅度也更易受 seed 影响。

高 target-modulation 是最集中的任务区域：high stratum 的 Local-RR mean 为 `1.2137 bpm`，而 medium/low 为 `0.3990/0.1329 bpm`；persistent Local-RR failure 为 `185/887=20.86%`，占全部 `262` 个 Local-RR persistent failures 的 `70.61%`。Core count `≥2` 在 high/medium/low 为 `323/887=36.41%`、`24/652=3.68%`、`5/1136=0.44%`，即 high stratum 占 `323/352=91.76%`。Target modulation 与 trajectory/local-RR/PCC-error 的 Spearman 分别为 `0.9143/0.7687/0.7842`。相反，256 个 persistent envelope-rank failures 中约 205 个落在 low stratum，表明低动态区主要暴露 rank metric/近常量轨迹困难，与高动态区的 rate/tracking/coverage 失败不是同一种签名。

样本整体排名受 modulation 构成明显混杂：按未校正整体 mean，samp 1308/952 的 Local-RR 为 `1.1788/1.0039 bpm`；但只看 high stratum 且 `n≥20` 时，956/1378/972 为 `2.6232/2.1595/2.0003 bpm`，反而高于 1308/952 的 `1.2906/1.0701 bpm`。因此不能把整体 samp 排名直接解释为 subject effect。Coupling state 12/18/8 的表面最差值只基于 `n=8/3/5`，同样不足以驱动结构选择。

IBI 在全部三 seed 均 interpretable 的窗口占 `69.61%`，全部不 interpretable 占 `25.94%`；后者 Local-RR mean `1.3756 bpm`，高于 all-interpretable 的 `0.2275 bpm`。至少 `2/3` seeds 命中 `|best_lag|=0.30 s` 的窗口占 `15.25%`，但其 Local-RR mean `0.5497 bpm` 与非持续边界的 `0.5573 bpm` 近似，因此 lag-boundary 不是当前 Local-RR 主失败轴。

Metadata follow-up 从干净 commit `7a1b29b` 生成，固定 index hash 与第一层 manifest/consensus hash 均记录。除 target modulation 外，与 Local-RR 最强的元数据关联是 waveform/rate/supervision confidence（Spearman `-0.5487/-0.5465/-0.5231`），其次是 transient-motion ratio（`+0.4811`）。Persistent Local-RR failures 的 waveform confidence mean/median 为 `0.5609/0.5327`，其余窗口为 `0.6978/0.6774`；motion ratio 为 `0.8294/0.8722` 对 `0.6651/0.7500`。Hard-valid 与 training-finite 均恒为 1，state-alignment-valid 近 1 且与 Local-RR 仅 `-0.0357`，说明这不是一般性非有限或 admission 失败。Confidence 与 modulation/samp composition 相关，以上差异仍不能解释为独立因果效应。

262 个 Local-RR persistent failure windows 合并为 97 个重叠窗口 episodes，其中 40 个 singleton；即 `222/262=84.73%` 的失败窗口处于多窗口连续片段，最长 9 windows/420 s。352 个 multimetric-core windows 合并为 142 个 episodes、69 个 singleton，`283/352=80.40%` 位于连续片段，最长 13 windows/540 s。失败因此更像高动态/较低置信度的连续时间段，而非均匀散布的独立窗口。最终诊断优先级固定为：先调查 high-modulation 连续 episode 的输入—目标可观测性与局部速率跟踪，再处理低-modulation rank 指标的适用性；不据此自动增加结构、重选 checkpoint 或启用 research-test。

## 24. CRD_102 high-modulation 配对可观测性诊断（2026-08-11）

本节是第 23.1 节结果触发的 validation-only exploratory follow-up，协议名固定为 `crd-v1.1-crd102-matched-observability-20260811`。它不重新运行 CRD_102，也不读取模型 prediction：输入仅为冻结 metadata consensus、dataset index，以及匹配窗口的 `bcg_rawish_wideband_state_aligned_segment_soft_z`、`bcg_resp_band_state_aligned_segment_soft_z` 与 `tho_waveform_segment_soft_z`。两个 BCG 信号分别作为 rawish direct proxy 与既有 F0 fixed-band proxy，复用冻结任务算子/metrics；另在 canonical `0.05–0.7 Hz` 信号上报告 Welch dominant-frequency error 与 `nperseg=2048/noverlap=1024` magnitude-squared coherence。Proxy 只回答输入中是否存在可恢复呼吸信息，不是新候选模型。

Case 固定为：Local-RR persistent failure 的多窗口 episode（至少 2 windows）中，central admitted window 为 high target-modulation；偶数长度取较早的中央窗。Control 固定为 high modulation、Local-RR 非 persistent、core failure count=0，并与同 samp 的任一 Local-RR failure window 在起始时间上至少相隔 180 s。Primary 要求同 samp、同 coupling state；sensitivity 只要求同 samp。两者分别在全体 high-modulation windows 的 IQR 上，对 target modulation、waveform confidence、motion ratio 计算等权 normalized L1 cost，caliper 固定 `≤2.0`；用 Hungarian assignment 最大化一对一匹配并最小化总 cost，时间距离只作 `1e-9` tie-break。Primary 少于 12 pairs 则停止，不能放宽规则。

每个 scheme 报告 CRD_102 三 seed mean、rawish proxy、fixed-band proxy 的八项任务指标 case/control paired difference 与 case-worse fraction；coherence 和 dominant-frequency error 另表。Primary 中，对每个 proxy 分别要求 Local-RR case-worse fraction 与 signed-PCC case-worse fraction均 `≥2/3` 才记 observability-failure signature：两 proxy 都命中为 `input_observability_associated`，都不命中为 `model_specific_tracking_associated`，仅一个命中为 mixed。该标签是配对关联证据而非可识别因果结论；不会重选 checkpoint、触发训练或访问 research-test。输出目录固定为 `runs/crd_v1/crd_102_matched_observability_diagnostic/`，存在时禁止覆盖。

### 24.1 冻结结果与解释

实现从干净 commit `48d8929` 冻结；首次执行在写出任何结果前因 decision 把 eligible IBI-MedAE 的 3 pairs 错纳入 primary pair-count 一致性检查而主动停止，未创建输出目录。修正只将 pair-count 审计限定到预先冻结的 Local-RR/PCC 两个 primary，commit `c951325` 通过全量 350 项测试后生成唯一结果。Manifest 固定 candidate-lock、metadata/index hashes、`research_test_used=false / model_inference_used=false`。

42 个 high-modulation multi-window case episodes 中，primary 得到 21 个同 samp/同 state pairs，覆盖 samp `952/1308/1378`；sensitivity 得到 28 个同 samp pairs，覆盖 `952/956/961/1308/1378`。Primary cases/controls 的 target modulation mean 为 `1.3384/1.2246`、waveform confidence `0.5158/0.5423`、motion ratio `0.8598/0.8780`，mean normalized L1 cost `0.9176`。因此 caliper 内仍有较小 modulation/confidence residual imbalance，且 14/21 primary pairs 来自 samp 952；primary 不能视作总体无混杂估计。

Primary 中 CRD_102 case/control Local-RR 为 `3.1827/0.7554 bpm`，21/21 cases 更差；PCC 为 `0.6979/0.7717`，16/21 更差。Rawish direct proxy 的 Local-RR 为 `5.1172/3.1014 bpm`、16/21 更差，但 PCC `0.5316/0.5546` 只有 13/21 更差，未达到 `2/3`；fixed-band proxy 的 Local-RR 为 `4.7234/2.4671 bpm`、17/21 更差，PCC `0.5369/0.5779`、16/21 更差，两项通过。故冻结 decision 为 `mixed_observability_and_model_tracking`，而非纯输入受限或纯模型失效。

配对内部，CRD Local-RR delta 与 rawish/fixed-band delta 的 Spearman 为 `0.4857/0.5740`，支持一部分 shared input-observability signature；两个 proxy 的 Local-RR/PCC delta 彼此相关 `0.8740/0.9182`。但 CRD PCC delta 与 rawish/fixed-band 仅 `0.2195/0.2130`，说明模型自身的相位/跟踪行为仍占明显部分。21 pairs 中 16 个在两个 proxy 上 Local-RR 都更差，12 个在两个 proxy 的 Local-RR/PCC 上同时更差；另有 4 个两个 proxy 的 Local-RR 都不更差，直接反对单一“输入完全不可观测”解释。

Primary broad-band coherence 没有稳定分离：rawish case/control mean `0.3887/0.3987`、case-worse `10/21`，fixed-band 为 `0.3791/0.4004`、`11/21`。Dominant-frequency absolute error 的 case mean 明显更大（rawish `3.77/0.98 bpm`，fixed `3.91/0.84 bpm`），但只在 `10/21`、`11/21` pairs 更差，反映少数大偏差与大量 ties/异质性。Same-samp sensitivity 中 rawish Local-RR/PCC case-worse 为 `17/28`、`16/28`，fixed-band 为 `18/28`、`20/28`，没有 proxy 同时通过两门槛；所以 primary 的输入可观测性成分对 state matching 和样本构成敏感。

最终判断：high-modulation failure episodes 至少包含两个亚型——proxy 同时恶化的输入可观测性关联亚型，以及 proxy 尚可但 CRD 仍失败的模型特异跟踪亚型。不能用一个全局新模块处理二者，也不能据此修改 admission 或标签。下一步若继续，应只对冻结 pairs 做三个 CRD_102 checkpoint 的 inference-only waveform/error decomposition，预先按 proxy-limited 与 proxy-available 亚型分层；未另立协议前不运行。

### 21.4 Energy representation 决策

E 与 A 各自相对 BASE，必须同时满足才称为 energy-eligible：

1. envelope trajectory MAE seed mean 相对改善至少 `1.0%`；
2. 至少 `2/3` paired seeds 的 trajectory 改善；
3. Local RR seed mean 相对恶化不超过 `1.0%`；
4. signed PCC seed mean 下降不超过 `0.003`；
5. global-envelope error seed mean 相对恶化不超过 `1.5%`。

选择 `X∈{E,A,none}` 的决策表：

| E eligible | A eligible | X |
|---|---|---|
| 否 | 否 | none |
| 是 | 否 | E |
| 否 | 是 | A |
| 是 | 是 | 先比较 A vs E |

两者都 eligible 时，A 只有在相对 E 同时满足 trajectory 改善 `≥1.0%`、`≥2/3` paired seeds 改善、Local RR 恶化 `≤1.0%`、PCC 下降 `≤0.003`、global-envelope 恶化 `≤1.5%` 才取代 E；否则选结构更简单且已有历史解释的 E。Whole RR、IBI、coverage 与三层 Spearman 只作 secondary。

### 21.5 Morphology 决策

M 相对 BASE 必须同时满足：

1. signed PCC seed mean 增加至少 `0.002`；
2. 至少 `2/3` paired seeds 的 signed PCC 改善；
3. Local RR seed mean 相对恶化不超过 `1.0%`；
4. IBI coverage seed mean 下降不超过 `0.01`；
5. trajectory MAE seed mean 相对恶化不超过 `1.5%`。

IBI-MedAE、interpretable fraction、prototype usage/entropy 和主体分布必须报告，但不能覆盖上述门槛。Prototype 激活若跨全部 validation 长期坍缩为单一 prototype，只能作为失败解释，不能在观察结果后修改 `T`、K 或 loss 权重重跑本阶段。

### 21.6 S2B：条件开放的组合与 capacity control

只有 `X != none` 且 M eligible，才开放恰好一个静态组合及其参数匹配对照：

| X | 静态组合 | Capacity control |
|---|---|---|
| E | `CRD_205_BASE_EM_STATIC` | `CRD_207_BASE_CAP_EM` |
| A | `CRD_206_BASE_AM_STATIC` | `CRD_208_BASE_CAP_AM` |

静态组合使用 BASE + selected energy + M，loss 为 `L_core+1e-3*r(u)*L_proto`。Capacity control 不读取 E/A/M representation，在 `z_B` 后加入 `CapacityResidualBlock(H)` stack：`GN(12,96)→Conv1x1 96→H→SiLU→Dropout(0.10)→Conv1x1 H→96→residual`，末层 zero-init。`N∈[1,8]`、`H∈{32,40,...,2048}`，只按完整 trainable parameter count 确定性穷举：最小相对参数差、再最小 MACs、再最小 N、再最小 H；要求与对应组合 `|delta params|≤2%`，否则构建失败。参数匹配不看 validation，MAC/VRAM/latency 只报告、不作为匹配目标。

组合必须同时满足：

1. 相对 BASE 的 Local RR mean 改善 `≥0.5%` 且 `≥2/3` paired seeds 改善；
2. 相对自己的 capacity control 的 Local RR mean 改善 `≥0.25%` 且 `≥2/3` paired seeds 改善；
3. 相对 BASE 的 PCC 下降 `≤0.003`、trajectory 恶化 `≤1.5%`、IBI coverage 下降 `≤0.01`。

全部通过才允许未来 S3 gate stage；否则 S3 整体跳过。无论结果如何，不补跑未被 X 选择的另一组合/capacity pair。

### 21.7 S2 关闭时的 BASE 选择

- X 与 M 均失败：保留 CRD_102 BASE；
- 仅 X eligible：选择 BASE+X；
- 仅 M eligible：选择 BASE+M；
- X/M 均 eligible 且组合通过：选择组合并开放 S3；
- X/M 均 eligible 但组合失败：S3 关闭，只在 BASE+X 与 BASE+M 中选择。Local RR mean 相对差异 `≥0.25%` 时选较低者；否则依次比较 trajectory MAE、signed PCC、trainable params，选择 trajectory 更低、PCC 更高、参数更少者，不构造加权总分。

所有选择只使用 validation-selected checkpoints。逐 seed、seed mean ± sample SD、paired window 与 paired `samp_id` descriptive differences 必须完整报告；不计算确认性 p-value/Holm/FWER。S2 结束前不得实现 gate/auxiliary/TCN/final ablation，不得访问 research-test。若未来需要强泛化结论，必须使用新的锁定 cohort 或外部数据。
