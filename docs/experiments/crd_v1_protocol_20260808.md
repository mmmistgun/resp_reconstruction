# CRD-Net v1.1 S0/S1 实现与实验协议（2026-08-08）

## 1. 权威性、范围与科学边界

本文是 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 35 节引用的规范性附件；发生冲突时以主协议为准。当前只激活：

- CRD-S0：旧 B0/T4 在新训练协议下重训；
- CRD-S1：decoder bridge、local Mamba、Direct frontend、可选 global Mamba 的顺序实验。

`docs/temp/` 中的讨论稿只保留设计历史，不是运行依据。AM、Morphology、gate、auxiliary、capacity/TCN control 和 S2 以后阶段不在本次实现或运行范围内，不能提前混入 S0/S1。

本阶段由第一轮 research-test 启发，但不回头修改旧 checkpoint 或旧结论。数据、split、target、正式算子 `Pi=S(B(.))`、core loss、评价指标和 Local-RR checkpoint selector 全部沿用主协议。S0/S1 只读 train/validation；research-test 在新的锁定队列和协议修订前禁止读取。

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

正式 seed 固定为 `20260811 / 20260812 / 20260813`，通过 `training.seed` 覆盖；`model.initialization_seed` 解析为同一值。六个原 S0/S1 配置与一个结果后诊断配置位于 `configs/crd_v1/`。

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
