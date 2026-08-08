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

D1 的主 gate 仍以 CRD_001 三 seed mean 为 comparator：Local RR 相对恶化严格大于 3%，或 signed PCC 下降严格大于 0.01，即判定 Direct-Coarse 失败。另报告 CRD_105 vs CRD_101 的全部指标差异用于 bridge 归因，但不另设事后阈值。

### 16.3 冻结的后续分支

1. 若 CRD_105 未通过 CRD_001 coarse gate，coarse decoder/output representation 路线停止，CRD_102/103/104 均不运行。
2. 若 CRD_105 通过，说明 Direct-Coarse package 可进入下一阶段，而原 patch-bridge 路线仍保持关闭；跳过 CRD_102，按 `105→103→104` 推进。
3. `103 vs 105` 使用第 13 节原 `102 vs 101` 的保留条件：Local RR seed mean 改善至少 0.5%、至少 2/3 配对 seed 方向改善、signed PCC 下降不大于 0.005、trajectory MAE 相对恶化不大于 1.5%。失败则 103 不保留且 104 不运行。
4. 103 通过后，`104 vs 103` 继续使用同一条件；通过则保留 104，否则保留 103。
5. 原 CRD_102 只有在未来协议预先冻结新的训练/selector 修订、并由对应的修订版 CRD_101 三 seed 重新通过 CRD_001 gate 后才可重新开放；D0 或单 seed 探索不能满足该条件。本修订不授权启动 CRD_102。

任何上述正式诊断 run 都只能读取 train/validation。不得因本修订读取 research-test、改变原 CRD_001/101 产物，或复用中断 run 的 best checkpoint。
