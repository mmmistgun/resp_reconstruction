# CRD_102 锚点控制线 C0/C1/C2 实验协议（2026-08-11）

## 1. 权威性、目的与当前开关

本文是 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 45 节引用的规范性附件；发生冲突时以主协议为准。该控制线在 CRD-v1.1 S2B-R 和后续 CRD_102 failure/matched-observability diagnostics 已冻结后另立，不续接已经关闭的 S3/S4/S5 命名，也不回写 S1/S2 或诊断阶段的历史结论。

控制线协议族固定为 `crd-v1.1-controls-research-informed-20260811`，证据属性为 **research-test-informed development/validation controls**。当前状态为：

- `C0`：无训练的 10-Hz/Fourier decoder round-trip 正式 validation 审计已完成并冻结；
- `C1`：parameter-matched full-context TCN 的工程验收、三个 formal runs 与冻结汇总均已完成，结果为 `mamba_retained_control_failure / retain CRD_102`，阶段关闭；
- `C2`：10-Hz capacity/100-Hz placement 两个候选的 CPU/CUDA 定向验收与独立 physical-batch-128 acceptance 已完成，formal 三 seed 队列已开放但尚未运行。

本控制线不读取 CRD research-test，不重新调用已关闭的 S1C 队列，不新增 gate、auxiliary、AM/Morphology、global stage、TCN+decoder 组合或 final ablation。即使 C1 与 C2 的候选分别通过，也不得在本协议内自动组合；非单调交互必须由未来另立协议验证。

## 2. CRD_102 锚点与公共冻结项

锚点固定为 `docs/experiments/crd_v1_candidate_lock_20260809.json` 中三个 CRD_102 `checkpoint_best_local_rr.pt`，lock SHA-256 为：

```text
9a14db8be8af22e1ce1c5a332b4912ab5c13c7fb03cdf1894fc5c6ed6ff7f8cc
```

锚点 checkpoint、seed、selected epoch、训练 commit/config/manifest/validation summary 的路径、大小和 hash 必须完整通过 lock 校验。CRD_102 不重训，不得替换为其他 timestamp、final checkpoint 或新的训练结果。

C1/C2 候选继续冻结：

- research v2 数据、admission、train/validation split、subject/session 隔离和 waveform target；
- 输入 `[B,1,18000]`、100 Hz、180 s；
- 正式 `Pi=S(B(.))`，频带 `0.05–0.70 Hz`；
- `L_sync + 0.25 L_effort`、全部 eligibility/finite 语义与评价指标；
- validation Local-RR checkpoint selector，严格 `<` 保留更早 epoch；
- formal seeds `20260811 / 20260812 / 20260813`；
- 80 epochs、physical batch 128、accumulation 1、6400 optimizer updates、AdamW/LR/no-decay 语义；
- 不计算确认性 p-value/Holm，不构造加权总分。

每个新训练候选必须先在同一干净 commit 下通过定向测试、CUDA synthetic forward/loss/backward finite 和独立 physical-batch-128 acceptance。Acceptance 仍为 128 train / 32 validation / 1 epoch / 1 update，只解除工程阻塞，不形成效果证据；不得因 OOM 临时缩小 formal physical batch。

## 3. C0：10-Hz/Fourier decoder round-trip 审计

### 3.1 科学问题

正式输出频带上限为 0.70 Hz，而 10 Hz 的 Nyquist 为 5 Hz。当前 `fourier_interpolate` 用 `rfft(norm="forward")` 将 1800 点频谱低频 bins 原样复制到 18000 点频谱，再 `irfft(norm="forward")`。因此 learned 100-Hz decoder 不能在未经审计时被表述为“恢复 10 Hz 丢失的呼吸带宽”。

C0 只回答：

> 对完整 validation target，正式呼吸带信号经 10-Hz 抽取和当前 Fourier 上采样后，是否在正式 `Pi` 与任务指标下近似无损？

C0 不运行模型、不读取 checkpoint tensor、不训练、不选择模型。

### 3.2 固定变换

对完整 admitted validation 的 2675 个窗口、7 个 `samp_id`，严格按 `dataset_row_id` 排序：

```text
raw target y100
→ canonical target c100 = S(B_0.05–0.70(y100))
→ c10 = c100[..., ::10]
→ raw round-trip r100 = FourierInterpolate(c10, 1800→18000)
→ canonical round-trip q100 = Pi(r100)
```

逐 sample 正式指标以 `r100` 为 prediction、原 raw target 为 target，通过现有 `evaluate_task_predictions` 计算。数值审计直接比较 `q100` 与 `c100`，并额外报告首尾各 15 秒的误差；不得裁剪边界、删行或填补非有限值。

### 3.3 固定产物与失败语义

固定输出目录为：

```text
runs/crd_v1/crd_102_decoder_roundtrip_audit/
```

一次正式执行必须在干净 Git commit 下生成且禁止覆盖：

```text
decoder_roundtrip_metrics.csv
decoder_roundtrip_metrics_summary.csv
decoder_roundtrip_numerical_audit.json
decoder_roundtrip_decision.json
decoder_roundtrip_manifest.json
```

必须记录 candidate-lock hash、dataset index hash、resolved data/metric identity、commit/dirty、窗口数、`samp_id` 数和 `research_test_used=false`。任一 prediction/target/canonical signal 非有限、行数或 split 不符、primary metric 缺失或已有输出目录时立即失败。

### 3.4 预先冻结的判定

`roundtrip_negligible=true` 必须同时满足：

1. 全部点 `max_abs_error ≤ 1e-4`；
2. 全部点 `RMSE ≤ 1e-5`；
3. 首 15 秒和末 15 秒各自 `RMSE ≤ 1e-5`；
4. Whole RR 与 Local RR direct mean 均 `≤1e-3 bpm`；
5. envelope trajectory 与 global-envelope error direct mean 均 `≤1e-4`；
6. lag-aware signed PCC direct mean `≥0.9999`；
7. 所有五项 primary 数值 finite，joint prediction degeneracy fraction 为 0。

若通过，后续 C2 只能称为 **nonlinear decoder capacity/placement control**，不得称为 10-Hz 采样带宽恢复。若失败，C2 继续关闭，必须先检查 Fourier/边界/数值语义并修订协议；不得根据失败结果临时改用另一上采样器。无论 C0 结果如何，C1 的 backbone 问题在登记 C0 后均可另行激活。

### 3.5 冻结结果与 C1 开放（2026-08-11）

C0 已在干净 commit `5de0c0f459c46e6021034d63bf3d4efbd8a39ac0` 下对完整 2675-window/7-`samp_id` validation 执行一次。Manifest 记录 `research_test_used=false / model_inference_used=false / checkpoint_tensor_read=false`，candidate-lock 与 dataset-index SHA-256 均通过冻结 identity。全部五项 primary finite，joint prediction degeneracy fraction 为 0。

数值结果为：全局 `max_abs_error=5.538454e-7`、RMSE `8.217932e-8`，首/末 15 秒 RMSE 为 `8.158691e-8 / 8.785890e-8`；Whole RR、Local RR、trajectory、global-envelope direct mean 为 `4.754389e-8 / 3.314116e-8 / 4.133708e-8 / 3.691569e-8`，signed PCC 为 `0.9999999999994397`。第 3.4 节全部判定通过，冻结为 `roundtrip_negligible=true`。

固定产物位于 `runs/crd_v1/crd_102_decoder_roundtrip_audit/`，decision/manifest/metrics/summary/numerical-audit SHA-256 依次为：

```text
bece7bf979ce53c73352e3b22fafff0d14840b157f86b8dcbbbcfc52be25a0a8
7f4ed5ac479598957aa44c62ef52ed6e6481571c7d711d28a49be2b4cdba18a3
c3fd06deaf9deb691ae66ca8deb80963beed7dce3dca0ac9a612c7387ecdb075
46aaa0f5b8398be89d0194700a2d7d8298b9a71a516f420c41107fa940db1c9a
f2447a7f755a8a42d1f404df85f573df5666faedb6bd94310238e123bfdf9f0f
```

C0 至此关闭，入口不得重复执行。该结果排除“正式 0.05–0.70 Hz waveform 因 10-Hz 采样而丢失带宽”作为 C2 理由；C2 未来只能检验 nonlinear decoder capacity/placement。现按预注册顺序开放 C1 的实现与工程验收，formal 三 seed 仍须等待当前实现提交下的 CUDA synthetic 和独立 batch-128 acceptance 全部通过并登记。

## 4. C1：CRD_102 vs parameter-matched full-context TCN

### 4.1 唯一问题与模型

C1 只回答：

> 在 CRD_102 的 frontend、refinement、head、loss 和训练协议完全不变时，六个 Local BiMamba2 是否能被参数量相当且覆盖完整 180 秒感受野的 TCN backbone 替代？

候选 ID 预冻结为 `CRD_C101_B0_LOCAL_TCN`。它与 CRD_102 共享并要求同 seed 初始 state 逐 tensor 相同的模块为：

```text
PatchTokenFrontend
2×local refinement
10-Hz CoarseWaveformHead
Fourier interpolation 1800→18000
```

唯一替换：

```text
6×Local BiMamba2(D=96)
→
10×LocalTCNBlock(C=96,H=488,d=1/2/4/8/16/32/64/128/256/512)
```

每个 TCN block 固定为：

```text
x
→ GroupNorm(12,96,eps=1e-5)
→ DepthwiseConv1d(96,k=5,dilation=d,padding=2d,bias=False)
→ SiLU
→ Conv1d(96→488,k=1,bias=False)
→ SiLU
→ Dropout(0.10)
→ Conv1d(488→96,k=1,bias=True)
→ residual add
```

depthwise/首个 pointwise 使用 Kaiming-normal，末层 weight/bias 为 0，GroupNorm 为 1/0。固定感受野为 `4093` 个 10-Hz tokens，大于完整输入的 1800 tokens。单 block 参数为 `94,464`，TCN trunk 为 `944,640`；CRD_102 Local Mamba trunk 为 `951,384`。候选总参数固定为 `1,062,001`，相对 CRD_102 的 `1,068,745` 少 `6,744`（`0.6310%`），满足 `|ΔP|≤2%`。该宽度来自固定 8-channel 网格中的最小绝对参数差，不得按 validation 调整。

该比较属于完整 backbone package 对照；TCN zero-init residual 与 Mamba 官方初始化不同，不得表述为只隔离算子而完全消除优化差异。

### 4.2 正式比较与资源报告

候选只新增三个 formal seeds。逐 sample direct mean、seed arithmetic mean ± sample SD、paired seed/window/`samp_id` differences 必须报告。资源报告固定包括 trainable params、MACs、peak allocated/reserved、forward latency、forward+backward latency、180-s inference latency与 windows/s；资源指标不进入质量主 gate。

相对冻结 CRD_102：

- **quality-superior**：Local RR mean 改善 `≥0.5%`、`≥2/3` paired seeds 改善、PCC drop `≤0.005`、trajectory worsening `≤1.5%`，四项全通过；
- **quality-near / efficiency-alternative**：未达到 quality-superior，但 Local RR worsening `≤0.5%`、PCC/trajectory 守住同一护栏，且同一冻结 benchmark 下 inference throughput 改善 `≥10%`；
- 否则为 **Mamba-retained control failure**。

C1 只形成候选/控制证据，不在本协议内替换 CRD_102 锚点。Whole RR、global envelope、IBI、三层 Spearman 与 failure-strata descriptives 为 secondary，不能覆盖质量门槛。

### 4.3 实现登记与当前工程门槛

实现配置固定为 `configs/crd_v1/crd_c101_b0_local_tcn.yaml`，输出根为 `runs/crd_v1/crd_c101_b0_local_tcn`。模型注册表只新增该一个 C1 variant；没有 C2、TCN+decoder、global TCN 或 width/depth 超参数入口。

结构测试已确认总参数 `1,062,001`、TCN trunk 参数 `944,640`、dilation/感受野、十个 block 初始化 identity，以及 CRD_102 frontend/refinement/head 的同 seed state 逐 tensor 相同。配置、模型、训练与实验定向回归通过；CPU synthetic batch-1 的 model/core-loss/backward finite、waveform shape `[1,1,18000]`、sync/effort eligibility 均为 1。CPU 结果只证明实现链路，不解除正式队列。

下一步必须在该实现提交后的同一干净 commit、目标 GPU 上依次完成：

1. `check_crd_variant.py` CUDA bf16 batch-1 output/input/全部 parameter gradients finite；
2. 独立 acceptance：128 train / 32 validation / 1 epoch / 1 optimizer update；
3. 两个 checkpoint、optimizer state、逐 sample primary metrics finite且 prediction nondegenerate；
4. peak reserved fraction `≤80%`。

任一失败均保持 C1 formal 队列关闭，不允许临时改变 H、block、dilation、batch 或 AMP。两项 CUDA 工程结果由主协议登记后才可开放三个 formal seeds。

上述工程门槛已在干净 commit `529de747cfee17675432fad1a969570703df1791` 下通过。CUDA bf16 synthetic batch-1 的 waveform shape 为 `[1,1,18000]`，loss `0.347592`，sync/effort eligibility 均为 1，output/input/全部 parameter gradients finite，peak allocated `92.88 MiB`。

独立 acceptance 位于 `/tmp/crd_c101_b0_local_tcn_batch128_acceptance/20260811_142949_605712`，严格使用 128/32 个 train/validation windows、1 epoch、1 optimizer update。Best/final checkpoint 均为 epoch 1、update `1/1`；各含 108 个 finite model tensors 和 324 个 finite optimizer tensors。32 条 validation metrics 的五项 primary 全部 finite、joint prediction degeneracy 为 0；peak allocated/reserved 为 `10018.08/10664.00 MiB`，reserved fraction `66.9161%`，低于 80% 工程线。单 epoch loss/Local RR/PCC 等数值不进入模型效果解释。

至此 C1 三个 formal seeds 的工程阻塞解除。Formal 必须在包含本登记的统一新干净 commit 上顺序或独立 GPU 并行运行，输出到 `runs/crd_v1/crd_c101_b0_local_tcn/seed_<seed>/`；任何中断 run 不凭 best checkpoint 纳入比较。C2 仍等待 C1 三 seed 完整性审计和冻结 decision，不得实现。

三个 formal runs 随后在统一干净 commit `930212ac66cb95697b659fc7504259cfa3cf71c2` 下完成，唯一 run 路径为：

```text
runs/crd_v1/crd_c101_b0_local_tcn/seed_20260811/20260811_143659_197397
runs/crd_v1/crd_c101_b0_local_tcn/seed_20260812/20260811_151742_317768
runs/crd_v1/crd_c101_b0_local_tcn/seed_20260813/20260811_143716_511886
```

三个 run 均完成 80 epochs/6400 updates、2675 条 validation metrics、best/final checkpoint 与 peak-reserved 审计；selected epochs 为 `25/26/12`。当前只允许预先实现的 `scripts/summarize_crd_c1.py` 在新干净 commit 上执行一次冻结汇总，重算逐 sample summary、paired window/`samp_id`、failure-strata descriptives 与第 4.2 节 decision；输出固定为 `runs/crd_v1/crd_c1_validation_summary/` 且禁止覆盖。冻结 summary 完成前不得实现 C2。

### 4.4 冻结结果与 C2 开放（2026-08-11）

冻结汇总器首次执行在创建输出目录前因把逐 sample 分层 Spearman 误当作 seed-summary 字段而停止，没有生成部分 decision 或结果产物；修正仅将 seed-level 汇总限定到现有七项 summary，未改变第 4.2 节模型门槛。修正后唯一 summary 从干净 commit `f0ac01b9f265fcc6f7737fb873c64285bca6f2da` 生成，重新审计 training commit、三个 lifecycle、checkpoint/optimizer、逐 sample metrics、candidate lock 与 failure-diagnostic identities，未读取 research-test。

三 seed arithmetic mean ± sample SD：

| 模型 | Whole RR ↓ | Local RR ↓ | Trajectory ↓ | Global envelope ↓ | Signed PCC ↑ | IBI MedAE ↓ | IBI coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CRD_102 | 0.515775 ± 0.020060 | 0.556161 ± 0.009055 | 0.150150 ± 0.001296 | 0.187512 ± 0.002438 | 0.864672 ± 0.001376 | 0.080181 ± 0.001384 | 0.839027 ± 0.006859 |
| C1 TCN | 0.465425 ± 0.014293 | 0.534893 ± 0.008805 | 0.153974 ± 0.001321 | 0.194656 ± 0.003877 | 0.860875 ± 0.002674 | 0.084177 ± 0.000604 | 0.835631 ± 0.005103 |

冻结门槛逐项结果：

1. Local RR 改善 `3.8241%`，通过 `≥0.5%`；
2. paired Local RR 为 `3/3` seeds 改善，通过；
3. signed PCC absolute drop `0.003797`，通过 `≤0.005`；
4. trajectory worsening `2.5468%`，失败 `≤1.5%`。

因此 `quality_superior=false`；quality-near 同样因 trajectory 护栏失败，不需要 efficiency benchmark，冻结 outcome 为 `mamba_retained_control_failure / retain CRD_102`。Whole RR 改善 `9.7621%` 与 Local RR 收益属于任务交换背景，不能覆盖 trajectory；global-envelope/IBI-MedAE/coverage 也分别退化 `3.8098%/4.9836%/0.003396 absolute`。

Failure-strata descriptives 显示，TCN 的 Local RR 改善集中在既有 persistent Local-RR failure、high-modulation failure 和 exact-state matched cases；相应 candidate-minus-base mean 约为 `-0.3468/-0.4703/-0.5933 bpm`，而非 persistent windows 为 `+0.0141 bpm`。这支持“full-context TCN 对既有难例存在局部补偿”的描述，不构成因果主张，也不推翻总体 gate；trajectory 在 high-modulation 层恶化约 `+0.00945`。

冻结产物位于 `runs/crd_v1/crd_c1_validation_summary/`。Selection/manifest/seed-summary/variant-summary/paired-window/paired-samp/failure-strata SHA-256 依次为：

```text
ba320e81a1533c974a3f4b68e73ba41d097957d6702b9a7be1d6540deecc7e0f
4c31f561f59eb0f57bd38dace89134b082698a49ceff270a0a86634096b979f4
c1e652481093ace32c79d27775f36fa233279c5a224de6e3708f8651c02f25d7
25c7f8a46f9bcd6371cc1a82ba1348a16e1e1861dfe7013f4f3d3da8b05c1cc1
7ecc6fadfd62ef73bb37c8a5388c35e6c9dd23f4ab476685fc183300dd3aa3c4
4b8fa45e3e47c202bd0867f4f6f94a55aeea7bd4e5bda52d35adef54e1c4b01a
7e0344eba1fa41a02b6b7f9e9a28a3fe09cbd448177779c9777e19019f79d15c
```

C1 至此关闭，summary 入口不得重复运行或用于改变门槛。按预注册顺序，现只开放 C2 两个候选的实现与工程验收；C2 继续固定使用 CRD_102 Mamba backbone，不得把 C1 TCN 与 decoder 组合。

## 5. C2：10-Hz capacity 与 learned 100-Hz placement

### 5.1 三臂因果结构

C2 固定继续从 CRD_102 分支，不因 C1 结果改变 backbone。reference 为锁定 CRD_102，不重训。两个候选为：

```text
CRD_C201_DECODER_10HZ_CAP
CRD_C202_DECODER_100HZ
```

令现有 `CoarseWaveformHead` 在最终 `Conv1d 32→1` 前的 feature 为 `h32∈R^(B×32×1800)`，原始标量输出为 `y10`。两个候选共享同一个参数结构：

```text
R(h) = Conv1d(32→32,k=1,bias=False)
       → SiLU
       → Conv1d(32→1,k=1,bias=True)
```

首层 Kaiming-normal；末层 weight/bias 全零；使用独立确定性子 seed `decoder_residual`。每个候选只增加 `1,057` 个参数，并在初始化时严格输出 CRD_102 的原 waveform。

```text
C201: waveform = Fourier(y10 + R(h32))
C202: waveform = Fourier(y10) + R(Fourier(h32))
```

C201 的正式 `waveform_10hz=y10+R(h32)`；C202 的 `waveform_10hz=y10` 仅作 provenance，loss/metrics 仍只消费 100-Hz `waveform`。C202 必须额外报告外部 `Pi` 前 residual 的 `0.70 Hz` 以上能量占比，但该值只作数值解释。

该设计用完全相同参数量区分“增加 nonlinear head capacity”和“把 nonlinear decoding 放到 100 Hz”；不得额外加入 temporal convolution、transposed convolution、skip、norm 或 learned interpolation。

### 5.2 冻结门槛与解释

C201/C202 各新增三个 formal seeds。两者各自相对 CRD_102 的基本资格门槛均为：Local RR mean 改善 `≥0.5%`、`≥2/3` paired seeds 改善、PCC drop `≤0.005`、trajectory worsening `≤1.5%`。

只有 C202 通过基本资格，且相对 C201 同时满足 Local RR 改善 `≥0.25%`、`≥2/3` paired seeds 改善、PCC drop `≤0.003`、trajectory worsening `≤1.5%`，才支持“100-Hz nonlinear placement 提供额外价值”。

冻结解释表：

| C201 vs 102 | C202 vs 102 | C202 vs C201 | 允许结论 |
|---|---|---|---|
| 通过 | 任意 | 未通过 | 额外 decoder capacity 有用；100-Hz placement 无独立证据 |
| 任意 | 通过 | 通过 | 100-Hz nonlinear placement 有额外开发证据 |
| 未通过 | 未通过 | — | 保留原 10-Hz scalar + Fourier decoder |
| 未通过 | 通过 | 未通过 | 只称 C202 decoder package 有效，不归因于 100 Hz |

### 5.3 实现登记与当前工程门槛

实现配置固定为：

```text
configs/crd_v1/crd_c201_decoder_10hz_cap.yaml
configs/crd_v1/crd_c202_decoder_100hz.yaml
```

两个候选总参数均为 `1,069,802`，恰比 CRD_102 增加 `1,057`。实现只把现有 `CoarseWaveformHead` 的最终 32-channel feature 暴露给同一个 `DecoderResidual`；旧 variant 继续调用原 `head(latent)`，state dict 与 forward 数学不变。C201/C202 的 frontend、六个 Local BiMamba2、refinement 和 coarse head 与 CRD_102 同 seed 逐 tensor 相同，两个 residual state 也逐 tensor相同；zero-init 下两个候选 waveform 与 CRD_102 逐点完全相同。

配置、参数、共享 state、zero-init identity、旧模型 forward 兼容和训练/experiment 定向测试必须全部通过。官方 Mamba fast path 不支持 CPU forward，因此 C2 不以 CPU synthetic 作为工程门槛；CPU 单元测试通过后，必须在同一干净 commit、目标 GPU 分别完成：

1. CUDA bf16 batch-1 output/input/全部 parameter gradients finite；
2. 各自独立 128 train / 32 validation / 1 epoch / 1 update acceptance；
3. 两个 checkpoint与 optimizer state finite，五项 primary finite，prediction nondegenerate；
4. 各自 peak reserved fraction `≤80%`。

任一 variant 工程失败只阻塞该 variant，不允许改变 residual、插值位置、batch 或 AMP。两个候选均通过并由主协议登记前，不开放任何 C2 formal seed。

上述工程门槛已在统一干净 commit `0e2d05824a50426e3ff0443a6875829422a6160d` 下通过。C201/C202 CUDA bf16 synthetic batch-1 的 waveform shape 均为 `[1,1,18000]`，loss 均为 `1.245275`，sync/effort eligibility 均为 1，output/input/全部 parameter gradients finite；peak allocated 分别为 `349.17/349.21 MiB`。

独立 acceptance 路径为：

```text
/tmp/crd_c201_decoder_10hz_cap_batch128_acceptance/20260811_162428_134016
/tmp/crd_c202_decoder_100hz_batch128_acceptance/20260811_162522_128723
```

两项均严格完成 128/32 个 train/validation windows、1 epoch、1 optimizer update；best/final checkpoint 与 optimizer state 全 finite，32 条 validation metrics 的五项 primary 全部 finite、joint prediction degeneracy 为 0。C201 peak allocated/reserved 为 `8480.77/9936.00 MiB`、reserved fraction `62.3479%`；C202 为 `8907.78/9192.00 MiB`、`57.6794%`，均低于 80% 工程线。单 epoch loss/metrics 不进入效果解释。

至此 C201/C202 各三个 formal seeds 的工程阻塞解除。Formal 必须来自包含本登记的统一新干净 commit，输出分别固定到 `runs/crd_v1/crd_c201_decoder_10hz_cap/seed_<seed>/` 与 `runs/crd_v1/crd_c202_decoder_100hz/seed_<seed>/`。任何中断 run 不凭 best checkpoint 纳入比较；在六个完整 run 与冻结 summary 完成前，不追加 temporal decoder、TCN+decoder 或 research-test。

## 6. 失败分层、停止条件与未来证据

C1/C2 必须预先复用已冻结的 CRD_102 failure diagnostic 定义，报告 persistent failure/control、waveform-confidence strata、high-modulation failure 与 matched-observability case/control 的 secondary descriptives。不得用这些 validation 分层重新加权 loss、筛样本、选择 checkpoint 或事后定义新 gate。

任一新候选未通过完整 lifecycle、finite、identity/parameter、acceptance 或正式 gate 时，保留 CRD_102，不补跑超参数变体。C0/C1/C2 均禁止读取 research-test；若最终形成新候选，只能建立新的 candidate lock，并在新锁定 cohort、外部数据或 prospective holdout 上获得强泛化证据。
