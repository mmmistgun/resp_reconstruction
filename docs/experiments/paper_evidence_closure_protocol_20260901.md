# 论文实验补充与证据闭环执行协议

日期：2026-09-01

协议 ID：`paper-evidence-closure-v1-20260901`

状态：**P0–P4、P4-S 与 P4-T 均已完成并冻结。Center-60 的 42-checkpoint 附件分支在独立 test 上支持“延长输入没有稳定 RR 收益”，60 s 是该任务当前 RR 优先的合理选择；center-30 的 validation 45 s 下限未获 test 确认，最短合理边界仍未定位。两种输出任务保持独立 panel，不回写论文主模型实验。效率 benchmark 未授权。**

## 1. 定位与边界

本协议把论文所需的补充实验整理为可执行、可审计的独立任务。设计来源文件
`/mnt/disk_code/marques/paper/resp_rec/paper_rewriting_output/实验补充与证据闭环计划.md`
保持只读，不由本协议修改、覆盖或补写。

本协议不重开、重跑或改写已经关闭的 CRD、CRD-TF、CRD-TF-W v2、RTM-v1 训练、评价、审计或汇总入口。
所有新增实现使用独立配置、入口和输出根目录；历史 `runs/`、checkpoint、cache、CSV、图表及 manifest 均保持原样。

本协议只处理实验可实施性和执行合同，不负责医学依据或文献来源整理。

用户已冻结以下论文决策：

1. `crd_tf102_w`（下称 W0）作为论文主模型。该选择以呼吸节律为最高任务优先级：W0 相对 W3 的
   Whole/Local RR 优势大于 W3 在努力与 PCC 轴上的收益。该定位是任务优先级下的论文主模型选择，
   不改写既有 Pareto 结果，也不声称 W0 在五项指标上逐项支配 W3。
2. 窗口长度实验统一采用“输入 60/90/180 s，只输出中心 60 s”的共同任务。
3. 窗口长度实验建立共同的 `Pi_60`、中心 60 s loss、checkpoint selector 和五项中心指标。
4. IoT 实验只预注册测量对象、测量方法和必报结果，不预设模型能否流式、实时或适合部署的结论。
5. IoT 结果不得被表述为 causal/streaming 重建证据；是否具有足够处理余量只能在结果产生后讨论。

## 2. 总体阶段

| 阶段 | 内容 | 是否训练 | 当前状态 |
|---|---|---:|---|
| P0 | 既有代表性方法协议兼容性与表格审计 | 否 | 已完成并冻结 |
| P1 | 中心 60 s 变长输入任务实现与定向测试 | 否 | 已完成 |
| P2 | 中心任务 synthetic GPU 工程验收 | 否 | 已完成并冻结；统一 `128×1` |
| P3 | C201/W-reduced × 60/90/180 s 单 seed 诊断矩阵 | 是，6 runs | 已完成并冻结；触发候选模型×上下文交互 |
| P4 | 条件触发的三 seed 窗口正式矩阵与冻结汇总 | 是，追加 12 runs | 18/18 完成并冻结；描述性结果留档 |
| P4-S | center-30 的 30/45/60/90 s 短窗口分支 | 条件训练，最终 24 runs | 24/24 与三 seed汇总已冻结；validation 阶段关闭 |
| P4-T | center-30/center-60 上下文任务完整独立测试集附件 | 否；42 checkpoints inference | 42/42 与分任务冻结汇总均完成；附件关闭 |
| P5 | 时频功能证据整理与可选局部干预 | 否 | 既有证据可整理；新增推理关闭 |
| P6 | 多属性波形图与 RR 区间分析 | 否或只读推理 | 待实现；test 访问关闭 |
| P7 | W0/W3/D4 端到端 IoT 效率测量 | 否 | 未授权执行 |

每个阶段必须使用新目录且禁止覆盖。实现、训练、独立测试集访问和长时间 benchmark 分别需要用户明确授权；
前一阶段完成不自动授权后一阶段。

## 3. P0：既有代表性方法兼容性审计

### 3.1 目标

只读检查冻结结果能否进入统一代表性方法表，不重评 checkpoint、不重新运行 test、不重算 prediction。
至少核对：

- dataset index、split、2310 个 test rows 与 `dataset_row_id` 集合；
- input/target 载体、100 Hz/180 s 窗口与 dataset admission；
- `Pi` 的频带、尺度规范化及五项 primary 实现语义；
- checkpoint 是否由完整 validation Local RR 选择；
- sample-direct 聚合、三 seed mean 与 `ddof=1` sample SD；
- prediction 退化、eligibility 和 nonfinite 处理；
- 训练模型是否完整保留三个 seed，确定性方法是否正确标为无 seed。

候选来源至少覆盖 F0、IEWT、早期 B0/T2/T4、C201/M/W/MS、W3 和 RTM-v1 五个冻结候选。
论文主表可按预先写入 manifest 的代表选择规则缩减，但完整兼容性审计不得静默遗漏已有 qualified/Pareto 候选。

W0 在论文中固定标为 `primary proposed model`。W3 固定标为 `scale-density trade-off`，D4 固定为
`validation-only efficiency trade-off`；不得把 D4 与拥有独立测试集结果的方法放在同一 test-quality 列中。

### 3.2 产物

固定输出到新目录：

```text
runs/paper_evidence_v1/p0_comparison_audit/
```

至少包含：

- `source_artifact_audit.csv`
- `protocol_compatibility.csv`
- `representative_method_rule.json`
- `paper_primary_metrics_table.csv`
- `manifest.json`

若任何方法无法证明 metric/split/target/checkpoint/aggregation 兼容，应在表中显式标记 `incompatible`，不得用
缺失字段推断兼容。RTM-v1 独立测试集结论锁未由用户确认前，其结果不得进入论文冻结表。

## 4. P1：中心 60 s 变长输入任务

### 4.1 独立任务身份

窗口长度任务使用子协议 ID：

```text
paper-center-context-v1-20260901
```

它是游离于论文主模型实验之外的独立 train/validation 上下文敏感性任务，不回写旧 180 s 模型结论，
不参与 W0/W3/C201/RTM 的既有选择、Pareto 或论文主模型定位。旧 W0/C201 checkpoint 只作为结构来源和参数合同参考，
不得用于初始化新训练；本任务的时频臂固定称为 `W-reduced-center60`，不得称为严格 W0 或 W0 窗长消融。

### 4.2 嵌套输入与共同 target

继续使用当前 research-v2 admitted 180 s rows 和既有 train/validation subject/session split。每个父 row
只产生一个实验样本身份，并构造三种确定性嵌套视图：

| 输入长度 | 100 Hz 父窗口切片 | 中心 target |
|---:|---|---|
| 60 s | `[6000:12000)` | `[6000:12000)` |
| 90 s | `[4500:13500)` | `[6000:12000)` |
| 180 s | `[0:18000)` | `[6000:12000)` |

所有长度使用相同父 `dataset_row_id`、`samp_id`、`coupling_state_id`、split 和 sample 顺序。Dataset admission
继续由父 180 s row 的冻结规则决定，不对 60/90 s 重新拟合质量阈值或删除样本。

输入直接裁剪既有 `bcg_rawish_segment_soft_z_key`，不按长度重新做 soft-z，使三种输入共享同一父窗口增益参考，
避免把“重新归一化”混入上下文长度。Target 直接裁剪既有 `target_waveform_segment_soft_z_key`；正式 target
表示随后统一经过 `Pi_60`。

必须审计三种视图：shape、finite、父 row 身份、中心 target 逐点相同、train/validation row 零交集和
subject/session 隔离不变。

### 4.3 两个模型轴

窗口矩阵只包含两个模型轴：

1. `C201-center60`：C201 时域主干、六层 Local BiMamba2、refinement 与 10-Hz capacity decoder；
2. `W-reduced-center60`：与 C201 完全相同的主干和 decoder，只增加全频程 49-scale 降密度 CWT branch 与 FiLM。

选择 C201/W-reduced 配对是为了在一个辅助实验内使模型间唯一科学差异保持为精简 W 时频条件，避免引入另一套
RTM 主干、宽度或 decoder。该比较只回答“合理精简 W 表征下的上下文效应”，不外推到严格 W0。
每个模型在 60/90/180 s 下参数数必须完全相同；同一 seed 下 C201/W-reduced 共享模块初始 state 必须逐 tensor 相同。

新实现不得放宽或修改既有 `CRDCoarseModel`、`CRDTfV1Model`、W cache reader 的固定 180 s 合同。
应新增独立的 variable-context model/runner，防止新语义进入历史入口。

### 4.4 变长 forward 与中心 decoder

对输入长度 `N∈{6000,9000,18000}`：

1. Patch frontend 的 token 输出插值到 `N/10` 个 10-Hz latent；
2. 六层 Local BiMamba2 在完整 `N/10` latent 上运行；
3. W-reduced 的 FiLM 在完整 latent 上生效；
4. refinement 和 coarse-head feature extractor 在完整 latent 上运行；
5. 从 32-channel head features 裁剪中心 600 个 10-Hz tokens；
6. coarse output 与 C201 decoder residual 只作用于中心 600 features；
7. 中心 10-Hz waveform 通过冻结 Fourier 插值生成 6000 点 raw output。

对应的 10-Hz 中心裁剪为：

| 输入长度 | latent 长度 | 中心裁剪 |
|---:|---:|---|
| 60 s | 600 | `[0:600)` |
| 90 s | 900 | `[150:750)` |
| 180 s | 1800 | `[600:1200)` |

先在完整 latent 上执行双向主干、FiLM、refinement 和带局部卷积的 head features，再裁剪中心，是为了让外侧输入
能够真实影响中心预测；不得在主干前裁剪成 60 s，也不得把外侧 target 纳入 loss。

### 4.5 变长 W 特征

W-reduced 对每种输入视图独立计算 CWT，禁止先对父 180 s 输入计算 CWT 再裁剪，以免 60/90 s 视图通过 CWT
边界或卷积读取范围外信息。

固定：

- 100 Hz 输入；
- Morlet `mu=13.4`；
- 从原 12 voices/octave、97 点全频程名义网格固定取偶数 indices `0,2,…,96`，得到 49 个 target frequencies，
  等效约 6 voices/octave；所有长度使用同一 49 点名义网格；
- reflect padding；
- `log1p(abs(CWT))`；
- 每 50 samples 平均池化到 2 Hz。

输出 shape 分别为 `[49,120] / [49,180] / [49,360]`。不同长度按相同 49 点名义频率网格做 direct
length-specific scale mapping，不从 97-scale feature 事后裁 view。连续 scales 必须严格递增；受离散 FFT 峰值量化影响，
保存的实际 mapped centers 允许非降且可重复，但必须记录 duplicate count 和最大名义频率绝对误差，不得按长度删 scale
或事后设置误差淘汰阈值。W encoder 保持同一组参数，只把其时间轴动态插值到 `N/10`，不得增加长度特定参数。

Cache 只允许 train/validation，按长度写入三个独立、不可覆盖的 input-only cache；不得读取 target，
也不得创建 test cache。Manifest 必须保存父 row/hash、切片、频率、shape、dtype、finite 和 access flags。

## 5. `Pi_60`、中心 loss 与中心指标

### 5.1 `Pi_60`

对 6000 点 raw prediction/target 定义：

```text
B_60 = 60 s / 6000-point hard FFT projection, 0.05–0.70 Hz, endpoints included
S_60 = center and divide by whole-center-60 RMS with eps=1e-8
Pi_60 = S_60 o B_60
```

它与旧 `Pi` 使用相同频带、FFT convention、finite 规则和不可辨识尺度语义，但作用域固定为中心 60 s。
不允许先在 90/180 s 上执行旧 `Pi` 再裁剪，因为那会让外侧 target 参与中心 target 的投影和尺度规范化。

### 5.2 中心 loss

唯一训练目标固定为：

```text
L_center = L_sync_60 + 0.25 * L_effort_60
```

- `L_sync_60`：在 `Pi_60` 输出上搜索 `-30…30` samples，使用共同 5940-point 支撑与 signed PCC；
- `L_effort_60`：复用所选 lag，对共同支撑上的 10 s log-RMS、5 s step 包络计算相关损失；
- target eligibility、prediction finite、无 eligible sample 聚合和梯度累积语义与现行 core loss 一致；
- loss 只读取中心 target，外侧 30/120 s 没有监督。

### 5.3 五项中心指标

原 180 s Whole RR 与 60 s Local RR 在中心 60 s 任务中会退化为同一个频谱估计，禁止用两个名称重复报告。
本任务固定五项中心指标：

1. `center_rr_mae_bpm`：中心 60 s single-periodogram dominant spectral RR MAE；
2. `center_ibi_medae_sec`：中心 60 s lag-aligned positive-peak IBI-MedAE；
3. `center_envelope_trajectory_mae`：10 s/5 s 的 11 点 median-centered log-RMS trajectory MAE；
4. `center_global_envelope_modulation_error`：上述 11 点 `Q90-Q10` 范围误差；
5. `center_lag_aware_signed_pcc`：中心 60 s、`±0.30 s` signed PCC。

`center_ibi_medae_sec` 必须与 `center_ibi_coverage`、target eligibility 和 interpretable fraction 同时报告；
coverage 是强制 companion，不构成第六项可独立排序指标。RR prediction 退化固定计 39 bpm，PCC prediction 退化
固定计 `-1`，非有限 prediction 使整个 checkpoint 评价失败。所有 dataset summary 继续 sample-direct mean。

不为中心任务计算 coherence、nDTW、原 Whole/Local RR、旧 envelope strata 或额外总分。

### 5.4 Checkpoint selector

每 epoch 在完整 validation 2675 个父 rows 上计算 `center_rr_mae_bpm`，严格选择更小值；完全相同时保留更早
epoch。其他四项中心指标和 IBI coverage 不参与 epoch 选择。Selected checkpoint 完成后才计算一次完整五项中心
指标与 companion。Early stopping 关闭。

## 6. P2：实现与工程验收

### 6.1 代码影响面

后续实现建议新增而不是修改历史入口：

```text
resp_train/paper_evidence/center_context_data.py
resp_train/paper_evidence/center_context_model.py
resp_train/paper_evidence/center_context_loss.py
resp_train/paper_evidence/center_context_metrics.py
resp_train/paper_evidence/center_context_experiment.py
resp_train/paper_evidence/center_context_acceptance.py
scripts/train_paper_center_context_v1.py
scripts/accept_paper_center_context_v1.py
configs/paper_evidence_v1/center_context_v1.yaml
tests/test_paper_center_context_v1_*.py
```

具体拆分可在实现时压缩，但不得复用旧 formal 输出目录或放宽旧 config gate。

### 6.2 定向测试

至少覆盖：

- 三种输入切片、共同中心 target 与 split 身份；
- 6000-point `Pi_60` 的 Torch/NumPy 等价、频带、尺度、finite 和梯度；
- 60/90/180 s C201/W-reduced forward shape 与各自跨长度参数数一致；
- 三种 latent 裁剪索引和中心对齐；
- W-reduced 特征 `[49,120/180/360]`、49-scale 偶数索引网格、实际中心重复/误差审计和无范围外泄漏；
- 同 seed C201/W-reduced 共享 state identity、不同长度同模型 state identity；
- `L_sync_60/L_effort_60` identity、延迟、反相、努力变化和无 eligible 语义；
- 五项中心指标的 identity/error ordering、11 点包络、IBI coverage 和退化失败值；
- strict center-RR checkpoint tie；
- train/validation-only access 与 test/cache 拒绝；
- synthetic 输入的确定性、三种长度 shape/finite、CPU 注入 Mamba 下的 loss/backward/optimizer step；
- P2 六臂 batch-1 与最大臂 batch-128 身份、output 不可覆盖、失败 lifecycle、access flags 与 artifact hash。

### 6.3 工程执行顺序

P2 不运行 CPU lifecycle 或真实数据 lifecycle。官方 Mamba2 fast path 由 CUDA 执行；CPU 仅用定向测试验证结构、
数据合同和注入 shape-preserving Mamba 后的 model/loss/backward/optimizer plumbing。

用户在实现提交后的干净 commit 手动执行唯一 P2 命令：

```bash
CUDA_VISIBLE_DEVICES=<GPU> ./.venv/bin/python scripts/accept_paper_center_context_v1.py \
  --config configs/paper_evidence_v1/center_context_v1.yaml \
  --device cuda:0 \
  --confirm-gpu-acceptance
```

该入口不读取 dataset/index、train/validation/test、cache 或 checkpoint。它先按固定顺序对 C201/W-reduced ×
60/90/180 s 六臂各执行一次 batch-1 真实 Mamba forward/loss/backward，再仅对计算上界
W-reduced-180 执行 physical batch 128 的 forward/loss/backward/AdamW step。W-reduced-180 包含与 C201 相同的主干并
增加 W branch，故其 batch-128 结果作为全六臂的保守显存准入。回执必须记录六臂完整性、finite、shape、参数数、
peak allocated/reserved、设备/依赖/commit、access flags 和最终 decision；固定 commit identity 目录禁止覆盖。

若 W-reduced-180 在冻结实现下不能满足 batch 128，必须在任何 P3 run 前为全部六臂统一选择
`64×2` 或 `32×4`，并增加同 batch 的身份/更新等价测试；不得按长度或模型单独降 batch。

P2 只形成实现证据，不形成窗口长度效果结论。按仓库约束，该 GPU 命令由用户执行。

### 6.4 冻结结果

P2 从干净 commit `7249fca27e5a960a3e74e43a5acf6ac0abc3a4b5` 在 NVIDIA GeForce RTX 4070 Ti SUPER
上完成。固定回执为
`runs/paper_evidence_v1/center_context/p2_gpu_acceptance/7249fca27e5a/acceptance_receipt.json`，SHA-256 为
`cb43ee480018afd483ce2bef15342d5bdaf29b6bc999d58ce209abac58df04ce`；artifact manifest SHA-256 为
`09b9e3347967cd7a4eac45b39c4054c4db689ccbbfe1bdaf2a2e23469957d582`。

六个 batch-1 arms 全部完成且 output/loss/gradient/parameter finite，参数数固定为 C201 `1,069,802`、W-reduced
`1,219,850`。W-reduced-180 的 physical batch 128 完成一次 AdamW step，peak allocated/reserved 为
`10,834.94 / 10,948.00 MiB`，reserved 占设备总显存 `68.70%`，decision=`batch128_accepted`。Access receipt 确认
dataset、cache、train/validation/test、checkpoint 均未访问。Batch-1 序列没有 warmup，首臂包含一次性 CUDA 初始化开销，
其显存记录只作 finite/shape 工程回执，不用于跨臂效率比较。

P2 据此冻结六臂统一 `physical batch=128 / gradient accumulation=1`，不触发 batch fallback。

## 7. P3/P4：窗口长度实验矩阵

### 7.1 固定配置

| 实验 ID | 模型 | 输入 | 输出 |
|---|---|---:|---:|
| `CCV1_C201_60` | C201-center60 | 60 s | 中心 60 s |
| `CCV1_C201_90` | C201-center60 | 90 s | 中心 60 s |
| `CCV1_C201_180` | C201-center60 | 180 s | 中心 60 s |
| `CCV1_WR_60` | W-reduced-center60 | 60 s | 中心 60 s |
| `CCV1_WR_90` | W-reduced-center60 | 90 s | 中心 60 s |
| `CCV1_WR_180` | W-reduced-center60 | 180 s | 中心 60 s |

三份 W cache 已完成只读验收并冻结：

| 输入 | 固定目录 | manifest SHA-256 | train/validation shape |
|---:|---|---|---|
| 60 s | `60s_215c24b05b2e438f311edf13d20903d617131378a952b893234582c670848c6f` | `e9d270c930d6862f9b5a9cbdb3c26765fe3b57d2573946640cba99e597389793` | `[10141/2675,49,120]` |
| 90 s | `90s_eb33cf00339545a75c459ca864bb97b333e605aa4d6c8f3d78f36b0df162587c` | `9442a33ea2c633b15642d033efa3d3e09278f30c4692a707abc9de460c784757` | `[10141/2675,49,180]` |
| 180 s | `180s_8414c1a1dd1acc27640bc22074805aad2f327ff7d56eca2ee4ea8193ab71b827` | `72402d8543adf3cda504967b87fc4f9528cacca58b9aa080f0dcfe06707037fe` | `[10141/2675,49,360]` |

三份 cache 的 dataset index、train row 与 validation row SHA-256 分别完全一致；所有 artifact hash 与 manifest 相符，
`complete/input_only/target_read/test_read=true/true/false/false`，lifecycle 均为 `complete`。Formal W configs 同时冻结
cache 目录与 manifest SHA-256，runner 在构建 dataloader 前再次校验。

共同训练合同：

- 完整 train/validation：`10141/2675` 父 rows；
- seeds：`20260811 / 20260812 / 20260813`；
- 80 epochs、6400 optimizer updates/run；
- AdamW，max/min LR=`3e-4/3e-5`，5% warmup + exact cosine；
- betas=`0.9/0.999`、eps=`1e-8`、weight decay=`1e-4`、grad clip=`1.0`；
- bf16、TF32=false、cuDNN benchmark=false、early stopping=false、resume=false；
- 同 seed 六臂使用相同 train shuffle order、父 row 顺序和初始化 seed；
- 只读取 train/validation，不创建或读取 test cache。

### 7.2 P3 单 seed 配置、命令与诊断

P3 固定运行六臂的 seed `20260811`，共 6 runs。六项配置必须在查看任何效果结果前完成冻结；除 OOM、非有限、
identity、lifecycle 或数据错误外，不得依据先完成 run 的 validation 结果取消其余正常 run。

固定配置为：

```text
configs/paper_evidence_v1/p3_ccv1_c201_60.yaml
configs/paper_evidence_v1/p3_ccv1_c201_90.yaml
configs/paper_evidence_v1/p3_ccv1_c201_180.yaml
configs/paper_evidence_v1/p3_ccv1_wr_60.yaml
configs/paper_evidence_v1/p3_ccv1_wr_90.yaml
configs/paper_evidence_v1/p3_ccv1_wr_180.yaml
```

用户从包含这些配置与本协议登记的统一干净 commit 执行全部六项；同一 GPU 队列中每项完成后继续下一项，出现工程失败则保留目录并停止：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<GPU> ./.venv/bin/python \
  scripts/train_paper_center_context_v1.py --config configs/paper_evidence_v1/p3_ccv1_c201_60.yaml \
  --confirm-formal-training

env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<GPU> ./.venv/bin/python \
  scripts/train_paper_center_context_v1.py --config configs/paper_evidence_v1/p3_ccv1_c201_90.yaml \
  --confirm-formal-training

env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<GPU> ./.venv/bin/python \
  scripts/train_paper_center_context_v1.py --config configs/paper_evidence_v1/p3_ccv1_c201_180.yaml \
  --confirm-formal-training

env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<GPU> ./.venv/bin/python \
  scripts/train_paper_center_context_v1.py --config configs/paper_evidence_v1/p3_ccv1_wr_60.yaml \
  --confirm-formal-training

env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<GPU> ./.venv/bin/python \
  scripts/train_paper_center_context_v1.py --config configs/paper_evidence_v1/p3_ccv1_wr_90.yaml \
  --confirm-formal-training

env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<GPU> ./.venv/bin/python \
  scripts/train_paper_center_context_v1.py --config configs/paper_evidence_v1/p3_ccv1_wr_180.yaml \
  --confirm-formal-training
```

实际调度使用两张同型号 GPU，各进程配置内均保持逻辑 `cuda:0`，只通过 `CUDA_VISIBLE_DEVICES` 选择物理卡；这只改变
运行排队，不增加实验维度。两个顺序队列为：物理 GPU 0 执行 `C201-60 → C201-90 → W-reduced-180`，物理 GPU 1
执行 `W-reduced-60 → W-reduced-90 → C201-180`。六项仍来自同一干净 commit，输出 identity 互不覆盖。

P3 gate 只接受 `stage=p3_single_seed / seed=20260811 / 80 epochs / physical batch 128 / accumulation 1`，固定输出根为
`runs/paper_evidence_v1/center_context/p3_single_seed/`。每项启动时显式设置并校验
`matmul TF32=false / cuDNN TF32=false / cuDNN benchmark=false`，实际状态写入 artifact manifest。P4 的另外两个 seed
当前不能通过 P3 gate。

该 seed 在未来 P4 中预注册为三个正式 seed 之一：若 P4 开放，只追加 `20260812/20260813` 的 12 runs，
不重跑或丢弃 seed `20260811`。P3 单独只能形成方向性诊断，不能写成稳定窗口长度结论。

P3 汇总按模型分别报告 `60→90`、`90→180`、`60→180` 的五项中心指标变化，并报告 W-reduced−C201 差异随长度的
变化。升级信号定义为以下任一可解释模式：

1. 至少一个模型在 RR、IBI、trajectory 或 global-envelope error 上随长度单调改善，且 60→180 相对改善
   至少 `0.5%`；
2. 60→90 达到上述材料性，90→180 的同指标绝对相对变化小于 `0.5%`，形成候选饱和模式；
3. W-reduced/C201 在同一中心指标上的长度响应方向不同，且至少一项 60→180 达到 error `0.5%` 或 PCC `0.002`，
   形成候选模型×上下文交互。

若所有四个 error 指标的 60→180 绝对相对变化均小于 `0.5%`、PCC 绝对变化小于 `0.002`，且没有一致的单调、
饱和或交互模式，则窗口队列停止。单 seed gate 只决定是否值得增加稳定性证据，不等于最终效果判决。

### 7.2.1 P3 冻结结果

六项训练均从干净 commit `f57583b91f04e5ab5ea9475042489e8aeabb4e9f` 完成，6/6 lifecycle 为
`complete`，每项均有 80 个 epoch、6400 次 optimizer update 与 2675 条完整 validation 指标。所有输入 artifact
manifest 的文件大小与 SHA-256 均通过只读复核；未读取 checkpoint 内容、dataset/index、signal/target 或 test，也未执行
训练、推理或 GPU 计算。

只读汇总入口 `scripts/summarize_paper_center_context_v1.py` 从干净 commit
`a7056cd0be489946530bb3954f3a341b136eecdf` 执行，固定输出位于
`runs/paper_evidence_v1/center_context/p3_validation_summary/`。`summary_receipt.json` SHA-256 为
`894acc31e0fa6219dba83774baa818eba632da5151bdbb5c644b53e00b7e9577`，`artifact_manifest.json` SHA-256 为
`5eed8250b98462e58be94c02d1bdae377c0a4b1838ef90ffc21f3c3cd1fe2a80`。

P3 没有触发单调改善或候选 90 s 饱和条件，但触发了四项候选模型×上下文交互：

- C201 的 60→180 RR 相对变化为恶化 `3.6528%`，W-reduced 为改善 `1.5962%`；
- C201 的 IBI 为改善 `0.2349%`，W-reduced 为恶化 `1.4721%`；
- C201 的 trajectory 为改善 `1.6855%`，W-reduced 为恶化 `15.0124%`；
- C201 的 global-envelope error 为改善 `3.3754%`，W-reduced 为恶化 `12.9560%`。

因此 P3 只支持“窗口长度响应依赖模型表征”的单 seed 方向性诊断，不支持稳定上下文效应、统一更长更好、唯一长度或
唯一模型结论。该升级信号只说明追加稳定性证据有价值；用户随后按第 7.3 节确认 P4 的额外 12 runs 成本。

### 7.3 P4 三 seed 扩展

P4 只有在 P3 完整汇总、实现/数据身份闭合且用户确认追加 12-run 成本后开放。P4 必须完成全部六臂的另外两个 seed，
最终共 18 个 formal-compatible runs。不得只扩展 P3 中数值较好的长度或模型。

用户已于 2026-09-02 确认追加成本。P4 只新增 seeds `20260812/20260813`，固定 12 项配置：

```text
configs/paper_evidence_v1/p4_ccv1_c201_60_seed20260812.yaml
configs/paper_evidence_v1/p4_ccv1_c201_90_seed20260812.yaml
configs/paper_evidence_v1/p4_ccv1_c201_180_seed20260812.yaml
configs/paper_evidence_v1/p4_ccv1_wr_60_seed20260812.yaml
configs/paper_evidence_v1/p4_ccv1_wr_90_seed20260812.yaml
configs/paper_evidence_v1/p4_ccv1_wr_180_seed20260812.yaml
configs/paper_evidence_v1/p4_ccv1_c201_60_seed20260813.yaml
configs/paper_evidence_v1/p4_ccv1_c201_90_seed20260813.yaml
configs/paper_evidence_v1/p4_ccv1_c201_180_seed20260813.yaml
configs/paper_evidence_v1/p4_ccv1_wr_60_seed20260813.yaml
configs/paper_evidence_v1/p4_ccv1_wr_90_seed20260813.yaml
configs/paper_evidence_v1/p4_ccv1_wr_180_seed20260813.yaml
```

P4 gate 固定为 `stage=p4_additional_seeds / execution_gate=p4_formal / 80 epochs / physical batch 128 /
accumulation 1 / logical device cuda:0`，输出根为
`runs/paper_evidence_v1/center_context/p4_additional_seeds/`。其余数据、cache、模型、loss、metric、selector、CUDA
数值开关和训练超参数逐项继承 P3 冻结合同；P4 拒绝 seed `20260811`、其他 seed、其他输出根和 test access。

两张同型号 GPU 各运行六项顺序队列，通过 `CUDA_VISIBLE_DEVICES=0/1` 选择物理卡，进程内均使用逻辑 `cuda:0`：

- 物理 GPU 0：seed 20260812 的 `C201-60 → C201-90 → W-reduced-180`，随后 seed 20260813 的
  `W-reduced-60 → W-reduced-90 → C201-180`；
- 物理 GPU 1：seed 20260812 的 `W-reduced-60 → W-reduced-90 → C201-180`，随后 seed 20260813 的
  `C201-60 → C201-90 → W-reduced-180`。

同一队列用 shell `&&` 串行，任一 run 失败即停止该队列并保留失败 identity。P4 训练仍由用户从统一干净 commit 手动执行。

正式汇总逐 seed 报告，再对三个 seed 报告 arithmetic mean ± sample SD 和 paired-seed 方向；不构造总分或 p-value。
该辅助实验的记录位置按三 seed 结果决定，但任何分支均不改变论文主模型或既有主实验结论：

- 稳定单调收益：可作为独立上下文效应结果进入正文或补充材料；
- 稳定 90 s 饱和：补充材料，正文只简述；
- 接近、非单调或跨 seed 不稳定：作为协议敏感性/负结果留档。

P4 validation 结果不得用于修改 `Pi_60`、loss、metric、selector、长度或模型。若需要独立测试集，只能在 18/18
完成、checkpoint 与汇总锁定后另立 test 附件并获得用户授权；为了检验长度效应，test 矩阵必须完整包含六臂×三 seed，
不得只评价 validation 最好的长度。

### 7.3.1 P4 冻结结果

新增 12 项均从干净 commit `55d515e74adc551a760fdd5914f5c4a1c1ced0a8` 完成，12/12 lifecycle 为
`complete`；与 P3 合并后为 6 arms × 3 seeds 共 18/18 runs。每项均有 80 个 epoch、6400 次 optimizer update、
2675 条完整 validation 指标，所有 artifact manifest 登记文件的大小与 SHA-256 均通过只读复核。

只读三 seed 汇总入口 `scripts/summarize_paper_center_context_p4_v1.py` 从干净 commit
`689344366bdd763932521b8d03aea14d4c8677e6` 执行，固定输出位于
`runs/paper_evidence_v1/center_context/p4_validation_summary/`。`summary_receipt.json` SHA-256 为
`7a8e5a3118e5059fc46ff287f4553952f2fea44653501c7ddedfd09145da9cb9`，`artifact_manifest.json` SHA-256 为
`c0806893d1b4350da46fb9bc02d2056a8b6c8e29d368205709be00aae9bacb2a`。汇总读取 48,150 条 validation metric
rows，未读取 checkpoint 内容、dataset/index、signal/target 或 test，未执行训练、推理或 GPU 计算。

三 seed 的 RR MAE（arithmetic mean ± sample SD）为：

| 模型 | 60 s | 90 s | 180 s |
|---|---:|---:|---:|
| C201-center60 | `0.51457 ± 0.00187` | `0.52245 ± 0.01069` | `0.54020 ± 0.00856` |
| W-reduced-center60 | `0.51104 ± 0.00593` | `0.51775 ± 0.00954` | `0.53214 ± 0.02195` |

paired-seed 方向显示：C201 的 60→180 RR 在 3/3 seeds 恶化，平均相对变化为恶化 `4.9782%`；W-reduced 的
60→180 RR 为 1/3 改善、2/3 恶化，平均相对变化为恶化 `4.1537%`。与此同时，C201 的 90→180 trajectory 与
global-envelope error 均在 3/3 seeds 改善，但其 60→90 两项均在 3/3 seeds 恶化；W-reduced 的 60→90 IBI 与 PCC
在 3/3 seeds 改善，而 90→180 PCC 在 3/3 seeds 恶化。模型间差异同样依赖长度：90 s 时 W-reduced trajectory
在 3/3 seeds 更好，180 s 时 W-reduced global-envelope error 在 3/3 seeds 更差。

协议未预注册把三 seed 结果自动判为“稳定”的精确阈值，因此汇总没有在结果可见后新增 gate。按已冻结的 P3 条件逐 seed
复核，三个 seed 均各自触发升级信号，但没有任何同一单调改善、候选 90 s 饱和或模型×上下文交互条件同时出现在全部
三个 seeds。最终只支持“窗口长度响应具有模型和 seed 依赖、没有简单单调规律”的 validation 描述，不支持稳定统一收益、
唯一窗口长度或唯一模型结论；该辅助窗口任务在 validation 阶段关闭并留档。

## 8. P5：时频条件功能证据

### 8.1 先复用冻结 P−1

W0 已有三个固定 seed、10 项完整 validation 干预：FULL、BETA_ONLY、GAMMA_ONLY、CONDITION_OFF、RESP、
CARRIER、CARRIER_L、CARRIER_H、TIME_MEAN、TIME_SHIFT_30S。该审计和 correction 已关闭，不得重复运行。

当前第一动作只允许只读生成论文表/图，保持 source CSV hash 不变。至少报告 FULL 对比 CONDITION_OFF、TIME_MEAN、
TIME_SHIFT_30S 及四个频率遮挡的五项 primary delta、三 seed 方向和 FiLM corrected statistics。

### 8.2 可选局部频率×时间干预

只有既有 P−1 无法满足论文图件需求时，才另行授权一次新的 validation-only inference protocol。候选网格固定为：

- 频率：`<=0.10 / (0.10,0.80] / (0.80,2.00] / >2.00 Hz`；
- 时间：180 s 内六个不重叠 30 s blocks；
- 干预：对应 W cache cell 在 encoder normalization 前置零；
- checkpoint：W0 三个 validation-selected checkpoint；
- 输出：五项 primary delta 的 `4×6` 网格、逐 seed 结果和 selected-sample FiLM map。

该阶段不训练、不读取 test、不修改 cache、不增加频带或块长搜索。它只定位冻结 W0 对局部条件的功能敏感性，
不参与模型选择或窗口 P3/P4 gate。

## 9. P6：多属性波形与 RR 区间

### 9.1 波形图

优先只使用 validation，避免为示例图重新访问独立测试集。先从冻结逐 sample metrics 和 target-only attributes 中按固定规则
选择 row，再对少量 row 做一次 frozen-checkpoint inference export；禁止浏览波形后手工换样本。

选择规则至少覆盖：典型、RR 困难、努力困难、RR—努力不一致和联合失败。每类使用明确分位点/最近距离规则，并保存
全部候选、最终 row ID、选择原因和 rule hash。导出 BCG、center/whole target、W0 prediction、log-RMS envelope、RR 与
五项指标；不得保存或发布无关身份元数据。

### 9.2 RR 区间

使用完整 admitted train targets，按现行 dominant spectral Whole RR 算法计算 target RR，并以 `linear` quantile 冻结
`1/3`、`2/3` cutpoints。Validation/test 只应用 train-frozen cutpoints，不重新估计。

既有逐 sample prediction metrics 未保存 target RR，因此必须新增 target-only attribute artifact，再按
`dataset_row_id` 一对一连接既有方法结果。连接必须审计 row 集合、重复、缺失、split、target hash 和 subject coverage；
不得重新运行已关闭的模型 test evaluator。

每个低/中/高区间报告窗口数、`samp_id` 数、五项 primary，以及 W0 和预注册关键比较方法的结果。当前默认聚合仍为
sample-direct mean；重叠窗口不解释为独立临床样本。连续 RR 误差曲线只作描述性图，不参与显著性或模型排序。

Test target-only attribute 读取属于新的 test 访问，必须在 train cutpoints、代码、方法 allowlist 和输出 schema 全部冻结后
获得用户明确授权。它不得触发 checkpoint、候选、阈值、频带或方法重选。

## 10. P7：IoT 端到端效率结果

### 10.1 目标与模型

只测量并完整报告实际计算结果，不设置“通过/失败”“可部署/不可部署”质量结论。模型固定为：

- W0：论文主模型；
- W3：6-voice scale-density trade-off；
- D4：四层 local trunk 的 validation-only efficiency trade-off。

D4 没有独立测试集质量结果，效率表必须明确其质量列来自 validation，不得与 W0/W3 的独立测试集质量混写。

### 10.2 平台与场景

至少在当前桌面 GPU 和桌面 CPU 测量 batch=1。若未来有真实边缘设备，再以新增平台行补充，不能用桌面结果替代边缘结果。

每个模型测两个场景：

1. `cached-W`：BCG 180 s waveform 与冻结 W cache 已在 host memory；记录 host→device、model forward 和正式
   180 s `Pi`；
2. `online-W`：BCG waveform 已在 host memory；在线执行当前冻结 97-scale CWT、pool/cache-view、host→device、
   model forward 和正式 180 s `Pi`。

W3 必须按当前实现从完整 97-scale source 取偶数 indices，不能在 benchmark 中改成新的 direct-49-scale extractor。
输入文件读取时间单独测量和报告，不与稳定内存内 pipeline 混成一个不可复现数字。

### 10.3 测量方法

- 固定软件环境、CPU 型号/线程数、GPU 型号、CUDA/cuDNN/PyTorch、dtype 和 checkpoint；
- batch=1，GPU 每段前后显式 synchronize；
- 每个场景至少 20 次 warm-up、100 次 timed iterations、5 个独立 rounds；
- 报告每段和端到端的 median、p95、mean、sample SD、min/max；
- GPU 报告 peak allocated/reserved，CPU 报告进程 RSS 增量；
- 同时报告参数数、权重文件大小和可审计 profiler 能覆盖的 FLOPs/MACs；无法完整覆盖 Mamba op 时必须标记 coverage，
  不得填入猜测值；
- 记录 30 s sliding step 下的 `30s - p95_end_to_end` 数值余量和更新吞吐，但不把正/负余量自动翻译成实时、流式或部署结论；
- 记录完整 180 s 上下文等待这一事实，明确 benchmark 只评价滑窗处理开销。

### 10.4 结果产物

固定新目录：

```text
runs/paper_evidence_v1/p7_iot_efficiency/
```

至少保存：

- `environment.json`
- `stage_latency_iterations.csv`
- `stage_latency_summary.csv`
- `memory_summary.csv`
- `model_resource_summary.csv`
- `quality_efficiency_table.csv`
- `manifest.json`

Manifest 的 `decision` 字段只允许记录 `measurement_complete/incomplete` 和失败原因，不允许预填部署结论。结果产生后再依据
实际数值撰写论文讨论。

## 11. 统一产物与失败规则

所有新增阶段必须：

- 使用干净 Git commit 和独立不可覆盖目录；
- 保存 resolved config、命令、seed、代码 commit、数据/row hash、checkpoint hash、依赖与 access flags；
- 训练保存 best/final checkpoint、完整 history、逐 sample validation metrics 和 summary；
- 非有限 input/target/prediction/checkpoint 立即 fail closed，不静默过滤；
- 失败目录和 lifecycle 保留，不删除、不覆盖、不用相同 identity 静默重跑；
- 只在协议明确开放的 split 上读取数据；
- 不因 partial seed、单个图或 benchmark 中间结果修改剩余矩阵。

## 12. 当前推进点与实现回执

用户已授权 P0/P1 实现和定向 CPU 测试，并将中心任务明确收窄为游离于主实验之外的
`C201-center60 / W-reduced-center60` 上下文敏感性比较。实现回执如下：

1. P0 从干净 commit `936d70813b5619c4eedb8be048806cc7ce3637e3` 完成正式只读审计，固定输出为
   `runs/paper_evidence_v1/p0_comparison_audit/`；manifest SHA-256 为
   `1e05c3de9a75c08ee016af379e075adc1b71b4d1162f7bbf81d716b4942d2542`，decision=`audit_complete`。
2. 12 组来源均为 `compatible`，独立测试集方法的 2310-row `dataset_row_id` 集合哈希一致；10 个预注册方法进入
   primary table，D4 保持 validation-only efficiency trade-off，RTM-v1 五候选等待结论锁。source manifest SHA-256 为
   `c1fcf7d55a25763013a9604d544a81c6306b3752e41cbfba2447a8dbb46f5b82`。本次首次记录各 source artifact 的
   observed SHA-256；因 source manifest 未预填逐 artifact expected SHA-256，`hash_match=true` 仅表示文件存在且未发生
   可选 expected-hash 冲突，不作为先验哈希比对声明。
3. P1 已实现嵌套数据视图、direct 49-scale length-specific input-only cache、C201/W-reduced 变长模型、`Pi_60`、
   中心 loss、五项中心指标、strict selector、不可覆盖训练 lifecycle 与独立入口/config gate。
4. C201/W-reduced 可训练参数分别为 `1,069,802 / 1,219,850`，差值为固定 W branch 的 `150,048`；
   165 个共享 state tensors 在同 seed 下逐 tensor 相同。各模型参数数不随输入长度变化。
5. `ssqueezepy==0.6.6` 的 49-scale 定向 CPU 映射在 60/90/180 s 上观察到实际中心重复数 `1/0/0`，
   最大名义频率绝对误差约 `0.159601 / 0.119571 / 0.077204 Hz`；按第 4.5 节只记录，不删 scale、不改网格。
6. 新增 P0/P1 测试 22 项均通过。P0 仅读取既有冻结结果 CSV，未评价 checkpoint、生成
   prediction 或读取数据集 signal/target；未运行训练、GPU、CPU lifecycle、全量 cache、benchmark 或新的独立测试集访问。
7. P2 从干净 commit `7249fca27e5a960a3e74e43a5acf6ac0abc3a4b5` 完成；六臂 batch-1 全部 finite，
   W-reduced-180 physical batch 128 完成 AdamW step，peak allocated/reserved 为 `10,834.94 / 10,948.00 MiB`，
   decision=`batch128_accepted`。固定 receipt SHA-256 为
   `cb43ee480018afd483ce2bef15342d5bdaf29b6bc999d58ce209abac58df04ce`，manifest SHA-256 为
   `09b9e3347967cd7a4eac45b39c4054c4db689ccbbfe1bdaf2a2e23469957d582`；P2 关闭并冻结统一 `128×1`。
8. 60/90/180 s 三份完整 W cache 均为 `10141 train / 2675 validation`、49 scales、float32、finite、input-only；
   manifest SHA-256 分别为 `e9d270c930d6862f9b5a9cbdb3c26765fe3b57d2573946640cba99e597389793`、
   `9442a33ea2c633b15642d033efa3d3e09278f30c4692a707abc9de460c784757`、
   `72402d8543adf3cda504967b87fc4f9528cacca58b9aa080f0dcfe06707037fe`。P3 六配置冻结为 seed `20260811`，
   formal gate 拒绝额外 seed、输出根、CUDA 数值运行合同或 cache identity 漂移。
9. P3 六项已完成，冻结汇总的 receipt/manifest SHA-256 为
   `894acc31e0fa6219dba83774baa818eba632da5151bdbb5c644b53e00b7e9577 / 5eed8250b98462e58be94c02d1bdae377c0a4b1838ef90ffc21f3c3cd1fe2a80`；
   触发模型×上下文交互，但只形成单 seed 方向性诊断。
10. 用户已确认 P4 追加 12 runs 成本；两个新增 seed 的 12 项显式配置、独立输出根与 `p4_formal` gate 已实现，
    不重跑 P3 seed。
11. P4 新增 12/12 与总计 18/18 runs 已闭合；冻结汇总 receipt/manifest SHA-256 为
    `7a8e5a3118e5059fc46ff287f4553952f2fea44653501c7ddedfd09145da9cb9 / c0806893d1b4350da46fb9bc02d2056a8b6c8e29d368205709be00aae9bacb2a`。
    结果作为非单调、跨 seed 不一致的描述性上下文敏感性证据留档；该阶段当时 47 项 paper-evidence CPU 定向测试通过。

center-60 窗口辅助任务的 validation 阶段已完成并关闭；不追加训练，也不回写论文主模型选择。后续独立测试集确认仅由
P4-T 附件开放，并保持与 validation 选择隔离；该附件现亦已完成并关闭。

P4-S 作为新的独立短窗口分支按
`docs/experiments/paper_center30_context_protocol_20260902.md` 推进；它不重开 center-60 P4。30/45 s input-only W cache、
八项单 seed formal 与只读 validation 汇总均已冻结。两种表征的 45/60/90 s RR 均优于 30 s，但 C201 最低点为
60 s、W-reduced 最低点为 45 s，因此当前只支持“30 s 较差，45–60 s 边界具有表征敏感性”的方向诊断。
用户确认成本后，P4-S3 从干净 commit `2107cf9935229b28224035aa7272515e91be0706` 完成新增 16/16，合计
24/24 runs。三 seed 冻结汇总 receipt/manifest SHA-256 为
`2c533117c735e5bedb65f31ca77fa9dd22663433c0a284b4ee7bd9dc0beee2ef / b18bbed8ab826b8a6e415866c4ad1b9d5171fd4eebc0d5c4090544428fc19bba`。
RR paired-seed 显示 30→45 在 C201/W-reduced 均为 3/3 材料改善，平均改善 `2.4783% / 2.9589%`；45→60
与 60→90 没有跨模型、seed 的稳定追加收益。故冻结为：center-30 输出下 45 s 是 RR 优先的合理输入下限，不能写成
五指标全局最优；该分支在 validation 阶段关闭，不回写论文主模型选择。

用户于 2026-09-04 授权 center-30 与 center-60 两种输出任务的完整独立测试集附件，按
`docs/experiments/paper_context_length_research_test_protocol_20260904.md` 推进。矩阵固定为 center-30 24 个与
center-60 18 个 validation-selected checkpoints，共 42 项；不得按 validation 或 test 数值缩减。P4-T1 先构建
30/45/60/90/180 s 五长度联合、2310-row、input-only W cache，只读取 test BCG，不读取 target；冻结 cache manifest
SHA-256=`9b475926258d129851fb7b9c10d2b342ac2e833b121842b18437585059bf1b98`。P4-T2 从干净 commit
`52d723891e34ad5051f18bdf93f2b0fbe4eb9d7f` 完成 42/42 项，共 97,020 条 test metric rows；全部 lifecycle、
artifact hash、checkpoint/data/runtime/access identity 通过只读复核。

P4-T3 从干净 commit `241d9ba96373dfd8fa3ff1d71782f7a8f9d7e926` 完成分任务冻结汇总，receipt/manifest
SHA-256=`84ac2153f7e30ece4838dfb37645ca5b261a788fbbb6a09f8524b6119100f937 / d69d47a80e514a9556000296938dff642d2642656c61ecff82d57c1a1baeeb66`。
Center-60 中 60→180 RR 平均变化为 C201 `−11.4546%`、W-reduced `−11.7991%`，90→180 在两模型中均
3/3 seeds 方向恶化，支持“更长输入没有稳定 RR 收益”；在该任务、当前模型与 RR 优先口径内，60 s 是合理选择。
Center-30 的 30→45 RR 在 test 中两模型均仅 2/3 seeds 材料改善，未确认 validation 的 45 s 稳定下限；30→90
则两模型均为 3/3 材料改善，说明 30 s 不足但最短边界未定位。两种输出任务始终独立汇总，不比较绝对指标、不重选长度。
新增 summary 测试后 89 项 `tests/test_paper*.py` 定向 CPU 回归全部通过。

本文件仍不授权 Codex 启动任何长时间 CPU/GPU 任务、训练、全量 cache、benchmark 或独立测试集访问。
