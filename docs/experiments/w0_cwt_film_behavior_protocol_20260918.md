# W0 CWT 条件 FiLM 调制行为分析方案

协议 ID：w0-cwt-film-behavior-v1-20260918。

状态：已完成代码、冻结来源、既有产物与质量元数据的只读核对；首轮最小分析设计、独立核心/运行入口、synthetic CPU 定向测试和当前 implementation lock 已经实现。真实 batch-1 smoke、三个 checkpoint 的 validation 推理与正式汇总由用户在实现提交后执行；尚未执行新的真实 checkpoint 推理或统计。本阶段只覆盖 validation 描述性分析。

## 1. 研究问题与边界

本阶段只分析冻结 W0（crd_tf102_w）的三个 validation-selected checkpoint：

1. CWT 条件分支实际改变了多少主干特征；
2. 原始与有效调制参数是否频繁接近 tanh 限幅边缘；
3. 调制行为是否随独立质量元数据变化，并与既有重建误差相关。

本阶段不训练、不修改 checkpoint、不改变模型结构、数据划分、预处理、loss、metrics 或 checkpoint selector；不访问 test。结果只属于冻结联合模型的描述性行为证据，不能单独证明 FiLM 的因果贡献、当前限幅有效、0.5 系数最优或某种质量变化由 FiLM 导致。

若结果用于选择新结构、阈值或限幅系数，后续实验仍只在 train/validation 中设计；固定方案的 test 评价须另立附件并重新授权。

## 2. 独立开发身份

本任务使用独立 worktree：

- worktree：/mnt/disk_code/marques/resp_reconstruction_w0_cwt_film
- branch：codex/w0-cwt-film-analysis
- 起点 commit：46ee918d59c7ce97efd7ddc7b106eff2c58c9381

代码和协议在该 worktree 中开发。历史 checkpoint、cache 与冻结 CSV 继续从主仓库的绝对路径只读使用；正式新产物写入主仓库下独立且不可覆盖的：

    /mnt/disk_code/marques/resp_reconstruction/runs/w0_cwt_film_behavior_v1/

worktree 删除不得影响正式产物。实现锁必须同时记录分析代码 commit、worktree 路径、源仓库根、产物根以及所有上游文件身份。

## 3. 冻结对象与数据条件核对

### 3.1 W0 checkpoint

| Seed | Selected epoch | 冻结 run |
|---|---:|---|
| 20260811 | 13 | runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861 |
| 20260812 | 15 | runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130 |
| 20260813 | 14 | runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006 |

每个 run 使用 checkpoint_best_local_rr.pt。训练来源 commit 为 68b3b85df2f18b3dc8ec59e02ac0780f2be42122，原 run manifest 记录工作树干净。

现有 E4 W0 来源审计已经核对：

- checkpoint 内 config 与 run config 一致；
- 三个 checkpoint 均可由当前 W0 路径 strict-load；
- 当前代码与训练提交构造出的 W0 各有 194 个同名 state tensor，逐 tensor 相等；
- W0 路径仍为六层 local trunk、97 个 CWT 尺度、相同 FiLM 计算；
- 完整模型 trainable parameters 为 1,219,850，W 条件分支为 150,048。

新实现锁仍须绑定并重新核验三个 checkpoint、config、run manifest、metrics.csv 与 metrics_summary.csv 的 size/SHA-256；不得仅依赖本文抄录的路径。

### 3.2 split、预处理与推理合同

冻结 config 的关键事实为：

- dataset_root：/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf
- dataset index：training/dataset_index.csv
- 输入键：bcg_rawish_segment_soft_z_key
- target 键：target_waveform_segment_soft_z_key
- 窗口：180 s、100 Hz、18,000 点
- validation：2675 窗口、7 个 samp_id，sample seed=20260611
- admission：filter_unusable=true，min_hard_valid_ratio=0.8，min_state_alignment_valid_ratio=0.8
- drop_nonfinite_windows=false；非有限输入、target、特征或统计必须显式失败
- 原正式推理 batch=128、BF16 AMP、drop_last=false

CWT cache 固定为：

- 路径：runs/crd_tf_v1/cache/bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0
- transform：ssqueezepy 0.6.6、Morlet mu=13.4、12 voices/octave、reflect padding
- 特征：mean50_log1p_abs
- 单窗口 shape：[97,360]，dtype=float32
- validation W shape：[2675,97,360]
- 实际 mapped frequency：0.03662109375–7.99560546875 Hz
- cache manifest 的 dataset_index_sha256：f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f
- validation row identity：2675 个唯一 dataset_row_id

分析必须沿用原生 data loader、W cache reader、模型 eval 路径和已有逐窗口 metrics.csv。调制—误差关联直接连接同 seed、同 dataset_row_id 的冻结 metrics.csv，不重新定义或替换重建误差。

## 4. 实现核对与符号

以 resp_train/crd/tf_v1_model.py 当前 W0 路径和冻结 state 为准。

主干首先执行：

    Z = model.base.encode_local(x)

其中 Z shape 为 [B,96,1800]，表示 refinement 前的 10 Hz latent。FiLM 位于六层 local trunk 之后、base.refinement 与 decoder 之前。

CWT 条件分支的数据流为：

    W [B,97,360]
      -> [B,1,97,360]
      -> conv_in/GN/SiLU [B,48,97,360]
      -> depthwise/conv_out/SiLU [B,96,97,360]
      -> scale mean [B,96,360]
      -> linear interpolate [B,96,1800]
      -> 3 temporal residual blocks
      -> active fill 96->65->96
      -> final 1x1 projection [B,192,1800]
      -> chunk gamma_raw, beta_raw，各 [B,96,1800]

冻结符号为：

$$
g=0.5\tanh(\gamma_{\mathrm{raw}}),\qquad
b=0.5\tanh(\beta_{\mathrm{raw}}),
$$

$$
Z'=Z\odot(1+g)+b.
$$

g 是增益修正量，实际乘法增益为 1+g。gamma_raw 与 beta_raw 是条件生成头的原始输出。对 W0 而言，g、b 同时随通道和时间变化；它们与 Z 同 shape，不是仅通道或仅窗口级常量。

条件分支中的 Conv 使用 Kaiming normal，GroupNorm weight/bias 初始化为 1/0。final_projection 的 weight 与 bias 均为零初始化，所以新模型初始时 gamma_raw=beta_raw=0，进而 g=b=0；这只描述初始化，不代表已训练 checkpoint 仍接近恒等映射。

## 5. 与既有分析的关系

CRD-TF-W v2 P−1 已在相同三个 W0 checkpoint、完整 validation 上保存 8025 行 corrected FiLM statistics：

- 每窗口 mean/median |g| 与 |b|；
- mean |Delta_t g| 与 mean |Delta_t b|；
- |g|>=0.49 与 |b|>=0.49 的元素比例。

其三 seed sample-direct mean 已知为：

- mean |g|=0.26685546，median |g|=0.28009504；
- mean |b|=0.24302296，median |b|=0.24550465；
- mean |Delta_t g|=0.02261490，mean |Delta_t b|=0.02396674；
- 旧饱和比例 g/b=0.02633766/0.02032499。

P−1 还完成 BETA_ONLY、GAMMA_ONLY、CONDITION_OFF、频带仅保留与时间负对照，已证明冻结输出对条件路径和时间排列存在功能敏感性，但没有记录 Z、相对特征变化、raw 参数分布、正负边缘、多阈值敏感性、通道结构、质量分组或与逐窗口误差的关联。

因此本阶段：

1. 不重跑 P−1 的十项干预；
2. 将 corrected film_statistics_corrected.csv 纳入来源锁；
3. 新推理中的重叠统计必须按 seed/dataset_row_id 与历史表核对；
4. 历史 |g|>=0.49 只作为兼容性锚点，新主报告使用第 7 节的严格多阈值定义。

既有 E1 记录的是尺度重排相对 FULL 的 gamma_raw/beta_raw 配对 MAE；它回答输入操作敏感性，不等同于本阶段对原生 FULL 分布与实际特征变化的分析。

## 6. 非侵入式采集与一致性检查

正式模型必须 model.eval() 并在 torch.inference_mode() 下运行。checkpoint 使用 strict=True 加载。不得修改 tf_v1_model.py 的原生 forward 公式。

分析包装器只在原生 forward 中保存张量引用，所有 detach、类型转换、归约、CPU 复制与绘图统计均在 waveform 输出完成后执行。历史 P−1 曾发现“在 decoder 前做统计”会改变 CUDA 执行/内存路径并产生小幅预测漂移，本阶段必须继承这一约束。

每个 batch 捕获：

- FiLM 前 Z；
- branch 输出 gamma_raw、beta_raw；
- refinement 输入处的实际 Z'；
- 原生 waveform 输出。

实现应优先使用只读 hook，并对 W0 的固定六层路径显式校验捕获位置。不得为方便而复制一套近似 forward。每个 batch 处理完统计后立即释放中间张量，不跨 batch 累积完整 latent。

一致性门槛：

1. synthetic CPU fixture 中，挂 hook 与不挂 hook 的 waveform 必须逐 tensor 相等；
2. 用户执行的真实 batch-1 smoke 保存 native/hooked 最大绝对差；目标为精确相等，任何差异均须记录，超过 1e-6 失败；
3. 捕获的 Z' 与由同 batch 的 Z、g、b 在统计精度下重算的结果进行 closure 检查，保存 max/mean absolute residual；
4. 新重叠 FiLM 统计与历史 corrected 表逐行核对，最大绝对差超过 1e-6 失败；
5. 每 seed 的 dataset_row_id 顺序、2675 行数、samp_id、split 与冻结 metrics.csv 必须完全一致。

若 hook 改变输出、捕获次数不是每 batch 恰好一次、shape 不符或任一关键张量非有限，保留失败 lifecycle 并停止，不生成科学汇总。

## 7. 统计定义

### 7.1 实际调制强度

对每个窗口 i，Frobenius 范数覆盖全部 96 个通道和 1800 个 10 Hz 时间点：

$$
\|A_i\|_F=\sqrt{\sum_{c=1}^{96}\sum_{t=1}^{1800}A_{i,c,t}^2}.
$$

使用实际 materialized 的 Z、g、b 与捕获的 Z'，在 native forward 完成后转 float32，并以 float64 累加平方和：

$$
R_{\mathrm{scale},i}=
\frac{\|g_i\odot Z_i\|_F}{\|Z_i\|_F+\varepsilon},
$$

$$
R_{\mathrm{shift},i}=
\frac{\|b_i\|_F}{\|Z_i\|_F+\varepsilon},
$$

$$
R_{\mathrm{total},i}=
\frac{\|Z'_i-Z_i\|_F}{\|Z_i\|_F+\varepsilon}.
$$

固定 epsilon=1e-12，仅作除零保护。另保存：

- z_fro_norm；
- z_rms=z_fro_norm/sqrt(96*1800)；
- denom_small，定义为 z_rms<=1e-6；
- scale、shift 与 total 的未归一化范数；
- closure residual。

denom_small 窗口不删除；单独标记并在主汇总中同时报告含/不含该标记的计数。R_scale、R_shift、R_total 始终分列，不能仅用 R_total 代表缩放和平移各自作用。

### 7.2 参数分布

每窗口分别对 gamma_raw、beta_raw、g、b 保存：

- signed mean、SD、min、max；
- p01、p05、p25、p50、p75、p95、p99；
- mean/median absolute value；
- 正值、负值和精确零的比例。

全量元素不落盘。分布图使用流式 histogram：

- g、b 固定在 [-0.5,0.5] 上分箱；
- tanh(gamma_raw)、tanh(beta_raw) 固定在 [-1,1] 上分箱；
- raw 参数使用固定 [-8,8] 分箱，并显式记录左右 overflow。

### 7.3 限幅边缘

操作性定义为：

$$
|\tanh(\gamma_{\mathrm{raw}})|>\tau,\qquad
|\tanh(\beta_{\mathrm{raw}})|>\tau,
$$

其中 tau 属于 {0.90,0.95,0.99}。主报告使用 tau=0.95，对应 |g|>0.475 或 |b|>0.475。raw 参数等价边界为：

| tau | atanh(tau) | 有效量边界 |
|---:|---:|---:|
| 0.90 | 1.4722194896 | 0.45 |
| 0.95 | 1.8317808231 | 0.475 |
| 0.99 | 2.6466524124 | 0.495 |

每个阈值对 gamma/beta 分别报告 positive-edge 与 negative-edge 比例，并给出二者之和。严格使用 >；不得与历史 P−1 的 >=0.49 混写。为兼容性核对，另计算 legacy_abs_ge_0p49，但不作为新主结果。

这些阈值只用于描述接近数学限幅的位置，不是公认饱和标准。高边缘比例不能直接推出应放宽约束。

### 7.4 通道与窗口内时间结构

为避免全局平均掩盖结构，保存：

- 每 seed、samp_id、channel 的 mean |g|、mean |b|、三项相对强度和正/负边缘比例；
- 每 seed、samp_id、5 s 相对时间 bin 的上述统计；180 s 窗口固定为 36 bins，每 bin 50 个 latent frames；
- 每窗口的 channel dispersion 与 time-bin dispersion；
- 选定典型窗口的完整 1800 点调制轨迹。

通道级相对强度的分母使用同通道 Z 的时间范数，并单独标记低能量通道；不得用全局分母伪装成通道自身强度。

### 7.5 重建误差关联

误差直接复用冻结 metrics.csv 的五个轴：

- whole_rr_abs_error_bpm；
- local_rr_mae_bpm；
- envelope_trajectory_mae；
- global_envelope_modulation_error；
- 1-lag_aware_signed_pcc。

每个 seed 分别计算调制统计与五个误差轴的 Spearman：

- pooled_windows；
- within_samp；
- samp_means。

三 seed 报告各自估计及 arithmetic mean/sample SD。窗口重叠，pooled 结果不是独立样本推断；不输出窗口级 p-value。首版不默认输出置信区间。若后续需要 CI，只能按 samp_id 整体重采样，并明确 validation 仅 7 个 samp_id 导致区间不稳定。

### 7.6 信号质量

现有 dataset index 提供独立于模型预测的质量元数据。首轮固定：

- primary group：waveform_confidence_level；
- continuous quality axes：waveform_confidence_score、transient_motion_ratio；
- confounding sensitivity：在冻结 envelope_target_stratum 内重复主要关联，并报告 within_samp 结果。

当前 validation 的 waveform_confidence_level 实际只有 medium=1744、high=931，没有 low。不得虚构 low 组或按重建误差补组。transient_motion_ratio 以连续变量为主；若绘图需要分层，其 cutpoints 只能在 prepare-lock 时由质量元数据单独计算并冻结，不能参考模型误差或调制结果。

各质量组报告 R_scale、R_shift、R_total、gamma/beta 正负边缘比例及五项误差。高误差不得定义为低质量。hard_valid_ratio 与 training_finite_ratio 在既有诊断中近乎常量，不作为主质量分组。

## 8. 典型窗口的预定义选择

第一遍完整推理只保存窗口统计，不保存全量 Z/g/b/prediction。三 seed 完成后，先按 dataset_row_id 计算三 seed mean，再固定选择：

1. 以 R_total 中位数和 local_rr_mae_bpm 中位数形成四个象限；
2. 每个象限在 [R_scale,R_shift,R_total,local RR,1-PCC] 的 robust-standardized 空间中，选择离该象限逐维中位点最近的窗口；
3. waveform confidence 的每个实际存在 level 再各选一个同规则 medoid；
4. 重复 dataset_row_id 去重；距离并列时取较小 dataset_row_id。

若某象限没有窗口，case_selection.csv 保留 candidate_count=0 / empty_cell，不从相邻象限强行补选。选择规则、cutpoints、候选数、距离和最终 row identity 写入 case_selection.csv。不得人工替换“看起来不好”的窗口。

第二遍按第一遍固定 batch=128 的原始 row 顺序和 batch 边界重建包含选定窗口的完整 batch，只保存其中选中的少量窗口；不得把案例拼成新的小 batch 而改变 CUDA 数值路径。保存：

- 输入 BCG、参考呼吸；
- 三个 W0 checkpoint 的重建呼吸；
- g、b 的通道均值、通道 RMS、正负边缘比例随时间轨迹；
- g*Z、b、Z'-Z 的通道 RMS 随时间轨迹；
- 对应质量元数据和五项冻结误差。

完整中间张量只允许保存在这些已选窗口的压缩 NPZ 中，并在 manifest 中列出 shape、dtype、size 与 SHA-256。

## 9. 最小实现与产物合同

计划实现路径：

- 核心：resp_train/paper_evidence/w0_cwt_film_behavior.py
- CLI：scripts/analyze_w0_cwt_film_behavior.py
- tests：tests/test_w0_cwt_film_behavior.py
- implementation lock：docs/experiments/w0_cwt_film_behavior_implementation_lock_20260918_r2.json

初版 implementation lock 只完成来源核验，未用于真实 smoke 或 analysis；当前 r2 同步最终命令合同、完成状态和案例原始 batch 上下文复现，并显式登记 supersedes。

每次 attempt 必须排他创建；失败 attempt 原地保留，不覆盖、不复用为完成结果。建议阶段：

- prepare-lock：只读核验上游身份并冻结配置；
- smoke：用户执行的真实 batch-1 非侵入一致性检查；
- analyze：用户按 seed 执行完整 validation 推理与流式统计；
- summarize：CPU 汇总、关联、质量分组与 case selection；
- render-cases：用户仅重推选定窗口；
- finalize：生成图、中文结论、manifest 与 freeze receipt。

最小产物：

1. source_audit.csv、resolved_config.yaml、commands.txt、environment.json、access_receipt.json；
2. window_statistics.csv，共 3*2675=8025 行；
3. subject_summary.csv、channel_summary.csv、time_bin_summary.csv；
4. saturation_threshold_summary.csv、quality_group_summary.csv；
5. modulation_error_associations.csv；
6. distribution_histograms.npz 与绘图数据 CSV；
7. case_selection.csv、selected_window_tensors/*.npz；
8. figures/parameter_distributions.png、saturation_by_sign_threshold.png、modulation_strength_error_association.png、quality_group_comparison.png、selected_windows/*.png；
9. conclusions_zh.md，分为观察结果、机制假设、待验证问题；
10. lifecycle、artifact_manifest.json、freeze_receipt.json。

window_statistics.csv 不保存逐通道逐时间 tensor；只保存窗口级归约、身份、质量元数据与冻结误差。subject/channel/time 汇总必须保留 seed，不得先混合三个模型。

## 10. 计划命令

以下命令为当前已实现接口。Python 环境复用主仓库 .venv，源码来自独立 worktree。prepare-lock 应在实现提交后执行并提交生成的 implementation lock；正式命令要求干净 worktree。

~~~bash
FILM_WT=/mnt/disk_code/marques/resp_reconstruction_w0_cwt_film
FILM_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python

cd "$FILM_WT"

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" -m pytest tests/test_w0_cwt_film_behavior.py -q

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py prepare-lock

# 用户：真实 checkpoint/data 的小规模一致性检查。
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py smoke \
  --seed 20260811 --device cuda:0

# 用户：三个冻结 checkpoint 的完整 validation 分析。
FILM_SMOKE=/完成的smoke_attempt目录
for FILM_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    "$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py analyze \
    --seed "$FILM_SEED" --device cuda:0 \
    --smoke-receipt "$FILM_SMOKE" || break
done

# CPU 汇总并冻结 case selection。
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py summarize \
  --runs /三个完成attempt目录

# 用户：只重推 case_selection.csv 中的少量窗口。
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py render-cases \
  --summary /完成summary目录 --device cuda:0

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py finalize \
  --summary /完成summary目录 --cases /完成case目录
~~~

正式实现时 CLI 不提供 test split、任意 checkpoint、任意阈值或覆盖输出的入口。tau 集合、epsilon、低能量门槛、时间 bin、质量字段和 case selection 规则均来自 implementation lock。

## 11. 验收标准

### 11.1 Codex 可执行

只运行 synthetic/disposable CPU 定向测试：

- shape、通道/时间轴、实际广播和三项范数公式；
- scale/shift 抵消时 R_scale、R_shift 仍分别非零；
- raw/effective 阈值、正负边缘与严格 > 语义；
- denom_small、finite 与低能量通道处理；
- streaming histogram 与直接统计一致；
- hook 捕获次数和无 hook/hook 输出一致；
- 历史 P−1 兼容字段的逐行核对逻辑；
- row identity、seed/samp 汇总、质量组与关联的失败条件；
- case selection 的确定性、去重和 tie-break；
- 排他输出、失败 lifecycle、manifest/hash。

这些测试不得读取真实 checkpoint、波形、W cache 或 test。

### 11.2 用户执行

smoke 验收：

- seed 20260811 checkpoint strict-load；
- 原生与采集 wrapper 的 waveform 差异通过第 6 节门槛；
- Z、raw、g/b、Z' shape 均为 [B,96,1800] 且 finite；
- closure、三项 R、边缘比例 finite；
- 不写历史 run/cache。

完整 validation 验收：

- 三 seed 各 2675 行、7 samp_id、相同 row order，无缺失/重复；
- 与历史 corrected FiLM 重叠字段逐行通过；
- 每 seed 全部窗口统计 finite，denom_small 计数显式；
- metrics join 为一对一，五项冻结误差与 eligibility 不变；
- quality metadata join 为一对一，实际 level/count 登记；
- 全部产物不可覆盖，完成 manifest/hash 闭合；
- 选定窗口二次推理的统计与第一遍对应行在容差内一致。

## 12. 解释与后续决策

结论必须分为：

- 观察结果：在当前三个 W0 checkpoint 和 validation 上直接测得的分布、关联和异质性；
- 机制假设：可能解释现象、但尚未由干预或重训练识别的机制；
- 待验证问题：需要新消融、限幅敏感性或输出敏感性实验回答的问题。

后续建议只依据完整三 seed、多阈值、通道/时间和质量敏感性共同形成：

- shift 持续较弱：可提出 scale-only 重训练消融，不把本阶段描述写成 beta 无效；
- 高频边缘伴随较大误差：可提出 0.5 系数或边界形式的 validation 敏感性实验，不把相关写成限幅致错；
- 调制随质量变化：先检查 subject 与 target-modulation 混杂，再设计性能收益干预；
- R_total 较小：先做冻结模型输出对调制缩放的局部敏感性，不直接判定条件分支无效；
- scale/shift 单项较大而 total 较小：优先研究二者抵消及 decoder 敏感性。

任何新增训练 arm、系数、结构、阈值、test 访问或结论性比较均需另立协议。
