# THO research v2 阶段整理：组会汇报底稿

日期：2026-07-08

更新：2026-07-09 按 `local_rr_*` 口径重评 G 系列测试指标，并同步更新相关表格与解释。

更新：2026-07-10 冻结 BCG 呼吸带二次谐波显著窗口定义，补充四模型在这类已知困难窗口上的离线分层结果。

## 文档定位

本文是面向组会阶段汇报的整理底稿，不是论文正文、完整实验清单或最终结论文件。
它只服务当前 THO research v2 soft-z 主线的阶段判断：

1. 当前项目在解决什么问题。
2. 当前比较方案中哪些证据最重要。
3. 当前 STFT 输入的本阶段基准方案是否已经可以确定。
4. 下一阶段为什么从“继续扩大输入”转向“难恢复窗口的选择性修正”。

本文不改变数据划分、标签、评价指标或既有实验产物。涉及当前评价标准和证据边界时，以
`docs/experiments/metric_schema.md` 与
`docs/experiments/current_evidence_ledger.md` 为准。

## 一句话摘要

本阶段的核心进展是：项目已经从“是否需要时频输入”推进到“确定一个可信的 STFT 输入基准方案”。
当前证据支持以宽频、更高时间分辨率的 STFT 辅助信息作为本阶段基准方案
（内部代号 `G3_C_wide_8p0`）；`G3_C_bandenergy` 保留为后续选择性修正的候选特征，
但暂不替代宽频 STFT。2026-07-09 `local_rr_*` 复评已覆盖当前比较方案，
结论仍支持这一判断。进一步的 THO 参考离线分析显示，四类网络都能在多数 BCG 二次谐波显著窗口中
抑制倍频并恢复目标基频；其中 `G3_C_bandenergy` 的呼吸率和周期计数误差最低，
`G3_C_wide_8p0` 的时延校正后形态更好。这一结果支持保留 bandenergy 作为选择性修正候选，
但不足以改变当前宽频 STFT 基准方案。

## 汇报主线

组会汇报建议采用“问题驱动型”叙事，而不是按脚本或实验编号流水账展开：

```text
BCG 到 THO 呼吸重建为什么难
  -> 为什么要按呼吸任务指标评价
  -> 为什么需要 STFT 辅助信息
  -> 当前比较方案如何比较
  -> 为什么选择宽频 STFT 配置作为本阶段基准方案
  -> 下一步为什么转向难恢复窗口选择性修正
```

推荐汇报重心放在“判断链条”：

- 先说明任务和指标为什么难。
- 再说明当前只比较哪些方案，以及为什么这些方案足够回答当前问题。
- 最后把听众带到当前决策：宽频 STFT 配置作为本阶段基准方案，后续围绕难恢复窗口和不确定窗口做选择性修正。

## 任务动机与问题定义

当前任务可以表述为：

```text
BCG 窗口 -> THO-like 呼吸波形 / 呼吸状态
```

但从信号角度看，BCG 到 THO 并不是简单的波形照抄。BCG 里既有呼吸运动成分，也混有
心冲击、体动、姿态变化、接触条件变化和设备噪声；THO 胸带波形本身也可能出现局部质量下降、
相位延迟、方向不一致、峰谷形态变化等问题。因此，模型即使能恢复整窗呼吸率，也不一定能恢复
局部波形形态；反过来，普通波形拟合误差较低也不一定代表呼吸任务指标更好。

本阶段的研究目标不是追求一个更复杂的网络，而是回答：

- 哪个 STFT 输入方案目前最可信？
- 频带能量特征是否足以替代宽频 STFT？
- 哪些指标能稳定反映呼吸任务，而不是只反映波形拟合误差？
- 下一步应如何从平均指标推进到难恢复样本分析？

## 实验范围与控制变量

当前主线默认指 THO research v2 soft-z 实验设置：

- 数据：`configs/tho_research_v2.yaml` 当前 soft-z 数据设置。
- 输入：BCG soft-z 窗口信号，以及当前比较方案中的 STFT / 分频带辅助信息；这些时频辅助信息只由
  BCG 输入在线计算，不使用 THO 目标信号。
- 目标：THO soft-z 呼吸波形及呼吸任务相关评价。
- 正式比较：固定数据划分、验证窗口、数据采样设置、重复训练次数和模型训练设置。
- 汇报范围：仅整理当前比较方案；不服务当前问题的分支不进入正文、不补测试集指标。

这一点在组会中需要提前说明，否则听众很容易把不同数据设置、不同训练设置和不同评价标准下的结果混在一起比较。

本轮比较的控制变量建议单独讲清楚。组会中可以把“保持不变”和“真正变化”的部分分开：

| 类型 | 本轮保持不变的设置 |
|---|---|
| 数据与划分 | 同一导出数据包、同一 train / val / test 划分、同一受试者隔离关系 |
| 输入预处理 | 同一滤波、窗口长度、采样率、归一化和 soft-z 处理 |
| 输出目标 | 同一 THO-like soft-z 波形目标 |
| 时序主干 | 同一 `patch_mixer1d` 时序主干和同一输出空间 |
| 训练设置 | 同一损失函数、优化器、训练轮数、批量大小、早停设置和重复训练次数 |
| 评价方式 | 同一独立测试集、同一指标计算方式、同一汇总方式 |

真正变化的是时频辅助信息的使用方式：

- `G0_time_only` 不使用时频辅助分支。
- `G0_f0_native_stft_pre_mixer` 使用上一版宽频 STFT，`30s` 窗、`5s` 步长。
- `G3_C_wide_8p0` 使用宽频 STFT，`20s` 窗、`2.5s` 步长。
- `G3_C_bandenergy` 使用频带能量特征，`20s` 窗、`2.5s` 步长。

这里的“频带能量特征”指的是：先对同一段 BCG soft-z 输入计算 `0.05-8Hz` 范围内的 STFT，
再对 `log(1 + |STFT|)` 特征按 5 个重叠频带求均值，得到 5 条随时间变化的频带特征序列。
默认频带为 `0.05-0.3Hz`、`0.1-0.7Hz`、`0.3-1.2Hz`、`0.7-3.0Hz` 和 `3.0-8.0Hz`。
它保留的是几个预设频带的强弱变化，而不是完整的频率-时间图。

这样表述可以避免听众误以为四个方案同时改变了数据、训练目标和评价方式。

## 数据集与预处理

当前报告使用的数据导出包为
`20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf`。
导出包的 manifest 记录数据制作仓库 commit 为 `7ab93efa0d4c8394b0fdf7c57972f75d2a98eec8`；
导出时 `configs/default.yaml` 等文件仍有未提交修改。因此，本节优先依据导出包 `provenance/`
中保存的配置快照和实际导出 metadata，不直接套用当前数据制作仓库的工作区状态。

导出包面向呼吸重建训练与评估，基本数据单位是 180 秒固定窗口；当前训练配置按
`training/dataset_index.csv` 中的 `window_start_s` / `window_end_s` 从整晚 NPZ 中切片。
采样率为 100 Hz，每个窗口 18,000 个采样点。索引中相邻窗口通常间隔 30 秒，因此同一受试者内窗口有重叠。

### 输入和目标信号

组会中建议把当前模型输入称为“宽频 BCG soft-z 输入”；字段名只用于结果追溯。对应关系如下：

- 训练输入字段：`bcg_rawish_wideband_state_aligned_segment_soft_z`。
- 训练目标字段：`tho_waveform_segment_soft_z`。
- 诊断对照字段：`bcg_resp_band_state_aligned`、`tho_waveform_observed`、`tho_event_phase_ref`、`tho_rate_ref` 等。

当前训练脚本不在读取窗口时重新做滤波和归一化，而是读取数据制作阶段已经导出的信号字段。

### 滤波处理

数据制作阶段为 BCG 构建了两类常用信号：

- 宽频 BCG：从 100 Hz BCG 信号出发，使用 4 阶零相位 Butterworth 带通滤波，频段为
  `0.03-20.0 Hz`。当前主线输入来自这条宽频信号的对齐和归一化结果。
- 呼吸带 BCG：同样使用 4 阶零相位 Butterworth 带通滤波，频段为 `0.05-0.7 Hz`。
  当前报告中主要作为诊断或对照信号，不作为 G 系列主线的基础输入。

THO 目标波形来自数据制作阶段导出的 THO 监督信号。事件、相位和呼吸率参考由 THO 呼吸带滤波结果生成；
当前波形回归主线使用的是 `tho_waveform_segment_soft_z`，而不是原始胸带幅值。

### 归一化和软压缩

这里更准确的中文表述是“归一化 / 标准化”，不是训练损失中的正则项。

- BCG 宽频输入先做受试者内状态对齐，再按 `coupling_state_id_sec` 分段估计 robust center / scale，
  得到分段 robust-z 信号。统计池排除状态对齐无效、BCG 不可观测、坏段、硬无效、极端体动、
  状态边界和状态过渡秒点。
- BCG soft-z 在分段 robust-z 之后，对全样本 z 空间做 `log1p_tail` 软压缩，避免极端体动把数值尺度拉得过大。
  宽频 BCG 的压缩参数为 `scale_abs=2.0`、`knee_abs=6.0`、`limit_abs=10.0`、压缩过渡 `2.0s`；
  分段统计要求每段至少 60 秒可靠样本，分段边界平滑过渡为 `5.0s`。
- THO 目标使用数据制作阶段导出的分段 robust-z / soft-z 波形；soft-z 版本在极端体动及其过渡区域做
  z 空间软压缩。THO 侧参数为 `soft_scale_abs=2.0`、`knee_abs=6.0`、`limit_abs=10.0`、
  体动过渡 `5.0s`；THO 的分段归一化按 `tho_amp_segment_id_sec` 估计统计量。

因此，当前模型学习的是“经过分段归一化和软压缩后的 BCG 到 THO 波形映射”，不是从未处理的 BCG
幅值直接回归胸带原始幅值。

### 窗口筛选和受试者分配

导出包原始索引包含 16,603 个窗口、48 名受试者。原始分组为训练集 11,104 个窗口 / 33 名受试者，
验证集 2,955 个窗口 / 7 名受试者，测试集 2,544 个窗口 / 8 名受试者。

当前训练读取器还会筛选可用于 waveform 任务的窗口：`allowed_losses` 必须包含 `waveform`，
`reason` 为空，`hard_valid_ratio >= 0.80`，`state_alignment_valid_ratio >= 0.80`。
经过这一步后，当前报告按以下可用窗口池统计：

| 划分 | 可用窗口数 | 受试者数 | 受试者编号 |
|---|---:|---:|---|
| 训练集 | 10,141 | 32 | 88, 221, 282, 285, 541, 579, 683, 684, 686, 703, 735, 736, 893, 933, 935, 939, 950, 954, 955, 960, 962, 967, 969, 1000, 1004, 1010, 1300, 1302, 1314, 1354, 1374, 1478 |
| 验证集 | 2,675 | 7 | 952, 956, 961, 971, 972, 1308, 1378 |
| 测试集 | 2,310 | 8 | 220, 229, 286, 670, 671, 704, 726, 1006 |

各受试者贡献的可用窗口数并不相同。按当前可用窗口池计入统计的窗口数如下：

| 划分 | 受试者编号 | 计入统计的可用窗口数 |
|---|---:|---:|
| 训练集 | 88 | 567 |
| 训练集 | 221 | 164 |
| 训练集 | 282 | 345 |
| 训练集 | 285 | 683 |
| 训练集 | 541 | 254 |
| 训练集 | 579 | 127 |
| 训练集 | 683 | 93 |
| 训练集 | 684 | 131 |
| 训练集 | 686 | 608 |
| 训练集 | 703 | 84 |
| 训练集 | 735 | 346 |
| 训练集 | 736 | 213 |
| 训练集 | 893 | 136 |
| 训练集 | 933 | 69 |
| 训练集 | 935 | 176 |
| 训练集 | 939 | 350 |
| 训练集 | 950 | 672 |
| 训练集 | 954 | 490 |
| 训练集 | 955 | 275 |
| 训练集 | 960 | 465 |
| 训练集 | 962 | 532 |
| 训练集 | 967 | 433 |
| 训练集 | 969 | 167 |
| 训练集 | 1000 | 359 |
| 训练集 | 1004 | 371 |
| 训练集 | 1010 | 246 |
| 训练集 | 1300 | 708 |
| 训练集 | 1302 | 282 |
| 训练集 | 1314 | 357 |
| 训练集 | 1354 | 66 |
| 训练集 | 1374 | 325 |
| 训练集 | 1478 | 47 |
| 验证集 | 952 | 579 |
| 验证集 | 956 | 213 |
| 验证集 | 961 | 468 |
| 验证集 | 971 | 820 |
| 验证集 | 972 | 182 |
| 验证集 | 1308 | 57 |
| 验证集 | 1378 | 356 |
| 测试集 | 220 | 283 |
| 测试集 | 229 | 298 |
| 测试集 | 286 | 302 |
| 测试集 | 670 | 79 |
| 测试集 | 671 | 504 |
| 测试集 | 704 | 575 |
| 测试集 | 726 | 214 |
| 测试集 | 1006 | 55 |

因此，后文窗口级指标汇总默认是按窗口统计，不是按受试者等权统计；汇报时需要把这一点作为
数据不均衡背景说明。

训练集、验证集和测试集之间没有受试者编号重叠。导出包原始训练分组中包含 `1009`，但经过当前
waveform 可用性筛选后没有进入可用窗口池；因此组会表格建议按上面的可用窗口池展示。

## 指标解释方式

本节只说明“后面为什么这样看结果”，不把“不能只看验证集损失”作为独立研究发现。
本阶段把训练 / 验证损失作为训练过程诊断，不作为最终排序依据；最终判断按呼吸任务指标展开。
评价指标按窗口逐个计算，再对窗口和多次独立训练做汇总。汇报时可以先说明统一记号：

- 预测信号记为 `\hat y(t)`，目标 THO 信号记为 `y(t)`。
- 当前窗口长度为 180 秒，采样率 `f_s=100 Hz`。
- 呼吸频段为 `0.05-0.7 Hz`，对应约 `3-42 bpm`。
- `B(x)` 表示 4 阶零相位 Butterworth 呼吸带滤波后的信号。
- 所有 `*_abs_error` 都是预测值和目标值的绝对差；越低越好。相关系数类指标越高越好。

### 整窗呼吸率

`rr_peak_band_robust_abs_error` 是当前最优先看的整窗呼吸率指标。计算过程是：

1. 先对预测和目标分别做呼吸带滤波：

```text
\hat y_b = B(\hat y),    y_b = B(y)
```

2. 对滤波信号估计一个较稳健的峰值呼吸率。实现上先用 Welch 功率谱找到呼吸带主频，
再把找峰的最小间距和峰突出度设得更保守：

```text
RR_robust(x) = 60 / median(diff(peaks(x)) / f_s)
```

其中 `peaks(x)` 是在呼吸带信号上找到的峰；最小峰间距至少 2 秒，并结合频谱主峰自适应放宽；
峰突出度同时参考信号标准差和 5-95 分位幅度范围。

3. 最后取预测和目标的绝对差：

```text
rr_peak_band_robust_abs_error
  = |RR_robust(\hat y_b) - RR_robust(y_b)|
```

它比普通峰值呼吸率更不容易被弱局部伪峰、局部尖峰或坏段牵着走。组会中可以把它解释为：
“模型是否恢复了整窗主要呼吸节律”。

`rr_spec_abs_error` 是辅助频域指标。它先用 Welch 方法估计呼吸带内归一化功率分布，
取功率最大的频点作为呼吸频率：

```text
RR_spec(x) = 60 * argmax_f P_x(f),    f in [0.05, 0.7]
rr_spec_abs_error = |RR_spec(\hat y) - RR_spec(y)|
```

它能说明频谱主峰是否接近，但如果它改善而周期计数或稳健峰值呼吸率变差，不应视为模型通过。

### 周期数量

`breath_count_zero_cross_abs_error` 判断模型是否恢复了合理的呼吸周期数量。计算过程是：

1. 对预测和目标做呼吸带滤波，得到 `\hat y_b` 和 `y_b`。
2. 分别统计上升过零和下降过零：

```text
N_up   = count(x[i-1] <= 0 and x[i] > 0)
N_down = count(x[i-1] >= 0 and x[i] < 0)
N_cycle = round((N_up + N_down) / 2)
```

3. 取预测和目标周期数的绝对差：

```text
breath_count_zero_cross_abs_error
  = |N_cycle(\hat y_b) - N_cycle(y_b)|
```

这个指标不关心每个峰的细节位置，更像是在问：“180 秒窗口里，模型恢复出的呼吸次数是否差不多？”
它对低质量窗口、低频形态不稳定窗口很有用。

为了和 RR 误差放在同一量级阅读，后文表格把这个周期数误差除以 3，换算为 180 秒窗口的
平均周期率误差：

```text
breath_count_zero_cross_bpm_error
  = breath_count_zero_cross_abs_error / 3
```

这个派生量的单位是 bpm，可理解为整窗平均意义上的周期计数误差；它不等同于 robust RR
或 local RR。

### 低频形态与时延

`band_limited_corr` 是零时延低频相关：

```text
band_limited_corr = corr(B(\hat y), B(y))
```

但 BCG 到 THO 可能有持续时延，所以它只能作为形态诊断，不能单独作为通过标准。

`best_lag_corr_4s` 和 `best_lag_sec_4s` 会在 `[-4s, +4s]` 范围内搜索最佳时延。设
`k` 为采样点级时延，正值表示预测相对目标滞后。对每个 `k`，只在重叠片段上计算相关：

```text
rho(k) = corr(shift(B(\hat y), k), B(y))
k* = argmax_k rho(k),    |k| <= 4 f_s
best_lag_corr_4s = rho(k*)
best_lag_sec_4s = k* / f_s
```

如果多个时延的相关性几乎相同，代码优先选择更接近 0 的时延。组会中建议同时看两个数：
相关性高说明低频形态相似；最佳时延过大则提示模型可能只恢复了节律，但时间位置仍有偏移。

### 相对呼吸强弱

`relative_envelope_corr_lag4s` 和 `relative_envelope_mae_lag4s` 用来判断模型是否跟上了
相对呼吸强弱变化，而不是只输出一个幅度稳定的规整波形。计算过程是：

1. 先用 `best_lag_sec_4s` 对齐预测和目标的重叠片段。
2. 对每个信号计算 2 秒 RMS 包络：

```text
env_x(t) = sqrt(MA_2s(x(t)^2))
```

3. 再用更宽的局部均值作为趋势项，构造相对包络轨迹：

```text
trend_x(t) = MA_long(env_x(t))
r_x(t) = log(env_x(t) / trend_x(t))
```

当前实现中 `MA_long` 使用 20 秒趋势参数的 2 倍平滑窗口，约 40 秒。这里使用对数比例，
是为了消除整体幅度缩放影响，让“增强”和“减弱”在相对尺度上更对称。

4. 最后计算相关性和平均绝对误差：

```text
relative_envelope_corr_lag4s = corr(r_{\hat y}, r_y)
relative_envelope_mae_lag4s  = mean(|r_{\hat y} - r_y|)
```

相关性越高，说明相对强弱起伏越同步；MAE 越低，说明相对强弱幅度越接近。

### 局部呼吸率

`local_rr_mae`、`local_rr_corr` 和 `local_rr_valid_frac` 用于评估局部节律曲线。
2026-07-09 复核后，`local_rr_*` 已切换为当前口径：40 秒滑动窗口、10 秒步长，
并使用更严格的 spectral-guided peak 间距，降低同一呼吸周期内双峰被误计为两个周期的概率。
计算过程是：

1. 先对预测和目标做呼吸带滤波。
2. 用 40 秒滑动窗口、10 秒步长，逐段计算局部稳健峰值呼吸率：

```text
RR_local(x, j) = RR_robust_strict(x_b[t_j : t_j + 40s])
```

3. 只保留预测和目标都能得到有效呼吸率的局部窗口，记为集合 `V`：

```text
local_rr_mae = mean_{j in V} |\hat RR_j - RR_j|
local_rr_corr = corr({\hat RR_j}_{j in V}, {RR_j}_{j in V})
local_rr_valid_frac = |V| / N_local
```

`local_rr_mae` 越低，说明局部 RR 曲线的绝对差更小；`local_rr_corr` 越高，说明局部
快慢变化趋势更同步。它仍不是逐呼吸事件匹配指标：若 `valid_frac` 很低，或 target 侧局部
RR 曲线本身有硬切换、谐波抢峰或低质量片段，仍需回到波形复核。legacy 20 秒 / 5 秒
寻峰口径和 v3 过零探针已退入历史，不再进入当前默认评价输出或汇报排序。

### 汇报时的判断原则

- 训练 / 验证损失只作为训练过程诊断，不直接决定最终方案排序。
- `rr_spec_abs_error` 改善但周期计数或稳健带通呼吸率误差变差时，不应视为通过。
- `band_limited_corr` 是形态诊断，不是单独的通过标准。
- 稳健整窗呼吸率、周期数量、4 秒时延校正后的形态、相对强弱和局部呼吸率要一起看。

## 模型结构概览

第一次汇报中需要补模型结构，但只建议讲到“分支和融合位置”这一层，不展开每层卷积、
通道数和实现细节。当前四组比较方案共享同一类时序主干、同一数据设置、同一输出目标和同一评价方式；
主要差异在于是否加入时频辅助分支，以及这个辅助分支如何表示频率信息。

可以用下面的简化流程说明模型：

```text
180 秒 BCG soft-z 波形
  -> 时序主干 patch_mixer1d
  -> 时序 token 表示

可选辅助分支：
  同一 BCG 波形 -> 在线 STFT / 频带能量 -> 时频 token 表示

融合：
  时频 token 在 pre-mixer 位置与时序 token 融合
  -> 后续时序混合
  -> THO-like soft-z 波形输出
```

因此，组会正文中不需要把 `time_stft_dual1d` 讲成一个复杂模型。更清楚的说法是：

- 所有方案都使用 `patch_mixer1d` 作为时序主干。
- `time_only` 只看 BCG 时序波形，不使用时频辅助分支。
- 其余三个方案在同一时序主干上加入时频辅助信息，并在 `pre_mixer` 位置融合。
- 当前比较不改变输出空间，仍然回归 THO-like soft-z 波形。
- 时频辅助分支只从 BCG 输入生成 STFT 或频带能量；THO 目标只用于训练损失和评价指标，
  不作为推理阶段的输入。

这页的目的不是证明网络结构新，而是让听众明白：本阶段主要在比较“时频信息是否有用、哪种表示更适合作为当前基准”，
不是同时改变模型、目标和评价方式。

## 训练流程与损失设置

第一次汇报还需要交代训练如何组织。建议正文只讲影响结论可信度的设置，
完整字段名和路径放在备份页。

训练、验证和测试的分工如下：

- 训练集用于更新模型参数。
- 验证集用于早停，并通过方向一致性筛选条件过滤明显不合理的训练结果。
- 测试集只在当前比较方案确定之后使用，用于最终复评；本轮没有用测试集调参。
- 每个方案保留 3 次独立训练；测试集结果先逐窗口计算指标，再汇总到单次训练级，
  最后对同一方案的 3 次独立训练取均值。

当前训练损失不是单一 MSE，也不是要求预测波形和 THO 在每个采样点严格重合。它更像一个弱同步约束：
允许 BCG 到 THO 的局部形态和相位存在一定差异，但要求预测结果在呼吸包络、呼吸频带能量分布、
相对呼吸强弱和方向一致性上接近目标信号。

可以用下面的简化式说明当前启用的主要分项：

```text
L = 1.0 L_env
  + 0.2 L_spec
  + 0.1 L_smooth
  + 0.2 L_high
  + 0.03 L_rel_env
  + w_corr(e) L_signed_corr
  + w_cos(e) L_signed_cos
```

其中，`e` 表示训练轮数。更具体地说，各分项可以这样理解：

| 分项 | 当前权重 | 主要计算方式 | 希望约束的性质 |
|---|---:|---|---|
| `L_env` | `1.0` | 对预测和目标分别计算 2 秒 RMS 包络，标准化后最大化相关性 | 呼吸强弱的整体起伏要同步 |
| `L_spec` | `0.2` | 比较 `0.05-0.7Hz` 呼吸频带内的归一化功率分布 | 主呼吸频率和频率分布要接近 |
| `L_smooth` | `0.1` | 计算预测波形相邻采样点差分的平均绝对值 | 抑制不合理的尖锐抖动 |
| `L_high` | `0.2` | 计算预测中高于 `0.7Hz` 的相对频谱能量 | 防止模型把非呼吸高频成分写进输出 |
| `L_rel_env` | `0.03` | 用 2 秒包络减去约 20 秒趋势，比较相对强弱曲线 | 保留局部呼吸增强或减弱的变化 |
| `L_signed_corr` | `0.6 -> 0.2` | 对 `0.05-0.7Hz` 带限波形标准化后最大化相关性 | 约束低频呼吸形态和方向一致性 |
| `L_signed_cos` | `0.1 -> 0` | 对中心化后的带限波形计算余弦相似度 | 训练早期辅助纠正整体方向 |

如果写成更接近实现的形式，可以把 `B(x)` 理解为 `0.05-0.7Hz` 带限波形，`Z(x)` 表示标准化，
`E_2(x)` 表示 2 秒 RMS 包络，`T_20(x)` 表示约 20 秒趋势，`P_B(x)` 表示呼吸频带内归一化功率分布：

```text
L_env         = 1 - corr(Z(E_2(y_hat)), Z(E_2(y)))
L_spec        = sum_f |P_B(y_hat)_f - P_B(y)_f|
L_smooth      = mean_t |y_hat_t - y_hat_{t-1}|
L_high        = power(y_hat, f > 0.7Hz) / power(y_hat, all f)
R(x)           = center(log(E_2(x)) - log(T_20(E_2(x))))
L_rel_env     = mean_t |R(y_hat)_t - R(y)_t|
L_signed_corr = 1 - corr(Z(B(y_hat)), Z(B(y)))
L_signed_cos  = 1 - cosine(center(B(y_hat)), center(B(y)))
```

这里的重点是：训练目标没有把“逐点波形误差最小”作为唯一目标，而是把呼吸任务真正关心的
节律、低频形态、相对强弱和方向一致性都放进了训练约束里。

`L_signed_corr` 和 `L_signed_cos` 用于缓解方向不一致问题：`signed_corr` 在第 1-6 轮训练
从 `0.6` 线性降到 `0.2`，之后保持 `0.2`；`signed_cos` 在第 1-6 轮训练从 `0.1`
线性降到 `0`。这里的 warmup 指的是训练早期对方向一致性的额外约束逐步减弱，而不是学习率 warmup。

当前 G 系列比较没有启用基于目标信号的 STFT 损失：STFT 距离、频带能量和峰值相关分项的权重均为 `0`。
因此，STFT 在本阶段是输入侧辅助信息，不是额外的输出监督目标。
同样关闭的还有相位对齐损失、带限波形 Smooth L1、曲率损失、节律自相关分布损失、SI-SDR 以及
F-B 辅助 STFT 分支相关损失；这些字段保留在配置中，主要是为了方便后续实验复用，不代表本轮已经启用。

正式训练配置在各 run 的 `config.yaml` 中已保存。当前四类比较方案使用相同训练设置：

| 设置 | 当前值 | 汇报时的含义 |
|---|---:|---|
| 最大训练轮数 | `50` | 给模型足够训练时间，由早停决定是否提前停止 |
| 批量大小 | `128` | 正式训练批量大小 |
| 优化器 | Adam | 学习率来自 `training.learning_rate` |
| 初始学习率 | `1e-3` | 本轮没有使用学习率 warmup |
| 学习率调度 | `none` | 训练中不使用学习率调度器 |
| 早停 | `patience=8`, `min_delta=0.001` | 验证损失连续无明显改善时停止 |
| 方向一致性筛选 | `auto_direction`, `max=0.5` | 避免方向明显不一致的训练结果进入后续复评 |
| AMP / 梯度裁剪 | `use_amp=false`, `grad_clip_norm=null` | 本轮未启用混合精度和梯度裁剪 |
| 重复训练 | `3` 次独立训练 | 降低单次训练偶然波动对阶段判断的影响 |

## 当前比较方案

本阶段正文只围绕四类方案展开。第一次汇报时建议先讲每个方案回答的问题，
实验代号只作为结果追溯信息保留在表格中。

| 方案 | 作用 | 为什么保留 |
|---|---|---|
| `G0_time_only` | 纯时序参照 | 判断只看 BCG 时序波形时，模型能恢复到什么程度 |
| `G0_f0_native_stft_pre_mixer` | 上一版 STFT 参照 | 判断时频辅助信息是否已经带来额外信息 |
| `G3_C_wide_8p0` | 宽频 STFT 方案 | 代表宽频、更高时间分辨率的 STFT 辅助信息 |
| `G3_C_bandenergy` | 频带能量特征方案 | 判断更简单的频带能量特征能否替代宽频 STFT，或只适合作为选择性修正的候选特征 |

这组方案足以回答组会阶段的主问题：

```text
时频辅助信息是否仍优于纯时序方案？
更高时间分辨率的宽频 STFT 是否优于上一版 STFT 参照？
频带能量特征是替代方案，还是选择性修正的候选特征？
下一阶段是否该围绕选择性修正继续？
```

四个方案的结构差异可以这样讲：

| 方案 | 时序主干 | 时频辅助分支 | 时频表示 | 时间设置 | 融合位置 | 回答的问题 |
|---|---|---|---|---|---|---|
| `G0_time_only` | `patch_mixer1d` | 无 | 无 | 无 | 无 | 只看 BCG 时序波形能恢复到什么程度 |
| `G0_f0_native_stft_pre_mixer` | `patch_mixer1d` | 有 | 宽频 STFT，`conv2d` 编码，`0.05-8Hz` | `30s` 窗，`5s` 步长 | `pre_mixer` | 加入上一版 STFT 后是否优于纯时序参照 |
| `G3_C_wide_8p0` | `patch_mixer1d` | 有 | 宽频 STFT，`conv2d` 编码，`0.05-8Hz` | `20s` 窗，`2.5s` 步长 | `pre_mixer` | 更高时间分辨率的宽频 STFT 是否更适合作为当前基准 |
| `G3_C_bandenergy` | `patch_mixer1d` | 有 | 重叠频带能量特征 | `20s` 窗，`2.5s` 步长 | `pre_mixer` | 更简单的频带能量特征能否替代宽频 STFT |

其中 `G3_C_bandenergy` 使用的是重叠频带能量序列，而不是完整频点图。它压缩了频率细节，
但保留了慢呼吸、常规呼吸主带、谐波 / 波形尖锐度、偏快呼吸和高频上下文等频带强弱变化。
因此它适合回答“频带能量特征是否足够”，不适合直接和宽频 STFT 混成同一种输入。

## 关键证据 1：时频信息能补充纯时序模型对呼吸节律的判断

四类比较方案的目的不是让听众记住实验编号，而是回答一个基础问题：
BCG 的原始时序波形之外，频率结构是否还能提供额外的呼吸信息。

从纯时序参照到 STFT 辅助信息，呼吸率、低频形态和局部呼吸率指标整体有改善。
这说明 STFT 的作用不是替代时序主干，而是帮助模型识别呼吸节律结构，尤其是在波形形态不稳定、
局部节律容易混淆的窗口中提供额外依据。

阶段判断：

> STFT 辅助信息值得保留；当前问题已经从“是否需要时频信息”转向“哪一种时频表示更适合作为当前基准”。

## 关键证据 2：宽频、更高时间分辨率的 STFT 更适合作为当前基准配置

G 系列回答两个问题：

1. STFT 输入的时间分辨率是否应该从原来的 `30s / 5s hop` 调整。
2. `0.05-8Hz` 宽频输入是否仍优于压缩后的频带能量特征。

当前比较方案中最重要的两个方案：

| 方案 | 本阶段定位 | 汇报解释 |
|---|---|---|
| `G3_C_wide_8p0` | 本阶段基准方案 | 宽频 STFT，时间分辨率高于上一版参照；稳健带通呼吸率误差和时延校正后形态综合表现更稳定 |
| `G3_C_bandenergy` | 选择性修正候选特征 | 频带能量特征，周期计数 bpm 误差和局部 RR 略优，但稳健带通 RR 和时延校正后形态不如宽频 STFT |

2026-07-09 按 `local_rr_*` 口径重评后的独立测试集结果如下。该批结果来自
`runs/test_eval_g_series_20260709_local_rr_canonical`。指标先在测试窗口上逐个计算，再汇总到单次训练级；
表中数值为每个方案 3 次独立训练的均值 ± 标准差。

| 方案 | 稳健带通 RR 误差 | 周期计数 bpm 误差 | 4 秒时延校正相关 | 局部 RR MAE | 局部 RR 相关 |
|---|---:|---:|---:|---:|---:|
| `g0_time_only` | 0.773193 ± 0.017278 | 0.778259 ± 0.081422 | 0.865265 ± 0.002625 | 0.786678 ± 0.015852 | 0.663619 ± 0.008084 |
| `g0_f0_native_stft_pre_mixer` | 0.733281 ± 0.006125 | 0.783405 ± 0.026986 | 0.868751 ± 0.003796 | 0.791792 ± 0.013671 | 0.660398 ± 0.003614 |
| `g3_c_wide_8p0` | 0.712186 ± 0.023403 | 0.716739 ± 0.025005 | 0.870774 ± 0.003077 | 0.780002 ± 0.021617 | 0.664304 ± 0.005835 |
| `g3_c_bandenergy` | 0.729792 ± 0.022269 | 0.684800 ± 0.057641 | 0.867238 ± 0.001328 | 0.773299 ± 0.017622 | 0.665137 ± 0.009487 |

相对 `g0_time_only`，`g3_c_wide_8p0` 同时改善稳健带通呼吸率误差、周期计数 bpm 误差、
时延校正后形态；局部 RR 也略有改善，但幅度明显小于旧 legacy 口径下的差异。
`g3_c_bandenergy` 在周期计数 bpm 误差和局部 RR 上略优，但整体稳定性不如宽频方案。因此当前推荐：

- 本阶段基准方案：宽频、更高时间分辨率的 STFT 方案（`G3_C_wide_8p0`）。
- 选择性修正候选特征：`G3_C_bandenergy`，仅在后续做选择性 / 条件融合时使用。
- 不建议动作：为不进入当前问题的分支补测试集指标。

## 关键证据 3：频带能量特征提示了局部改进方向，但还不足以替代宽频 STFT

`G3_C_bandenergy` 的结果说明，频带能量特征不是没有价值；它在周期计数 bpm 误差和局部 RR 上略优，
提示这类频带特征可能对一部分窗口有帮助。但它在稳健带通 RR 和时延校正后形态上不如宽频 STFT 稳定，
因此目前还不足以作为当前基准输入配置。

因此，第一次汇报里建议把它讲成下一阶段选择性修正的候选特征：

```text
频带能量特征不是当前基准方案，而是后续做难恢复窗口选择性修正时值得保留的候选特征。
```

## 关键证据 4：分层结果揭示选择性修正的潜在收益

整体均值只能说明当前方案的平均表现，不能回答“收益来自哪些窗口”。因此在 2026-07-09
测试集复评结果上，又基于逐窗口 metrics 做了离线分层分析。该分析不重新推理、
不重评 checkpoint、不覆盖历史 run；它按 `dataset_row_id` 对齐各方案的可用窗口，直接统计
每个分层中各模型的原始指标均值，不以方案间 delta 作为汇报结果。分层汇总输出在：

```text
runs/test_eval_g_series_20260709_local_rr_canonical/stratified_analysis_20260709/
```

### Baseline hard / easy 与低谱相似窗口

为了直接横向比较，这里的 baseline 指用于定义窗口难度的参考模型 `g0_time_only`，
不是最终推荐方案。`baseline hard` 指 `g0_time_only` 在该窗口上的
`rr_peak_band_robust_abs_error > 1.0 bpm`；`baseline easy` 指同一指标
`<= 0.25 bpm`。同一分层内各模型对应同一批窗口。`low spectrum` 指
`g0_time_only` 输出与 target 的 `spectrum_similarity` 不高于该次结果中的中位数。三种分层
都使用了目标侧评价信息，因此只用于回顾性理解收益出现的位置，不能直接作为推理时的窗口选择条件。

| 分层 | 模型 | 窗口数 | robust RR | count bpm | lag4 corr | local RR MAE |
|---|---|---:|---:|---:|---:|---:|
| overall | `g0_time_only` | 2310 | 0.7732 | 0.7783 | 0.8653 | 0.7867 |
|  | `g0_f0_native_stft_pre_mixer` | 2310 | 0.7333 | 0.7834 | 0.8688 | 0.7918 |
|  | `g3_c_wide_8p0` | 2310 | 0.7122 | 0.7167 | 0.8708 | 0.7800 |
|  | `g3_c_bandenergy` | 2310 | 0.7298 | 0.6848 | 0.8672 | 0.7733 |
| baseline hard | `g0_time_only` | 308 | 4.3790 | 1.8908 | 0.7375 | 2.5755 |
|  | `g0_f0_native_stft_pre_mixer` | 308 | 3.8786 | 1.8884 | 0.7425 | 2.5792 |
|  | `g3_c_wide_8p0` | 308 | 3.8079 | 1.8448 | 0.7407 | 2.5697 |
|  | `g3_c_bandenergy` | 308 | 3.9029 | 1.8362 | 0.7463 | 2.5723 |
| baseline easy | `g0_time_only` | 1396 | 0.0968 | 0.4158 | 0.9042 | 0.3528 |
|  | `g0_f0_native_stft_pre_mixer` | 1396 | 0.1420 | 0.4167 | 0.9077 | 0.3539 |
|  | `g3_c_wide_8p0` | 1396 | 0.1380 | 0.3563 | 0.9104 | 0.3523 |
|  | `g3_c_bandenergy` | 1396 | 0.1437 | 0.3452 | 0.9051 | 0.3526 |
| low spectrum | `g0_time_only` | 1155 | 1.3275 | 1.0742 | 0.8094 | 1.2631 |
|  | `g0_f0_native_stft_pre_mixer` | 1155 | 1.2757 | 1.0914 | 0.8124 | 1.2733 |
|  | `g3_c_wide_8p0` | 1155 | 1.2362 | 1.0369 | 0.8142 | 1.2552 |
|  | `g3_c_bandenergy` | 1155 | 1.2588 | 0.9873 | 0.8127 | 1.2458 |

这张表的核心读法是：`G3_C_wide_8p0` 的整体 robust RR 最低，从 `g0_time_only` 的
`0.7732` 降到 `0.7122`；收益主要来自 baseline hard 和 low-spectrum 窗口。baseline hard
中，wide 的 robust RR 是 `3.8079`，优于 F0 的 `3.8786` 和 bandenergy 的 `3.9029`；
low-spectrum 中，wide 也是 robust RR 最低的方案。bandenergy 的价值更偏 count bpm 和 local：
整体 count bpm 为 `0.6848`、local RR MAE 为 `0.7733`，baseline hard 的 count bpm 也最低。
但 baseline easy 中，所有时频方案的 robust RR 都高于 `g0_time_only`。这提示后续若设计
选择性修正，easy 窗口应被视为需要保护的对象；这一点仍需用不依赖目标信息的判据另行验证。

### 周期计数失败窗口

周期计数分层用 `g0_time_only` 的 `breath_count_zero_cross_abs_error` 定义。
`baseline count-error > 0` 指参考模型已经数错呼吸周期数；`baseline count-error = 0`
指参考模型周期数完全正确，也就是 clean count 窗口。这个分层提示后续如果做选择性修正，
应优先考虑保护 clean count 窗口，并验证 count-error 或 ambiguous 窗口是否更适合开放更强修正。

| 分层 | 模型 | 窗口数 | robust RR | count bpm | local RR MAE |
|---|---|---:|---:|---:|---:|
| baseline count-error > 0 | `g0_time_only` | 1259 | 1.2386 | 1.4277 | 1.1846 |
|  | `g0_f0_native_stft_pre_mixer` | 1259 | 1.1800 | 1.3877 | 1.1911 |
|  | `g3_c_wide_8p0` | 1259 | 1.1453 | 1.2598 | 1.1711 |
|  | `g3_c_bandenergy` | 1259 | 1.1705 | 1.2008 | 1.1565 |
| baseline count-error = 0 | `g0_time_only` | 1051 | 0.2157 | 0.0000 | 0.3104 |
|  | `g0_f0_native_stft_pre_mixer` | 1051 | 0.1978 | 0.0593 | 0.3133 |
|  | `g3_c_wide_8p0` | 1051 | 0.1927 | 0.0656 | 0.3116 |
|  | `g3_c_bandenergy` | 1051 | 0.2019 | 0.0677 | 0.3144 |

在已有 count-error 的窗口中，`G3_C_wide_8p0` 的 robust RR 最低，为 `1.1453`；
`G3_C_bandenergy` 的 count bpm 和 local RR MAE 最低，分别为 `1.2008` 和 `1.1565`。
这提示 bandenergy 对“周期数修正”和局部 RR 长尾可能有价值，但不一定带来最好的整窗 RR。
在 baseline count-error 为 0 的 clean 窗口中，`g0_time_only` 的 count bpm 本来就是
`0.0000`，所有时频方案都会引入非零 count bpm 误差。因此 count-error 窗口适合作为修正入口，
clean count 窗口需要保护。count-error 窗口能否作为实际修正入口，仍需用不依赖目标信息的判据验证。

### RR 分层

RR 分层不只看 fast RR，而是按 target 的整窗呼吸率分成慢、中、偏快几档。脚本优先使用
`target_rr_peak_band_robust_bpm`，缺失时才回退到旧 RR 列。当前测试集中没有 `rr_ge24`
档样本，因此本轮只解释 `<10`、`10-14`、`14-18` 和 `18-24 bpm`。

| RR 档 | 模型 | 窗口数 | robust RR | count bpm | rel env corr | local RR MAE |
|---|---|---:|---:|---:|---:|---:|
| `<10 bpm` | `g0_time_only` | 230 | 4.2577 | 1.4425 | 0.5889 | 2.1106 |
|  | `g0_f0_native_stft_pre_mixer` | 230 | 3.8698 | 1.4324 | 0.6179 | 2.1675 |
|  | `g3_c_wide_8p0` | 230 | 3.8450 | 1.3850 | 0.6140 | 2.1636 |
|  | `g3_c_bandenergy` | 230 | 4.0787 | 1.4411 | 0.6066 | 2.2515 |
| `10-14 bpm` | `g0_time_only` | 1339 | 0.4070 | 0.8805 | 0.6730 | 0.5367 |
|  | `g0_f0_native_stft_pre_mixer` | 1339 | 0.3821 | 0.8841 | 0.6939 | 0.5436 |
|  | `g3_c_wide_8p0` | 1339 | 0.3691 | 0.7823 | 0.6981 | 0.5209 |
|  | `g3_c_bandenergy` | 1339 | 0.3600 | 0.7222 | 0.6850 | 0.4964 |
| `14-18 bpm` | `g0_time_only` | 545 | 0.2761 | 0.3539 | 0.4336 | 0.6293 |
|  | `g0_f0_native_stft_pre_mixer` | 545 | 0.3024 | 0.3725 | 0.4369 | 0.6194 |
|  | `g3_c_wide_8p0` | 545 | 0.2635 | 0.3649 | 0.4352 | 0.6228 |
|  | `g3_c_bandenergy` | 545 | 0.2697 | 0.3537 | 0.4400 | 0.6205 |
| `18-24 bpm` | `g0_time_only` | 196 | 0.5684 | 0.4802 | 0.5359 | 1.3785 |
|  | `g0_f0_native_stft_pre_mixer` | 196 | 0.6497 | 0.4768 | 0.5280 | 1.3519 |
|  | `g3_c_wide_8p0` | 196 | 0.6275 | 0.4626 | 0.5280 | 1.3637 |
|  | `g3_c_bandenergy` | 196 | 0.6058 | 0.4626 | 0.5461 | 1.3552 |

RR 分层的直接横向结论更清楚：`G3_C_wide_8p0` 在 `<10 bpm` 和 `14-18 bpm` 的
robust RR 最低，说明宽频 STFT 对慢 RR 和中等 RR 更稳；但 `10-14 bpm` 和
`18-24 bpm` 中，`G3_C_bandenergy` 的 robust RR 更低，且 local RR MAE 也最低。
不过 bandenergy 在 `<10 bpm` 明显失败，robust RR 为 `4.0787`，count bpm 为 `1.4411`，
都接近或劣于 `g0_time_only`。因此 RR 分层提示 bandenergy 更适合探索条件使用，而不是统一替换；
其具体选择条件仍是下一阶段需要独立验证的问题。

## 关键证据 5：BCG 二次谐波显著窗口大多可以恢复，但不是时频分支独有能力

前面的 hard / easy 分层从模型误差出发，不能直接说明输入 BCG 为什么难。为了更贴近“原始 BCG
呼吸节律不对，但其中双峰或倍频结构仍可能被网络恢复”的问题，本轮又建立了一个 THO 参考的离线
先验分层。该分析只回答“模型在这类已知困难窗口上表现如何”，不参与训练、不重新选择 checkpoint，
也不构造推理时 gate。

### 冻结规则与覆盖率

分析先对 BCG 和 THO 的 `0.05-0.7Hz` 低频分量计算频谱。THO 的稳健呼吸率与频谱呼吸率相差
不超过 `1 bpm`，且 `2*f_THO` 仍位于分析频带内时，窗口才进入可判定集合。随后只在完整验证集
的 2675 个窗口、7 名受试者上生成并人工复核候选，冻结 `candidate_040`：

- BCG 主峰与 `2*f_THO` 的相对偏差不超过 `10%`，记为主峰倍频证据。
- 二倍频 / 基频邻域能量比不低于 `0.5393`，且二倍频占全分析带能量不低于 `0.1647`，
  记为谐波能量显著证据。
- 两类证据都成立为 `strong_harmonic`；只有主峰证据为 `peak_doubling`；只有能量证据为
  `harmonic_prominent`。三者并集称为 BCG 呼吸带二次谐波显著窗口。

冻结后一次性应用到完整独立测试集，覆盖率如下。不可判定窗口被排除，但单独报告覆盖率；
`占可判定窗口` 只对可判定集合计算。

| 状态 / 分层 | 窗口数 | 占全部窗口 | 占可判定窗口 | 受试者数 |
|---|---:|---:|---:|---:|
| 全部测试窗口 | 2310 | 100.00% | — | 8 |
| THO 参考不稳定 | 373 | 16.15% | — | 8 |
| THO 二倍频超出 `0.7Hz` | 13 | 0.56% | — | 3 |
| 可判定窗口 | 1924 | 83.29% | 100.00% | 8 |
| `strong_harmonic` | 214 | 9.26% | 11.12% | 5 |
| `peak_doubling` | 5 | 0.22% | 0.26% | 2 |
| `harmonic_prominent` | 233 | 10.09% | 12.11% | 5 |
| 二次谐波显著窗口并集 | 452 | 19.57% | 23.49% | 6 |
| `harmonic_negative` | 1472 | 63.72% | 76.51% | 8 |

`peak_doubling` 只有 5 个窗口，不能单独形成稳定结论；本轮的主要证据来自
`strong_harmonic` 和 `harmonic_prominent`，两层在汇总中保持分开。

### 四模型在固定阳性窗口上的任务表现

下表只统计同一批 452 个二次谐波显著窗口。数值仍是逐窗口计算后先汇总到单次训练级，
再对 3 次独立训练取均值。谐波纠正定义为：模型输出主峰回到 THO 基频的 `10%` 容差内，
且二倍频 / 基频能量比相对输入下降至少 `20%`。

| 方案 | 稳健带通 RR 误差 | 周期计数 bpm 误差 | 4 秒时延校正相关 | 相对包络相关 | 局部 RR MAE | 谐波纠正率 |
|---|---:|---:|---:|---:|---:|---:|
| `g0_time_only` | 0.6343 | 2.0973 | 0.8252 | 0.6018 | 0.6520 | 95.94% |
| `g0_f0_native_stft_pre_mixer` | 0.5984 | 2.1013 | 0.8353 | 0.6219 | 0.6743 | 95.94% |
| `g3_c_wide_8p0` | 0.5192 | 1.7652 | **0.8475** | **0.6525** | 0.5897 | 96.24% |
| `g3_c_bandenergy` | **0.4734** | **1.5541** | 0.8278 | 0.5994 | **0.5455** | **97.20%** |

在定义更严格的 214 个 `strong_harmonic` 窗口中，方向更加清楚：

| 方案 | 稳健带通 RR 误差 | 周期计数 bpm 误差 | 4 秒时延校正相关 | 相对包络相关 | 局部 RR MAE | 谐波纠正率 |
|---|---:|---:|---:|---:|---:|---:|
| `g0_time_only` | 0.6576 | 3.4278 | 0.7997 | 0.6136 | 0.8616 | 94.86% |
| `g0_f0_native_stft_pre_mixer` | 0.6984 | 3.4361 | 0.8132 | 0.6363 | 0.9166 | 95.02% |
| `g3_c_wide_8p0` | 0.5543 | 2.8769 | **0.8318** | **0.6800** | 0.7430 | 95.79% |
| `g3_c_bandenergy` | **0.4716** | **2.4564** | 0.8039 | 0.6077 | **0.6552** | **97.82%** |

作为参照，1472 个 `harmonic_negative` 窗口上的五项任务指标如下：

| 方案 | 稳健带通 RR 误差 | 周期计数 bpm 误差 | 4 秒时延校正相关 | 相对包络相关 | 局部 RR MAE |
|---|---:|---:|---:|---:|---:|
| `g0_time_only` | 0.4082 | **0.3311** | 0.8982 | 0.5996 | 0.5613 |
| `g0_f0_native_stft_pre_mixer` | 0.3826 | 0.3374 | **0.8999** | **0.6126** | **0.5574** |
| `g3_c_wide_8p0` | **0.3752** | 0.3371 | 0.8992 | 0.6063 | 0.5618 |
| `g3_c_bandenergy` | 0.3912 | 0.3459 | 0.8991 | 0.6101 | 0.5584 |

阳性并集的纠正状态完整分布为：

| 方案 | 已纠正 | 部分纠正 | 未纠正 |
|---|---:|---:|---:|
| `g0_time_only` | 95.94% | 2.88% | 1.18% |
| `g0_f0_native_stft_pre_mixer` | 95.94% | 3.02% | 1.03% |
| `g3_c_wide_8p0` | 96.24% | 2.88% | 0.89% |
| `g3_c_bandenergy` | 97.20% | 2.14% | 0.66% |

这组结果有四层含义：

1. 四类网络的谐波纠正率都在约 `95%` 或以上，说明从 BCG 倍频恢复 THO 基频主要是当前网络和
   监督目标共同学到的能力，不是 STFT 分支独有能力。纯时序模型也能解决其中大多数窗口。
2. 高纠正率不等于呼吸任务已经完全解决。尤其在 `strong_harmonic` 层，纯时序和 F0 的周期计数
   误差仍超过 `3.4 bpm`。因此必须同时看 RR、周期计数、局部 RR 和形态指标，不能只报纠正状态。
3. `G3_C_bandenergy` 在阳性并集和 strong 层的稳健 RR、周期计数与局部 RR 均最低；
   `G3_C_wide_8p0` 的时延校正后形态相关最高。这与整体测试集上的角色分工一致：bandenergy
   更像选择性修正候选，wide 仍是综合表现更稳定的本阶段基准方案。
4. 在 harmonic-negative 参照层中，四模型差异明显收窄，bandenergy 也不再占优；这进一步说明
   它的价值更可能集中在特定困难形态，而不是全窗口统一替换 wide。

### 代表案例与证据边界

案例图按预先定义的类别平衡选择，而不是只挑成功样本：`dataset_row_id=640` 为四模型均纠正，
`873` 为四模型均未纠正，`1353` 为模型间分歧，`3584` 为阈值附近案例。案例清单和图位于：

```text
runs/bcg_second_harmonic_20260710/figures/model_case_manifest.csv
runs/bcg_second_harmonic_20260710/figures/
```

代表性全模型纠正案例：

![BCG 二次谐波显著窗口中四模型均恢复目标基频的案例](../../runs/bcg_second_harmonic_20260710/figures/all_corrected_seed_20260837_row_640.png)

统计边界也很明显：452 个阳性窗口中有 285 个来自受试者 671；214 个 strong 窗口中有
175 个来自该受试者。再加上 180 秒窗口之间存在重叠，窗口级结果不能当作 452 个独立样本进行
统计推断。本结果适合证明“存在一类可恢复的输入倍频困难窗口”并比较四模型的初步表现，
不适合宣称某个方案已经在跨受试者层面稳定解决双峰问题。

## 测试集复评记录

测试集复评已经完成，范围只覆盖当前比较方案。2026-07-09 补充复评只更新
`local_rr_*` 口径，不重训、不重选 checkpoint。复评目的不是重排所有做过的实验，
而是确认当前阶段结论在独立测试集上是否仍成立：

1. `G3_C_wide_8p0` 是否仍优于 `G0_time_only` 和 `G0_f0_native_stft_pre_mixer`。
2. `G3_C_bandenergy` 是否仍表现为“部分指标更好，但整体不如宽频方案稳”。

完整性检查：`g_series_test_eval_manifest.csv` 中 12 个任务全部存在逐窗口指标、汇总表和运行清单。
该批运行清单记录的实际指标计算参数为 `metric_workers=16`、`metrics_chunk_size=64`。

派生汇总文件：

- 单次训练级汇总：`runs/test_eval_g_series_20260709_local_rr_canonical/g_series_local_rr_canonical_seed_summary.csv`
- 方案级汇总：`runs/test_eval_g_series_20260709_local_rr_canonical/g_series_local_rr_canonical_label_summary.csv`
- 分层分析汇总：`runs/test_eval_g_series_20260709_local_rr_canonical/stratified_analysis_20260709/`
- 二次谐波测试标签与覆盖率：`runs/bcg_second_harmonic_20260710/test_v2/`
- 二次谐波四模型任务汇总：`runs/bcg_second_harmonic_20260710/model_metrics/`
- 二次谐波模型输出纠正汇总：`runs/bcg_second_harmonic_20260710/corrections/`

不建议为非重点方案补测试集指标。那些结果即使补齐，也不会改变当前阶段决策，反而会增加汇报噪声和整理成本。

## 阶段结论

当前阶段可以向组会明确汇报以下结论：

1. 在纯时序 BCG 输入之外，STFT 辅助信息能提供额外的呼吸节律信息。
2. 当前更适合作为基准的是宽频、更高时间分辨率的 STFT 配置（`G3_C_wide_8p0`）。
3. 频带能量特征呈现局部收益信号，但更适合作为后续选择性修正的候选特征，不适合直接替代宽频 STFT。
4. 回顾性分层显示，宽频 STFT 在 baseline hard、low-spectrum 和 count-error 窗口中有较明显的
   收益；bandenergy 的正信号也集中在困难或部分 RR 分层中，但会误伤 easy、慢 RR 和部分 subject。
   这些结果揭示了选择性修正的潜在收益，尚不能直接证明可部署的选择条件。
5. 在 THO 参考定义的 BCG 二次谐波显著窗口中，四类网络都能在多数窗口抑制倍频并恢复基频，
   说明这是一种网络共有的可恢复能力；bandenergy 的呼吸率和周期计数误差最低，wide 的时延校正后
   形态最好。该结果支持 bandenergy 作为选择性修正候选，但不改变 wide 的本阶段基准地位。
6. 本次测试集复评只覆盖当前比较方案，不补不服务当前问题的分支。
7. 下一阶段关键问题从“继续扩大输入表示”转向“什么时候使用辅助时频信息”。

## 风险与不确定性

组会中需要主动说明以下风险，避免把阶段性证据讲成过度结论：

- 当前整理只适用于 THO research v2 soft-z 实验设置。
- 波形目标可能不是唯一合理输出，呼吸状态、事件和局部呼吸率可能更接近真实任务目标。
- 难恢复窗口和不确定窗口的后续定义必须避免数据泄漏，不能直接用目标信号上的失败结果作为训练期选择条件。
- 当前 hard/easy、count-error 和 low-spectrum 分层均含有目标侧评价信息；它们只用于事后理解，不能直接作为推理时的窗口选择规则。
- BCG 二次谐波分层同样使用 THO 参考频率，只是离线先验分析，不是 BCG-only 质量检测器或推理时 gate。
- 二次谐波阳性窗口集中在少数受试者，且 peak-only 层只有 5 个窗口；纠正率和模型差异不能外推为跨受试者稳定结论。
- 当前结论基于重点方案，不代表所有既有实验都需要或已经按当前指标完整复评。
- 当前测试集统计单位首先是 180 秒窗口；同一受试者内相邻窗口通常有重叠，因此这些窗口不能理解为彼此完全独立的受试者样本。

## 下一阶段建议

### 短期整理

- 基于本次测试集复评结果准备 G 系列代表结果表。
- 将二次谐波的全模型纠正、全模型失败、模型分歧和阈值边界案例作为平衡备份页，不用单个案例替代定量结果。
- 将当前稳定结论沉淀到 `findings.md` 或阶段报告中。

### 中期研究

- 设计不泄漏目标信息的难恢复 / 不确定窗口判据。
- 在更多受试者或受试者级重采样下验证二次谐波分层，重点确认 bandenergy 在 strong 层的收益是否稳定。
- 验证 `bandenergy` 或高频上下文是否能作为条件使用的修正信息，而不是全窗口无差别融合；
  若未来要做在线选择，再另行设计完全由 BCG 输入计算的无泄漏判据。
- 重新思考输出空间：从单一波形回归，逐步转向 latent respiratory state、局部呼吸率、事件或不确定性辅助目标。

### 暂不建议继续的方向

- 不对非重点方案做测试集补评。
- 不对所有既有实验做全量重评。
- 不用单次实验或单个指标改善作为新主线依据。

## PPT 页面骨架与备份页

后续转成第一次组会 PPT 时，不预设固定页数；以首次接触本研究的听众能够完整理解任务、方法、
证据和边界为准。正文采用“研究问题 -> 可信对照 -> 整体证据 -> 收益来源 -> 特殊困难窗口 ->
阶段决策”的判断链条，预计约 23--25 页。完整公式、复现实验命令和历史分支退场原因放到备份页
或现场问答。

正文页：

### 第一部分：研究问题

1. **标题页**：THO research v2 阶段进展；副标题直接给出阶段判断——已确定当前 STFT 输入基准，
   下一阶段转向选择性修正。
2. **本次汇报回答三个问题**：STFT 辅助信息是否有用；哪一种 STFT 表示适合作为当前基准；
   下一步为什么不继续无差别扩大输入。
3. **任务动机**：为什么需要从 BCG 恢复 THO-like 呼吸信息，以及这一任务的应用价值和信号来源。
4. **任务定义与主要困难**：`BCG -> THO-like 波形 / 呼吸状态`；主要困难包括心冲击、体动、
   接触变化、方向不一致、时延和局部质量下降。优先用一组典型 BCG / THO 对照波形建立直观认识。

### 第二部分：实验是否足以支持判断

5. **数据集与受试者划分**：180 秒窗口、100 Hz；训练 / 验证 / 测试分别为
   32 / 7 / 8 名受试者和 10141 / 2675 / 2310 个可用窗口；同时说明受试者隔离、窗口重叠和
   受试者间窗口数不均衡。
6. **输入、目标与预处理**：宽频 BCG soft-z 输入、THO soft-z 目标，以及滤波、状态对齐、
   分段 robust-z 和 soft-z 的作用；正文不展开全部参数。
7. **整体任务流程与模型结构**：BCG 进入时序主干；可选 STFT / 频带能量分支在 `pre_mixer`
   位置融合；最终输出 THO-like soft-z 波形。
8. **训练与监督逻辑**：训练 / 验证 / 测试分工、3 次独立训练；说明损失约束的是呼吸节律、
   低频形态、相对强弱和方向一致性，而不是只做逐点波形拟合。完整损失权重放到备份页。
9. **严格控制的四方案对照**：同页区分保持不变的设置和真正变化的时频辅助信息，并说明四类方案
   分别回答什么问题。
10. **评价框架**：用“指标 -> 回答的问题 -> 读表方向”说明整窗 RR、周期计数、4 秒时延校正形态、
    相对包络和局部 RR；完整公式放到备份页。

### 第三部分：当前基准方案如何确定

11. **独立测试集整体结果**：展示四方案主结果表，标明各指标的优劣方向，并突出 wide 和
    bandenergy 的不同优势。
12. **证据一——STFT 确实补充了纯时序信息**：比较 time-only、上一版 STFT 和 wide STFT，
    说明当前问题已经从“是否需要时频信息”转向“哪种表示更适合作为基准”。
13. **证据二——wide 更适合作为当前基准**：wide 的稳健 RR、时延校正形态和综合稳定性更好；
    bandenergy 的周期计数和局部 RR 略好，但在慢 RR 和部分窗口上存在退化。明确两者分别是
    “本阶段基准方案”和“选择性修正候选特征”。
14. **长尾与灾难性误差**：补充 robust RR p95、误差超过 1 / 2 bpm 的窗口比例、周期计数 p95
    和局部 RR MAE p95；说明 wide 更稳定地压低 robust RR 长尾，bandenergy 更擅长压低周期计数
    和局部 RR 长尾。

### 第四部分：收益发生在哪里

15. **分层分析的定位**：分层只用于回顾性理解收益来源；hard / easy、count-error 和 low-spectrum
    均含目标侧评价信息，不能直接作为推理时 gate。
16. **Hard / easy 与 low-spectrum**：wide 的收益主要来自 hard 和 low-spectrum 窗口；
    easy 窗口存在被时频方案误伤的现象，引出“选择性修正必须保护已正确窗口”。
17. **Count-error 与 clean-count**：已有周期计数错误的窗口中，bandenergy 呈现修正价值；
    clean-count 窗口中，时频方案可能引入少量非零计数误差。
18. **不同 RR 区间的角色分化**：wide 在 `<10 bpm` 和 `14--18 bpm` 更稳；bandenergy 在
    `10--14 bpm` 和 `18--24 bpm` 有局部优势，但不适合统一替代 wide。

### 第五部分：BCG 二次谐波困难窗口

19. **为什么单独研究二次谐波**：从输入信号机制解释倍频 / 双峰问题；说明这类分层比单纯按模型
    误差定义 hard window 更接近可观察的输入困难，但当前定义仍依赖 THO 参考。
20. **冻结规则、覆盖率与样本分布**：说明 strong / peak-doubling / harmonic-prominent 三层，
    阳性并集为 452 个窗口、占全部测试窗口 19.57%；同时展示阳性窗口和 strong 窗口集中在少数
    受试者的问题。
21. **四模型在阳性窗口上的结果**：四模型谐波纠正率均约为 95% 或以上；bandenergy 的稳健 RR、
    周期计数和局部 RR 最低，wide 的时延校正后形态最好。强调倍频恢复是网络共有能力，不是 STFT
    分支独有能力。
22. **平衡案例**：按预先定义的四类案例展示四模型均纠正、四模型均失败、模型间分歧和阈值附近窗口；
    不用单个成功案例替代定量结果。

### 第六部分：阶段决策与下一步

23. **阶段决策**：wide STFT 确定为本阶段基准；bandenergy 保留为选择性修正候选；不补评非重点
    历史分支，不继续无差别扩大输入表示。
24. **证据边界与风险**：当前结论只适用于 research v2 soft-z；统计单位首先是重叠窗口而非独立
    受试者；二次谐波阳性集中在少数受试者；当前分层不能直接作为无泄漏 gate；波形回归可能不是
    最终输出空间。
25. **下一阶段与三条带走结论**：建立 BCG-only 难度 / 不确定性判据；验证“wide 默认 + 条件
    bandenergy 修正”；做受试者级复核或重采样；探索 respiratory state、局部 RR、事件或不确定性
    辅助目标。页面结尾再次总结：STFT 值得保留、wide 是当前基准、bandenergy 用于条件修正探索。

备份页：

- 代表性预测图：正文使用四类平衡案例的概览；备份页保留单案例大图、易恢复窗口、纯时序和 STFT
  差异明显窗口，以及各案例的频谱细节。
- 模型结构细节：`patch_mixer1d`、STFT 窗长 / 步长、encoder 类型和 `pre_mixer` 融合位置。
- 训练配置细节：损失权重、warmup、早停、方向一致性筛选条件和 3 次独立训练。
- 指标公式：稳健带通 RR、周期计数、4 秒时延校正相关、相对包络和局部 RR。
- 分层分析细节：完整 hard / easy、count-error、RR 分层表，8 名受试者直接结果表，二次谐波
  子层完整结果、覆盖率和 analysis manifest。
- 数据导出包 provenance：commit、配置快照和实际 metadata。
- 复现实验命令和输出路径。
- 现场问答：受试者独立性、soft-z 含义、为什么不用 `val_loss` 排序、为什么不补非重点方案测试集、如何避免数据泄漏。

## 备份页材料

### 长尾和灾难性错误

这里的长尾指逐窗口误差分布的高端部分，主要看 p95 和超过固定阈值的窗口比例；灾难性错误
指 `robust RR >1 bpm` 或 `>2 bpm` 的窗口占比。长尾统计显示，`G3_C_wide_8p0`
对稳健带通 RR 的灾难性错误最稳，而 `G3_C_bandenergy` 更擅长压低周期计数和局部 RR 长尾。

| 方案 | robust RR mean | robust RR p95 | robust RR >1 bpm | robust RR >2 bpm | count bpm mean | count bpm p95 | local RR MAE p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `g0_time_only` | 0.7732 | 3.5090 | 0.1333 | 0.0661 | 0.7783 | 4.0111 | 3.2778 |
| `g0_f0_native_stft_pre_mixer` | 0.7333 | 3.1780 | 0.1198 | 0.0638 | 0.7834 | 4.0000 | 3.3840 |
| `g3_c_wide_8p0` | 0.7122 | 2.7691 | 0.1153 | 0.0589 | 0.7167 | 3.5556 | 3.2492 |
| `g3_c_bandenergy` | 0.7298 | 2.7883 | 0.1163 | 0.0605 | 0.6848 | 3.2222 | 3.1019 |

### 8 名测试受试者的直接结果

本表按测试集中的 `samp_id` 聚合逐窗口指标，用来检查宽频 STFT 的收益是否集中在少数受试者。
数值为 3 次独立训练的方案内均值；测试集只有 8 名受试者，且同一受试者内相邻窗口有重叠，
因此本表只作为备份页和坏例选择依据，不作为正式统计推断。

表内每项均按“`g0_time_only` / `g3_c_wide_8p0`”顺序直接列出，不表示 delta。

| subject | 窗口数 | robust RR | count bpm | lag4 corr | local RR MAE |
|---:|---:|---:|---:|---:|---:|
| 220 | 283 | 0.7083 / 0.6924 | 0.2760 / 0.2748 | 0.8599 / 0.8645 | 0.5862 / 0.5750 |
| 229 | 298 | 0.2760 / 0.2597 | 0.1864 / 0.1805 | 0.9211 / 0.9231 | 0.2676 / 0.2630 |
| 286 | 302 | 0.7298 / 0.6424 | 0.5717 / 0.5530 | 0.8691 / 0.8707 | 0.5919 / 0.5713 |
| 670 | 79 | 7.4582 / 7.1635 | 2.0394 / 1.9606 | 0.6234 / 0.6378 | 5.3693 / 5.6486 |
| 671 | 504 | 0.7482 / 0.6743 | 2.1122 / 1.8247 | 0.8058 / 0.8204 | 0.9801 / 0.9326 |
| 704 | 575 | 0.3467 / 0.2748 | 0.3480 / 0.3766 | 0.9108 / 0.9134 | 0.5197 / 0.5228 |
| 726 | 214 | 0.3840 / 0.3769 | 0.2056 / 0.2056 | 0.8976 / 0.8988 | 0.6015 / 0.5873 |
| 1006 | 55 | 0.6390 / 0.6071 | 0.3939 / 0.4000 | 0.8593 / 0.8618 | 0.8565 / 0.8294 |

宽频 STFT 在 8 名测试受试者上的 robust RR 均低于纯时序方案，说明当前整体收益并非由单一受试者拉动；
但 count bpm、lag4 corr 和 local RR MAE 的受试者内表现并不完全一致，仍需结合波形复核理解。

## 复现实验命令

先做预演，确认 12 个复评任务、设备分配和输出路径：

```bash
./.venv/bin/python scripts/run_g_series_test_eval.py \
  --specs configs/eval_specs/g_series_test_eval_20260705.csv \
  --output-dir runs/test_eval_g_series_20260709_local_rr_canonical \
  --manifest runs/test_eval_g_series_20260709_local_rr_canonical/g_series_test_eval_manifest.csv \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 2 \
  --metric-workers 16 \
  --metrics-chunk-size 64 \
  --start-stagger-sec 30 \
  --dry-run
```

确认预演没问题后正式运行：

```bash
./.venv/bin/python scripts/run_g_series_test_eval.py \
  --specs configs/eval_specs/g_series_test_eval_20260705.csv \
  --output-dir runs/test_eval_g_series_20260709_local_rr_canonical \
  --manifest runs/test_eval_g_series_20260709_local_rr_canonical/g_series_test_eval_manifest.csv \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 2 \
  --metric-workers 16 \
  --metrics-chunk-size 64 \
  --start-stagger-sec 30
```

如果只用一张 GPU，把设备和并发改成：

```bash
./.venv/bin/python scripts/run_g_series_test_eval.py \
  --specs configs/eval_specs/g_series_test_eval_20260705.csv \
  --output-dir runs/test_eval_g_series_20260709_local_rr_canonical \
  --manifest runs/test_eval_g_series_20260709_local_rr_canonical/g_series_test_eval_manifest.csv \
  --device cuda:0 \
  --max-parallel 1 \
  --metric-workers 16 \
  --metrics-chunk-size 64 \
  --start-stagger-sec 30
```

运行结束后，汇总以下目录：

```text
runs/test_eval_g_series_20260709_local_rr_canonical
```

重点汇总：

- 方案级均值和多次训练间差异。
- `G3_C_wide_8p0` 相对 `G0_time_only`、`G0_f0_native_stft_pre_mixer` 的变化。
- `G3_C_bandenergy` 与 `G3_C_wide_8p0` 的指标分歧。
- 是否支持当前基准配置判断，以及是否需要补代表预测图。

测试集复评完成后，生成分层分析表：

```bash
./.venv/bin/python scripts/stratified_eval_analysis.py \
  --eval-root runs/test_eval_g_series_20260709_local_rr_canonical \
  --output-dir runs/test_eval_g_series_20260709_local_rr_canonical/stratified_analysis_20260709 \
  --dataset-index /mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/training/dataset_index.csv \
  --seeds 20260700 20260837 20260901 \
  --comparison wide_vs_time=g3_c_wide_8p0:g0_time_only \
  --comparison bandenergy_vs_wide=g3_c_bandenergy:g3_c_wide_8p0 \
  --comparison bandenergy_vs_time=g3_c_bandenergy:g0_time_only \
  --comparison f0_vs_time=g0_f0_native_stft_pre_mixer:g0_time_only \
  --window-seconds 180
```

BCG 二次谐波分层的完整 discover、验证图复核、freeze、apply、预测导出和纠正汇总命令见
`scripts/README.md` 的“BCG 呼吸带二次谐波显著窗口分层”。本轮活动口径对应：

```text
验证集特征与复核：runs/bcg_second_harmonic_20260710/validation_full_v2/
冻结阈值：runs/bcg_second_harmonic_20260710/harmonic_thresholds.json
测试集固定标签：runs/bcg_second_harmonic_20260710/test_v2/
四模型任务指标：runs/bcg_second_harmonic_20260710/model_metrics/
模型阳性窗口预测：runs/bcg_second_harmonic_20260710/predictions/
谐波纠正汇总：runs/bcg_second_harmonic_20260710/corrections/
平衡案例图：runs/bcg_second_harmonic_20260710/figures/
```

其中 12 个 checkpoint 的预测导出是长时间 GPU 命令，由用户手动执行；分析脚本只读取保存后的
预测并生成派生统计。`validation/`、`validation_full/` 和 `test/` 是阈值冻结前的早期检查目录，
不进入本报告结论。

## 证据索引

- 当前证据账本：`docs/experiments/current_evidence_ledger.md`
- 当前指标说明：`docs/experiments/metric_schema.md`
- G 系列规划与结果解释：`docs/experiments/g_series_stft_input_resolution_band_plan.md`
- 测试集复评任务清单：`configs/eval_specs/g_series_test_eval_20260705.csv`
- 测试集复评脚本：`scripts/run_g_series_test_eval.py`
- 分层分析脚本：`scripts/stratified_eval_analysis.py`
- 分层分析输出：`runs/test_eval_g_series_20260709_local_rr_canonical/stratified_analysis_20260709/`
- 二次谐波分析设计：`docs/superpowers/specs/2026-07-10-bcg-second-harmonic-stratified-analysis-design.md`
- 二次谐波信号与标签实现：`resp_train/analysis/second_harmonic.py`
- 二次谐波特征、冻结和汇总脚本：`scripts/analyze_bcg_second_harmonic.py`
- 二次谐波预测导出脚本：`scripts/export_harmonic_predictions.py`
- 二次谐波复核与案例图脚本：`scripts/plot_bcg_second_harmonic.py`
- 二次谐波冻结阈值：`runs/bcg_second_harmonic_20260710/harmonic_thresholds.json`
- 二次谐波测试标签与覆盖率：`runs/bcg_second_harmonic_20260710/test_v2/`
- 二次谐波四模型任务指标：`runs/bcg_second_harmonic_20260710/model_metrics/`
- 二次谐波纠正率与案例：`runs/bcg_second_harmonic_20260710/corrections/`
  与 `runs/bcg_second_harmonic_20260710/figures/`
- 数据导出包 README：`/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/README.md`
- 数据导出包 manifest：`/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/manifest.json`
- 数据导出包配置快照：`/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/provenance/config_default.yaml`
- 数据导出包 git 记录：`/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/provenance/git_commit.txt`
  与 `/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/provenance/git_status_short.txt`
- 实际导出 metadata 抽查：`/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/whole_night/signal_bank/88/research_v2_signal_bank.json`
  与 `/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/whole_night/alignment/88/research_v2_alignment.json`
- 数据制作阶段记录：`/mnt/disk_code/marques/resp_prepare/findings.md`
