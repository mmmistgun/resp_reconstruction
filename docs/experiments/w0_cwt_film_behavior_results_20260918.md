# W0 CWT-FiLM 调制行为分析：完成记录

状态：三个冻结 W0 validation checkpoint 的调制行为分析、六个预定义案例的原始 batch 上下文复现及最终 bundle 已完成并通过完整性核验。本阶段没有训练、没有修改 checkpoint/cache、没有访问 test。

协议 ID：w0-cwt-film-behavior-v1-20260918。

## 1. 冻结身份与产物

- 执行 commit：9ef1082134ccfeb7361835f3dcb29804a876d881，执行时工作树干净。
- implementation lock SHA-256：c4bfc1907266599a097f4f8cfba88530bec92fc51b13b808e6fd8df1e706f9f2。
- W0 seed：20260811 / 20260812 / 20260813；selected epoch：13 / 15 / 14。
- split：validation，2675 个唯一窗口、7 个 samp_id；三个 seed 共 8025 行。
- summary：
  /mnt/disk_code/marques/resp_reconstruction/runs/w0_cwt_film_behavior_v1/summary/summary_c4bfc1907266_20260918T055139Z_773b7065a629
- summary manifest SHA-256：d6dee7a97d2fd1a5b7892ef9c8b4f7dca53739a43361d267af8b2bf6ca0ceff1。
- cases：
  /mnt/disk_code/marques/resp_reconstruction/runs/w0_cwt_film_behavior_v1/cases/cases_c4bfc1907266_20260918T055529Z_997fc0466f01
- cases manifest SHA-256：e56a1cba531434bb9c96fecf694858fd7f7882117cc63c05dd676982077d1bdf。
- final：
  /mnt/disk_code/marques/resp_reconstruction/runs/w0_cwt_film_behavior_v1/final/final_c4bfc1907266_20260918T061530Z_4d7b2b6b6c73
- final manifest SHA-256：c4cfcd610b7155f84ad1650666718513fc73d5e3b5e471843197cf6eb48991e7。

三个 analysis attempt 均为 completed。每个 seed：

- 2675 窗口、7 samp_id，无缺失或重复；
- denom_small=0；
- 历史 P−1 corrected FiLM 八项逐窗口统计的最大差约 1e-16，远低于 1e-6 门槛；
- checkpoint/cache 未修改，research_test_used=false。

六个预定义窗口按原始 batch=128 上下文重推。三个 seed 共保存 18 份选定张量，实际访问 640 个 validation batch rows/seed；首遍与二遍 R_scale/R_shift/R_total 最大绝对差为 8.33e-17。

## 2. 实际调制强度

Frobenius 范数覆盖每窗口全部 96 通道×1800 时间点。

| Seed | R_scale P05 / median / P95 | R_shift P05 / median / P95 | R_total P05 / median / P95 |
|---|---:|---:|---:|
| 20260811 | 0.279 / 0.319 / 0.344 | 0.173 / 0.209 / 0.229 | 0.341 / 0.393 / 0.428 |
| 20260812 | 0.267 / 0.311 / 0.353 | 0.148 / 0.196 / 0.242 | 0.312 / 0.376 / 0.432 |
| 20260813 | 0.223 / 0.277 / 0.321 | 0.134 / 0.181 / 0.214 | 0.277 / 0.361 / 0.418 |

三 seed 的窗口中位数均值为：

- R_scale=0.3020；
- R_shift=0.1954；
- R_total=0.3768。

shift/scale 的窗口中位比为 0.654 / 0.629 / 0.649。偏移路径弱于缩放路径，但约为其三分之二，不属于可忽略量。

由三项范数按余弦定理得到的 scale-delta 与 shift-delta 夹角余弦中位数为 0.082 / 0.040 / 0.202；R_total/(R_scale+R_shift) 中位数为 0.752 / 0.737 / 0.789。两项变化整体接近正交并略同向，没有观察到系统性强抵消。

观察结论：条件分支对 FiLM 前 latent 的实际改变量不小；“总调制变化很小”和“beta 路径近似无效”均不符合当前三 seed validation 描述。

## 3. 参数方向、raw 分布与限幅边缘

### 3.1 增益修正主要为负

| Seed | mean g | mean |g| | g<0 元素比例 | mean b | mean |b| |
|---|---:|---:|---:|---:|---:|
| 20260811 | -0.1630 | 0.2833 | 74.49% | 0.0011 | 0.2688 |
| 20260812 | -0.2174 | 0.2838 | 83.00% | 0.0110 | 0.2478 |
| 20260813 | -0.1383 | 0.2335 | 73.73% | -0.0226 | 0.2125 |

gamma_raw 的逐窗口统计均值显示明显负偏：

- seed 20260811：mean=-0.475，P05/P50/P95=-1.817/-0.532/1.108；
- seed 20260812：mean=-0.648，P05/P50/P95=-1.937/-0.639/0.597；
- seed 20260813：mean=-0.387，P05/P50/P95=-1.513/-0.371/0.710。

beta_raw 总体靠近零且正负更平衡。当前 W0 的主要增益行为是衰减部分 latent 位置，不是对称地放大/缩小；平移行为没有同等明显的全局符号偏置。

### 3.2 多阈值边缘比例

下表为三个 seed 的窗口 mean fraction 再平均，百分数：

| 参数 | tau | 正边缘 | 负边缘 | 合计 |
|---|---:|---:|---:|---:|
| g | 0.90 | 1.552 | 9.952 | 11.504 |
| g | 0.95 | 0.841 | 4.921 | 5.762 |
| g | 0.99 | 0.252 | 1.119 | 1.371 |
| b | 0.90 | 5.009 | 4.074 | 9.084 |
| b | 0.95 | 2.627 | 1.954 | 4.581 |
| b | 0.99 | 0.594 | 0.433 | 1.027 |

主阈值 tau=0.95 的合计比例按 seed 为：

- g：6.750% / 7.112% / 3.424%；
- b：6.691% / 5.037% / 2.015%。

观察结论：边缘元素不是零，但 tau=0.99 下只约 1%，不支持“广泛贴边饱和”。g 的边缘明显偏负，与其整体衰减方向一致；b 边缘相对平衡。三个 seed 的边缘比例有明显幅度差异，因此不能用单 checkpoint 代表固定饱和水平。

## 4. 信号质量与调制

独立质量组实际只有 waveform confidence high=931、medium=1744，没有 low。三个 seed 组均值再平均：

| 量 | High | Medium | High−Medium |
|---|---:|---:|---:|
| R_scale | 0.3063 | 0.2969 | +0.0094 |
| R_shift | 0.2076 | 0.1837 | +0.0240 |
| R_total | 0.3904 | 0.3634 | +0.0269 |
| Local RR MAE | 0.1855 | 0.7466 | -0.5610 |
| trajectory MAE | 0.0881 | 0.1872 | -0.0992 |
| 1-PCC | 0.0634 | 0.1728 | -0.1094 |

调制与连续质量轴的三 seed pooled Spearman 均值：

| 调制量 | waveform confidence | transient motion ratio |
|---|---:|---:|
| mean |b| | +0.500 | -0.429 |
| R_shift | +0.508 | -0.371 |
| mean |g| | +0.279 | -0.274 |
| R_scale | +0.158 | -0.171 |
| R_total | +0.335 | -0.227 |

within-samp 后方向仍多数保持。例如 mean |b| 与 confidence 在 21 个 seed×samp 估计中 18 个为正，与 motion 在 18 个为负；R_shift 对应也是 18/21 与 18/21。

观察结论：FiLM 行为确实随独立质量元数据变化。高 confidence、较低 motion 的窗口通常有更强的调制，差异主要来自 shift/beta 路径。该结果不能解释为“低质量需要更强补偿”；当前观察方向相反。

## 5. 调制与重建误差

完整窗口 pooled Spearman 的三 seed 均值中：

- mean |b|—trajectory=-0.614；
- mean |b|—1-PCC=-0.596；
- mean |b|—Local RR=-0.579；
- R_shift—trajectory=-0.539；
- R_shift—Local RR=-0.504；
- mean |g| 与五项误差约为 -0.214 至 -0.328；
- R_scale 与五项误差约为 -0.082 至 -0.169；
- R_total 与五项误差约为 -0.192 至 -0.313。

within-samp 敏感性仍以负方向为主：

- mean |b|—Local RR：均值 -0.485，中位数 -0.580，18/21 为负；
- mean |b|—trajectory：均值 -0.396，中位数 -0.560，17/21 为负；
- R_shift—trajectory：均值 -0.317，中位数 -0.487，16/21 为负；
- R_total—Local RR：均值 -0.308，中位数 -0.416，18/21 为负。

按 samp_id 均值计算时，mean |b|—Local RR 的三个 seed 相关为 -0.857/-0.571/-0.821；但每次只有 7 个 samp_id，只作描述。

### 5.1 关键分层反转

pooled 负相关不能直接解释成调制降低误差。按冻结 target-envelope modulation stratum 分层后：

- tau=0.95 的 g 边缘比例与 Local RR，在 high/low/medium 中约为 +0.298/-0.215/-0.221；
- g 边缘比例与 trajectory 约为 +0.371/-0.452/-0.207；
- tau=0.95 的 b 边缘比例与 Local RR 约为 +0.307/-0.220/-0.245；
- b 边缘比例与 trajectory 约为 +0.311/-0.434/-0.261；
- R_total—Local RR 在 high/low/medium 中约为 +0.213/-0.074/-0.216。

观察结论：全局“调制更强、误差更低”很大程度上混合了质量和任务难度。在 high target-modulation 难例中，边缘比例反而与较大误差同向。这一反转比 pooled 相关更适合指导后续实验。

## 6. 通道与时间结构

每个 seed 的 96 通道平均 R_total 范围为：

- 20260811：0.217–0.632，通道 CV=0.198；
- 20260812：0.221–0.539，通道 CV=0.206；
- 20260813：0.181–0.587，通道 CV=0.245。

通道内存在明显异质性。三个独立训练 seed 的通道 R_total rank Spearman 为 0.047/0.021/-0.178。由于 latent channel 存在置换/旋转自由度，不能把不同 seed 的同编号通道当作相同生理通道；这些低相关不等于功能不稳定，也不支持按固定通道编号删减。

跨全部窗口平均后的相对时间位置 profile 较平：

- R_total 的 36 个 5 s bins CV 为 0.5% / 0.7% / 2.0%；
- R_scale CV 为 0.6% / 0.6% / 1.8%；
- R_shift CV 为 0.8% / 1.2% / 2.8%。

但单窗口 time-bin R_total SD 中位数为 0.027/0.023/0.030，六个预定义案例也显示局部明显峰值。观察更符合“调制随窗口内容局部变化”，而不是“模型在固定窗口位置施加相同模板”。这与既有 TIME_MEAN/TIME_SHIFT 功能干预证据方向一致，但仍不是因果机制证明。

## 7. 案例图

固定案例为 dataset_row_id：

- 10785：低 R_total、低 Local RR；
- 12826：低 R_total、高 Local RR；
- 12324：高 R_total、低 Local RR；
- 8330：高 R_total、高 Local RR；
- 12001：high waveform confidence medoid；
- 10409：medium waveform confidence medoid。

案例图位于 cases/figures/。六图中 scale feature-delta RMS 通常高于 shift，但两者均随时间变化；局部峰值和 seed 幅度差异可见。案例由固定 medoid 规则选取，不能替代全量统计。

## 8. 机制假设与后续实验优先级

### 8.1 当前不优先 scale-only 重训练

shift 的相对强度约为 scale 的 63%–65%，并且 beta/shift 与质量和误差的关联最强。既有 P−1 的 BETA_ONLY/GAMMA_ONLY 也均未触发删减融合训练。当前证据不支持把 beta 视为冗余路径。

### 8.2 当前不支持全局放宽限幅

tau=0.99 的边缘比例约 1%，没有广泛饱和；pooled 中边缘比例还与较低误差同向。直接扩大 0.5 系数的全局训练实验缺少充分依据。

更有针对性的后续问题是：在 high target-modulation 窗口中，较高边缘比例究竟是对困难输入的必要响应，还是限制下的失配信号。优先设计 validation-only、冻结 checkpoint 的局部剂量/边界敏感性干预，并预先按 target stratum 与质量轴报告；不能只看 pooled 改善。

### 8.3 调制—质量变化需要收益干预验证

高 confidence/低 motion 窗口调制更强且误差更低，但这不能证明更强调制带来收益。下一步如检验收益，应在相同 samp、target modulation 和质量范围内做冻结模型的配对调制剂量干预，或另立严格匹配的训练消融。

### 8.4 不需要以“总调制过小”为理由启动输出敏感性

R_total 中位数约 0.36–0.39，条件分支对 latent 的实际变化不小。后续输出敏感性若执行，应服务于 high-modulation 分层反转和边缘机制，而不是验证一个已不成立的“分支几乎恒等”前提。

## 9. 证据边界

- validation-only，研究历史知情，不是确认性 test 证据；
- 2675 个窗口存在重叠，只有 7 个 samp_id；
- 相关系数没有因果含义，不输出窗口独立 p-value；
- 质量与 target modulation、samp composition 存在混杂；
- 通道编号不能跨独立训练 seed 作语义对齐；
- 本阶段不能证明 FiLM 相对无条件模型的因果贡献、0.5 最优或扩大/缩小限幅会提高性能。
