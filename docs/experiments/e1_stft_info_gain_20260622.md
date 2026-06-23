# E1 STFT 输入信息增益实验记录

## 结论摘要

本轮 E1 在干净配对口径下给出分支化结论：**patch wrapper 上 STFT 输入有稳定正增益**，
主要体现为降低 peak-band 长尾谐波误拣；`multiscale_decomp_mixer1d` 上仍只有趋势信号，
受 wrapper 优化多稳态和 gate failure 限制，暂不能给强结论。

> **E1 里程碑（patch 路线已闭环）**：STFT 输入在 `patch_mixer1d` 上的信息增益验证通过——
> 9 个 8Hz 成功配对 `frac_gt_1` 9/9 改善（均值 `-2.34pp`）、mean/median 9/9 改善、
> band/best-lag corr 9/9 提升，且 3Hz/12Hz 频带稳健。E1 核心问题“STFT 到底有没有用”在
> patch 路线上回答为**有用、证据干净**，给出继续投入 STFT 输入方向的 GO 信号。
> E1-D 进一步确认：轻 warmup（E1-C1）在 6 个代表性 seed 上得到 6/6 完整配对，
> dual 相对 time_only 的 `frac_gt_1` 平均 `-2.23pp`、`frac_gt_2` 平均 `-1.38pp`，
> 且 band/best-lag corr 平均小幅提升；因此 warmup 修复极性稳定性**不伤 STFT 净增益**，
> 可进入推荐 patch 训练配方。
> 两点收尾认知：(1) 旧记录的“相关性 trade-off”是基线错配假象（用原生 E1a 而非 E1a' 对比），
> 干净配对下 corr 与误拣率同向改善、无此消彼长；(2) 12Hz 未优于 8Hz，印证“8Hz 以上能量低、
> 更宽频带未必更好”，**8Hz 是甜点**，不再往宽推。
> 融合方法：当前 `concat → deep FusionHead` 已足以支撑 E1 正结论，**融合优化（gating/FiLM/
> attention）非 E1 必需，列为后续可选杠杆**（见“下一步 / 优先级 4”），且应在极性稳定性
> 处理之后再做。

- `multiscale_decomp_mixer1d` 的主指标崩坏**不是 STFT 造成的**：容量桩 E1a'（只有时序分支
  + 融合头、不含 STFT）的 peak-band MAE 已从 E1a 的 `0.483` 暴涨到 `1.325`，E1b 加入 STFT
  后（`1.19~1.43`）与 E1a' 同量级。崩坏发生在 STFT 进入之前，凶手是融合/上采样路径本身。
- `patch_mixer1d` 上 STFT 对长尾误拣有稳定净收益：补 seed 后，8Hz N0 在 9 个成功配对上
  `frac_gt_1` 9/9 下降，平均 `-2.34pp`；mean/median 9/9 下降，band/best-lag corr 9/9 提升。
  3Hz/12Hz 频带复核也保持负向 delta，说明 8Hz 不是单点挑选。
- 门控核查（见“疑点核查结果 / 疑点 2”）：方向门控**没有误杀**，初版“过严/幸存者偏差”担心
  已推翻——E1c `stft_only` 全 fail 是 magnitude STFT 极性盲（`val_signed_corr≈0.99`，corr≈−0.98）
  的必然结果，零星 time_only fail 是 seed 级真反向。被拦的都是真坏 run，弱信号结论不是门控美化的。

当前主候选更新为 `patch_mixer1d + conv2d + N0 8Hz + deep + fuse_len=600`，训练侧默认采用
E1-C1 轻 warmup。两个混淆因子核查后：门控已澄清（无误杀，正信号可信）；multiscale 崩坏不是
STFT 导致。**逐窗分布核查
（见“疑点 1 续”）整体修正了机制**：原生 E1a 与各 wrapper 变体的 peak-band **中位数**全在
`0.18~0.21`，不存在波形重建退化；全部差异都在**谐波误拣率（长尾）**。主因是「裸 backbone vs
wrapper 包装」带来的优化多稳态（误拣率方差），解码器结构（deep vs lite）只是次要因子；
peak-band **均值**因被 10~20% 误拣窗口主导而脆弱，不宜单独做 STFT 增益主判据。由此确定干净
判定路径：复刻原生重建的融合头（multiscale 用 lite+`fuse_len=18000`，patch 用 deep+`fuse_len=600`，
backbone 相关、不可迁移）+ 以 E1a'（非原生 E1a）为基线 + 改用误拣率判据 + 加 seed。
patch_mixer 路线已经按该口径补到 9 个 8Hz 成功配对，并完成 3Hz/12Hz 稳健性复核。

补做的干净 multiscale 复核（lite 头、`fuse_len=18000`、6 seed、time_only wrapper vs dual）
给出更清晰但仍非决定性的信号：12 个预期条目中 11 个成功、1 个 `dual/20260710`
因 checkpoint gate 未通过无 metrics；5 个同 seed 成功配对上，dual 的 `frac_gt_1`
平均低 `1.07pp`、`frac_gt_2` 平均低 `1.01pp`，median 也略低。这支持“STFT 有降低长尾谐波
误拣的趋势”，但 `20260700` 反向、`20260710` dual 直接 gate failure，说明优化稳定性仍是主要风险。

补做的干净 patch 复核已把证据强度推到可收口：`patch_mixer1d + deep + fuse_len=600`
在 8Hz N0 上补到 9 个成功配对，`frac_gt_1` 为 9/9 负向、平均 `-2.34pp`，`frac_gt_2`
为 8/9 负向、平均 `-1.87pp`，mean/median 均 9/9 下降；`band_limited_corr` 和
`best_lag_corr` 也是 9/9 提升。频带稳健性上，3Hz N0 的 4 个配对 `frac_gt_1` 4/4
改善，12Hz N0 为 3/4 改善，且两档 corr 均为 4/4 提升。由此可给出 E1 的干净结论：
**在 patch wrapper 口径下，STFT 输入稳定降低 peak-band 长尾谐波误拣，且未牺牲当前波形相关性；
该结论不再依赖单 seed 或均值口径。**

## 实验口径

- 计划与设计：
  - `docs/superpowers/plans/2026-06-21-e1-stft-info-gain.md`
  - `docs/superpowers/specs/2026-06-21-e1-stft-info-gain-design.md`
- 结果根目录：`runs/tho_research_v2_20260620_e1_stft_info_gain/`
- 推荐分析表：
  - `runs/tho_research_v2_20260620_e1_stft_info_gain_canonical_joined.csv`
  - `runs/tho_research_v2_20260620_e1_stft_info_gain_canonical_grouped_stats.csv`
  - `runs/tho_research_v2_20260620_e1_stft_info_gain_expected_status.csv`
- 主指标：`selection_task_rr_peak_band_abs_error_mean`，越低越好。
- 次指标：`selection_task_rr_spec_abs_error_mean`，越低越好。
- 辅助指标：`selection_task_relative_envelope_mae_mean`，越低越好。
- 波形诊断：`selection_waveform_band_limited_corr_mean`、
  `selection_waveform_best_lag_corr_mean`，越高越好。

`canonical_*` 表按 manifest tag 去重，一行对应一个预期实验结果，适合做最终统计。
`summary.csv` 会保留部分重复或额外 run，不建议直接用于最终均值。

## 术语说明

- `selection_*`：用于本轮模型选择和横向比较的统一指标列。它通常来自模型输出侧的评估结果，
  但通过统一前缀明确“这是当前实验选择口径下采用的指标”，避免和 `model_*`、`baseline_*`
  等原始汇总列混淆。
- `selection_task_*`：任务主指标或任务辅助指标，直接用于判断候选是否值得继续。例如
  `selection_task_rr_peak_band_abs_error_mean` 是本轮主指标，
  `selection_task_rr_spec_abs_error_mean` 是频域呼吸率护栏。
- `selection_waveform_*`：波形形态诊断指标，不直接作为模型胜负的唯一依据。它用于判断主指标改善
  是否伴随波形质量下降，例如 `band_limited_corr` 和 `best_lag_corr`。
- `canonical_*`：去重后的规范统计表。若同一个 manifest tag 有多次 run，canonical 表只保留
  一个用于最终分析的完整 run，避免重跑或 zero/main 复用导致某个配置被重复计入均值。
- `manifest`：manifest 生成器生成的预期实验清单，记录每个 tag 对应的 label、backbone、
  `high_hz`、encoder、seed 和 overrides。后续统计用它把时间戳 run 目录回连到实验参数。
- `tag`：一个预期实验配置的稳定名字，例如
  `E1b_patch_mixer1d_conv2d_8hz_20260700`。它比时间戳 run id 更适合做参数到指标的映射。
- `checkpoint gate` / `gate failure`：训练过程中用于筛掉明显不合格 checkpoint 的门控。
  本轮使用方向门控；未通过 gate 的 run 不进入最终均值，若属于已知系统性失败则记为
  `expected_no_result`。
- `expected_no_result`：预期无结果，不等同于漏跑。本轮主要指 E1c `stft_only` 在早期 gate
  失败后，为避免同类配置反复 fail-fast 而按计划跳过。
- `N0` / `N1`：STFT magnitude 的归一化口径。N0 仅使用 `log1p`；N1 在 `log1p` 后再按
  train 集 per-frequency-bin 鲁棒尺度做频带尺度对齐，只除尺度、不减中心。

## 结果数量

本轮最终主矩阵包含 51 个预期条目，其中 32 个完成，19 个为预期无结果，没有非预期缺失。

| 阶段 | 完成 | 预期无结果 | 说明 |
|---|---:|---:|---|
| main | 29 | 19 | E1a/E1a'/E1b 主线与 E1c 跳过项 |
| n1 | 3 | 0 | N1 频带尺度对照 |
| 合计 | 32 | 19 | `missing_unexpected=0` |

无结果原因：

| 标签 | backbone | 数量 | 原因 |
|---|---|---:|---|
| E1c | `patch_mixer1d` | 9 | `stft_only` 分支早期未通过 checkpoint gate；为避免 fail-fast 反复中断，后续同类实验按预期跳过 |
| E1c | `multiscale_decomp_mixer1d` | 9 | 同上 |
| E1a_prime | `patch_mixer1d` | 1 | seed `20260837` 的 `time_only` 对照未通过 checkpoint gate，未纳入均值 |

零号消融与重复探针另见 `runs/tho_research_v2_20260620_e1_stft_info_gain_zero_status.csv`，
不纳入 51 个最终主矩阵条目。

## 分组结果

| 阶段 | 标签 | backbone | high Hz | n | peak-band MAE | spec MAE | rel-env MAE | band corr | best-lag corr |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| main | E1a | `multiscale_decomp_mixer1d` | - | 3 | 0.483444 | 0.544867 | 0.240630 | 0.785895 | 0.839492 |
| main | E1a | `patch_mixer1d` | - | 3 | 0.489366 | 0.588676 | 0.231147 | 0.791260 | 0.842559 |
| main | E1a_prime | `multiscale_decomp_mixer1d` | - | 3 | 1.324590 | 0.502884 | 0.234933 | 0.798050 | 0.845876 |
| main | E1a_prime | `patch_mixer1d` | - | 2 | 0.538742 | 0.583473 | 0.243456 | 0.730683 | 0.780169 |
| main | E1b | `multiscale_decomp_mixer1d` | 3 | 3 | 1.281431 | 0.497773 | 0.235204 | 0.801923 | 0.850000 |
| main | E1b | `multiscale_decomp_mixer1d` | 8 | 3 | 1.431665 | 0.499963 | 0.234968 | 0.800326 | 0.847820 |
| main | E1b | `multiscale_decomp_mixer1d` | 12 | 3 | 1.193187 | 0.514384 | 0.237869 | 0.796982 | 0.845041 |
| main | E1b | `patch_mixer1d` | 3 | 3 | 0.498270 | 0.537383 | 0.241858 | 0.756971 | 0.806165 |
| main | E1b | `patch_mixer1d` | 8 | 3 | 0.473554 | 0.553811 | 0.239734 | 0.755005 | 0.805364 |
| main | E1b | `patch_mixer1d` | 12 | 3 | 0.482549 | 0.550526 | 0.239124 | 0.753432 | 0.802920 |
| n1 | E1n1 | `patch_mixer1d` | 3 | 3 | 0.472526 | 0.542129 | 0.241543 | 0.757923 | 0.806596 |

## 补充 seed 复核

在初版 3 seed 结果之后，追加 `20260901`、`20260911` 两个 seed，对比
`patch_mixer1d + conv2d + N1 3Hz` 与 `patch_mixer1d + conv2d + N0 8Hz`。

| 标签 | high Hz | norm | seed | run | 状态 | peak-band MAE | spec MAE | rel-env MAE | band corr | best-lag corr | 说明 |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---|
| E1n1 | 3 | N1 | 20260901 | `20260622_082439_153903` | gate failure | - | - | - | - | - | 训练到 38 epoch，但未通过 checkpoint gate |
| E1b | 8 | N0 | 20260901 | `20260622_082439_153340` | gate failure | - | - | - | - | - | 训练到 30 epoch，但未通过 checkpoint gate |
| E1n1 | 3 | N1 | 20260911 | `20260622_083101_649718` | complete | 0.493336 | 0.552534 | 0.238547 | 0.756890 | 0.809244 | 成功完成 |
| E1b | 8 | N0 | 20260911 | `20260622_083101_647270` | complete | 0.541439 | 0.539391 | 0.247670 | 0.756382 | 0.807732 | 成功完成，但主指标明显偏差 |

补充后仅按已完成 run 粗略合并：

- E1n1 / N1 3Hz：4 个完整 seed 的 peak-band MAE 均值约 `0.477729`。
- E1b / N0 8Hz：4 个完整 seed 的 peak-band MAE 均值约 `0.490525`。

这说明 N1 3Hz 相对 N0 8Hz 更稳，但相对 E1a 纯时序 baseline 的优势不再明确。
同时两个 `20260901` 都 gate failure，提示这条 STFT 路线对训练 seed 仍敏感。

## 分析判断

### PatchMixer 路线

`patch_mixer1d` 是本轮唯一有继续价值的 STFT 输入路线。

- E1a 纯时序 baseline 的 peak-band MAE 为 `0.489366`。
- E1b 8Hz 的 peak-band MAE 降到 `0.473554`，主指标略好。
- E1n1 3Hz 的 peak-band MAE 为 `0.472526`，略优于 N0 3Hz，且稳定性更好。
- 补充 seed 后，E1n1 / N1 3Hz 的 4 个完整 seed 均值约 `0.477729`；
  E1b / N0 8Hz 的 4 个完整 seed 均值约 `0.490525`。
- **归因视角修正**：E1b 该减去的不是纯时序 E1a，而是同样付出融合头容量代价的 E1a'。
  E1a' patch peak-band MAE 为 `0.538742`（n=2，1 个 seed gate failure），E1b 8Hz（`0.474`）
  与 N1 3Hz（`0.473`）都明显优于 E1a'。即“融合头容量本身先把指标退化到 `0.539`，STFT
  又把它拉回到 `0.47`”——相对正确基线，STFT 的正贡献比“E1b 微弱优于 E1a”更明确。
  但 E1a' patch 仅 2 个 seed，这条基线本身偏弱，结论强度受此限制（见“疑点核查结果 / 疑点 2”）。

但该收益不全面，也不够稳定：E1b/E1n1 的 `band_limited_corr` 和 `best_lag_corr`
明显低于 E1a，`relative_envelope_mae` 也没有同步改善。补充 seed 还暴露出 gate failure
和 N0 8Hz 主指标波动。因此当前只能记录为“PatchMixer 上存在条件性局部信号”，
不能判定为“STFT 输入稳定改善波形重建质量”。

### MultiScale 路线

`multiscale_decomp_mixer1d` 主指标崩坏，但 **E1a' 容量桩与 fuse_len 对照坐实凶手不是 STFT**。

- E1a baseline peak-band MAE 为 `0.483444`。
- E1a_prime（time_only，不含 STFT，仅时序主干 + 融合头）已是 `1.324590`（std `0.3139`，极不稳定）。
- E1b 3/8/12Hz 分别为 `1.281431`、`1.431665`、`1.193187`，与 E1a_prime 同量级。
- E1a_prime 将 `fuse_len` 从 `600` 提高到 `18000` 后，3 seed peak-band MAE 降到
  `0.941701`，有改善但没有回到 `0.48` 量级。
- 继续把 FusionHead 从 deep 改为 lite（纯 `1x1`、无 GroupNorm、2 层）后，3 seed peak-band MAE
  降到 `0.671496`；两个 seed 回到 `0.45~0.48`，但 seed `20260710` 仍为 `1.081937`。
- 单拆 deep 头后，`k1_norm` 均值 `0.628577`，`k3_no_norm` 均值 `0.767004`。GroupNorm 单独
  不是主因；去掉 `kernel_size=3` 更有帮助，但仍被 seed `20260710` 拉高。

E1a' 不含任何 STFT 输入，却已经把 peak-band MAE 抬到 E1b 的水平。这把原先“更像是融合头
破坏”的推测坐实为：**主指标崩坏在 STFT 进入之前就已发生，与 STFT 信息无关**。但逐窗分布
核查（见“疑点 1 续”）进一步澄清了崩坏的**性质**：**不是波形重建退化（中位数与原生 E1a
持平，均在 `0.18~0.21`），而是 wrapper 包装路径的优化多稳态抬高了谐波误拣率（长尾）**；
解码器结构（deep vs lite）只是次要加尾因子。因此 multiscale 这条线既不能记为“STFT 破坏
主指标”，也不应记为“融合头结构破坏重建”，而应记为“wrapper 双分支路径相对裸主干有更高的
误拣率方差，需用 E1a' 为基线、误拣率为判据、并加 seed 才能在其上判 STFT 增益”。
具体见“疑点核查结果 / 疑点 1”。

### STFT-only 路线

E1c `stft_only` 在两个 backbone 上均未通过早期 checkpoint gate，后续同类实验按预期跳过。
当前结论是：在现有 loss、gate 和模型容量设置下，STFT magnitude 单分支不足以作为主输入路线。

### N1 频带尺度对齐

N1 只在 `patch_mixer1d + high_hz=3` 代表档做了 3 seed 对照。
相对 N0 3Hz，N1 的主指标略好，spec MAE 略差，其余指标基本持平。

这一结果支持继续保留 N1 作为候选归一化方式，但不足以单独证明 STFT 输入稳定有效。

## 疑点核查结果

对两个混淆因子做了核查（读代码 + 翻 gate failure run 的 `val_signed_corr` 轨迹）。
结论之一推翻了初版的“门控过严”担心。

### 疑点 1：融合路径有损压缩（fuse_len 对照后机制修正）

- 代码坐实：`TimeStftDual1D.forward`（`resp_train/models/stft_branch.py:257`）确认 multiscale 的
  `(B, C, 18000)` 中间特征经 `align_to_time` 降到 `fuse_len=600`，`FusionHead` 再 linear 上采样回 18000。
- 机制修正：初版写的“600 点抹掉呼吸峰位”不准确——patch_mixer 用更粗的 T'≈140 反而没崩，
  且 600 点（每次呼吸约 10-16 点）足以表达呼吸。
- 真正证据：multiscale 崩坏 run 的 `band_limited_corr=0.798`、`best_lag_corr=0.846`（均不低于
  baseline），但 `peak-band MAE=1.325`（baseline 的 2.7 倍）。整体波形相关性好、极性也正常
  （time_only multiscale 三 seed `val_signed_corr≈0.20`，全过 gate），唯独 RR 提取极差。
  这指向“18000 高分辨率特征被约 30× 有损压缩 + 浅层 FusionHead 上采样，引入破坏 peak 检测的
  局部伪结构”，而 patch_mixer 的 token（140）本就是低密度抽象特征、140→600 不丢信息，故未崩。
- `fuse_len` 对照已完成：对 `multiscale_decomp_mixer1d + time_only(E1a')` 提高到
  `fuse_len=18000`，3 seed 均完成，结果如下。

| seed | run | epochs | peak-band MAE | spec MAE | rel-env MAE | band corr | best-lag corr |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20260700 | `20260622_100436_562206` | 17 | 0.875878 | 0.530082 | 0.237111 | 0.786642 | 0.839979 |
| 20260710 | `20260622_100436_562376` | 30 | 1.057582 | 0.512558 | 0.225379 | 0.792558 | 0.845298 |
| 20260837 | `20260622_101029_183942` | 21 | 0.891645 | 0.539939 | 0.231104 | 0.793604 | 0.845021 |
| 均值 | - | - | 0.941701 | 0.527526 | 0.231198 | 0.790935 | 0.843433 |

随后在相同 `fuse_len=18000` 下，把 FusionHead 改成仿原生 fuse 的 `fusion_decoder=lite`
（纯 `1x1`、无 GroupNorm、2 层），结果如下。

| seed | run | epochs | peak-band MAE | spec MAE | rel-env MAE | band corr | best-lag corr |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20260700 | `20260622_102936_000616` | 18 | 0.451012 | 0.526249 | 0.239523 | 0.788869 | 0.841683 |
| 20260710 | `20260622_102936_018967` | 31 | 1.081937 | 0.518034 | 0.236363 | 0.793126 | 0.846246 |
| 20260837 | `20260622_103612_728379` | 23 | 0.481539 | 0.551438 | 0.236763 | 0.789554 | 0.843479 |
| 均值 | - | - | 0.671496 | 0.531907 | 0.237550 | 0.790516 | 0.843803 |

对比 deep `fuse_len=18000`，lite 头把 peak-band 均值从 `0.941701` 降到 `0.671496`，
其中 seed `20260700` 从 `0.875878` 降到 `0.451012`，seed `20260837` 从 `0.891645` 降到
`0.481539`；但 seed `20260710` 从 `1.057582` 变为 `1.081937`，没有改善。当时（仅看均值）的
判断是“FusionHead 结构是主要混淆因子、lite 头未稳定回到 `0.483`、不宜直接用 lite 重跑”。
**该判断已被“疑点 1 续”逐窗分布核查修正：median 证明不存在重建退化，结构只是次要因子，
`20260710` 的偏高来自 wrapper 误拣率方差而非解码器层；最终结论改为采用 lite 头、误拣率判据
与加 seed（见“疑点 1 续”与“下一步 / 优先级 0”）。**

继续在相同 `fuse_len=18000` 下单拆 deep 头中的 `kernel_size=3` 与 GroupNorm：

- `k3_no_norm`：保持 deep 的两层 `kernel_size=3` 局部卷积，去掉 GroupNorm。
- `k1_norm`：保持 lite 的纯 `1x1` 解码，但加入 GroupNorm。

| decoder | seed | run | epochs | peak-band MAE | spec MAE | rel-env MAE | band corr | best-lag corr |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `k1_norm` | 20260700 | `20260622_104959_415765` | 18 | 0.452780 | 0.536105 | 0.239015 | 0.787238 | 0.840762 |
| `k1_norm` | 20260710 | `20260622_105746_883578` | 31 | 0.909519 | 0.511463 | 0.231213 | 0.790322 | 0.844841 |
| `k1_norm` | 20260837 | `20260622_110619_493408` | 23 | 0.523433 | 0.514201 | 0.239992 | 0.787682 | 0.841127 |
| `k1_norm` 均值 | - | - | - | 0.628577 | 0.520590 | 0.236740 | 0.788414 | 0.842243 |
| `k3_no_norm` | 20260700 | `20260622_104958_221481` | 23 | 0.804188 | 0.523511 | 0.236176 | 0.787662 | 0.842522 |
| `k3_no_norm` | 20260710 | `20260622_105842_828298` | 31 | 1.029698 | 0.515844 | 0.233074 | 0.790711 | 0.844311 |
| `k3_no_norm` | 20260837 | `20260622_110725_583216` | 10 | 0.467125 | 0.577723 | 0.243532 | 0.782656 | 0.838955 |
| `k3_no_norm` 均值 | - | - | - | 0.767004 | 0.539026 | 0.237594 | 0.787010 | 0.841929 |

四种 decoder 的合并均值：

| decoder | peak-band MAE | spec MAE | rel-env MAE | band corr | best-lag corr |
|---|---:|---:|---:|---:|---:|
| `k1_norm` | 0.628577 | 0.520590 | 0.236740 | 0.788414 | 0.842243 |
| `lite` | 0.671496 | 0.531907 | 0.237550 | 0.790516 | 0.843803 |
| `k3_no_norm` | 0.767004 | 0.539026 | 0.237594 | 0.787010 | 0.841929 |
| `deep` | 0.941701 | 0.527526 | 0.231198 | 0.790935 | 0.843433 |

结论：GroupNorm 不是单独主因。若 GroupNorm 是主因，`k1_norm` 应明显劣于 `lite`，但实际
`k1_norm` 还略好于 `lite`。`kernel_size=3` 局部卷积路径更可疑：`k3_no_norm` 虽优于 deep，
但仍明显差于 `k1_norm` / `lite`。不过所有变体在 seed `20260710` 上都没有稳定回到 `0.48`，
说明还存在 seed 级训练/峰值提取敏感性，不能只把问题归因于一个层。
**（注：以上均基于 peak-band「均值」，该结论被下一节逐窗分布核查整体修正——见“疑点 1 续”。）**

### 疑点 1 续：逐窗分布核查（中位数证伪“结构是真因”）

前述 deep / lite / k1_norm / k3_no_norm 对照都基于 peak-band **均值**。补做逐窗分布核查后，
发现均值结论具有误导性，需整体修正。分析脚本：`scripts/peak_band_misclass_rate.py`。

关键对照——把**原生 E1a（裸 multiscale，无 wrapper，走 native fuse）**按 seed 摊开，
与各 wrapper 变体在同 seed 对齐：

| 配置（time_only，wrapper 为 fuse_len=18000） | 20260700 | 20260710 | 20260837 | 均值 |
|---|---:|---:|---:|---:|
| 原生 E1a（无 wrapper，native fuse） | 0.527 | **0.448** | 0.476 | **0.483** |
| wrapper deep | 0.876 | 1.058 | 0.892 | 0.942 |
| wrapper k3_no_norm | 0.804 | 1.030 | 0.467 | 0.767 |
| wrapper lite | 0.451 | 1.082 | 0.482 | 0.672 |
| wrapper k1_norm | 0.453 | 0.910 | 0.523 | 0.629 |

决定性发现：**seed `20260710` 上原生 E1a 恰恰是三 seed 里最好的 `0.448`，而所有 wrapper 变体
最差。** 这证伪了「`20260710` 是 peak-band 口径本身不稳」的猜测——口径在原生路径上稳得很，
不稳的是 wrapper 路径。

再看 seed `20260710` 各配置的**逐窗分布**（n=2675，误拣率 = 逐窗误差 > 阈值的占比）：

| seed710 | mean | **median** | 截尾95均值 | p95 | max | 误拣率>1.0 | 误拣率>2.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原生 E1a | 0.448 | **0.180** | 0.288 | 1.68 | 9.0 | 11.3% | 3.9% |
| wrapper lite | 1.082 | **0.192** | 0.644 | 7.53 | 12.8 | 20.6% | 13.9% |
| wrapper k1_norm | 0.910 | **0.201** | 0.550 | 5.76 | 12.1 | 19.8% | 12.0% |
| wrapper deep | 1.058 | **0.209** | — | 6.20 | — | 22.4% | 15.0% |

三条结论，整体修正前述“结构是真因”：

1. **四者 median 全在 `0.18~0.21` → 不存在波形重建退化。** 典型窗口大家一样准；
   “FusionHead 破坏 peak-band”是被均值放大的假象。
2. **全部差异 = 谐波误拣率（长尾）。** seed710 上 native→wrapper(lite) 误拣率 `11.3%→20.6%`
   （+9.3pp，大头），wrapper-lite→wrapper-deep 仅 `20.6%→22.4%`（+1.8pp，小头）。
   **主因是「裸 backbone vs wrapper 包装」本身，解码器结构（deep vs lite）只是次要因子。**
   代码核查确认：time_only + fuse_len=18000 + lite 头与 native **计算图等价**——`align_to_time`
   （`stft_branch.py:48`）与 `FusionHead.forward`（尺寸一致时短路返回）均无重采样，唯一差别是
   解码头初始化 RNG 位置不同 → 端到端训练收敛到不同 basin。即 wrapper 多一个解码头带来
   **优化多稳态**，在某些 seed 收敛到误拣率翻倍的解；3 个 seed 平均不掉（原生三 seed 紧，
   `0.527/0.448/0.476`；wrapper-lite 散，`0.451/1.082/0.482`）。
3. **peak-band 均值是被 10~20% 谐波误拣窗口主导的脆弱口径**，不适合单独做 STFT 增益的主判据。

方法论修正（直接决定后续 E1 判定方式）：

- **STFT 增益的基线应是 E1a'（time_only wrapper），不是原生 E1a。** wrapper 多稳态是共模噪声，
  E1b（dual）和 E1a'（time_only）都带它，对比 `(E1b − E1a')` 时抵消；当初拿 E1b 比原生 E1a 是错配。
- **判据改为谐波误拣率（逐窗 peak-band > 阈值的占比）+ median 作平稳性 sanity，均值仅参考。**
  且 STFT 若有用，价值大概率正在降低谐波误拣（提供显式频率结构区分基频 vs 2× 谐波），
  该收益恰好活在长尾/误拣率，不在 median。
- **必须加到 5~8 seed**，把初始化方差压到噪声地板以下，否则单 seed 的尾巴就淹没 STFT 信号。
- **融合头要复刻各自 backbone 的原生重建路径，且该选择是 backbone 相关的，不能跨 backbone 迁移**：
  - `multiscale`：原生重建是「全分辨率 `(B,48,18000)` 特征 → 纯 `1x1` fuse」，故 wrapper 用
    `fusion_decoder=lite`（纯 `1x1`）+ `fuse_len=18000` 才复刻原生，deep 反而加尾。
  - `patch_mixer`：原生重建是「140 token →`patch_head`(Linear) → overlap-add 展开回 18000」，
    是有容量的展开式解码；lite 撑不起 `140→18000` 的展开（smoke 实测 `fuse_len=140,lite` 崩到
    median `0.857`、`fuse_len=600,lite` median `0.283`），需 `fusion_decoder=deep`+`fuse_len=600`
    才复刻原生（smoke `deep,600` mean `0.534`/median `0.227`，与旧 E1a' seed700 `0.5346/0.2265`
    逐位对齐）。**旧 patch E1a'/E1b 本就是 deep+600，从未受融合头混淆因子污染——FusionHead
    混淆因子是 multiscale 独有的；patch 干净复核只需补 seed/补 837 基线/做配对统计，无需改口径。**

### 疑点 1 续二：干净 multiscale 复核（lite + 全分辨率 + 6 seed）

按上面的修正口径，补跑 `multiscale_decomp_mixer1d` 的干净复核：

- 共同配置：`fusion_decoder=lite`、`fuse_len=18000`、`stft_high_hz=3.0`、`stft_norm=n0`、
  `stft_encoder_type=conv2d`、`baseline.enabled=false`。
- 只切 `branch_mode`：`time_only` 作为 E1a' 基线，`dual` 作为 E1b。
- 6 个 seed：`20260700`、`20260710`、`20260837`、`20260901`、`20260911`、`20260920`。
- 结果文件：
  - `runs/e1_clean_ms_detail.csv`
  - `runs/e1_clean_ms_grouped.csv`
  - `runs/e1_clean_ms_status.csv`
  - `runs/e1_clean_ms_paired_delta.csv`

预期结果数量：2 个分支 × 6 seed = 12 个。实际 11 个成功，1 个失败且原因明确：

| seed | 分支 | 状态 | run | mean | median | p95 | frac_gt_1 | frac_gt_2 | 无结果原因 |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 20260700 | time_only | ok | `20260622_143358_403721` | 0.451 | 0.171 | 1.736 | 10.2% | 4.2% | - |
| 20260700 | dual | ok | `20260622_143402_590652` | 0.599 | 0.173 | 2.764 | 13.2% | 7.0% | - |
| 20260710 | time_only | ok | `20260622_144142_695286` | 1.082 | 0.192 | 7.526 | 20.6% | 13.9% | - |
| 20260710 | dual | failed | `20260622_144244_794376` | - | - | - | - | - | checkpoint gate 未通过，未保存 checkpoint/metrics；best_epoch=34，best_val_loss=0.857046，val_signed_corr_min=0.686316，gate_passed=0/42 |
| 20260837 | time_only | ok | `20260622_145119_030597` | 0.482 | 0.181 | 2.002 | 11.9% | 5.0% | - |
| 20260837 | dual | ok | `20260622_145121_102966` | 0.488 | 0.175 | 1.850 | 11.4% | 4.6% | - |
| 20260901 | time_only | ok | `20260622_145908_564562` | 0.720 | 0.176 | 3.815 | 15.3% | 8.1% | - |
| 20260901 | dual | ok | `20260622_145748_740946` | 0.560 | 0.172 | 2.631 | 14.5% | 7.4% | - |
| 20260911 | time_only | ok | `20260622_150705_747445` | 0.929 | 0.193 | 5.650 | 19.0% | 12.4% | - |
| 20260911 | dual | ok | `20260622_150535_797784` | 0.607 | 0.174 | 2.633 | 13.8% | 6.7% | - |
| 20260920 | time_only | ok | `20260622_151356_516338` | 0.540 | 0.184 | 2.314 | 13.5% | 6.3% | - |
| 20260920 | dual | ok | `20260622_151354_158208` | 0.490 | 0.169 | 2.144 | 11.7% | 5.5% | - |

成功 run 的跨 seed 聚合（注意 dual 只有 5 个成功 seed，time_only 有 6 个）：

| 分支 | n | mean | median | p95 | frac_gt_1 | frac_gt_2 |
|---|---:|---:|---:|---:|---:|---:|
| time_only | 6 | 0.701 | 0.183 | 3.840 | 15.1% | 8.3% |
| dual | 5 | 0.549 | 0.173 | 2.404 | 12.9% | 6.2% |

更公平的主判据是 5 个**同 seed 成功配对**的差值（dual - time_only）：

| seed | Δ frac_gt_1 | Δ frac_gt_2 | Δ mean | Δ median |
|---:|---:|---:|---:|---:|
| 20260700 | +2.92pp | +2.80pp | +0.148 | +0.002 |
| 20260837 | -0.49pp | -0.49pp | +0.007 | -0.006 |
| 20260901 | -0.75pp | -0.79pp | -0.160 | -0.004 |
| 20260911 | -5.23pp | -5.79pp | -0.322 | -0.019 |
| 20260920 | -1.79pp | -0.79pp | -0.050 | -0.015 |
| 平均 | **-1.07pp** | **-1.01pp** | **-0.075** | **-0.008** |

解释：

1. **median sanity 通过**：两臂 median 都在 `0.17~0.19`，dual 没有伤害典型窗口；
   这继续支持“差异主要在长尾误拣，而非波形重建退化”。
2. **STFT 有降低谐波误拣的趋势**：5 个成功配对中 4 个 `frac_gt_1` 为负，平均 `-1.07pp`；
   `frac_gt_2` 同样平均 `-1.01pp`。
3. **证据仍不够强**：`20260700` 反向，`20260710` 的 dual 分支 gate failure。也就是说，
   STFT 对长尾可能有净收益，但 dual wrapper 的优化稳定性仍会吞掉部分收益。后续若要给强结论，
   应补到 8 seed 或做稳定性改造后再复核。

### 疑点 1 续三：patch_mixer 干净配对复核（deep + fuse_len=600 + 8Hz N0）

patch 路线不能套 multiscale 的 `lite+fuse_len=18000`：patch 原生重建依赖
`140 token -> patch_head -> overlap-add` 的展开式解码，smoke 已确认 `fuse_len=140,lite`
和 `fuse_len=600,lite` 都会伤害 E1a' 基线；`fuse_len=600,deep` 可复现旧 E1a' seed700
（mean `0.5345` / median `0.2266`）。因此本轮正式复核使用：

- `time_backbone=patch_mixer1d`
- `fusion_decoder=deep`
- `fuse_len=600`
- `stft_high_hz=8.0`
- `stft_norm=n0`
- `stft_encoder_type=conv2d`
- 6 seed：`20260700`、`20260710`、`20260837`、`20260901`、`20260911`、`20260920`
- 只切 `branch_mode`：`time_only`（E1a'） vs `dual`（E1b）

产物：

- `runs/e1_clean_patch600_deep_detail.csv`
- `runs/e1_clean_patch600_deep_grouped.csv`
- `runs/e1_clean_patch600_deep_status.csv`
- `runs/e1_clean_patch600_deep_paired_delta.csv`
- `runs/e1_clean_patch600_deep_corr_summary.csv`
- `runs/e1_clean_patch600_deep_corr_paired_delta.csv`

完整状态表：

| 分支 | seed | 状态 | 原因 |
|---|---:|---|---|
| time_only | 20260700 | ok | - |
| time_only | 20260710 | ok | - |
| time_only | 20260837 | gate fail | 未满足 `checkpoint_gate(auto_direction<=0.5)` |
| time_only | 20260901 | gate fail | 未满足 `checkpoint_gate(auto_direction<=0.5)` |
| time_only | 20260911 | ok | - |
| time_only | 20260920 | ok | - |
| dual | 20260700 | ok | - |
| dual | 20260710 | ok | - |
| dual | 20260837 | ok | time_only 失败，无法配对 |
| dual | 20260901 | gate fail | 未满足 `checkpoint_gate(auto_direction<=0.5)` |
| dual | 20260911 | ok | - |
| dual | 20260920 | ok | - |

跨 seed 聚合（仅有效 run）：

| 分支 | n | mean | median | p95 | frac_gt_1 | frac_gt_2 |
|---|---:|---:|---:|---:|---:|---:|
| time_only | 4 | 0.589 | 0.231 | 2.348 | 14.0% | 5.9% |
| dual | 5 | 0.497 | 0.196 | 1.984 | 12.4% | 4.9% |

同 seed 成功配对差值（dual - time_only）：

| seed | Δ frac_gt_1 | Δ frac_gt_2 | Δ mean | Δ median |
|---:|---:|---:|---:|---:|
| 20260700 | -1.53pp | -0.52pp | -0.071 | -0.023 |
| 20260710 | -1.35pp | -1.12pp | -0.049 | -0.009 |
| 20260911 | -0.15pp | +0.56pp | -0.045 | -0.042 |
| 20260920 | -2.24pp | -2.21pp | -0.169 | -0.047 |
| 平均 | **-1.32pp** | **-0.82pp** | **-0.084** | **-0.030** |

同 seed 成功配对的波形相关性差值（dual - time_only）：

| seed | Δ band_limited_corr | Δ best_lag_corr |
|---:|---:|---:|
| 20260700 | +0.019 | +0.020 |
| 20260710 | +0.025 | +0.026 |
| 20260911 | +0.028 | +0.032 |
| 20260920 | +0.030 | +0.036 |
| 平均 | **+0.026** | **+0.029** |

解释：

1. **STFT 对 patch 的长尾误拣有稳定净收益**：4 个成功配对的 `frac_gt_1` 全部为负，
   平均降低 `1.32pp`；mean/median 也全部为负，说明改善不是只靠单个极端窗口。
2. **证据强于 multiscale，但不是完整 6 配对**：time_only 的 `20260837`、`20260901`
   gate fail，dual 的 `20260901` gate fail，最终只有 4 个可配对 seed。方向性已满足
   “至少 4 个 seed 为负”的判读条件，但有效样本数不足 6，需要把训练稳定性作为限制条件写入结论。
3. **`frac_gt_2` 不是全胜**：4 个配对中 3 个为负，`20260911` 小幅反向（+0.56pp）。
   STFT 的主收益更稳定地体现在 `frac_gt_1` 和整体 mean/median 上。
4. **gate failure 本身是 patch wrapper 的优化稳定性问题**：`20260837/20260901`
   在 time_only 侧失败，说明多稳态不是 STFT 独有；`dual/20260837` 反而成功，不能简单解释为
   “加 STFT 更不稳”。后续若要扩大结论，应先把 seed 数补到 8 或用稳定性改造减少 gate fail。
5. **本轮没有复现旧记录的相关性 trade-off**：成功配对中 `band_limited_corr` 和 `best_lag_corr`
   都是 4/4 提升，均值分别 `+0.026`、`+0.029`。这让 patch 路线的正结论更干净：STFT 不仅降低
   peak-band 长尾误拣，也没有牺牲当前两项波形相关性均值。

### 疑点 1 续三 B1：patch_mixer 补 seed 与频带稳健性

为避免“4 个成功配对不足 6 个”的样本量风险，继续沿用上节干净 patch 配方：

- `time_backbone=patch_mixer1d`
- `fusion_decoder=deep`
- `fuse_len=600`
- `stft_norm=n0`
- `stft_encoder_type=conv2d`
- time_only 复用同 seed E1a' wrapper 基线；dual 只切 `stft_high_hz` 与 `branch_mode`

8Hz N0 先补 5 个新 seed：`20260931`、`20260945`、`20260952`、`20260963`、`20260974`。
加上既有结果后，11 个预期 seed 中 9 个成功配对，2 个因 gate fail 无法配对：

| seed | time_only | dual 8Hz | 配对状态 | 原因 |
|---:|---|---|---|---|
| 20260700 | ok | ok | ok | - |
| 20260710 | ok | ok | ok | - |
| 20260837 | gate fail | ok | not paired | time_only 未通过 `checkpoint_gate(auto_direction<=0.5)` |
| 20260901 | gate fail | gate fail | not paired | 两臂均未通过 `checkpoint_gate(auto_direction<=0.5)` |
| 20260911 | ok | ok | ok | - |
| 20260920 | ok | ok | ok | - |
| 20260931 | ok | ok | ok | - |
| 20260945 | ok | ok | ok | - |
| 20260952 | ok | ok | ok | - |
| 20260963 | ok | ok | ok | - |
| 20260974 | ok | ok | ok | - |

8Hz 同 seed 成功配对差值（dual - time_only）汇总：

| 口径 | n | 方向 | 平均差值 |
|---|---:|---|---:|
| `frac_gt_1` | 9 | 9/9 为负 | **-2.34pp** |
| `frac_gt_2` | 9 | 8/9 为负 | **-1.87pp** |
| mean | 9 | 9/9 为负 | **-0.120** |
| median | 9 | 9/9 为负 | **-0.033** |
| `band_limited_corr` | 9 | 9/9 为正 | **+0.028** |
| `best_lag_corr` | 9 | 9/9 为正 | **+0.030** |

逐 seed 看，8Hz 的 `frac_gt_1` 没有任何反向 seed；新增的 `20260931/20260952`
time_only 长尾更重，dual 把 `frac_gt_1` 分别拉低 `5.57pp`、`4.60pp`。这说明 STFT 的收益
确实主要在长尾谐波误拣，而不是靠 median 或少数窗口偶然拉动。`frac_gt_2` 唯一反向 seed
仍是旧的 `20260911`（`+0.56pp`），但整体均值和 8/9 方向已经足够稳。

随后做频带稳健性复核：只跑 dual 的 3Hz/12Hz，复用相同 seed 的 8Hz time_only 基线
（time_only 分支不建 STFT 分支，频带参数不影响其计算图）。四个 seed 全部成功：
`20260700`、`20260710`、`20260911`、`20260920`。

| 频带 | n | Δ frac_gt_1 | 方向 | Δ frac_gt_2 | 方向 | Δ mean | Δ median | Δ band corr | Δ best-lag corr |
|---:|---:|---:|---|---:|---|---:|---:|---:|---:|
| 3Hz | 4 | **-1.29pp** | 4/4 负 | **-0.86pp** | 4/4 负 | -0.082 | -0.025 | +0.027 | +0.029 |
| 8Hz | 9 | **-2.34pp** | 9/9 负 | **-1.87pp** | 8/9 负 | -0.120 | -0.033 | +0.028 | +0.030 |
| 12Hz | 4 | **-1.26pp** | 3/4 负 | **-0.64pp** | 3/4 负 | -0.064 | -0.028 | +0.025 | +0.027 |

产物：

- `runs/e1_clean_patch600_deep_b1_8hz_detail.csv`
- `runs/e1_clean_patch600_deep_b1_8hz_grouped.csv`
- `runs/e1_clean_patch600_deep_b1_8hz_status.csv`
- `runs/e1_clean_patch600_deep_b1_8hz_paired_delta.csv`
- `runs/e1_clean_patch600_deep_b1_8hz_corr_summary.csv`
- `runs/e1_clean_patch600_deep_b1_8hz_corr_paired_delta.csv`
- `runs/e1_clean_patch600_deep_b1_3hz_detail.csv`
- `runs/e1_clean_patch600_deep_b1_3hz_grouped.csv`
- `runs/e1_clean_patch600_deep_b1_3hz_status.csv`
- `runs/e1_clean_patch600_deep_b1_3hz_paired_delta.csv`
- `runs/e1_clean_patch600_deep_b1_3hz_corr_summary.csv`
- `runs/e1_clean_patch600_deep_b1_3hz_corr_paired_delta.csv`
- `runs/e1_clean_patch600_deep_b1_12hz_detail.csv`
- `runs/e1_clean_patch600_deep_b1_12hz_grouped.csv`
- `runs/e1_clean_patch600_deep_b1_12hz_status.csv`
- `runs/e1_clean_patch600_deep_b1_12hz_paired_delta.csv`
- `runs/e1_clean_patch600_deep_b1_12hz_corr_summary.csv`
- `runs/e1_clean_patch600_deep_b1_12hz_corr_paired_delta.csv`

解释：

1. **E1 patch 结论可以收口为正**：8Hz 在 9 个成功配对上 `frac_gt_1` 9/9 为负，且 mean、
   median、corr 全部同向改善。这个结果已经越过“单 seed 长尾偶然性”和“均值脆弱口径”的风险。
2. **频带不是单点挑选**：3Hz/12Hz 的 4 seed 复核方向仍为负，尤其 3Hz 两个误拣率阈值均 4/4
   改善；12Hz 略弱，但总体仍负。8Hz 目前是三档中信号最强的 N0 频带。
3. **epoch=100 暂无必要**：本轮长训练 seed 的最佳点多在 20~44 epoch；个别跑满 50 的 run
   后段已平台化或回退，不是持续刷新上限导致的欠训练。为保持 B1 口径一致，没有混入 100 epoch
   变体。
4. **剩余主要风险转向训练稳定性**：`20260837/20260901` 的缺失仍来自 gate fail/极性多稳态，
   不是 STFT 频带选择问题。后续 B1 或方法开发应优先减少反相 basin，而不是继续扩大 E1 频带网格。

### 疑点 1 续四：gate-fail 根因核查（初始化极性多稳态，非样本分布）

针对干净 patch 复核中的 3 个 gate fail（`time_only` `20260837/20260901`、`dual` `20260901`），
翻它们的 `train_history.csv`（fail run 不存 checkpoint/metrics，但保留逐 epoch 的 val 指标）
与成功 run 对比，定位根因。`val_signed_corr` 映射关系约为 corr ≈ 1 − 2·signed_corr。

| run | val_signed_corr 全程 | ≈ corr | 性质 |
|---|---|---:|---|
| OK `time_only` 20260700 | 0.267~0.286（平） | +0.46 | 正极性，epoch1 即锁定 |
| OK `dual` 20260837 | 0.242~0.283（平） | +0.50 | 正极性 |
| FAIL `time_only` 20260837 | 起 1.476 → 收 0.762 | −0.52 | 反极性，0/36 epoch 越过 0.5 |
| FAIL `time_only` 20260901 | 起 1.573 → 收 0.835 | −0.67 | 反极性，0/50 |
| FAIL `dual` 20260901 | 起 1.381 → 收 0.775 | −0.55 | 反极性，0/30 |

根因 = **初始化决定的极性多稳态，不是样本分布**。三条证据排除样本分布假设：

1. **数据划分跨所有 run 固定**（`train_sample_seed=20260610`、`val_sample_seed=20260611`）。
   fail 与 ok 看同一个 val 集，唯一变量是 `training.seed`（只控制初始化和 shuffle 顺序，
   不改变 val 内容）。同一份样本分布既出 +0.46 又出 −0.52，样本分布不可能是 fail/ok 的区分变量。
2. **双峰、gate 阈值 0.5 落在空隙**：ok 全部 firmly 在 0.27（corr +0.46），fail 全部 firmly 在
   0.75+（corr −0.5~−0.67），中间没有任何 run。门控不是误杀边界 run，而是干净隔开两个 basin。
3. **fail run 在 epoch1 即起步于 signed_corr 1.4~1.57（corr≈−0.9，近完美反相）**，训练只是
   精修这个反相解、改不动符号。极性在初始化那一刻就已确定。

结论与影响：

- WeakSyncLoss 的符号约束不足以在训练初期破除「正相 / 反相」两个稳定解，初始化（`training.seed`）
  决定落入哪个，约 1/3 seed 落入反相、被 gate 正确拦掉。这与历史 `stft_only` magnitude 极性盲、
  m7/m8 signed_cosine 课程实验是同一条极性不稳定主线。
- **对 E1 结论是利好**：gate 干净隔开两 basin、fail 是真反相、配对统计只建立在真正同极性的
  run 上，patch 的正结论不受影响（B1 补 seed 后已是 9/9 同向，见“疑点 1 续三”及里程碑）。
- 局限：fail run 无逐窗预测，无法直接区分「全局翻转 vs 子集病态样本」；但 train_history 已强烈
  指向全局翻转（epoch1 即近完美反相、聚合 corr 均匀 −0.5）。若要坐实，需重跑 1 个 fail seed、
  临时关 gate dump 预测，边际较低，暂不做。
- 降低翻转率属 loss/训练侧干预（极性锚定 / signed 项课程），与 E1 收尾是两条线。B1 已完成、
  E1 patch 已闭环，此项现为当前主攻方向（见“下一步 / 优先级 3”）。

### E1-C：早期 signed warmup 极性稳定性验证

目的：验证“epoch1 基本决定极性”这个观察下，训练前 6 个 epoch 临时提高 signed 项权重，是否能把
反相 basin 填浅并降低 gate fail。此轮只看极性稳定性，不重判 STFT 增益；若 warmup 有效，后续再用
同一 E1a' vs E1b 配对口径复核 `Δf1` 是否仍稳定为负。

固定配置：

- 模型：`patch_mixer1d + time_stft_dual1d + conv2d STFT encoder + N0 8Hz + deep FusionHead + fuse_len=600`。
- 分支：`branch_mode=dual`。
- 训练：`signed_corr_weight=0.2`、`checkpoint_gate.metric=auto_direction`、`checkpoint_gate.max=0.5`。
- seeds：`20260837`、`20260901`、`20260952`、`20260963`、`20260974`。
- 输出：
  - 明细：`runs/e1_c_polarity_status.csv`
  - 聚合：`runs/e1_c_polarity_grouped.csv`

编号：

| 编号 | run_root | 配置含义 |
|---|---|---|
| E1-C0 | `runs/e1_c0_polarity_baseline` | B1 正常无 warmup 对照，保留 `signed_corr_weight=0.2` |
| E1-C1 | `runs/e1_c1_polarity_warmup_light` | 轻 warmup：`signed_corr 0.6→0.2`，`signed_cosine 0.1→0.0`，epoch 1-6 线性退火 |
| E1-C2 | `runs/e1_c2_polarity_warmup_heavy` | 重 warmup：`signed_corr 1.0→0.2`，`signed_cosine 0.3→0.0`，epoch 1-6 线性退火 |

聚合结果：

| 编号 | 有效 run | gate fail | final `val_signed_corr>0.5` | final `val_signed_corr` mean | median |
|---|---:|---:|---:|---:|---:|
| E1-C0 | 4/5 | 1/5 | 1/5 | 0.351672 | 0.246663 |
| E1-C1 | 5/5 | 0/5 | 0/5 | 0.246688 | 0.246003 |
| E1-C2 | 5/5 | 0/5 | 0/5 | 0.247509 | 0.247789 |

逐 seed 结果：

| seed | E1-C0 | E1-C1 | E1-C2 | 备注 |
|---:|---|---|---|---|
| 20260837 | pass，final 0.243319 | pass，final 0.244446 | pass，final 0.243670 | 三臂均正极性 |
| 20260901 | **gate fail**，final 0.775296，best 0.746049 | pass，final 0.244373 | pass，final 0.247789 | 关键样本：warmup 把 C0 反相 basin 拉回正极性 |
| 20260952 | pass，final 0.246663 | pass，final 0.246003 | pass，final 0.245433 | 三臂均正极性 |
| 20260963 | pass，final 0.248901 | pass，final 0.248711 | pass，final 0.250222 | C0 epoch1 反相很强，但第 2 epoch 起回正；warmup 也稳定 |
| 20260974 | pass，final 0.244183 | pass，final 0.249905 | pass，final 0.250433 | 三臂均正极性 |

判定：

1. **早期 signed warmup 对极性稳定性有效**：在 5 个高风险/补充 seed 上，C0 出现 1/5 gate fail；
   C1/C2 都是 5/5 通过，且 final `val_signed_corr` 全在 `0.244~0.252` 的正极性区间。
2. **轻 warmup 已足够**：C1 和 C2 的 final `val_signed_corr` 均值几乎相同（`0.246688` vs
   `0.247509`），重 warmup 没有明显额外收益；后续默认优先用 E1-C1，避免不必要地抬高早期 loss。
3. **关键样本是 seed `20260901`**：C0 从 epoch1 到 early stop 都处于反相区间（best 仍
   `0.746049`，未过 `0.5` gate），C1/C2 均被拉回正极性。这直接支持“前期 signed 项可破 basin”
   这个机制判断。
4. **本轮不是 STFT 增益复核**：只证明 warmup 降低 dual 分支极性失败率；还不能自动替代 B1 的
   E1a' vs E1b `Δf1` 结论。若要把 warmup 写进推荐训练配方，应补一轮 time_only/dual 配对，
   确认 `frac_gt_1` 净增益仍稳定为负。

### E1-D：warmup 口径下的 STFT 增益复测

目的：验证 E1-C1 轻 warmup 在降低极性失败后，是否仍保留 patch 口径下的 STFT 净增益。
本轮使用代表性 6 seed，不再按 gate fail 额外冗余；结果实际得到 6/6 完整配对。

固定配置：

- 模型：`patch_mixer1d + time_stft_dual1d + conv2d STFT encoder + N0 8Hz + deep FusionHead + fuse_len=600`。
- 分支：`branch_mode=time_only` vs `branch_mode=dual`。
- warmup：E1-C1 轻口径，`signed_corr 0.6→0.2`、`signed_cosine 0.1→0.0`，epoch 1-6 线性退火。
- seeds：`20260700`、`20260710`、`20260837`、`20260911`、`20260920`、`20260901`。
- 输出：
  - 明细：`runs/e1_d_warmup_detail.csv`
  - 聚合：`runs/e1_d_warmup_grouped.csv`
  - 配对 delta：`runs/e1_d_warmup_paired_delta.csv`
  - 相关性汇总：`runs/e1_d_warmup_corr_summary.csv`
  - 相关性配对 delta：`runs/e1_d_warmup_corr_paired_delta.csv`
  - run 状态：`runs/e1_d_warmup_status.csv`

运行状态：

| 分支 | 完整 run | 不完整 run | 说明 |
|---|---:|---:|---|
| `time_only` | 6 | 0 | 6 个代表性 seed 全部有 `metrics.csv` |
| `dual` | 6 | 1 | 6 个代表性 seed 全部有完整 `metrics.csv`；另有 `20260911` 一条中断残留目录，仅有 `train_history.csv` |

聚合结果：

| 分支 | n | mean | median | p95 | `frac_gt_1` | `frac_gt_2` |
|---|---:|---:|---:|---:|---:|---:|
| `time_only` | 6 | 0.634631 | 0.239052 | 2.592558 | 0.147726 | 0.067165 |
| `dual` | 6 | 0.517425 | 0.188671 | 2.114233 | 0.125421 | 0.053396 |

配对 delta（`dual - time_only`）：

| 指标 | 平均 delta | 方向 |
|---|---:|---|
| mean | -0.117206 | 改善 |
| median | -0.050381 | 改善 |
| p95 | -0.478325 | 改善 |
| max | -0.346793 | 改善 |
| `frac_gt_1` | -0.022305 | 改善（-2.23pp） |
| `frac_gt_2` | -0.013769 | 改善（-1.38pp） |

逐 seed 的 `frac_gt_1` 配对结果：

| seed | `time_only` | `dual` | delta |
|---:|---:|---:|---:|
| 20260700 | 0.149533 | 0.152150 | +0.002617 |
| 20260710 | 0.121495 | 0.124860 | +0.003364 |
| 20260837 | 0.151776 | 0.107290 | -0.044486 |
| 20260911 | 0.156262 | 0.130841 | -0.025421 |
| 20260920 | 0.147664 | 0.122617 | -0.025047 |
| 20260901 | 0.159626 | 0.114766 | -0.044860 |

相关性配对 delta（`dual - time_only`，逐窗口均值后按 seed 配对）：

| 指标 | 平均 delta | 方向 |
|---|---:|---|
| `envelope_corr` | +0.030545 | 改善 |
| `relative_envelope_corr` | +0.030534 | 改善 |
| `spectrum_similarity` | +0.001775 | 改善 |
| `band_limited_corr` | +0.027440 | 改善 |
| `best_lag_corr` | +0.030810 | 改善 |

判定：

1. **warmup 不伤 STFT 净增益**：E1-D 的 `Δfrac_gt_1=-2.23pp`，接近 B1 冻结配方的
   `-2.34pp`，并且 mean、median、p95、`frac_gt_2` 全部同向改善。
2. **gate fail 率显著下降**：6 个代表性 seed 的两臂均有完整 `metrics.csv`。历史上
   `20260837/20260901` 容易因反相被 gate 拦；本轮在 warmup 下全部给出有效配对。
3. **相关性没有 trade-off**：`band_limited_corr` 与 `best_lag_corr` 平均 delta 为正，说明
   warmup 口径下 STFT 降低谐波误拣并未牺牲当前波形相关性。
4. **局部反例不推翻总体结论**：`20260700/20260710` 的 `frac_gt_1` 小幅反向（+0.26pp、+0.34pp），
   但幅度远小于后 4 个 seed 的改善，聚合和相关性均支持 warmup 作为推荐口径。
5. **增益集中在高风险 seed（机制性观察）**：改善大头在 `20260837/20260901/20260911/20260920`
   （`Δf1` -2.5~-4.5pp），这些 seed 的 `time_only` `frac_gt_1` 本就偏高（0.15~0.16，即更靠近反相/
   高误拣工作区）；`20260700/20260710` 的 `time_only` 本就低（~0.12~0.15），STFT 边际自然小。
   即 warmup 的价值不是“普涨增益”，而是**把原本高风险的 seed 救回到 STFT 能发挥作用的工作区**，
   使 STFT 增益在更宽的 seed 范围内可复现，而不只在好运 seed 上——这反而是 warmup 该进推荐口径
   的理由，而非“700/710 几乎无差”所暗示的“warmup 没用”。

结论：E1-C1 轻 warmup 同时满足“修极性”和“不伤增益”，后续 patch STFT 主线可默认采用该
warmup；历史 B1 冻结结果仍保留为无 warmup 对照基线。

> **patch STFT 推荐训练配方（定稿）**：
> `patch_mixer1d + time_stft_dual1d + conv2d STFT encoder + N0 8Hz + deep FusionHead + fuse_len=600`，
> 叠加极性 warmup：`signed_corr` schedule linear `0.6→0.2`（ep1-6，ep6 后常驻 0.2），
> 以及 `signed_cosine` schedule linear `0.1→0.0`（ep1-6，ep6 后归零）；方向门控
> `auto_direction/max=0.5` 不变。已写入 `configs/tho_research_v2.yaml` 默认 loss 段
> （注意：仅对本数据/patch 路线验证过，multiscale 等其他 backbone 未验证 warmup 安全性）。

### B2-0：native_inject 原生解码融合复测

目的：验证 patch 双分支是否必须避开 `concat -> FusionHead` 通用解码头，改为在 patch token
栅格上做零初始化加性注入，再走 `PatchMixer1D` 原生 `decode_from_features`。判定门来自计划：
`time_only` 误拣率应回到原生约 `11%`；`dual - time_only` 的 `Δfrac_gt_1` 应不弱于 E1-D
基准 `-2.34pp`；gate fail 不劣于 E1-D。

固定配置：

- 模型：`patch_mixer1d + time_stft_dual1d + fusion_mode=native_inject + conv2d STFT encoder + N0 8Hz`。
- 解码：`time_only` 走原生 patch 解码；`dual` 在 token 特征上加 `stft_proj(stft_feats)` 后走同一原生解码。
- warmup：沿用 E1-C1 默认轻 warmup。
- seeds：`20260700`、`20260710`、`20260837`、`20260901`、`20260911`、`20260920`。
- 输出：
  - 明细：`runs/b2_native_detail.csv`
  - 聚合：`runs/b2_native_grouped.csv`
  - 配对 delta：`runs/b2_native_paired_delta.csv`
  - 相关性配对 delta：`runs/b2_native_corr_paired_delta.csv`
  - run 状态：`runs/b2_native_status.csv`

运行状态：

| 分支 | 完整 run | 不完整 run | 说明 |
|---|---:|---:|---|
| `time_only` | 6 | 6 | 6 个完整结果；另有 6 个早期误启动/中断残留目录，未进入配对统计 |
| `dual` | 6 | 0 | 6 个代表性 seed 全部有 `metrics.csv` |

聚合结果：

| 分支 | n | mean | median | p95 | `frac_gt_1` | `frac_gt_2` |
|---|---:|---:|---:|---:|---:|---:|
| `time_only` | 6 | 0.600554 | 0.195961 | 2.621072 | 0.147913 | 0.067539 |
| `dual` | 6 | 0.554411 | 0.190620 | 2.220951 | 0.133956 | 0.057695 |

配对 delta（`dual - time_only`）：

| 指标 | 平均 delta | 方向 |
|---|---:|---|
| mean | -0.046143 | 改善 |
| median | -0.005340 | 改善很小 |
| p95 | -0.400120 | 改善 |
| `frac_gt_1` | -0.013956 | 改善（-1.40pp） |
| `frac_gt_2` | -0.009844 | 改善（-0.98pp） |

逐 seed 的 `frac_gt_1` 配对结果：

| seed | `time_only` | `dual` | delta |
|---:|---:|---:|---:|
| 20260700 | 0.146916 | 0.124860 | -0.022056 |
| 20260710 | 0.168598 | 0.121121 | -0.047477 |
| 20260837 | 0.126355 | 0.142430 | +0.016075 |
| 20260901 | 0.135327 | 0.129720 | -0.005607 |
| 20260911 | 0.179813 | 0.149907 | -0.029907 |
| 20260920 | 0.130467 | 0.135701 | +0.005234 |

相关性配对 delta（`dual - time_only`，逐窗口均值后按 seed 配对）：

| 指标 | 平均 delta | 方向 |
|---|---:|---|
| `envelope_corr` | -0.004363 | 小幅下降 |
| `relative_envelope_corr` | -0.002451 | 小幅下降 |
| `spectrum_similarity` | -0.000430 | 基本持平 |
| `band_limited_corr` | -0.001031 | 基本持平 |
| `best_lag_corr` | -0.000115 | 基本持平 |

判定：

1. **gate 稳定性通过**：6 个 seed 的两臂全部产出完整 `metrics.csv`，且所有 run 的
   `max_val_signed_corr` 都未超过 `0.5`，不劣于 E1-D。
2. **time_only 没回到原生约 11%**：`frac_gt_1=14.79%`，比旧 wrapper 约 `22%` 明显好，
   但仍高于计划门槛里的原生 `~11%`。架构前向等价不能自动保证训练结果完全回到原生分布。
3. **STFT 净增益门未通过**：`Δfrac_gt_1=-1.40pp`，4/6 seed 为负，方向有改善但弱于
   E1-D 的 `-2.23pp` 和 B1 的 `-2.34pp`。因此 B2-0 不能作为进入 B2-1 gating 的 substrate。
4. **机制含义（B2-0 设计前提被部分证伪）**：`native_inject` 消掉通用解码头后，并没有放大
   STFT 信息增益，反而双双变弱（`time_only` 误拣率没回到原生约 `11%`、净增益从 `-2.34pp`
   缩到 `-1.40pp`）。这**部分推翻了立 B2-0 时的核心前提**——当时据 multiscale 的 296-318 行
   断言「`concat → deep FusionHead` 是纯 handicap」。实测在 patch 上：该头**确有一部分是
   handicap**（`time_only` 从约 `22%` 降到 `14.79%` 成立），但**另一部分是在帮 STFT**：
   deep 头在 `fuse_len=600` 超采样栅格上的 `k3` 卷积给了 STFT 一个跨 token、可学习的混合
   空间；`native_inject` 把 STFT 压回 140 token 栅格做 `1×1` 加性注入，信道更窄更「原生」，
   反而削弱了 STFT 的发挥。教训：「substrate 更原生干净」与「对 STFT 更有利」不是同一件事，
   不能混为一谈。按预设判定门，此处应停下，不盲目追加 gating/FiLM。

结论：B2-0 是有效的负结果。它证明原生解码注入能稳定训练，但没达到“头修正 + STFT 净增益更强”
的目标；后续 patch 主线仍以 E1-D 定稿配方为准，不进入 B2-1 gating。

实现备注（零初始化注入，标准 ReZero）：`native_inject` 的 `stft_proj` 末层零初始化保证 dual
在 init 时输出逐元素等价原生解码。其副作用是**首个 backward 里 STFT encoder 梯度为 0**
（`∂out/∂stft_encoder = decode' · stft_proj.weight = 0`）——但这是零初始化残差（ReZero/
LayerScale/Fixup 一类）的**标准无害行为**：`stft_proj.weight` 自身首步即获非零梯度并离开
零点，STFT encoder 从**第二步**起正常学习，「延迟一步」在数千 step 训练里可忽略。早期版本曾
加过一个零值梯度桥试图让 STFT encoder 首步即拿梯度，但核查发现：该桥注入的是绕过 `stft_proj`
的**代理方向**（非真实 `∂loss/∂out`），会污染 Adam 首步矩，副作用比它解决的「晚一步」更实，
已删除。因此 B2-0 的负结果与「首步梯度」无关，是 substrate 本身的性质。

### E2c：分频带能量 STFT 表征探针

目的：在 B2-0 的 `native_inject` substrate 上，单独替换 STFT encoder 输入表征，验证“完整
2D 频谱图是否过宽/过噪”，以及分频带能量序列是否能在同一融合路径下更好地降低谐波误拣。
该实验只比较 `dual` 臂：E2c `bandenergy dual` 对 B2-0 `conv2d fullband dual` 同 seed 配对，
不和 E1-D 的 concat 强基线混口径。

固定配置：

- 模型：`patch_mixer1d + time_stft_dual1d + fusion_mode=native_inject + N0 8Hz`。
- STFT 表征：`encoder_type=bandenergy`，默认 5 个重叠频带，带内均值池化后得到 `(B,5,T)`，
  再经 `conv1d` 编码到 `stft_out_channels=16`。
- warmup：沿用 E1-C1 默认轻 warmup。
- seeds：`20260700`、`20260710`、`20260837`、`20260901`、`20260911`、`20260920`。
- 输出：
  - 明细：`runs/e2c_band_energy_detail.csv`
  - 聚合：`runs/e2c_band_energy_grouped.csv`
  - 对 B2-0 配对 delta：`runs/e2c_band_energy_vs_b2_paired_delta.csv`
  - 全频带 sanity：`runs/e2c_fullband_sanity/20260623_110446_645152`

sanity：E2c 脚本里的 `conv2d fullband dual` 复现 B2-0 同 seed `20260700`，`frac_gt_1`
为 `0.127477`，B2-0 为 `0.124860`，差值 `+0.26pp`；`frac_gt_2` 为 `0.047103`，
B2-0 为 `0.048224`，差值 `-0.11pp`。差异很小，说明复用 B2-0 fullband 桩的口径成立。

E2c `bandenergy dual` 聚合结果：

| n | mean | median | p95 | `frac_gt_1` | `frac_gt_2` |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.571132 | 0.190173 | 2.463660 | 0.139003 | 0.062555 |

对 B2-0 `fullband dual` 的同 seed 配对 delta（E2c - B2-0）：

| seed | E2c `frac_gt_1` | B2-0 `frac_gt_1` | delta | E2c median | B2-0 median | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 20260700 | 0.109907 | 0.124860 | -0.014953 | 0.185510 | 0.191969 | -0.006460 |
| 20260710 | 0.111402 | 0.121121 | -0.009720 | 0.184280 | 0.181163 | +0.003117 |
| 20260837 | 0.128224 | 0.142430 | -0.014206 | 0.192181 | 0.185510 | +0.006671 |
| 20260901 | 0.132710 | 0.129720 | +0.002991 | 0.185510 | 0.200673 | -0.015164 |
| 20260911 | 0.161495 | 0.149907 | +0.011589 | 0.191537 | 0.190270 | +0.001267 |
| 20260920 | 0.190280 | 0.135701 | +0.054579 | 0.202020 | 0.194137 | +0.007883 |

配对汇总：

| 指标 | 平均 delta | 方向 |
|---|---:|---|
| mean | +0.016721 | 变差（4/6 seed 为负，但两个补 seed 长尾过大） |
| median | -0.000447 | 基本持平（2/6 seed 为负） |
| `frac_gt_1` | +0.005047 | 变差（3/6 seed 为负，+0.50pp） |
| `frac_gt_2` | +0.004860 | 变差（4/6 seed 为负，但均值被补 seed 长尾反转） |

gate：6 个 E2c run 的 `max_val_signed_corr` 最大值为 `0.221028`，未触发方向 gate 失败；
补 seed 的劣化不是极性失败造成的。

判定：E2c 补到 6 seed 后未过判定门。前 4 seed 的 `-0.90pp` 是弱正小样本信号，
但补入 `20260911/20260920` 后变成 `Δfrac_gt_1=+0.50pp`，且 `20260920` 明显反向
（`+5.46pp`）。因此不能收“分频带能量赢 fullband”，也不需要继续补 seed。E2c 作为
B2-0 substrate 上的表征探针被否决；这会连带降低 E2b/E2d 这类“压缩/改写 STFT 表征”
路线优先级。E1-D concat-deep 仍是当前 patch 主线定稿配方。

### 疑点 2：方向门控（核查后推翻“过严/幸存者偏差”）

gate `auto_direction/max=0.5` 解析为 `val_signed_corr ≤ 0.5`，等价要求 corr≥0（极性不反）。
翻 gate failure run 的 `val_signed_corr` 轨迹：

| run | val_signed_corr | 含义 |
|---|---:|---|
| `stft_only` patch ×3（全 fail） | 全程 `0.99`（≈corr −0.98） | 压倒性极性反向 |
| `time_only` patch seed `20260837`（fail） | `0.755`（≈corr −0.5） | 真半反向，非边界误杀 |
| `time_only` patch 另 2 seed | `0.27`（≈corr +0.46） | 正向，过 gate |
| `time_only` multiscale ×3 | `0.20`（≈corr +0.6） | 全正向，全过 gate |

- **门控没有误杀任何 run，初版“过严 / 幸存者偏差”担心不成立。**
- E1c `stft_only` 全 fail 的真因：magnitude STFT 丢相位、根本学不出极性（corr≈−0.98），
  门控正确拦截。这是设计层结论（magnitude 单分支极性盲），印证设计判断，**不是 bug 也不是过严**。
  注：`stft_only` 实跑 3 个（均 seed `20260700`）即 fail-fast，其余 manifest 配置按预期跳过；
  初版“18 个 gate failure”应理解为“3 个实跑全 fail + 其余预期跳过”。
- 零星 `time_only`/`dual` gate failure（如 patch seed `20260837`，`0.755`）是双分支训练的
  seed 级极性不稳定（真反向），非门控阈值误杀（`0.755` 离 `0.5` 门槛有明显距离）。
- 推论：被门控拦掉的都是真坏 run，计入均值反而错——“STFT 弱信号”**不是门控人为美化的**，
  这让弱信号结论更可信，而非更可疑。

## 当前决策

- 主候选更新为：`patch_mixer1d + conv2d + N0 8Hz + deep + fuse_len=600`，训练侧默认采用
  E1-C1 轻 warmup。这是当前 E1 最干净的正结论配置：无 warmup B1 的 9 个成功配对里
  `frac_gt_1` 9/9 下降，且 corr 9/9 提升；E1-D 进一步确认 warmup 口径下 6 个代表性 seed
  仍保持 `Δfrac_gt_1=-2.23pp`，且 corr 平均提升。
- `patch_mixer1d + conv2d + N0 3Hz/12Hz` 保留为频带稳健性证据，不作为优先候选：
  3Hz 4/4 改善、12Hz 3/4 改善，但信号强度弱于 8Hz。
- 旧 `N1 3Hz` 结果保留为归一化候选线索，但未进入本轮干净配对 B1；后续若重启 norm 扫描，
  必须复用同样的 E1a' 配对与误拣率口径。
- `multiscale_decomp_mixer1d + STFT`：**不记为“STFT 不适用”**。崩坏已定位为 wrapper 包装路径的
  优化多稳态抬高谐波误拣率（非 STFT、**非重建退化**、极性正常；中位数与原生持平 `0.18~0.21`）。
  干净复核显示 dual 相对 time_only 在 5 个成功配对上平均降低 `frac_gt_1` 约 `1.07pp`，
  有长尾收益趋势；但 `dual/20260710` gate failure、`20260700` 反向，暂不能给强结论。
- `patch_mixer1d + STFT`：**给出当前 E1 的正向结论**。deep+`fuse_len=600`、8Hz N0 的
  9 个成功配对上，dual 的 `frac_gt_1` 全部低于 time_only，平均 `-2.34pp`；mean/median
  同向改善，且 `band_limited_corr` / `best_lag_corr` 也 9/9 提升。无 warmup 冻结口径的限制从
  “样本量不足”转为“仍有 gate fail/极性多稳态”；E1-D 显示轻 warmup 已能显著缓解该问题，
  且不伤 STFT 净增益。
- `native_inject` 原生解码注入：**不进入当前主线**。B2-0 证明它能稳定训练（6/6 配对、0 gate
  fail），但 `time_only` 误拣率未回到原生约 `11%`，且 STFT 净增益只有 `Δfrac_gt_1=-1.40pp`，
  弱于 E1-D/B1 的 `-2.23~-2.34pp`。因此不在这个 substrate 上继续投 gating。
- `stft_only`：**确认在 magnitude 口径下不可用**——极性盲已坐实（corr≈−0.98），
  要用 STFT 单分支必须引入相位（E4/CWT），否则放弃单分支路线。
- 门控保持 `auto_direction/max=0.5`：核查证明它正确拦截真反向，不放宽。
- 暂不进入 CWT/SST 或更复杂频域前端；E1 已证明 magnitude STFT 在 patch wrapper 上有用，
  下一阶段优先做训练稳定性与融合使用方式，而不是继续扩大频域前端复杂度。

## 下一步

优先级 0（在干净口径下重判 STFT 增益，已完成第一轮）：

1. 第一轮已按 `fusion_decoder=lite`、`fuse_len=18000`、E1a' vs E1b、误拣率判据完成 6 seed 复核。
2. 当前证据：dual 在 5 个成功配对上平均降低 `frac_gt_1` `1.07pp`，但有 1 个反向 seed 和
   1 个 dual gate failure。
3. 下一步若继续 multiscale：优先补 2 个 seed（例如 `20260931`、`20260945`）或先处理 dual
   优化稳定性，再确认 `frac_gt_1` 净收益是否稳定为负。

优先级 1（patch_mixer 干净配对复核，已完成 B1，可收口）：

1. 当前证据：8Hz N0 的 9 个成功配对中 `Δf1` 9/9 为负，均值 `-2.34pp`；mean/median
   也 9/9 为负，corr 9/9 为正。这支持“STFT 在 patch 上降低谐波误拣”的 E1 正向结论。
2. 频带稳健性：3Hz 的 `Δf1` 4/4 为负，12Hz 的 `Δf1` 3/4 为负；8Hz 是当前三档中信号最强者。
3. 限制：B1 无 warmup 冻结口径中，time_only `20260837/20260901` gate fail，
   dual `20260901` gate fail；B1 已不缺配对数，但 wrapper 极性多稳态仍会造成约 1/3 run
   无结果。E1-D 显示轻 warmup 可把这类代表性 seed 拉回完整配对，后续 patch 主线默认采用
   E1-C1 口径（见优先级 3）。
4. trade-off 已复核：8Hz 有效配对里 `band_limited_corr` / `best_lag_corr` 都是 9/9 提升，
   3Hz/12Hz 也都是 4/4 提升。这两列已确认无 trade-off，作为后续固定 sanity。
   融合方法优化移至优先级 4（极性稳定后再做）。

优先级 2（multiscale 后续，暂缓）：

1. multiscale 干净复核已完成第一轮，证据偏弱（5 配对 `Δf1` 均值 `-1.07pp` 被单 seed 主导、
   `dual/20260710` gate failure）。在 patch 出结论前不再追加 multiscale seed，避免边际投入过低。
2. 若后续仍要追 multiscale：优先诊断 dual 的极性翻转/gate failure（加 STFT 抬高了训练不稳），
   这是 STFT 路线的共性风险。

优先级 3（极性多稳态修复，E1-C/E1-D 已完成）：

1. 根因已查清（见“疑点 1 续四”）：约 1/3 seed 因初始化落入反相 basin、被 gate 拦掉。
   病根在 WeakSyncLoss 的符号约束不足（正相/反相近简并），**不是 init**——标准 init 方法
   符号无关、治不动；靠 init 偏向正相需要数据相关末层 init 这类脆弱 hack，不采用。
2. 正确杠杆在 **loss 侧**：早期重加权 signed 项 / 极性锚定 warmup，把反相 basin 填浅。
   **这与历史 m7/m8 signed_cosine 课程是同一条线**，应复用而非另起。
3. E1-C 第一轮已按单独实验线完成：C0 正常对照 1/5 gate fail，C1/C2 warmup 均 5/5 通过。
   轻 warmup（C1：`signed_corr 0.6→0.2`、`signed_cosine 0.1→0.0`，epoch 1-6）已足够，
   见“E1-C：早期 signed warmup 极性稳定性验证”。
4. 这是改优化景观的改动，会让已完成的 E1a'/E1b 配对不可比，故必须保留现有冻结配方的
   E1 结论作为对照基线；不回改 B1 的历史结果。
5. E1-D 已完成合入推荐配方前的增益复测：6 个代表性 seed 全配对，`Δfrac_gt_1=-2.23pp`、
   `Δfrac_gt_2=-1.38pp`，band/best-lag corr 平均提升。因此轻 warmup 可进入后续 patch
   STFT 推荐训练口径。
6. 剩余验收目标从“是否能推荐 warmup”转为“扩大到后续实验矩阵时是否仍保持低 gate fail”。
   后续新结构或新频带仍应保留同 seed 的 time_only 对照，避免把结构收益和 STFT 信息增益混在一起。
7. **已落地**：定稿 warmup（`signed_corr 0.6→0.2` + `signed_cosine 0.1→0.0`，ep1-6）已写入
   `configs/tho_research_v2.yaml` 默认 loss 段，成为 patch STFT 主线默认口径。仅对本数据/patch
   路线验证过；multiscale 等其他 backbone 未验证 warmup 安全性（按当前决策不再跟进 multiscale）。

优先级 4（patch 融合方法优化，**B2-0/E2c 后整体暂停**）：

1. 定位：当前 `concat → deep FusionHead` 已足以支撑 E1 正结论，融合优化是“能否从 STFT 榨出
   更多”的性能问题，**非 E1 必需**。预期边际收益不大且不确定（concat+有容量解码器本身是强基线）。
2. B2-0 已完成原生解码注入复测：`fusion_mode=native_inject` 得到 6/6 完整配对、0 gate fail，
   但 `Δfrac_gt_1=-1.40pp`，弱于 E1-D 的 `-2.23pp`，未过“进入 B2-1 gating”的门槛。
   这说明旧通用融合路径可能在 patch 上提供了额外可用容量，不能假设“更原生的解码”一定更干净。
3. 因此 **不继续 B2-1 gating**，除非先提出新的、能解释为何 gating 会弥补 B2-0 增益变弱的机制假设。
   当前默认回到 E1-D 定稿配方推进后续研究，而不是继续扩融合结构。E2c 补到 6 seed 后也未过门，
   说明“STFT 表征压缩/分带”在当前 B2-0 substrate 上不能作为现成重启入口。
4. 若以后重启融合优化，判定口径（强制）：每个新融合都跑 dual + time_only 两臂 × 多 seed，比较
   `(新融合 dual − 新融合 time_only)` 的 `Δf1` 是否比 concat 基准 `-2.34pp` 更负。
   **time_only 对照桩不可省**——它确保涨幅来自“更会用 STFT”而非“新结构在裸 backbone 上就更强”。
5. **patch gating/FiLM 扩结构线现整体封存**（不是只停 B2-1）：B2-0 已证明「换更原生解码」方向走不通，
   且没有现成的下一个候选优于 concat 强基线。重启扩结构的唯一合理入口是 B2-0 反转出的新假设——
   **STFT 增益依赖一个跨 token 的宽混合空间**（deep 头在 `fuse_len=600` 上恰好提供了它，
   `1×1` token 栅格注入提供不了）。即如要再投入，应验证「在保留原生解码的同时给 STFT 一条
   更宽的跨 token 通路」（例如 token 栅格上的 `k3`/多 token 感受野注入，而非 `1×1`），
   而不是直接上 gating/FiLM——后者并不针对这个被实测指认的瓶颈。在提出并接受这样的机制假设前，
   patch 主线一律以 E1-D 定稿配方为准。E2c 已单独作为表征侧探针否决，不等价于重启 gating/FiLM 扩结构线。

已闭环（无需再做）：方向门控核查——确认无误杀，stft_only 在 magnitude 口径下极性盲，
保持门控不放宽；multiscale `fuse_len`/decoder 对照 + 逐窗分布核查——已定位崩坏为 wrapper
包装路径的误拣率方差（非重建退化、非单一层结构），中位数与原生持平，后续按“优先级 0”的
干净口径重判，不再扩大 deep/lite 这类纯 decoder 结构对照。
