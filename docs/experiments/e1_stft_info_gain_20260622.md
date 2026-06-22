# E1 STFT 输入信息增益实验记录

## 结论摘要

本轮 E1 实验没有给出“STFT 输入带来稳定、全面信息增益”的强证据，但也**不能据此判定
“STFT 无用”**——容量桩 E1a' 暴露出至少一个与 STFT 无关的混淆因子尚未拆解，需先排除再定论。

- `multiscale_decomp_mixer1d` 的主指标崩坏**不是 STFT 造成的**：容量桩 E1a'（只有时序分支
  + 融合头、不含 STFT）的 peak-band MAE 已从 E1a 的 `0.483` 暴涨到 `1.325`，E1b 加入 STFT
  后（`1.19~1.43`）与 E1a' 同量级。崩坏发生在 STFT 进入之前，凶手是融合/上采样路径本身。
- `patch_mixer1d` 上 STFT 对主指标有正贡献：相对正确的对照基线 E1a'（`0.539`，已含融合头容量
  代价），E1b 8Hz（`0.474`）/ N1 3Hz（`0.473`）把主指标拉回并超过；但伴随波形相关性
  （band/best-lag corr）下降，trade-off 真实存在。
- 门控核查（见“疑点核查结果 / 疑点 2”）：方向门控**没有误杀**，初版“过严/幸存者偏差”担心
  已推翻——E1c `stft_only` 全 fail 是 magnitude STFT 极性盲（`val_signed_corr≈0.99`，corr≈−0.98）
  的必然结果，零星 time_only fail 是 seed 级真反向。被拦的都是真坏 run，弱信号结论不是门控美化的。

当前可保留的候选方向是 `patch_mixer1d + conv2d + N1 3Hz`（降级为待复核候选）。两个混淆因子
核查后：门控已澄清（无误杀，弱信号可信）；multiscale 崩坏不是 STFT 导致。**逐窗分布核查
（见“疑点 1 续”）整体修正了机制**：原生 E1a 与各 wrapper 变体的 peak-band **中位数**全在
`0.18~0.21`，不存在波形重建退化；全部差异都在**谐波误拣率（长尾）**。主因是「裸 backbone vs
wrapper 包装」带来的优化多稳态（误拣率方差），解码器结构（deep vs lite）只是次要因子；
peak-band **均值**因被 10~20% 误拣窗口主导而脆弱，不宜单独做 STFT 增益主判据。由此确定干净
判定路径：lite 头 + 以 E1a'（非原生 E1a）为基线 + 改用误拣率判据 + 加到 5~8 seed。
对 patch_mixer 路线，可在“STFT 相对 E1a' 有正贡献、但波形相关性下降”的口径下继续。

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
- canonical 头用 `lite`（消掉 deep 的次要加尾，白送的干净；按误拣率 lite≈k1_norm，均优于 k3）。

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

- 暂保留候选：`patch_mixer1d + conv2d + N1 3Hz`，降级为“待复核候选”。
- 保留不稳定对照：`patch_mixer1d + conv2d + N0 8Hz`，补充 seed 后不再视作强候选。
- `multiscale_decomp_mixer1d + STFT`：**不记为“STFT 不适用”**。崩坏已定位为 wrapper 包装路径的
  优化多稳态抬高谐波误拣率（非 STFT、**非重建退化**、极性正常；中位数与原生持平 `0.18~0.21`）。
  判 STFT 增益需改用「lite 头 + E1a' 基线 + 误拣率判据 + 5~8 seed」，在此之前继续暂停扩展。
- `stft_only`：**确认在 magnitude 口径下不可用**——极性盲已坐实（corr≈−0.98），
  要用 STFT 单分支必须引入相位（E4/CWT），否则放弃单分支路线。
- 门控保持 `auto_direction/max=0.5`：核查证明它正确拦截真反向，不放宽。
- 暂不进入 CWT/SST 或更复杂频域前端；patch_mixer 的 STFT 信号尚弱（有 trade-off），
  multiscale 的融合 bug 未排除，均未达到“STFT 输入稳定增益”的进入条件。

## 下一步

优先级 0（在干净口径下重判 STFT 增益）：

1. canonical 头改用 `fusion_decoder=lite`（消掉 deep 的次要加尾），`fuse_len` 用全分辨率。
2. 基线改用 E1a'（time_only wrapper），不再拿 E1b 比原生 E1a——wrapper 多稳态是共模噪声，
   只有同为 wrapper 的 E1a' 与 E1b 相减 `(E1b − E1a')` 才抵消。
3. 主判据改为谐波误拣率（逐窗 peak-band 误差 > 阈值的占比）+ median 作平稳性 sanity，
   均值仅作参考。STFT 若有用，收益大概率正在降低误拣率。
4. 加到 5~8 seed，把初始化方差压到噪声地板以下，再比 `(E1b − E1a')` 的误拣率。
5. 误拣率从现有各 run 的 `metrics.csv` 逐窗值后处理即可（`scripts/peak_band_misclass_rate.py`），
   不需改训练链路。

优先级 1（patch_mixer STFT 信号复核）：

1. 对 `patch_mixer1d + N1 3Hz` 继续做少量 seed 或独立 split 复核；N0 8Hz 只保留为不稳定对照。
2. 抽样绘制代表窗口，比较 E1a baseline、N1 3Hz 成功 seed 与 multiscale 崩坏 run，
   重点看“band/lag corr 好但 peak-band 差”的窗口，确认是否为浅解码上采样引入的伪峰。
3. 若 STFT 仍只有主指标小幅改善而波形诊断退化，再把 STFT 路线降级为
   条件分支或诊断特征，不作为默认输入主线。

已闭环（无需再做）：方向门控核查——确认无误杀，stft_only 在 magnitude 口径下极性盲，
保持门控不放宽；multiscale `fuse_len`/decoder 对照 + 逐窗分布核查——已定位崩坏为 wrapper
包装路径的误拣率方差（非重建退化、非单一层结构），中位数与原生持平，后续按“优先级 0”的
干净口径重判，不再扩大 deep/lite 这类纯 decoder 结构对照。
