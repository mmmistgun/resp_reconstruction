# E1 STFT 输入信息增益实验记录

## 结论摘要

本轮 E1 实验没有给出“STFT 输入带来稳定、全面信息增益”的强证据。更准确的结论是：
STFT 分支在 `patch_mixer1d` 上对主指标有局部收益，但伴随波形相关性和相对包络指标退化；
在 `multiscale_decomp_mixer1d` 上，STFT 相关路径明显破坏主指标，不建议作为下一阶段主线。

当前可保留的候选方向是 `patch_mixer1d + conv2d + N1 3Hz`，但补充 seed 后证据变弱：
它仍好于 N0 8Hz 的补充表现，但已经不能认为相对纯时序 baseline 有稳定优势。
N0 8Hz 更应作为不稳定对照，而不是主候选。

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
- `manifest`：批次编排脚本生成的预期实验清单，记录每个 tag 对应的 label、backbone、
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

但该收益不全面，也不够稳定：E1b/E1n1 的 `band_limited_corr` 和 `best_lag_corr`
明显低于 E1a，`relative_envelope_mae` 也没有同步改善。补充 seed 还暴露出 gate failure
和 N0 8Hz 主指标波动。因此当前只能记录为“PatchMixer 上存在条件性局部信号”，
不能判定为“STFT 输入稳定改善波形重建质量”。

### MultiScale 路线

`multiscale_decomp_mixer1d` 加入 STFT 后主指标明显恶化。

- E1a baseline peak-band MAE 为 `0.483444`。
- E1a_prime 为 `1.324590`。
- E1b 3/8/12Hz 分别为 `1.281431`、`1.431665`、`1.193187`。

虽然 spec MAE、相对包络和波形相关性有局部改善，但主指标恶化幅度过大。
这更像是融合头或输出形态改变后破坏了 peak-band RR 提取，而不是 STFT 信息带来的正收益。

### STFT-only 路线

E1c `stft_only` 在两个 backbone 上均未通过早期 checkpoint gate，后续同类实验按预期跳过。
当前结论是：在现有 loss、gate 和模型容量设置下，STFT magnitude 单分支不足以作为主输入路线。

### N1 频带尺度对齐

N1 只在 `patch_mixer1d + high_hz=3` 代表档做了 3 seed 对照。
相对 N0 3Hz，N1 的主指标略好，spec MAE 略差，其余指标基本持平。

这一结果支持继续保留 N1 作为候选归一化方式，但不足以单独证明 STFT 输入稳定有效。

## 当前决策

- 暂保留候选：`patch_mixer1d + conv2d + N1 3Hz`，但需要降级为“待复核候选”。
- 保留不稳定对照：`patch_mixer1d + conv2d + N0 8Hz`，补充 seed 后不再视作强候选。
- 暂停扩展：`multiscale_decomp_mixer1d + STFT`。
- 暂停扩展：`stft_only`。
- 暂不进入 CWT/SST 或更复杂频域前端；E1 还没有满足“STFT 输入稳定增益”的进入条件。

## 下一步

1. 对 `patch_mixer1d + N1 3Hz` 继续做少量 seed 或独立 split 复核；N0 8Hz 只保留为不稳定对照。
2. 抽样绘制代表窗口，重点比较 E1a baseline、N1 3Hz 成功 seed 和 gate failure seed，
   检查 peak-band MAE 改善是否以波形相关性下降为代价。
3. 对 `multiscale_decomp_mixer1d` 的高 peak-band 误差做少量坏例诊断，确认是 peak 提取失真、
   输出形态问题，还是融合头容量/上采样路径问题。
4. 若下一轮仍只有主指标小幅改善而波形诊断退化，应把 STFT 路线降级为条件分支或诊断特征，
   不作为默认输入主线。
