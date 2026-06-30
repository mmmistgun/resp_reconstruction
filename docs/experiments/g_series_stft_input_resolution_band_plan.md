# G 系列 STFT 输入时间分辨率与频带范围实验规划

本文档规划 G 系列实验。目标是在 `F0_native_stft_pre_mixer` 已作为当前时频输入 anchor 的基础上，拆清两个问题：

1. 当前在线 STFT 输入的时间分辨率是否过低，`30s / 5s hop` 是否把动态呼吸线索压得太粗。
2. 当前 `0.05-8Hz` fullband 是否过宽，收益到底来自呼吸主带、谐波/中频上下文、分频带归纳偏置，还是高频心冲击/扰动上下文。

G 系列不是 target-STFT loss 实验，也不改变输出空间。所有第一阶段候选仍使用 waveform 输出、`native_inject pre_mixer` 融合和当前基础 loss。实验只改变 STFT 输入分支的时间频率参数或编码方式。

## 0. 固定 anchor 与研究问题

主 anchor：

| label | 模型口径 | 作用 |
|---|---|---|
| `F0_native_stft_pre_mixer` | `patch_mixer1d + conv2d STFT + native_inject pre_mixer + 0.05-8Hz + win3000/hop500` | G 系列主 baseline |
| `F0_native_time_only` | 同一 `patch_mixer1d` substrate，不启用 STFT 分支 | 判断 STFT 输入仍是否有净贡献 |

固定项：

- 数据：`configs/tho_research_v2.yaml` 当前 research v2 soft-z 口径。
- 时间域骨干：`patch_mixer1d`。
- 融合方式：`fusion_mode=native_inject`。
- 注入位置：`stft_inject_position=pre_mixer`。
- STFT encoder 默认：`conv2d`，`stft_out_channels=16`，`stft_norm=n0`。
- 输出：一维 respiration waveform，长度 `18000`。
- loss：不启用 target-STFT loss，`loss.stft_dist_weight=0.0`、`loss.stft_band_energy_weight=0.0`。
- paired 对照：同 seed、同数据 seed、同验证集、同 checkpoint gate。

核心问题：

- 如果只提高 STFT 时间栅格，是否能减少 `37 -> 140` 的大比例插值带来的信息平滑。
- 如果缩短 STFT 窗口，是否改善局部 waveform / RR 跟随，还是因为频率分辨率下降而加重主频/谐波误拣。
- 如果收窄频带，是否能减少高频噪声注入；如果保留高频，是否确实帮助 hard / low-spectrum 窗口。
- `bandgroup` / `bandenergy` 是否比 fullband conv2d 更容易表达呼吸主带、谐波和高频上下文的分工。

## 1. 为什么不直接跑完整 4 x 6 矩阵

候选维度：

STFT 参数：

以下帧数按在线 `STFTEncoder` 当前实现估算：`center=True`，`n_fft=stft_win`。

| 代号 | `stft_win` | `stft_hop` | 直观含义 | 预期帧数 |
|---|---:|---:|---|---:|
| A | 3000 | 500 | 当前 F0，30s 窗、5s hop | 37 |
| B | 3000 | 250 | 只提高时间栅格，保留 30s 频率分辨率 | 73 |
| C | 2000 | 250 | 20s 平衡窗口，2.5s hop | 73 |
| D | 1500 | 128 | 15s 激进高时间分辨率，1.28s hop | 141 |

频带/编码候选：

| 代号 | 频带/编码 | 目的 |
|---|---|---|
| R1 | fullband `0.05-1.2Hz` | 呼吸主带 + 一阶谐波上沿，去掉高频上下文 |
| R2 | fullband `0.05-3.0Hz` | 呼吸主带 + 谐波/低阶心冲击上下文 |
| R3 | fullband `0.067-1.2Hz` | 更保守呼吸频带，去掉极低频漂移 |
| R4 | `bandgroup` | 重叠分频带保留带内频率结构 |
| R5 | `bandenergy` | 分频带能量序列，强先验、低维输入 |
| R6 | high-only ablation | 高频上下文单独分支，判断高频是否有独立贡献 |

完整析因至少 `4 x 6 = 24` 个配置；如果每个 3 seed，再加 F0/time-only paired anchor，成本接近正式阶段，而归因仍会被二阶交互污染。G 系列应先用分阶段矩阵减少自由度。

推荐顺序：**先粗定 STFT 时间参数，再在胜出的时间参数上扫频带范围，最后做少量交互复核。**

理由：

- 当前最大结构疑点是 STFT 帧数只有 37，而注入 token 是 140；时间栅格过粗会影响所有频带候选。
- 频带收益会依赖窗口长度。比如 `0.067-1.2Hz` 在 30s 窗下有约 34 个 bin，在 15s 窗下只有约 17 个 bin；如果先用当前 30s 窗选频带，可能选出一个只适合低时间分辨率的频带。
- 但不能完全忽略频带，因为 `0.05-8Hz` 太宽时可能掩盖 hop/window 的收益。因此第一阶段用两个代表频带做参数筛选，而不是只用 8Hz。

## 2. G0：anchor 复现与形状/成本 dry-run

目的：确认 G 系列基线仍是当前 F0，不因为后续 runner 或配置改动引入额外变量。

配置：

| label | branch | STFT 参数 | 频带 | encoder |
|---|---|---|---|---|
| `G0_time_only` | time only | 无 | 无 | 无 |
| `G0_f0_native_stft_pre_mixer` | dual | A: `3000/500` | `0.05-8Hz` | `conv2d` |

要求：

- 只在实现 runner 后做 smoke / dry-run，不在本文档阶段启动正式训练。
- 正式实验前检查 `config.yaml` 中 `model.stft_win`、`model.stft_hop`、`model.stft_low_hz`、`model.stft_high_hz`、`model.stft_encoder_type`、`model.fusion_mode` 和 `model.stft_inject_position`。
- 记录每个参数组的 STFT frame 数、对齐到 patch token 后的插值比例和单 batch 显存/耗时。

## 3. G1：先确定 STFT 时间参数

G1 只回答：在 `native_inject pre_mixer` 中，STFT 输入应该保持粗时间栅格，还是提高时间分辨率。

### 3.1 代表频带

G1 不直接用全部频带候选，只用两个代表档：

| 档位 | 频带 | 原因 |
|---|---|---|
| `wide_anchor` | `0.05-8.0Hz` | 当前 F0 anchor，保留历史可比性 |
| `resp_mid` | `0.05-3.0Hz` | 去掉 3-8Hz 高频，仍保留呼吸主带、谐波和低阶心冲击上下文 |

不建议 G1 首批只用 `0.067-1.2Hz`，因为它太保守，可能把“STFT 时间参数”问题和“高频上下文是否有用”问题混在一起。

### 3.2 G1 矩阵

| label | STFT 参数 | 频带 | encoder | 目的 |
|---|---|---|---|---|
| `G1A_wide` | A `3000/500` | `0.05-8.0` | `conv2d` | 当前 F0 复现 |
| `G1B_wide` | B `3000/250` | `0.05-8.0` | `conv2d` | 只提高 STFT 帧密度 |
| `G1C_wide` | C `2000/250` | `0.05-8.0` | `conv2d` | 平衡频率/时间分辨率 |
| `G1D_wide` | D `1500/128` | `0.05-8.0` | `conv2d` | 高时间分辨率极限探针 |
| `G1A_resp_mid` | A `3000/500` | `0.05-3.0` | `conv2d` | 当前时间参数下的中频对照 |
| `G1B_resp_mid` | B `3000/250` | `0.05-3.0` | `conv2d` | 中频下只提高帧密度 |
| `G1C_resp_mid` | C `2000/250` | `0.05-3.0` | `conv2d` | 中频平衡窗口 |
| `G1D_resp_mid` | D `1500/128` | `0.05-3.0` | `conv2d` | 中频高时间分辨率 |

第一批 seed：

- pilot：每个候选 2 seed，先看方向和稳定性。
- 若 B/C/D 任一在两个频带上都优于 A，再扩到 3 seed。
- 若只有单一频带改善，进入 G2 前要补交互复核，避免误把频带收益当作时间参数收益。

G1 推荐判定：

- 如果 B 在 `wide_anchor` 和 `resp_mid` 都改善或至少不伤主护栏，优先把 B 作为 G2 默认参数。它只改变 hop，归因最干净。
- 如果 C 优于 B，说明 30s 窗确实偏长，G2 用 C 作为默认参数，并保留 B 作为保守参照。
- 如果 D 指标波动、count error 或 fast-RR 明显变差，停止更短窗方向；D 只保留为高时间分辨率负/边界证据。
- 如果 B/C/D 都没有超过 A，G2 仍用 A，但频带扫应解释为“当前 30s/5s STFT 输入的频带选择”，不能推广到高时间分辨率 STFT。

## 4. G2：在胜出 STFT 参数上扫频带与编码方式

G2 使用 G1 选出的默认 STFT 参数 `T*`。截至 2026-06-29 的 G1 三 seed 结果，`T*=C: win2000/hop250`；`B: win3000/hop250` 保留为 G3 的保守交互复核候选。

### 4.1 G2 fullband 频带矩阵

| label | STFT 参数 | 频带 | encoder | 目的 |
|---|---|---|---|---|
| `G2_R1_resp_1p2` | `T*` | `0.05-1.2Hz` | `conv2d` | 主呼吸/谐波上沿，去高频 |
| `G2_R2_resp_3p0` | `T*` | `0.05-3.0Hz` | `conv2d` | 呼吸 + 谐波/低阶高频上下文 |
| `G2_R3_strict_resp` | `T*` | `0.067-1.2Hz` | `conv2d` | 去极低频漂移，保守呼吸带 |
| `G2_R0_wide_8p0` | `T*` | `0.05-8.0Hz` | `conv2d` | 与 F0 宽频 anchor 对齐 |

判定重点：

- `R1/R3` 如果改善 easy windows 但伤 hard/low-spectrum，说明高频或谐波上下文对困难窗口仍有价值。
- `R2` 如果接近或优于 `R0`，说明 3-8Hz 高频不是必要变量，可以收窄默认频带。
- `R0` 如果仍明显最好，需要进一步用 high-only ablation 判断它是真高频信息，还是宽频模型容量/归一化副作用。

### 4.2 G2 分频带输入矩阵

| label | STFT 参数 | 频带配置 | encoder | 目的 |
|---|---|---|---|---|
| `G2_R4_bandgroup` | `T*` | 默认重叠 bands | `bandgroup` | 保留带内频率结构，同时显式分组 |
| `G2_R5_bandenergy` | `T*` | 默认重叠 bands | `bandenergy` | 只给分频带能量轨迹，测试强先验是否足够 |

默认重叠 bands 沿用现有 STFT encoder 逻辑：

| band | 解释 |
|---|---|
| `0.05-0.3Hz` | 慢呼吸 / 主节律低端 |
| `0.1-0.7Hz` | 常规呼吸主带 |
| `0.3-1.2Hz` | 谐波 / 波形尖锐度 |
| `0.7-3.0Hz` | 偏快呼吸、谐波和低阶心冲击上下文 |
| `3.0-8.0Hz` | 高频心冲击/体动上下文 |

判定重点：

- `bandgroup` 优于 fullband：说明显式频带分组有帮助，但仍需要带内频率结构。
- `bandenergy` 优于 fullband：说明模型主要需要慢变频带能量，不需要细粒度频点图。
- `bandenergy` 变差但 `bandgroup` 持平/改善：说明仅能量轨迹太粗，频带内峰位或谐波结构仍重要。

### 4.3 high-only ablation

高频分支不建议直接和主模型同权作为默认候选，而应作为解释实验：

| label | STFT 参数 | 频带 | encoder | 目的 |
|---|---|---|---|---|
| `G2_R6_high_1p2_8p0` | `T*` | `1.2-8.0Hz` | `conv2d` | 高频上下文单独是否有任务信息 |
| `G2_R6_high_3p0_8p0` | `T*` | `3.0-8.0Hz` | `conv2d` | 去掉呼吸/低阶谐波，只留更纯高频 |

解释规则：

- high-only 接近 time-only 或更差：高频不能独立承担任务；若 `0.05-8Hz` 好，收益可能来自低频 + 高频协同。
- high-only 明显优于 time-only：必须进一步检查是否是心肺耦合、运动/subject 捷径或 split 泄漏风险；不能直接把它当作稳定生理收益。
- high-only 改善 spec 但伤 peak-band/count：高频更像频谱上下文或捷径，不应作为主输入默认。

## 5. G3：交互复核，而不是扩大全矩阵

G3 只复核 G1/G2 中最可能存在交互的少数组合。

建议进入条件：

- G1 选出一个非 A 的时间参数 `T*`。
- G2 选出一个非 `0.05-8Hz` 的频带/编码 `R*`。
- `T* + R*` 至少 2 seed 主护栏不劣于 F0，且 hard/low-spectrum 分层有收益。

最小复核矩阵：

| label | 目的 |
|---|---|
| `G3_A_Rstar` | 判断 `R*` 在原始 A 参数下是否也有效 |
| `G3_Tstar_R0` | 判断 `T*` 在宽频 8Hz 下是否仍有效 |
| `G3_Tstar_Rstar` | 最终候选 |
| `G3_A_R0` | F0 anchor |

如果 `G3_Tstar_Rstar` 只在单 seed 或单分层改善，不进入下一阶段；如果三 seed paired delta 一致，再考虑作为新的 H 系列 anchor 或回写 F 系列后续分支。

## 6. 指标、分层和通过条件

主指标：

- `rr_peak_band_abs_error`
- `rr_spec_abs_error`
- `frac_gt_1`、`frac_gt_2`
- `breath_count_zero_cross_abs_error`

波形与同步护栏：

- `relative_envelope_corr`
- `relative_envelope_mae`
- `band_limited_corr`
- `best_lag_corr`
- `best_lag_sec`

必须分层：

- baseline easy / hard。
- low `spectrum_similarity`。
- fast RR。
- baseline count-error = 0 vs >0。
- high harmonic-ratio 或 peak-bin shift 窗口。

通过条件：

- overall 主指标不劣于 F0，最好相对同 seed F0 有 paired 改善。
- hard 或 low-spectrum 分层至少一个稳定改善。
- baseline easy 不出现三 seed 同向退化。
- fast RR 和 breath count 不系统性恶化。
- `rr_spec_abs_error` 的改善不能以明显 peak-band/count 退化换取。

停止条件：

- 只改善 `rr_spec_abs_error`，但 `rr_peak_band_abs_error`、count 或 easy windows 变差。
- high-only 明显改善但分层显示集中在疑似 subject/运动捷径，且没有跨 split 证据。
- D 档短窗导致主频/谐波误拣率升高，即使局部相关性改善，也不作为默认参数。
- 分频带输入只改善 loss 或频谱相似度，不改善任务护栏。

## 7. 实验记录与后续实现要求

当前已实现 G0/G1/G2/G3 编排脚手架：

- runner：`scripts/run_g_series_stft_input_probe.py`
- 默认 manifest：`runs/g_series_stft_input_manifest.csv`
- 阶段参数：`--stage g0`、`--stage g1`、`--stage g2`、`--stage g3` 或默认 `g0-g1`
- pilot seed：`20260700`、`20260837`
- summary：复用 `scripts/summarize_f_a_stft_loss.py`，已支持 `G1/G2/G3` 候选标签；`G0` anchor 不进入 candidate delta。

准备命令：

```bash
./.venv/bin/python scripts/run_g_series_stft_input_probe.py \
  --stage g0-g1 \
  --dry-run \
  --manifest runs/g_series_stft_input_manifest.csv

./.venv/bin/python scripts/run_g_series_stft_input_probe.py \
  --stage g0-g1 \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 4 \
  --manifest runs/g_series_stft_input_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/eval_topk_checkpoints.py \
  --runs-root runs/g_series_stft_input \
  --top-k 3 \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 4 \
  --metric-workers 4 \
  --start-stagger-sec 30 \
  --output-prefix runs/g_series_stft_input_topk

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/g_series_stft_input_manifest.csv \
  --output runs/g_series_stft_input_summary.csv \
  --paired-output runs/g_series_stft_input_paired_delta.csv \
  --strata-output runs/g_series_stft_input_strata_delta.csv
```

G2 频带探针命令：

```bash
./.venv/bin/python scripts/run_g_series_stft_input_probe.py \
  --stage g2 \
  --seed 20260700 \
  --seed 20260837 \
  --seed 20260901 \
  --dry-run \
  --manifest runs/g_series_stft_input_g2_manifest.csv

./.venv/bin/python scripts/run_g_series_stft_input_probe.py \
  --stage g2 \
  --seed 20260700 \
  --seed 20260837 \
  --seed 20260901 \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 4 \
  --manifest runs/g_series_stft_input_g2_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/eval_topk_checkpoints.py \
  --runs-root runs/g_series_stft_input \
  --top-k 3 \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 4 \
  --metric-workers 4 \
  --start-stagger-sec 30 \
  --output-prefix runs/g_series_stft_input_topk

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/g_series_stft_input_g2_manifest.csv \
  --output runs/g_series_stft_input_g2_summary.csv \
  --paired-output runs/g_series_stft_input_g2_paired_delta.csv \
  --strata-output runs/g_series_stft_input_g2_strata_delta.csv
```

G3 交互复核命令：

```bash
./.venv/bin/python scripts/run_g_series_stft_input_probe.py \
  --stage g3 \
  --seed 20260700 \
  --seed 20260837 \
  --seed 20260901 \
  --dry-run \
  --manifest runs/g_series_stft_input_g3_manifest.csv

./.venv/bin/python scripts/run_g_series_stft_input_probe.py \
  --stage g3 \
  --seed 20260700 \
  --seed 20260837 \
  --seed 20260901 \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 4 \
  --manifest runs/g_series_stft_input_g3_manifest.csv \
  --start-stagger-sec 90

./.venv/bin/python scripts/eval_topk_checkpoints.py \
  --runs-root runs/g_series_stft_input \
  --top-k 3 \
  --device cuda:0 \
  --device cuda:1 \
  --max-parallel 4 \
  --metric-workers 4 \
  --start-stagger-sec 30 \
  --manifest runs/g_series_stft_input_topk_g3_eval_manifest.csv \
  --output-prefix runs/g_series_stft_input_topk

./.venv/bin/python scripts/summarize_f_a_stft_loss.py \
  --manifest runs/g_series_stft_input_g3_manifest.csv \
  --output runs/g_series_stft_input_g3_summary.csv \
  --paired-output runs/g_series_stft_input_g3_paired_delta.csv \
  --strata-output runs/g_series_stft_input_g3_strata_delta.csv
```

正式训练前仍需要补：

- smoke：小窗口或 1 epoch 验证 run 产物完整，不改变 split、标签、核心指标口径。
- 结论汇总前跑 `eval_topk_checkpoints.py --top-k 3`；topK 结果属于同验证集任务指标择优，表述时必须标注 validation top-k selection。

正式训练前必须确认：

- 不改变数据 split、目标定义、评价指标和 checkpoint gate。
- 不覆盖历史 F0/F-A/F-B run 产物。
- 所有候选都能回连到 resolved `config.yaml` 和 manifest。

## 8. 当前推荐结论

实验顺序建议：

1. `G0` 复现 anchor。
2. `G1` 用 `0.05-8Hz` 与 `0.05-3Hz` 两个代表频带先筛 STFT 参数。
3. G1 三 seed 后，G2 默认用 `C: win2000/hop250`；`B: win3000/hop250` 保留为 G3 保守交互复核候选。
4. `G2` 在 `T*` 上扫 `0.05-1.2Hz`、`0.05-3Hz`、`0.067-1.2Hz`、`0.05-8Hz`、`bandgroup`、`bandenergy` 和 high-only ablation。
5. `G3` 只复核少量 `T* x R*` 交互，不扩成完整 24 配置全矩阵。

当前最值得优先验证的假设是：

```text
G1C_wide 显示 20s/2.5s STFT 输入在 hard 和 low-spectrum 分层上更有收益，
但 easy windows 有轻微退化风险。
因此 G2 优先在 win=2000/hop=250 上扫频带与编码方式，
同时把 easy windows 作为护栏，并把 B 档留给 G3 交互复核。
```

## 9. G0/G1 pilot 2026-06-29 结果

运行范围：

- 训练 manifest：`runs/g_series_stft_input_manifest.csv`
- 第三 seed 训练 manifest：`runs/g_series_stft_input_seed20260901_manifest.csv`
- 普通 checkpoint 汇总：`runs/g_series_stft_input_summary.csv`
- 第三 seed 普通 checkpoint 汇总：`runs/g_series_stft_input_seed20260901_summary.csv`
- 普通 paired delta：`runs/g_series_stft_input_paired_delta.csv`
- 第三 seed paired delta：`runs/g_series_stft_input_seed20260901_paired_delta.csv`
- 普通分层 delta：`runs/g_series_stft_input_strata_delta.csv`
- 第三 seed 分层 delta：`runs/g_series_stft_input_seed20260901_strata_delta.csv`
- top3 复评 manifest：`runs/g_series_stft_input_topk_eval_manifest.csv`
- top3 全量结果：`runs/g_series_stft_input_topk_all_metrics.csv`
- top3 任务指标择优：`runs/g_series_stft_input_topk_best_by_rr.csv`

完成性：

- G0/G1 首批 20 个训练 run 与第三 seed 10 个训练 run 全部完成；每个 label 均有 `20260700`、`20260837`、`20260901` 三个 seed。
- top3 复评共 90 个 checkpoint 评价结果，`best_by_rr` 共 30 行。
- 未改变 split、target、metrics 或 checkpoint gate；top3 结论属于同验证集任务指标择优，后续表述需标注 validation top-k selection。

G0 anchor：

- top3 口径下，`G0_f0_native_stft_pre_mixer` 相对 `G0_time_only` 的 `rr_peak_band_abs_error_mean` 三 seed 均改善，平均 `-0.0502`。
- `frac_gt_1` 平均下降 `-1.47pp`，`frac_gt_2` 平均下降 `-0.96pp`；说明当前 STFT 输入 anchor 仍有净贡献。

G1 时间参数结论：

| label | top3 `rr_peak_band_abs_error_mean` Δ | top3 `rr_spec_abs_error_mean` Δ | top3 count Δ | 普通 checkpoint `rr_peak_band_abs_error_mean` Δ | 判断 |
|---|---:|---:|---:|---:|---|
| `G1B_wide` | -0.0138 | +0.0155 | -0.0209 | -0.0225 | 只提高 hop 有收益，但 spec 仍偏伤 |
| `G1C_wide` | -0.0145 | +0.0057 | -0.0212 | -0.0351 | 普通 checkpoint 最稳，wide 下优先 |
| `G1D_wide` | -0.0225 | -0.0078 | -0.0308 | -0.0249 | top3 peak 最强，但短窗同步/中频稳定性不足 |
| `G1B_resp_mid` | -0.0039 | -0.0188 | +0.0110 | -0.0052 | 中频下收益弱且 count 不稳 |
| `G1C_resp_mid` | -0.0044 | -0.0075 | -0.0233 | -0.0209 | 中频下仍优于 B，支持 C |
| `G1D_resp_mid` | +0.0012 | -0.0411 | +0.0000 | -0.0181 | spec 强但 peak/count 不够稳 |

分层观察：

- `G1C_wide` 在 `baseline_hard` 上改善最明显：普通 checkpoint `rr_peak_band_abs_error_mean` 平均 `-0.3435`，`breath_count_zero_cross_abs_error_mean` 平均 `-0.1259`。
- `G1C_wide` 在 `low_spectrum` 上也改善：peak-band 平均 `-0.0371`，spec 平均 `-0.0270`。
- `baseline_easy` 上 B/C/D 都有 peak-band 小幅退化，`G1C_wide` 平均 `+0.0242`；因此 G2 需要继续把 easy windows 作为护栏。
- `G1D_wide` 第三 seed 后整体 top3 peak-band 最强，但 `G1D_resp_mid` top3 peak-band 不稳定，且 D 系列在 relative-envelope corr 上仍偏退化；短窗方向暂不作为默认时间参数。

阶段判断：

- G1 三 seed 结果支持 `C: win2000/hop250` 作为 G2 主时间参数；`B: win3000/hop250` 保留为保守参照，`D: win1500/hop128` 作为短窗边界/解释候选。
- G2 建议以 `T*=C` 扫 `0.05-1.2Hz`、`0.05-3Hz`、`0.067-1.2Hz`、`0.05-8Hz`、`bandgroup`、`bandenergy` 和 high-only ablation。
- 若 G2 中 C 的收窄频带候选只在 hard/low-spectrum 改善、easy 继续退化，应在 G3 用 `B` 或 `A` 做少量交互复核，避免把时间窗收益和 easy 退化绑定成默认方案。

## 10. G2 pilot 2026-06-30 结果

运行范围：

- G2 manifest：`runs/g_series_stft_input_g2_manifest.csv`
- 普通 checkpoint 汇总：`runs/g_series_stft_input_g2_summary.csv`
- 普通 paired delta：`runs/g_series_stft_input_g2_paired_delta.csv`
- 普通分层 delta：`runs/g_series_stft_input_g2_strata_delta.csv`
- G2 top3 复评 manifest：`runs/g_series_stft_input_topk_g2_eval_manifest.csv`
- top3 全量结果：`runs/g_series_stft_input_topk_all_metrics.csv`
- top3 任务指标择优：`runs/g_series_stft_input_topk_best_by_rr.csv`

完成性：

- `runs/g_series_stft_input_g2_manifest.csv` 中 24 个训练 run 全部完成。
- G2 top3 共 72 个 checkpoint 评价全部完成；当前 top3 全量结果共 162 行，`best_by_rr` 共 54 行。
- 复查 `eval_topk_checkpoints.py --dry-run` 无待评价任务。
- 未改变 split、target、metrics 或 checkpoint gate；top3 结论属于同验证集任务指标择优，表述时必须标注 validation top-k selection。

G2 相对 `G2_R0_wide_8p0` 的 top3 任务指标择优均值 delta：

| label | `rr_peak_band_abs_error_mean` Δ | `rr_spec_abs_error_mean` Δ | count Δ | `frac_gt_1` Δ | 判断 |
|---|---:|---:|---:|---:|---|
| `G2_R1_resp_1p2` | +0.0271 | -0.0336 | +0.0285 | +0.0041 | spec 改善但主护栏退化 |
| `G2_R2_resp_3p0` | +0.0101 | -0.0122 | -0.0020 | +0.0052 | 接近 R0，但整体 peak 仍退化 |
| `G2_R3_strict_resp` | +0.0231 | -0.0234 | +0.0606 | +0.0090 | strict resp 不适合作默认 |
| `G2_R4_bandgroup` | +0.0134 | -0.0235 | +0.0239 | +0.0072 | 显式分组未超过 fullband |
| `G2_R5_bandenergy` | -0.0099 | -0.0144 | -0.0172 | -0.0004 | top3 下最有希望，但 corr 略退化 |
| `G2_R6_high_1p2_8p0` | -0.0054 | -0.0047 | -0.0032 | +0.0001 | high-only 有解释价值，不可直接默认 |
| `G2_R6_high_3p0_8p0` | +0.0164 | -0.0046 | +0.0147 | +0.0056 | 更纯高频不成立 |

普通 checkpoint 均值 delta 显示，所有非 R0 候选的 overall `rr_peak_band_abs_error_mean` 均比 `G2_R0_wide_8p0` 更差；其中 `G2_R5_bandenergy` 虽然 count 近似持平，但 peak 仍退化 `+0.0249`。因此 G2 的候选收益主要出现在 validation top-k selection 口径，不应直接升级为默认训练配置。

分层观察：

- `G2_R2_resp_3p0` 在普通 checkpoint 的 `baseline_hard` 分层 peak 改善 `-0.0792`，但 `baseline_easy` 退化 `+0.0317`，说明收窄到 3Hz 可能帮助困难窗口，却继续伤 easy 护栏。
- `G2_R5_bandenergy` 在 `baseline_hard` 分层 peak 改善 `-0.1687`、count 改善 `-0.0513`，同时 overall peak 仍退化；它更像候选归纳偏置，而不是已验证默认输入。
- `G2_R6_high_1p2_8p0` 的 top3 peak/count 相对 R0 有小幅改善，且相对 time-only 仍改善；但 seed 表现不一致，relative-envelope corr 下降，不能解释为稳定高频生理收益。

阶段判断：

- G2 不支持把 `0.05-1.2Hz`、`0.067-1.2Hz`、`0.05-3Hz` 或 `bandgroup` 直接设为默认 STFT 输入；`G2_R0_wide_8p0` 仍是普通 checkpoint 口径下最稳的 C 档 fullband anchor。
- `G2_R5_bandenergy` 是唯一值得进入 G3 的频带编码候选，但必须标注其证据来自 validation top-k selection，且需要用普通 checkpoint 或跨时间参数交互复核确认。
- high-only 分支只保留为解释实验；若后续继续追，需要先检查 subject/session 捷径、运动高频和 split 泄漏风险。
- 建议 G3 不扩大全矩阵，只复核 `C/B x {wide_8p0, bandenergy}`，可选加一个 `high_1p2_8p0` 解释臂。
