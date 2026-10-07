# ε-RespDiff 多轨迹推理 v2 结果

2026-10-07 状态：固定 source-equivalent checkpoint 的 diagnostic subset、按规则选 N 及一次完整 DDIM6 validation 均完成，本轮关闭。未重新训练或改变 ε parameterization，独立测试集未开放。

本轮结论：多 trajectory 明显改善波形与幅值。DDIM6 到 N16 仍未满足预定 plateau，N16 作为预算上限完成全量评价；不能宣称 ensemble 已收敛。相近总 denoiser 调用数下，DDPM50 在多项波形指标和能量集中度上更好，但极端幅值随 N 非单调。其证据仍仅为 diagnostic subset，未做全量 DDPM。

## 固定身份与核验

- 协议：[ε 推理 v2](respdiff_bcg_epsilon_inference_v2_protocol_20261007.md)。源码 commit=`7cf62ad6e1da31862504f11ceb4ec2a897a9a1f8`。
- 主模型：source-equivalent，train seed=20260811，final update=6400；checkpoint SHA256=`08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6`。
- 基础噪声 seed=20261003。trajectory 0 的 initial noise 与旧推理完全兼容，其余 trajectory 独立；各 N nested。B64、FP32、GPU 0（RTX 4070 Ti SUPER）。
- 子集：64 个等距 parent 与 row 10382/12226 取并集，实际为 66 parents、858 chunks。复核选取 indices 与协议完全一致，没有用指标挑样本。
- 完整 validation：2675 parents、34775 chunks，544 batches，与旧 source run 的 parent 身份和顺序一致。
- run 根目录：`runs/respdiff_bcg_epsilon_inference_v2/`；子集 `subset_seed20261003_v1`、完整评价 `full_validation_seed20261003_v1`。
- 两份 receipt 均 complete；子集 109 项和全量 36 项文件 SHA256 校验全部匹配，共 145 项。参数与所有 buffer 的前后 hash 一致，checkpoint 文件字节前后不变，当前文件 hash 也匹配。
- 全量 raw/postfiltered 波形与参考均有限且为 `(2675,18000)`，五项主指标均覆盖 2675 个 parent。参考数组 hash 与旧 source validation 完全相同：`4e82d87d3f99c2abda11d74c7e259e0d10acce4bd7544508896e465477421ee7`；子集参考也逐值等于全量参考相应行。

本次完成后的核验只读已有产物，不重新运行 sampler 或公共指标。

## DDIM6 固定子集曲线

主结果为 parent-level 1 Hz postfiltered 输出。时间为在共享最大 N 执行中记录的累计前缀采样墙钟；不包含另记的重建/指标时间。

| N | calls/chunk | 采样秒 | Whole RR MAE | Local RR MAE | 包络轨迹 MAE | 全局调制误差 | signed PCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 12.57 | 1.5764 | 1.9434 | 0.3046 | 0.4337 | 0.5511 |
| 2 | 12 | 24.84 | 0.9579 | 1.5374 | 0.2944 | 0.3942 | 0.5509 |
| 4 | 24 | 49.38 | 1.5373 | 1.6089 | 0.2989 | 0.4283 | 0.5788 |
| 8 | 48 | 98.48 | 1.0654 | 1.5010 | 0.2832 | 0.3807 | 0.6180 |
| 16 | 96 | 196.78 | 0.9832 | 1.3819 | 0.2601 | 0.3940 | 0.6593 |

| N | raw P99 max | post P99 max | raw global max | post global max | post std 比中位数 | post top1% 能量，% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1254.19 | 439.69 | 1268.73 | 512.09 | 1.098 | 36.35 |
| 2 | 627.23 | 255.46 | 634.32 | 283.21 | 1.003 | 40.29 |
| 4 | 376.75 | 169.15 | 469.07 | 213.61 | 1.000 | 43.24 |
| 8 | 192.92 | 133.39 | 234.30 | 157.63 | 1.063 | 42.38 |
| 16 | 123.16 | 97.61 | 134.85 | 136.03 | 0.952 | 36.53 |

P99 和 global max 的整体下降明显，PCC 在较大 N 下持续提高。Whole RR、调制误差和能量集中度不是逐 N 单调改善，因此不能只依据一项指标决定收敛。

## 固定 plateau 规则

| 比较 | P99 相对变化 | top1% 相对变化 | Whole RR 相对变化 | PCC 绝对变化 | 同时通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2→4 | 33.78% | 7.33% | 60.49% | 0.02789 | 否 |
| 4→8 | 21.14% | 2.00% | 30.70% | 0.03919 | 否 |
| 8→16 | 26.82% | 13.80% | 7.71% | 0.04133 | 否 |

前三项要求相对变化严格 <5%，PCC 要求绝对变化严格 <0.01。三对均未通过，重算结果与保存的 selection 一致：`selected_N=16`，状态为 **`ensemble not converged by N=16`**。因此全量 N16 是本轮固定预算上限，不是经过收敛证明的最小 N。本轮不扩展至 N32，也不改阈值。

## DDPM50：同一子集的质量与成本

| N | calls/chunk | 采样秒 | Whole RR MAE | Local RR MAE | 包络轨迹 MAE | 全局调制误差 | signed PCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 103.27 | 0.8674 | 1.4865 | 0.2699 | 0.4018 | 0.6263 |
| 2 | 100 | 206.61 | 0.9317 | 1.3307 | 0.2449 | 0.3558 | 0.6949 |
| 4 | 200 | 413.47 | 0.9458 | 1.1269 | 0.2174 | 0.3193 | 0.7586 |

| N | post P99 max | post global max | post std 比中位数 | post top1% 能量，% | raw >1 Hz 能量，% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 92.10 | 167.39 | 0.964 | 14.63 | 0.0733 |
| 2 | 108.38 | 256.84 | 0.866 | 14.37 | 0.0912 |
| 4 | 79.26 | 132.11 | 0.829 | 13.41 | 0.0945 |

同等数量 N 并不是同等成本。较公平的预算对比如下：

- DDIM6 N8（48 calls，98.48 s）与 DDPM50 N1（50 calls，103.27 s）：后者 Whole RR/PCC/P99 更好，但全局调制误差和 global max 没有同时更好。
- DDIM6 N16（96 calls，196.78 s）与 DDPM50 N2（100 calls，206.61 s）：DDPM 的 PCC=0.6949 优于 0.6593，Local RR=1.3307 优于 1.3819，包络轨迹/全局调制误差也更小；top1% 能量 14.37% 明显低于 36.53%。但 P99 和 global max 分别为 108.38/256.84，差于 DDIM 的 97.61/136.03。
- DDPM50 N4 的 PCC、Local RR 和包络指标最好，耗时约为 DDIM6 N16 的 2.10 倍。DDPM 的 P99/global max 从 N1 到 N2 上升，再在 N4 下降，不能声称增加 N 保证最坏幅值单调改善。

DDPM raw >1 Hz 能量已经很低，额外 parent LPF 对其波形的影响较小。较多 reverse steps 在本子集改善了多项波形质量；较多 independent trajectories 改善了 DDIM 的随机输出和尾部幅值。当前没有一个设置在质量、极端幅值和成本三方面全面支配其他设置。DDPM 的范围仍是 66-parent diagnostic subset，不外推全量结论或自动执行全量 DDPM。

## 完整 validation：DDIM6 N16

以下三列使用同一 source-equivalent checkpoint 和相同 2675 个参考父窗口。旧 N1 来自[baseband 结果](respdiff_bcg_baseband_v1_results_20261007.md)的保存输出，新 N16 raw/post 来自本轮。旧运行在 RTX 2080 Ti，本轮在 RTX 4070 Ti SUPER；库环境与来源均保留，合成测试已验证新 N1 更新公式与旧 sampler 精确兼容。

| 指标 | 旧 DDIM6 N1 raw | 新 DDIM6 N16 raw | 新 DDIM6 N16 postfilter |
| --- | ---: | ---: | ---: |
| Whole RR MAE，bpm | 1.4143 | 0.7517 | **0.7517** |
| Local RR MAE，bpm | 1.7185 | 0.9639 | **0.9628** |
| 包络轨迹 MAE | 0.3250 | 0.2613 | **0.2613** |
| 全局包络调制误差 | 0.4971 | 0.3866 | **0.3862** |
| lag-aware signed PCC | 0.5313 | 0.6511 | **0.6513** |

新主结果相比旧 N1 raw：Whole RR 降低 46.85%，Local RR 降低 43.98%，包络轨迹误差降低 19.61%，全局调制误差降低 22.30%，PCC 增加 0.11995。逐 parent 改善比例分别为 60.49%、74.58%、72.60%、60.22%、74.62%。这些比例为描述性统计，未把相关窗口当成独立样本做显著性检验。

在新 N16 内，raw/post 的五指标差别很小，说明本轮五指标主要增益已在 ensemble raw 输出中出现；后滤波进一步改善保幅输出的频带与幅值诊断。公共指标内部采用呼吸带处理，不能用五指标的微小 raw/post 差异否定后滤波的高频清理收益。

| 幅值/能量诊断 | 旧 N1 raw | 新 N16 raw | 新 N16 postfilter |
| --- | ---: | ---: | ---: |
| parent max P95 | 274.92 | 60.70 | 29.22 |
| parent max P99 | 597.68 | 99.56 | 73.13 |
| 全体最大绝对幅值 | 1241.20 | 273.78 | 260.12 |
| prediction/target std 均值 | 2.639 | 1.322 | 1.205 |
| prediction/target std 中位数 | 1.319 | 1.130 | 1.031 |
| top1% 时间点能量均值，% | 49.17 | 44.01 | 34.82 |
| >1 Hz 能量均值，% | 16.92 | 14.98 | 0.361 |

target parent max P99 约 11.71，top1% 时间点能量均值约 8.79%。因此典型幅值比例明显改善，幅值长尾和过度能量集中仍存在。

完整运行墙钟 8085.58 s（约 2 h 14 min 46 s），累计采样时间 7912.08 s，重建/指标 127.74 s；其余包括来源核验、数据准备、绘图及产物哈希。每个 chunk 96 次 denoiser，整轮共 52224 次 batched denoiser forward。

## 原异常 row 与剩余最坏窗口

| row / max | 旧 N1 raw | 新 N16 raw，全量 | 新 N16 post，全量 |
| --- | ---: | ---: | ---: |
| 10382 | 1233.41 | 77.32 | 6.11 |
| 12226 | 1241.20 | 78.10 | 24.34 |

row 10382 的 std 比为 raw 2.417、post 1.304；row 12226 为 raw 2.389、post 1.858。两条已知异常显著改善。子集 N16 的相应 post max 为 6.27/24.25；子集与全量 batch 组合不同，模型 BN 使用当前 batch 统计，因此 prediction 不保证逐值相同，参考波形则逐值相同。

同一子集的 DDPM50 N4 中，这两条 row 的 post max 为 1.66/1.70，且 N1 时已为 2.98/2.81；这是较多 reverse steps 有效改善这两条异常的直接证据，但不能代表全部尾部窗口。

全量剩余最坏窗口是 row 10322：raw max=273.78，post max=260.12，target max=3.72，post std 比=16.73，top1% 时间点能量=98.45%。post 峰值发生在 123.53 s；排除 parent 两端各 1 s 或 5 s 后最大幅值不变，故不是仅出现在 parent 边缘的异常。该窗口 post >1 Hz 能量仅 0.051%，呼吸带能量约 87.05%，仍可有巨大低频幅值；本轮 1 Hz 后滤波不能保证消除它。

## 图与交付入口

子集图均保存在 `runs/respdiff_bcg_epsilon_inference_v2/subset_seed20261003_v1/`：

- `metrics_vs_N.png`
- `max_amplitude_vs_N.png`
- `runtime_vs_N.png`
- `row_10382_ensemble_progression.png`
- `row_12226_ensemble_progression.png`

曲线 CSV/JSON、runtime_profile、selection、两条 row 的 progression NPZ 位于同目录；各 sampler_N 子目录保存 raw/post 波形、诊断和逐 parent metrics。全量主结果入口为 `full_validation_seed20261003_v1/validation_summary.csv` 及该目录 raw/post waveform/diagnostics；完整逐 parent 指标在 `ddim_N16/`。

本轮交付为经预算规则选择的 ε-RespDiff DDIM6 N16 + parent LPF 推理结果。它显著提升了完整 validation 性能和典型幅值恢复，但 ensemble 尚未收敛，极端幅值问题尚未完全解决。DDPM50 提供质量与成本的子集 anchor；新的 DDPM 全量或更大 N 实验需另立后续协议。
