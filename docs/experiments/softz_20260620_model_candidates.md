# 20260620 soft-z 模型候选重跑

本文档记录 2026-06-20 新数据集上的候选模型重跑结果。旧的 rawish state-aligned
结论只能作为候选模型来源，不能和本轮结果直接比较。

## 数据口径

- 数据集：`20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf`
- 配置：`configs/tho_research_v2.yaml`
- BCG 输入字段：`bcg_rawish_segment_soft_z_key`
- 实际 BCG signal key：`bcg_rawish_wideband_state_aligned_segment_soft_z`
- THO 目标字段：`target_waveform_segment_soft_z_key`
- 实际 THO signal key：`tho_waveform_segment_soft_z`
- 训练窗口：全量可用窗口，`batch_size=128`
- 数据抽样 seed：`train_sample_seed=20260610`，`val_sample_seed=20260611`
- 训练 seed：`20260700`、`20260710`、`20260837`
- 输出：`runs/tho_research_v2_20260620_softz_model_candidates/`

## 汇总产物

- 原始 run 汇总：`runs/tho_research_v2_20260620_softz_model_candidates_summary.csv`
- 带模型和 seed 的排序表：`runs/tho_research_v2_20260620_softz_model_candidates_ranked.csv`
- 按模型聚合表：`runs/tho_research_v2_20260620_softz_model_candidates_by_model.csv`

## 按模型聚合

| 模型 | seeds | best val loss 均值 | best val loss 最小 | RR peak-band error 均值 | RR peak-band error 最小 | RR spec error 均值 | rel env MAE 均值 | rel env corr 均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiscale_decomp_mixer1d` | 3 | 0.583703 | 0.578145 | 0.483444 | 0.447530 | 0.544867 | 0.240630 | 0.501694 |
| `patch_mixer1d` | 3 | 0.587490 | 0.573943 | 0.489164 | 0.463629 | 0.588493 | 0.231147 | 0.518642 |
| `patch_hann_bandlimited_output1d` | 3 | 0.589328 | 0.584010 | 0.496856 | 0.449160 | 0.598532 | 0.236325 | 0.508186 |
| `period_aware_patch_hann_bandlimited1d` | 3 | 0.589296 | 0.587345 | 0.512247 | 0.485173 | 0.562756 | 0.247155 | 0.511969 |
| `multiscale_patch_hann_bandlimited1d` | 3 | 0.577882 | 0.576160 | 0.580652 | 0.524660 | 0.545050 | 0.237518 | 0.521515 |
| `polyphase_patch_hann_bandlimited1d` | 3 | 0.561825 | 0.557547 | 0.651780 | 0.621987 | 0.527891 | 0.232119 | 0.525493 |

## 当前判断

按验证 loss 看，`polyphase_patch_hann_bandlimited1d` 最强，三个 seed 都稳定在
0.5575-0.5703。但它的 `rr_peak_band_abs_error` 明显偏高，不宜直接作为任务主选。

按当前任务选择护栏看，`multiscale_decomp_mixer1d`、`patch_mixer1d` 和
`patch_hann_bandlimited_output1d` 更接近主线。单 run 最好的 RR peak-band error 是
`multiscale_decomp_mixer1d:20260710` 的 0.447530；`patch_hann_bandlimited_output1d`
的 0.449160/0.457921 两个 seed 也很接近；`patch_mixer1d:20260700` 在相对包络 MAE
和综合稳定性上更好。

下一步如果继续收窄，建议保留这三类模型做更细的 loss/训练策略对照，同时单独解释
`polyphase` 为什么 loss 和频谱指标好但 RR peak-band error 差。
