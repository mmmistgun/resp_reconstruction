# THO research v2 实验记录与性能对比

本文档作为稳定实验台账，记录 THO research v2 相关 run 的核心配置、性能对比和阶段结论。更细的实现计划和执行流水见 `docs/superpowers/plans/`。

## 对比口径

- 数据集：`research_v2_waveform`
- 目标：预测 `r_tho_hat(t)` 波形，指标仅用于评估和诊断
- 验证集：固定 `data.val_sample_seed=20260611`
- 窗口数：2557 个 val 窗口
- 主模型：`patch_mixer1d`
- 关键指标方向：
  - `relative_envelope_corr` 越高越好
  - `relative_envelope_mae` 越低越好
  - `rr_peak_abs_error` / `rr_spec_abs_error` 越低越好
  - `spectrum_similarity` 越高越好

## 当前基准与相对包络实验

| 实验 | run | 关键变化 | best epoch | best val loss | `relative_envelope_mae` mean / median / p75 | `relative_envelope_corr` mean / median | `rr_peak_abs_error` mean / median | `rr_spec_abs_error` mean | `spectrum_similarity` mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HF penalty 基线 | `runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400` | `high_freq_weight=0.2` | 未记录 | 未记录 | 0.222915 / 0.188702 / 0.288078 | 0.444623 / 0.458991 | 0.469880 / 0.208348 | 0.482934 | 0.974992 | 当前主要对照基线 |
| 相对包络 loss 0.01 | `runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320` | `relative_envelope_weight=0.01` | 6 | 0.625069 | 0.219451 / 0.184972 / 0.280613 | 0.440967 / 0.457596 | 0.651732 / 0.278480 | 0.443979 | 0.975401 | 包络 MAE 小幅改善，但 peak RR 明显恶化，不通过 |
| 相对包络 loss 0.005 | `runs/tho_research_v2_patch_mixer_rawish_relenv0005/20260616_220652_976917` | `relative_envelope_weight=0.005` | 18 | 0.618409 | 0.222391 / 0.188642 / 0.286219 | 0.442177 / 0.457183 | 0.466059 / 0.207346 | 0.477779 | 0.975085 | peak RR 未恶化，但包络收益几乎为零 |

## 阶段结论

相对包络诊断是有价值的：它能把模型在局部增强/下降处的坏例暴露出来。但把它作为训练 loss 的收益不足。

`relative_envelope_weight=0.01` 有轻微降低 `relative_envelope_mae` 的效果，但 `rr_peak_abs_error` mean 从 0.469880 恶化到 0.651732，超过 0.15 bpm 的容忍阈值。`relative_envelope_weight=0.005` 消除了 peak RR 恶化，但包络指标几乎回到基线，说明继续调这个权重的边际价值很低。

当前决策：

- 保留 `relative_envelope_corr` 和 `relative_envelope_mae` 作为诊断指标。
- 不建议继续保留 `relative_envelope_loss` 作为默认训练目标。
- 后续优先查坏例数据质量、事件段/稳定呼吸段混合、输入与目标对齐，而不是继续放大相对包络 loss。

## 坏例观察

`relative_envelope_weight=0.005` 的最差相对包络 row：

| row | `samp_id` | 主要现象 |
|---:|---:|---|
| 12430 | 971 | 预测在强事件附近响应明显，但目标稳定呼吸段错配严重 |
| 12431 | 971 | 与 12430 相邻，疑似同一事件段连续影响 |
| 15210 | 1308 | 目标稳定振荡段与预测事件响应混合错配 |
| 15209 | 1308 | 与 15210 相邻，连续坏例 |
| 12909 | 972 | `rr_spec_abs_error` 和相对包络误差都偏高 |
| 15208 | 1308 | 与 15209/15210 同段 |
| 12914 | 972 | 相对包络误差高，但 spectrum 仍可接受 |
| 8273 | 952 | 相对包络误差高，需结合相邻窗口复核 |

这些坏例不是完全随机散布，而是集中到少数 `samp_id` 和相邻窗口。下一步应把分析重点从 loss 权重转到样本结构与对齐问题。

## 建议下一步

1. 生成坏例审计表：覆盖 top relative MAE 的 row、相邻窗口、`samp_id`、`allowed_losses`、有效率、RR 误差、频谱相似度和包络指标。
2. 对每个坏例扩展绘制前后 3-5 个窗口，确认是单窗口异常、连续事件段，还是某个 `samp_id` 系统性问题。
3. 检查输入 BCG 强事件与目标 THO 稳定呼吸段是否存在不可学习或错位现象。
4. 若坏例集中在事件冲击/状态切换段，考虑分桶评估或训练样本降权；若存在明显错位，优先修对齐。

## 相关产物

- `0.01` 最差图：`runs/tho_research_v2_patch_mixer_rawish_relenv001/20260616_005516_616320/plots_diagnostic_relenv_worst/`
- `0.005` 最差图：`runs/tho_research_v2_patch_mixer_rawish_relenv0005/20260616_220652_976917/plots_diagnostic_relenv_worst/`
- 详细计划：`docs/superpowers/plans/2026-06-15-relative-envelope-diagnostics.md`
