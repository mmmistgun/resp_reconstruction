# RespDiff-BCG 双低通三臂 validation 结果

2026-10-07 状态：三臂 seed=20260811、B64、6400-update train/validation 均完成；本轮关闭。GPU 合成检查包含 B64 梯度 probe，修复后全部通过。证据限于本轮单 seed、2675 个 validation parent。独立测试集未开放。

本轮结论：`source_equivalent` 在五项主指标上均优于另外两臂。`snr_resp_spectral` 比 epsilon-only 有小幅改善，且确实保持了较小的谱项参数梯度比，但三臂原始输出均残留千级尖峰。当前结果不支持把 SNR 呼吸带目标认定为本轮性能最优或尖峰问题已解决。

## 产物与身份核对

- 执行协议：[baseband v1](respdiff_bcg_baseband_v1_protocol_20261006.md)。任务为双 1 Hz 低通后的 BCG→THO，训练与 validation 参考均使用低通 THO。
- 回传根目录：`runs/respdiff_bcg_baseband_v1_from_t630_20261007/`；三臂目录为 `<objective>_seed20260811_v1/`。
- 运行代码版本：`278bea66dea5a6b3d3738752c548561ae13a70b3`。三臂 source file hashes 一致，resolved config 除 objective 外一致。
- 三份 receipt 均 complete，每份所列 22 项产物 SHA256 全部匹配，共 66 项；包含 checkpoint、源码快照、完整历史、波形及逐 parent 指标。
- 每臂 history 恰为连续 update 1…6400，probe 30 条。三臂训练 chunk 顺序与 timestep 逐 update 一致；train/val parent、chunk、validation batch、probe 输入及数据身份文件哈希一致。
- validation 输出与参考均为有限的 `(2675,18000)`；三臂参考数组字节 SHA256 一致：`4e82d87d3f99c2abda11d74c7e259e0d10acce4bd7544508896e465477421ee7`。
- 三项成功 GPU receipt 对应 epsilon-only/source-equivalent 的 `gpucheck_v1` 和 SNR 的 `gpucheck_v2`。包含 probe 的峰值 allocated 约 7.10 GiB，设备 RTX 2080 Ti。SNR `gpucheck_v1` 的 OOM lifecycle 原样保留。

本次只读取回传产物、核验哈希与汇总已有逐 parent 记录；未重训、重采样或重评公共指标。

| objective | final.pt SHA256 |
| --- | --- |
| epsilon_only | `85b1e4ad664b0426bab882f927e0b0c059cabaf94cc4a2cdd6422e6870977436` |
| source_equivalent | `08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6` |
| snr_resp_spectral | `8b8e1eee6a7ae722a75141a1bf59c83c0cfda4097f8b13ca183335bd3e30cd33` |

## 完整 validation 五项主指标

以下均为相同 2675 个 parent 的均值，五项指标均有完整有效计数；最终 checkpoint 由固定 update 选择。

| 指标 | epsilon_only | source_equivalent | snr_resp_spectral |
| --- | ---: | ---: | ---: |
| Whole RR MAE ↓，bpm | 2.6734 | **1.4143** | 2.4518 |
| Local RR MAE ↓，bpm | 2.8059 | **1.7185** | 2.5979 |
| 包络轨迹 MAE ↓ | 0.5022 | **0.3250** | 0.4818 |
| 全局包络调制误差 ↓ | 0.9822 | **0.4971** | 0.9208 |
| lag-aware signed PCC ↑ | 0.4173 | **0.5313** | 0.4304 |

source-equivalent 相比 SNR 候选：Whole RR 降低 42.3%，Local RR 降低 33.9%，包络轨迹误差降低 32.5%，全局调制误差降低 46.0%，PCC 增加 0.1010。逐 parent 严格改善比例分别为 57.7%、67.2%、90.7%、77.3%、72.4%，故优势不只来自某个极端窗口。窗口可能相关，这些比例是描述性统计，不作为独立样本显著性检验。

SNR 候选相比 epsilon-only：Whole RR 降低 8.3%，Local RR 降低 7.4%，包络轨迹误差降低 4.1%，全局调制误差降低 6.3%，PCC 增加 0.0131。它提供了一定辅助增益，但仍落后于 source-equivalent。

## 原始输出幅值与频带诊断

表内能量比、std 比均为逐 parent 值的均值；括号另列中位数。幅值来自保幅的 100 Hz 重建输出，未使用指标内部归一化。

| 诊断 | epsilon_only | source_equivalent | snr_resp_spectral | THO target |
| --- | ---: | ---: | ---: | ---: |
| 呼吸带能量比，% | 81.44 | 76.87 | 78.75 | 99.60 |
| >1 Hz 能量比，% | 10.18 | 16.92 | 13.22 | 0.0665 |
| prediction/target std 均值 | 6.987 | 2.639 | 6.600 | 1 |
| prediction/target std 中位数 | 4.410 | 1.319 | 4.152 | 1 |
| parent max 的 P95 | 375.48 | 274.92 | 411.04 | 9.37 |
| parent max 的 P99 | 712.14 | 597.68 | 732.29 | 11.71 |
| 全体最大绝对幅值 | 1231.61 | 1241.20 | 1347.59 | — |
| top 1% 时间采样点能量占比，% | 60.11 | 49.17 | 59.99 | 8.79 |

source-equivalent 改善了典型幅值比例及幅值分布的 P95/P99，但全体最坏幅值仍为千级，且其 >1 Hz 能量占比最高。能量占比是相对量，不能直接推断绝对高频能量大小。SNR 候选相对 epsilon-only 的主指标改善没有同步带来极端幅值改善。

三臂 top 1% 时间点携带约 49%–60% 的能量，target 为 8.8%。低通输入与 target 没有把模型输出强制限制到低频，也没有消除尖峰。

## 固定 timestep 梯度与第一步采样

最终 update=6400 的 `r_t=||grad(weighted_spec)||/||grad(noise)||`：

| t | source_equivalent | snr_resp_spectral |
| --- | ---: | ---: |
| 0 | 3.85e-9 | 0.00277 |
| 9 | 1.87e-5 | 0.01588 |
| 19 | 1.46e-4 | 0.02366 |
| 29 | 0.00738 | 0.00758 |
| 39 | 0.79120 | 0.00606 |
| 49 | 374.55 | 0.00735 |

epsilon-only 的 r_t 恒为零。SNR 候选在全部 30 个固定 probe 中 r_t 约为 0.00207–0.03998，即谱项参数梯度范数约占主损失的 0.21%–4.00%；本轮固定 probe 支持其辅助梯度角色。source-equivalent 即使系数为 0.01/128，t=49 仍明显由谱项主导。

| 最终固定 train probe | epsilon_only | source_equivalent | snr_resp_spectral |
| --- | ---: | ---: | ---: |
| q_sample t=49 epsilon residual RMS | 0.02288 | 0.01742 | 0.02364 |
| reverse 第一步 x0_hat 绝对值 P99.9 | 320.13 | 393.44 | 333.84 |
| reverse 第一步 x0_hat 最大绝对值 | 1315.36 | 1348.69 | 1362.08 |

上述 reverse 第一步来自固定 train probe 的真实初始噪声，不能当作 validation 全集幅值分位数。它确认三臂在第一次 x0_hat 计算时已经存在千级峰值。SNR balancing 改善了训练目标的 timestep 梯度尺度，但没有改变 epsilon→x0 的低 SNR 逆映射；这两项问题需要分别判断。

source-equivalent 的高噪声步残差较小，与其更强调高噪声步误差的目标一致；但三臂同时改变谱项频带、窗口与 SNR 权重，当前结果不能把全部性能差异唯一归因于其中一个因素。

## 固定异常窗口

| row / 量 | epsilon_only | source_equivalent | snr_resp_spectral |
| --- | ---: | ---: | ---: |
| row 10382 最大绝对幅值 | 1100.35 | 1233.41 | 1202.95 |
| row 10382 std 比 | 29.24 | 32.98 | 31.91 |
| row 10382 >1 Hz 能量比，% | 92.89 | 93.90 | 93.25 |
| row 12226 最大绝对幅值 | 1231.61 | 1241.20 | 1347.59 |

row 10382 的最强尖峰三臂均位于 12.75 s，均为负峰；相同时刻 target 约 0.1087，target 全窗最大绝对幅值约 2.2634。它距最近 15 s hop 边界 2.25 s。

三臂全体最大绝对幅值都来自 row 12226，峰值同在 14.89 s，target 当时约 −0.3208，距 15 s hop 边界 0.11 s。这说明异常在三臂中有共同定位；共享输入和初始噪声也可能导致共同位置，单凭这一现象不能归因于 OLA 或某个具体网络模块。

## 后续判断边界

本轮排序以 source-equivalent 为最强 validation 候选，保留三臂固定结果。SNR 候选达到了参数梯度平衡目的，未达到性能最优或消除异常幅值的目的。三臂均不能宣称已经解决保幅波形重建问题。

这是单 seed 结果，不能外推多 seed 稳健性。旧未低通 target 结果属于不同任务数值合同，不与本表直接混排；本轮也没有同 target 的“只改低通”独立因素对照，不能据此单独判断双低通的净增益。
