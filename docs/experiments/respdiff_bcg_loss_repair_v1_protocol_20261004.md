# RespDiff-BCG loss 修复验证 v1

## 当前状态与实验范围

2026-10-05 核对：**两个 objective 的 GPU 检查、seed=20260811 的 6400-update train/validation 均已由用户执行完成。** 两个 run 的完成回执、最终 checkpoint、17 项产物 hash、完整训练日志和 validation 集合核对通过。与原版同 seed 相比，RR 与 PCC 显著改善，epsilon-only 五项主指标均略优于 SNR 加权 FFT；但原始预测仍有千级极值，幅值比例偏差未解决。因此本轮为部分修复，不能宣称尖峰已根治；独立测试集未开放。具体结果见本文末尾。

本实验源于用户 2026-10-04 的“尝试修复”要求。实现阶段通过 11 项相关合成 CPU 测试，后续新训练由用户执行；本次结果核对只读取保存产物，不重新调用模型。

依据见 [固定 checkpoint 诊断](respdiff_bcg_checkpoint_diagnostic_20261004.md)：首次 t=49 的 epsilon 尾部残差被放大约 2193 倍；实测该 t 的频谱/噪声参数梯度范数比约 47927，两项方向近乎一致。修复候选针对已确认的额外 timestep 权重放大，是否改善尾部拟合与最终采样由新训练检验。

本轮预先定义两项 seed=20260811 的修复验证任务：

| Objective | 训练目标 | 角色 |
|---|---|---|
| `snr_weighted_fft` | mean noise MSE + 0.01 × mean(SNR(t) × per-sample FFT magnitude MSE) | 修复候选 |
| `epsilon_only` | mean noise MSE | 去除频谱项的对照 |

参考基线为已有 `train_b64_seed20260811_r01`，不重训该 identity。两个新任务均从相同 seed 初始化、训练 6400 updates；本轮仅定义这两个单 seed 任务，后续多 seed 扩展另行明确范围。单 seed 结果不作为多 seed 稳健性结论。

## 损失定义与稳定实现

对每个样本的 timestep，令 `SNR(t)=alpha_bar/(1-alpha_bar)`，`e=epsilon-epsilon_hat`：

\[
L_{\rm repair}=\operatorname{mean}(e^2)
+0.01\operatorname{mean}_B\left[\mathrm{SNR}_t\operatorname{mean}_{K,L}
(|F\hat x_0|-|Fx_0|)^2\right].
\]

权重先按样本应用，再对 batch 归约。训练 batch 内 t 不同，不能用 batch 平均 SNR 乘 batch 平均 FFT loss。

根据 `sqrt(SNR)*x0_hat = sqrt(SNR)*target + e`，频谱项按以下数学等价形式计算：

\[
\operatorname{mean}_{K,L}\left(
\left|\sqrt{\mathrm{SNR}_t}Fx_0+Fe\right|
-\left|\sqrt{\mathrm{SNR}_t}Fx_0\right|\right)^2.
\]

FFT 仍为完整正交 FFT（`norm=ortho`）。该实现避免先除极小的 `sqrt(alpha_bar)` 再计算谱差；训练前向加噪与 epsilon 预测参数化保持一致。

正交 FFT 与幅度操作的非扩张性质给出：对 epsilon 预测输出空间，频谱项的梯度范数不超过噪声 MSE 梯度范数的 0.01 倍（理想算术下）。**这不是网络参数梯度范数的统一上界**，因为网络 Jacobian 可能对两项方向产生不同作用。

对于此前每个 batch 统一使用单个 t 的固定 probe，新频谱梯度在数学上等于旧频谱梯度乘 SNR(t)。按已有测量解析换算，t=49 的参数梯度比从约 47926.9 变为约 0.00996；t=19 约为 0.01206。这是对同一 checkpoint/probe 的解析对应关系，不是新候选重新训练后的实测结果。

幅度 FFT 对相位及反号不敏感的性质仍然存在；本次加权只处理额外的 `1/SNR` 放大。减少高噪声步频谱权重是否反而削弱尾部校正，不能靠公式单独判断，因此保留 epsilon-only 对照和全量波形检验。

## 保持一致的条件

沿用 parent protocol `respdiff-bcg-v1-20261003` 的数据、父窗口/split、soft-z、20 Hz AA、30 秒片段与 15 秒 hop、reflect padding、13 块 Hann OLA、公共 Fourier 插值，以及六层 hidden=1024 的 source_fft 网络。

两项任务均为 batch=64、FP32、Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，6400 updates；LR 在 4480/6336 updates 后衰减。使用最终 checkpoint，在训练结束后进行完整 validation。推理固定六步 DDIM、N=1、相同 parent-keyed noise seed=20261003、相同 batch=64 和排序，保持公共五指标定义一致。独立测试集未开放。

本轮只改变训练目标，需重新训练。既有 checkpoint 与 v1 正式结果保留。

## 实现与产物

- 损失与合同：`resp_train/respdiff_bcg/repair.py`。
- 入口：`scripts/run_respdiff_bcg_loss_repair_v1.py`。
- 配置：`configs/respdiff_bcg_loss_repair_v1/snr_weighted_fft.yaml` 和 `epsilon_only.yaml`。
- 原始 v1 模型、loss 和采样实现保持原定义。共用 runner 新增可选 checkpoint schema/metadata 与指标 method 标签，默认调用行为保持 v1；历史运行复现按各自保存的源码快照定位代码版本。
- 新 checkpoint schema 为 `respdiff-bcg-loss-repair-v1`，`experiment` 字段记录 protocol、objective 和 seed，避免混淆损失版本。
- 终端/`run.log` 继续显示进度、loss、lr、ETA 与显存，`history.jsonl` 保存每步记录；source snapshot、data identity、父/片段清单、预测、逐父指标、summary 和 receipt 沿用完整追溯结构。

`snr_weighted_fft` 的日志字段：`loss_noise`、SNR 加权但尚未乘 0.01 的 `loss_fft_snr`、实际加入总损失的 `loss_fft_weighted`，以及仅记录的 `diagnostic_fft_raw`（原始 x0 FFT 误差，不参与优化）。总损失为 `loss_noise + loss_fft_weighted`。epsilon-only 只计算/记录噪声项。

## 用户执行命令

以下记录本轮已完成的 GPU 0 顺序执行命令：各 objective 先做三次更新的合成 GPU 检查，再进入 train/validation；任一命令失败即停止。所列 identity 已存在且完成，不再重跑或覆盖。

```bash
(
  set -e
  cd /home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction

  for objective in snr_weighted_fft epsilon_only; do
    repair_config="configs/respdiff_bcg_loss_repair_v1/${objective}.yaml"

    env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 \
      /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
      scripts/run_respdiff_bcg_loss_repair_v1.py gpu-check \
      --config "$repair_config" \
      --output "runs/respdiff_bcg_loss_repair_v1/gpu_${objective}_b64_r01"

    env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 \
      /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
      scripts/run_respdiff_bcg_loss_repair_v1.py train \
      --config "$repair_config" --confirm-training \
      --output "runs/respdiff_bcg_loss_repair_v1/${objective}_seed20260811_r01"
  done
)
```

执行完成后，先核对 6400 updates、来源和完整 val 身份，再与 v1 同 seed 比较五项主指标及保存的 20 Hz 输出：每父最大幅值分布、呼吸带能量占比、稀疏尖峰能量占比和原异常 row 10382。训练 loss 数值下降不能代替这些重建检查；两个 objective 的 loss 定义不同，不直接比较总 loss 数值判优。

## 已执行验证

`tests/test_respdiff_bcg_repair.py` 加旧入口的完整合成流程与 6400-update/LR 测试，共 **11 passed，17.63 秒**。覆盖：

- 50 个混合 timestep 下，稳定公式与显式逐样本 SNR×FFT 公式的数值及梯度一致。
- 噪声输出空间的梯度尺度上界和零误差情况。
- epsilon-only 与手算 MSE 一致，模型初始化权重与 v1 相同。
- 两个 objective 的 NPZ 合成训练→采样→公共指标流程，新 checkpoint schema/目标标签、输出不可覆盖与真实数据入口门禁。
- 配置拒绝无关预算、数据 split、采样和权重漂移；旧入口 checkpoint/指标语义与 6400 updates 保持兼容。

两个配置差异检查确认，只有 `objective.name` 和 `objective.fft_weight` 不同。实现验收阶段未进行新 GPU 执行或新真实数据访问；后续用户执行结果见下节。

## 同 seed 结果与完整性核对（2026-10-05）

比较来源：

- 原版：`runs/respdiff_bcg_v1/train_b64_seed20260811_r01`。
- SNR FFT：`runs/respdiff_bcg_loss_repair_v1/snr_weighted_fft_seed20260811_r01`。
- epsilon-only：`runs/respdiff_bcg_loss_repair_v1/epsilon_only_seed20260811_r01`。

两个新 run 的 receipt 均为 complete，各 17 项产物 SHA256 匹配，包括 checkpoint、源码快照、history、波形及指标。checkpoint schema 为 `respdiff-bcg-loss-repair-v1`、selector 为 `final_update`、update 为 6400，objective 元数据正确。history 均连续 6400 条，记录数值有限。

两个候选的源码 hash 相同，resolved config 除 objective 外相同；训练片段顺序、最终 CPU/CUDA 训练 RNG 状态相同。两个候选与原版的 data identity、train/val 父清单、片段清单、validation 批次文件 hash 相同。保存的 validation row 顺序和 reference 数组逐元素一致。每个候选均有 2675 个完整父窗口，五项主指标全部有限，逐父均值与原 summary 一致。

两项工程检查均在 RTX 4070 Ti SUPER 上通过，训练 batch=64、推理检查 batch=1/64；峰值 allocated 显存约 7.10 GiB。

| 指标（逐父 direct mean） | 原版 | SNR 加权 FFT | epsilon-only |
|---|---:|---:|---:|
| Whole RR MAE ↓，bpm | 13.923317 | 2.635463 | 2.576107 |
| Local RR MAE ↓，bpm | 13.271213 | 2.767897 | 2.670269 |
| 包络轨迹 MAE ↓ | 0.553600 | 0.441631 | 0.437797 |
| 全局调制误差 ↓ | 1.076499 | 0.817223 | 0.804887 |
| 有符号 PCC ↑ | 0.051164 | 0.446961 | 0.453306 |

两种目标均明显改善了当前同 seed 的五项指标；epsilon-only 略优，但单 seed 的小差异不足以形成多 seed 稳健性结论。当前尚未观察到 SNR FFT 相对 epsilon-only 的额外收益。

末 100 updates 的 noise MSE：原版约 0.28257，SNR FFT 约 0.01931，epsilon-only 约 0.01930。SNR FFT 的末 100 步加权频谱项约 0.0001006，总 loss 约 0.01941。这里只比较相同定义的 noise MSE，不把不同目标的总 loss 当作性能排序依据。

## 残余问题：原始幅值与尖峰

后续关于手动滤波、末期训练走势和 final checkpoint 的检查见 [滤波与继续训练诊断](respdiff_bcg_postfilter_probe_20261005.md)。该记录只比较固定小规模保存输出，不替换本节完整 validation 结果。

统计覆盖全部 2675 个保存的 20 Hz 父窗口。reference 使用与训练相同的 AA 降到 20 Hz。频带占比按逐父去均值能量计算；“最大 1%”为每父能量最大的 36/3600 个点。下表按父窗口取中位数或明确标注的 P95/全体最大值：

| 原始输出诊断量 | 原版 | SNR 加权 FFT | epsilon-only |
|---|---:|---:|---:|
| 呼吸带能量占比，中位数 | 11.47% | 86.61% | 87.05% |
| >2 Hz 能量占比，中位数 | 68.22% | 1.77% | 1.55% |
| 最大 1% 采样点能量占比，中位数 | 92.40% | 63.76% | 62.28% |
| 每父最大绝对幅值，中位数 | 45.29 | 44.76 | 42.56 |
| 每父最大绝对幅值，P95 | 459.94 | 379.47 | 362.49 |
| 全体最大绝对幅值 | 1861.78 | 1376.24 | 1301.39 |
| 原异常 row 10382 最大绝对幅值 | 1861.78 | 1109.16 | 988.00 |
| prediction std / target std，中位数 | 1.390 | 2.968 | 2.882 |
| prediction std / target std，P95 | 10.17 | 21.58 | 20.71 |

两个候选的新最大极值均在 row 12637。新模型的高频能量显著减少、呼吸带结构和相关性改善，但仍有强烈幅值异常。标准差比例中位数及 P95 反而变大，因此不能把评价改善直接解释为原始幅值已恢复。公共 Pi 包含幅度归一化；其指标必须与这些原始输出诊断一起理解。

在本轮范围内，可把 epsilon-only 作为后续残余尖峰排查的简洁候选，同时保留 SNR FFT 为对照；尚未据此扩展训练 seed、修改 sampler 或访问独立测试集。要证明剩余异常仍源于同一高噪声尾部机制，需要对新 checkpoint 做新的有界诊断，当前本次汇总没有执行该诊断。

## 本次比较产物

入口：`scripts/summarize_respdiff_bcg_loss_repair_v1.py`。本次只读已保存产物的 CPU 分析已完成，输出 identity：

`runs/respdiff_bcg_loss_repair_v1/comparison_seed20260811_20261005_r01`

其中 `validation_comparison.csv` 和 `summary.json` 保存数值，`per_parent_signal_diagnostics.csv` 保存逐父幅值/能量统计；`fixed_outlier_waveforms.png` 固定展示原异常 row 10382 的三个版本，`spectrum_comparison.png` 展示全部父窗口的归一化频谱中位数；`manifest.json` 保存来源和产物 hash。原 run 未补写或覆盖。

| Objective | 最终 checkpoint SHA256 |
|---|---|
| snr_weighted_fft | `e96cc9ea03de5023b810f3c5b8c28f847530dd43a183e4708e5e162240375ebc` |
| epsilon_only | `64a29fc3e63d6250a6175b2c73c6ff221a7fb982f8f03a9fad1262f984383485` |
