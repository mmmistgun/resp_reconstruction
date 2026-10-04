# RespDiff-BCG loss 修复验证 v1

## 当前状态与实验范围

2026-10-04，用户在固定 checkpoint 诊断后要求“尝试修复”。已实现逐样本 SNR 加权 FFT 候选与 epsilon-only 对照，通过 11 项相关合成 CPU 测试。**尚未执行这两个候选的 GPU 检查、训练或 validation，尚无模型重建已修复的结论。** 前一轮 GPU 0 授权只覆盖已完成的有限诊断；本轮训练由用户执行下文命令。

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

以下在 GPU 0 顺序完成两个 objective，各自先做三次更新的合成 GPU 检查，再进入 train/validation；任一命令失败即停止。所有输出目录须未存在。

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

两个配置差异检查确认，只有 `objective.name` 和 `objective.fft_weight` 不同。本次未进行新 GPU 执行或新真实数据访问。
