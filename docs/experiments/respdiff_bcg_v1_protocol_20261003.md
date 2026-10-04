# RespDiff-BCG v1：20 Hz 父窗口重建

## 当前状态与授权

2026-10-04：**batch=64 的 GPU 工程检查及 seeds 20260811/20260812/20260813 的 6400-update train/validation 均已完成。** 三个 run 的完成回执、最终 checkpoint 元数据、全部产物 hash、训练步数及 validation 身份核对通过；结果见本文末尾。当前三 seed 重建相关性较低，不能据此形成有效重建结论；独立测试集未开放。本轮已完成的训练与评价不自动重跑。

本版本由用户确认 A+B+C+D+E+F+G+H，H 明确为固定 **6400 optimizer updates**。前期 batch=32 的工程检查成功，其真实训练在 548 updates 后被用户中断；用户随后授权清理失败产物，并改为尝试 batch=64、显示进度与日志。

按本次用户清理授权，确认无活动训练进程后，已删除以下三个失败/中断目录，共约 170 MiB。其运行身份不复用：

| 已清理目录（相对 `runs/respdiff_bcg_v1/`） | 终止原因 | 已完成 updates |
|---|---|---|
| `gpu_fp32_b128_b64_r01` | backward OOM | 0 |
| `train_seed20260811_r01` | 默认 batch=128，backward OOM | 0 |
| `train_b32_seed20260811_r01` | 用户 KeyboardInterrupt | 548 |

清理前核对的 batch=128 OOM：设备总显存 15.56 GiB，进程占用 12.32 GiB、剩余 3.24 GiB，申请 4.00 GiB 失败。

成功工程检查使用独立配置 `configs/respdiff_bcg_v1/gpu_b32_b64.yaml`，仅将 training batch 从 128 调为 32，inference batch 保持 64。`runs/respdiff_bcg_v1/gpu_fp32_b32_b64_r01/receipt.json` 为 complete；`gpu_check.json` 记录 RTX 4070 Ti SUPER、三次训练更新、batch=1/64 的推理和峰值 PyTorch allocated 显存 4169157632 bytes（约 3.88 GiB）。该检查已完成，无需重跑。

实际采用 `configs/respdiff_bcg_v1/gpu_b64_b64.yaml`，training/inference batch 均为 64；另外两个 seed 对应 `train_b64_seed20260812.yaml` 和 `train_b64_seed20260813.yaml`。batch=64 工程检查位于 `runs/respdiff_bcg_v1/gpu_fp32_b64_b64_r01`，receipt 为 complete，峰值 PyTorch allocated 显存 7624296960 bytes（约 7.10 GiB）。命令之间不会自动继承配置，检查和训练均显式传入各自配置。

batch=64、6400 updates 的片段暴露量约为 batch=32 的两倍，并改变每批统计及优化轨迹；均值归约只保证损失定义不随 batch 大小成倍缩放。采用新配置从头执行独立训练。

实现位置为独立工作树 `/home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction`，分支 `codex/respdiff-paper-settings`。本协议只约束 `respdiff_bcg_v1`。源码对齐版的参数证据由 `respdiff_parameter_alignment_20261003.md` 管理。

## 科学合同

这是经用户确认的 BCG 适配版。原 RespDiff 来源为 commit `3ff05545c34f67e1ad415e36e4daea80269880b0`；网络实现复用本地 `resp_train/respdiff/model.py`。以下适配需要独立训练，既有源码对齐版或 APOR 的 checkpoint/结果不能直接作为本版本的结果。

| 项 | 固定定义 |
|---|---|
| A 损失 | `mean((epsilon - epsilon_hat)^2) + 0.01 * mean((abs(FFT(x0_hat)) - abs(FFT(target)))^2)`；FFT `norm=ortho`，两项均对 batch/channel/time 取均值 |
| B 幅值 | 继承 `bcg_rawish_segment_soft_z_key` 与 `target_waveform_segment_soft_z_key` 的现有父窗口值 |
| C 降采样 | 输入与 target 同用 100→20 Hz，255 taps、9 Hz cutoff、Kaiser β=8.6、`scipy.signal.resample_poly(up=1, down=5, padtype=line)` |
| D 分块与重建 | 180 秒父窗口为 3600 点，两端 reflect 300 点；600 点片段、300 点 hop，共 13 块；periodic Hann 加权 OLA 后除以权重和，裁取 `[300:3900]`，公共 Fourier 插值至 18000 点 |
| E 扩散 | 训练 50 步，linear beta `[0.0001, 0.5]`；DDIM η=0，时间步 `[49,39,29,19,9,0]`，终点 alpha_bar=1 |
| F 采样 | 每个片段一个 realization；evaluation noise seed 默认 `20261003` |
| G 初始噪声 | 按 split、父 row ID、noise seed 生成独立 4200 点高斯场；13 个片段切片共享重叠位置的初始噪声 |
| H 预算 | 固定 6400 次 optimizer update；保存最后一步 checkpoint，训练结束后完成一次全量 validation |

训练时对补边区间同样计算损失。反射只用于信号，4200 点初始噪声独立生成。每个片段分别完成反向过程，重叠处后续预测可以不同，再通过 OLA 合成。预测保留 soft-z 幅值空间。公共插值使用 `resp_train.crd.spectral_ops.fourier_interpolate`，含其偶数长度 Nyquist bin 处理规则。

网络保留 source_fft 的六层双向 RNN、hidden=1024、output=128、fine/coarse 编码器及 diffusion embedding。优化器 Adam：lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；FP32。训练 batch 默认 128，shuffle、保留尾批，梯度每个 batch 更新一次。原 400-epoch 调度的 70%/99% 位置映射到本预算的 4480/6336 updates：在 optimizer.step 后 scheduler.step，updates 4481、6337 起分别使用 1e-5、1e-6。该映射属于本适配合同。

## 数据身份与评价

使用配置指定的 research_v2 索引及现有 train/val 筛选，保留全部合格父窗口，sample seeds 为 20260610/20260611。先核对 row、subject 与源记录隔离，再访问父窗口波形。相对 NPZ 路径按索引目录规范化；每个片段记录父身份和相对 20 Hz 区间。13 个片段是计算单位，统计单位仍为 180 秒父窗口。

验证按父 row ID 排序后的片段顺序连续组 batch，默认 batch=64，保留尾批。source 网络的 BatchNorm 使用当前 batch 统计，因此 batch 组成影响预测。运行记录保存有序片段列表、batch size 和尾批大小。共享噪声确保噪声身份可复现；变更 inference batch 需要重新评价。变更 training batch 同时改变 6400 updates 的样本暴露量，应作为独立配置身份运行。

原始 20 Hz/100 Hz 预测和原始 100 Hz target 保存在 `validation_waveforms.npz`。评价调用现有 `metrics.task.evaluate_task_predictions` 与 `summarize_task_metrics`，使用公共 Pi 和逐 sample direct mean，主指标为 Whole RR、Local RR、包络轨迹 MAE、全局包络调制误差、lag-aware signed PCC。逐父窗口 CSV 保留 eligibility/degeneracy 标记；target 不合格的指标空值按公共合同保留并由 summary 的 n 计数说明。合格样本的主指标非有限值显式失败。

## 运行与产物

依赖沿用当前项目环境。每条命令的输出目录必须未存在，失败默认保留，清理需用户授权。下列 batch=64 命令记录本轮已完成运行的来源，已有 identity 不再执行。

```bash
cd /home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction

# CPU disposable NPZ + 小网络：两次更新、完整 13 块重建和公共指标。
/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_respdiff_bcg_v1.py synthetic-smoke \
  --output /tmp/respdiff_bcg_v1_cpu_r01

# batch=64 的三次 GPU 更新与 batch=1/64 的六步采样检查。
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/run_respdiff_bcg_v1.py gpu-check \
  --config configs/respdiff_bcg_v1/gpu_b64_b64.yaml \
  --output runs/respdiff_bcg_v1/gpu_fp32_b64_b64_r01

# 上述检查通过后：batch=64，seed=20260811，6400 updates 与完整 validation。
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/run_respdiff_bcg_v1.py train \
  --config configs/respdiff_bcg_v1/gpu_b64_b64.yaml \
  --confirm-training \
  --output runs/respdiff_bcg_v1/train_b64_seed20260811_r01
```

GPU 检查验收：`receipt.json` 为 complete；`gpu_check.json` 记录 native 网络 train batch 的三次更新、batch=1/64 的采样和峰值显存；各 loss、gradient、prediction 有限。本机 batch=32/64 已通过，默认 batch=128 已观察到 OOM。需要调整 batch 时复制配置并显式传 `--config`，新输出 identity 保存实际参数；配置允许改 seed、device 和 batch size。

正式运行验收：history 恰好 6400 条；`final.pt` 的 schema=`respdiff-bcg-v1`、selector=`final_update`、update=6400；val manifest 每父窗口恰好 13 块，输出每父窗口 3600/18000 点；指标 CSV 与 val 父集合一致；receipt 为 complete。CPU smoke 的执行记录明确标记小网络、两次更新、batch=2/8，不能作为正式训练或 GPU 资源证据。

启动时在终端明确显示实际 mode、seed、device、训练/推理 batch、updates 和输出位置。训练输出首步、每 25 步、末步进度；超过 30 秒未报告时在下一步完成后输出。内容包括 loss/noise/FFT、lr、速度、elapsed、ETA、CUDA allocated/reserved 显存。验证按采样、父窗口重建和指标计算分别报告进度，末尾显示指标均值。常规日志同时写入终端与 `run.log`（Asia/Shanghai 时间），异常堆栈保存在 `run.log`，每步结构化记录仍写入 `history.jsonl`。

每次运行保存 resolved config、源码 hash 与源码快照、commit、运行环境、命令、seed、父/片段清单、完整 history、最终模型/optimizer/scheduler/RNG 状态、原始预测、逐父指标、汇总和产物 SHA256。数据记录含完整索引 SHA256 与被选中源文件的 size/mtime，结束时再次核对；这些文件 stat 不是 NPZ 内容摘要。运行失败保存 `failure.json` 的阶段与异常，成功后才写 receipt。当前入口从头执行一个独立 attempt。

数据读取沿用 ResearchV2 的整晚数组缓存，宿主内存需求仍需在用户执行真实数据阶段观察。新片段层另有 64 个父窗口的 LRU 缓存。当前合成 GPU 检查只覆盖 GPU 算子与 batch 显存，不能证明真实数据 I/O 吞吐或宿主内存适配。

## 已执行验证

2026-10-03，在项目 `.venv` 中执行 `PYTHONPATH=. python -m pytest tests/test_respdiff_bcg.py -q`：**15 passed，14.19 秒**。覆盖损失及梯度解析对照、batch 均值归约、DDIM 独立公式对照、噪声重叠/RNG 隔离、OLA 边界脉冲与常量重建、抗混叠频率/幅值/对齐、6400 次 scalar optimizer 更新及 LR 边界、NPZ 端到端流程、失败/覆盖保护、配置合同、相对路径、主体隔离及不可访问的 test fixture。

日志改动后运行两个定向 CPU 测试：端到端 NPZ 流程及 6400-update/LR 边界，**2 passed，14.86 秒**。额外验证终端/文件包含各阶段进度、loss 与 ETA，日志 SHA256 与 receipt 一致，异常堆栈已落盘，分批计算的指标与直接调用公共评价结果一致。

## 三 seed validation 结果（2026-10-04）

后续保存输出与训练目标的只读诊断见 [输出与 FFT loss 诊断](respdiff_bcg_output_fft_diagnostic_20261004.md)，含原始波形/频谱图、尖峰统计、timestep 相对权重推导及证据边界。

用户后续授权的 GPU 0 有限排查已完成，见 [固定 checkpoint 六/五十步与梯度诊断](respdiff_bcg_checkpoint_diagnostic_20261004.md)。该定向诊断不替换本节全量 validation 结果。

来源为 `runs/respdiff_bcg_v1/train_b64_seed{20260811,20260812,20260813}_r01/` 中的完成回执、`validation_per_parent.csv` 与 `validation_summary.csv`。本次只读核对已有产物，未执行训练、模型推理或独立测试集评价。

三个 run 均满足：

- 17 项 receipt 产物 SHA256 全部匹配，含 checkpoint、history、源码快照、预测波形与指标。
- `final.pt` 为 `respdiff-bcg-v1`，`update=6400`、`selector=final_update`，scheduler 最后一步为 6400。
- history 连续 6400 条，loss/noise/FFT/lr 均有限，学习率边界符合 4480/6336 的约定。
- 源码 hash 相同，resolved config 除 training seed 外完全一致；数据 identity 与 val 父清单 hash 相同。
- train 为 10141 个父窗口、131833 个片段；每个 seed 实际处理 409579 个片段，全部训练片段至少访问一次，包含保留尾批。
- validation 为相同的 2675 个父窗口、34775 个片段，batch=64；片段次序完整，逐父指标身份与清单一致。
- 五个主指标每个 seed 均为 2675 个有效值，逐父均值与已有 summary 一致。

表中每个 seed 先按父窗口 direct mean；最后一行为三 seed 均值 ± 样本标准差（ddof=1），不是置信区间。

| Seed | Whole RR MAE ↓ (bpm) | Local RR MAE ↓ (bpm) | 包络轨迹 MAE ↓ | 全局调制误差 ↓ | 有符号 PCC ↑ |
|---|---:|---:|---:|---:|---:|
| 20260811 | 13.923317 | 13.271213 | 0.553600 | 1.076499 | 0.051164 |
| 20260812 | 13.919834 | 13.292655 | 0.553256 | 1.079246 | 0.048651 |
| 20260813 | 13.891075 | 13.341791 | 0.537756 | 1.019616 | 0.044827 |
| mean ± SD | 13.911409 ± 0.017695 | 13.301887 ± 0.036183 | 0.548204 ± 0.009050 | 1.058454 ± 0.033662 | 0.048214 ± 0.003191 |

训练日志首 100 / 末 100 updates 的平均总 loss 分别为：seed 20260811，178.2928 / 0.4149；seed 20260812，167.4232 / 0.4139；seed 20260813，159.1980 / 0.4437。这是随机训练片段、噪声和 timestep 上的记录，不是固定 probe 或 validation loss。优化目标下降，但本配置的最终六步采样重建 PCC 仍接近零，RR 误差较大，且三个 seed 一致偏弱。已有产物完整性检查通过不能排除方法或实现层面的其他问题；仅凭本轮汇总尚不能确定训练预算、采样路径或目标函数各自的影响。

| Seed | 最终 checkpoint SHA256 |
|---|---|
| 20260811 | `c4e57bf59fef460839a02ee12795986d6fc062c498514ba68de0767919657a15` |
| 20260812 | `36b7904871fa6fe345bf327cefb9187755f11412f775773786a04574b41267b9` |
| 20260813 | `1f98b0af4e60b9474d4d455d4b8ebc221c64602c6f405e62165f796abe5bece3` |
