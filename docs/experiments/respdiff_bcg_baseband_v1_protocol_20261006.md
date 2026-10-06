# RespDiff-BCG respiratory baseband v1

协议 ID：`respdiff-bcg-baseband-v1-20261006`。

当前状态：按用户 2026-10-06 定案实现；初版 41 项 CPU 定向测试通过。用户在 RTX 2080 Ti 上运行主候选 GPU 检查，B64 训练与采样通过，参数梯度 probe OOM；修复及重试说明见末节。正式训练和真实 validation 尚未执行。真实运行由用户执行或另行明确授权代跑。本协议仅开放 train/validation。

## 科学合同与数据

研究任务为 BCG respiratory baseband → THO respiratory waveform。沿用 research-v2 soft-z 数据、父窗口身份、train/val 准入和隔离检查。BCG 与 THO 均先在完整 180 s、100 Hz 父窗口进行 1 Hz Butterworth 双向低通，然后执行原 255-tap Kaiser 抗混叠 100→20 Hz 降采样、两侧各 300 点 reflect padding、600 点 chunk 与 300 点 hop。每 parent 共 13 块；输出用 periodic Hann 归一化 OLA，保留 3600 点，再 Fourier interpolation 至 18000 点。

低通固定为 8 阶 Butterworth、`sosfiltfilt(padtype='odd', padlen=27)`，float64 计算后转 FP32，不再归一化。这是 Butterworth + filtfilt 的 SOS 数值实现；1 Hz 是单向设计截止频率，双向滤波使幅频响应平方，并非理想砖墙。边缘延拓发生在 parent 边界；不在 chunk 边界重新滤波。系数 SHA256、SciPy 版本及完整配置保存在每次来源记录中。

训练 target 和 validation 的 `tho_ref` 均为上述 100 Hz 低通 THO；公共五项指标算子、eligibility、聚合方法与 row identity 沿用原合同。输入与数值 target 已改变，三个 objective 均须从头训练。旧 checkpoint、结果及失败 lifecycle 保留，不作为本轮同 target 对照，也不修改其历史指标。

## 固定三臂

各臂主损失 `loss_noise = mean((epsilon - epsilon_hat)^2)`，在 B、K、L 上全部取 mean。

| objective | loss_spec | weighted contribution |
| --- | --- | --- |
| epsilon_only | 0（未启用） | 0 |
| source_equivalent | 原作者全频、无 Hann、正交 FFT magnitude MSE(x0_hat,target) | `(0.01/128) * loss_spec` |
| snr_resp_spectral | 逐样本 SNR × 呼吸带 Hann-rFFT magnitude MSE(x0_hat,target)，再 batch mean | `0.01 * loss_spec` |

正式候选为 `snr_resp_spectral`。600 点、20 Hz 的频率间隔是 1/30 Hz；inclusive 0.05–0.70 Hz 选中 bin 2…21，共 20 个。分析窗为 `hann_window(600, periodic=True)`，与 OLA 同定义，`rfft(norm='ortho')`，不再除窗 RMS。取所选 bin 幅度差平方的 mean，不加入单边谱能量倍数。

令 `e=epsilon-epsilon_hat`，稳定计算 `R=sqrt(SNR)*rFFT(w*target)`、`E=rFFT(w*e)`，然后对所选 bin 计算 `(|R+E|-|R|)^2`。逐样本 SNR 在归约前生效，训练损失不显式构造 x0_hat。source-equivalent 也用稳定恒等式计算全频幅度差，再逐样本除 SNR，以保留 raw FFT objective；系数的 128 是作者参考 batch 常数，不随本轮 batch=64 变化。

source-equivalent 仅控制作者源码的相对 loss balance；由于预处理、输入模态和训练合同不同，不声称是完整作者复现。它仍保留低 SNR 放大，作为既定控制运行。SNR balancing 消除了正式候选中解析的 1/SNR 放大，不保证经模型 Jacobian 后参数梯度范数始终小于主损失；以 r_t 实测。固定 Hann 后，幅度项也不再严格循环平移不变，仍不足以唯一恢复相位与极性。波形监督由 epsilon 主损失承担。

## 训练和采样

三臂固定 seed=20260811、相同初始化与数据 shuffle；原 6 层双向 RNN、hidden=1024、output=128；FP32；Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0；train/inference batch=64；6400 updates，LR milestones=[4480,6336]、gamma=0.1。训练完成后仅保存 final-update checkpoint 并做完整 validation，不进行中途选择。

沿用 50 步 beta schedule 与 DDIM [49,39,29,19,9,0]，N=1，parent-keyed noise seed=20261003，输出保幅。三份配置除 objective 外相同；CLI 仅允许更改执行用 cuda:N，其余合同校验拒绝偏离。本轮固定一个 seed，不自动扩展矩阵。

## 诊断与解释

每 update 的 `history.jsonl` 保存 loss_noise、loss_spec、loss_spec_weighted、total、全部 timestep、chunk indices、epsilon residual 的 RMS、绝对值 P99/P99.9/max、lr 和 update。

固定 update=[0,1600,3200,4800,6400] 执行 probe，取 train 前 64 个全局 chunk，独立 CPU generator seed=20261006 固定 forward noise，遍历 t=[0,9,19,29,39,49]。逐 t 计算完整可训练参数上的 `r_t=||grad(weighted_spec)||/||grad(noise)||`；epsilon-only 的分子为零。模型参数、optimizer、已有 .grad、模式、buffer 和全局 torch RNG 不因 probe 改变。probe 输入及身份单独保存。

梯度执行采用两个独立的完整 B64 前向，各自反向后释放计算图。两次使用相同参数、输入、t 和 noise；当前网络无 dropout，BN 不保存 running stats，故保持原梯度定义与 batch 统计。epsilon-only 只需一次反向。`probe_identity.json` 记录执行方式。

每次 probe 另从固定 parent-keyed reverse initial noise，在 eval mode 实际执行 DDIM 第一步 t=49 的 x0_hat，记录绝对值 P99.9/max（另带 RMS/P99）。该统计与 q_sample 后的重建分开定义，写入同一 update 的六条 probe 记录；不为每个训练 update 额外执行一次采样。记录 r_t>1 标志，不根据中间诊断改变系数、训练矩阵或 checkpoint selector。主损失梯度为零导致比值无定义时显式失败。

完整 validation 保存公共五指标：lag-aware signed PCC、Whole RR、Local RR、envelope trajectory、global envelope modulation error。同时新增逐 parent 原始重建输出诊断：0.05–0.70 Hz 与 >1 Hz 能量比、prediction/target std、最大绝对幅值、top 1% 时间采样点能量占比，及 parent max 的 P95/P99 汇总。能量谱采用完整 180 s 去均值、无窗 rFFT，非 DC/Nyquist bin 双倍实现 Parseval；top 1% 为原波形最大 180 个平方值占总平方能量。预测与参考均报告。零能量或零 target std 的比值记 null，并显式记录零分母与有效/无定义计数；非有限波形或溢出失败。

row 10382 单独保存输出诊断与公共指标；若本轮 validation 身份中不存在，明确记录 present=false，不额外访问其他 split。CPU fixture 中它正常缺席。

## 执行命令与产物

在独立 RespDiff worktree 运行；每个输出目录必须不存在，失败目录保留。先 native GPU 合成检查，再按主候选、epsilon-only、source-equivalent 顺序执行固定训练。执行过程中不要修改来源文件。下列命令没有在本次实现阶段代跑：

```bash
cd /home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
for objective in snr_resp_spectral epsilon_only source_equivalent; do
  "$PY" scripts/run_respdiff_bcg_baseband_v1.py gpu-check \
    --config "configs/respdiff_bcg_baseband_v1/$objective.yaml" \
    --output "runs/respdiff_bcg_baseband_v1/${objective}_gpucheck_v1" || break
  "$PY" scripts/run_respdiff_bcg_baseband_v1.py train --confirm-training \
    --config "configs/respdiff_bcg_baseband_v1/$objective.yaml" \
    --output "runs/respdiff_bcg_baseband_v1/${objective}_seed20260811_v1" || break
done
```

GPU 检查使用合成 B64 原网络的三次训练 update、B1/B64 六步采样与 B64 六 timestep 参数梯度 probe。CPU smoke 则使用小网络、B2、两次更新、update=[0,2] probe；其 `execution.json` 明确记录覆盖，不能视为原网络科研证据。

验收产物：`receipt.json status=complete`；checkpoint schema=`respdiff-bcg-baseband-v1` 且 update=6400；history 有 6400 条；training_probes 有 30 条；validation 波形与 CSV 包含全部 val parent；`validation_output_summary.json` 包含幅值分位数、诊断计数及 row 10382。每次保存 resolved config、环境、命令、源码快照和哈希、parent/chunk 清单、数据身份、完整 history、optimizer/scheduler/RNG 和失败回执。训练完成不等于科学目标已验证；需据三臂最终五指标、尖峰诊断与 r_t 决定结论。

CPU 定向测试命令：

```bash
PYTHONPATH=. "$PY" -m pytest tests/test_respdiff_bcg_baseband.py tests/test_respdiff_bcg_repair.py tests/test_respdiff_bcg.py -q
```

2026-10-06 执行结果：41 passed in 35.46s；仅 CPU、小网络及合成 NPZ，临时目录 `/tmp/respdiff_baseband_tests_20261006_v1`。覆盖父窗口滤波增益/相位与先后顺序、全 50 timestep 的稳定 loss/梯度等价、source-equivalent 常数系数、三臂相同网络初始化、probe 前后训练参数/.grad/RNG 逐值一致、输出能量比与零分母、独立 checkpoint/配置身份、不可覆盖输出和 train 授权门。`git diff --check` 通过。

## 2026-10-06 GPU probe 显存修复

用户提供的 21:50 日志：RTX 2080 Ti（可用总容量 10.57 GiB）上主候选 B64 三次训练 update、B1/B64 六步采样通过，已报告阶段峰值 allocated=7.10 GiB；随后 B64 probe 在 `autograd.grad(..., retain_graph=True)` 请求额外 2.11 GiB 时 OOM，完整 GPU 验收失败。原目录 `snr_resp_spectral_gpucheck_v1` 保留。此记录依据用户日志，未在本机复跑 GPU。

修复将共享并保留的图改为上述独立前向、各自释放；保留 train/inference/probe B64、t 网格、样本及科学合同。CPU 定向测试 20 passed in 32.11s，包含三臂重算梯度与原共享图范数一致、禁止 probe 使用 retain_graph、训练轨迹/.grad/RNG 无扰动及三臂合成全链路。临时产物位于 `/tmp/respdiff_baseband_probe_fix_20261006_v1`。GPU 峰值仍待用户重试确认，CPU 测试不证明 GPU 显存已适配。

同步修复代码后，在用户机器仓库目录重试：

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/run_respdiff_bcg_baseband_v1.py gpu-check \
  --config configs/respdiff_bcg_baseband_v1/snr_resp_spectral.yaml \
  --output runs/respdiff_bcg_baseband_v1/snr_resp_spectral_gpucheck_v2
```

以完整六 timestep probe 后的 `receipt.json status=complete` 为通过标准；中途“GPU 检查通过”只覆盖训练与采样。若仍需减小 probe batch，须三臂统一修订并记录：BN 使用当前 batch 统计，缩小 probe batch 会影响 r_t，不能宣称与 B64 数值等价。
