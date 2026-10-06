# RespDiff-BCG 实验过程与结果索引

截至 2026-10-05，原版三 seed 与两项单 seed loss 修复验证均已完成。修复后的节律和波形相关性明显改善，但原始输出仍存在尖峰与幅值偏差。当前证据覆盖 train/validation，独立测试集未开放。

2026-10-06 新阶段：[双 1 Hz 呼吸基带三臂](respdiff_bcg_baseband_v1_protocol_20261006.md)已实现，41 项 CPU 定向测试通过。正式候选为 Hann-rFFT 呼吸带 SNR spectral，配套 epsilon-only 与 source-equivalent 控制；GPU 和真实 train/validation 待执行。新 target 为低通 THO，下表仍保留原 target 合同结果。

## 实验过程

| 阶段 | 内容与结论 | 证据入口 |
|---|---|---|
| 参数来源 | 整理论文与源码参数，保留来源对齐实现 | [参数对齐记录](respdiff_parameter_alignment_20261003.md) |
| BCG 适配 | 20 Hz、30 秒片段、15 秒 hop、六步 DDIM、共享初始噪声、Hann 重建；batch=64、6400 updates、最终 checkpoint | [BCG v1 协议与三 seed 结果](respdiff_bcg_v1_protocol_20261003.md) |
| 保存输出诊断 | 原始输出能量由稀疏尖峰主导；FFT loss 存在 timestep 相对权重失衡 | [输出与 FFT 分析](respdiff_bcg_output_fft_diagnostic_20261004.md) |
| 固定模型诊断 | 六步回放与保存输出零误差；t=49 的噪声残差经约 2193 倍放大；五十步仍残留极端峰值 | [采样轨迹与梯度](respdiff_bcg_checkpoint_diagnostic_20261004.md) |
| Loss 修复验证 | SNR 加权 FFT 与 epsilon-only 各运行 seed=20260811；五项指标均明显改善，后者本轮略优 | [修复协议与完整结果](respdiff_bcg_loss_repair_v1_protocol_20261004.md) |
| 后处理诊断 | 固定等距 64 窗口及两个异常窗口；普通低通收益很小，3/5 点中值有有限帮助 | [滤波与训练走势](respdiff_bcg_postfilter_probe_20261005.md) |

## 同 seed 完整 validation 对比

下表为 seed=20260811、6400 updates、相同 2675 个父窗口的逐父均值。所有模型使用最终 checkpoint；训练结束后执行 validation。

| 指标 | 原版 | SNR 加权 FFT | epsilon-only |
|---|---:|---:|---:|
| Whole RR MAE ↓，bpm | 13.9233 | 2.6355 | 2.5761 |
| Local RR MAE ↓，bpm | 13.2712 | 2.7679 | 2.6703 |
| 包络轨迹 MAE ↓ | 0.5536 | 0.4416 | 0.4378 |
| 全局调制误差 ↓ | 1.0765 | 0.8172 | 0.8049 |
| 有符号 PCC ↑ | 0.0512 | 0.4470 | 0.4533 |

两项修复任务的训练片段顺序和最终训练 RNG 状态一致，配置仅 objective 不同。数据身份及 validation 批次与原版一致；每个新 run 的 17 项产物 SHA256 核对通过。checkpoint hash、逐父 CSV 与原始波形路径由修复协议定位。

## 当前判断

epsilon-only 的呼吸带能量占比中位数从原版的约 11.5% 提升到 87.1%，但全体最大绝对幅值仍约 1301，prediction/target 标准差比中位数约 2.88。公共指标包含幅度归一化，必须与原始幅值诊断共同判断修复程度。

在固定等距的 64-window 后处理诊断中，PCC 为原输出 0.4565、1 Hz 低通 0.4567、5 点中值 0.4666。这些是诊断子集结果，不能替换上表全量 validation 或直接确立正式后处理。

两项修复目前各只有一个 seed，epsilon-only 的小幅领先尚不足以证明多 seed 稳健优势。训练约 4500 步后 noise MSE 基本平台化，最终 lr=1e-6；现有证据不能保证简单续训消除尖峰。后续按上述双低通新协议从头训练三臂，并用固定 timestep 参数梯度与实际 reverse first-step 幅值诊断检查新合同。

## 代码与产物管理

比较脚本为 `scripts/summarize_respdiff_bcg_loss_repair_v1.py`，后处理诊断为 `scripts/probe_respdiff_bcg_postfilters.py`。两者只分析已保存 validation 输出，使用独立、不可覆盖的输出目录。

Git 保存代码、配置和说明。`runs/` 下的 checkpoint、波形、逐父指标、日志、图表与 manifest 保留在本地；各协议记录运行 identity 和校验信息。已完成的训练、评价与诊断按已有产物读取，不自动重跑。
