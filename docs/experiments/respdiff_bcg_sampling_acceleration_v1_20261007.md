# 单 GPU ε 推理加速 v1

当前状态：condition 缓存与 trajectory 分组已实现，13 项 CPU 合成定向测试通过（12 项整文件测试＋新增 DDPM N4 定向测试）。用户明确要求本次不调用 GPU；GPU benchmark 未执行，未重训、未重跑完整 validation。实现分支为 `codex/respdiff-paper-settings`。

## 合同与实现

来源仍是 source-equivalent、seed=20260811、final update=6400；checkpoint SHA256 为 `08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6`。训练、权重、参数名、shape、state_dict 及既有数据适配不变。FP32、TF32=False、cudnn.benchmark=False 与[ε 推理 v2](respdiff_bcg_epsilon_inference_v2_protocol_20261007.md)一致。

参考 sampler `resp_train/respdiff_bcg/sampling.py` 保留。独立加速模块为 `sampling_accelerated.py`，主要接口：

- `encode_condition`：按当前原始 B 计算 fine condition 与 weighted coarse condition，返回短生命周期缓存，不写入模型。
- `forward_cached`：保留 `(fine_condition + timestep_embedding) + weighted_coarse_condition` 顺序；embedding 按 B 计算后广播，noise encoder/RNN/decoder 对 G*B 计算。
- `accelerated_groups`：G=1/2/4/8，逐组处理连续 trajectory_id，支持不足 G 的最后一组。
- `accelerated_prefixes`：每组完成后按原 trajectory_id 顺序累加 FP64 sum，产出所有所需 nested prefix。只保留当前组状态；不持有完整 N 维 trajectory tensor。

缓存由一次原始 batch 的 iterator 创建并私有持有；不能跨不同 condition tensor、不同 denoiser 或原地修改后的 condition 复用。模型须完整 eval；非 condition 分支若新增 BatchNorm 则显式拒绝合批。原始 B64 或输入尾 batch 的 condition BatchNorm 统计按 B 计算，G 不参与统计。缓存不是 checkpoint 参数或 buffer。

DDIM6 与 DDPM50 的 timestep、更新算式、FP32 调度使用方法、initial/reverse Gaussian noise 身份均复用现有合同。所有噪声逐 trajectory 调用参考 noise helper，再按 trajectory-major 排列为 `[G*B,1,600]`，G 不写入 noise identity。同一 parent 的 13 chunks 仍共享 Gaussian 场；trajectory0 的 initial noise 与历史版本完全一致。

parent 重建仍为 ensemble chunk mean→Hann OLA→100 Hz Fourier interpolation→完整 180 s parent 1 Hz 后滤波，复用既有重建函数。缓存、合批不增加幅值变换。

当前 SamplerSpec 的预算继续为 DDIM≤16、DDPM≤4；本次不改变 N 实验矩阵。分组循环和 noise identity 不绑定总 N，后续更大 N 可在独立预算修订后复用该实现。当前 DDPM G8 的最后一组最多包含 4 条轨迹，不能据此报告真正的 G8满组吞吐。

## 验证前确定的 FP32 容差

`acceleration_checks.py` 固定以下容差，比较逐元素 `abs(actual-reference) <= atol+rtol*abs(reference)`：

| 阶段 | atol | rtol |
| --- | ---: | ---: |
| epsilon prediction | 1e-4 | 1e-3 |
| 每个 reverse step 输出 | 0.02 | 1e-3 |
| 每条 trajectory 终态 | 0.02 | 1e-3 |
| 每个 ensemble prefix | 0.02 | 1e-3 |
| raw/postfiltered parent waveform | 0.02 | 1e-3 |
| initial/DDPM reverse noise | 0 | 0 |

原始/缓存 G1 的 CPU forward 和 sampler 另要求逐值完全一致。G>1 改变 GEMM/卷积/RNN 计算 batch，数学等价不意味着不同 batch 形状逐位一致；GPU 使用上述固定 FP32 容差并报告实际误差。reverse 的绝对容差大于 ε，因为 t=49 的 ε→x0 映射具有约 2193 倍的误差放大。

记录每个阶段的最大绝对误差、RMS、最大容差占比、超限元素数、finite 和 passed。初始/逐步 noise 必须逐值相同。参考 trace 通过原 sampler 的 forward hook 获取 epsilon、实际下一步输入及最终输出，不重写参考更新公式；trace 仅驻留当前 G 条轨迹。验收检查每个 trajectory 每一步的 epsilon/reverse，不能只以均值一致判通过。

容差修订依据：首版 CPU 试验使用更严格 ε `atol=3e-6,rtol=3e-5`，G4/G8 的一个 DDIM 轨迹在 t19 有一个元素超限，最大 ε 差异为 6.75e-6；reverse/终态最大差异为 9.77e-4。同一 noisy state 下缓存合批 forward 的 ε 差异仅 2.24e-8，说明主要是递推输入中的舍入差异传播。用户随后明确要求阈值不要过低；按该指令在下一轮 CPU 验证前冻结上表工程容差。首版失败记录保留，实际误差不会因容差修订被省略。

## 独立推理入口和 runtime 口径

`scripts/run_respdiff_bcg_epsilon_accelerated_v1.py` 沿用原 v2 模式和科学合同，显式增加 `--trajectory-group-size`；原 v2 CLI 不指定 acceleration 时仍走串行参考。每次新输出保存 cache/G identity；同一 accelerated subset/full-validation 必须使用相同 G。本次只实现入口，不执行这些真实推理阶段。

`predict_prefixes(..., acceleration=AccelerationSpec(G))` 区分三个数：

1. 请求 prefix N 的逻辑预算：NFE×N calls/chunk。
2. prefix 实际 ready 时完成的轨迹数，以及分组合批的 forward 次数。
3. 共享最大 N 执行中的 group 完成墙钟与 prefix 实际交付墙钟。

例如最大 N8、G8 时，N1/N2/N4/N8 都在同一组计算完成后 ready；每个 prefix 不能被解释为一次独立 N 推理耗时。其请求逻辑预算依次为 6/12/24/48，但 ready 时实际均已完成 8 条轨迹的逻辑工作和 6 次 G*B forward。`runtime_profile` 明确保存该口径和完成工作量，组内 readiness 时间相同，delivery 时间可因依次累积和拷贝略有不同。

对于固定最终 N，实际 forward 次数为 `NFE*ceil(N/G)`，逻辑计算量 NFE×N 不变。例如 DDIM6 N8 的 forward 次数随 G1/2/4/8 为 48/24/12/6；fine/coarse 编码从参考的 48 次各减少为一次。实际墙钟收益和显存占用须经硬件实测，不能由调用数直接推断。

## 用户执行的短时 GPU benchmark

本次不执行下列命令。默认使用已冻结的真实 checkpoint 和合成 parent 输入，经过原 parent LPF/AA/chunk 数据适配；不读取真实 train/validation/test waveform，也不计算科研指标。

```bash
cd /home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/benchmark_respdiff_bcg_sampling_acceleration.py \
  --sampler ddim --n-trajectories 8 --batch-size 64 --warmup 1 --repeats 2 \
  --source-run runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1 \
  --output runs/respdiff_bcg_sampling_acceleration_v1/ddim6_N8_B64_benchmark_v1
```

该命令依次比较 serial_reference、condition_cache_G1、condition_cache_G2/G4/G8。每个变体先逐 step/trajectory 数值验收，再单独 warmup 和测量；每次 invocation 都新建当前 batch cache，缓存构建计入采样时间。timed pass 不带数值 trace observer。报告重复次数、采样秒的中位数、ensemble chunks/s、相对串行与缓存 G1 的加速比，以及 warmup 后 peak allocated/reserved；prefix readiness 另存独立 CSV。

可用以下较小 B 的 DDPM 工程验收检查完整 50 步、输入尾 batch 和 posterior noise；其性能结论仅适用于 B13，不外推 B64：

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/benchmark_respdiff_bcg_sampling_acceleration.py \
  --sampler ddpm --n-trajectories 4 --batch-size 13 --warmup 1 --repeats 1 \
  --source-run runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1 \
  --output runs/respdiff_bcg_sampling_acceleration_v1/ddpm50_N4_B13_benchmark_v1
```

每个变体固定相同 checkpoint、input、noise、sampler 和 N；不自动降 B/G。某个 G OOM 时保留错误和 profile status=oom，继续记录其他预定变体；整体 receipt 标记 complete_with_oom。任何非有限输出、误差超容差或 checkpoint 变化则显式失败，不给该变体加速建议。

验收产物：`numerical_errors.csv`（逐 step/trajectory 和重建误差）、`runtime_profile.csv`、`shared_prefix_runtime_profile.csv`、`summary.json`、输入 fixture、source snapshot/identity、environment、benchmark identity、receipt/failure。checkpoint 文件、参数和所有 buffer 前后不变；最快 G 仅从已通过数值验收且未 OOM 的实测变体中选择。

GPU group 建议尚待实测。G1 的缓存收益可独立评估；G2/G4/G8 可减少 forward 次数但增加计算 batch 与临时内存。本次不承诺任何实际加速倍数。

## CPU 验证与交付边界

使用小网络 hidden=8/layers=1/output=4、合成 condition 与 disposable checkpoint，未执行真实 checkpoint 的 GPU 推理或性能测试。测试在 CPU 单线程运行，并将 CUDA 初始化、可用性查询、全设备 seed 和 synchronize 调用设为失败；另显式设置 `CUDA_VISIBLE_DEVICES=`。

命令：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$PY" -m pytest \
  tests/test_respdiff_bcg_sampling_acceleration.py -q
```

最终执行记录：整文件当时的 12 项测试在 57.11 s 内通过，产物 `/tmp/respdiff_sampling_acceleration_cpu_20261007_v4`；随后为完整 DDPM N4 新增一项参数化案例，仅运行该新增案例，1 passed in 47.35 s，产物 `/tmp/respdiff_sampling_acceleration_ddpm4_cpu_20261007_v1`。当前文件共 13 个案例，均按修订后的冻结容差通过。

覆盖原始 B3/B64 缓存 forward 逐值一致、缓存跨 batch/原地改动拒绝、G1/G2/G4/G8、DDIM N9 与 DDPM N3/N4、逐 step epsilon/reverse、逐 trajectory 终态和各 nested prefix、initial/reverse noise 逐值一致、最后不足 G 的组、输入 B5/B5/B3 尾 batch、完整 parent OLA/插值/后滤波、checkpoint/parameter/buffer 不变、非有限失败与合理超限拒绝。weakref 案例还验证上一组 Tensor 在下一组计算前释放，未累计持有旧组。

| CPU 案例 | G | ε 最大绝对误差 | reverse/终态最大绝对误差 | prefix 最大绝对误差 |
| --- | --- | ---: | ---: | ---: |
| DDIM6 N9 | 1/2 | 0 | 0 | 0 |
| DDIM6 N9 | 4/8 | 6.75e-6 | 9.77e-4 | 4.88e-4 |
| DDPM50 N3 | 1/2/4/8 | 0 | 0 | 0 |
| DDPM50 N4 | 1/2 | 0 | 0 | 0 |
| DDPM50 N4 | 4/8 | 5.16e-6 | 2.44e-3 | 1.46e-3 |

所有记录 finite 且通过当前逐元素门槛。完整 parent 重建案例（B13、N3）各 G 的 raw/postfiltered waveform 逐值一致。源真实 checkpoint 文件的 SHA256 再核对仍为固定值；CPU fixture 的参数和 buffer 前后不变。语法检查和 benchmark `--help` 检查通过。

这些结果验证实现与误差检查机制，不代表 native 六层 H1024 模型在某张 GPU 上的精度、显存或速度验收。当前未实测 condition 缓存的墙钟收益、G2/G4/G8 的额外收益或最佳 G；使用上面的 GPU 命令取得通过的 native checkpoint 实测后，再按最快且未 OOM 的组选择。
