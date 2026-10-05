# Patch-aligned TF-Mamba CUDA 工程验证

2026-10-05，经用户授权执行合成 CUDA 验证。代码提交为 `a95f8007ecec9a94f23d0672e8c6462c60cd5f17`，开始时工作树干净。验证使用官方 Mamba2，GPU 0 为 RTX 4070 Ti SUPER（16 GiB 标称显存），PyTorch 2.12.0+cu130，CUDA runtime 13.0。

## 结果

| 精度 | Batch | 优化步骤 | 结果 | 峰值分配 GiB | 峰值预留 GiB |
|---|---:|---:|---|---:|---:|
| FP32 | 1 | 2 | 通过 | 0.480 | 0.732 |
| BF16 | 128 | 未完成 | CUDA OOM | — | — |
| BF16 | 32 | 2 | 通过 | 4.687 | 4.906 |
| BF16 | 64 | 2 | 通过 | 9.100 | 9.938 |

通过项均验证输出 `(B,1,18000)`、有限 loss、参数和参数梯度，并验证 FiLM 输出投影更新后的第二步条件输入梯度非零。模型参数量为 1,084,681。

batch=128 在条件编码器 forward 阶段显存不足。其原始失败报告已按用户要求删除，产物索引同步更新。batch=64 已通过本轮两步工程检查；尚未进行最大批量搜索。

## 验证范围

输入为生成的 180 秒幅度调制正弦信号，对该合成信号计算原生 H-CWT，再复制为指定 batch。模型使用默认 2 秒 patch、0.5 秒步长、D=96、6 层双向 Mamba2、0.79 秒 waveform stem、patch_chunk_size=16 与局部 checkpoint 重算。

两步检查使用合成正弦目标和 MSE、AdamW(lr=1e-3)，从零初始化 FiLM 开始优化。峰值覆盖 forward、backward 和 optimizer step，包含优化器状态分配。该结果证明上述工程路径可运行；正式任务 loss、数据管线与长期收敛仍须按正式协议验证。

本轮未读取真实数据、train/validation/test cache 或目标文件，未启动正式训练。此前的临时 cache→dataset→engine 接口测试记录见模型实现说明。

## 产物与复现

产物位于当前 worktree 的 `runs/patch_aligned_tf_mamba_checks/`：

- `cuda_fp32_b1_20261005.json`：FP32 成功报告。
- `cuda_bf16_b32_20261005.json`：BF16 batch=32 成功报告。
- `cuda_bf16_b64_20261005.json`：BF16 batch=64 成功报告。
- `provenance_20261005.json`：源码提交与关键文件 SHA-256。
- `validation_manifest_20261005.json`：命令、报告 SHA-256 和执行边界。

报告和运行产物不进入 Git。复验使用新的报告路径：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/check_patch_aligned_tf_mamba.py --device cuda:0 --dtype bfloat16 --batch-size 64 --steps 2 --report /tmp/patch_tf_cuda_bf16_b64_recheck.json
```

正式运行交由用户执行。若正式协议要求有效 batch=128，需在该协议中明确微批量与梯度累积，并验证更新次数和 loss 缩放；本轮未更改正式训练配置。
