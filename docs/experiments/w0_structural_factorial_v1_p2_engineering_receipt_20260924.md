# W0 structural factorial v1：P2 工程验收回执

- 日期：2026-09-24
- 协议：`w0-structural-factorial-v1-es30p15-20260923`
- 状态：`P2_COMPLETE`
- 运行提交：`309c2418d994897a8834a1b4db254b31c523e525`
- 实现锁：`docs/experiments/w0_structural_factorial_v1_implementation_lock_r2_20260924.json`
- 实现锁 SHA-256：`32141eab672ea41435055c45cbd7ec96325f8ddef2481a210db2941222c9e9f3`
- 设备：NVIDIA GeForce RTX 4070 Ti SUPER，15.56 GiB
- 软件：Python 3.12.13，PyTorch 2.12.0+cu130，cuDNN 92000

## 1. 执行边界

本阶段仅使用由固定 seed 构造的 synthetic waveform、target 与 W tensor。运行入口读取 r2 实现锁中的 resolved baseline 配置；未读取 dataset index、真实波形、W cache、历史 checkpoint 或 test。所有训练路径使用 BF16 AMP、AdamW、physical batch 128、accumulation 1 和正式 6400-update 学习率预算定义。

运行前专项及相邻回归为 `53 passed`，Python 编译、`check-config`、`describe`、`check-lock` 和 `git diff --check` 均通过。工作树在 GPU 运行时保持干净。

## 2. GPU acceptance

八个 arm 均完成 batch-1 forward/core-loss/backward/optimizer update。所有可训练参数张量均获得 finite gradient；零梯度张量只出现在预期的 identity-start 路径，且不存在 missing 或 nonfinite gradient。每组均有参数实际更新。

最大资源臂 `sfv1_conv20_tm3_ref2` 完成三次 physical-batch-128 原生 update：

- 三步非零梯度张量数：`138/174 → 162/174 → 174/174`；
- 三步后发生更新的参数张量：`174/174`；
- peak allocated：9,526.06 MiB；
- peak reserved：10,872.00 MiB；
- peak reserved fraction：`68.2213%`。

同一最大资源臂随后完成一次原生 `128 train / 32 validation / 1 epoch / 1 update` lifecycle：

- `train_history.csv` 为 1 epoch、1 optimizer update；
- best-Local-RR 与 final checkpoint 均存在且 state finite；
- validation metrics 为 32 行，row identity/order 完整；
- `joint_prediction_degenerate=0`，`envelope_spearman_prediction_degenerate=0`；
- peak reserved fraction：`68.1711%`；
- resolved config、audit、optimizer groups、history、两个 checkpoint、metrics、summary、runtime 与 manifest 均已冻结。

acceptance identity：

```text
runs/w0_structural_factorial_v1_es30p15/gpu_acceptance/
  gpu_acceptance_32141eab672e_20260924T074944Z_44fa12038854/
```

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `gpu_acceptance.json` | 12,635 | `0a1d642795871d85bfb039c55fcf0284443c31922c695dcc5231ac7e95fe843e` |
| `manifest.json` | 4,890 | `6d49aa1b5889c67d176d205e7a035d5d541a45f5ab473a5b4502d7ff54674dd9` |
| `freeze_receipt.json` | 190 | `48669632ad609f2884ca535236a511d0c12cf498696a7778996a1a0fd462f9a6` |

## 3. 独立进程 benchmark

每个 arm/mode 使用一个新进程；eval 为 batch 1，train 为 physical batch 128。每项 3 次 warmup 后记录 10 次重复的中位数、IQR、吞吐和 CUDA peak memory。

| arm | eval ms | eval sample/s | train ms | train sample/s | train peak reserved |
|---|---:|---:|---:|---:|---:|
| `sfv1_patch_tm3_ref2` | 20.803 | 48.070 | 555.814 | 230.293 | 63.50% |
| `sfv1_patch_tm3_ref0` | 20.731 | 48.237 | 541.311 | 236.463 | 63.18% |
| `sfv1_patch_tm0_ref2` | 19.644 | 50.906 | 525.594 | 243.534 | 63.50% |
| `sfv1_patch_tm0_ref0` | 19.426 | 51.478 | 502.661 | 254.645 | 63.16% |
| `sfv1_conv20_tm3_ref2` | 20.175 | 49.566 | 617.545 | 207.272 | 68.27% |
| `sfv1_conv20_tm3_ref0` | 20.284 | 49.299 | 616.187 | 207.729 | 66.56% |
| `sfv1_conv20_tm0_ref2` | 19.635 | 50.929 | 581.839 | 219.992 | 68.61% |
| `sfv1_conv20_tm0_ref0` | 21.981 | 45.494 | 572.520 | 223.573 | 66.90% |

benchmark identity：

```text
runs/w0_structural_factorial_v1_es30p15/benchmark/
  benchmark_32141eab672e_20260924T075037Z_ab5670de3e59/
```

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `benchmark.json` | 26,401 | `f21a073c269351baa2b0a6ee7924adf0557efe726f98cf2061147c9d2dc561fe` |
| `manifest.json` | 6,265 | `214f4cd8b555569600f7dcf5577a2a6899d5bf00650c3f5be817b2527118568b` |
| `freeze_receipt.json` | 190 | `234f4a15b6923fb9c105b14e148c222290e8977c0852b83373dd9c6ecc773e9d` |

## 4. P2 判定

P2 工程验收通过。八组所需 CUDA 路径均 finite；最大资源臂的三步梯度启动和完整 checkpoint lifecycle 闭合；所有 physical-batch-128 测量的 peak reserved fraction 均低于 80%。当前证据只支持工程可运行性、资源口径和生命周期正确性，不形成结构质量结论。

P3 的 24-run formal 仍需单独授权。正式运行应继续绑定上述 r2 lock，保持八组统一 batch、accumulation、dtype、训练预算与 selector。
