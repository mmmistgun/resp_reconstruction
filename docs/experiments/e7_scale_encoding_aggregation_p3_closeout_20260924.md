# E7 P3 GPU 工程验收与效率测量收口

日期：2026-09-24。状态：**P3 GPU acceptance 与 benchmark 已完成并冻结；P4 formal train/validation 可引用本收口固定的唯一成功 GPU receipt。**

机器可读收口为 `docs/experiments/e7_scale_encoding_aggregation_p3_closeout_20260924.json`。

## 1. 冻结身份

- P1 implementation lock：`299448a12c4006f2f918f56645000838f83f044fbbf31d7cc281faa6540b11e1`。
- P3 engineering lock：`773f5e78c5e8d7b08cadaac00797ede8dcd9f3f7a77d0775fc5679650a29d32d`。
- 执行代码 commit：`2d1552f33c4a69de96fbd914235d3497f0730fef`。
- 环境：RTX 4070 Ti SUPER、CUDA 13.0、cuDNN 9.20、PyTorch 2.12.0+cu130、Python 3.12.13。

GPU acceptance 成功 attempt：

```text
runs/e7_scale_encoding_aggregation/gpu_acceptance/
gpu_acceptance_773f5e78c5e8_20260924T082610Z_62e8f9b7eaef
```

Benchmark 成功 attempt：

```text
runs/e7_scale_encoding_aggregation/benchmark/
benchmark_773f5e78c5e8_20260924T084537Z_32c4b00a1fe3
```

两份 freeze receipt 与 manifest 已逐文件回载验证；manifest SHA-256 分别为 `6adaf2ca83776f1d9bbd0ae92ca688f1890a7f8fc05cee6949370c18d00d22fe` 和 `452a01a0d2f8b7d69dfdcc05a40e8827502879bdefb7ee0b057a99401e924364`。

## 2. GPU acceptance

- 完成 18 个 batch-1 cell 与 6 个 batch-128 cell，共 24 项。
- 24 项均在第 5 次原生 update 满足 tracked 参数梯度非零且相对初值真实变化。
- 三 seed batch-1 的公共 W0 state、CPU/CUDA RNG、聚合前后 shape、waveform 初始等价与有限性均通过。
- batch-128 使用 BF16、physical batch 128、branch checkpoint chunk 8 和完整任务损失。
- 最大 `peak_reserved/device_total=0.687735`，低于冻结的 0.8 上限。
- 输入全部为 synthetic；未读取真实 train/validation 或 research-test。

## 3. Benchmark

完整 36 个独立测量进程均完成：六 arm × eval/train × 三组；每项 warm-up 5 次并记录 20 次。下表是三个独立 measurement 的 median 再取算术平均；训练吞吐对应 batch 128，eval 吞吐对应 batch 1。

| Arm | eval ms | eval samples/s | train ms | train samples/s | train peak reserved MiB（最大） |
|---|---:|---:|---:|---:|---:|
| `s0_shallow__mean` | 21.899 | 45.949 | 567.897 | 225.393 | 10118 |
| `s0_shallow__frequency_attention` | 22.546 | 44.557 | 681.803 | 187.744 | 10116 |
| `s1_deep_local__mean` | 24.541 | 40.754 | 2389.942 | 53.558 | 10790 |
| `s1_deep_local__frequency_attention` | 26.185 | 38.366 | 2488.188 | 51.444 | 10998 |
| `s2_axis_spanning__mean` | 24.527 | 40.772 | 2405.336 | 53.217 | 10790 |
| `s2_axis_spanning__frequency_attention` | 25.395 | 39.383 | 2484.998 | 51.509 | 10998 |

Benchmark 最大 `peak_reserved/device_total=0.690119`。该结果只描述 synthetic cached-input model-only 路径，不含 CWT/cache 构建或磁盘 I/O；参数量、局部 MAC 与实测速度分开解释。

## 4. 失败生命周期

保留三个失败 attempt：GPU acceptance 与首个 benchmark attempt 因运行环境优先加载了不匹配的 cuDNN 9.8；第二个 benchmark attempt 被干净工作树预检拒绝。随后在相同 P1/P3 科学身份下使用正确运行环境并保持工作树干净，形成上述唯一成功 attempt。失败目录不计入成功矩阵，也没有被删除或覆盖。

## 5. P4 使用边界

P4 formal train/validation 只接受本收口登记的 GPU acceptance 成功目录作为单一 `--gpu-receipt`。Formal 入口必须重新核验其 freeze receipt、manifest、24-cell 完整性、P1/P3 lock identity 和关键运行环境。Benchmark 是效率证据，不作为 formal 运行门控。P4 仍限于 train/validation，不读取 research-test。
