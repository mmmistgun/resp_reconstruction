# E8 P2：GPU 工程验收与效率测量收口

日期：2026-09-25。状态：**P2 synthetic GPU acceptance 与独立进程 benchmark 已完成并冻结；两份成功 attempt 的 freeze receipt、manifest 及全部受管文件已回载验证。**

## 1. 冻结身份

- 协议：`e8-film-decoder-redesign-factorial-v1-20260925`。
- 执行代码 commit：`ea5a6aa2881f522b0aa47e46eedbcf3b72508b3c`。
- Engineering identity：`aa80977f0c737906bdf011bd9b39367d1c77b39130404a39db9c1f6b2670f3d4`。
- 环境：RTX 4070 Ti SUPER、CUDA 13.0、cuDNN 9.20、PyTorch 2.12.0+cu130、Python 3.12.13、BF16。
- 物理 GPU 由 `CUDA_VISIBLE_DEVICES=1` 隔离，进程内设备为 `cuda:0`。

GPU acceptance：

```text
runs/e8_film_decoder_redesign_v1/gpu_acceptance/
gpu_acceptance_aa80977f0c73_20260925T115916Z_e20ea0dd8f28
```

Benchmark：

```text
runs/e8_film_decoder_redesign_v1/benchmark/
benchmark_aa80977f0c73_20260925T120044Z_4ddffe1e62b0
```

Acceptance manifest SHA-256 为 `a169d42fca40585a55eeb58de9dbb5b9bd3decd21429f9eeef9458b3238c041b`，含 54 个受管文件；benchmark manifest SHA-256 为 `23f4d285d586df25227ddc33c5d5751089de03ee7b7bc278523cacabd9cce9dc`，含 53 个受管文件。

## 2. GPU acceptance

- 完成 12 arms × 3 seeds 的 36 个 batch-1 cell，每项三次原生 loss/optimizer update。
- 36 项最后一步的所有适用 factor gradient 均已开启；FiLM projection、条件 refiner 和 decoder residual 的适用参数均相对初值真实变化。
- Batch-1 最大 `peak_reserved/device_total=0.038386`。
- 最大资源 arm `e8_res192_temporal` 完成 batch-128、三次原生 update，峰值 allocated 8.600 GiB、reserved 9.912 GiB，`peak_reserved/device_total=0.636628`，低于冻结的 0.8 上限。
- 同一最大资源 arm 完成一轮原生 synthetic trainer lifecycle：128 train windows、32 validation windows、1 次 optimizer update；config、history、best/final checkpoint、逐窗口 metrics、summary 与 runtime 产物齐全。该路径峰值 reserved fraction 为 `0.635875`。
- 全程只使用确定性 synthetic tensor；dataset index、真实 waveform、W cache、历史 checkpoint 与 research-test 均未读取。

## 3. Benchmark

24 个独立进程全部完成：12 arms × eval/train。每项 warm-up 5 次、测量 20 次；eval batch=1，train batch=128。下表为各进程中 20 次测量的 median；该测量只描述 synthetic cached-input model-only 路径，不含真实数据读取、CWT/cache 构建或磁盘 I/O。

| Arm | eval ms | eval samples/s | train ms | train samples/s | train peak reserved fraction |
|---|---:|---:|---:|---:|---:|
| `e8_fill65_pointwise` | 20.782 | 48.118 | 565.828 | 226.217 | 0.634746 |
| `e8_fill65_single` | 20.527 | 48.716 | 566.571 | 225.920 | 0.633492 |
| `e8_fill65_temporal` | 23.962 | 41.733 | 568.041 | 225.336 | 0.636001 |
| `e8_direct_pointwise` | 20.526 | 48.719 | 577.263 | 221.736 | 0.634746 |
| `e8_direct_single` | 20.653 | 48.419 | 573.453 | 223.209 | 0.633492 |
| `e8_direct_temporal` | 20.948 | 47.738 | 556.791 | 229.889 | 0.636001 |
| `e8_res96_pointwise` | 24.332 | 41.099 | 563.559 | 227.128 | 0.634746 |
| `e8_res96_single` | 21.014 | 47.587 | 560.831 | 228.233 | 0.633492 |
| `e8_res96_temporal` | 20.499 | 48.783 | 588.919 | 217.347 | 0.636126 |
| `e8_res192_pointwise` | 24.273 | 41.198 | 562.641 | 227.498 | 0.634872 |
| `e8_res192_single` | 20.479 | 48.831 | 571.626 | 223.923 | 0.633617 |
| `e8_res192_temporal` | 24.261 | 41.219 | 573.706 | 223.111 | 0.636126 |

Benchmark 最大 reserved fraction 为 `0.636126`。单进程 benchmark 没有独立重复组，因此细小速度差仅作工程描述，不作为结构优劣证据。

## 4. 完整性核验

- Acceptance：`passed=true`，36 个唯一 `(arm,seed)`，全部 `updates=3`，所有最后一步 factor gradient 非零，所有适用 factor 参数发生变化。
- Benchmark：24 个唯一 `(arm,mode)`，全部 `warmup=5/repeats=20`，eval/train 两种模式与 12 arms 完整覆盖。
- 两份 freeze receipt 均与 manifest 字节身份一致；manifest 登记的 107 个文件已逐文件核对 size 与 SHA-256。
- 两阶段 environment 均登记同一 clean commit、engineering identity、依赖和设备环境。

## 5. 后续边界

P2 已关闭，不以同一或新 identity 重跑 acceptance/benchmark。Formal 入口必须显式引用本收口的两份成功 attempt，重新核验 freeze receipt、manifest、矩阵完整性、显存上限和运行环境兼容性。正式训练仍限于 train/validation；research-test 未开放。
