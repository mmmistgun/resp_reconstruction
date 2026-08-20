# RTM-v1 GPU engineering 协议

日期：2026-08-20

状态：**用户已授权实现统一 GPU engineering harness；尚未执行 GPU benchmark。Formal training、validation evaluation 与 research-test 仍未授权。**

## 1. 目的与证据边界

本附件只验证五项 locked candidate 在目标 GPU 上的执行可行性与工程成本，不产生模型质量证据，不读取任何 dataset/index/split/target/prediction/checkpoint，不训练数据模型，也不改变 signal-substrate、candidate、loss、metrics、selector 或正式矩阵。

输入只允许确定性 synthetic tensors。Benchmark 内执行的 optimizer update 只用于测量既有 `RespirationTaskLoss` 下的 forward/backward/optimizer 工程成本，必须标记为 `synthetic_training_like_update`，不得登记为科研 run 或训练结果。

## 2. 冻结输入与候选

- CPU implementation receipt：`docs/experiments/resp_temporal_v1_cpu_implementation_receipt_20260820.json`，SHA-256=`6ef3ca0d48e1ac819ff55bab4247d542c8bae0adfddb6951c8a026e81fea7872`；
- 候选顺序固定为 T0、TCN d9/H384、BiMamba2 D96×6、BiLSTM H96×2、multiscale 10/2/1-Hz H384；
- benchmark 必须在同一干净 Git commit、同一设备、同一 precision 与同一 synthetic contract 下顺序执行五项候选；
- 不允许 checkpoint、预训练权重、family-specific frontend、候选替换或结构 fallback；
- 入口只接受冻结 GPU engineering config，不提供 candidate、batch、repeat、device 或 output CLI 覆盖。

## 3. 设备与数值合同

- 目标设备：`cuda:0`，设备名必须为 `NVIDIA GeForce RTX 4070 Ti SUPER`，总显存不得低于15 GiB；
- precision：CUDA autocast `bfloat16`；固定 FIR 内部仍按实现合同使用 float32；
- `allow_tf32=false`、`cudnn.benchmark=false`；
- seed固定 `20260820`；输入/target固定 `[B,1,18000]`、100 Hz、180秒的解析 synthetic 波形；
- 每次测量必须同步 CUDA，报告 CUDA event latency；所有输出、loss、gradient/parameter检查必须 finite；
- 只有 `torch.cuda.OutOfMemoryError` 或明确 CUDA OOM runtime error 可记为 OOM。其他错误必须使本次 benchmark fail closed，不能伪装成候选硬件不可执行。

## 4. 统一测量合同

每项候选固定执行：

1. batch-1 inference：10次 warmup、50次 timed；
2. fixed batch-8 forward：5次 warmup、20次 timed；
3. fixed batch-8 forward+真实冻结 loss backward：3次 warmup、10次 timed，不做 optimizer step；
4. physical-batch ladder按 `128×1 → 64×2 → 32×4` 顺序测试，每项保持effective batch=128；每个scheme执行1次 warmup update、3次 timed update，包含真实冻结 loss、精确eligible-count梯度累积、gradient clipping与AdamW step；
5. 第一个完整finite通过的scheme即为该候选的执行scheme。较小scheme不再运行；该规则只解决可执行性，不比较吞吐来选择结构；
6. 三个scheme均OOM时标记 `not_evaluated_on_current_hardware`，不替换候选、不减宽度/深度，也不形成family能力负证据。

Latency报告p50/p90；显存报告timed阶段peak allocated/reserved。Update throughput固定为`128 / p50_update_seconds`。Compute-only 6400-update估计为`p50_update_seconds × 6400`；规划wall-time另乘固定1.25透明余量。二者只作工程描述，不是候选gate。

## 5. 输出、不可覆盖与验收

唯一允许的手动执行命令为：

```bash
./.venv/bin/python scripts/benchmark_resp_temporal_v1_gpu.py --config configs/resp_temporal_v1/gpu_engineering_v1.yaml
```

固定输出目录：

```text
runs/resp_temporal_v1/gpu_engineering/rtm_v1_gpu_engineering_v1/
```

目录必须预先不存在。入口使用同父目录临时目录完成写入后原子发布；禁止覆盖。固定产物：

- `resolved_config.json`
- `candidate_engineering.json`
- `candidate_engineering.csv`
- `artifact_manifest.json`
- `access_receipt.json`
- `access_receipt.sha256`

Receipt必须严格记录config/commit/dependency/device、五项candidate count、参数数、latency、peak memory、physical scheme、throughput、wall-time估计、finite/null闭合、artifact hashes及access flags。以下flags必须为false：dataset/index/train/validation/research-test/checkpoint access、formal training、validation evaluation、research-test evaluation。`gpu_used`与`synthetic_tensor_used`必须为true。

成功状态只允许：

- `complete`：五项均找到finite physical scheme；
- `complete_with_unavailable_candidates`：至少一项三个scheme均OOM，其余结果完整。

任何非OOM异常、dirty Git、设备不匹配、依赖/hash/schema漂移、已有输出目录、非有限值或产物/hash不闭合均必须失败并保留失败receipt，不得据此开放formal training。用户返回完整receipt/manifest/hash后，Codex才能分析工程结果并起草GPU engineering lock；是否开放formal training仍需用户再次明确授权。
