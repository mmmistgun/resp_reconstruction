# R3/GN r2：原生输出上的统计增量重放

日期：2026-09-21。协议 ID 沿用 `e4-r3-temporal-normalization-v1-20260921`，实现修订为 r2。

本附件仅修订 [执行协议](e4_r3_temporal_normalization_protocol_20260921.md) 的 GN 数值实现、诊断记录和 signals 复用。三个原 W0 selected checkpoint、22 条件、2675 个 validation 窗口、偏移、参考、五指标、eligibility 与配对口径沿用原协议。主干、训练与独立测试集不受影响。

验证状态：定向 synthetic CPU 测试 **27 项通过，24.64 s**。包含完整微型评价/汇总、信号流程和本次新增回归。r2 GPU smoke 与真实 validation 推理由用户执行，尚未验收。

## 1. 失败证据与修复范围

r1 锁 SHA 为 `7cbf4b4d9412fdadb24c08ae6f5cf1851cd0a47a244b1dd19541c1cde8aed6a2`。用户执行的 `gpu_smoke_7cbf4b4d9412_20260921T072724Z_43c5104f505b` 在 seed=20260811、batch=1 的 FULL 波形一致性检查失败：15181/18000 点超容差，最大绝对差 0.00402939。它尚未通过 GPU 门槛，不能据此进入正式干预评价。

旧日志未记录具体 FULL 模式；能够确定的是失败来自整体波形检查，而不是此前的逐层 GN 同源检查。小的 FP32 舍入差异经过 BF16 下游计算可能放大，是需消除的数值风险；仅凭旧日志不能认定它已被证实为 CUDA 故障的唯一原因。r2 为每条件记录开始、完成或失败、GN 自重放/公式误差；FULL 另记录波形误差、超容差点数及逐层自然统计与 FULL 的差异。异常包含条件和 batch，诊断文件在失败时保留。

旧失败目录、r1 锁和已完成 signals 保持原字节。修复使用新文件 `e4_r3_temporal_normalization_lock_r2_20260921.json` 和独立 attempt identity。

## 2. 数值定义和验收

设当前 GN 原生输出为 `yN`，当前输入为 `x`，原生统计为 `(muN,rN)`，要求使用的统计为 `(muF,rF)`，原 affine 参数为 `gamma,beta`。固定统计输出改为：

```text
yF = yN + gamma * [(x - muN) * (rF - rN) + (muN - muF) * rF]
```

在实数运算下它等于 `(x-muF)*rF*gamma+beta`。实现以 FP32 计算统计增量，回到原 GN 输出 dtype；沿 batch 分块限制临时显存。所有固定统计条件执行同一公式，统计差为零时增量自然为零。均值/rstd 必须有限、rstd 为正，窗口/组 shape 必须一致。

该实现保留原生 GN 的浮点舍入残差，因而并非承诺与另一种仿射实现逐位一致。为约束这一残差，继续独立计算直接仿射公式：

- 每层原生输出与其同源直接仿射重放仍按 `rtol=1e-5, atol=1e-6` 检查。
- 每个固定统计输出再与对应直接仿射公式检查。FP32 使用同一容差；若 GN 输出本身是 BF16/FP16，则该附加公式检查使用该 dtype 的 `eps` 作为相对/绝对误差下限，容纳低精度原生输出的量化残差。该规则仅用于附加公式检查，不改变下面的波形门槛。
- FULL/NAT、FULL/GN1_FIXED、FULL/ALL_W_GN_FIXED 相对原生完整模型的波形门槛仍为 `rtol=1e-5, atol=1e-6`。不能用局部检查代替完整模型检查。
- 原生 FULL 五指标相对冻结 validation 均值门槛仍为 `rtol=1e-3, atol=0`；非有限值、样本/eligibility 漂移继续失败。

CPU 回归额外覆盖：直接仿射与原生值仅相差一 ULP、局部容差通过但 BF16 量化结果分离的示例；新重放保留原生值；改变统计后与独立 FP64 公式相符；注入波形偏差仍失败且准确记录条件。GPU smoke 的三 seed×batch 1/128×22 条件仍需用户实际验收。

## 3. 复用已完成 signals

用户已完成：

```text
runs/e4_r3_temporal_normalization_v1/signals/signals_7cbf4b4d9412_20260921T072756Z_ff00a900b6a9
```

signals 不运行模型，其数值计算不依赖 GN 重放。新锁仅接纳这个指定的完成目录，并执行以下核验：

- r1 来源锁、signals manifest/freeze receipt 及 manifest 中全部文件的字节身份。
- r1/r2 的 checkpoint/config、完整有序 rows、展示窗口、cache/索引、参考、频率和实际偏移一致。
- 除 GN 核心与运行模块外，所有 `resp_train` 原文件仍与 r1 字节一致，包括信号函数、数据处理、指标和共享依赖。
- 对两个修改模块，从 signals 记录的干净 commit 读取旧代码并核验其原锁 SHA；比较信号路径 `TRANSFORMS`、`make_shifts`、`COUNT`、`validation_data`、`signals` 的 AST 身份。新锁记录这些定义的身份和原 signals 的 manifest 身份。
- signals 回执的数量、变换、偏移、参考及 `model_inference=false` 一致。

评价和汇总都复核指定旧产物的身份，其他旧锁下的 signals 不自动兼容。`signals` CLI 在新锁已绑定该完成目录时核验并返回原目录，不重复计算。r1/r2 的 GPU 和评价回执不能混用。

## 4. 执行顺序与产物

锁准备完成后，将本次代码、测试、新附件及新锁提交，使工作树干净。提交后不再修改执行代码，GPU 验收与正式评价须使用同一环境和 commit。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_r3_temporal_normalization.py gpu-smoke --device cuda:0
```

验收应生成 r2 identity 的成功 `gpu_smoke.json`（6 个用例）、`condition_diagnostics.jsonl`（每例完整 22 条件的开始/完成记录）、manifest 和 freeze receipt。若失败，保留输出目录，诊断中应给出具体条件。

GPU smoke 成功后，沿用完成的 signals，并把下面 GPU 路径替换为本次成功目录：

```bash
E4_R3_SIGNALS='/mnt/disk_code/marques/resp_reconstruction/runs/e4_r3_temporal_normalization_v1/signals/signals_7cbf4b4d9412_20260921T072756Z_ff00a900b6a9'
E4_R3_GPU='/本次成功的r2_gpu_smoke目录'
(
  for E4_R3_SEED in 20260811 20260812 20260813; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
      scripts/eval_e4_r3_temporal_normalization.py evaluate \
      --seed "$E4_R3_SEED" --device cuda:0 \
      --signals "$E4_R3_SIGNALS" --gpu-receipt "$E4_R3_GPU" || exit $?
  done
  env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
    scripts/eval_e4_r3_temporal_normalization.py summarize
)
```

定向 synthetic CPU 验证：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv/bin/python -m pytest tests/test_e4_r3_norm.py -q
```

实现锁准备命令仍为 `scripts/eval_e4_r3_temporal_normalization.py prepare-lock`，只创建 r2 新锁，不覆盖 r1。准备过程只读既有产物、代码与身份，不运行模型或重新计算真实信号。
