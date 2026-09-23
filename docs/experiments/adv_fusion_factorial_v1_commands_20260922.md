# ADV 融合因子实验：执行与验收

适用协议：[adv-fusion-factorial-v1-es30p15-train-val-20260922](adv_fusion_factorial_v1_protocol_20260922.md)。以下 GPU、真实数据和 formal 命令由用户执行。任一命令失败即停止该矩阵，保留失败目录；修复后使用新 identity，成功项不重跑。运行期间保持执行源码与协议固定。

## 1. 环境与结构检查

在 Bash 中执行；所有输出均位于新的融合工作树。

```bash
set -euo pipefail
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/adv_fusion_factorial_v1
FUSION_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
FUSION_RUNS=runs/adv_fusion_factorial_v1
FUSION_CONFIGS=configs/adv_fusion_factorial_v1
fusion() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    SSQ_PARALLEL=0 SSQ_GPU=0 NUMBA_CACHE_DIR=/tmp/adv-fusion-numba-cache \
    "$FUSION_PY" scripts/run_adv_fusion_factorial_v1.py "$@"
}
fusion describe
```

原生结构统计应为：A/B 各480,593参数；C/D 各480,657；E/F 各488,785。波形编码1,664、CWT编码3,136、主干463,248、读出65、两路adapter各2,112。`describe` 只实例化模型并统计参数，不执行原生 Mamba 前向。

## 2. 六组 GPU 合成验收

```bash
for arm in A B C D E F; do
  fusion gpu-synthetic --confirm-gpu \
    --config "$FUSION_CONFIGS/$arm.yaml" \
    --output "$FUSION_RUNS/gpu_${arm}_es30p15_20260922_r1" || exit
done
```

默认 `cuda:0`、batch32、bf16；如设备编号不同，统一追加 `--override training.device=cuda:1`。使用合成BCG的实际CWT、原生Mamba、任务Pi/loss、AdamW，共5次更新；保存 `report.json`、源码/配置/环境及生命周期。

验收：六个 completed.json，输出 `(32,1,18000)`、loss/预测/所有梯度有限，第二步起两路前端及attention Q/K/V权重梯度非零。报告包含每模块参数、融合MAC、峰值allocated/reserved显存及后3步中位耗时。若 OOM，不按组分别缩小正式batch；应先核对资源条件和修订整个矩阵的预算。

## 3. 有限真实数据 smoke

固定64 train、32 validation窗口、2epochs、batch32、accumulation4；样本选择和训练仍为既有冻结规则。CWT cache只读取所选输入，train读取对应target。

```bash
fusion_smoke() {
  fusion "$@" \
    --override protocol.run_role=smoke \
    --override data.max_train_windows=64 \
    --override data.max_val_windows=32 \
    --override training.epochs=2
}
FUSION_SMOKE_CACHE="$FUSION_RUNS/cache_smoke_es30p15_20260922_r1"
fusion_smoke cache --output "$FUSION_SMOKE_CACHE"
for arm in A B C D E F; do
  fusion_smoke train --config "$FUSION_CONFIGS/$arm.yaml" \
    --cache "$FUSION_SMOKE_CACHE" \
    --output "$FUSION_RUNS/${arm}_smoke_es30p15_20260922_r1" || exit
done
```

预期 cache payload 为96×97×1800×4=67,046,400 bytes，另有文件头和元数据。每组应完成2epochs/2次optimizer更新，保存完整history、初始化、模型报告、sample哈希、checkpoint与64/32窗口provenance，末尾重载selected checkpoint并输出32行validation metrics。Smoke成功不作为formal质量结论。

## 4. 正式 cache 与完整18组训练

优先使用数学表示和 provenance 完全匹配的 ADV formal cache；入口在正式读取时逐项核验。此路径是只读输入。

```bash
FUSION_FORMAL_CACHE=/mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1/runs/aligned_dual_view_v1/cache_formal_20260921_r1
for arm in A B C D E F; do
  for seed in 20260811 20260812 20260813; do
    fusion train --config "$FUSION_CONFIGS/$arm.yaml" \
      --override "training.seed=$seed" \
      --cache "$FUSION_FORMAL_CACHE" \
      --output "$FUSION_RUNS/${arm}_seed${seed}_formal_es30p15_20260922_r1" || exit
  done
done
```

每项最大80epochs/6400更新、10141 train/2675 validation；严格复核所有缓存行和输入内容后开始优化。统一停止参数为min_epochs30、patience15、min_delta0；每次完整validation后检查，保存实际末轮及最早最佳checkpoint。学习率始终按80epochs总更新数计算。不得依据部分组或seed结果修改停止规则或余下矩阵，配置拒绝改为test。

若旧cache不可用，先确认原因；按相同数学定义重新生成时使用本轮新目录：

```bash
FUSION_FORMAL_CACHE="$FUSION_RUNS/cache_formal_es30p15_20260922_r1"
fusion cache --output "$FUSION_FORMAL_CACHE"
```

新cache特征payload约8.336 GiB；仅在需要时执行，禁止为绕过哈希/来源错误而静默换cache。整个18组矩阵必须使用同一个cache身份。

## 5. 一次完整汇总

全部formal完成后执行；只读取已保存产物，不重算模型预测或访问原始数据。

```bash
FUSION_SOURCES=()
for arm in A B C D E F; do
  for seed in 20260811 20260812 20260813; do
    FUSION_SOURCES+=("$FUSION_RUNS/${arm}_seed${seed}_formal_es30p15_20260922_r1")
  done
done
fusion summary --runs "${FUSION_SOURCES[@]}" \
  --output "$FUSION_RUNS/summary_full_es30p15_20260922_r1"
```

验收：`per_seed.csv`18行，含实际epochs、更新数和停止原因；`across_seed.csv`30行，`contrasts_per_seed.csv`240行，`contrasts_across_seed.csv`80行；manifest列出全部18来源。逐run回放history核验最早合法停止点、完整更新数、固定6400更新的学习率计划及最早最佳选点。任何缺失、截断、初始化错配、样本或源码/依赖不同均显式失败。四项error的负差值和PCC的正差值代表改善；交互单独解释，不作总分排序。

## 6. CPU 工程验证入口

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  SSQ_PARALLEL=0 SSQ_GPU=0 NUMBA_CACHE_DIR=/tmp/adv-fusion-numba-cache \
  "$FUSION_PY" -m pytest -q -p no:cacheprovider tests/test_adv_fusion_factorial_v1.py
```

该集合仅使用 synthetic/disposable 数据及明确CPU Mamba替身。当前验证和实现身份见本目录 `adv_fusion_factorial_v1_implementation_lock_es30p15_20260922.json`；上一版实现锁原位保留。
