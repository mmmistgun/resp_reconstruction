# E2：最终 W0 的相对努力损失消融

协议 ID：`e2-w0-effort-ablation-v1-20260916`；日期：2026-09-16。

状态：**E2 已开放实施；独立入口及 synthetic CPU 验证已准备，GPU 验收和三 seed 正式训练由用户执行。**

## 1. 科学问题与单一变量

在最终 W0、相同初始化与训练预算下，比较完整目标 `L_sync + 0.25 L_effort` 与 `L_sync`，检验相对努力项对包络轨迹、包络范围及 RR/PCC 的影响。完整 W0 为冻结对照，新训练臂为 `w0_sync_only`。

科学变量仅为 `loss.effort_weight: 0.25 → 0.0`。沿用原生 loss 的同步对齐、target eligibility 和分量归一化；努力分量继续计算并作为诊断记录，训练总目标中它的系数为零。原生 loss、metrics、W0 模型、优化器、scheduler 与 checkpoint selector 均直接复用。

完整 W0 对照采用 `crd_tf_w_v2_candidate_lock_20260817.json` 的三个 W0 原始 validation summary，seed 为 `20260811/20260812/20260813`，selected epoch 为 `13/15/14`。来源身份与代码身份记录在独立实现锁。

## 2. 固定合同

| 项目 | E2 合同 |
|---|---|
| 输入、参考与投影 | 冻结 180 s、100 Hz 任务，BCG 与 THO 载体、admission 和 Pi 沿用 W0 |
| 训练 / validation | 10141 / 2675 窗口，32 / 7 个 samp_id；沿用 subject/session 隔离与 row identity |
| CWT | 原冻结 train/validation `[97,360]` W cache，按 row identity 只读使用 |
| 模型 | `crd_tf102_w`，六层 BiMamba2、FiLM、refinement、decoder；1,219,850 个可训练参数 |
| 初始化与 seed | 每个 seed 的初始化及 module 子 seed 与同 seed W0 相同；从新初始化开始训练 |
| 预算 | 每 seed 80 epochs、每 epoch 80 次更新，共 6400 updates；early stopping 关闭 |
| Batch | physical/effective 128，accumulation=1，drop_last=false |
| 优化器 | 原生 AdamW 参数分组；lr 3e-4→3e-5，weight decay=1e-4，betas=(0.9,0.999)，eps=1e-8 |
| Schedule | step-exact warm-up cosine，warm-up fraction=0.05，gradient clip=1.0 |
| 精度 | BF16 AMP；loss 的原生 float32 路径保持一致 |
| 样本 seed | train=20260610、val=20260611；完整 split，训练 shuffle 由同训练 seed 初始化 |
| Selector | 每 epoch 完整 validation Local RR 严格最小；相等时保留较早 epoch |
| 最终评价 | selected checkpoint 的完整 validation 五主指标及原生资格/诊断记录 |

当前阶段为 train/validation 三 seed 消融。若后续问题需要 test，使用匹配的独立评价附件明确模型身份、问题和评价范围。

## 3. 实现与身份锁

入口为 `scripts/run_e2_w0_effort_ablation.py`，实现为 `resp_train/paper_evidence/e2_effort_ablation.py`。E2 从同 seed W0 resolved config 派生，独立校验全部字段；允许差异为努力权重、E2 协议名/执行标识、输出根及运行设备/进度显示。模型计算族保留原生 `stage=tf`，实验协议名为本协议 ID。

训练由 `EffortAblationExperiment` 调用原生 `CRDExperiment.train`，子类只增强最终指标身份、finite/degeneracy 和 arm 标签校验。准备阶段生成：

```text
docs/experiments/e2_w0_effort_implementation_lock_20260916.json
```

锁保存三个完整 W0 配置、三个 E2 resolved template、checkpoint/config/summary/manifest 来源、train/validation cache 与 row/frequency 文件的大小及 SHA-256、数据索引哈希、运行合同和代码身份。准备阶段只读核验来源字节身份。实际运行重新核验锁和来源，在读取真实波形前登记访问范围，并保存 train/val rows、样本 seed 与 row 顺序哈希。

## 4. 验收与生命周期

1. synthetic CPU 定向测试覆盖单变量配置隔离、零努力目标及梯度、非有限失败、三个 seed 的真实 W0 初始化一致性、原生训练器短程 fixture、严格最早 Local RR selector、checkpoint/history 溯源和完整矩阵汇总。
2. GPU 验收使用新初始化 W0、动态合成波形和随机 W，执行一个 batch-1 的原生 forward/backward/AdamW update。检查与完整目标 W0 的同 seed 初始参数完全相同、总 loss 等于同步 loss、梯度/参数/optimizer state 有限且参数确有更新。
3. 正式训练要求同一 implementation lock 的成功 GPU 验收回执和干净 Git 工作树。每 seed 使用独立不可覆盖 attempt；成功回执绑定 run_dir、80 epochs/6400 updates、selected epoch、完整 validation 指标与分母。
4. 训练过程沿用原生 input/target finite 和非有限梯度显式失败；完成后核验全部 history、best/final checkpoint、optimizer state，核对 checkpoint 与对应 epoch 的 history 指标及 update 数。指标逐窗口身份与 validation rows 一致，prediction degeneracy 为零。
5. 同一实现身份下已完成的 seed 拒绝重跑；工程失败的 attempt 保留 lifecycle 和部分产物。在 implementation lock 和运行合同保持一致时，排除执行故障后可用新 attempt 从头执行失败 seed，并复用其余完成 seed。若修复涉及源码或科学配置变更，应先修订实现身份及可比性记录，明确已有 seed 的适用范围；当前汇总入口要求三个 seed 使用同一实现锁。
6. 三个 seed 全部完成后才开放汇总。中间 seed 的效果不用于修改矩阵、超参数、指标或 selector。
7. 每个 attempt 的完成依据是 `manifest.json` 与 `freeze_receipt.json`；失败时保留 `lifecycle_failed.json`。原生训练目录保存 config、run manifest、train.log、history、optimizer 分组、best/final checkpoint、逐窗口 metrics 与 summary；外层增加源身份、环境和访问回执。

## 5. 三 seed 配对结果

主要观察 `envelope_trajectory_mae` 与 `global_envelope_modulation_error`，完整并列报告 Whole RR、Local RR 和 signed PCC。

对四项 error，逐 seed delta 为 `100 × (sync_only − W0_FULL) / W0_FULL`；PCC delta 为 `W0_FULL − sync_only`。正值均表示完整目标更好。保存两臂的逐 seed 指标、各自三 seed mean/sample SD（ddof=1）、逐 seed delta、delta 的 mean/sample SD 和方向数；另列三 seed 均值之间的相对百分比变化（error）或绝对下降（PCC）。

结果完成后按幅度、seed 方向与其他属性代价判断努力项的贡献。两项努力指标均为 3/3 同方向时，可描述其方向一致；单一努力指标的均值和多数 seed 改善时，结论绑定该属性。科学论述使用完整连续结果，训练实例间 SD 描述优化随机性。

汇总输出包括 `seed_metrics.csv`（2 arms×3 seeds）、`paired_seed_delta.csv`（3 seeds×5 metrics）、`three_seed_comparison.csv`（5 metrics）和来源回执。比较基准为原始冻结 W0 summary，E2 结果作为新实验记录。

## 6. 执行命令

### 6.1 实现准备与 CPU 验证

```bash
./.venv/bin/python -m pytest tests/test_e2_effort_ablation.py -q
./.venv/bin/python scripts/run_e2_w0_effort_ablation.py prepare-lock
```

准备命令排他创建实现锁。提交本轮源码、协议与锁，保持工作树干净，再由用户执行以下阶段。

### 6.2 GPU synthetic 验收

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e2_w0_effort_ablation.py gpu-smoke --device cuda:0
```

取得成功输出的 `gpu_smoke` attempt 目录后，设置下面的任务变量为该实际路径。

### 6.3 三 seed 正式训练

```bash
E2_GPU_RECEIPT='/实际完成的E2_gpu_smoke_attempt目录'
for E2_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    ./.venv/bin/python scripts/run_e2_w0_effort_ablation.py formal \
    --device cuda:0 --seed "$E2_SEED" --gpu-receipt "$E2_GPU_RECEIPT" || break
done
```

每个命令输出本 seed 的 attempt 目录。循环在工程错误时停止，以便保留现场并排查；继续时仅执行尚未完成的 seed。正式训练总计 3×80 epochs，不设置按效果停掉剩余 seed 的规则。

### 6.4 完整汇总

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e2_w0_effort_ablation.py summarize --runs \
  '/seed_20260811的完成attempt目录' \
  '/seed_20260812的完成attempt目录' \
  '/seed_20260813的完成attempt目录'
```

输出根为 `runs/e2_w0_effort_ablation_v1/`。汇总要求三个不同且完成的 seed、相同 implementation lock、原生五指标分母和完整生命周期；同一实现身份的完成汇总直接复用。

## 7. 实施记录

2026-09-16 synthetic CPU 定向测试结果：`24 passed`，覆盖原生训练器的短程三 seed fixture、checkpoint/history 对应关系和重复运行保护。该 fixture 使用 2 epochs 和合成数据，作为生命周期验证；正式入口固定为本协议的完整 80 epochs。GPU 验收、正式训练与最终结果尚待用户执行，结果完成后另建冻结结果记录并同步论文证据。
