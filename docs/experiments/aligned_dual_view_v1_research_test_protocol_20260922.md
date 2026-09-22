# ADV-v1：固定候选的 research-test 评价协议

日期：2026-09-22。协议 ID：`aligned-dual-view-v1-research-test-20260922`。

**当前状态：用户授权实现 test 相关代码。候选锁已从完整 validation 产物生成；本次仅执行只读来源检查、代码实现及 CPU synthetic 验证。真实 test index/波形/target、test cache 与 GPU 评价尚未由 Codex 访问或执行。**

本协议专用于 ADV-v1。既有 CRD、RTM、论文附件的已关闭 test 阶段维持原状态。Test split 已参与以往研究，所有新结果标记为 **reused research/development evidence**，不称为未触及 held-out 或无偏独立确认。

## 1. 固定问题和候选矩阵

评价完整 180 s 重建任务中四种输入配置的 test 表现，保留五主指标及其属性取舍。沿用既有 dataset/admission、target、Pi、指标、eligibility 和 sample direct mean；test 结果不改变本轮 checkpoint 或阈值。

| 配置 | seed20260811 selected epoch | seed20260812 | seed20260813 |
|---|---:|---:|---:|
| joint | 12 | 7 | 5 |
| waveform | 8 | 9 | 15 |
| cwt | 22 | 7 | 21 |
| joint_scale_mean | 7 | 7 | 8 |

共 12 个 validation-selected checkpoint。矩阵覆盖上一阶段所有配置，不按 partial test 改变候选或 seed 集合。每项仅加载锁定的 checkpoint path/hash；不存在以 test 指标重选 epoch 的入口。

来源：

- Validation 汇总：`runs/aligned_dual_view_v1/summary_full_20260922_r1/manifest.json`，SHA=`cdc03f23f299b24bbf6234cd73c5c1678d7e9d7b22065e7f8cef58f9a0d1577d`。
- 候选锁：`configs/aligned_dual_view_research_test_v1/candidate_lock_20260922.json`，文件 SHA=`146e89f4c196fd1b2742ba66bb3ad600f11e25a6ccf98ebfcbc7ece012947418`。
- 冻结训练代码身份：`19b92747cb2a26a5cf45d6954aeebaae1ce70b3de5a01a266677de4761284040`。

新实现放在 `resp_eval/adv_test/`，普通训练与 validation 入口继续使用原代码。专项执行身份分别记录原训练文件和本协议/评价代码，使原训练 provenance 仍可核验。

## 2. 数据与前处理

数据集、输入/target key 及 admission 从锁定训练配置读取。完整 test 样本规格来自[既有 CRD-TF research-test 协议](crd_tf_v1_research_test_protocol_20260816.md)第 3、6 节的数据身份元数据，本次没有重新打开 test 数据来生成这些常数：

- Index SHA：`f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f`。
- 完整 test：2,310 窗口、8 个 samp_id；sample seed=20260612；row ID 严格递增且唯一。
- little-endian int64 row-ID SHA：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。
- 访问时重新核对 train/val/test 的 subject 与源文件隔离、index、admission 数量和 row-ID 哈希。

test 使用新建、input-only 的 `[2310,97,1800]` float32 CWT cache。复用 ADV-v1 的实际 97 scales、100 Hz `log1p(abs(CWT))`、固定 FIR 与对齐抽取。缓存绑定候选锁、表示、前处理实现、依赖、选择行、输入与特征哈希。旧 2 Hz 表示和 train/validation cache 均不作为此入口的 test cache。

Cache builder 只取输入 NPZ 的 BCG key，不取 target array、不执行模型。预估特征 payload 为 1,613,304,000 bytes（约 1.503 GiB），另有选择行、元数据、源码快照等。

## 3. 访问与执行约束

`cache`、`evaluate`、`summary` 均要求显式 `--confirm-research-test`。用户亲自执行命令即表明本阶段操作意图；Codex 代跑真实 test 或 GPU 阶段仍须获得当次明确授权。本次“实现代码”的授权不扩展为执行真实 test。

模型加载前校验 checkpoint 哈希；载入时验证所有 checkpoint tensor/标量有限、参数结构、固定 FIR、训练 identity 与 selected epoch。推理使用 `eval()`、`no_grad()`、CUDA bf16，固定 FIR 为 float32 IEEE 卷积，默认 batch=原配置的 32。

**每个候选先完成全部 2,310 窗口的 input-only 推理，再开始读取 target/mask。** Target 只用于既有评价器；输入校验在取样与 target 阶段再次执行。`access.jsonl` 记录推理开始/完成及 target 阶段顺序。原有 dataset 负责 target 和 mask 语义，原有 metric 函数负责 Pi 与指标。

五主指标：Whole RR、Local RR、包络轨迹、全局包络调制、signed PCC。同步输出既有 IBI/coverage、分层 envelope Spearman、呼吸带 coherence 和 constrained nDTW。Eligible 主指标与 coherence/nDTW 必须有限；原定义允许的 target-ineligible NA 保留。统计单位仍为窗口，三个训练 seed 不作为独立人群。

## 4. 产物与失败生命周期

- Cache 和 summary 的输出目录必须事先不存在；不能在数据源、源码或历史 run/cache 目录内补写。
- 新阶段每个 artifact 保存命令、依赖、代码/协议快照、候选锁、原配置、manifest 与完成回执；异常/可捕获中断保存 `failed.json` 和已有文件。
- 一个 cache 对应固定评价目录 `<cache目录名>_evaluations/`。每个 entry 的尝试目录为 `<entry>/attempt_<attempt>`。
- 同一 cache/entry 用文件锁防止并发重复；有成功回执后拒绝再次评价。失败后可用新 attempt 标识重试，保留原失败目录。未闭合尝试需要先核实中断原因，不能忽略或删除。
- 更换 cache/output 名称不构成重新评价成功候选的授权。修复若改变评价语义或影响已有成功结果，应先明确协议修订和受影响范围。
- 每项保存逐 sample metrics、summary、selection、input/target/mask 哈希、预测内容哈希、checkpoint/cache 身份与 access trace。数据内容、执行代码、precision 或 batch 不一致的评价不能进入同一完整汇总。
- 原训练目录、checkpoint、history、validation 指标和旧 test 产物均保持原状。

## 5. 入口和用户执行命令

唯一专项入口：`scripts/eval_aligned_dual_view_v1_research_test.py`。

候选锁已经准备。以下检查不读取数据集：

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1
ADV_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  "$ADV_PY" scripts/eval_aligned_dual_view_v1_research_test.py check-lock
```

下面的 cache、评价与汇总命令由用户执行，或在新增执行授权后代跑。每步成功再进入下一步；失败先调查原因。

```bash
adv_test() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    SSQ_PARALLEL=0 SSQ_GPU=0 NUMBA_CACHE_DIR=/tmp/adv1-numba-cache \
    "$ADV_PY" scripts/eval_aligned_dual_view_v1_research_test.py "$@"
}
ADV_TEST_CACHE=runs/aligned_dual_view_v1_research_test/cache_20260922_r1

# 固定全部四配置与三个 seed；任意一项失败即停止循环。
adv_test_matrix() {
  for view in joint waveform cwt joint_scale_mean; do
    for seed in 20260811 20260812 20260813; do
      adv_test evaluate --confirm-research-test --cache "$ADV_TEST_CACHE" \
        --entry "${view}_seed${seed}" --device cuda:0 --attempt r1 || return
    done
  done
}
adv_test cache --confirm-research-test --output "$ADV_TEST_CACHE" &&
adv_test_matrix &&
adv_test summary --confirm-research-test --cache "$ADV_TEST_CACHE" \
  --output runs/aligned_dual_view_v1_research_test/summary_20260922_r1
```

Cache 验收：完整 selection、row/input/feature 哈希、`test_cwt.npy`、`target_read=false` 与成功回执。每项评价验收：固定 epoch/hash、2,310 个同序 test rows、eligible 指标有限、完整 access trace 与成功回执。实际耗时/显存尚未测量，nDTW 等辅助指标在 CPU 上计算。

## 6. 汇总与证据边界

汇总要求每个锁定 entry 恰有一个成功评价，且所有失败尝试保留、没有未闭合尝试；拒绝 partial seed、模型错配、损坏产物、跨 cache 或不一致的 sample/precision/code。

输出 `per_seed.csv`、`across_seed.csv`、`paired_deltas.csv` 和来源 manifest。Primary 的每 seed sample direct mean 再按三个 seed 等权汇总，并与 joint、waveform 报告配对差。辅助指标如存在合法 NA，保留 NA，并报告有限 seed 数，不静默缩小三 seed 分母。不得从 test 指标选择本轮 epoch，或把描述性 seed 波动解释成人群显著性。

本阶段不自动引入 W0/C201 的新 test 评价，不与 center-30/60/90 的不同任务区间混表。若后续需要额外 anchor 或新结构，另行定义其比较口径和阶段授权。

## 7. 已完成的工程验证

2026-09-22 的最小定向集合 **12 项通过（32.13 s）**：11 项 research-test 专项测试，以及既有 train/validation 配置拒绝 test 的边界测试。覆盖固定锁可复现、授权标志先于数据访问、完整样本/subject 隔离、实际 CWT 的 synthetic NPZ 缓存、全部预测完成后读取 target、12 候选矩阵汇总、错配/损坏/非有限输出拒绝、失败重试、并发互斥和未闭合尝试拒绝。

涉及模型前向和指标计算的用例使用 disposable 数据与 checkpoint；模型接线使用显式 CPU Mamba 替身。候选锁的来源核验只读既有 validation 产物。训练执行代码身份仍为 `19b92747cb2a26a5cf45d6954aeebaae1ce70b3de5a01a266677de4761284040`。真实 test cache、原生 GPU test 评价和真实 test 汇总尚未执行。

当前实现哈希与验证记录见 `aligned_dual_view_v1_research_test_implementation_lock_20260922.json`。
