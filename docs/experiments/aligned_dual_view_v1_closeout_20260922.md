# ADV-v1 实验收尾索引

日期：2026-09-22。状态：**本轮实现、四配置三 seed formal 训练、validation 汇总、固定候选 research-test 评价与汇总均已完成，阶段关闭。**

本文件为 ADV-v1 的当前状态入口。早期协议和实现锁中的“尚未执行”等描述是对应实现阶段的历史状态；实际完成情况以本索引、两份结果报告和原始产物完成回执为准。原执行协议及源码已纳入哈希身份，保持原文以便复核。已完成的 cache、训练、评价和汇总不重复执行。

## 1. 阅读入口

| 内容 | 位置 |
|---|---|
| 网络、训练及 validation 执行契约 | [实现协议](aligned_dual_view_v1_protocol_20260920.md) |
| Validation 完整结果与来源核对 | [validation 结果](aligned_dual_view_v1_results_20260922.md) |
| 固定 test 候选及访问边界 | [research-test 协议](aligned_dual_view_v1_research_test_protocol_20260922.md) |
| Research-test 完整结果与来源核对 | [research-test 结果](aligned_dual_view_v1_research_test_results_20260922.md) |
| 核心工程验证 | [source lock](aligned_dual_view_v1_source_lock_20260920.json) |
| Train/validation 实现与测试记录 | [runtime lock](aligned_dual_view_v1_runtime_lock_20260920.json) |
| Research-test 实现与测试记录 | [test implementation lock](aligned_dual_view_v1_research_test_implementation_lock_20260922.json) |
| 十二个固定 checkpoint 的身份 | [candidate lock](../../configs/aligned_dual_view_research_test_v1/candidate_lock_20260922.json) |

## 2. 实现与实验范围

- 分支：`codex/aligned-dual-view-v1`。
- 工作树：`/mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1`。
- 开发基点：`cebc7f5a478da492458fe2c0f3f8d8acdf2f2001`。正式运行时的 Git 状态、逐文件哈希与源码快照保存在各产物目录中；收尾提交用于保存实现，不替代实际运行身份。
- 网络：180 s、100 Hz 输入；波形卷积与 97 尺度 CWT 幅值经相同 FIR 对齐到 10 Hz；D64、六层双向 Mamba2 主干，输出插值回原时间网格。
- 固定配置：`joint`、`waveform`、`cwt`、`joint_scale_mean`；训练 seeds 为 20260811、20260812、20260813。
- 每项 formal 训练 80 epochs、6,400 次 optimizer 更新、effective batch 128；按完整 validation Local RR 最小值选择最早并列 epoch。
- Validation 每项 2,675 窗口；research-test 每项 2,310 窗口，四配置三 seed 共十二项。
- 数据划分、target、Pi、loss、核心指标及窗口 direct mean 聚合沿用执行契约。三个训练 seed 等权汇总，seed SD 仅描述训练随机性。

| 实现 | 路径 |
|---|---|
| 网络、CWT、FIR、训练与 validation | `resp_train/aligned_dual_view/` |
| 网络与训练配置 | `configs/aligned_dual_view_v1/` |
| 核心工程检查 | `scripts/check_aligned_dual_view_v1.py` |
| Train/validation 管线 | `scripts/run_aligned_dual_view_v1.py` |
| Research-test 管线 | `resp_eval/adv_test/`、`scripts/eval_aligned_dual_view_v1_research_test.py` |
| 专项 CPU 测试 | `tests/test_aligned_dual_view_v1.py`、`tests/test_aligned_dual_view_runtime.py`、`tests/test_aligned_dual_view_research_test.py` |

## 3. 完成产物与冻结身份

以下路径相对于本工作树；文件均原位保留。

| 阶段 | 产物位置 | 完成情况 |
|---|---|---|
| Train/validation 正式 cache | `runs/aligned_dual_view_v1/cache_formal_20260921_r1` | 完成 |
| Formal 训练及 validation | `runs/aligned_dual_view_v1/{view}_seed{seed}_formal_20260921_r1` | 12/12 完成 |
| Validation 完整汇总 | `runs/aligned_dual_view_v1/summary_full_20260922_r1` | 完成 |
| Research-test input-only cache | `runs/aligned_dual_view_v1_research_test/cache_20260922_r1` | 完成 |
| Research-test 评价 | `runs/aligned_dual_view_v1_research_test/cache_20260922_r1_evaluations/{view}_seed{seed}/attempt_r1` | 12/12 完成 |
| Research-test 完整汇总 | `runs/aligned_dual_view_v1_research_test/summary_20260922_r1` | 完成 |

Smoke cache 和训练记录同样保留在 `runs/aligned_dual_view_v1/`，正式比较使用上表 formal 与 research-test 产物。

| 身份 | SHA-256 |
|---|---|
| 训练执行代码 | `19b92747cb2a26a5cf45d6954aeebaae1ce70b3de5a01a266677de4761284040` |
| Research-test 执行代码 | `3a2b0c877c4e6832ff3d997ddafa96048cf9d074994e70e5a59ba948fce66d55` |
| Candidate lock 文件 | `146e89f4c196fd1b2742ba66bb3ad600f11e25a6ccf98ebfcbc7ece012947418` |
| Validation summary manifest | `cdc03f23f299b24bbf6234cd73c5c1678d7e9d7b22065e7f8cef58f9a0d1577d` |
| Research-test summary manifest | `b21da372dfca62807831e97362c1a3cd81fe0b84657c1b5bca7972cd74b12bc3` |

各 summary 的 `manifest.json` 列出完整来源，`per_seed.csv`、`across_seed.csv`、`paired_deltas.csv` 保留逐 seed、跨 seed 及配对结果。Checkpoint 路径、哈希和 selected epoch 以 candidate lock 为准。Cache 身份与比较身份见对应结果报告和 manifest。

`runs/`、checkpoint、缓存与日志由文件系统保留，不进入 Git。恢复实验记录需要同时保留该工作树中的运行产物及代码提交；仅克隆 Git 不包含运行数据。

## 4. 保留的结论与证据边界

1. Joint 相对 waveform 的 PCC 均值在 validation 和 research-test 均提高，两个 split 内均为三个 seed 同向；其他主指标存在取舍。
2. 尺度均值相对完整尺度投影的四项主指标均值更好，全局包络调制误差更大。这一方向在两个 split 上一致，当前证据不支持完整尺度投影的优越性。
3. 在 research-test 上，尺度均值相对 joint 的 Whole RR、Local RR、包络轨迹误差分别降低 7.11%、8.30%、4.08%，三个 seed 均改善；全局包络调制误差均值增加 5.29%，PCC 均值增加 0.00330。
4. CWT 单视图在当前结构与训练协议下明显较弱。其 research-test IBI 误差仅来自每 seed 38–55 个可解释窗口，必须结合 coverage 与计数理解。
5. Test split 为 **reused research/development evidence**。本轮结果不构成未触及 held-out 上的独立确认，不作人群统计显著性或优于 W0/C201 的结论。

完整五主指标、辅助指标及 seed 差异均保留在结果报告中。属性取舍和比较边界属于本轮结论的一部分。

## 5. 验证记录与维护边界

- 核心及 train/validation CPU synthetic 测试：27 项通过，23.88 s；research-test 专项及入口边界测试：12 项通过，32.13 s。后者包含前者中的一个边界测试，两个数字不应简单相加为独立测试数。
- CLI help 和候选锁检查通过；真实 GPU 训练、评价与汇总已由用户完成。
- 完成后的只读核对覆盖固定 checkpoint 文件哈希、样本身份、指标文件哈希、访问阶段顺序及存储指标到汇总的数值一致性，详见两份结果报告。
- 原协议中的执行命令作为复现记录保留。成功产物不可覆盖或使用相同 identity 重跑，历史失败和 smoke 记录也应保留。文档修订使用 Git 历史追踪。

只读候选锁与当前源码身份检查入口：

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1
ADV_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  "$ADV_PY" scripts/eval_aligned_dual_view_v1_research_test.py check-lock
```

该命令检查锁文件、当前训练源码和前处理身份，不读取数据集，也不替代结果报告记录的完整产物核对。
