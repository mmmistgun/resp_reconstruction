# 当前 THO 与 CRD 实验入口

本文只描述当前冻结的新呼吸重建协议。旧 E/F/G probe、旧 loss、旧 metrics、旧 gate/topK 和历史 checkpoint 语义不再属于当前 workflow；旧代码与说明通过 Git 追溯，历史 run 原地保留。

唯一主协议见 `docs/experiments/loss_metrics_restart_plan_20260729.md`；CRD-v1.1 S0/S1 的规范附件见 `docs/experiments/crd_v1_protocol_20260808.md`。

## 当前固定口径

- 数据：2026-06-20 research v2 soft-z。
- 输入：`bcg_rawish_segment_soft_z_key`。
- target：`target_waveform_segment_soft_z_key`。
- 当前独立测试集评价集合：B0 PatchMixer、T2 宽频 native、T4 宽频 bandenergy、F0 固定呼吸带和 IEWT。T1/T3 已由 validation 退出。
- 训练 loss：`L_sync + 0.25 L_effort`；rhythm 与短期 polarity 已由消融删除。
- 正式输出：$\Pi=S\circ B$，统一 `0.05–0.70 Hz`。
- checkpoint：完整 validation Local RR MAE 最小 epoch。
- early stopping：关闭。
- 包络主指标：`envelope_trajectory_mae` 与 `global_envelope_modulation_error`；
  `target_stratified_envelope_spearman` 只按 train-frozen Low/Medium/High 分层补充报告。
- 独立测试集：现有 `test` 可在阶段性整理后重复评价，并可形成后续独立科研问题；不得用于重选既有run的epoch/checkpoint。
- CRD 训练与普通 `eval_crd.py` 仍只读 train/validation；S1C 只允许 candidate lock 中的 12 个 checkpoint 通过专用入口各读取一次现有 research-test。

## W0 测试集定性导出

按需入口新增 `index`、`select-cases`、`intervene-r3`、`render-intervention`；`render` 支持
`--cases`、`--views` 和分类 PNG 目录。完整参数、案例选择规则、四条件科学合同与命令见
[按需定性分析协议](../docs/experiments/w0_qualitative_analysis_protocol_20260927.md)。

入口：`scripts/export_w0_test_qualitative.py`；固定原始 W0 seed `20260812`、epoch `15`，
全量 2310 test windows，F0 固定呼吸频带与 IEWT 对照。
[专项协议与运行命令](../docs/experiments/w0_test_qualitative_export_plan_20260927.md)说明完整数据合同和验收条件。

- `export --output <新目录> --device cuda:0 --confirm-research-test-export`：保存预测、CWT、FiLM 张量与统计、RR/包络轨迹和逐窗口指标。真实数据/GPU 导出由用户执行。
- `render --source <完成的导出目录> --output <新绘图目录>`：离线生成四联图、FiLM/CWT 附图、轨迹附图与可检索 `index.html`；默认全部窗口 PNG。
- `render` 可指定 `--rows <row_id> ... --zoom 30 60`，从保存文件生成 PNG 局部放大图。
- `finalize --source <已有完整窗口导出目录>`：核验保存产物，追加描述性回放差异汇总、窗口索引和完成清单；不运行模型。验收采用 `qualitative-export-v2`，身份、完整性和非有限值检查保持严格。
- 快速 synthetic CPU 检查：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv/bin/python -m pytest tests/test_w0_test_qualitative.py -q`。

导出与绘图都拒绝覆盖已有目录；全量 FiLM 未压缩约 12.8 GB，建议为导出预留至少 20 GB，图形另计。
单 seed 按已有 test 表现选择用于定性展示；模型性能仍按已有三 seed 结果报告。

## ADV 融合方式 × 位置 v1

当前状态：六组工程验收、18组formal、validation与research-test及完整汇总均已完成并关闭；
统一入口为[收尾索引](../docs/experiments/adv_fusion_factorial_v1_closeout_20260923.md)。
以下协议中的命令作为复现记录保留，已完成阶段不重复执行。
Validation结果与来源核对见[完成报告](../docs/experiments/adv_fusion_factorial_v1_results_20260923.md)。

协议：[融合因子实验](../docs/experiments/adv_fusion_factorial_v1_protocol_20260922.md)；
命令与验收：[执行说明](../docs/experiments/adv_fusion_factorial_v1_commands_20260922.md)；
训练过程：[early stopping评估](../docs/experiments/adv_fusion_factorial_v1_earlystop_review_20260922.md)。
入口为 `scripts/run_adv_fusion_factorial_v1.py`，实现为 `resp_fusion/`，配置为
`configs/adv_fusion_factorial_v1/`。固定六组 × 三seed，共18次formal，最大80epochs；
统一early stopping：min_epochs30、patience15、min_delta0，学习率计划保持80epochs；
支持独立cache、训练、validation、完整因子汇总、原生参数统计和用户执行的GPU合成验收。
历史训练曲线只读分析入口为 `scripts/review_adv_fusion_training_history.py`。

## ADV 融合因子 research-test

当前状态：18项固定checkpoint评价及完整汇总已完成，见[research-test结果报告](../docs/experiments/adv_fusion_factorial_v1_research_test_results_20260923.md)。

专项协议及用户执行命令：[固定18项research-test](../docs/experiments/adv_fusion_factorial_v1_research_test_protocol_20260923.md)。
入口为 `scripts/eval_adv_fusion_factorial_v1_research_test.py`，实现位于 `resp_eval/fusion_test/`。
`check-lock` 仅读取固定候选锁与源码；`cache/evaluate/summary` 要求显式
`--confirm-research-test`。每项完成全部input-only预测后再读取target，完整汇总保留
方式、位置、交互差分和辅助指标。候选沿用es30p15训练的18个已选checkpoint。

## ADV-v1：时间对齐双视图网络

当前状态：本轮四配置 × 三 seed 的 formal 训练、validation 和 research-test 均已完成并关闭。
当前状态与证据入口：[ADV-v1 收尾索引](../docs/experiments/aligned_dual_view_v1_closeout_20260922.md)。
原协议中的运行命令作为复现记录保留；已完成的 cache、训练、评价和汇总不重复执行。

专项说明：`docs/experiments/aligned_dual_view_v1_protocol_20260920.md`。
模型核心位于 `resp_train/aligned_dual_view/`，配置位于 `configs/aligned_dual_view_v1/`。
独立管线入口为 `scripts/run_aligned_dual_view_v1.py`，包含 `cache`、`train`、
`validation` 和 `summary`。`experiment.yaml` 固定 train/val、既有 loss/metrics、
80 epochs、effective batch 128；真实数据/GPU 的用户执行命令及产物验收见专项说明第 7 节。

```bash
ADV_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SSQ_PARALLEL=0 SSQ_GPU=0 \
  NUMBA_CACHE_DIR=/tmp/adv1-numba-cache \
  "$ADV_PY" -m pytest -q tests/test_aligned_dual_view_v1.py tests/test_aligned_dual_view_runtime.py
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  "$ADV_PY" scripts/check_aligned_dual_view_v1.py describe
```

`describe` 用原生 Mamba 实例统计参数，不运行 GPU。用户执行的原生 GPU 合成
前后向命令和报告验收标准见专项说明；测试替身的 CPU 结果不代表 GPU 已通过。

## ADV-v1：research-test

十二个固定 checkpoint 的评价与完整汇总已完成，结果见
[research-test 结果报告](../docs/experiments/aligned_dual_view_v1_research_test_results_20260922.md)。
专项协议：`docs/experiments/aligned_dual_view_v1_research_test_protocol_20260922.md`。
入口：`scripts/eval_aligned_dual_view_v1_research_test.py`；实现位于 `resp_eval/adv_test/`。
候选锁固定四配置 × 三 seed 的 validation-selected checkpoints；普通训练/validation
代码身份保持不变。`check-lock` 不访问数据集，实际 test 操作需要显式确认标志及阶段授权。

```bash
ADV_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  "$ADV_PY" scripts/eval_aligned_dual_view_v1_research_test.py check-lock
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SSQ_PARALLEL=0 SSQ_GPU=0 \
  NUMBA_CACHE_DIR=/tmp/adv1-numba-cache \
  "$ADV_PY" -m pytest -q tests/test_aligned_dual_view_research_test.py
```

用户执行的 input-only cache、十二项评价与完整汇总命令、输出位置和验收标准见专项协议。
本实现不重新训练或选择 checkpoint；所有结果继续使用 reused research/development evidence 标签。

## E1：W0 尺度重排敏感性审计

专项协议：`docs/experiments/e1_w0_scale_topology_protocol_20260915.md`。入口为
`scripts/eval_e1_w0_scale_topology.py`，当前矩阵是完整 validation 的三个 W0 checkpoint × 四条件。
三个 FULL 均通过冻结五指标 `rtol=1e-3, atol=0` 复现后，执行三个尺度重排条件并自动汇总。

实现准备与 synthetic CPU 定向验证：

```bash
./.venv/bin/python -m pytest tests/test_e1_scale_topology.py -q
./.venv/bin/python scripts/eval_e1_w0_scale_topology.py prepare-locks
```

`prepare-locks` 只读核验 W0/validation 来源文件的字节身份，并排他生成索引锁与实现锁；已生成的锁直接复用。
正式执行前提交待执行代码、协议和锁，保持工作树干净。以下两步由用户执行：

```bash
./.venv/bin/python scripts/eval_e1_w0_scale_topology.py gpu-smoke --device cuda:0
./.venv/bin/python scripts/eval_e1_w0_scale_topology.py validation \
  --device cuda:0 \
  --gpu-receipt '/实际完成的gpu_smoke_attempt目录'
```

第一步只使用合成输入与新初始化 W0；将其输出目录作为第二步的 `--gpu-receipt`。两步都创建独立 attempt，
输出根为 `runs/e1_w0_scale_topology_v1/`。完整评价保留约 10.3 GiB 的 FULL raw FiLM 文件供配对审计，
并交付 32100 条逐窗口指标、同量 FiLM 配对记录、三 seed 汇总及 delta 表。
以 `manifest.json` 和 `freeze_receipt.json` 确认完成，失败 lifecycle 与部分产物原地保留。

## E4：结项入口

截至 2026-09-22，E4 的 15 次新训练、15 次候选 test 评价、两轮 validation 机制检查及效率测量均已完成。
当前状态、完整五指标比较、结论边界和产物索引统一见
[E4 结项报告](../docs/experiments/e4_closeout_20260922.md)。以下 E4 命令保留用于追溯执行入口；已完成阶段保持冻结。

结项数据见 [性能、代价与机制证据数据集](../docs/experiments/e4_closeout_data_20260922/README.md)。
`collect_e4_closeout_data.py` 从冻结 CSV/benchmark JSON 整理数据并登记来源身份，可用 `--output` 指定新的核对目录。

## E5：W0 时域前端替换

专项协议为 [E5 时域前端方案](../docs/experiments/e5_temporal_frontend_protocol_20260922.md)。唯一候选
`e5_tfe101_aa10_res_w0` 使用固定 511-tap `100→10 Hz` 抗混叠、`1→96,k11` 局部 embedding
和 zero-init `96-channel,k5` depthwise residual；不在前端增加 normalization 或 pointwise mixer。
W 条件分支、六层 BiMamba2、FiLM、decoder 和完整 loss 固定。训练最多 80 epochs/6400 planned updates；
完整 validation Local-RR early stop 固定为 `min_epoch=30 / patience=15 / min_delta=0`，最早在 epoch 30 触发，
LR 日程不因提前停止重标定。

P1 代码、synthetic CPU 验收、三 seed formal 和一次性 validation 汇总均已完成。Validation 结果为
`net negative for this candidate`，不替换 W0；完整数值与来源见
[E5 validation 结果](../docs/experiments/e5_temporal_frontend_results_20260923.md)。训练阶段已冻结，不重跑相同身份。

训练实现的历史验收入口为：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e5_temporal_frontend.py tests/test_crd_tf_v1_model.py \
  tests/test_resp_temporal_v1_models.py -q
./.venv/bin/python scripts/run_e5_temporal_frontend.py check-config
```

Test 专项协议为
[E5 research-test 附件](../docs/experiments/e5_temporal_frontend_test_protocol_20260923.md)，入口为
`scripts/eval_e5_temporal_frontend_test.py`。它固定使用 validation 预选的 epoch `9/13/5`，复用同 seed
冻结 W0 test 指标作对照；test 只作为重复使用 research-test 上的开发性描述，不改变 validation 决定。
三 seed test 与一次性汇总均已完成，结果继续支持 E5 net-negative 判断；冻结汇总位于
`runs/e5_temporal_frontend_test/summary/summary_5b452091dc28_20260923T031427Z_c128f3353d19/`。
以下命令只保留为历史入口：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e5_temporal_frontend_test.py -q
```

```bash
./.venv/bin/python scripts/eval_e5_temporal_frontend_test.py prepare-lock

for E5_TEST_SEED in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/eval_e5_temporal_frontend_test.py evaluate \
    --seed "$E5_TEST_SEED" --device cuda:0 || break
done

./.venv/bin/python scripts/eval_e5_temporal_frontend_test.py summarize --runs \
  '/seed_20260811完成attempt' '/seed_20260812完成attempt' '/seed_20260813完成attempt'
```

## E6：20-Hz 学习解调后形成 10-Hz latent

状态：**已结项并关闭。** 完整来源、validation/test 指标、效率与解释边界统一见
[E6 结项记录](../docs/experiments/e6_temporal_frontend_closeout_20260924.md)。以下内容保留为历史入口。

专项协议为 [E6 时域前端方案](../docs/experiments/e6_temporal_frontend_protocol_20260923.md)。唯一候选
`e6_tfe201_aa20_demod10_w0` 直接复用 RTM 冻结 `TemporalStem`：显式 `100→20 Hz` 抗混叠，
在 20 Hz 完成 `1→48` carrier-sensitive filtering、depthwise filtering、`48→96` projection 及
channel-only normalization/SiLU，再显式 `20→10 Hz`。只替换 W0 `base.frontend`；W 分支、
六层 BiMamba2、FiLM、decoder、loss 和数据保持固定。参数量只报告，不参与结构选择。

训练最多 80 epochs/6400 planned updates，继续使用完整 validation Local-RR early stop：
`min_epoch=30 / patience=15 / min_delta=0`，LR 不随提前停止重标定。实现锁、GPU acceptance、
benchmark、三 seed formal 和一次性 validation 汇总均已完成；三个 seed 都在 epoch 30 停止并选择 epoch 4。
Validation 形成 trajectory 局部收益但 Local RR/PCC 等整体退化的属性权衡，不替换 W0。完整记录见
[E6 validation 结果](../docs/experiments/e6_temporal_frontend_results_20260924.md)。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e6_temporal_frontend.py tests/test_e5_temporal_frontend.py \
  tests/test_resp_temporal_v1_models.py tests/test_crd_tf_v1_model.py -q

./.venv/bin/python scripts/run_e6_temporal_frontend.py check-config
```

Test 专项协议为
[E6 research-test 附件](../docs/experiments/e6_temporal_frontend_test_protocol_20260924.md)，入口为
`scripts/eval_e6_temporal_frontend_test.py`。固定使用三个 epoch-4 checkpoint，复用同 seed W0 test 指标；
结果只作重复使用 research-test 上的开发性描述，不改变 validation 决定。三 seed test 与一次性汇总均已完成；
结果在五项 candidate mean 上均不利，E6 不替换 W0。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e6_temporal_frontend_test.py -q
```

Test lock、三个 evaluation attempt 与 summary 已冻结，命令只保留 provenance；失败 attempt 和已完成 identity 均不可覆盖。

## E4：R3 时间结构与 GN 统计控制

入口 `scripts/eval_e4_r3_temporal_normalization.py`，科学矩阵见
[执行协议](../docs/experiments/e4_r3_temporal_normalization_protocol_20260921.md)，当前实现和命令见
[r2 修复附件](../docs/experiments/e4_r3_temporal_normalization_r2_20260921.md)。
原 W0 三个 selected checkpoint、完整 validation、每 seed 22 条件；r2 GPU smoke、三 seed 评价及汇总已完成。
共享 signals 与完整结果见 [R3/GN 结果记录](../docs/experiments/e4_r3_temporal_normalization_results_20260921.md)。

## E4：频带编码与聚合干预诊断

专项协议：`docs/experiments/e4_band_encoding_aggregation_protocol_20260918.md`。
入口 `scripts/eval_e4_band_encoding_aggregation.py` 固定 W0＋四候选、三个 seed、原 selected checkpoint，
在完整 validation 上保存聚合前 X/实际 alpha，并比较均匀恢复、四区先验恢复、两种入口参考×四区替换。
完整 X 约 250.55 GiB；两种入口参考预先覆盖全矩阵，共 210 个条件。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest tests/test_e4_band_audit.py -q
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_band_encoding_aggregation.py prepare-lock \
  --validation-summary '/mnt/disk_code/marques/resp_reconstruction/runs/e4_scale_aggregation_v2/summary/summary_71e6b8f753cc_20260918T055104Z_da596d615deb'
```

上述准备与全部 15 次 `evaluate`、`summarize` 均已完成，见
[频带/聚合结果记录](../docs/experiments/e4_band_encoding_aggregation_results_20260919.md)。checkpoint、原 cache 及评价保持原身份。

## W0 CWT-FiLM 调制行为分析

专项协议为 docs/experiments/w0_cwt_film_behavior_protocol_20260918.md。独立入口
scripts/analyze_w0_cwt_film_behavior.py 只分析三个冻结 W0 validation checkpoint，不训练、不访问 test。
分析复用冻结逐窗口误差和历史 corrected FiLM 统计，新增实际特征变化、raw/有效参数分布、
正负多阈值边缘、通道/时间结构及质量关联。
当前实现锁为 docs/experiments/w0_cwt_film_behavior_implementation_lock_20260918_r2.json。

实现和 synthetic CPU 验证：

~~~bash
FILM_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
"$FILM_PY" -m pytest \
  tests/test_w0_cwt_film_behavior.py tests/test_crd_tf_v1_model.py -q
"$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py prepare-lock
~~~

提交实现与新锁并保持独立 worktree 干净后，由用户运行真实 smoke 和完整 validation：

~~~bash
"$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py smoke \
  --seed 20260811 --device cuda:0

FILM_SMOKE='/完成的smoke attempt目录'
for FILM_SEED in 20260811 20260812 20260813; do
  "$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py analyze \
    --seed "$FILM_SEED" --device cuda:0 --smoke-receipt "$FILM_SMOKE" || break
done

"$FILM_PY" scripts/analyze_w0_cwt_film_behavior.py summarize \
  --runs '/seed 20260811 attempt' '/seed 20260812 attempt' '/seed 20260813 attempt'
~~~

summary 固定案例身份后，render-cases --summary ... --device cuda:0 只重建包含案例的原始
batch-128 上下文并保存选中窗口；最后用 finalize --summary ... --cases ... 生成中文结论与闭合回执。
产物根为主仓库的 runs/w0_cwt_film_behavior_v1/；每阶段排他创建 attempt，失败现场保留。

## E8：FiLM 条件末端 × 波形解码端

状态：P0–P6 全部完成并关闭。十二个模型组成
`4 condition refiners × 3 decoders` 完整矩阵；三 seed 共 36 个 formal cell、P5 validation
汇总、36 个固定 checkpoint research-test 评价及完整汇总均已冻结。最终状态与结论统一由
`docs/experiments/e8_film_decoder_redesign_v1_closeout_20260926.md` 路由。

以下检查与执行命令只保留为 provenance；已完成阶段不得重复执行：

```bash
./.venv/bin/python -m pytest tests/test_e8_film_decoder_redesign_v1.py -q
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py check-p1
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py describe
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py formal-plan
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py \
  gpu-acceptance --device cuda:0
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py \
  benchmark --device cuda:0
./.venv/bin/python scripts/summarize_e8_film_decoder_redesign_v1.py \
  --confirm-p5-summary
```

Formal 执行合同见
`docs/experiments/e8_film_decoder_redesign_v1_p3_formal_protocol_20260925.md`；validation 输出与证据边界见
`docs/experiments/e8_film_decoder_redesign_v1_p5_validation_summary_protocol_20260926.md`，结果见
`docs/experiments/e8_film_decoder_redesign_v1_validation_results_20260926.md`。

E8 research-test 专项入口为 `scripts/run_e8_film_decoder_redesign_v1_test.py`，协议见
`docs/experiments/e8_film_decoder_redesign_v1_research_test_protocol_20260926.md`。36-checkpoint allowlist 已冻结为
`runs/e8_film_decoder_redesign_v1/research_test/allowlist/allowlist_8b67c189b10c_20260926T083406Z_b300e95e53bc`，SHA-256 为
`76ef0c8ac05cdddcf89caab46abb0a03b78d0d3e72d09b6ed5773521dc2576db`。生成阶段未读取 test arrays。
36 项评价与完整汇总均已完成；research-test 入口不得再次执行。

## E7：聚合前尺度编码 × 尺度聚合

状态：P2 已完成。独立六臂模型、严格配置、early-stop 回放、析因汇总、表征诊断、
source audit 与 P1 implementation lock 已建立。协议为
`docs/experiments/e7_scale_encoding_aggregation_factorial_protocol_20260924.md`。

```bash
./.venv/bin/python scripts/run_e7_scale_encoding_aggregation.py check-p1
./.venv/bin/python scripts/run_e7_scale_encoding_aggregation.py check-lock
./.venv/bin/python -m pytest tests/test_e7_scale_encoding_aggregation.py -q
```

冻结来源与实现分别为
`docs/experiments/e7_scale_encoding_aggregation_source_audit_20260924.json` 和
`docs/experiments/e7_scale_encoding_aggregation_p1_implementation_lock_20260924.json`。
GPU acceptance、benchmark、正式 18 次 train/validation 和 research-test 均按协议阶段门控由用户执行。

P3 GPU 工程入口已实现并由独立 engineering lock 固定，协议为
`docs/experiments/e7_scale_encoding_aggregation_p3_engineering_protocol_20260924.md`。用户执行：

```bash
./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p3.py check-lock
./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p3.py \
  gpu-acceptance --device cuda:0
./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p3.py \
  benchmark --device cuda:0
```

GPU acceptance 为 18 个 batch-1 cell 加 6 个 batch-128 cell；benchmark 为六臂、
eval/train 两场景、三组独立进程，共 36 个 measurement。二者只使用 synthetic 输入。

P3 已完成并由 `docs/experiments/e7_scale_encoding_aggregation_p3_closeout_20260924.md`
收口。P4 formal 入口复用该收口固定的单一 GPU receipt；正式训练由用户执行：

```bash
git status --short  # 必须无输出

E7_GPU_RECEIPT='runs/e7_scale_encoding_aggregation/gpu_acceptance/gpu_acceptance_773f5e78c5e8_20260924T082610Z_62e8f9b7eaef'

for E7_SEED in 20260811 20260812 20260813; do
  for E7_ARM in \
    s0_shallow__mean \
    s0_shallow__frequency_attention \
    s1_deep_local__mean \
    s1_deep_local__frequency_attention \
    s2_axis_spanning__mean \
    s2_axis_spanning__frequency_attention; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p4.py formal \
      --arm "$E7_ARM" --seed "$E7_SEED" --device cuda:0 \
      --gpu-receipt "$E7_GPU_RECEIPT" || break 2
  done
done
```

同一 P4 execution lock 下已完成 cell 拒绝重跑；失败现场保留，后续从未完成 cell 继续。全部完成后运行
`scripts/run_e7_scale_encoding_aggregation_p4.py check-completed` 核验唯一 18-cell 矩阵。

P5 复用 P4 execution lock。先生成 CPU validation summary，再由用户运行 18-cell selected-checkpoint
diagnostics，最后冻结完整 P5：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p5.py summarize

for E7_SEED in 20260811 20260812 20260813; do
  for E7_ARM in \
    s0_shallow__mean \
    s0_shallow__frequency_attention \
    s1_deep_local__mean \
    s1_deep_local__frequency_attention \
    s2_axis_spanning__mean \
    s2_axis_spanning__frequency_attention; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p5.py diagnose \
      --arm "$E7_ARM" --seed "$E7_SEED" --device cuda:0 || break 2
  done
done

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p5.py check-diagnostics
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p5.py finalize
```

P5 summary 与 diagnostics 只访问 validation；research-test 保持关闭。

P5 已完成并由 `docs/experiments/e7_scale_encoding_aggregation_p5_closeout_20260926.md`
收口。当前默认不再执行 validation 训练、诊断或汇总。

E7 research-test 已完成并由
`docs/experiments/e7_scale_encoding_aggregation_test_results_20260926.md` 收口。18 个固定 checkpoint
评价全部通过质量验收，完整汇总冻结于
`runs/e7_scale_encoding_aggregation/research_test/summary/summary_0fcba39436c9_20260926T040914Z_8436abbf29e6/`。
下列命令保留为历史执行记录：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python -m pytest tests/test_e7_scale_encoding_aggregation_test.py -q

./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py prepare-lock
./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py check-lock

for E7_TEST_SEED in 20260811 20260812 20260813; do
  for E7_TEST_ARM in \
    s0_shallow__mean \
    s0_shallow__frequency_attention \
    s1_deep_local__mean \
    s1_deep_local__frequency_attention \
    s2_axis_spanning__mean \
    s2_axis_spanning__frequency_attention; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      ./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py evaluate \
      --arm "$E7_TEST_ARM" --seed "$E7_TEST_SEED" --device cuda:0 || break 2
  done
done

./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py check-completed
./.venv/bin/python scripts/eval_e7_scale_encoding_aggregation_test.py summarize --completed
```

每项读取完整 2,310-window research-test，18 项共新增 41,580 条逐窗口指标。结果没有形成稳定五指标
Pareto 改善，validation 决定保持冻结。

## E4 v2：四种尺度聚合固定矩阵

状态：12 次训练、12 次 test 评价、GPU 验收、benchmark 与两份汇总均已完成并结项。下列为历史执行命令。

协议：`docs/experiments/e4_scale_aggregation_v2_protocol_20260917.md`。
当前工程修订：`docs/experiments/e4_scale_aggregation_v2_engineering_r2_20260917.md`；
当前实现锁为 `docs/experiments/e4_scale_aggregation_v2_implementation_lock_r2_20260917.json`。
四个 arm 为 `static_scale`、`scale_attention`、`frequency_attention`、`channel_region`，
各三个 seed、80 epochs / 6400 updates，共 12 次训练。新增参数分别为 97/784/792/384；
固定 W0 公共初始化、fill=65、完整 loss 与训练合同。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e4_aggregation_v2_model.py tests/test_e4_aggregation_v2.py \
  tests/test_e4_aggregation_v2_test.py tests/test_e4_aggregation_v2_engineering.py -q
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_aggregation_v2.py prepare-lock
```

实现锁排他创建，已有锁直接复用。提交代码、协议和锁并保持工作树干净后，用户执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_aggregation_v2.py gpu-acceptance --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_aggregation_v2.py benchmark --device cuda:0
```

验收包含四个 arm 各三个 seed 的 batch-1 对照，以及各一个 batch-128 验收，每项至少 5、最多 20 次更新。
全部新增参数均有本步非零梯度且相对初值真实变化后通过；逐步诊断及异常现场保存在各用例的 `*_diagnostics/`。
benchmark 为 W0 与四候选的 eval/train 两场景×三组，共 30 个独立进程。
将成功验收路径填入变量后，按固定顺序执行完整训练；子 shell 在工程失败时停止：

```bash
E4_V2_GPU_RECEIPT='/实际成功的v2_gpu_acceptance目录'
(
  for E4_V2_SEED in 20260811 20260812 20260813; do
    for E4_V2_ARM in static_scale scale_attention frequency_attention channel_region; do
      env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
        scripts/run_e4_aggregation_v2.py formal --arm "$E4_V2_ARM" --seed "$E4_V2_SEED" \
        --device cuda:0 --gpu-receipt "$E4_V2_GPU_RECEIPT" || exit $?
    done
  done
)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_aggregation_v2.py summarize --completed
```

`--completed` 仅定位当前锁下每个 arm/seed 的唯一成功 attempt，缺失或重复即失败；
汇总仍完整核验来源与 checkpoint。也可用 `--runs` 显式传入 12 个目录。
完成 cell 禁止重跑；工程失败后只单独执行未完成的 cell。

取得完整 validation 汇总目录后，准备 test 锁并提交，再由用户执行完整 12 次 test：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_aggregation_v2_test.py prepare-lock \
  --validation-summary '/实际完成的v2_validation_summary目录'

# 提交新生成的 test 锁并保持工作树干净后执行。
(
  for E4_V2_SEED in 20260811 20260812 20260813; do
    for E4_V2_ARM in static_scale scale_attention frequency_attention channel_region; do
      env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
        scripts/eval_e4_aggregation_v2_test.py evaluate \
        --arm "$E4_V2_ARM" --seed "$E4_V2_SEED" --device cuda:0 || exit $?
    done
  done
)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_aggregation_v2_test.py summarize --completed
```

训练输出根为 `runs/e4_scale_aggregation_v2/`，test 输出根为 `runs/e4_scale_aggregation_v2_test/`。
两种 split 均生成 15 行 seed metrics、60 行配对差值、20 行 `four_arm_comparison.csv`。
test 每次完整 2310 窗口、8 samp_id，12 次新记录共 27720 条；有限退化结果保留质量标志。

## E4：W0 四区域尺度聚合

状态：三 seed 训练、三次 test 评价、GPU 验收、benchmark 与两份汇总均已完成并结项。下列为历史执行命令。

专项协议：`docs/experiments/e4_w0_scale_aggregation_protocol_20260917.md`。
独立入口 `run_e4_w0_scale_aggregation.py` 保持 W0 公共初始化、fill=65 和完整目标，
增加零初始化的四区域 384→96 残差投影。全模型 1,256,714 参数；固定三 seed，
各 80 epochs / 6400 updates，范围为 train/validation。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e4_scale_aggregation.py tests/test_e2_effort_ablation.py -q
./.venv/bin/python scripts/run_e4_w0_scale_aggregation.py prepare-lock
```

已存在的实现锁直接复用，准备命令拒绝覆盖。提交实现、协议与锁、保持干净工作树后，
由用户执行 GPU synthetic 验收和匹配效率测量：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_w0_scale_aggregation.py gpu-acceptance --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_w0_scale_aggregation.py benchmark --device cuda:0
```

GPU 验收包括三个 seed 的 batch-1 初始等值/三步梯度检查，以及 seed 20260811 的
batch-128 三步 acceptance。正式训练要求相同代码锁、commit 和关键运行环境。
benchmark 使用合成输入和独立进程，报告 batch-1 eval 与 batch-128 training 的耗时和显存。

```bash
E4_GPU_RECEIPT='/实际成功的E4_gpu_acceptance目录'
for E4_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
    scripts/run_e4_w0_scale_aggregation.py formal --seed "$E4_SEED" \
    --device cuda:0 --gpu-receipt "$E4_GPU_RECEIPT" || break
done

env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_w0_scale_aggregation.py summarize --runs \
  '/seed_20260811的完成attempt目录' \
  '/seed_20260812的完成attempt目录' \
  '/seed_20260813的完成attempt目录'
```

输出根为 `runs/e4_w0_scale_aggregation_v1/`。完成以 manifest/freeze receipt 为准，
失败现场保留，同身份完成 seed 禁止重跑。汇总包括完整五指标、配对差值、seed SD、
方向数、质量标志和容量/计算口径。test 需要匹配的后续专项附件。

## E4 test：固定四区域聚合 checkpoint 的评价

附件：`docs/experiments/e4_w0_scale_aggregation_test_protocol_20260917.md`。
固定 E4 selected epoch **10/5/17**，每 seed 完整评价 2310 窗口、8 个 samp_id，
与同 seed 冻结 W0 test CSV 配对。独立模型工厂严格加载 E4 状态，保留完整目标与四区配置。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python -m pytest tests/test_e4_scale_aggregation_test.py -q
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/eval_e4_w0_scale_aggregation_test.py prepare-lock
```

锁只创建一次；准备仅核验冻结产物与 metadata，真实 test 数组在用户执行阶段读取。
提交本轮 test 源码、协议与锁，保持干净工作树后执行：

```bash
for E4_TEST_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    ./.venv/bin/python scripts/eval_e4_w0_scale_aggregation_test.py evaluate \
    --device cuda:0 --seed "$E4_TEST_SEED" || break
done

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/eval_e4_w0_scale_aggregation_test.py summarize --runs \
  '/seed_20260811的完成test_attempt目录' \
  '/seed_20260812的完成test_attempt目录' \
  '/seed_20260813的完成test_attempt目录'
```

输出根 `runs/e4_w0_scale_aggregation_test_v1/`，三 seed 新记录共 6930 条。
完成以 manifest/freeze receipt 为准，失败现场保留；五指标、target eligibility、
退化计数与质量标志共同交付。test 是复用研究测试集，结论与 validation 分开解释。

## E3：W0 五指标关联与 RR—努力不一致比例

**已完成**：两个 split、三个 seed 的再分析及完整敏感性矩阵已冻结。
结果、统计定义与验收见 `docs/experiments/e3_w0_metric_association_results_20260917.md`。
下列命令为执行记录，后续引用已有完成产物。

专项协议：`docs/experiments/e3_w0_metric_association_protocol_20260916.md`。
三个冻结 W0 的 validation/test 分别分析，读取已有指标 CSV 与时间元数据，使用 CPU。
交付 Spearman 相关、逐 samp_id 分布、完整阈值敏感性矩阵及不重叠窗口视图。
默认描述性设置为两项 RR 误差同时 ≤1 bpm，其他属性超过 validation 参考分布第 75 百分位；
完整统计定义、分母与证据边界以专项协议为准。

实现准备（锁已生成时复用）：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python -m pytest tests/test_e3_metric_association.py -q
./.venv/bin/python scripts/analyze_e3_w0_metrics.py prepare-lock
```

提交实现与锁后执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/analyze_e3_w0_metrics.py analyze
```

输出位于 `runs/e3_w0_metric_association_v1/analysis/`，以 manifest 与 freeze receipt 确认完成。
相同锁的已完成分析校验后复用；失败 attempt 保留。

## E2：最终 W0 相对努力损失消融

**已完成**：三个 seed 的训练、validation 与 test 及配对汇总均已冻结。
当前状态、结果路径与解释见 `docs/experiments/e2_w0_effort_results_20260916.md`。
下列命令保留为执行记录；后续分析读取已有完成产物。

专项协议：`docs/experiments/e2_w0_effort_ablation_protocol_20260916.md`。E2 使用原生 W0
训练器，仅将 `effort_weight` 从 0.25 设为 0，按三个冻结 seed 分别训练 80 epochs / 6400 updates，
随后与原始冻结完整 W0 validation 结果配对。完整运行步骤、来源及验收口径以专项协议为准。

实现准备：

```bash
./.venv/bin/python -m pytest tests/test_e2_effort_ablation.py -q
./.venv/bin/python scripts/run_e2_w0_effort_ablation.py prepare-lock
```

准备锁、提交实现与协议并保持干净工作树后，由用户执行 synthetic GPU 验收：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e2_w0_effort_ablation.py gpu-smoke --device cuda:0
```

将成功的 E2 GPU attempt 路径作为 `--gpu-receipt`，分别执行三个 seed：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e2_w0_effort_ablation.py formal \
  --seed 20260811 --device cuda:0 --gpu-receipt '/实际E2_gpu_smoke目录'
```

其余 seed 为 `20260812`、`20260813`。三个完成 attempt 齐全后执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e2_w0_effort_ablation.py summarize --runs \
  '/seed_20260811的完成attempt目录' '/seed_20260812的完成attempt目录' '/seed_20260813的完成attempt目录'
```

输出根为 `runs/e2_w0_effort_ablation_v1/`。每个 seed 保存原生 history、best/final checkpoint、
逐窗口 validation metrics、配置/来源/环境回执；外层 `manifest.json` 与 `freeze_receipt.json` 标识完成。
已完成的同身份 seed 直接复用，失败 attempt 和部分产物保留；完整汇总要求全部三个 seed。

## E2 test：固定损失消融 checkpoint 评价

三个 seed 的 test 评价与汇总已完成，结果索引同上。

专项协议：`docs/experiments/e2_w0_effort_test_protocol_20260916.md`。
入口为 `scripts/eval_e2_w0_effort_test.py`；固定 E2 三个 validation-selected checkpoint
（epoch 31/36/33），每个评价完整 2310-window test，并与同 seed 的冻结 W0 test 结果配对。

```bash
./.venv/bin/python -m pytest tests/test_e2_effort_test.py -q
./.venv/bin/python scripts/eval_e2_w0_effort_test.py prepare-lock
```

准备阶段核验已冻结结果与 metadata，正式 test 数组在评价阶段读取。锁准备完成并提交本轮代码、协议和锁后，
由用户逐 seed 执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/eval_e2_w0_effort_test.py evaluate --seed 20260811 --device cuda:0
```

其余 seed 为 `20260812`、`20260813`。三个成功 attempt 后执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/eval_e2_w0_effort_test.py summarize --runs \
  '/seed_20260811的完成test_attempt目录' '/seed_20260812的完成test_attempt目录' '/seed_20260813的完成test_attempt目录'
```

输出根为 `runs/e2_w0_effort_test_v1/`，新增 6930 条逐窗口记录。
test 五指标、配对变化、来源/访问/环境回执与不可覆盖 manifest 独立保存，完整运行循环和验收标准见专项协议。

## 数据与 split 审计

数据审计：

```bash
./.venv/bin/python scripts/audit_tho_dataset.py \
  --config configs/tho_research_v2.yaml \
  --output /tmp/tho_restart_audit.csv
```

Split 独立性审计：

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config configs/tho_research_v2.yaml \
  --output-dir runs/audits/split_independence_restart
```

CRD S1C 使用严格 CRD 配置 loader 的复现命令：

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config-kind crd \
  --config configs/crd_v1/crd_105_direct_coarse.yaml \
  --output-dir runs/crd_v1/crd_s1c_split_audit_20260809
```

这些审计不是每个训练 run 的重复前置步骤。协议首次实现或数据/split 发生变化时执行并保存结果；普通 run 只保留加载、shape 与 finite 断言。

### RTM-v1 train-only 信号审计（已完成并关闭）

该入口只允许完整 eligible train，配置哈希、频带、算子、输出目录与 access receipt schema 均已冻结；不接受 split、subset、频带或输出覆盖。完整审计已成功完成，以下命令只作 provenance，**不得重复运行**：

```bash
./.venv/bin/python scripts/audit_resp_temporal_v1_signal.py --config configs/resp_temporal_v1/signal_audit_train_v1.yaml
```

Receipt/manifest/summary SHA-256分别为`1dbdc18836c13d8c2cd888c186477c40b044e190afed3e8817c082bd9b235428 / 19ab1ece34e40b0b970f1f620f20015742c7ae1de57df3f20a2f904403763eb2 / 61b5c5d17b11b71731bf399ac2d12c684c724ac1aebe4979b577797a96b49483`。Signal/candidate、exact CPU与GPU engineering均已完成并锁定；这些入口和产物不得重跑或覆盖。

### RTM-v1 formal（仅用户手动执行）

冻结矩阵固定5项candidate×3 seeds。旧`formal_v1.yaml`在0/15时被双GPU修订取代，不得使用。每条命令只运行一个identity，不接受data/split/checkpoint/epoch/batch/device/output/resume override；`LOCKED_GPU`与candidate/seed分组不匹配时会在数据/GPU访问前失败：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<LOCKED_GPU> ./.venv/bin/python scripts/train_resp_temporal_v1.py --plan configs/resp_temporal_v1/formal_dual_gpu_v2.yaml --candidate-id <LOCKED_ID> --seed <LOCKED_SEED>
```

`gpu_0/CUDA_VISIBLE_DEVICES=0`固定8项，`gpu_1/CUDA_VISIBLE_DEVICES=1`固定7项；两组可并行、组内按plan串行。输出固定为`runs/resp_temporal_v1/formal/<candidate_id>/seed_<seed>/`，目录必须预先不存在。失败目录必须保留且同identity不得重跑；15/15完成前不得汇总partial validation，也没有独立eval或research-test入口。

Formal 15/15与冻结validation summary现均已完成，以上训练及以下summary入口只作provenance，不得重跑、覆盖或补写：

```bash
./.venv/bin/python scripts/summarize_resp_temporal_v1.py --config configs/resp_temporal_v1/validation_summary_v1.yaml
```

固定输出为`runs/resp_temporal_v1/formal_validation_summary_v1/`且禁止覆盖。Receipt/manifest SHA-256=`6f9f1e873b8910b22241bc0e9f2c510909835b0bbc2a9edc12f0e1788fa59aad / aab22094d6efd11927c952e9f12bcbab24e30cedc9a2a5f282f61056f1f193dc`，用户确认的validation lock为`docs/experiments/resp_temporal_v1_validation_lock_20260821.json`，SHA-256=`989ef0a3a5941ead3e80aba25606878f88d315bd23a1cf5ca4260317f3cffce6`。命令只读formal validation metrics/receipts与GPU engineering artifacts，不读取checkpoint内容、dataset/signal或research-test；RTM-v1 validation阶段现已关闭。

五候选validation `mean ± SD`主指标表已冻结为`docs/experiments/resp_temporal_v1_primary_metrics_table_20260821.md`。用户随后明确授权独立测试集评价；以下一次性入口现已完成，只作provenance，不得重跑：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/eval_resp_temporal_v1_research_test.py --config configs/resp_temporal_v1/research_test_v1.yaml
```

固定输出为`runs/resp_temporal_v1/research_test/rtm_v1_research_test_v1/`。入口已依次评价五候选×三seed的15个冻结`checkpoint_best_local_rr.pt`，保存逐sample metrics、15行seed summary和5行`mean ± sample SD`主指标表；不训练、不重选checkpoint/candidate，不读取validation metrics/target/prediction，也不计算secondary、Pareto、排名或p-value。Receipt/manifest/主指标表 SHA-256=`f9b1b216f80c4bde5a7be27aa9df76c66b8e3765ebe491ebe45fa880dd380a1e / a18d1459a3e4e9f11728c2bcebffd64f1f016f65129345c359794eee2a39ce4b / 91be5de5607de03a678b34f597100991ce81051f150b7c5cfdd7dfb508613454`；不得重跑、覆盖或补写。

## 训练

统一入口：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set training.device=cuda:0
```

### T2 实现 smoke

Smoke 不是科研结果：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=32 \
  --set data.max_val_windows=32 \
  --set training.epochs=2 \
  --set training.batch_size=8 \
  --set training.seed=20260811 \
  --set training.device=cuda:0 \
  --set outputs.run_root=/tmp/tho_restart_t2_smoke
```

T2、T3、T4 均已分别通过一次性 CPU 生命周期 smoke；数值不构成科研结果。

### Batch 128 GPU 验收（已通过）

T2 native 路径：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.batch_size=128 \
  --set training.seed=20260811 \
  --set training.device=cuda:0 \
  --set outputs.run_root=/tmp/tho_restart_t2_batch128_acceptance
```

T3 concat-deep 使用不同 decoder，需单独验收：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2_t3_concat.yaml \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.batch_size=128 \
  --set training.seed=20260811 \
  --set training.device=cuda:0 \
  --set outputs.run_root=/tmp/tho_restart_t3_batch128_acceptance
```

T4 与 T2 共用内存更低的 native 路径，T2 通过后不重复 batch 128 验收。

验收目录分别为 `/tmp/tho_restart_t2_batch128_acceptance/20260805_155009_160323` 和 `/tmp/tho_restart_t3_batch128_acceptance/20260805_155030_379024`。

### T2 三 seed 正式 validation

先只运行第一顺位 T2：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_tho.py \
    --config configs/tho_research_v2.yaml \
    --set training.epochs=50 \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/tho_restart_t2_g3c_wide_native/seed_${seed}" \
    || exit 1
done
```

T2、T3、T4 validation 均已完成。当前保留 T2/T4，并与 B0、F0、IEWT 一起进入阶段性 research-test；B0/M1/T1/T3 和 loss 消融由各自 run manifest 与 Git 历史追溯。

## CRD-v1.1 S0/S1/S2A/S2B-R

八个 S0/S1、三个 S2A 与四个 S2B-R 配置：

```text
configs/crd_v1/crd_001_b0_retrain.yaml
configs/crd_v1/crd_002_t4_retrain.yaml
configs/crd_v1/crd_101_b0_coarse.yaml
configs/crd_v1/crd_102_b0_local_mamba.yaml
configs/crd_v1/crd_103_direct_local_mamba.yaml
configs/crd_v1/crd_104_direct_hier_mamba.yaml
configs/crd_v1/crd_105_direct_coarse.yaml
configs/crd_v1/crd_106_b0_hier_mamba.yaml
configs/crd_v1/crd_202_base_legacy_energy.yaml
configs/crd_v1/crd_203_base_analytic_am.yaml
configs/crd_v1/crd_204_base_morphology.yaml
configs/crd_v1/crd_205_base_em_static.yaml
configs/crd_v1/crd_206_base_am_static.yaml
configs/crd_v1/crd_207_base_cap_em.yaml
configs/crd_v1/crd_208_base_cap_am.yaml
```

先确认固定原生依赖：

```bash
./.venv/bin/python -c \
  "from resp_train.crd.config import check_crd_dependencies; p=check_crd_dependencies(); print(p); raise SystemExit(bool(p))"
```

在目标 GPU 上执行 actual CRD BiMamba2 的 bf16 fast-path forward/backward 短验收：

```bash
./.venv/bin/python scripts/check_crd_mamba.py --device cuda:0
```

选定 variant 的完整 synthetic 单 microbatch 链路验收：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_103_direct_local_mamba.yaml \
  --device cuda:0 \
  --batch-size 1
```

### CRD CPU 生命周期 smoke

CPU smoke 只建议用于不含 Mamba 的 001/002/101/105；下面以 101 为例。它不形成科研结果：

```bash
./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_101_b0_coarse.yaml \
  --set protocol.run_role=smoke \
  --set data.max_train_windows=4 \
  --set data.max_val_windows=2 \
  --set training.epochs=1 \
  --set training.batch_size=2 \
  --set training.gradient_accumulation_steps=1 \
  --set training.device=cpu \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_101_cpu_smoke
```

### CRD GPU physical-batch-128 acceptance

每种实际要进入正式队列的结构都应单独验收。下面的一次 run 包含 1 个 physical-batch-128 microbatch、一次 optimizer update 和 32 个 validation windows：

```bash
./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_103_direct_local_mamba.yaml \
  --set protocol.run_role=acceptance \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_103_batch32_acceptance
```

如发生 OOM，停止，不在正式命令中临时改变 batch/accumulation；按协议先复验并统一修订为 runner-up `64×2`。Acceptance 目录与数值不进入正式比较。

正式队列前的等效 batch-scaling 工程 benchmark：

```bash
./.venv/bin/python scripts/benchmark_crd_batch_scaling.py \
  --config configs/crd_v1/crd_103_direct_local_mamba.yaml \
  --device cuda:0 \
  --schemes 32x4 64x2 128x1 \
  --repeats 3 \
  --output /tmp/crd_batch_scaling_20260808.json
```

首轮 cold 不进入吞吐中位数；脚本记录 PyTorch allocated/reserved 峰值并按协议的 10% 吞吐增益、80% 显存安全线给出工程 recommendation。它不运行 validation/test，也不形成模型效果证据。

2026-08-08 的固定 CRD_103 benchmark 中，`32×4 / 64×2 / 128×1` 稳态吞吐分别为 `215.899 / 249.812 / 281.438 samples/s`，peak reserved 分别为 `3114 / 5448 / 10868 MiB`。`128×1` 相对基线提升 30.36%，占 16 GiB 设备显存的 68.17%，因此当前 formal/acceptance 配置已统一冻结为 physical batch 128、accumulation 1；effective batch 仍为 128。

### CRD 正式三 seed

必须按协议决策链逐 variant 推进，而不是一次性启动 001–104。以下模板只在对应上一阶段已经通过且工作树/commit 已冻结时使用：

```bash
variant=crd_001_b0_retrain
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
    || exit 1
done
```

不要覆盖 `epochs/batch/accumulation` 或任何 `max_*_windows`；formal loader 会拒绝。S0 的 001/002 与 S1 的 101 已完成。101 已触发 signed-PCC 停止线，原 `101 → 102 → 103 → 可选104` 队列关闭；当前只允许按协议第 15–16 节执行 final-checkpoint 归因复评与 105 诊断。

CRD validation checkpoint 复评：

```bash
./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/<variant>/seed_<seed>/<timestamp>/checkpoint_best_local_rr.pt \
  --set training.device=cuda:0 \
  --metrics-output /tmp/crd_validation_metrics.csv
```

该入口只读 validation，不接受 `--split test`。

### CRD_001/101 paired final-checkpoint 诊断复评

以下复评不改变原 Local-RR checkpoint 结论，输出进入独立诊断目录，不写回六个正式 run。先复评 CRD_001：

```bash
./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_001_b0_retrain/seed_20260811/20260808_035754_258179/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_001_b0_retrain/seed_20260811/validation_metrics.csv

./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_001_b0_retrain/seed_20260812/20260808_042713_710965/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_001_b0_retrain/seed_20260812/validation_metrics.csv

./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_001_b0_retrain/seed_20260813/20260808_045849_155730/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_001_b0_retrain/seed_20260813/validation_metrics.csv
```

再复评 CRD_101：

```bash
./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_101_b0_coarse/seed_20260811/20260808_154855_237029/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_101_b0_coarse/seed_20260811/validation_metrics.csv

./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_101_b0_coarse/seed_20260812/20260808_155229_535580/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_101_b0_coarse/seed_20260812/validation_metrics.csv

./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_101_b0_coarse/seed_20260813/20260808_161853_510867/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_101_b0_coarse/seed_20260813/validation_metrics.csv
```

若目标文件已经存在，先停止并核对，不得覆盖。D0 结果只判断 selector 是否可能贡献 PCC 下降，不能事后重选 CRD_101 或开放 CRD_102。

D0 已完成：fixed-final 下 CRD_001/101 的 Local RR seed mean 为 `0.661868/0.640469`，signed PCC seed mean 为 `0.841586/0.788269`；PCC 仍下降 `0.053317`，三个配对 seed 全部下降。selector 不是原 PCC 退化的主要解释，CRD_102 继续暂停，下一步进入 CRD_105。

### CRD_105 Direct-Coarse 诊断

先在目标 GPU 验证完整 synthetic 链路，再运行独立 physical-batch-128 acceptance：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_105_direct_coarse.yaml \
  --device cuda:0 \
  --batch-size 1

./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_105_direct_coarse.yaml \
  --set protocol.run_role=acceptance \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_105_batch128_acceptance
```

两项通过、代码提交且工作树干净后，才运行三 formal seeds：

上述两项已在 commit `be21ba0` 下通过；acceptance 路径为 `/tmp/crd_105_batch128_acceptance/20260808_204245_249478`。该单 epoch 数值只属于工程验收，不作效果解释。当前允许执行：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_crd.py \
    --config configs/crd_v1/crd_105_direct_coarse.yaml \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="runs/crd_v1/crd_105_direct_coarse/seed_${seed}" \
    || exit 1
done
```

105 通过 CRD_001 coarse gate 后才允许按 `105 → 103 → 可选104` 推进；105 失败则停止。原 CRD_102 仍处于暂停状态。

CRD_105 三 formal seeds 已在 commit `2bee3e5` 下完成：Local RR seed mean `0.581234`、signed PCC seed mean `0.846330`，相对 CRD_001 分别改善 8.1006% 和增加 0.006043，明确通过 gate。当前跳过 CRD_102 并开放 CRD_103；CRD_104 仍等待 `103 vs 105` 结果。未来 CRD_103/104 formal run 与 105 一样记录 `crd-v1.1-s1d-20260808`，不再使用原队列的 protocol 标识。

CRD_103 三 formal seeds 已在 commit `ed9d68e` 下完成：相对 CRD_105，Local RR 恶化 0.6905%、配对方向为 0/3、trajectory MAE 恶化 5.2364%，违反三项必要 gate；signed PCC 虽增加 0.006461，不能单独推翻停止规则。S1D 当前保留 CRD_105、CRD_103 不保留，CRD_102/104 不进入模型选择队列；其后仅可按下面的 S1E 身份探索性补跑。

### CRD S1E 探索性补全

结果后协议第 17 节允许补跑 CRD_102/104，但不重新选择本轮模型。两者使用独立 `crd-v1.1-s1e-20260809` protocol；先分别执行 synthetic 与 acceptance：

```bash
for variant in crd_102_b0_local_mamba crd_104_direct_hier_mamba; do
  ./.venv/bin/python scripts/check_crd_variant.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --device cuda:0 \
    --batch-size 1 \
    || exit 1

  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set protocol.run_role=acceptance \
    --set data.max_train_windows=128 \
    --set data.max_val_windows=32 \
    --set training.epochs=1 \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="/tmp/${variant}_batch128_acceptance" \
    || exit 1
done
```

两个 variant 均通过后，执行完整三 seed：

两项 synthetic/acceptance 已在 commit `d60b050`、`git_dirty=false` 下通过；acceptance 路径分别为 `/tmp/crd_102_b0_local_mamba_batch128_acceptance/20260809_021443_770877` 与 `/tmp/crd_104_direct_hier_mamba_batch128_acceptance/20260809_021601_879119`。其单 epoch 指标不作效果解释。工作树重新确认干净后即可执行：

```bash
for variant in crd_102_b0_local_mamba crd_104_direct_hier_mamba; do
  for seed in 20260811 20260812 20260813; do
    ./.venv/bin/python scripts/train_crd.py \
      --config "configs/crd_v1/${variant}.yaml" \
      --set training.seed="${seed}" \
      --set training.device=cuda:0 \
      --set training.show_progress=false \
      --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
      || exit 1
  done
done
```

S1E 运行当时不得覆盖 epochs/batch/accumulation、任何 `max_*_windows` 或读取 research-test。102/104 的原保留条件只作描述性参照；S1E 结果不会自动推翻当前保留的 CRD_105。后续 test 授权只来自下面另立的 S1C。

六个 S1E run 已在 commit `f8fa658` 下完成并通过审计。102 相对 101 除 trajectory 恶化 3.4048% 外，在 RR、PCC、global envelope 与 IBI 上均大幅改善；104 相对 103 描述性满足原四项条件。相对冻结的 105，102 的 Local RR 改善 4.3138%、3/3 paired 改善、PCC 增加 0.018342且 trajectory 仅恶化 0.2882%；104 的 Local RR/PCC 也改善，但 trajectory 恶化 3.0375%。这些结果只用于下一版确认协议设计；S1E 本身不重选模型、不授权 research-test。

### CRD candidate lock

CRD_102/104/105 的九个候选 checkpoint 与 CRD_001 三个 reference 已冻结在：

```text
docs/experiments/crd_v1_candidate_lock_20260809.json
```

验证 checkpoint、resolved config、run manifest 和 validation summary 的大小/SHA-256：

```bash
./.venv/bin/python scripts/verify_crd_candidate_lock.py
```

选择规则固定为相对 CRD_105 的四项资格门槛后进行五项 primary Pareto；多于一个非支配候选时保留 Pareto set，不强制单赢家。

### CRD S1C research-test 确认

独立测试集已为上述12个frozen checkpoints完成评价，且不参与本阶段checkpoint或candidate选择。以下冻结命令仅保留作provenance，12份access receipt已齐备，**不得再次运行**：

```bash
./.venv/bin/python scripts/verify_crd_candidate_lock.py

set -o pipefail
./.venv/bin/python scripts/verify_crd_candidate_lock.py --print-checkpoints | \
while IFS= read -r checkpoint; do
  ./.venv/bin/python scripts/eval_crd_s1c.py \
    --checkpoint "${checkpoint}" \
    --device cuda:0 \
    --confirm-research-test \
    || exit 1
done
```

冻结规则汇总器也已运行，以下命令只作 provenance：

```bash
./.venv/bin/python scripts/summarize_crd_s1c.py
```

运行使用干净 commit `3b28001`。12 份 metrics 各 2310 行，总计 27720 行，manifest/lock/split/finite 检查全部通过。冻结结果为：102 通过全部 eligibility，104 未通过 Local-RR 与 trajectory 门槛；eligible set 为 `{102,105}`，102 在五项 primary 上严格支配 105，因而是唯一 non-dominated candidate。输出固定在 `runs/crd_v1/crd_s1c_research_confirmation/`。CRD_001 reference 的 Whole RR、IBI-MedAE 与 coherence 仍优于 102，不能把选择结果写成所有轴全面占优。S1C 队列现已关闭；下一阶段仍需另立 research-test-informed 协议。

### CRD S1F：CRD_106 B0-Hier-Mamba

S1F 只补齐 `CRD_102 + CRD_104 同构 global/FiLM` 这一格，不开放其他 S1/S2 模型，也不读取 research-test。提交后先执行 synthetic 与独立 physical-batch-128 acceptance：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_106_b0_hier_mamba.yaml \
  --device cuda:0 \
  --batch-size 1

./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_106_b0_hier_mamba.yaml \
  --set protocol.run_role=acceptance \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_106_b0_hier_mamba_batch128_acceptance
```

两项已在干净 commit `8fa56f8` 下通过并完成产物审计；acceptance 路径为 `/tmp/crd_106_b0_hier_mamba_batch128_acceptance/20260809_142221_810744`。现允许三个 formal seeds：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_crd.py \
    --config configs/crd_v1/crd_106_b0_hier_mamba.yaml \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="runs/crd_v1/crd_106_b0_hier_mamba/seed_${seed}" \
    || exit 1
done
```

三个 formal runs 已在干净 commit `80e6350` 下完成并通过完整性审计。相对冻结 CRD_102，106 的 Local RR 改善 `2.2123%`、3/3 paired seeds 改善，PCC 下降 `0.003178` 仍在护栏内；但 trajectory 恶化 `5.0351%`，超过 `1.5%`。因此 106 不满足四项全通过条件，不保留；S1F 关闭并固定未来 `S2 BASE=102`，不追加 variant、不读取 research-test。Whole RR 改善 `8.5536%` 只作任务交换背景，不能覆盖停止规则。

### CRD S2 表征分支

S2 BASE 固定为 candidate lock 中的 CRD_102，别名 `CRD_201_BASE`，不重训。当前只激活以下三个 S2A variant 的实现与工程验收：

```text
CRD_202_BASE_LEGACY_ENERGY
CRD_203_BASE_ANALYTIC_AM
CRD_204_BASE_MORPHOLOGY
```

三者都在 BASE PatchTokenFrontend 后、local Mamba 前以 zero-init static residual 加入；不含 Direct/global/gate/auxiliary。当前冻结的配置与 trainable parameter 数为：

| Variant | 配置 | Trainable params |
|---|---|---:|
| `CRD_202_BASE_LEGACY_ENERGY` | `configs/crd_v1/crd_202_base_legacy_energy.yaml` | 1,071,449 |
| `CRD_203_BASE_ANALYTIC_AM` | `configs/crd_v1/crd_203_base_analytic_am.yaml` | 1,197,785 |
| `CRD_204_BASE_MORPHOLOGY` | `configs/crd_v1/crd_204_base_morphology.yaml` | 1,106,857 |

先逐一运行 synthetic forward/backward，再运行彼此独立的 physical-batch-128 acceptance：

```bash
for variant in \
  crd_202_base_legacy_energy \
  crd_203_base_analytic_am \
  crd_204_base_morphology; do
  ./.venv/bin/python scripts/check_crd_variant.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --device cuda:0 \
    --batch-size 1 \
    || exit 1

  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set protocol.run_role=acceptance \
    --set data.max_train_windows=128 \
    --set data.max_val_windows=32 \
    --set training.epochs=1 \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="/tmp/${variant}_batch128_acceptance" \
    || exit 1
done
```

每个 synthetic 必须报告 finite output/gradient；每个 acceptance 必须完成一次 optimizer update、validation 和完整 checkpoint lifecycle。每个 run 的 `runtime_summary.json` 会固化 peak allocated/reserved 及 reserved/总显存比例，尤其检查 204。三项结果返回并审计、代码 commit 固定前，不启动 formal seeds。Energy 和 morphology 使用不同的 primary/guardrail，S2B 只在两类都 eligible 时条件开放；S2 全程禁止 research-test。

首轮结果中，202/203 已通过，peak reserved fraction 为 `62.18%/74.25%`；204 synthetic 通过但 batch-128 训练 OOM。204 现已加入不改变数学定义的 per-chunk activation checkpoint，保持 physical batch 128。只重跑 204：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_204_base_morphology.yaml \
  --device cuda:0 \
  --batch-size 1

./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_204_base_morphology.yaml \
  --set protocol.run_role=acceptance \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_204_base_morphology_checkpointed_batch128_acceptance
```

验收要求仍是完整 lifecycle、output/gradient/checkpoint finite、所有 eligible primary metrics finite，且 `runtime_summary.json` 的 `peak_reserved_fraction≤0.80`；按 eligibility flag 显式未定义的指标不伪造有限值。若仍失败，停止，不临时减小 batch 或 chunk。

204 checkpointed 重验已在干净 commit `a149913` 下通过，peak allocated/reserved 为 `8,636.81/10,406 MiB`、reserved fraction `65.30%`。三个 S2A variant 至此均完成工程验收；正式队列在本记录提交且工作树干净后开放。依次运行九个 formal runs：

```bash
for variant in \
  crd_202_base_legacy_energy \
  crd_203_base_analytic_am \
  crd_204_base_morphology; do
  for seed in 20260811 20260812 20260813; do
    ./.venv/bin/python scripts/train_crd.py \
      --config "configs/crd_v1/${variant}.yaml" \
      --set training.seed="${seed}" \
      --set training.device=cuda:0 \
      --set training.show_progress=false \
      --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
      || exit 1
  done
done
```

不得覆盖 epochs、batch、accumulation 或任何 `max_*_windows`，不得读取 research-test。九个 runs 完成后先做完整性与 checkpoint 集合审计，再按附件第 21.4–21.5 节分别判断 energy representation 与 morphology eligibility；在结果审计前不实现 S2B。

九个 formal runs 已在统一干净 commit `41ed41d` 下完成并通过完整性审计。Selected epochs 为 202=`10/18/12`、203=`22/11/12`、204=`8/11/12`。冻结门槛得到 `X=none` 且 M 不 eligible，因此 S2B/S3 关闭并保留 CRD_102；prototype 描述不能推翻该结果。为满足第 21.5 节的完整报告要求，只读三个 204 validation-selected checkpoints：

```bash
for checkpoint in \
  runs/crd_v1/crd_204_base_morphology/seed_20260811/20260809_224710_306894/checkpoint_best_local_rr.pt \
  runs/crd_v1/crd_204_base_morphology/seed_20260812/20260810_011253_712445/checkpoint_best_local_rr.pt \
  runs/crd_v1/crd_204_base_morphology/seed_20260813/20260810_033710_435683/checkpoint_best_local_rr.pt; do
  ./.venv/bin/python scripts/eval_crd_morphology_prototypes.py \
    --checkpoint "${checkpoint}" \
    --device cuda:0 \
    || exit 1
done
```

三项完成后，在干净工作树运行冻结汇总：

```bash
./.venv/bin/python scripts/summarize_crd_s2a.py
```

汇总入口会重新审计 BASE lock、九个 formal runs、逐 sample identity/metrics summary、checkpoint finite/hash、prototype manifest/hash，并输出 seed mean±sample SD、paired window/samp 描述、prototype usage/entropy 与最终 decision；已有输出拒绝覆盖。

上述三项 prototype 描述与冻结汇总已在干净 commit `3c3598c` 下完成。Prototype 未发生全局单类坍缩：三个 seed 的 global dominant hard fraction 为 `37.26%/47.50%/34.06%`，soft-usage entropy 为 `0.9021/0.8809/0.9540`；但这不能覆盖 M 已失败的 PCC、Local-RR 与 coverage 门槛。最终产物：

```text
runs/crd_v1/crd_s2a_validation_summary/s2a_decision.json
runs/crd_v1/crd_s2a_validation_summary/s2a_seed_summary.csv
runs/crd_v1/crd_s2a_validation_summary/s2a_variant_summary.csv
runs/crd_v1/crd_s2a_validation_summary/s2a_paired_descriptives.csv
runs/crd_v1/crd_s2a_validation_summary/s2a_prototype_seed_summary.csv
runs/crd_v1/crd_s2a_validation_summary/s2a_summary_manifest.json
```

Decision 固定为 `X=none / M ineligible / S2B=false / S3=false / retain CRD_102`。S2A 已关闭，以上 prototype/summary 命令只保留 provenance，不得重复执行或用 research-test 重选。

### CRD S2B-R 双因素交互补救

研究者在获知 S2A 结果后明确要求探索单因素失败、多因素非线性补偿，因此新增的 S2B-R 是 result-informed exploratory stage，不改写上面的 S2A decision，也不是原 S2B 条件自然触发。四个模型为：

| Variant | 结构 | Params | Capacity match |
|---|---|---:|---|
| `crd_205_base_em_static` | BASE + E + M | 1,109,561 | — |
| `crd_206_base_am_static` | BASE + A + M | 1,235,897 | — |
| `crd_207_base_cap_em` | BASE + capacity control | 1,109,257 | `N=2,H=104`，差 `-304` |
| `crd_208_base_cap_am` | BASE + capacity control | 1,235,785 | `N=4,H=216`，差 `-112` |

以下四项 synthetic 与独立 physical-batch-128 acceptance 命令现只保留 provenance：

```bash
for variant in \
  crd_205_base_em_static \
  crd_206_base_am_static \
  crd_207_base_cap_em \
  crd_208_base_cap_am; do
  ./.venv/bin/python scripts/check_crd_variant.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --device cuda:0 \
    --batch-size 1 \
    || exit 1

  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set protocol.run_role=acceptance \
    --set data.max_train_windows=128 \
    --set data.max_val_windows=32 \
    --set training.epochs=1 \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="/tmp/${variant}_batch128_acceptance" \
    || exit 1
done
```

每项要求完整 lifecycle、eligible primary finite、joint prediction nondegenerate 且 `peak_reserved_fraction≤0.80`。若某项 OOM 或越线，停止，不临时减 batch/结构。S3 gate 仍未实现，S2B-R 全程禁止 research-test。

四项已在统一干净 commit `0e541f8` 下通过；peak reserved fraction 为 `67.37%/67.15%/65.41%/71.46%`。以下 12 个 formal runs 命令现只保留 provenance，不得重复运行：

```bash
for variant in \
  crd_205_base_em_static \
  crd_206_base_am_static \
  crd_207_base_cap_em \
  crd_208_base_cap_am; do
  for seed in 20260811 20260812 20260813; do
    ./.venv/bin/python scripts/train_crd.py \
      --config "configs/crd_v1/${variant}.yaml" \
      --set training.seed="${seed}" \
      --set training.device=cuda:0 \
      --set training.show_progress=false \
      --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
      || exit 1
  done
done
```

12 项已在统一干净 training commit `c2bcfb0` 下完成；selected epochs 为 205=`8/11/12`、206=`19/11/9`、207=`13/11/12`、208=`10/18/15`。冻结汇总命令如下，现只保留 provenance：

```bash
./.venv/bin/python scripts/summarize_crd_s2br.py
```

产物位于：

```text
runs/crd_v1/crd_s2br_validation_summary/s2br_decision.json
runs/crd_v1/crd_s2br_validation_summary/s2br_seed_summary.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_variant_summary.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_paired_descriptives.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_factorial_seed_summary.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_factorial_paired_descriptives.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_summary_manifest.json
```

两个组合的 BASE/control/best-constituent Local RR 门槛均失败：205 的相对改善为 `-1.0387%/-0.9260%/-2.5769%`，206 为 `-1.0295%/-1.4515%/-1.0295%`。Factorial descriptives 显示非线性补偿方向存在，但绝对组合效果不足；decision 固定为 `passing_combinations=[] / retain CRD_102 / S3=false`。S2B-R 已关闭，以上训练与汇总入口不得重复执行或用于读取 research-test 后重选。

### CRD_102 validation 误差分层与失败模式

该诊断只读取 candidate-lock 中 CRD_102 三个冻结 run 的既有 validation `metrics.csv`，不需要 GPU、不重新 eval、不访问 research-test。以下命令现只保留 provenance，不得重复运行：

```bash
./.venv/bin/python -m pytest tests/test_crd_failure_diagnostics.py
./.venv/bin/python scripts/summarize_crd102_failures.py
```

第一层已从干净 commit `81fab55` 输出到 `runs/crd_v1/crd_102_failure_diagnostic/`；worst-decile/persistent failure、固定分层、association 和签名口径见协议附件第 23 节。

第一层满足 metadata follow-up 条件后，以下命令已从新的干净 commit 运行，现只保留 provenance：

```bash
./.venv/bin/python -m pytest tests/test_crd_failure_metadata.py
./.venv/bin/python scripts/summarize_crd102_failure_metadata.py
```

第二层已从干净 commit `7a1b29b` 输出到 `runs/crd_v1/crd_102_failure_metadata_diagnostic/`，只连接冻结 validation consensus 与 dataset index；没有读取波形、重新 eval 或访问 research-test。核心结果为：high modulation 占 Local-RR/multimetric persistent failures 的 `70.61%/91.76%`，三 seed Local-RR 难例高度一致，且 `84.73%/80.40%` 的相应失败窗口位于多窗口连续片段。两层均已关闭，所有入口不得用于重选 checkpoint 或自动触发新模型阶段。

### CRD_102 matched observability

该入口按附件第 24 节冻结规则选择 high-modulation failure/control pairs，只读取匹配窗口的 rawish/fixed-band/target 波形并计算 CPU proxy metrics，不执行模型 inference。以下命令现只保留 provenance：

```bash
./.venv/bin/python -m pytest tests/test_crd_matched_observability.py
./.venv/bin/python scripts/summarize_crd102_matched_observability.py
```

唯一结果从干净 commit `c951325` 输出到 `runs/crd_v1/crd_102_matched_observability_diagnostic/`。21 个 exact-state primary pairs 中 rawish proxy 的 Local-RR/PCC case-worse 为 `16/21、13/21`，fixed-band 为 `17/21、16/21`，decision 为 `mixed_observability_and_model_tracking`；28 个 same-samp sensitivity pairs 也不支持纯输入受限结论。该阶段已关闭，入口不得重复运行，结果不构成因果证据或新候选选择。

### CRD_102 C0 decoder round-trip

C0 只读取完整 validation target，执行 `Pi(target) → ::10 → Fourier 1800→18000 → Pi`；不运行模型、不读取 checkpoint tensor、不训练，也不访问 research-test。先运行定向测试：

```bash
./.venv/bin/python -m pytest tests/test_crd_decoder_roundtrip.py
```

实现与协议提交后，已在干净 commit `5de0c0f` 执行一次：

```bash
./.venv/bin/python scripts/audit_crd102_decoder_roundtrip.py
```

固定输出为 `runs/crd_v1/crd_102_decoder_roundtrip_audit/`；全局最大绝对误差/RMSE 为 `5.538454e-7 / 8.217932e-8`，五项 primary 全部通过冻结门槛，decision 为 `roundtrip_negligible=true`。C0 已关闭，以上命令只保留 provenance，不得重复运行。当前只开放 C1 TCN 的实现与工程验收；formal 三 seed 和 C2 仍关闭。

### CRD C1 parameter-matched full-context TCN

唯一候选 `crd_c101_b0_local_tcn` 已完成实现与 CPU 定向验收。实现提交后，先在目标 GPU 执行 synthetic：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_c101_b0_local_tcn.yaml \
  --device cuda:0
```

通过后运行独立 physical-batch-128 acceptance：

```bash
./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_c101_b0_local_tcn.yaml \
  --set protocol.run_role=acceptance \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set outputs.run_root=/tmp/crd_c101_b0_local_tcn_batch128_acceptance
```

Acceptance 已在干净 commit `529de74` 下完成：恰有一次 optimizer update、两个 finite checkpoint、32 条 primary-finite validation metrics、无 prediction degeneracy，peak reserved fraction 为 `66.9161%`。上述 synthetic/acceptance 命令现只保留 provenance，不得覆盖重跑。

工程结果登记后，三个 formal seeds 已由研究者在统一干净 commit `930212a` 完成；以下命令只保留 provenance：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_crd.py \
    --config configs/crd_v1/crd_c101_b0_local_tcn.yaml \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/crd_v1/crd_c101_b0_local_tcn/seed_${seed}" \
    || exit 1
done
```

不得覆盖 epochs、batch、accumulation、数据上限、loss、metrics 或 selector。当前只允许先运行定向测试，再从新干净 commit 生成一次冻结 summary：

```bash
./.venv/bin/python -m pytest tests/test_crd_c1_selection.py
./.venv/bin/python scripts/summarize_crd_c1.py
```

固定输出为 `runs/crd_v1/crd_c1_validation_summary/`，目录存在时拒绝覆盖。唯一 summary 已从干净 commit `f0ac01b` 生成，decision 为 `mamba_retained_control_failure / retain CRD_102`：Local RR 改善 `3.8241%`、paired `3/3`、PCC drop `0.003797` 均通过，但 trajectory 恶化 `2.5468%` 超过 `1.5%`。C1 已关闭，以上 summary 命令只保留 provenance；当前只开放 C2 两个 decoder controls 的实现与工程验收。

### CRD C2 decoder capacity/placement controls

两个候选为 `crd_c201_decoder_10hz_cap` 与 `crd_c202_decoder_100hz`。实现提交后分别执行 CUDA synthetic：

```bash
for variant in crd_c201_decoder_10hz_cap crd_c202_decoder_100hz; do
  ./.venv/bin/python scripts/check_crd_variant.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --device cuda:0 \
    || exit 1
done
```

两项均通过后，分别运行独立 physical-batch-128 acceptance：

```bash
for variant in crd_c201_decoder_10hz_cap crd_c202_decoder_100hz; do
  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set protocol.run_role=acceptance \
    --set training.epochs=1 \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set data.max_train_windows=128 \
    --set data.max_val_windows=32 \
    --set outputs.run_root="/tmp/${variant}_batch128_acceptance" \
    || exit 1
done
```

两项 acceptance 已在统一干净 commit `0e2d058` 下通过；C201/C202 peak reserved fraction 为 `62.3479%/57.6794%`，checkpoint/optimizer 与 32 条 primary metrics 全 finite、无 prediction degeneracy。上述 synthetic/acceptance 命令现只保留 provenance，不得覆盖重跑。

工程结果登记后，六个 formal runs 已由研究者在统一干净 commit `4ec7371` 完成；以下命令只保留 provenance：

```bash
for variant in crd_c201_decoder_10hz_cap crd_c202_decoder_100hz; do
  for seed in 20260811 20260812 20260813; do
    ./.venv/bin/python scripts/train_crd.py \
      --config "configs/crd_v1/${variant}.yaml" \
      --set training.seed="${seed}" \
      --set training.device=cuda:0 \
      --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
      || exit 1
  done
done
```

不得覆盖 epochs、batch、accumulation、数据上限、loss、metrics 或 selector。冻结 summary 前先运行 residual diagnostic 的定向测试，然后按固定 checkpoint 顺序执行三项 validation inference：

```bash
./.venv/bin/python -m pytest tests/test_crd_decoder_diagnostics.py

for checkpoint in \
  runs/crd_v1/crd_c202_decoder_100hz/seed_20260811/20260811_163019_812824/checkpoint_best_local_rr.pt \
  runs/crd_v1/crd_c202_decoder_100hz/seed_20260812/20260811_173925_328791/checkpoint_best_local_rr.pt \
  runs/crd_v1/crd_c202_decoder_100hz/seed_20260813/20260811_184846_178964/checkpoint_best_local_rr.pt; do
  ./.venv/bin/python scripts/eval_crd_c202_residual_spectrum.py \
    --checkpoint "${checkpoint}" \
    --device cuda:0 \
    || exit 1
done
```

固定输出为 `runs/crd_v1/crd_c2_decoder_diagnostics/crd_c202_decoder_100hz/seed_<seed>/`。三项均已从干净 commit `3ef5cf0` 完成，带外能量比例 seed mean 为 `3.1199%/18.7155%/10.6439%`；以上 diagnostic 命令只保留 provenance，不得重复运行。

冻结门槛测试与 C2 summary 使用以下命令完成：

```bash
./.venv/bin/python -m pytest tests/test_crd_c2_selection.py
./.venv/bin/python scripts/summarize_crd_c2.py
```

固定输出为 `runs/crd_v1/crd_c2_validation_summary/`，目录存在时拒绝覆盖。唯一 summary 已从干净 commit `6e893a3` 生成，decision 为 `decoder_capacity_supported_100hz_placement_not_supported`：C201/C202 相对 CRD_102 均通过 basic gate，但 C202 相对 C201 的 Local RR 为 `-0.1141%`，未达到 `+0.25%` placement gate；最终选择 `crd_c201_decoder_10hz_cap` 并保留 CRD_102 Mamba backbone。以上测试与 summary 命令只保留 provenance，不得重复运行；research-test、TCN+decoder 与其他 decoder 继续关闭。

### CRD-TF v1 synthetic calibration 与 fixed cache

P1 已从干净 commit `6d169760` 完成；规范见 `docs/experiments/crd_tf_v1_protocol_20260812.md`。以下命令只保留 provenance，不得重复运行或覆盖：

```bash
./.venv/bin/python scripts/calibrate_crd_tf_v1.py
```

固定 calibration 为 `runs/crd_tf_v1/calibration/7e29795edc13fe8dc2e12ada8d619c22d0ae19fa8d13feb729d2f9b261fd5535/calibration.json`，四项全部通过，S 参数为 `2.0 / 2`。随后使用的 cache 命令为：

```bash
./.venv/bin/python scripts/build_crd_tf_v1_cache.py \
  --calibration runs/crd_tf_v1/calibration/7e29795edc13fe8dc2e12ada8d619c22d0ae19fa8d13feb729d2f9b261fd5535/calibration.json
```

Cache 已固定写入 `runs/crd_tf_v1/cache/bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0/`，只包含 10141 train + 2675 validation 的 M/W/S 和 L input spectrum，共 3.6398 GiB；逐文件 hash/shape/dtype/finite 审计通过，不读取 target、不生成 test cache。P2 已完成；cache 不得重建或覆盖。

以下 partial-cache smoke 也随 P1 关闭，不再运行；命令只说明历史调试接口，任何 partial cache 都不能进入训练：

```bash
./.venv/bin/python scripts/build_crd_tf_v1_cache.py \
  --calibration <passed-calibration.json> \
  --max-windows-per-split 1 \
  --allow-dirty-smoke
```

### CRD-TF v1 P3 CUDA 与 batch-128 acceptance

P3 必须从同一个干净 commit 顺序执行。先覆盖全部 15 个 variant 的 CUDA synthetic：

```bash
git status --short
./.venv/bin/python scripts/check_crd_tf_v1_cuda.py --device cuda:0
```

首行必须无输出。Synthetic receipt 会写到 `runs/crd_tf_v1/p3_cuda_synthetic/<commit>_<timestamp>/synthetic_receipt.json`。15 项必须全部报告 output/input/parameter gradient finite，且 active branch final projection gradient 非零。

随后依次运行冻结的最大 single/pair/triple；不得并行，以免污染显存证据：

```bash
for variant in crd_tf102_w crd_tf204_wl crd_tf302_wls; do
  ./.venv/bin/python scripts/run_crd_tf_v1_acceptance.py \
    --variant "$variant" \
    --device cuda:0
done
```

每项必须完成 `128 train / 32 validation / 1 epoch / 1 optimizer update`、best/final 两个 checkpoint、五项 finite primary、prediction nondegenerate，并满足 `peak_reserved_fraction≤0.80`。三个命令各自打印唯一 run 目录。拿到路径后执行冻结审计：

```bash
./.venv/bin/python scripts/audit_crd_tf_v1_p3.py \
  --synthetic-receipt <synthetic_receipt.json> \
  --tf102-run <crd_tf102_w_run_dir> \
  --tf204-run <crd_tf204_wl_run_dir> \
  --tf302-run <crd_tf302_wls_run_dir>
```

若任一 acceptance OOM、非有限、生命周期失败或超过 80% 显存线，立即停止并返回完整错误；不要自行改 batch。P3 单 update 的 validation 数值不作效果解释，P4 formal 仍关闭。

最终 P3 已在 commit `56cabf1` 完成。TF102/TF204/TF302 的 peak reserved fraction 分别为 `63.45% / 64.14% / 64.76%`，统一审计输出为 `runs/crd_tf_v1/p3_acceptance_audit/0b4af9bd0c1c6441460702ad893cc5713f8ce3578cc055e51e86e60ad285234e/p3_acceptance.json`，固定 batch 为 `128×1`。以上 P3 命令现只保留 provenance，不得重复运行；用户已确认的 P4 入口如下。

### CRD-TF v1 P4 完整 formal matrix

用户已确认完整 45-run，但每项独立运行，不使用统一队列入口。每个 arm 有独立 formal 配置；以下示例运行 CTRL1 的第一个 seed：

```bash
./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_tf_v1/crd_tf_ctrl1_formal.yaml \
  --set training.seed=20260811
```

同一 arm 的其余 seed 只替换最后一项：

```bash
--set training.seed=20260812
--set training.seed=20260813
```

15 个独立配置按协议第 19 节顺序为：

```text
crd_tf_ctrl1_formal.yaml
crd_tf101_m_formal.yaml
crd_tf102_w_formal.yaml
crd_tf103_l_formal.yaml
crd_tf104_s_formal.yaml
crd_tf_ctrl2_formal.yaml
crd_tf201_mw_formal.yaml
crd_tf202_ml_formal.yaml
crd_tf203_ms_formal.yaml
crd_tf204_wl_formal.yaml
crd_tf205_ws_formal.yaml
crd_tf206_ls_formal.yaml
crd_tf_ctrl3_formal.yaml
crd_tf301_mls_formal.yaml
crd_tf302_wls_formal.yaml
```

每项完成后把打印的 run 目录返回审计，再手动启动下一项。失败或中断时先返回错误；重跑不会覆盖旧 timestamp 目录。不得根据中间 validation 数值删减剩余 arm；全部 45 项完成前不得执行 P5。

P4 已完成 45/45。当前唯一允许的 P5 命令为：

```bash
./.venv/bin/python scripts/summarize_crd_tf_v1.py
```

输出固定为 `runs/crd_tf_v1/p5_validation_summary/`，存在即拒绝覆盖。脚本会显式排除并登记 WLS seed 20260811 的早期 incomplete 目录；P5 结果冻结前不得执行 research-test 或 P6。

P5 已从 commit `c7b65b1` 完成，候选并集固定为 `crd_tf101_m / crd_tf102_w / crd_tf203_ms`。上述 summary 命令现只保留 provenance，不得重复运行；research-test 继续关闭。

### CRD-TF v1 P6a 工程验收

P6a 固定为 `MWS-ADD / MWS-GATE / CTRL-GATE × 3 seeds`；完整规范见 `docs/experiments/crd_tf_v1_p6a_protocol_20260815.md`。当前只运行工程验收。提交代码并确认工作树干净后，先执行三个新 variant 的 CUDA synthetic：

```bash
git status --short
./.venv/bin/python scripts/check_crd_tf_v1_p6a_cuda.py --device cuda:0
```

首行必须无输出。随后在同一张空闲 GPU 上依次执行两个最大 gated arm 的独立 batch-128 acceptance：

```bash
for variant in crd_tf402_mws_gate crd_tf403_ctrl_gate; do
  ./.venv/bin/python scripts/run_crd_tf_v1_p6a_acceptance.py \
    --variant "${variant}" \
    --device cuda:0 \
    || exit 1
done
```

Synthetic 与两项 acceptance 已从 engineering commit `3802423` 通过，最大 reserved fraction 为 `67.48%`；统一冻结 receipt 为：

```text
runs/crd_tf_v1/p6a_acceptance_audit/9304ae7abd2056c9c28b09702d8fcfc88672a4b53ea412e2922d8d7c7ee821b1/p6a_acceptance.json
SHA-256 = a23c1dd9aca724ecae3d867429911043a0da1843ede61a3784578abbf99040a9
```

以上 engineering 命令与下列审计命令现只保留 provenance，不得重复运行：

```bash
./.venv/bin/python scripts/audit_crd_tf_v1_p6a.py \
  --synthetic-receipt runs/crd_tf_v1/p6a_cuda_synthetic/38024237059a_20260815_220244_034450/synthetic_receipt.json \
  --mws-gate-run runs/crd_tf_v1/p6a_acceptance/crd_tf402_mws_gate/20260815_220344_143254 \
  --ctrl-gate-run runs/crd_tf_v1/p6a_acceptance/crd_tf403_ctrl_gate/20260815_220430_380745
```

9-run formal 已完成。三项配置与通用 `train_crd.py` 命令现只保留 provenance，不得重跑。唯一冻结汇总命令为：

```bash
./.venv/bin/python scripts/summarize_crd_tf_v1_p6a.py
```

该命令已从干净 commit `e658d42` 一次性完成，现不得重复运行。9/9 formal 无 incomplete，全部 early stop；冻结 summary 为 `runs/crd_tf_v1/p6a_validation_summary/p6a_summary.json`，SHA-256=`b970a6ea6d77e6d6858d8b7dbd77ed4c2ed8ff633c7f48eca15aeba227ef6e64`。Decision 为 `no_p6a_candidate_retain_p5_pool`：MWS-ADD 未优于 MS 且未过 PCC 护栏，MWS-GATE 未优于 MWS-ADD。P6a 关闭，research-test/P6b 不自动开放。

### CRD-TF v1 冻结候选 research-test cache

用户已明确授权独立测试集评价，固定矩阵为 C201/M/W/MS × 3 validation-selected checkpoints。当前只构建一次独立的完整 2310-window test input-only M/W/S cache：

```bash
git status --short
./.venv/bin/python scripts/build_crd_tf_v1_research_test_cache.py
```

首行必须无输出。预计耗时与 P1 完整 cache 的 test-window 比例相当；只读 BCG input，不读取 test target array，不运行模型。命令打印唯一 `cache_manifest.json` 路径。返回 manifest 并冻结前，不启动 checkpoint research-test evaluation。

Cache 已从干净 commit `dfd9313` 完成并冻结，manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。上述 builder 命令现只保留 provenance，不得重复运行。

专用评价入口为：

```bash
./.venv/bin/python scripts/eval_crd_tf_v1_research_test.py \
  --checkpoint <冻结矩阵内的 checkpoint_best_local_rr.pt> \
  --device cuda:0 \
  --confirm-research-test
```

入口只接受 C201/M/W/MS × 3 的 12 个精确 path/hash，输出写回 checkpoint 原 run 目录且拒绝覆盖。可由两张相同 GPU 用 shell `for` 分组并行，但每个 checkpoint 仍是独立入口；完整命令见 research-test 协议执行记录。12 项齐备前不得运行汇总或根据中间结果删减矩阵。

12 项已从干净 commit `9f429da` 完成；评价入口现只保留 provenance。唯一汇总命令为：

```bash
./.venv/bin/python scripts/summarize_crd_tf_v1_research_test.py
```

该命令已从干净 commit `a1ce90c` 一次性完成，现不得重复运行。冻结 summary 为 `runs/crd_tf_v1/research_test_summary/research_test_summary.json`，SHA-256=`e9430d3449e1e75cbab1804f1c887803ba8c12dcc4b11582f94090a6a1d7c6c0`。M/W/MS 均 qualified，Pareto 为 W/MS，W 是 Local-RR lead；未构造总分或唯一赢家。

### CRD-TF-W v2 P−1 validation 功能审计

P0 candidate lock 固定为 `docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json`，SHA-256=`6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6`。P−1 只评价锁定的三个 W0 validation checkpoints，不训练、不改 cache/checkpoint、不读取 research-test，也不做 `samp_id` 分析。

实现定向测试：

```bash
./.venv/bin/python -m pytest -q \
  tests/test_crd_tf_w_v2_audit.py \
  tests/test_crd_tf_v1_model.py \
  tests/test_crd_tf_v1_data.py \
  tests/test_crd_experiment.py
```

完整 audit 的原始命令为：

```bash
git status --short
./.venv/bin/python scripts/eval_crd_tf_w_v2_functional_audit.py \
  --candidate-lock docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json \
  --split val \
  --device cuda:0
```

该命令已从干净 commit `d91db6e` 完成 30/30 evaluations，固定 source manifest SHA-256=`249c761799b1f8020a77ed51875985776718d9cf1e9701900ce5dae783492f3f`，现不得重复运行或修改 source audit。冻结 P2 decision=`retain_film_no_p2_training`，不训练 ADD/SCALE。

独立复核发现 source `film_statistics.csv` 的两项相邻帧统计错误地使用 `mean(diff(abs(x)))`，而协议要求 `mean(abs(diff(x)))`；prediction、primary、干预 summary 和 P2 decision 不受影响。correction 的历史命令为：

```bash
git status --short
./.venv/bin/python scripts/eval_crd_tf_w_v2_film_statistics_correction.py \
  --candidate-lock docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json \
  --split val \
  --device cuda:0
```

该命令已从干净 commit `4eb9b3c` 完成，固定输出为 `runs/crd_tf_w_v2/p_minus_1_film_statistics_correction/`；manifest SHA-256=`1c6f7218c280b2a1579b2169a2b4752d7c10416d89de67ed5e9a1cc58d104463`，corrected CSV SHA-256=`2b7a4edef8e9828356a336c3c1ec7880235914207b00d164bf016ce1cd7a5203`。correction 与 source audit 均不得重复运行或改写。P−1 已关闭，P2 已关闭；P1 实现与定向测试现已完成。

### CRD-TF-W v2 P1 implementation、stress 与 formal

P1 implementation lock 为 `docs/experiments/crd_tf_w_v2_p1_implementation_lock_20260817.json`，SHA-256=`fb822ca7f8607e45443e07a94d91150bcc108217fb25c47f0b4504fbdf644f58`。只注册 W1 RESP、W2 CARRIER、W3 full-band 6V；P2/P3 未注册。

定向实现验收：

```bash
./.venv/bin/python -m pytest -q \
  tests/test_crd_tf_w_v2_p1.py \
  tests/test_crd_tf_w_v2_audit.py \
  tests/test_crd_tf_v1_model.py \
  tests/test_crd_tf_v1_data.py \
  tests/test_crd_tf_v1_formal_configs.py \
  tests/test_crd_config.py \
  tests/test_crd_experiment.py \
  tests/test_crd_training.py
```

结果固定登记为 `112 passed`。三个 GPU stress 已从干净 commit `1d1b22e` 完成，receipt SHA-256 依 W1/W2/W3 为 `489b49cd4765f13ddd30908931eed1604862ad7395080b1c3c07b89045c05a6c / c1db21673ec58e25b16c345944e31c9fda8adb2ada64ad429bd4ae18a772fce8 / b1bc8a1dcb14b4213a84cf6a9f20fd701515ecb0af9288e2ffaf19cd309f8414`。以下 stress 命令只保留 provenance，不得重复运行：

```bash
git status --short
./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_tf_w_v2/crd_tfw_v2_w1_resp_12v_film_d6_stress.yaml

./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_tf_w_v2/crd_tfw_v2_w2_carrier_12v_film_d6_stress.yaml

./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_tf_w_v2/crd_tfw_v2_w3_full_6v_film_d6_stress.yaml
```

每臂均完成 5 epochs、physical batch `128×1`、完整 train/validation、400 optimizer updates，并通过 finite、degeneracy、梯度、throughput/latency/validation peak 与显存线。

P1 9/9 formal 也已从同一 commit 完成；W1/W2/W3 selected epoch 分别为 `9/18/29`、`10/9/15`、`9/12/12`。所有 run 完整 80 epochs / 6,400 updates，checkpoint 与 validation 指标核验通过。三组 formal 命令现同样只保留 provenance，不得重复运行：

```bash
for variant in \
  crd_tfw_v2_w1_resp_12v_film_d6 \
  crd_tfw_v2_w2_carrier_12v_film_d6 \
  crd_tfw_v2_w3_full_6v_film_d6; do
  for seed in 20260811 20260812 20260813; do
    ./.venv/bin/python scripts/train_crd.py \
      --config "configs/crd_tf_w_v2/${variant}_formal.yaml" \
      --set training.seed="$seed"
  done
done
```

W1/W2 不通过严格质量门槛；W3 进入质量候选池，但 throughput/peak-allocated 不满足效率门槛。P1 已关闭；P3 进展与当前入口见下一节。

### CRD-TF-W v2 P3 D4 implementation 与 isolation stress

P3 implementation lock 为 `docs/experiments/crd_tf_w_v2_p3_implementation_lock_20260818.json`，SHA-256=`0aa2f520a52a667640ea6550a6d26c48b0f65dd088de6b4616938705a7e40da7`。唯一 variant 为 `crd_tfw_v2_d4_full_12v_film`：保持 full-12V W branch/FiLM 与同 seed 共享 state，只移除 local blocks 4/5；准确参数为 `902,722`。P2/D8/频带或 6V 复合 D4 均未实现。定向结果为 `123 passed`。

以下 GPU stress 命令已从干净 commit `6c4f622` 完成，现只保留 provenance，不得重复运行：

```bash
git status --short
./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_tf_w_v2/crd_tfw_v2_d4_full_12v_film_stress.yaml
```

固定 stress receipt SHA-256=`431604d505f10733082ad5a380a75c2544ff45649f86fc3452301d7d0bb61a8c`，finite/degeneracy/梯度与显存线均通过。D4 3/3 formal 也已完成，selected epoch=`13/26/14`；其历史命令同样不得重复运行。

D4 参数/throughput/peak-allocated 改善 `25.9973% / 24.8925% / 22.5204%`，但 Local RR/trajectory 相对 W0 恶化 `1.8308% / 1.2847%`，超过效率池质量保护线，因此不进入质量或效率池。P3 已关闭；P4 随后将 D4 冻结为描述性、非灾难性效率 trade-off。

### CRD-TF-W v2 P4 validation summary

P4 summarizer 已实现并通过 `127 passed`；随后从干净 commit `cd81f68b65125f3bf604b25cb2fd010056018077` 一次性运行：

```bash
git status --short
./.venv/bin/python scripts/summarize_crd_tf_w_v2.py \
  --candidate-lock docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json
```

固定输出 `runs/crd_tf_w_v2/p4_validation_summary/` 的六文件 schema 已完整冻结，summary SHA-256=`0f5a62448aa8db6b3bc9b07633bedf2e857b6369427792d37872851841cbd70b`，manifest SHA-256=`d9f32d26fa9bf8b98ec739762a30717a881d62a32ccb44099e603fafe94b0e97`。严格质量池/Pareto 为 W3，严格效率池/Pareto 为空，描述性质量—效率 Pareto 为 W3/D4。D4 的显著计算收益与轻度、非灾难性质量损失均被保留，但不会越过硬门槛进入 P5 allowlist。

P4 已关闭，上述命令只保留 provenance，不得重复运行。P5 随后获得明确授权，且只允许 W3 的三个 validation-selected checkpoints；其完成登记见下节。

### CRD-TF-W v2 P5 最小独立测试集评价

P5 获用户授权后从干净 commit `4ef901c17ef6167f2531232e43e0941361563c81` 实现；冻结 checkpoint allowlist SHA-256=`c3fe1320a8342a9580fff2864218c948c451b21c05a863db63efe269f1358b07`，只包含 W3 三个 validation-selected checkpoints（epoch `9/12/12`）。实现阶段定向回归为 `141 passed`，当时尚未运行 GPU 或读取 research-test target。

从干净工作树串行评价三个 seed：

```bash
git status --short
./.venv/bin/python scripts/eval_crd_tf_w_v2_research_test.py \
  --candidate-lock docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json \
  --checkpoint-allowlist docs/experiments/crd_tf_w_v2_p5_checkpoint_allowlist_20260819.json \
  --device cuda:0 \
  --confirm-research-test
```

首行必须无输出。固定顺序为 seed `20260811 → 20260812 → 20260813`；隔离输出位于 `runs/crd_tf_w_v2/research_test/evaluations/crd_tfw_v2_w3_full_6v_film_d6/seed_<seed>/`。中断恢复可增加 `--seed <seed>`；已存在的完整输出只校验后跳过，任何产物均不覆盖。

3/3 evaluation 与一次性 `scripts/summarize_crd_tf_w_v2_research_test.py` 随后从干净 commit `508a936b5c3b997c19c5861b6a6e0874e41dcd43` 完成。固定 summary 位于 `runs/crd_tf_w_v2/research_test/summary/`，summary SHA-256=`d2d2f24ba9628a5f099c0698137c88918b51e802c0aa2ee0faf6f8ff1212f862`，manifest SHA-256=`3169d062b3b93131d1bb20f38f7870ad23cd5c9536706efc50a1854d23ec415e`。

W3 research-test mean ± sample SD 为 Whole/Local/trajectory/global/PCC=`0.677887 ± 0.024346 / 0.653025 ± 0.011574 / 0.137116 ± 0.001865 / 0.166488 ± 0.001985 / 0.878694 ± 0.001267`。相对 W0，Whole/Local RR 恶化 `9.8265% / 7.1294%`（均 `0/3` seed 更优），trajectory/global/PCC 改善 `1.7417% / 3.9962% / +0.002117`（`3/3 / 2/3 / 3/3`）。因此只保留 W0=RR rate 优势、W3=morphology/correlation 优势的混合结论，不宣称唯一赢家。

P5 不评价 D4/W1/W2，不重选 checkpoint；D4“显著计算收益、轻微非灾难性质量损失”的 validation trade-off 继续保留，不因未进入 strict allowlist 而被抹除。不做 `samp_id` 分析，不构造总分。P5 已关闭，上述 evaluation/summarizer 命令只保留 provenance，不得重复运行。

## 固定呼吸带传统基线

`F0_fixed_band_bcg` 直接使用当前数据集的
`bcg_resp_band_state_aligned_segment_soft_z` 作为预测，数据 admission、validation split、
canonical 输出算子、五项 primary、IBI、eligibility 和逐 sample direct mean 均与深度学习一致。
该方法是确定性基线，不训练、不选择 checkpoint，也不报告 seed 方差。

完整 validation 统计：

```bash
./.venv/bin/python scripts/eval_tho_fixed_band_baseline.py \
  --config configs/tho_research_v2.yaml \
  --split val \
  --run-root runs/tho_fixed_band_baseline
```

实现 smoke 可以额外传入 `--max-windows 8`，但 smoke 不构成科研结果。Research-test 需
显式添加 `--split test --confirm-research-test`。

## 包络分层阈值复现

当前配置已经冻结 training target modulation 的三分层边界。以下命令只用于复现阈值与 provenance，
不得用 validation/test 重算：

```bash
./.venv/bin/python scripts/freeze_envelope_strata.py \
  --config configs/tho_research_v2.yaml \
  --output runs/envelope_strata_train_20260805.json
```

冻结结果为 `low=0.30875308839006915`、`high=0.7031542121234101`，来自 10141 个 admitted
training targets、linear quantile、train sample seed `20260610`。

## IEWT 传统基线

`IEWT` 使用与深度学习相同的
`bcg_rawish_wideband_state_aligned_segment_soft_z`，在 100 Hz 下执行协议化 Python IEWT：
三阶多项式去趋势、三阶 1 Hz 零相位低通、35 秒上下文生成 30 秒输出、六块拼接及整段
1 Hz 零相位后低通。两处低通均使用 `sosfiltfilt`，阶数和截止频率与原适配协议一致；
它只读取 BCG，target 只进入公共评价器。

完整 validation 命令为：

```bash
./.venv/bin/python scripts/eval_tho_iewt_baseline.py \
  --config configs/tho_research_v2.yaml \
  --split val \
  --run-root runs/tho_iewt_baseline
```

实现 smoke 可传入 `--max-windows 1` 或其他小值。零相位版本已完成单 sample CPU smoke 和完整
validation 新包络指标重评；此前因果版本只作系统群延迟诊断。当前实现不要求 MATLAB 数值对照，只能描述为依据
现有 MATLAB 源码定义并进行零相位适配的 Python IEWT，不能声称逐点等价。Research-test 要求
`--split test --confirm-research-test`。

该入口保存 `resolved_config.yaml`、`run_manifest.json`、`sample_metrics.csv` 和
`summary.csv`；不生成 checkpoint。

## Checkpoint 复评

Validation 复评：

```bash
./.venv/bin/python scripts/eval_tho.py \
  --checkpoint runs/<run>/checkpoint_best_local_rr.pt \
  --split val \
  --metrics-output /tmp/tho_restart_val_metrics.csv
```

Research-test 必须显式确认：

```bash
./.venv/bin/python scripts/eval_tho.py \
  --checkpoint runs/<run>/checkpoint_best_local_rr.pt \
  --split test \
  --confirm-research-test \
  --metrics-output runs/<run>/research_test_metrics.csv
```

Research-test 命令会同时计算五项 primary、IBI + coverage、三层 envelope Spearman、coherence 与 nDTW。结果可解释阶段性差异并形成后续独立研究任务，但不得重选既有 run 的 epoch/checkpoint；所有结论都必须注明是 research/development evidence。

### 当前阶段 research-test 命令

B0、T2、T4 各运行三个 validation-selected seed checkpoint，逐 run 保存结果：

```bash
for checkpoint in \
  runs/tho_restart_b0_final_loss_patchmixer/seed_*/20*/checkpoint_best_local_rr.pt \
  runs/tho_restart_t2_g3c_wide_native/seed_*/20*/checkpoint_best_local_rr.pt \
  runs/tho_restart_t4_g3c_bandenergy_native/seed_*/20*/checkpoint_best_local_rr.pt; do
  run_dir="$(dirname "${checkpoint}")"
  ./.venv/bin/python scripts/eval_tho.py \
    --checkpoint "${checkpoint}" \
    --split test \
    --confirm-research-test \
    --set training.device=cuda:0 \
    --metrics-output "${run_dir}/research_test_metrics.csv" \
    || exit 1
done
```

F0 与 IEWT 不训练、无 seed：

```bash
./.venv/bin/python scripts/eval_tho_fixed_band_baseline.py \
  --config configs/tho_research_v2.yaml \
  --split test \
  --confirm-research-test \
  --run-root runs/tho_fixed_band_research_test

./.venv/bin/python scripts/eval_tho_iewt_baseline.py \
  --config configs/tho_research_v2.yaml \
  --split test \
  --confirm-research-test \
  --run-root runs/tho_iewt_research_test
```

## 当前 run 产物

- `config.yaml`：resolved config。
- `run_manifest.json`：运行命令、Git commit 与 dirty 状态。
- `audit.csv`：数据加载审计摘要。
- `train_history.csv`：每 epoch 仅含 `train_loss_total`、`train_loss_sync`、`train_loss_effort`、`val_core_loss` 和 `val_local_rr_mae`。
- `checkpoint_best_local_rr.pt`：Local RR 严格最小 epoch；完全并列时保留更早 epoch。
- `checkpoint_final.pt`：实际完成的最后 epoch，仅用于追溯；旧阶段等于固定预算最后 epoch，P6a 可能是 early-stop epoch。
- `metrics.csv`：选中 checkpoint 的完整 validation 逐 sample 指标。
- `metrics_summary.csv`：逐 sample direct-mean validation 汇总。
- `research_test_metrics.csv` / `research_test_metrics_summary.csv`：显式 research-test 评价产物。
- `*_metrics_manifest.json`：checkpoint 复评的命令、split、配置与代码版本。
- `train.log`：训练日志。

CRD run 额外保存 `optimizer_parameter_groups.json` 与 `runtime_summary.json`；后者记录训练到最终 validation 复评期间的 CUDA peak allocated/reserved 和显存占比。`train_history.csv` 还记录 optimizer update、每 epoch 首末 LR。CRD checkpoint 的 `extra_state` 保存协议版本、actual update index、planned total updates、依赖版本与 `resume_supported=false`；P6a 另存 early-stop monitor/patience/wait/triggered/completed epochs。

不再生成或解释旧 `checkpoint.pt`、`checkpoint_best_rr.pt`、`checkpoint_best_task.pt`、`checkpoint_topN.pt`、`epoch_metrics.csv`、旧 target-feature cache 或旧指标 summary。
