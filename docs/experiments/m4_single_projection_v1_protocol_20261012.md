# 原始 P1-M4 单投影配对对照

协议 ID：`m4-single-projection-v1-20261012`。当前范围为独立实现、来源核对和 synthetic CPU 检查；正式 GPU 验收、训练和 validation 由用户执行。矩阵固定为 M4/SP × seeds 20260811、20260812、20260813。test 未开放；完成本轮并冻结 checkpoint 后另立固定 checkpoint 评价协议和执行授权。

## 来源与复用决定

本分支 `codex/m4-single-projection` 基于归并后的 `d6b85b2679499507e1034e375dfe547dd0bc54ae`，worktree 为 `/mnt/disk_code/marques/resp_reconstruction/.worktrees/m4-single-projection`。历史 `codex/patch-aligned-tf-mamba` 本地分支已在归并时删除；远端引用 `origin/codex/patch-aligned-tf-mamba` 指向 `d45d16c553313b3f3a5673e497c183ede402e2d6`，核心 P1/M4 文件与新分支基点无差异。

原始 M4 正式来源为 `p1_components_v1/formal_v1_20261006`，session 记录的 Git commit 为 `ea7106f9f3f4dd4742b052e974d59903d676f377`，当时实验扩展尚未提交，故实际身份以 session 的逐文件 SHA-256 和源码快照为准，不能只用 commit 代替。原模型为 `scripts/p1_components_model.py` 的 `P1ComponentsModel("M4")`；训练为 `scripts/p1_components_runtime.py`，正式配置由 `configs/p1_components_v1/experiment.yaml` 和 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 解析。

历史实体根目录：

`/mnt/disk_code/marques/resp_reconstruction/runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/model-architecture-review/runs`

来源核对记录为 `configs/m4_single_projection/source_audit.json`，包含两个历史 session 的路径、哈希、commit、源码比对及三份 M4 resolved config/环境身份。P1 session SHA 为 `34d0c62cb64adae16fca4db512687d28bb96e74e89857c049a506d66c4e71fb6`；M4-v2 session SHA 为 `eeb7d56625a3a8a15a9d83745e30204de936ca939d8f280ee2df371aba3ecb5d`。两者冻结源码集合逐文件比对均无差异；三个 M4 的 data/window/loss/evaluation/training（除执行 device）和 architecture 均与本轮一致。

历史 M4 环境为 PyTorch 2.12.0+cu130、CUDA 13.0、mamba-ssm 2.3.2.post1、causal-conv1d 1.6.2.post1、RTX 4070 Ti SUPER；当前 CPU 检查环境的软件版本一致，尚未执行 GPU 核验。默认优先只读复用三个历史 M4，仅训练三个 SP。`reuse-m4` 核验历史全源码、配置、completion 与引用产物哈希，保存新 session 下的引用，不改写历史文件。开跑和汇总均检查软件版本、GPU 型号及动态库环境设置；环境不匹配时显式失败，使用全新 session 同期运行六 cell。复用方式须在首个训练 cell 开始前确定；不得依据 SP 结果切换对照。

旧 M4 可在上述门槛通过后作为匹配对照：同 seed 的共有模块从初始化开始一致，训练数学、数据、选点和指标保持一致。GPU 型号一致不等于内核绝对确定性，历史与当前的负载也不能假设一致；因此效率记录只按已定义口径描述，不据此宣称精确加速比。若要求消除时段/负载差异，预先选择同期六 cell 方案。

已有 M4-v2 的 A0：W-full、物理偏置、4 头局部 attention、单个零初始化输出 Linear、原始 z 残差和主干/解码几何均符合本轮 SP 结构；训练配置亦相同。但它使用 `m4_residual.*` 初始化 namespace，连 waveform encoder、CWT encoder、Q/K/V、坐标网络、trunk 和 decoder 的初始化都不同，不能作为本轮严格配对的 SP。可以复用其结构实现依据与工程经验；已有结果保留为独立历史实验。本轮复用原 P1 的 attention forward，权重对齐后与 A0 的开放残差路径输出作合成等价检查。

历史结果入口：[原始 P1 validation](p1_components_v1_validation_results_20261007.md)、[M4-v2 validation](m4_residual_v2_validation_results_20261008.md)。本次不重评历史 checkpoint，不以旧 test 结果调整矩阵。

## 固定结构与初始化

| Arm | 计算路径 | 正式参数量 |
|---|---|---:|
| M4 | `u=z+W_ext(W_att(concat_heads))`；内部 affine 常规初始化，外部 affine 权重/偏置为零 | 1,056,357 |
| SP | `u=z+W_out(concat_heads)`；唯一输出 affine 权重/偏置为零 | 1,047,045 |

参数减少 9,312，恰为一个带偏置的 Linear(96,96)。Q 的 Linear(96,96) 属于 query 投影，计数时与多头拼接之后的输出投影区分；主干和 decoder 中的 Linear 保持原定义。

SP 通过原始 M4 构造器建立完全相同的初始化流，然后将 attention.output 置零，并将外部 ResidualAdd 替换成无参数 DirectAdd。丢弃的外部 affine 不进入最终 module/state_dict/optimizer/forward。保留 attention.output 构造时的 RNG 消耗位置，确保后续坐标网络与原 M4 相同。各模块继续使用 `p1_components.*` 隔离 seed；除 condition.output 的初始化及移除的 fusion.projection 外，全部共有 parameter/buffer 要求 `torch.equal`。

P=100、S=50、N=359、D=96，100 Hz×180秒，W-full 97尺度×360时间格，每 patch 读取2格、相对中心±0.25秒，4头；原始 CNN/GroupNorm、物理频率/时间偏置、6层 BiMamba2、dropout=0.1、MLP decoder、positive Hann OLA（epsilon=0.001）全部保持。

初始融合严格 `u=z`。首个 backward：M4 的外部投影有梯度，attention.output 及条件上游梯度为零；SP 的唯一输出投影有梯度，Q/K/V、坐标网络和 CWT encoder 梯度为零。残差主路与 decoder 可有梯度。一次更新打开零输出层后，上游梯度应非零。某些 softmax 平移不变方向可持续零梯度，不要求每一个 bias 都非零；AdamW weight decay 导致的参数变化也不能替代任务梯度检查。

这是从头训练的参数化对照，同时改变线性层数及参数化/初始化后的优化路径。不能把结果直接归因为“额外投影的表达能力”；不对已训练权重做推理折叠。

## 数据、训练与选点

沿用原始 M4 的 resolved config：train 10141窗口/32人，validation 2675窗口/7人；BCG `bcg_rawish_segment_soft_z_key`、THO `target_waveform_segment_soft_z_key`、原数据预处理、冻结 W cache 与数据/split/sample seed 不变。原 data config 中保留的 test 字段不是访问许可，入口仅构建 train/val。

物理 batch=32、梯度累积=4、有效 batch=128，BF16，尾组保留，每 epoch 80 updates。AdamW、参数分组、lr 3e-4→3e-5、5% update warmup+cosine、weight decay=1e-4、clip=1.0；loss=sync+0.25×effort，各累积组按真实 eligible 数归一化。最大80 epoch/6400 updates；早停 min_epoch=30、patience=15、min_delta=0。每 epoch 完整 validation，以 Local RR MAE 严格最小选择 best，并列保留最早 epoch。

独立 runtime 保留 P1 的数据构造、task loss、optimizer 和 `legacy.run_epochs` 完整 epoch 恢复路径。源码/配置/协议/测试/审计记录由 `prepare` 一次冻结到 session 和快照；运行及汇总要求来源不变。新产物使用独立 session，已有输出拒绝覆盖，失败 attempt 保留；`--resume` 只从最后完整 epoch 恢复。

## 固定汇总口径

完整六 cell（包括三份有效历史引用）后才能汇总。逐 seed、三 seed mean/sample SD（ddof=1）、SP−M4 配对差及差值 mean/SD/改善 seed 数全部报告，覆盖五项主要指标：Local RR MAE、Whole RR AE、Envelope trajectory MAE、Global envelope modulation error、Lag-aware signed PCC。误差 Δ<0、PCC Δ>0 为改善；保留实际方向，不挑选指标。三个 seed 共享 validation 人群，不能当作三个独立人群；validation 用于选点，结果为描述性开发证据，未预设非劣效界限，不据小差值宣称统计等价。

同时输出 per-subject、subject-macro、Local RR tail、资格分母/退化计数、完整原生 validation summary（含已有 IBI、coverage、分层 Spearman、lag 等），参数量、best epoch、完成 epoch/update 和早停状态。原生辅助指标保留其资格定义；未计算的 test-only coherence/nDTW 不填充。

效率按两种既有口径保存：①每个完整 epoch 的 train+validation+LocalRR选点耗时（不包含 epoch checkpoint 序列化）；②同 batch=32、累积=4、BF16、官方 Mamba2 的两次更新 synthetic 工程耗时和 peak allocated/reserved 显存。后者包含初次内核开销，且同 arm/GPU 的 receipt 可被多个 seed 复用，不能当作三次独立 benchmark。两种口径分别报告，不混算；本轮不估算未完整覆盖自定义 Mamba 算子的 FLOPs。

汇总校验工程回执的成功状态、对应实验臂和计时范围，耗时及峰值显存必须为有限正值；best Local RR 回执须为有限值并与所选 checkpoint 的 validation 指标一致。

## 启动与汇总（由用户执行）

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/m4-single-projection
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
SESSION=/mnt/disk_code/marques/resp_reconstruction/runs/m4_single_projection_v1/formal_v1_20261012
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py plan
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py prepare --session "$SESSION"
# 在任何训练启动前决定引用历史 M4；核验只读历史产物，不重新评价。
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py reuse-m4 --session "$SESSION"
# shard 1/2 恰为三个 SP；同 GPU 串行执行，包含各臂正式 GPU synthetic 验收。
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py run --session "$SESSION" --shard-index 1 --shard-count 2 --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py status --session "$SESSION"
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py summarize --session "$SESSION"
```

若运行环境不匹配或预先决定同期重跑：采用另一个全新 session，不执行 `reuse-m4`，运行完整六 cell（按 seed 交错 M4、SP）。不要在看到结果后更改对照来源。

```bash
SESSION=/mnt/disk_code/marques/resp_reconstruction/runs/m4_single_projection_v1/concurrent_v1_20261012
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py prepare --session "$SESSION"
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py run --session "$SESSION" --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PY" scripts/run_m4_single_projection.py summarize --session "$SESSION"
```

恢复时在同一 `run` 命令后加 `--resume`；单 cell 使用 `--arm SP --seed 20260811`，不可再混用 shard 选项。独立 GPU synthetic 验收可用 `smoke --session "$SESSION" --arms M4 SP --device cuda:0`，同样由用户执行。CPU 复核命令：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. "$PY" -m pytest -q tests/test_m4_single_projection.py
```

预期产物：session.json、source_snapshot.tar.gz；可选 cells/M4_seed*/historical_reference.json；新 cell 的 config.yaml、environment.json、attempts（数据身份、样本行、优化器分组、日志/失败记录）、epochs（checkpoint/RNG/optimizer/history）、completed.json、best checkpoint 的 validation_metrics.csv/validation_summary.csv；engineering receipt；summary/<identity>/ 下 per_seed、across_seed、paired_delta、paired_across_seed、native_validation_summary、per_subject、subject_macro、local_rr_tail、denominators、parameters、efficiency、selection CSV 和哈希回执。

验收：六 cell 身份完整、共有初始化全张量一致；每个 validation 为固定2675窗口/7人且样本/资格一致；全部主要指标在规定资格域有限，退化情况保留分母；SP 仅有一次拼接后输出投影、零初始化融合严格恒等、参数差9312；GPU 正式 task loss/32×4 两次更新后上游梯度开启；环境和所有来源哈希匹配；checkpoint 选点遵守预设规则。任一失败显式报错，保留失败 lifecycle。

## 本轮检查记录

2026-10-12：CPU 定向测试 **20 passed（31.13秒）**。包括三个 seed 的官方 Mamba2 CPU 构造、原始 M4 完整 state_dict 一致、两臂199个共有 parameter/buffer 逐张量一致、正式参数量、完整 P=100/N=359/D=96/6层几何下的实际投影调用数及 (B,1,18000) 输出、零初始化 u=z、首步门控/次步上游梯度、M4-v2 权重对齐后的前向等价、原训练合同、BF16 正式 task loss 的两次四微批更新、非有限输入失败、完整五指标配对/SD及缺失 cell 拒绝、source snapshot、历史只读引用、环境/产物变更拒绝和完整汇总生命周期。前后向使用显式 Mamba 替身；未运行官方 GPU 内核。

来源/config/软件版本的只读核对已完成，结果见 source_audit.json。完善快照依赖集合后，另外复核了 prepare、历史引用/完整汇总及复用方式冻结三项测试，3 passed（8.78秒）；CLI 的 SP 分片准确列出三个预设 seed，`git diff --check` 通过。官方 Mamba2 的 forward/backward、GPU显存和正式数据链由用户启动阶段验证。没有启动训练、GPU任务或真实数据评价。

提交前复核：实现、配置与协议一致；补齐效率回执和选点值校验后，完整定向 CPU 集合 **27 passed（29.55秒）**，覆盖异常耗时/显存、失败或错臂验收、计时范围变化及非有限选点回执的拒绝。
