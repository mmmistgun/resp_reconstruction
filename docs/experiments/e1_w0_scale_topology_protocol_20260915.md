# E1：冻结 W0 的尺度重排敏感性审计

协议 ID：`e1-w0-scale-topology-v1-20260915`；日期：2026-09-15。

状态：实现与 26 项 synthetic CPU 定向测试已完成；索引/实现锁由第 7 节入口生成并核验；GPU 验收和完整 validation 推理待用户执行。

## 1. 任务与数据范围

本协议承接论文工作区的 `实验项目待补实验清单_20260915.md` 第 3 节，并落实用户在 2026-09-15 对 E1 数据访问、结果解释及工程定义的决定。E1 按本协议执行；loss、五项指标和任务投影沿用既有冻结定义。历史 P5、CRD-TF-W v2 产物继续作为只读来源。

科学问题是：冻结 W0 的预测质量与 FiLM 条件输出，对指定尺度重排有多大响应，以及响应是否随指标和训练 seed 改变。三个操作分别改变多种尺度结构属性；解释时结合其边界效应。

首轮固定完整 validation：三个 validation-selected W0 checkpoint，seed 为 `20260811/20260812/20260813`，selected epoch 为 `13/15/14`，每个 checkpoint 对应 2675 个窗口、7 个 `samp_id`。当前问题可由这一轮回答。

用户允许在科学问题确有需要时访问 test，既有 test 评价经历本身不构成拒绝访问的理由。例如，validation 得到的尺度重排响应是否在既有 test 人群中保持，可以构成后续评价问题。实际访问前，以配套附件明确问题、访问历史、checkpoint/cache/row 身份、完整条件矩阵、指标和输出 identity；按对应 split 分别汇总。相关结果的证据定位应反映其开发和访问历史。E1 当前实现和运行合同只覆盖上述 validation 矩阵，test 附件尚未建立。

正式 GPU 验收及完整推理由用户执行；Codex 当前可完成实现、协议、身份锁准备及 synthetic CPU 定向测试。

## 2. 冻结模型与输入身份

W0 为 `crd_tf102_w`。身份来源为 `docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json` 的 W0 三 seed 条目、对应历史 resolved config、validation summary 和 W cache 条目。正式实现锁应逐项登记并验证路径、大小与 SHA-256，记录当前推理代码及依赖环境。

输入 W cache 的单窗口 shape 为 `[97,360]`，轴语义为尺度、时间。模型输入为 `[B,97,360]`，进入 `CwtBranch.conv_in` 时为 `[B,1,97,360]`。操作施加在从只读 cache 复制出来的 batch 上，对应尺度轴分别是 `dim=1` 和 `dim=2`。每个尺度的 360 点时间序列整体搬移，模型参数、任务投影和归一化沿用原生路径。

频率文件作为原始尺度到频率的身份依据保持只读；每个条件的槽位频率通过 `original_frequencies[index]` 描述，保留操作后的次序。

## 3. 尺度操作与索引锁

统一使用 gather 语义：`output[b,s,t] = input[b,index[s],t]`。

| 条件 | 索引定义 | 解释 |
|---|---|---|
| `FULL` | `index[s] = s` | 原生 forward 的复现锚点 |
| `SCALE_SHIFT_12` | `index[s] = (s - 12) % 97` | 正向循环移动 12 槽，等价于 `roll(shifts=+12)` |
| `SCALE_REVERSE` | `index[s] = 96 - s` | 反转尺度顺序 |
| `SCALE_PERMUTE_FIXED` | `Generator(PCG64(20260915)).permutation(97)` | 使用独立 NumPy RNG 一次性生成，全部窗口和三个 checkpoint 共用 |

在查看本轮干预结果前，生成并保存四个条件的实际 97 项索引、逆索引 `argsort(index)`、NumPy 版本、算法名、种子及自检回执。后续执行直接读取保存的数组。数组哈希使用 C-order、little-endian int64 的连续字节；另记录索引锁 JSON 文件本身的 SHA-256。

正向移动意味着原槽位 `s` 的内容到达 `(s+12)%97`。12 voices/octave 对应名义一个倍频程的索引间隔；实际 mapped frequency 与循环回绕部分按频率文件描述。

代码中的尺度卷积使用零填充。SHIFT 保留多数内部邻接，但改变两端内容并产生高低频接缝；REVERSE 保留无方向的相邻尺度对，同时改变方向和边界内容；PERMUTE 同时改变槽位、邻接和边界内容。响应归属于指定重排操作。单个置换只提供该固定置换下的证据。

自检包含：索引为 `0..96` 的双射、正逆组合为 identity、shape/dtype 一致、原 batch 未被修改，以及带有尺度和时间唯一编码的 synthetic tensor 经操作后满足逐元素 gather 定义。对每个时间点核对跨尺度数值多重集，并核对完整尺度时间序列搬移，以防误换时间轴。

## 4. 五项指标与结果解释

沿用冻结五主指标及其 eligibility/degeneracy 定义，逐窗口生成结果，在每个 seed 内执行既有 sample-direct 聚合。

对 error 指标 E，逐 seed 配对变化为：

`delta_E(seed) = 100 * (E_intervention(seed) - E_FULL(seed)) / E_FULL(seed)`。

对 signed PCC，逐 seed 配对变化为：

`delta_PCC(seed) = PCC_FULL(seed) - PCC_intervention(seed)`。

两者正值均表示恶化，负值表示改善。error 的 FULL 分母必须大于零；不满足时显式失败，保留原始值及失败原因。

完整交付每个条件的逐 seed 原始指标、三 seed arithmetic mean 和 sample SD（`ddof=1`），以及 paired delta 的 mean、sample SD、正/负/零方向数。另列以三 seed 指标均值计算的相对变化，与 paired delta mean 使用独立列名。方向数按所存完整精度的符号计算，避免显示舍入影响计数。`samp_id` 用于身份追溯；三个训练 seed 描述训练随机性。

本轮采用连续效应量和方向报告。结果完成后，综合变化幅度、seed 一致性、指标间权衡和 FULL 数值复现情况评估解释力度；届时采用的阈值或分类规则登记为结果后探索性解释，并保留完整结果。接近或方向混合时，区分“观察到的敏感性较弱/不一致”与“模型不使用尺度结构”两类结论。

E4 的立项结合 E1 结果和新模型研究目标另行决定。

## 5. FiLM 配对统计

本轮的原始 FiLM 明确定义为 `final_projection(context).chunk(2, dim=1)` 的两个输出，即进入 `0.5*tanh` 前的 `gamma_raw` 与 `beta_raw`，每个形状为 `[B,96,1800]`。

对同 seed、同 `dataset_row_id` 的窗口 i，分别计算：

`gamma_raw_pair_mae[i] = mean_{channel,time} abs(gamma_raw_intervention[i] - gamma_raw_FULL[i])`。

`beta_raw_pair_mae[i] = mean_{channel,time} abs(beta_raw_intervention[i] - beta_raw_FULL[i])`。

在原生输出完成后，对捕获张量脱离计算图并转至 CPU，以 float64 作差和归约；有限性检查覆盖原始张量、差值与统计结果。先得到每窗口配对 MAE，再对 2675 个窗口直接平均，最后计算三 seed mean 和 sample SD。FULL 与自身的配对 MAE 应为零。

配对要求 seed、row identity、通道和时间位置完全一致。使用逐位置绝对差的均值；各条件的幅值均值可作为背景信息，但不代替配对 MAE。历史 corrected FiLM 表使用有界有效量，引用时明确其与本轮 raw 量的区别。

捕获机制在原生 forward 内只保留张量引用，统计计算安排在输出完成之后。通过 synthetic identity 测试与正式 FULL 复现核对包装器的影响，所有条件复用 W0 原生 forward。

## 6. 验收与输出合同

1. 新 evaluator、wrapper 与输出 identity 独立于历史入口。正式运行前冻结 implementation lock、索引锁、resolved config、命令、代码 commit 和环境记录。
2. FULL 逐 seed 对冻结 validation summary 的五项主指标执行 `rtol=0, atol=1e-6` 检查。正式 batch 固定为历史配置的 128，AMP 为 bf16；锁定 row 顺序与末 batch 行为。全部三个 FULL 通过后开放正式干预评价。
3. GPU batch-1 finite 验收使用 synthetic fixture，检查运行能力。完整 validation FULL 负责正式批量条件下的数值复现。超差时保留失败 lifecycle 和差值回执，再排查身份、实现和环境。
4. 每条件 8025 条唯一记录，四条件共 32100 条；唯一键为 `(split, condition, seed, dataset_row_id)`，每个 `(condition,seed)` 的 row 集合及顺序与 FULL 一致。
5. 输入、prediction 与关键指标均有限，prediction degeneracy 为零，完整记录既有指标资格与分母。失败显式中止并保留产物。
6. 输出逐窗口 metrics、FiLM 配对表、逐 seed summary、三 seed summary、paired delta 表、验收与访问回执、lifecycle 和 manifest。
7. 新输出根建议为 `runs/e1_w0_scale_topology_v1/validation/`，每次执行使用独立 attempt 子目录，名称同时包含阶段、implementation lock 标识和 attempt ID，以排他创建保证不可覆盖。failed attempt 原地保留。
8. 完成 manifest 记录全部被管理产物的相对路径、大小与 SHA-256；manifest 自身的哈希由外层冻结回执记录，避免自引用。推理和汇总冻结后再交付论文工作区。

## 7. 实现、准备与执行

实现入口：`scripts/eval_e1_w0_scale_topology.py`；核心操作与统计为 `resp_train/paper_evidence/e1_scale_topology.py`，运行与产物管理为同目录的 `e1_scale_topology_runtime.py`。

固定锁路径：

- 索引锁：`docs/experiments/e1_w0_scale_indices_20260915.json`。
- 实现锁：`docs/experiments/e1_w0_scale_implementation_lock_20260915.json`。

准备命令只读核验 W0 三 seed checkpoint/config/manifest/validation summary、validation W cache、row 文件及频率文件的大小与 SHA-256。checkpoint/cache 仅按字节计算身份，随后保存索引自检结果和代码身份。实际运行重新核验来源、shared dataset index、索引锁与代码锁。

```bash
./.venv/bin/python scripts/eval_e1_w0_scale_topology.py prepare-locks
```

该命令排他创建锁文件。实现锁同时覆盖 E1 入口、定向测试、本文和 `resp_train` Python 源码，以记录原生推理与指标依赖；后续源码修订应形成新实现身份。锁中的 preparation Git 状态说明准备时的工作树，正式 attempt 另行记录干净执行 commit。

### 7.1 CPU 验收回执

```bash
./.venv/bin/python -m pytest tests/test_e1_scale_topology.py -q
```

2026-09-15 执行结果：`26 passed`。验证覆盖索引正逆与每时间点数值多重集、完整尺度时间序列搬移、原生 forward 精确一致、hook 时序和异常清理、FiLM float64 配对公式、尾 batch 与 row 顺序、target-only eligibility、完整三 seed 配对统计、来源锁漂移、失败产物保留及 manifest 完整性。

完整流程测试使用 7 个 disposable synthetic 窗口和 mock W0；真实 CWT/FiLM/decoder 的 CPU exact 测试仅在 fixture 中把 CUDA Mamba blocks 替换为 identity。测试没有访问真实波形或历史 checkpoint。正式原生 Mamba 数值验收由下面两个 GPU 阶段完成。

### 7.2 用户执行顺序

整理并提交待执行代码、协议和锁文件，确保工作树干净。先执行 synthetic GPU batch-1 验收：

```bash
./.venv/bin/python scripts/eval_e1_w0_scale_topology.py gpu-smoke --device cuda:0
```

验收使用新初始化 W0 和合成输入，并激活原本零初始化的 FiLM 投影以覆盖条件路径。记录四条件 output shape/finite、raw FiLM 配对 MAE、FULL 与原生输出的实际差值，以及输入未修改回执。正式 batch 的 FULL 复现仍由完整 validation 阶段验收。

命令输出新 attempt 目录。该目录的 `gpu_acceptance.json`、`manifest.json` 和 `freeze_receipt.json` 完整后，将实际目录传给完整评价：

```bash
./.venv/bin/python scripts/eval_e1_w0_scale_topology.py validation \
  --device cuda:0 \
  --gpu-receipt '/实际完成的gpu_smoke_attempt目录'
```

GPU 验收回执必须绑定同一 implementation lock。完整评价按三个 FULL → FULL 总体验收回执 → 三 seed 各三个干预的顺序执行，最终自动汇总完整矩阵。每个 seed/condition 目录单独保存 `metrics.csv`、`film_pair_metrics.csv`、`summary.csv` 和评价回执，根目录保存原始三 seed 汇总、paired delta、环境及访问记录。

为了跨三个 FULL 验收门配对且保留可审计来源，FULL raw FiLM 使用 float32 `.npy` 存储，原生 bf16 数值可无损表示。三个 seed 的六个文件合计约 10.3 GiB；分批读写与计算 MAE，完整 attempt 还需容纳 metrics 和元数据。原始 FiLM 文件保留并进入 manifest。

每次运行创建新 attempt。出现异常时保存 `lifecycle_failed.json` 和已有文件；成功以 `manifest.json` 加 `freeze_receipt.json` 为完成依据。全部最终记录共 32100 条 metrics 与 32100 条 FiLM 配对记录。首轮范围为完整 validation，test 按第 1 节规定的科学问题与附件开放。
