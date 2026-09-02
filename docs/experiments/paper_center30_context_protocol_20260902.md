# Center-30 短窗口上下文敏感性协议

日期：2026-09-02

协议 ID：`paper-center30-context-v1-20260902`

状态：**实现、30/45 s input-only W cache 与八项单 seed formal 配置均已完成并冻结；等待用户手动执行 P4-S2。独立测试集访问未开放。**

## 1. 科学问题与位置

本任务作为论文证据闭环总协议中 P4-S 短窗口分支，游离于论文主模型实验之外。它不重开已冻结的 center-60 P3/P4，
也不回写 W0/W3/C201/RTM 的既有选择、Pareto 或独立测试集结论。

center-60 P4 只说明：当输出固定为中心 60 s 时，输入从 60 s 延长至 90/180 s 没有一致、材料性的跨 seed 收益；
它不能证明 60 s 是合理下限。本协议回答：当输出固定为中心 30 s 时，输入 30/45/60/90 s 的额外上下文是否产生收益，
以及最短可用上下文落在何处。

此前冻结的 180→180、center-60 的 60→60 与本任务的 30→30 可形成任务尺度的描述性证据梯级，但三者的输出长度、
指标采样性质和训练任务不同，不得把绝对指标拼成一条可直接排序的曲线。

## 2. 固定矩阵与嵌套视图

父窗口固定为 180 s、100 Hz、18,000 点；共同 target 为中心 30 s：`[7500,10500)`。

| arm | input slice | input/output ratio | target 两侧额外上下文 |
|---|---|---:|---:|
| input 30 s → center 30 s | `[7500,10500)` | 1.0 | 0 s |
| input 45 s → center 30 s | `[6750,11250)` | 1.5 | 每侧 7.5 s |
| input 60 s → center 30 s | `[6000,12000)` | 2.0 | 每侧 15 s |
| input 90 s → center 30 s | `[4500,13500)` | 3.0 | 每侧 30 s |

模型固定为 `C201-center30 / W-reduced-center30`，形成 2×4 共八臂。不得称 W-reduced-center30 为严格 W0。
同一 parent row 的四种输入视图必须逐点共享同一 center-30 target；train/validation 的 row、samp_id、subject/session
隔离沿用 center-60 冻结身份，只允许访问 train/validation。

## 3. 模型与 49-scale W 表征

两种模型保持 center-60 C201/W-reduced 的 frontend、六层双向 Mamba trunk、refinement、coarse head、decoder residual
和初始化命名空间。输入 latent 长度为 `N/10`，先在完整输入上下文上执行 trunk/refinement/head features，再裁共同中心
300 latent points，经 10 Hz head 与 Fourier interpolation 输出 3000 点。

可训练参数数固定为 C201 `1,069,802`、W-reduced `1,219,850`，差值仍为 W branch 的 `150,048`；参数数不随输入
长度变化。同 seed 的共享 state tensor 必须逐 tensor 相同。

W-reduced 继续使用完整 12-voice 网格的偶数索引，保持 49 个名义 scales、input-only、`log1p(abs(CWT))` 后每 50 点
平均。60/90 s 直接复用已冻结的 center-60 input-only cache；30/45 s 需要新建独立 cache，不读取 target/test。

短窗有限长度会降低实际频率中心的可实现性。`ssqueezepy==0.6.6` 定向映射在 30/45 s 上观察到：

| input | feature shape | actual center 重复数 | 全网格最大名义频率误差 | 0.05–0.70 Hz 名义区间最大误差 |
|---:|---:|---:|---:|---:|
| 30 s | `49×60` | 1 | `0.577335 Hz` | `0.358809 Hz` |
| 45 s | `49×90` | 3 | `0.324396 Hz` | `0.187911 Hz` |

该限制必须随 W-reduced 结果报告；不事后删 scale、换网格或把 W-reduced 的短窗变化纯粹解释为上下文效应。
C201-center30 作为不受 W scale mapping 影响的结构对照。

## 4. Center-30 任务合同

固定 `Pi_30`：对 3000 点 prediction/target 分别进行 0.05–0.70 Hz FFT hard band projection、去均值并按中心 RMS
归一化，`eps=1e-8`。loss 为 `L_sync_30 + 0.25 L_effort_30`：

- lag 搜索固定 ±0.30 s，tie-break 沿用冻结 lag priority；
- effort log-RMS 固定 10 s window / 5 s step；
- prediction 非有限使整个 checkpoint 失败；
- target 无资格时按冻结 eligible-count 聚合，不静默丢弃 prediction failure。

五项 center-30 primary metrics 使用独立 `center30_*` 命名：

1. `center30_rr_mae_bpm`；
2. `center30_ibi_medae_sec`；
3. `center30_envelope_trajectory_mae`；
4. `center30_global_envelope_modulation_error`；
5. `center30_lag_aware_signed_pcc`。

RR prediction 退化固定计 39 bpm，PCC prediction 退化固定计 `-1`。IBI 必须同时报告 coverage、interpretable fraction
和 target-eligible count。30 s 的 FFT bin spacing 为 2 bpm，3 bpm 目标在 30 s 内仅约 1.5 cycles；10 s/5 s envelope
在完整 30 s 输出上固定形成 5 个轨迹点。这些是 center-30 任务本身的观测性质，不通过过滤样本、零填充解释或修改窗口
来规避；若短窗在这些指标上不能覆盖，作为支持更长任务窗口的证据报告。

checkpoint selector 固定为完整 validation 2675 rows 上更小的 `center30_rr_mae_bpm`，完全相同时保留更早 epoch。

## 5. 相对变化与跨任务联系

本任务优先报告 paired-seed 相对变化。四个 error 指标的有向改善定义为 `(baseline-new)/baseline`，PCC 同时报告绝对
差和 `(new-baseline)/abs(baseline)` 的补充相对变化。以 30→30 为基线报告 30→45、30→60、30→90，并报告
45→60、60→90 的增量变化；不构造总分或唯一赢家。

与 center-60 P4 只比较相对增益方向：

- ratio 1.5：本任务 45→30 对比 center-60 的 90→60；
- ratio 3.0：本任务 90→30 对比 center-60 的 180→60；
- ratio 2.0 的 60→30 是本任务新增中间点。

由于 target 长度不同，跨任务比较不得使用绝对指标差、p-value 或合并 seed 统计。

结论分支预先固定为：

- 30/45 均较差、60 改善且 90 无一致追加收益：支持 60 s 是合理上下文；
- 30 较差但 45 已恢复到 60/90 水平：支持 45 s，而非单独支持 60 s；
- 30 与更长输入接近或更好：30 s 可能足够；
- 90 仍有一致材料收益：现有矩阵尚未定位饱和点；
- 模型或 seed 方向不一致：作为表征/协议敏感性结果留档。

“材料性”沿用 center-60 P3：error 相对变化 `0.5%`，PCC 绝对变化 `0.002`。单 seed 只形成方向诊断，不形成稳定结论。

## 6. 分阶段执行与成本

### P4-S0：实现与 CPU 定向测试

实现独立 config/data/model/loss/metrics/cache 空间，验证四种 nested views、3000 点输出、五项指标、参数身份、49-scale
shape、train/validation-only access 与 cache 不覆盖。实现配置不得执行真实 lifecycle。

### P4-S1：30/45 s W cache

用户从干净 commit 手动执行：

```bash
./.venv/bin/python scripts/build_paper_center30_context_w_cache_v1.py \
  --config configs/paper_evidence_v1/center30_context_v1.yaml \
  --input-sec 30 \
  --confirm-cache-build

./.venv/bin/python scripts/build_paper_center30_context_w_cache_v1.py \
  --config configs/paper_evidence_v1/center30_context_v1.yaml \
  --input-sec 45 \
  --confirm-cache-build
```

完整 cache 必须为 `10141 train / 2675 validation`、49 scales、float32、finite、input-only；目录禁止覆盖。60/90 s
复用既有冻结 cache 及 manifest SHA-256，不重建。

P4-S1 已从干净 commit `693ce63` 完成并冻结：

- 30 s：`runs/paper_evidence_v1/center30_context_w_cache/30s_127e60642b74716a5afa1255060f52e770e275f8426c12d24775bf1194123b58/`，
  manifest SHA-256=`d64e696a686ebe3f11ec279c276aa1666a74b81d889ecd3c5fca8075bbdb79ed`；
- 45 s：`runs/paper_evidence_v1/center30_context_w_cache/45s_9b56099708bccd89b53e8ee6d8716863342f741f13b94a67ae9f772255edd603/`，
  manifest SHA-256=`f5e525719906ebd4bdf4836f8ad4b7e9b722ebb2e195d48c28d40a643fbe199f`。

两份 cache 均为 `10141 train / 2675 validation`、49 scales、float32、finite、input-only，lifecycle=`complete`；
target/test 均未读取，manifest 登记的全部文件大小与 SHA-256 已复核。train/validation row hashes 与 center-60 cache
逐项一致。

### P4-S2：单 seed 八臂

30/45 s cache 完整、hash 冻结后创建以下八项 formal configs：

```text
configs/paper_evidence_v1/p4s_c30v1_c201_30.yaml
configs/paper_evidence_v1/p4s_c30v1_c201_45.yaml
configs/paper_evidence_v1/p4s_c30v1_c201_60.yaml
configs/paper_evidence_v1/p4s_c30v1_c201_90.yaml
configs/paper_evidence_v1/p4s_c30v1_wr_30.yaml
configs/paper_evidence_v1/p4s_c30v1_wr_45.yaml
configs/paper_evidence_v1/p4s_c30v1_wr_60.yaml
configs/paper_evidence_v1/p4s_c30v1_wr_90.yaml
```

固定入口为 `scripts/train_paper_center30_context_v1.py`，stage/gate 为 `p4s_single_seed / p4s_formal`，输出根为
`runs/paper_evidence_v1/center30_context/p4s_single_seed/`。固定 seed `20260811`、80 epochs、统一
physical batch `128×1`、bf16、TF32=false、cuDNN benchmark=false、early stopping=false、resume=false。
现有 W-reduced-180 batch-128 acceptance 比本矩阵最大 W-reduced-90 更保守，不新增 GPU acceptance。

### P4-S3：条件三 seed

单 seed 八臂全部完成后先冻结方向汇总。若继续稳定性证据，追加 seeds `20260812/20260813` 的 16 runs，最终 24/24；
不得只扩展数值较好的长度或模型。P4-S3 成本需要用户另行确认。

## 7. 当前实现回执

- 实现配置：`configs/paper_evidence_v1/center30_context_v1.yaml`；
- 参数数：`1,069,802 / 1,219,850`，W branch 差值 `150,048`；
- 30/45/60/90 s 输入均输出 3000 点，latent center 均为 300 points；
- center-30 metric 保持 30 s 自身的 2 bpm FFT spacing、5-point envelope 与 IBI coverage 口径；
- 两份新 cache manifest SHA-256 为
  `d64e696a686ebe3f11ec279c276aa1666a74b81d889ecd3c5fca8075bbdb79ed / f5e525719906ebd4bdf4836f8ad4b7e9b722ebb2e195d48c28d40a643fbe199f`；
- 新增 15 项 center-30 定向测试，并与 47 项既有 paper-evidence 回归合计 62 项通过；
- 当前开放 P4-S2 八项单 seed formal，由用户手动执行；P4-S3、benchmark 和 test 均未开放。

下一推进点是用户从新的统一干净 commit 按双 GPU 顺序队列执行八项 P4-S2 formal runs。
