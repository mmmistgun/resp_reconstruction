# Center-90 输出上下文尺度协议

日期：2026-09-04

协议 ID：`paper-center90-context-v1-20260904`

状态：**独立辅助实验已完成并冻结。C201-center90 / W-reduced-center90 × 90/135/180 s 输入 × 3 seeds 的
18/18 formal lifecycle 与 validation 汇总均闭合；结果支持“center-90 上下文长度效应弱且依赖表征”，不进行长度或模型重选。**

## 1. 科学问题与证据位置

本任务回答：当输出固定为中心 90 s 时，输入覆盖输出本身、增加半个输出长度上下文、增加一个输出长度上下文，是否带来
稳定的 RR 或波形属性收益。它是独立的输出任务尺度实验，结果以单独 panel 报告，不参与论文主模型、既有 center-30、
center-60 或 180→180 结果的重选。

Center-30、center-60、center-90 可按 input/output ratio 描述上下文响应，但各任务的 `Pi`、FFT 分辨率、包络轨迹点数、
训练 target 和 checkpoint selector 不同。跨任务只比较相对变化方向，不合并绝对指标、不构造总分或统一排名。

## 2. 冻结矩阵

| 模型 | 输入 | 输出 | ratio | seeds | runs |
|---|---:|---:|---:|---|---:|
| C201-center90 | 90/135/180 s | 中心 90 s | 1.0/1.5/2.0 | 20260811/12/13 | 9 |
| W-reduced-center90 | 90/135/180 s | 中心 90 s | 1.0/1.5/2.0 | 20260811/12/13 | 9 |

全部 18 项作为同一 formal 矩阵执行。训练固定 80 epochs、physical batch 128、
gradient accumulation 1、bf16、TF32=false、cuDNN benchmark=false；若 P2 最大臂验收表明 batch 128 不可行，必须在正式
训练前统一修订 18 项 batch/accumulation 合同并记录等效 optimizer batch，不允许只修改单臂。

## 3. 数据与嵌套视图

继续使用冻结的 180 s admitted parent rows、train/validation split、采样种子和原始 soft-z 载体：

- input key=`bcg_rawish_segment_soft_z_key`；
- target key=`target_waveform_segment_soft_z_key`；
- target 固定为中心 `[4500,13500)`，共 9000 点；
- train/validation rows 固定为 `10141 / 2675`，共享既有 row-ID 与 subject/session 隔离身份；
- 只访问 train/validation。

| 输入 | input slice | latent 长度 | 输出 latent slice | 每侧额外上下文 |
|---:|---:|---:|---:|---:|
| 90 s | `[4500,13500)` | 900 | `[0,900)` | 0 s |
| 135 s | `[2250,15750)` | 1350 | `[225,1125)` | 22.5 s |
| 180 s | `[0,18000)` | 1800 | `[450,1350)` | 45 s |

同一 parent row 的三种输入视图必须逐点共享同一 center-90 target。输入继续直接裁 parent soft-z，不按长度重新归一化。

## 4. 模型与 W 表征

模型沿用同一 C201 backbone、六层 Local BiMamba2、refinement、coarse head 与 10-Hz capacity decoder：

- `C201-center90`：仅时域输入；
- `W-reduced-center90`：增加完整频程的 49-scale input-only W branch 与 FiLM。

两模型的共享模块按相同 seed 逐 tensor 同初始化；各输入长度内参数数固定为 `1,069,802 / 1,219,850`。W 表征继续使用
Morlet `mu=13.4`、12-voice 完整网格偶数索引、49 个名义 scales、reflect padding、`log1p(abs(CWT))` 后每 50 点平均。

90 s 与 180 s 输入的切片和 transform spec 与已冻结 train/validation W cache 完全相同，直接复用：

- 90 s manifest SHA-256=`9442a33ea2c633b15642d033efa3d3e09278f30c4692a707abc9de460c784757`；
- 180 s manifest SHA-256=`72402d8543adf3cda504967b87fc4f9528cacca58b9aa080f0dcfe06707037fe`。

135 s 使用独立 `[10141/2675,49,270]`、float32、finite 的 input-only cache，输出根为
`runs/paper_evidence_v1/center90_context_w_cache/`；固定 manifest SHA-256=
`9454fe918aaa6a4844239a5c8479665cc80f31cf6c64cfb30fc06d069cd7c473`。

## 5. `Pi_90`、loss、指标与 selector

`Pi_90` 固定为 9000-point hard FFT band projection `0.05–0.70 Hz`，随后对中心 90 s 整体去均值并按整体 RMS
归一化，`eps=1e-8`。Loss 固定为 `L_sync_90 + 0.25 L_effort_90`：lag ±0.30 s，effort log-RMS 使用
10 s window / 5 s step。

五项 primary 使用独立 `center90_*` 命名：

1. `center90_rr_mae_bpm`：90 s hard-FFT dominant RR，频率间距 `1/90 Hz`、RR 间距约 `0.666667 bpm`；
2. `center90_ibi_medae_sec`，同时报告 coverage、interpretable fraction 与 eligible count；
3. `center90_envelope_trajectory_mae`：17 点 log-RMS envelope；
4. `center90_global_envelope_modulation_error`；
5. `center90_lag_aware_signed_pcc`。

Checkpoint 固定由完整 2675-row validation 的 `center90_rr_mae_bpm` 严格更小选择，同值保留更早 epoch，文件名为
`checkpoint_best_center90_rr.pt`。Prediction 非有限使整个 checkpoint 评价失败；IBI 不可解释样本按 coverage 与 eligible-count
口径显式保留。

## 6. 执行阶段

### P0：实现与 CPU 定向测试

实现独立 config/data/model/loss/metrics/cache/experiment/CLI 空间，验证三种 nested views、9000 点输出、900 点 decoder
latent、17 点 envelope、参数身份、49-scale shape、train/validation access、formal gate 和输出不可覆盖。CPU 测试只使用
合成 tensor 或 fixture，不创建真实训练 lifecycle。

### P1：135 s W cache

用户从 P0 干净实现 commit 执行：

```bash
./.venv/bin/python scripts/build_paper_center90_context_w_cache_v1.py \
  --config configs/paper_evidence_v1/center90_context_v1.yaml \
  --input-sec 135 \
  --confirm-cache-build
```

完成后只读核对 lifecycle、manifest、全部文件 size/SHA-256、row identity、shape/dtype/finite、frequency identity 与 access
flags，再冻结 cache path/hash。

P1 已由用户从干净 commit `2c151cba2e41056a115dd5f7f0293fa486666c8d` 完成，耗时 `06:57`。固定目录为：

```text
runs/paper_evidence_v1/center90_context_w_cache/
  135s_7bbce891b3e158079f1b6d59b6a97fe273aa8293bec5f6c0964a9eb85d913cd3/
```

`cache_manifest.json` SHA-256=`9454fe918aaa6a4844239a5c8479665cc80f31cf6c64cfb30fc06d069cd7c473`。
只读验收确认 lifecycle=`complete`、train/validation=`10141/2675`、row-ID SHA-256 与冻结 split 一致且零交集；feature
shape=`[10141/2675,49,270]`、float32、全量 finite，49 个 frequency centers 在 split 间完全相同。Manifest 登记的
5 个 `.npy` 文件 size/SHA-256 均匹配，duplicate center=0，max nominal frequency error=`0.035647 Hz`；access
identity 为 train/validation input-only，未读取 target 或 test。

### P2：最大臂 GPU 工程验收

工程验收只用 synthetic tensors，对 `W-reduced-center90 / input 180 s / batch 128` 执行一次真实
forward/loss/backward/AdamW step，确认 output、loss、gradient、optimizer state 与显存状态。它不读取 dataset、cache、
checkpoint 或任何 split，只决定统一 `128×1` 运行合同是否成立。

固定命令为：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<GPU> \
./.venv/bin/python scripts/accept_paper_center90_context_v1.py \
  --config configs/paper_evidence_v1/center90_context_v1.yaml \
  --device cuda:0 \
  --confirm-gpu-acceptance
```

P2 已由用户从同一干净 commit 在 NVIDIA GeForce RTX 4070 Ti SUPER 上完成。固定输出为
`runs/paper_evidence_v1/center90_context/gpu_acceptance/2c151cba2e41/`；receipt/manifest SHA-256=
`36ab2e4c9c2d3276078728bf657463619b82e02776aa4ff4238101643c891b87 / 721fe0a8d59460809666723daf9ba9dee9cfde1f4aab28934297d83cd1e25935`。
最大臂 batch-128 输出 shape=`[128,1,9000] / [128,1,900]`，loss、gradient、parameter 与 optimizer state 均
finite，peak allocated/reserved=`11021.50 / 11628.00 MiB`，占设备总显存约 `69.16% / 72.97%`，
decision=`batch128_accepted`。Access receipt 确认仅使用 synthetic tensor，统一 formal 合同冻结为 `128×1`。

### P3：18 项 formal 与冻结汇总

P1/P2 冻结后已生成 18 项显式配置，文件集合为：

```text
configs/paper_evidence_v1/c90v1_<c201|wr>_<90|135|180>_seed<20260811|20260812|20260813>.yaml
```

每项使用独立不可覆盖目录：

```text
runs/paper_evidence_v1/center90_context/formal/C90V1_<C201|WR>_<90|135|180>/seed_<seed>/
```

全部 18/18 完成前不汇总。正式汇总逐 seed 报告，再报告 arithmetic mean ± sample SD (`ddof=1`)；按模型分别计算
90→135、135→180、90→180 的 paired-seed 有向变化。四个 error 的材料阈值为相对 `0.5%`，PCC 为绝对 `0.002`。
结果只形成 center-90 validation 上下文尺度证据。

两张 GPU 各运行 9 项顺序队列；每个 seed 的两个模型×三个长度在两卡间互补分配，每张卡对每个 seed 均覆盖
90/135/180 s 各一项。进程内统一使用逻辑 `cuda:0`，物理卡只由 `CUDA_VISIBLE_DEVICES=0/1` 指定；同一队列用
shell `&&` 串行，任一失败使该队列停止并保留 lifecycle。

P3 已由用户从干净 commit `033e3ff2a9fdb793ad0ffc93fbb9f7821c25b24a` 完成。只读冻结汇总从干净 commit
`572fd51ae236c7b5e55520b624f8eb871977966a` 执行，确认 18/18 lifecycle complete、每项 80 epochs / 6400 updates、
全部登记 artifact 的 size/SHA-256、相同 2675-row validation identity、逐样本指标重算与 strict-lower selector 均闭合。
未读取 checkpoint 内容、dataset/signal/target array 或 test，未执行训练、推理或 GPU 计算。

五主指标的三 seed arithmetic mean ± sample SD 如下（四项 error 越小越好，PCC 越大越好）：

| 模型 | 输入 | RR MAE bpm | IBI MedAE s | trajectory MAE | global modulation error | signed PCC |
|---|---:|---:|---:|---:|---:|---:|
| C201-center90 | 90 | 0.518453 ± 0.004980 | 0.089797 ± 0.004639 | 0.140956 ± 0.002154 | 0.196425 ± 0.014899 | 0.857213 ± 0.006581 |
| C201-center90 | 135 | 0.513817 ± 0.010833 | 0.086035 ± 0.001671 | 0.138107 ± 0.002733 | 0.185759 ± 0.006142 | 0.862294 ± 0.002151 |
| C201-center90 | 180 | 0.510009 ± 0.011051 | 0.088705 ± 0.002945 | 0.145000 ± 0.012126 | 0.185518 ± 0.008555 | 0.860816 ± 0.003082 |
| W-reduced-center90 | 90 | 0.499880 ± 0.007252 | 0.088859 ± 0.004431 | 0.140258 ± 0.003825 | 0.185696 ± 0.002134 | 0.861503 ± 0.004302 |
| W-reduced-center90 | 135 | 0.501253 ± 0.003332 | 0.086944 ± 0.001790 | 0.145575 ± 0.007098 | 0.188194 ± 0.002416 | 0.868063 ± 0.001041 |
| W-reduced-center90 | 180 | 0.502855 ± 0.007258 | 0.089193 ± 0.002589 | 0.147384 ± 0.009732 | 0.191919 ± 0.008453 | 0.864583 ± 0.001802 |

同 seed 有向变化的三 seed 均值如下；正值表示改善。四项 error 为相对变化百分比，PCC 为绝对变化：

| 模型 | 对比 | RR | IBI | trajectory | global modulation | PCC |
|---|---:|---:|---:|---:|---:|---:|
| C201-center90 | 90→135 | +0.901% | +4.053% | +2.004% | +4.942% | +0.005081 |
| C201-center90 | 135→180 | +0.702% | −3.096% | −4.923% | +0.001% | −0.001479 |
| C201-center90 | 90→180 | +1.615% | +1.020% | −2.841% | +5.388% | +0.003603 |
| W-reduced-center90 | 90→135 | −0.286% | +1.952% | −3.754% | −1.346% | +0.006560 |
| W-reduced-center90 | 135→180 | −0.325% | −2.576% | −1.445% | −1.955% | −0.003480 |
| W-reduced-center90 | 90→180 | −0.623% | −0.616% | −5.193% | −3.344% | +0.003080 |

RR 的逐 seed 最低输入长度为：C201=`180/135/135 s`，W-reduced=`135/180/90 s`。C201 的 90→180 RR
平均改善 `1.615%`，但仅 `2/3` seeds 改善；W-reduced 平均恶化 `0.623%`，仅 `1/3` seed 改善。其余指标也呈混合方向，
因此没有稳定的单调上下文收益，也没有共同最优长度。该结果只支持 center-90 validation 上“效应弱且依赖表征”的描述，
不用于更改主实验、center-30、center-60 或 180→180 的结论。

冻结输出位于 `runs/paper_evidence_v1/center90_context/validation_summary/`。`summary_receipt.json` / `artifact_manifest.json`
SHA-256=`b8e3428ec4f43094ec7508b4897f9d2b07a54c702dc07749e3fd5fbb9f959886 / cc532fe3930861dd27ef4eb0f8efbb55381ffd26ad5cb88b7cf5f3c0839d47bc`。

## 7. 当前访问边界

P0–P3 已关闭并冻结。当前协议不包含独立测试集评价；任何后续数据访问需另立附件。

## 8. P0 实现回执

P0 已实现独立的 config/data/model/`Pi_90`/loss/metrics/cache/experiment/acceptance/CLI 空间。定向 CPU 验证确认：

- 三种输入切片逐点共享 `[4500,13500)` target，latent center 分别为 `[0,900)`、`[225,1125)`、
  `[450,1350)`；
- C201/W-reduced 均输出 9000 点 waveform 与 900 点 10-Hz waveform，参数数固定为
  `1,069,802 / 1,219,850`，共享 state 初始化一致；
- `Pi_90`、RR selector、五主指标、17 点 envelope、IBI coverage 与 prediction finite 合同闭合；
- 135 s 实际 scale mapping 为 49 个有限非降且无重复 centers；全网格与 `0.05–0.70 Hz` 名义区间最大频率误差分别约
  `0.035647 / 0.020064 Hz`；
- 135 s cache fixture 的 `[N,49,270]`、input-only、hash、row-ID 与不可覆盖合同通过，90/180 s 冻结 cache 可按既有
  manifest hash 打开；
- 最大臂 synthetic CPU 注入完成 forward/loss/backward；cache、训练和 GPU acceptance CLI 均有显式 confirmation gate。

P1/P2 冻结后新增 4 项 formal 定向测试，覆盖 18-config 矩阵、三份 cache hash、GPU acceptance receipt 与 formal/CLI
gate；center90 共 17 项定向测试通过，与全部 paper-evidence 回归合计 106 项通过。Formal gate 已开放，固定输入为
上述 18 项配置。
