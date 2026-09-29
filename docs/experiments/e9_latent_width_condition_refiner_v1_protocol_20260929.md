# E9：Latent Width and Condition Refiner Study

日期：2026-09-29。协议 ID：`e9-latent-width-condition-refiner-v1`。

状态：**P0/P1 已完成；P2 synthetic GPU acceptance 已完成并冻结；P3 formal runtime 已实现，等待在干净提交上生成并提交唯一 implementation lock。锁回载前18-cell formal train/validation 硬门控关闭。**

独立身份：

```text
config:              configs/e9_latent_width_condition_refiner_v1/experiment.yaml
model/control:       resp_train/paper_evidence/e9_latent_width_condition_refiner_v1*.py
cli:                 scripts/run_e9_latent_width_condition_refiner_v1.py
implementation lock: docs/experiments/e9_latent_width_condition_refiner_v1_implementation_lock_20260929.json
output:              runs/e9_latent_width_condition_refiner_v1/
validation summary: runs/e9_latent_width_condition_refiner_v1/validation_summary/
research-test:      runs/e9_latent_width_condition_refiner_v1/research_test/
```

P3 在 P2 成功后的干净提交上生成唯一 implementation lock，并由全部 formal cell 与后续汇总共同回载。E8 的协议、源码、`runs/e8_film_decoder_redesign_v1/`、checkpoint、summary 与锁保持冻结。

## 1. 科学问题与解释边界

本实验只研究全局潜在通道宽度 D 与 FiLM 前条件末端通道残差重组的关系：

1. D=96 时，历史 `H=65` 能否规整为 H=64，或压缩为 H=48；
2. D=64 时，条件末端应删除、保留等宽 H=64，还是保留 H=48 窄瓶颈。

条件末端固定为：

```text
c_out = c + W2(SiLU(W1(c)))
W1/W2: Conv1d(kernel_size=1)
```

D 是 Patch trunk、CWT context、FiLM gamma/beta 与调制后表示的共享通道数；H 只是在 FiLM 参数投影前的条件末端隐藏通道。H=64/48 是预注册候选，不具有理论最优含义。结论限于当前任务、数据、训练和评价合同。

不研究或改变：CWT 表示/尺度/频率/池化、Patch 长度/步长/mixer、Mamba 层数/双向方式、FiLM 位置/界限、post-FiLM refinement 的层数和 dilation、waveform decoder 类型、输出采样率/Fourier 插值、loss、optimizer、batch/AMP、数据准入/split、指标或 checkpoint selector。不同 D 不做参数填充，不增加 inert 参数。

## 2. 六个实验臂

### 2.1 E9-A：D=96

三个 arm 均从头训练，固定原 97-scale Morlet CWT、条件 dilation 1/2/4、六层 BiMamba2、`0.5*tanh` FiLM、post-FiLM refinement 与 pointwise waveform residual decoder。

- `e9a_d96_h65`：`96→65→96`，本轮同合同锚点；
- `e9a_d96_h64`：`96→64→96`；
- `e9a_d96_h48`：`96→48→96`。

禁止从任何 E8/W0 checkpoint 切片、复制、warm-start 或部分加载。

### 2.2 E9-B：D=64

共同基础模型合同：

- Patch：长度1800，adapter `16→64`，GN(8,64)，SiLU；
- BiMamba2：六层双向，`d_model=64,d_state=64,d_conv=4,expand=2,headdim=32,ngroups=1,chunk_size=256`；`64 % 32 = 0`，concat merge `128→64`；
- CWT：`1→48`、GN、SiLU、48-channel DWConv2D 不变，PWC `48→64`，scale mean 与插值1800不变，三个64-channel temporal blocks 使用 GN 8 groups；
- FiLM：最终 `64→128` 严格零初始化，split 64-channel gamma/beta，`0.5*tanh` 与公式不变；
- refinement：两个64-channel residual blocks，dilation 1/2；
- head：输入64，保持 `64→64,k5 → DW64,k5 → 64→32→1`；pointwise residual `32→32→1`，残差末层零初始化；10 Hz 生成后 Fourier 插值至100 Hz。

三个 arm：

- `e9b_d64_direct`：条件时间块直接进入 FiLM projection；
- `e9b_d64_h64`：`64→64→64`；
- `e9b_d64_h48`：`64→48→64`。

## 3. 初始化与配对

同一 seed、同一宽度族的共享模块使用完全相同的命名子 seed 并逐 tensor 相同。条件末端使用独立命名子 seed：

```text
e9_condition_refiner_d96_h64
e9_condition_refiner_d96_h48
e9_condition_refiner_d64_h64
e9_condition_refiner_d64_h48
```

FiLM 最终投影为零，因此同 seed、同一 D 家族的三个 arm 在 eval 模式下初始 waveform 逐元素相同。D64 与 D96 因 shape 不同，只共享 training seed、模块命名和初始化规则，不要求跨 D tensor 或初始 waveform 相同。

零初始化会分阶段打开梯度：首次 update 先更新 FiLM 最终投影，第二/第三次 update 才能验证 refiner 内层获得非零有限梯度。因此单次 backward 不构成模块活性验收。

## 4. 结构、参数与计算表

模型文件大小是当前 PyTorch 环境对原生 `state_dict` 的序列化字节数，仅作工程基线；正式 P2 必须在冻结环境重新记录。`condition MACs` 是 refiner 两个逐点卷积；`declared covered MACs` 覆盖 E9 明确受管的 Patch adapter、六个 Mamba merge linear、CWT Conv/PWC、五个 D-channel residual blocks、condition refiner、FiLM projection、waveform head 与 pointwise residual。它排除 Patch encoder、Mamba 内部 selective-scan/conv/projection、norm/activation 和 Fourier interpolation，不能称为完整模型 FLOPs。

| Arm | D | 条件末端 | Trainable params | state_dict bytes | MiB | condition MACs | declared covered MACs |
|---|---:|---|---:|---:|---:|---:|---:|
| `e9a_d96_h65` | 96 | 96→65→96 | 1,219,850 | 4,954,427 | 4.725 | 22,464,000 | 856,224,000 |
| `e9a_d96_h64` | 96 | 96→64→96 | 1,219,658 | 4,953,659 | 4.724 | 22,118,400 | 855,878,400 |
| `e9a_d96_h48` | 96 | 96→48→96 | 1,216,586 | 4,941,371 | 4.712 | 16,588,800 | 850,348,800 |
| `e9b_d64_direct` | 64 | identity | 592,706 | 2,441,217 | 2.328 | 0 | 445,985,280 |
| `e9b_d64_h64` | 64 | 64→64→64 | 600,962 | 2,475,323 | 2.361 | 14,745,600 | 460,730,880 |
| `e9b_d64_h48` | 64 | 64→48→64 | 598,914 | 2,467,131 | 2.353 | 11,059,200 | 457,044,480 |

E8 冻结历史参照只作上下文，不填充 E9 结果：

| E8 historical arm | Params | E8 factor-covered MACs | 角色 |
|---|---:|---:|---|
| `e8_direct_pointwise` | 1,207,274 | 1,900,800 | 删除条件末端 |
| `e8_fill65_pointwise` | 1,219,850 | 24,364,800 | 历史 W0；E9 本轮仍须训练 H65 锚点 |
| `e8_res96_pointwise` | 1,225,802 | 35,078,400 | 历史等宽重组 |
| `e8_res192_pointwise` | 1,244,234 | 68,256,000 | 历史扩展重组 |

E8 的 factor-covered MAC 范围与 E9 的 declared covered MAC 范围不同，不直接相减。历史结果来自另一轮随机训练，不替代 E9-A 同轮配对。

## 5. 训练合同

- seeds：`20260811 / 20260812 / 20260813`；六臂均从头训练；
- AdamW、原 weight decay/gradient clipping/step-exact warm-up cosine；
- physical/effective batch 128，accumulation 1，BF16，完整尾 batch；
- 每 epoch 80 updates，最多80 epochs，planned 6,400 updates；
- early stopping：`min_epoch=30 / patience=15 / min_delta=0`，wait 从 epoch 1 累积；
- selector：完整 validation Local RR 严格最小，并列取最早 epoch；
- loss：`L_sync + 0.25 L_effort`；
- 五主指标：Whole RR MAE、Local RR MAE、envelope trajectory MAE、global envelope modulation error、signed PCC。

矩阵为 E9-A `3×3=9`、E9-B `3×3=9`，共18次。不得根据 partial seed、速度、显存或中途 validation 删除剩余 cell。

## 6. CPU/synthetic 验收计划与当前结果

当前已执行：

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/test_e9_latent_width_condition_refiner_v1.py -q
# 22 passed

PYTHONPATH=. ./.venv/bin/python \
  scripts/run_e9_latent_width_condition_refiner_v1.py check-p1
```

覆盖：六臂 spec/derived config、精确结构与参数、optimizer 全覆盖、state_dict strict round-trip、state finite、D64 逐模块 shape、同宽族共享 tensor 初始化、同宽族初始 waveform 逐元素相同、四种新增 refiner 的三次真实 AdamW update 梯度开启、显式 shape/nonfinite failure、18-cell formal gate 关闭。初始函数测试为 disposable CPU tensor，并把昂贵的公共 Mamba blocks 替换为 Identity；它只检验 FiLM/refiner 等价性，不能替代 P2 原生 Mamba forward/backward。

`96` 全仓审计见 `e9_latent_width_condition_refiner_v1_96_hardcode_audit_20260929.md`。审计结论是建立 E9 专属 D64 路径，不修改冻结的通用 CRD、E4–E8、W0 或 RTM 文件。

## 7. Synthetic GPU acceptance

P2 只使用确定性 synthetic input/target/W tensor，不读 dataset index、真实 waveform、W cache、历史 checkpoint 或 test。

1. 六臂 × 三 seeds 的18个 batch-1 cell，每项三次原生 loss/optimizer update；要求 output/loss/model/optimizer finite，所有活跃参数被 optimizer 精确覆盖，FiLM 和适用 refiner 参数真实变化，第三步 refiner 内层梯度非零有限。
2. 六臂逐项 native batch-1 forward/backward shape/finite，并登记 params、state_dict bytes、condition MACs、declared covered MACs 与 peak allocated/reserved。
3. 最大资源 arm 固定为 `e9a_d96_h65`；batch=128、三次原生 update，branch checkpoint chunk=8。预注册 `peak reserved / device total ≤ 0.85`；P3 只能接受相同软件栈且总显存不低于验收设备的运行环境。
4. 使用排他、不可覆盖 attempt；成功后以 freeze receipt/manifest 固定，失败 lifecycle 原位保留。

冻结产物：

```text
runs/e9_latent_width_condition_refiner_v1/gpu_acceptance/
gpu_acceptance_dfd2efe9fa1b_20260929T052057Z_bd9c52bd4c27
```

- Engineering identity：`dfd2efe9fa1be8834c607b157f960f48e779af7348b57ce6bb45697212a7fb48`；
- Manifest SHA-256：`37bef5f86c28fe239bf0ea0d65eabea544c05afd008ba1e768016b0141bdbeaa`，24个受管文件；
- 18/18 batch-1 cell 均完成三次 update，最后一步所有适用 factor 梯度非零有限，参数真实变化；
- 最大资源 arm `e9a_d96_h65` 完成 batch-128 三次 update，peak allocated=`9,204,315,648` bytes，peak reserved=`10,622,074,880` bytes，reserved fraction=`0.635653`；
- `passed=true`，freeze receipt 与全部 manifest 文件已逐项回载验证。

执行命令记录为：

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py gpu-acceptance --device cuda:0
```

## 8. Implementation lock 与 formal 门控

P2 完成后，在干净 Git commit 上生成唯一 implementation lock。锁至少固定：

- 协议/spec/六臂合同、18-cell 矩阵与所有 derived baseline configs；
- E9 model/control/engineering/formal/CLI/tests 及依赖的 CRD/training/loss/metrics 源码逐文件 SHA-256；
- W0 train/validation 来源锁及数据/cache/row identity；
- P2 acceptance freeze receipt、manifest、环境和安全阈值；
- 输出根、selector、early stop、loss/optimizer/batch/BF16 与 planned updates。

正式入口必须回载并逐项核验该锁；相同 `(lock,arm,seed)` 已成功 cell 拒绝重跑，失败现场保留。当前只待生成并提交 implementation lock；锁存在且工作树干净后 formal 才开放。

在包含 P3 runtime 的干净提交上生成锁：

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py prepare-formal-lock
git add docs/experiments/e9_latent_width_condition_refiner_v1_implementation_lock_20260929.json
git commit -m '冻结E9正式实验实现'
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py check-formal-lock
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py matrix-status
```

训练前矩阵必须为 `pending=18 / running=0 / failed=0 / completed=0`。

## 9. 18次 formal 计划命令

以下命令固定18-cell矩阵；提交 implementation lock 并通过回载核验后即可运行。

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h65 --seed 20260811 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h64 --seed 20260811 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h48 --seed 20260811 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_direct --seed 20260811 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_h64 --seed 20260811 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_h48 --seed 20260811 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h65 --seed 20260812 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h64 --seed 20260812 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h48 --seed 20260812 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_direct --seed 20260812 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_h64 --seed 20260812 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_h48 --seed 20260812 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h65 --seed 20260813 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h64 --seed 20260813 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9a_d96_h48 --seed 20260813 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_direct --seed 20260813 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_h64 --seed 20260813 --device cuda:0 --confirm-formal-training
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal --arm e9b_d64_h48 --seed 20260813 --device cuda:0 --confirm-formal-training
```

机器可读计划：

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_e9_latent_width_condition_refiner_v1.py formal-plan
```

## 10. 汇总、计划对比与决策规则

18/18 唯一成功 cell 完成后只汇总一次，不构造跨指标加权总分。输出模板见 `e9_latent_width_condition_refiner_v1_validation_summary_template_20260929.md`。

E9-A 固定对比：`H64−H65`、`H48−H65`、`H48−H64`。E9-B 固定对比：`D64-H64−D64-direct`、`D64-H48−D64-direct`、`D64-H48−D64-H64`。每个 D64 arm 另与 E9 同轮 D96-H65、D96-H64描述性比较，同时报告质量、参数与 covered MAC；不得把 D 改变造成的容量—效率联合变化解释成单一 refiner 效应。

材料性阈值：四项 error 相对0.5%，PCC 绝对0.002。只有至少一项材料性改善且其余主属性均无材料性退化，才称 tolerance-aware Pareto 改善。报告三 seed mean ± sample SD、paired-seed方向、window-direct、subject-macro、Local RR median/P90/P95、`>2/>5 bpm`、参数与 covered MAC。Synthetic acceptance 的显存记录只作运行安全回执。

预注册判断：

1. E9-A：H64 对 H65 全部在容差内且参数/MAC更低，优先更规整的 H64；H48 只有形成 Pareto 改善或质量保护线内的明确效率收益才保留；RR/PCC 受损时不因更小替换；不按单项最好选择。
2. E9-B：direct 对 H64/H48 无材料性退化则支持删除；H64 或 H48 形成 Pareto 改善才支持对应映射；指标分化则报告节律—形态—效率取舍；D64 组内最佳只有对 D96 锚点满足质量保护线时才可建议替换，否则仅为效率候选。

## 11. Research-test 门控

必须先完成全部 train/validation、一次性汇总、checkpoint 选择与 validation 决策冻结，再生成18项 allowlist。模板见 `e9_latent_width_condition_refiner_v1_research_test_templates_20260929.md`。

若继续使用现有 research-test：

- 全部18个 validation-selected checkpoint 在任何 test array 访问前固定；
- 一次性评价完整矩阵，test 不参与 checkpoint、阈值、结构或候选选择；
- 明确标记为 **reused research/development evidence**；
- 看到 test 后不得新增 D/H 候选；
- E8 已观察到 validation→test 反转，且同一 test 已参与多轮开发，因此该结果不能表述为独立 held-out 确认。它可以描述跨 split 稳定性，不能单独证明主模型泛化替换。

## 12. 结论措辞模板与证据边界

允许措辞：

> 在固定数据、训练和评价合同下，E9-A 的 H={候选} 相对同轮 H=65 在五主属性上表现为 {容差内等效 / tolerance-aware Pareto改善 / 明确属性交换}，并减少 {参数/MAC/实测资源}；该结果支持 {规整/保留候选/不替换}，不证明该宽度具有理论最优性。

> 在 D=64 模型族内，{direct/H64/H48} 相对其余候选表现为 {结果}。这一结论描述64维全局表示下是否需要条件末端重组；它不外推到其他潜在宽度。

> D=64 相对同轮 D=96 锚点呈现 {质量变化} 与 {效率变化}。由于全局宽度同时改变容量和计算量，该差异不能归因于单一条件瓶颈效应。

> Research-test 结果来自既有、反复参与研究开发的 split，仅作为 reused research/development evidence；它不构成独立外部确认，也不用于追加候选或重选结构。

禁止措辞：理论最优、普遍最优、无损压缩（除非所有预注册属性及尾部/subject证据均在容差内且限定当前合同）、独立测试确认、仅凭参数更少宣称速度更快、仅凭单项最优宣称全面赢家。

## 13. 需要修改/新增的文件清单

本阶段新增：

- `configs/e9_latent_width_condition_refiner_v1/experiment.yaml`；
- `resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_model.py`；
- `resp_train/paper_evidence/e9_latent_width_condition_refiner_v1.py`；
- `resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_engineering.py`；
- `resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_formal.py`；
- `scripts/run_e9_latent_width_condition_refiner_v1.py`；
- `tests/test_e9_latent_width_condition_refiner_v1.py`；
- 本协议、硬编码审计、validation 模板与 research-test 模板。

P3 后续产物为唯一 implementation lock；formal 完成后再建立 validation summary runtime。无需修改 `resp_train/crd/{tf_v1_model,frontends,blocks,model}.py` 或 E8 文件。
