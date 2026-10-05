# CWT 时频信息与频带作用 v1：工程接手与最小对照设计

日期：2026-09-30。拟议实验 ID：`cwt-time-frequency-v1-20260930`。

状态（2026-10-01更新）：**独立实现及定向CPU检查已完成；用户已授权后台进行合成校准和GPU工程验收。** 实现、检查记录与运行范围见 [执行协议](cwt_time_frequency_v1_protocol_20261001.md)。实际运行结果由attempt回执记录；真实数据和正式训练尚未开放。

## 1. 来源与工作区

- 论文侧计划：`/mnt/disk_code/marques/paper/resp_rec/paper_rewriting_output/CWT时频信息与频带作用_实验交接计划_20260930.md`。
- 新 worktree：`/home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction`。
- 分支：`codex/cwt-time-frequency-v1`；起点：`f8e10854c2123c5d56317bf55d139d9d723490b7`。
- 原工作区：`/mnt/disk_code/marques/resp_reconstruction`。
- 原工作区尚未提交的 PATCH/APOR、网格×解码和激活系列源码、配置、测试及相关说明已按原路径复制，逐文件核对 SHA-256，见 [工作区来源清单](cwt_time_frequency_v1_worktree_sources_20260930.json)。该清单记录迁移来源，不承担实验冻结功能。
- 原工作区的修改、历史 runs、cache、checkpoint 和数据保持原位置。新 worktree 未链接整个历史 runs；新输出应使用本 worktree 的 `runs/cwt_time_frequency_v1/`，历史输入按明确绝对路径只读引用。

共享解释器已验证可用：`/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python`。在新 worktree 使用 `PYTHONPATH=.`，确认导入的 `resp_train` 来自新 worktree。依赖元数据为 torch 2.12.0+cu130、NumPy 1.26.4、pandas 3.0.3、OmegaConf 2.3.0、ssqueezepy 0.6.6、mamba-ssm 2.3.2.post1、causal-conv1d 1.6.2.post1；本次未验证 CUDA kernel。

## 2. 基准：采用原 W0

主矩阵采用 `CRDTfV1Model("crd_tf102_w", seed)`，与论文侧计划及历史 CWT/GN 分析一致。以 `B` 标识本轮同期 W0，避免与 APOR 的 A0/N0/U0 重名。

| 项目 | 同期基准 B |
|---|---|
| 时域路径 | PatchTokenFrontend 对齐到 1800 个时间点，D96、六层双向 Mamba2 |
| 条件输入 | Morlet μ=13.4、12 voices/octave、97 尺度、360 帧 |
| W 编码 | Conv2d(5,3)、GN、depthwise(3,3)、1×1 projection、scale mean |
| 条件末端 | 对齐至 1800 点、三个 temporal residual blocks、H65 pointwise fill |
| FiLM | `z' = z * (1 + 0.5*tanh(gamma_raw)) + 0.5*tanh(beta_raw)` |
| 输出 | 原 refinement、C201 residual readout、Fourier 插值及任务 Pi |
| 参数量 | 历史静态审计为 1,219,850；新实现验收须再次核对 |

实现入口：`resp_train/crd/tf_v1_model.py`、`resp_train/crd/model.py`。M0（PATCH/TM0/REF0）、完整 APOR、E9 D96/H64 均有不同模型合同，不能作为 B 的已有训练实例。APOR 的训练调度、参照复用审计和结果组织可作为工程参考。

原 W0 的三个 temporal residual blocks 在时间对齐后执行；不能将 APOR 的浅层编码及五位置条件采样接入本轮，仍声称只比较 CWT 表示。

## 3. 最小充分训练矩阵

这里的“最小”指保留交接计划的问题和建议水平，消除跨子矩阵的重复训练；不将 μ、voices、时间池化、频率覆盖做全因素交叉。每配置使用三个 seed：20260811、20260812、20260813。

| 子矩阵 | 配置 | 唯一新增配置 | 新增训练 cell | 共享与比较 |
|---|---|---:|---:|---|
| TF-S2 | μ∈{6,13.4,20} × voices∈{4,8,12,24}；完整至 8 Hz；0.5 s | 12 | 36 | 包含 B；保留 μ×voices 交互比较 |
| TF-S3 | 基准 μ/voices/频带；池化 0.25、0.5、1 s | 2 | 6 | 0.5 s 直接共享 B 的三个 cell |
| TF-S4 | 基准 μ/voices/池化；上限 0.8、2、4、8、12、20 Hz | 5 | 15 | 8 Hz 直接共享 B；按频带相邻上限及对 B 配对 |
| TF-S5 | L、H、L+H；L≤0.8 Hz，H=(0.8,8] Hz | 1 | 3 | L 共享 S4 的 0.8-Hz 臂，L+H 共享 B；只新增 H |
| 合计 | 20 个唯一配置 | 20 | **60** | 其中同期 B 为 3 个 cell |

规划 cell 清单见 [矩阵 CSV](cwt_time_frequency_v1_planned_cells_20260930.csv)。频率字段是名义定义；实际数组和尺度数由 TF-S1 固定。TF-S2～S4 合计 57 次训练，加入 TF-S5 的 H 后为 60 次。

TF-S1 不训练；TF-S6/S7 最小设计采用本轮 B 的三个 validation-selected checkpoints，分别做配对干预和案例导出，不增加训练 cell。需要研究其他参数配置上的机制时，另列评价预算和 checkpoint 清单。

### 3.1 该矩阵支持的比较

- `L+H − L`：宽带时域输入下，加入高频 CWT 条件的增量变化。
- `L+H − H`：宽带时域输入下，加入低频 CWT 条件的增量变化。
- `H − L`：两种条件频段的完整表示—编码配置比较。
- S2 的 12 个点：实际小波分辨率、尺度密度及其交互的任务响应。
- S3、S4：以共同 B 为交点的时间压缩和频率覆盖响应。

若要回答“H 相对完全不用 CWT 是否有收益”，需要同期无 W/FiLM 的 C201 对照，另加 1 配置×3 seeds，共 63 次训练。当前 60-cell 矩阵不估计这个效应，也不据三臂误差计算超加性的互补收益；该扩展应由研究目标决定。

## 4. 结果复用判定

| 来源 | 当前判定 | 理由 |
|---|---|---|
| 本轮 B | 跨 S2/S3/S4/S5 共享三个 cell | 相同配置、同一运行实例，只建立结果引用 |
| 本轮 S4 的 L | S5 共享三个 cell | 必须共用同一个实际频率/尺度数组、缓存和模型定义 |
| 历史原 W0 | 作为历史参照；不抵扣同期 B | 计划要求同期基准；已有同 seed 数值轨迹与选点差异记录 |
| E7 S0-MEAN、E8 fill65/pointwise、三因素 PATCH/TM3/REF2 | 历史同结构参照 | 早停和实际数值执行记录不同；不能挑更好的一组拼接新矩阵 |
| 历史 W1/W2 | 不复用新 L/H 的训练结果 | 旧实现保持 97 槽位并将其他频带置零；新实验使用原生连续频带输入，卷积边界、GN 和 scale mean 均不同 |
| 历史 W3 | 不复用 S2 训练结果 | 旧条件为从 12V 缓存取偶数尺度的 6V；本计划水平为 4/8/12/24V |
| M0、A0/N0/U0、E9 | 不复用主矩阵训练结果 | 时域路径、条件末端或读出合同不同 |
| 历史 W0 的 97×360 cache | 允许作为待审计的输入来源 | 需满足相同数据键、row 身份、完整变换定义和字节身份；缓存复用不等于训练结果复用 |
| 历史 R3 GN/配对导出 | 复用方法与组件 | 本轮 checkpoint、频带和偏移重新绑定；历史 `[73:97]` 不能充当精确 (2,8] Hz 定义 |

当前可直接计入本轮的历史训练 cell 为 **0**。这是按本计划同期比较定义作出的判断，不表示所有旧 checkpoint 失效。历史结论、产物和冻结表保持其原有范围。

## 5. 训练合同

本设计默认沿用论文侧计划的原 W0 **固定 80 epochs，early stopping disabled**。已实际解析 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 及父配置，确认该设置；APOR runner 会额外开启 30/15/0 早停，不可原样照用。

- train/validation：10141/2675 窗口，32/7 个 samp_id；180 s、100 Hz；沿用资格、划分、BCG/THO 数据键。
- sample seed：train=20260610、validation=20260611；三个训练 seed 如上。
- loss：原 `L_sync + 0.25 L_effort`；Pi 与核心指标函数保持原定义。
- physical/effective batch=128，accumulation=1；BF16；drop_last=false。
- AdamW：LR 3e-4→3e-5，betas=(0.9,0.999)，eps=1e-8，weight decay=1e-4，warmup=0.05，gradient clip=1。
- 80 updates/epoch、6400 planned updates/cell；完整 validation Local RR 严格最小，平局保留最早 checkpoint。
- 公共模块按同 seed 逐 tensor 初始化一致；配置维度扩展不引入新模型参数时，预期全部 20 臂参数量相同，但仍须实际核验。

如为预算改用早停，应在正式运行前统一修订全矩阵训练合同及预算。不能沿用 APOR 的默认早停后仍将训练合同标为原 W0。

## 6. 原生表示与缓存：必须先落实的实现细节

1. **变换次序：**历史 W 实际计算 `mean_pool(log1p(abs(CWT(x))))`，不是先平均幅度再 log。0.25/0.5/1 s 对应 25/50/100 点池化，帧数为 720/360/180。记录帧中心 `(p*j+(p-1)/2)/100` 秒及完整池化行为。
2. **网格锚点：**原基准名义网格为 0.03125–8 Hz，不能静默改成从 0.03 Hz 起步。S2 在八个 octave 上的名义尺度数为 33/65/97/193；实际中心频率、重复映射和响应均需校准。历史 97-scale 映射含重复中心；不能自动去重改变原基准。
3. **频率覆盖：**建议采用相同 μ/voices 的共同对数网格及嵌套频率选择，避免移动上限时平移所有低频尺度。基准 scales 与实际频率须原样保留。上限 12/20 Hz 的扩展 scales 如何映射、边界按名义还是实际频率处理，由 TF-S1 写入执行协议；实际边界数组是最终身份。
4. **L/H 的一致性：**用实际频率数组按 `≤0.8` / `(0.8,8]` 精确分割，S4 的 L 与 S5 的 L 必须是同一配置。固定后报告真实最大/最小中心频率，不将名义端点当作测得中心。母网格子集符合原生频带输入要求，不补零恢复成 97 槽位。
5. **缓存派生：**只有变换 scales、padding、库调用和输出在既定容差内等价时，才能从母缓存裁剪尺度；不能只凭中心频率相同判定。0.25 s 不能从 0.5 s 恢复；0.5→1 s 的再平均须检查浮点累加顺序。统一保存派生来源、实际数组与校验结果。
6. **代码接点：**`tf_v1_features.py` 固定 μ、8-Hz 锚点、pool50；`CwtBranch` 固定 49/97 尺度和 360 帧；`tf_v1_cache.py` 固定 cache shape；`TfV1CacheReader` 固定 manifest hash；`crd/config.py` 还固定历史 cache 路径。因此只改模型 shape 检查不足以执行新矩阵。
7. **扩展方式：**为本系列增加配置驱动的变换/缓存适配器与实验入口，复用原数据加载、trainer、optimizer、loss、metrics。原入口维持其冻结合同。新输入继续保留完整 shape、finite、row 与 hash 检查。
8. **比较解释：**尺度卷积槽位核不变，物理覆盖、GN 统计、scale mean 和时间插值行为可随输入改变。这些结果是完整表示—编码配置的效应；单独归因于“频率信息量”需要进一步控制。

建议记录 conv_in 五尺度覆盖和经过 depthwise 后七尺度有效覆盖的低/中/高位置。时间核覆盖分别报告原生帧轴与对齐后轴，边界填充单列。20-Hz 端点及压缩前信号可用性仍待 TF-S1 核查。

## 7. TF-S6/S7 的最小机制矩阵

建议预设两个干预区域：H=(0.8,8] Hz，以及 H2=(2,8] Hz；采用实际频率布尔索引。最小条件为 FULL、WINDOW_MEAN、三个共同循环 SHIFT，分别使用 NAT 与 ALL_W_GN_FIXED。两个区域共享 FULL，故为 `2 + 2区域×4干预×2模式 = 18` 个条件/checkpoint，即三 seed 共 54 个条件评价。该数目是建议预算，需在正式协议固定偏移后落实。

沿用现有按有序 row、PCG64 固定生成三个偏移并保存数组的方式，偏移范围拟为 30–150 s；频带、seed 之间使用同一 row 的偏移。多偏移分别保留且汇总，不按干预效果选取。FULL_FIXED 必须重放 FULL_NAT；固定四层 W-GN 的统计全部来自同窗口 FULL，保留原生 dtype 路径与独立数值检验。

WINDOW_MEAN 检查移除时间变化的影响；共同 SHIFT 检查与 BCG/其他尺度/参考的对应依赖。两者**不能单独区分高频整体幅度曲线与跨尺度相对结构的贡献**。若论文需要后一个结论，按交接计划另立完整表示与单曲线表示的重训练比较及编码合同；本轮不预先给出频谱或相位机制结论。

S7 复用 `w0_cwt_film_behavior.py` 的 capture 及 `w0_qualitative_intervention.py` 的同批保护检查。完整保存预选案例的 `gamma_raw/beta_raw/g/b/Z/Z_prime`、修正量、配对输出、指标曲线、实际 CWT 与坐标；全样本保存紧凑指标和身份。案例选择规则先于干预结果固定。历史 R3 入口绑定固定 shape、旧 checkpoint 和旧频带，应提取组件而非直接重跑。

## 8. 运行工程、资源与验收顺序

可借鉴 `patch_apor_v1.py` 的 session/cell 生命周期和交错分片，`apor_grid_decoder_v2_runtime.py`、`apor_activation_v1_runtime.py` 的历史参照合同校验，以及对应 research-test 入口的 allowlist 和三视图汇总。新 runtime 需落实以下差异：

- 公共 validation reference 由父进程单次准备，worker 只读核验；共享写入竞争不能作为训练失败。必要的收尾恢复须有独立“只导出”阶段与 checkpoint 身份，不把 `--retry-failed` 描述成断点续训。
- 源码来源检查基于明确依赖清单及其闭包，历史参照单独核验其冻结快照；新增无关文件不应自动使旧证据失效，实际依赖变化必须失败或使用新身份。
- 多卡是独立 cell 进程，固定交错分片，各 worker OMP/MKL/OpenBLAS=4；部署环境和设备记录一致。先核对 FP32/BF16 基准等价，再验收不同尺度/帧数的 forward/loss/backward；零初始化 FiLM 需多步更新检查条件内部梯度。
- 保留 selected validation prediction、row IDs、公共 reference 及诊断标量。正式 summary 包含 per_seed、across_seed、paired_delta、paired_summary、per_subject、subject_macro、local_rr_tail、denominators。
- 四正文指标与包络调制诊断保留现行定义；配对差统一 error=候选−参照，PCC 下降=参照−候选。seed 和重叠窗口不增加独立受试者数。后续 research-test 同时保存 full/exclude670/subject670，不能用其中一视图重选 checkpoint。

固定 80 epochs 的 60-cell 上限为 384,000 optimizer updates、48,676,800 次训练窗口呈现、12,840,000 次逐 epoch validation 窗口评价，另计收尾和干预评价。墙钟时间和显存未测量，不能套用 APOR 的时间。

float32 条件缓存大小公式为 `N_windows × N_scales × N_frames × 4 bytes`；基准 train+validation 的 W 数组约 1.67 GiB。每 cell 保存 2675×18000 的 validation prediction 约 184 MiB，60 个 cell 约 10.76 GiB，公共 reference 另存一次。checkpoint/history、其他缓存、失败产物和 S7 特征另计；应在全量执行前依据已冻结 shapes 列出磁盘预算。

建议顺序：实现新配置/变换/缓存适配与 synthetic 定向检查 → TF-S1 校准及 train/validation 信号检查 → 固定实际矩阵与资源预算 → 用户运行 GPU 工程验收和 formal train/validation → 完整 validation 汇总、固定 S6/S7 定义和 checkpoint allowlist → 用户开放专项 research-test → 收尾。

## 9. 2026-09-30接手时的完成状态

已完成：新 worktree/分支创建；36 个未提交工程参考文件的无覆盖复制和哈希核验；基准配置解析、共享解释器/依赖元数据检查；源码与历史合同差异核对；20 配置/60 cell 规划清单及本记录。

尚未执行：新模型/缓存/runtime 实现、CWT 合成校准、真实数据读取与分析、GPU 验收、训练、benchmark、checkpoint 推理及 test 访问。当前不提供虚构的新实验运行命令。

可复用的只读配置检查方式：

```bash
cd /home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -c \
  'from resp_train.crd.config import load_crd_config; c=load_crd_config("configs/crd_tf_v1/crd_tf102_w_formal.yaml"); print(c.training)'
```

该基础 YAML 仍含指向原工作区的历史 output_root；后续新入口必须解析并覆盖为本系列独立路径，再开放训练。复制过来的历史 APOR CLI 仍绑定旧实验身份，不能用作本轮 CWT 命令。
