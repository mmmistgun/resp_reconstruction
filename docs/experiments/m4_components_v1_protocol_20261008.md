# 原始 P1-M4 组件消融 train/validation 协议

协议 ID：`m4-components-v1-20261008`。本轮实现用户确认的八臂 × seeds 20260811/20260812/20260813；B0/B1 使用原 P1 六个冻结 cell，新增 B2–B7 共 18 个训练 cell。当前为实现与合成 CPU 验证阶段；正式 GPU 验收及 train/validation 由用户执行。

## 研究问题与来源

研究原始 P1-M4 的完整 CWT 条件、读取模块、频带表示、显式 BiMamba2 及波形 patch pooling 的增量。原始 M4 使用 attention 输出投影和零初始化 residual-add 投影，保留两级 affine 参数化。M4-v2 的单投影及初始化合同不同，结果只作为历史开发证据，不代替本轮基线。

历史入口：[P1 协议](p1_components_v1_protocol_20261006.md)、[P1 validation](p1_components_v1_validation_results_20261007.md)、[P1 research-test](p1_components_research_test_results_20261007.md)、[M4-v2 结果](m4_residual_v2_research_test_results_20261008.md)。已有 research-test 曾影响设计，本轮为开发迭代。当前入口只开放 train/validation，后续 research-test 必须另立匹配协议并取得当次授权；重复使用的 test 仍属于 research/development evidence。

## 固定矩阵

| Arm | 名称 | 相对 B0 的变量 | 执行身份 |
|---|---|---|---|
| B0 | M4-Full | 原始 Full CWT attention + 双投影 residual-add | 引用 P1 M4 |
| B1 | NoTF | 不构造或读取 CWT 条件分支，u=z | 引用 P1 M1 |
| B2 | MeanTF-Add | 编码后 F×2 均值 → C→D affine → 零初始化 D→D affine → 加到 z | 新训练 |
| B3 | ContentAttention-Add | 移除频率/相对时间坐标 bias 网络，保留物理窗口和 Q/K/V/output | 新训练 |
| B4 | H-Add | 使用 (0.8,8] Hz 的 41 scales | 新训练 |
| B5 | L-Add | 使用 ≤0.8 Hz 的 56 scales | 新训练 |
| B6 | NoMamba-Add | BiMamba2 替换为 Identity | 新训练 |
| B7 | UniformPatchPool-Add | 波形 patch attention pooling 改为均匀均值，保留位置表示和输出映射 | 新训练 |

共有结构：100 Hz、180 秒波形，continuous CNN，P=100、hop=50、N=359、D=96；适用时 BiMamba2×6；token 解码 100 点，positive Hann（epsilon=0.001）归一化 overlap-add。完整 W 有 97 scales、360 时间格，每个 token 读取两个对齐格（相对中心 ±0.25 秒）。沿用冻结 cache、实际频率坐标及原 GroupNorm 条件编码器。

所有同名共有模块使用原 `p1_components.*` 种子命名和构造顺序。B0/B1 的 state dict 和原 P1 M4/M1 一致；B3 在原 attention 初始化完成后替换坐标网络为无参数零 bias，以保持 Q/K/V/output 完全一致。B7 沿用原 UniformPatchEncoder 对 score 模块的处理方式。模型构造不消耗外部 RNG，继续使用原训练 seed 与采样顺序合同。

B2 同时替换 Q/K/V 读取参数化，比较定义为读取模块的增量。B3 只检验显式 bias，不检验物理窗口对齐。B4/B5 裁剪支持域同时改变 GroupNorm 统计域和卷积边界邻域，属于 representation variant。B6 保留 CWT 时间支持、整窗 GroupNorm 和 CNN 邻域，检验的是显式 BiMamba2 增量。

正式 Mamba2 在 CPU 上构造核验的可训练参数量：

| Arm | 参数量 |
|---|---:|
| B0 | 1,056,357 |
| B1 | 1,020,125 |
| B2 | 1,034,365 |
| B3 | 1,056,157 |
| B4 | 1,056,357 |
| B5 | 1,056,357 |
| B6 | 104,973 |
| B7 | 1,056,292 |

## 历史参照复用

固定来源为 `runs/p1_components_v1/formal_v1_20261006/`，session.json SHA-256：`34d0c62cb64adae16fca4db512687d28bb96e74e89857c049a506d66c4e71fb6`。

`audit-references` 校验原 session、源码快照和当前依赖源码身份、六个 cell 的完整 artifacts（含 checkpoint、配置、validation CSV 和 history）、科学配置、参数量；不运行模型或读取原始数据。`prepare` 将参照路径、完成记录及环境哈希写入新 session。后续入口继续核验参照身份，汇总时只在内存将 M4/M1 映射为 B0/B1，原始文件不复制、不修改。默认 plan/run 仅包含 18 个新增 cell，禁止 B0/B1 进入训练。

运行时要求训练软件版本与 GPU 型号与历史参照一致。若来源、科学配置或环境不符，显式失败，不自动以新训练替代历史参照。需要重新建立完整矩阵时另行修订协议并使用新输出身份。P1-M0 的 FiLM 结果可在讨论中作为既有融合对照，本轮汇总仅包含上述八臂。

## 数据、优化与选点

沿用 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 及 P1 数据合同：train=10141 窗口/32 人，validation=2675 窗口/7 人；sample seeds、target、loss、metrics 与 split 不变。入口核验冻结 index、样本 ID 内容、分母与受试者隔离。非有限输入、cache、模型、预测及关键指标显式失败。

microbatch=32、累积=4、有效 batch=128、BF16，每 epoch 80 次 optimizer 更新。各累积组按实际 eligible 数归一化 loss 分量并保留尾组。AdamW、lr=3e-4→3e-5、5% update warmup+cosine、weight decay=1e-4、clip=1，loss=sync+0.25×effort。最多 80 epoch/6400 updates；early stopping min_epoch=30、patience=15、min_delta=0。每 epoch 完整 validation，按 Local RR MAE 严格最小选 checkpoint，并列保留最早 epoch。

## 预设汇总

只在完整 24-cell（18 新训练 + 6 引用）齐备后汇总。五项主指标为 Whole RR、Local RR、envelope trajectory、global envelope modulation 与 lag-aware signed PCC。报告逐 seed 和三 seed mean/sample SD、同 seed 配对差、逐受试者及 subject-macro、Local RR tail、资格分母及参数量。不同 seed 共用同一 validation 人群，该集用于选点，证据为开发阶段描述性比较。

主配对为 B1–B7 各对 B0；递增配对为 B2−B1（均匀 TF 条件）、B3−B2（内容读取及其参数化）、B0−B3（显式坐标）。delta=候选−参照，improvement 对误差取负 delta、对 PCC 取正 delta。Full/H/L 按主配对共同解读，不以单项均值或 partial seed 改写矩阵。保留形态与节律的指标取舍，不将最优 Whole RR 均值等同于整体最优。

## 执行

进入 TF-Mamba 工作树：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
```

只读审计与准备各执行一次：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_components.py audit-references
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_components.py prepare --session runs/m4_components_v1/formal_v1_20261008
```

分别在两个终端执行，每张 GPU 串行 9 个新 cell：

```bash
bash scripts/run_m4_components_shard.sh runs/m4_components_v1/formal_v1_20261008 0 0
```

```bash
bash scripts/run_m4_components_shard.sh runs/m4_components_v1/formal_v1_20261008 1 1
```

恢复时末尾加 `--resume`。run 在训练前执行对应臂的正式 GPU 合成验收：32×4 累积、正式 task loss/optimizer、两次更新及 validation loss；检查上游梯度/参数更新，并记录工程耗时与 allocated/reserved 显存。该耗时含初次内核成本，不作为稳态 benchmark。验收仅在同 session、同设备环境及正式批量下复用。

保存完整 epoch 的模型、optimizer、调度 update index、早停与 Python/NumPy/CPU/CUDA RNG；失败 attempt 原地保留，完成产物校验后复用。新 session 保存配置、命令、Git 状态、源码快照及参照身份。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_components.py plan --include-references
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_components.py status --session runs/m4_components_v1/formal_v1_20261008
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_components.py summarize --session runs/m4_components_v1/formal_v1_20261008
```

验收标准：18 个新 cell 完成，六个参照持续通过来源核验；各 validation 2675 窗口/7 人、样本与资格一致；完整五指标、主配对与三条递增比较、subject-macro、tail 和来源表齐备；不含 test 访问。

## 合成验证入口

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_m4_components.py
bash -n scripts/run_m4_components_shard.sh
```

合成测试使用 Mamba 替身，验证原始参照与共有初始化一致、内容 bias 消融、双投影均值、频带 adapter、全部臂 FP32/BF16 梯度、两步四微批正式 task loss、来源破坏拒绝、参照只读复用及完整汇总。正式 Mamba2 GPU 内核与真实数据运行由用户执行阶段验收。

2026-10-08：41 项定向 CPU 合成测试通过。正式 Mamba2 的 CPU 构造检查在全部三个 seed 上确认共有参数逐张量一致，B0/B1 与 P1 M4/M1 的完整 state dict 一致；未执行正式 Mamba2 前向或 GPU 内核。只读历史来源审计通过，六个参照的源码、快照、科学配置和完整 artifacts 均匹配。未启动 GPU 验收、训练、真实数据 smoke 或 test 访问。
