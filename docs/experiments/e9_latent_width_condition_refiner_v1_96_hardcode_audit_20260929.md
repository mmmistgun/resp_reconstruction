# E9：`96` 硬编码全仓审计

日期：2026-09-29。作用域：E9 设计与实现前置审计。扫描表达式为单词边界 `96`，排除 `.git/`、`.venv/`、`runs/` 与 `__pycache__/`。扫描是写入本审计文档前的快照，因此文档自身后续出现的数字不属于该快照。

## 1. 扫描覆盖

| 区域 | 命中数 | 文件数 | 处置 |
|---|---:|---:|---|
| `resp_train/**/*.py` | 196 | 38 | 逐类核对可执行语义 |
| `tests/**/*.py` | 96 | 26 | 历史合同测试保持只读；新增 E9 独立测试 |
| `configs/**/*.{yaml,yml,json}` | 9 | 4 | 历史配置不改；新增 E9 独立配置 |
| `scripts/**/*.{py,md,sh}` | 6 | 2 | 历史入口不改；新增 E9 独立入口 |
| `docs/**/*.{md,json,yaml,yml}` | 649 | 65 | 历史协议、锁、结果均保持只读 |
| **合计** | **956** | **135** | 全部命中均纳入以下分类 |

## 2. E9 执行路径中的全局潜在宽度

| 来源 | 硬编码角色 | D=64 所需合同 | E9 处置 |
|---|---|---|---|
| `resp_train/crd/frontends.py:28-29` | Patch adapter `16→96` 与 GN | `16→64`、GN(8,64) | 在 E9 专属 `PatchTokenFrontend64` 重建；不改冻结类 |
| `resp_train/crd/model.py:267` | 六层 local BiMamba2 的 `d_model` | 六层 `d_model=64` | E9 专属 `D64CoarseModel` 重建 |
| `resp_train/crd/blocks.py:160` | BiMamba concat merge 为 `2D→D` | 自动成为 `128→64` | 复用已参数化的 `BidirectionalMamba2Block(64)` |
| `resp_train/crd/tf_v1_model.py:196` | CWT PWC `48→96` | `48→64` | E9 专属 D64 CWT branch 重建 |
| `resp_train/crd/tf_v1_model.py:127-132` | 条件上下文、末端、FiLM `96→192` | `[B,64,1800]`、`64→128` | E9 专属 D64 branch 重建并零初始化末投影 |
| `resp_train/crd/tf_v1_model.py:139` | 三个条件时间残差块 | 64 channels、GN 8 groups | 复用 `ResidualDWBlock(64)`，dilation 1/2/4 |
| `resp_train/crd/model.py:283-287` | FiLM 后两个 refinement block | 64 channels、GN 8 groups | E9 专属 D64 base 重建，dilation 1/2 |
| `resp_train/crd/blocks.py:185-189` | waveform head 输入 `96→64` | 输入改为64，后续64→64(DW)→32→1不变 | E9 专属 `CoarseWaveformHead64` |
| `resp_train/crd/model.py:306,347` | latent shape guards | `(64,1800)` | E9 专属显式 guard |
| `resp_train/paper_evidence/e8_film_decoder_redesign_v1_model.py` | D96 条件末端和 E8 冻结模型 | E9-A 仍为 D96 | 只复用原生 W0/E8 锚结构；不修改 E8 文件 |

## 3. 不应随 E9 改动的命中

| 分类 | 文件/模式 | 原因 |
|---|---|---|
| 历史冻结 CRD/E4–E8/W0 实现 | `resp_train/paper_evidence/e[1-8]*`、`w0_*`、对应 tests/configs | 这些数字描述各自已冻结模型、shape、MAC、诊断通道或结果身份；修改会破坏历史复现 |
| RTM-v1 独立模型族 | `resp_train/temporal/{blocks,config,model}.py` | `96` 是 RTM 的冻结 latent/LSTM/Mamba 合同，不属于 E9 |
| CRD 其他表示与容量对照 | `resp_train/crd/{capacity,representations}.py` | 属于未进入 E9 W-only 主路径的历史表示/容量模型 |
| Direct frontend 与 global stage | `resp_train/crd/frontends.py:87-92`、`model.py:177-193` | E9 固定 Patch frontend、无 global stage；不是本实验路径 |
| Local TCN | `resp_train/crd/blocks.py:62-100` | C1 独立对照，E9 固定六层 BiMamba2 |
| 频率/尺度索引 | `tf_w_v2.py`、E4/E7 中 `0..96`、`48_96` | 表示第97个尺度、lag pair 或尺度数组索引，不是通道宽度 |
| basis count | `resp_train/models/{registry,timeseries,lowfreq}.py` | 表示基函数数量，语义与潜在通道无关 |
| qualitative/diagnostic shape | `w0_test_qualitative*`、`w0_cwt_film_behavior.py` | 读取冻结 W0 的96通道中间量；E9 不复用为 D64 诊断 |
| 历史协议、实现锁与结果 | `docs/experiments/**` | 审计证据，必须保持字节身份或 Git 历史，不作批量替换 |

## 4. 审计结论

不能对仓库执行全局 `96→64` 替换。E9 只在新的实验命名空间中改变七处张量合同：Patch adapter/GN、六层 BiMamba2 与 merge、CWT PWC、条件时间块、条件末端与 FiLM 投影、post-FiLM refinement、waveform head 输入。基础 Patch encoder、CWT 48-channel 二维编码、decoder 的 32-channel residual、长度1800/18000以及所有历史实现保持不变。
