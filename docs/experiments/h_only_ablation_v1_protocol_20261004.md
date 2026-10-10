# H-only＋H64 消融实验 v1

协议 ID：`h-only-ablation-v1-20261004`。日期：2026-10-04。

状态：实现与 synthetic CPU 定向验收已完成；GPU 验收、真实数据访问、训练、HA16 历史产物复用核验和 research-test 均待用户执行。本协议只作用于 `h_only_ablation_v1`。

代码基于 `52948d2`，工作分支 `codex/h-only-ablation-v1`。每个 session 保存配置、源码快照及哈希、Git 身份、环境版本、来源 manifest 与执行命令；本系列使用独立的运行 identity。旧实验的模型、训练逻辑与产物保持其冻结身份。

## 1. HA0 与独立消融矩阵

HA0：APOR，180 s/100 Hz 单通道 BCG 输入和 18000 点输出；两个 Mixer、96 通道适配、六层 BiMamba2，140-token 主干。CWT 使用实际标定中心频率 `(0.8,8] Hz`、Morlet μ=13.4、12 voices/octave；`mean_pool(log1p(abs(CWT)))`，50 点池化，41×360。

条件卷积为 `5×3` 输入卷积、GN、SiLU、`3×3` depthwise 与通道投影、SiLU、尺度平均。每 patch 的五个位置为 `128j + {0,63.75,127.5,191.25,255}`；池化 bin 中心为 `24.5+50k`，线性插值并沿用边界延拓。五位置保留原拼接顺序并经 `480→96` 投影。

条件残差为 `h + project(SiLU(expand(h)))`，`96→64→96`；expand 无 bias，project 有 bias。FiLM 参数投影 `96→192` 零初始化，输出原始 `a,b`。HA0 使用 `z'=[1+0.5 tanh(a)]z+0.5 tanh(b)`。

片段头为 `Linear(96,96)→SiLU→Linear(96,256)`。合成沿用现有实现：非周期 Hann 的权重下限为 `1e-3`，256 点片段、128 步长、float32 重叠相加与逐位置权重归一化，18048 点裁剪至 18000 点。该端点处理与历史 H65 一致。

各行从 HA0 独立修改：

| 配置 | 唯一改动 | 构造参数量 |
|---|---|---:|
| HA0 | 完整 H-only＋H64；从头训练 | 1,077,448 |
| HA1 | 删除 CWT 条件分支与 FiLM，保留原片段头 | 994,312 |
| HA2 | 两个 patch-mixing 残差子层整体恒等，移除对应归一化及参数 | 1,076,712 |
| HA3 | 两个 channel-mixing 残差子层整体恒等，移除对应归一化及参数 | 1,075,240 |
| HA4 | 六层 BiMamba2 整体恒等，保留前端 96 通道适配 | 126,064 |
| HA5 | `5×3→1×3`、`3×3→1×3`；其余编码保持一致 | 1,076,584 |
| HA6 | 五个采样位置均为 `128j+127.5`，保留五槽和 `480→96` | 1,077,448 |
| HA7 | 整个 H64 残差模块恒等，保留最终 FiLM 投影 | 1,065,064 |
| HA8 | 只注册 β 投影，`z'=z+0.5 tanh(b)` | 1,068,136 |
| HA9 | 只注册 γ 投影，`z'=[1+0.5 tanh(a)]z` | 1,068,136 |
| HA10 | 片段头改为 `Linear(96,256)` | 1,068,136 |
| HA11 | 重叠权重全 1；保留逐位置归一化与裁剪 | 1,077,448 |
| HA12 | 努力损失权重 0 | 1,077,448 |
| HA13 | 乘性增益 `1+tanh(a/2)` | 1,077,448 |
| HA14 | 乘性增益 `1+2 tanh(a/4)` | 1,077,448 |
| HA15 | 乘性增益 `1+0.5a` | 1,077,448 |
| HA16 | 高频 H-only＋H65 宽度参照 | 1,077,640 |

HA0/13/14/15 的理论增益范围分别是 `(0.5,1.5)`、`(0,2)`、`(-1,3)`、实数域，均有 `s(0)=1,s'(0)=0.5`，加性项相同。浮点 tanh 在极端输入下可能数值饱和至理论区间端点。

所有模型先构造同 seed 的历史 H65 初始参数，然后独立应用变体。H64 取 expand 前 64 行和 project 前 64 列，保留 project bias；同名同 shape 的公共参数逐 tensor 相同。HA8/9 从原 192 通道最终投影的对应半部复制。HA5 和 HA10 的新层使用既有模块级确定性初始化命名；其他层不受随机数消耗变化影响。

## 2. 数据、优化与选点

沿用冻结受试者划分、窗口集合、预处理、target-only 资格和输出投影 Π。train 为 10141 窗/32 人，validation 为 2675 窗/7 人；数据 index、行集合/顺序和缓存来源均核验。HA1 的 loader 不附加 CWT 条件。

三 seed 固定为 `20260811/20260812/20260813`。batch128、梯度累积1、BF16、AdamW、`3e-4→3e-5`、5% warm-up、cosine 与原梯度裁剪均从冻结 W0 配置继承。每轮80次更新，最多80轮、最少30轮、patience15、min_delta0；提前停止不压缩6400次计划更新的学习率日程。

loss 为 `L_sync+0.25 L_effort`，仅 HA12 改为 `L_sync`。所有配置由自身完整 validation Local RR 最小值选择 checkpoint，并列取最早。新 history 校验器按该配置的努力权重核验完整训练目标；LR、停止轨迹、optimizer state 和选点同时校验。HA12 是新的从头训练实例，旧 checkpoint 的其他 loss 结果不能充当 HA12。

先验收并完成 HA0 三实例，再执行 HA1–HA15；这三个 HA0 实例直接进入最终矩阵。HA16 只有通过历史 H-only/H65 同 seed 的配置、频率网格、完整 history、checkpoint/optimizer 和初始化身份核验后才能登记复用；全频 A0 会被拒绝。无法证明兼容时执行独立 HA16 训练。复用全部通过则新增48次训练，否则最多51次；整个矩阵始终是17配置×3seed。

## 3. 诊断与失败处理

所有配置保存裁剪前梯度范数，每次反向传播对应一个 update。HA0/13/14/15 额外保存每训练 batch 的增益统计及最终 validation/test 的逐窗口增益统计：分位数 `0/.01/.05/.25/.5/.75/.95/.99/1`、min/max、负增益比例、`|s|<0.1` 近零比例和 `|s|>3` 极端比例。上述阈值在训练之前固定。逐窗口分位数不冒充整个 split 的合并分位数。

非有限 input/target/prediction/checkpoint/关键指标或梯度显式失败，保留 traceback 和所有已写产物。HA15 沿用同一优化合同；稳定性是实验结果，额外稳定化须另立编号。改善先归因于整个参数化，负增益比例只辅助解释。

阶段目录 `attempt_<timestamp>_<随机标识>` 不可覆盖，receipt 绑定 session 与 cell。成功 cell 在批量入口中只读跳过。失败重试须显式 `--retry-failed`，保留原 attempt。训练已完成而 validation 导出失败时，必须使用 `recover-export`；训练入口会拒绝再次训练这个 cell。

## 4. 用户执行顺序

以下命令由用户运行。先在新 worktree 下进入 Bash，按机器情况调整 `HA_DEVICE`。`prepare` 只读核验已有 train/validation 缓存；不重建历史缓存。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-ablation-v1
HA_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
HA_SOURCE=/home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction/runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731
HA_SESSION="$PWD/runs/h_only_ablation_v1/session_v1_001"
HA_DEVICE=cuda:0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

"$HA_PY" scripts/run_h_only_ablation_v1.py plan
"$HA_PY" scripts/run_h_only_ablation_v1.py prepare --session "$HA_SESSION" \
  --source-session "$HA_SOURCE" --confirm-train-validation

"$HA_PY" scripts/run_h_only_ablation_v1.py acceptance --session "$HA_SESSION" \
  --arms HA0 --device "$HA_DEVICE" --confirm-gpu
"$HA_PY" scripts/run_h_only_ablation_v1.py train --session "$HA_SESSION" \
  --arms HA0 --device "$HA_DEVICE" --confirm-training

HA_REST=(HA1 HA2 HA3 HA4 HA5 HA6 HA7 HA8 HA9 HA10 HA11 HA12 HA13 HA14 HA15)
"$HA_PY" scripts/run_h_only_ablation_v1.py acceptance --session "$HA_SESSION" \
  --arms "${HA_REST[@]}" --device "$HA_DEVICE" --confirm-gpu
"$HA_PY" scripts/run_h_only_ablation_v1.py train --session "$HA_SESSION" \
  --arms "${HA_REST[@]}" --device "$HA_DEVICE" --confirm-training

for HA_SEED in 20260811 20260812 20260813; do
  "$HA_PY" scripts/run_h_only_ablation_v1.py reuse-ha16 --session "$HA_SESSION" \
    --seed "$HA_SEED" --source-session "$HA_SOURCE" || break
done
```

`acceptance` 使用原生 CUDA/Mamba、batch128、BF16、合成 input/target/CWT 执行两次真实 loss 反传和 optimizer update；要求所有注册参数均有有限梯度、更新后参数有限，并保存环境及参数量。验收模型不用于正式初始化。它不替代正式训练的实际数据结果。

如某个 HA16 seed 兼容性失败，保留诊断；对该 seed 运行 `acceptance --arms HA16 --seeds <seed> --confirm-gpu`，然后 `train --arms HA16 --seeds <seed> --confirm-training`，均带相同的 `--session` 和 `--device`。不得替换已经登记成功的 HA16。

完成全部51项后：

```bash
"$HA_PY" scripts/run_h_only_ablation_v1.py summary --session "$HA_SESSION"
"$HA_PY" scripts/run_h_only_ablation_v1.py freeze --session "$HA_SESSION"
```

训练完成后的导出恢复示例：

```bash
"$HA_PY" scripts/run_h_only_ablation_v1.py recover-export --session "$HA_SESSION" \
  --arm HA0 --seed 20260811 --source-attempt /该cell失败attempt的绝对路径 \
  --device "$HA_DEVICE" --retry-failed --confirm-train-validation
```

完整 checkpoint allowlist 固定后，以下命令由用户在本轮授权 research-test 时执行：

```bash
"$HA_PY" scripts/run_h_only_ablation_v1.py test-prepare --session "$HA_SESSION" --confirm-research-test
"$HA_PY" scripts/run_h_only_ablation_v1.py test --session "$HA_SESSION" \
  --arms HA0 HA1 HA2 HA3 HA4 HA5 HA6 HA7 HA8 HA9 HA10 HA11 HA12 HA13 HA14 HA15 HA16 \
  --device "$HA_DEVICE" --confirm-research-test
"$HA_PY" scripts/run_h_only_ablation_v1.py summary --session "$HA_SESSION" \
  --split test --confirm-research-test
```

`test-prepare` 在本 session 下生成独立 test reference/H-only cache，核验2310窗/8人及与 development 的受试者隔离；test 不进入训练/选点入口。任何矩阵缺项会阻止 allowlist 固定。发生失败后保留全部预定配置和失败 lifecycle；不根据 partial seed/test 结果替换配置或阈值。

## 5. 保存、汇总与证据边界

每 cell 保存 resolved config、初始化 tensor 身份、代码和环境、执行命令、sample seed/训练 seed、best/final checkpoint 与 optimizer state、完整 history、梯度与所需增益统计、逐窗预测、行身份、逐窗五指标和阶段 receipt。训练导出及恢复产物分别绑定原 checkpoint 身份。

五指标为 Whole RR MAE、Local RR MAE、包络轨迹 MAE、全局包络调制误差、lag-aware signed PCC。汇总要求完整矩阵，输出逐窗口、逐seed、逐受试者、三seed均值与样本SD、对 HA0 的配对变化及改善/恶化/持平方向数。误差的 `degradation=candidate-reference`，PCC 为 `reference-candidate`。

test full 同时报窗口平均及受试者等权；预定敏感性视图为 exclude670（2231窗/7人）和 subject670（79窗/1人）。报告按结构消融、损失消融、增益形式、64/65宽度对照分组。seed SD 只描述训练随机性。原 research-test 已在前序系列使用，本轮属于后续研究性评价；不能将其表述为从未触达的确认性测试队列。

## 6. 实现验收

CPU 定向测试入口：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q tests/test_h_only_ablation_v1.py
```

测试使用 synthetic/disposable fixture，覆盖三seed嵌套初始化、历史 H65 state 精确一致、17臂参数量与公共层、前向/反向参与性、五槽中心坐标、单路投影切片、增益初值/导数/范围、HA12 loss 与最早并列选点、固定LR、受试者等权统计、完整矩阵门控及失败保留。CPU 前向测试显式替换 Mamba 为 Identity，不能据此宣称原生 CUDA kernel 或真实数据训练已经验收。
