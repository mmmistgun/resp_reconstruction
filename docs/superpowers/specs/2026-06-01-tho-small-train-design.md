# 胸带参考小规模训练骨架设计

## 背景

当前仓库已有研究方案文档和 Stage 2.1 数据集快照，但尚未建立深度学习训练代码。第一阶段目标是在胸带参考条件下，搭建一个可训练、可评价、可扩展的最小研究闭环。

数据集路径：

`/mnt/disk_code/marques/dataset/RespPairs/20260530_tho_ramp5_stage2_1`

训练入口：

`training/dataset_index.csv`

该数据集包含 180 秒窗口、100 Hz 采样率、多个 `input_set`，其中首版默认使用 `mixed_zscore`。长期目标是完整实验骨架，但当前里程碑先完成小规模训练。

## 目标

首个里程碑采用“瘦 C 骨架”：

- 先跑通胸带参考 `r_tho_hat(t)` 的小规模训练和评价。
- 代码目录、配置、模型注册、run 产物按后续完整实验骨架预留。
- 默认只训练 `mixed_zscore`，但 `input_set` 可配置切换。
- 默认只输出单一路径 `r_tho_hat(t)`，包络、频谱和节律指标从输出信号派生。
- 默认使用包络、频谱和平滑损失，避免弱同步条件下的逐点 MSE 主监督。

验收级别为“小规模训练”：

- 能加载部分 train/val 窗口。
- 能完成若干 epoch 训练。
- 能保存 checkpoint、metrics 和少量诊断预测。
- 能通过评价结果判断训练闭环是否成立。

默认小规模训练参数：

- `epochs=3`
- `batch_size=8`
- `learning_rate=1e-3`
- `num_workers=0`
- `preload_windows=true`

## 非目标

首版不实现以下内容：

- 腹带参考训练。
- 胸腹综合呼吸努力参考训练。
- 多参考多头输出。
- 完整时频双分支模型。
- 全量 input_set 消融。
- 完整绘图系统。
- 完整实验管理平台。
- 默认逐点波形 MSE 或复杂 shift-invariant 波形损失。

这些内容作为后续扩展点保留接口。

## 总体架构

建议新增以下结构：

```text
configs/
  tho_small.yaml

resp_train/
  data/
  models/
  losses/
  metrics/
  engine/
  utils/

scripts/
  train_tho_small.py
  eval_tho_small.py

runs/
  tho_small/
```

职责边界：

- `configs/`：保存数据路径、默认 `input_set`、窗口上限、训练参数、loss 权重和输出目录。
- `resp_train/data/`：读取 `dataset_index.csv`，解析相对 `source_npz`，缓存整晚 NPZ 数组并按窗口切片。
- `resp_train/models/`：提供模型注册表，默认实现轻量 1D 时域模型，后续扩展时频模型。
- `resp_train/losses/`：实现包络、频谱和平滑损失。
- `resp_train/metrics/`：实现评价指标。
- `resp_train/engine/`：纯 PyTorch 训练与验证循环。
- `resp_train/utils/`：日志、配置、随机种子、设备选择、run 目录等工具。
- `scripts/`：提供面向用户的训练和评价入口。

## 数据加载与缓存

Dataset 默认读取 `training/dataset_index.csv`，并按配置过滤：

- `input_set=mixed_zscore`
- `split=train` 或 `split=val`
- `max_train_windows=1024`
- `max_val_windows=256`

每条样本使用 CSV 中的信息：

- `source_npz`
- `bcg_signal_key`
- `target_signal_key`
- `valid_sec_key`
- `window_start_sample`
- `window_end_sample`
- `target_fs`
- `dataset_row_id`
- `samp_id`
- `segment_id`
- `window_id_in_segment`

默认缓存策略：

- `WholeNightCache` 按 `source_npz` 缓存整晚数组。
- 每个 NPZ 文件首次访问时读取所需数组，后续窗口从缓存中切片。
- 返回张量形状为 `[1, 18000]`，对应 180 秒、100 Hz。

可选加速策略：

- 小规模训练可开启 `preload_windows=true`，提前把当前 split 的窗口切片放入内存。
- 预缓存受 `max_train_windows`、`max_val_windows` 限制。
- 后续正式训练可扩展为 LRU 或 memmap，不改变 Dataset 对外接口。

`valid_sec_key` 在首版用于过滤和记录，不默认引入复杂 mask loss。若窗口有效率不足，应在 Dataset 初始化阶段过滤或记录警告。

## 模型设计

首版模型采用插件式注册：

- 默认模型名为 `unet1d_tiny`。
- 输入为 `[B, 1, 18000]`。
- 输出为 `[B, 1, 18000]`。
- 输出语义为 `r_tho_hat(t)`，即胸带参考条件下的 BCG 时间轴呼吸努力估计信号。

模型不默认输出：

- RR 辅助头。
- 包络辅助头。
- 峰谷概率。
- 腹带或综合努力分支。

后续扩展通过模型注册表新增：

- `tcn1d_*`
- `timefreq_*`
- `dual_branch_*`
- `abd_*`
- `effort_*`

## 损失函数

默认总损失：

```text
L = w_env * L_envelope
  + w_spec * L_spectrum
  + w_smooth * L_smooth
```

`L_envelope`：

- 对 `r_tho_hat(t)` 和 `tho_ref(t)` 提取包络。
- 训练损失默认使用可微滑窗 RMS 包络，默认窗口为 2 秒。
- Hilbert 包络只作为后续评价或消融实现，不作为首版默认训练损失。
- 比较归一化后的趋势，避免直接依赖逐点相位一致。

`L_spectrum`：

- 在呼吸频带内比较归一化频谱结构，默认频带为 0.05-0.7 Hz。
- 用于避免模型只学习平滑趋势而忽略呼吸节律。

`L_smooth`：

- 默认使用一阶差分 L1 约束输出连续性。
- 抑制无意义高频抖动。

首版不默认使用：

- 逐点 MSE。
- RR 辅助头损失。
- shift-invariant waveform loss。

这些内容可作为后续消融实验加入配置。

## 评价与产物

每次训练创建一个轻量 run 目录，例如：

```text
runs/tho_small/<timestamp>/
  config.yaml
  checkpoint.pt
  train.log
  metrics.csv
  predictions.npz
```

必需产物：

- `config.yaml`：本次有效配置快照。
- `checkpoint.pt`：模型状态、优化器状态、epoch 和最佳指标。
- `train.log`：关键入参、数据统计、每轮训练摘要。
- `metrics.csv`：窗口级和汇总评价指标。

可选诊断产物：

- `predictions.npz` 仅保存少量窗口，默认 `max_prediction_windows=32`。
- 它用于快速画图复查，不作为正式全量预测归档。
- 正式训练时可只保存 checkpoint 和 metrics；需要绘图时再运行预测或评价脚本生成预测文件。

首版默认不生成 PNG。后续可基于 `predictions.npz` 增加绘图脚本。

## 评价指标

首版评价从 `r_tho_hat(t)` 和 `tho_ref(t)` 派生：

- loss 分解。
- 包络相关或包络趋势误差。
- 呼吸频带频谱相似度。
- 输出平滑度。
- 基础有效性统计，例如 finite ratio、窗口数量、split、input_set。

波形相关性最多作为辅助诊断，不作为首版主指标。

## 配置与依赖

运行环境使用仓库当前目录的 `.venv`，由全局 `nv` 管理。

实现前只检查已安装库，不擅自安装依赖。若缺依赖，应先报告：

- 包名。
- 用途。
- 建议安装命令。

等待确认后再安装。

允许使用成熟依赖，避免重复造轮子：

- 默认依赖候选：`torch`、`numpy`、`pandas`、`scipy`、`tqdm`、`omegaconf`。
- 可选依赖候选：`torchinfo`、`torchmetrics`。
- 暂不默认引入 Hydra；首版先用 `omegaconf` 读取 YAML。

## 错误处理与日志

错误处理：

- 数据路径不存在、CSV 缺列、NPZ 缺 key、窗口越界、张量存在非有限值时 fail-fast。
- 单个窗口无效但可恢复时记录并过滤。
- 不静默吞掉数据异常。

日志策略：

- 记录配置、数据集路径、split、input_set、窗口数量、缓存策略和设备信息。
- 记录每个 epoch 的训练/验证 loss 和主要指标。
- 不在高频 batch 内输出大量日志，batch 进度交给 `tqdm`。

## 验证计划

实现后的最小验证顺序：

1. 检查 `.venv` 中依赖是否可导入。
2. Dataset smoke test：读取 `mixed_zscore` 的少量 train/val 窗口，确认形状、finite ratio 和 key 解析正确。
3. Forward/loss smoke test：一个 batch 能通过模型、损失和 backward。
4. 小规模训练：完成若干 epoch，生成 run 目录。
5. 产物检查：确认 `config.yaml`、`checkpoint.pt`、`train.log`、`metrics.csv` 存在；若开启诊断预测，确认 `predictions.npz` 只保存少量窗口。

## 扩展路径

后续从该骨架扩展：

- 添加 `legacy_v1`、`mixed_mild`、`mixed_rawish` 消融。
- 添加整夜推理。
- 添加绘图脚本。
- 添加时频输入模型。
- 添加传统基线评价。
- 在新数据集准备好后，切换到腹带参考实验。
- 在胸带和腹带实验稳定后，再研究综合呼吸努力参考。

这些扩展不应改变首版 Dataset、模型注册和 run 目录的基本接口。
