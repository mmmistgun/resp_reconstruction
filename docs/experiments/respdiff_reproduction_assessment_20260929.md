# RespDiff 复现接入评估

日期：2026-09-29。状态：初次代码与任务调研记录；补充论文后的现行决定见[移植方案v2](respdiff_tho_migration_plan_v2_20260929.md)，正式实验合同尚待确定。

## 1. 工作树与来源

- 分支：`codex/resp-diff`。
- 工作树：`/home/marques/.codex/worktrees/resp-diff/resp_reconstruction`。
- 基线提交：`0cc7b796df23526829f6d634adc4b28aca83166d`，来自本地 main；创建时 main 比 origin/main 领先156个提交，工作区干净。
- RespDiff 来源：`/mnt/disk_code/marques/reference_repos/RespDiff`，remote 为 `https://github.com/MYY311/RespDiff.git`。
- 来源提交：`3ff05545c34f67e1ad415e36e4daea80269880b0`，检查时工作区干净。只有 README 与四个 Python 文件；没有受 Git 管理的数据、checkpoint 或依赖清单。

Git worktree 共享 Git 对象和引用；历史 runs、数据与 .venv 等忽略文件不会自动复制。现有配置包含指向主仓库 runs/cache 的绝对路径，后续接入必须显式指定新的输出根并检查防覆盖，不能直接沿用既有实验入口的输出配置。

## 2. 当前项目与任务

这里将“原本模型”解释为完整 W0（`crd_tf102_w`，PATCH/TM3/REF2），同时把 E9 视为正在开展的结构实验。

- W0 三因素结构实验已经关闭，保留完整 W0；E5/E6/E7/E8 等阶段按各自 closeout 冻结。结构实验未找到同时保持全部评价属性的简化，不能据单项结果改选基线。
- 最近代码任务为 E9：D=96 的条件末端 H=65/64/48，以及 D=64 的 direct/H64/H48，六臂三 seed。源码协议开头仍描述 runtime amendment 过渡状态，但 amendment JSON 已存在，且本地 formal 产物已有完成回执，说明正式训练已经推进。
- 本次只读抽查到 `e9a_d96_h48/seed_20260813/formal_8bb3b1992f64_20260929T115729Z_4d1458ba5dce/formal_receipt.json`：30 epochs、2400 updates、2675 validation rows。另见 D64 H64 seed 20260813 的 access receipt。未核验完整18-cell矩阵，不能据此宣称 E9 已全部完成。
- 另有 W0 最终五指标协议 `w0_final_evaluation_protocol_20260927.md`。其指标与旧结构实验五指标不同，RespDiff 后续比较需明确采用哪套指标，不能混用同名近似指标。该文档仍记载真实 test 待执行，本次未审计它的实际完成状态。

任务列表中相关聊天包括“实现并冻结 E9 结构实验”“保存W0测试结果与可视化”“实现 W0 test 评价指标”；聊天摘要用于定位，实验状态以协议和产物为准。

## 3. RespDiff 原代码如何使用

入口为 `breathing_bidmc.py` + `model.py`，或 `breathing_bidmc_fft.py` + `model_fft.py`。两个训练脚本的实质差异只有导入的模型模块。

原脚本直接加载当前目录 `bidmc_data.mat`，循环53个 subject，按被留出的 subject 做评价。PPG 与参考呼吸信号按0.24倍重采样、代码按30 Hz使用；参考另做八阶1 Hz低通。随后切成不重叠5 s、150点片段，PPG逐片段映射至[-1,1]，参考逐片段映射至[0,1]。变量名 `co2` 实际取自 MAT 的 `ref.resp_sig` 字段，不能凭变量名断言其传感器模态。

训练参数固定为：`diffusion_pipeline(384, 1024, 6, 128, device)`；六层双向 vanilla tanh RNN，Adam lr=1e-4，batch=128，400 epochs，MultiStepLR 在280/396处降为0.1倍。`val_loader` 实际装的是留出 subject，脚本训练结束才评价，保存最后一轮 checkpoint；虽然声明 best_val_loss，却没有 validation selector。

模型 API（示意，未执行）：

```python
from model import diffusion_pipeline  # FFT版对应 from model_fft import ...

model = diffusion_pipeline(384, 1024, 6, 128, device).to(device)
# condition 与 target 均为 float tensor [B, 1, L]；原脚本 L=150。
loss = model(condition, co2=target, flag=0)
# 调用方再执行 backward/optimizer.step。
model.eval()
with torch.inference_mode():
    samples = model(condition, n_samples=100, flag=1)  # [B, 100, 1, L]
    prediction = samples.mean(dim=1)  # [B, 1, L]
```

训练时随机选一个扩散时间步，对参考波形加噪并预测噪声。推理从高斯噪声出发，逐步条件去噪，不需要参考波形。默认50步DDPM，beta线性从0.0001到0.5；原脚本采样100条轨迹取均值，相当于每个输入batch调用去噪网络5000次。FFT文件另有 flag=2 DDIM入口，但训练脚本使用 flag=1，DDIM并非原脚本默认评价方法。

直接运行原脚本会绑定 `cuda:2`，执行53轮留一受试者训练及留出评价，在当前目录保存固定名称的 checkpoint/CSV。依赖、MAT结构、运行资源与结果隔离均未在本次验证，接入时应先拆出无副作用的模型模块与独立CLI。

## 4. 与 W0 的核心差异

| 项目 | 完整 W0 | RespDiff 原代码 |
|---|---|---|
| 输入与目标 | 单通道 BCG → THO 呼吸波形 | PPG → MAT参考呼吸信号 |
| 时长/采样 | 180 s、100 Hz，输入[B,1,18000] | 5 s、30 Hz，输入[B,1,150] |
| 幅值处理 | 现有 segment soft-z 数据键 | 逐5秒 min-max，输入[-1,1]、目标[0,1] |
| 主干 | Patch前端 → 六层BiMamba2，10 Hz latent | 多尺度普通/空洞Conv1d → 六层双向RNN，逐采样点运算 |
| 条件机制 | CWT分支生成受限gamma/beta，FiLM调制主干 | 条件编码与带噪目标编码拼接，同时注入扩散步嵌入 |
| 波形生成 | FiLM后refinement与head，10 Hz波形Fourier插值到100 Hz | 预测噪声，50步反向扩散产生波形 |
| 训练目标 | L_sync + 0.25 L_effort | 噪声平方误差；FFT版再加0.01×幅度谱误差 |
| 推理 | 一次前向，返回waveform字典 | 多步随机采样，返回[B,N,1,L]再取均值 |
| 模型选择 | 完整validation Local RR最小，并列取最早 | 固定400 epochs后保存最终checkpoint |
| 数据划分 | 固定train/validation/test，主体隔离 | 53折leave-one-subject-out |
| 原生评价 | 项目RR、波形/相关、包络体系 | 片段MAE、拼接滤波后MAE、60 s FFT-RR、duty-cycle相关 |

代码依据：`resp_train/crd/tf_v1_model.py:369`、`resp_train/crd/model.py:301`、`configs/crd_tf_v1/crd_tf101_m_smoke.yaml`；来源仓库 `model.py:132`、`:211`、`:224` 和 `breathing_bidmc.py:57`、`:115`、`:161`、`:213`。

## 5. 接入前必须处理的具体问题

1. **FFT版不是纯loss变化。** `model.py:149` 给噪声编码加step embedding，`model_fft.py:188` 没有这项。若比较FFT贡献，应固定同一个架构再切loss；若复现发布代码，则分别记录这两个版本的实际身份。
2. **噪声loss的batch缩放。** 原实现为平方误差 `sum()/L`，FFT项为全batch均值。单通道时噪声项等于batch size乘MSE，尾batch或改变batch会改变相对权重。不能直接改为 `.mean()` 后仍声称与原优化合同一致。
3. **幅值与时长是科学选择。** 逐5秒独立目标min-max会丢失跨片段幅值关系；简单拼接并不能恢复THO努力包络。移植到180 s且沿用segment归一化更贴近现有任务，但属于任务适配。若保持30 Hz，180 s为5400点，RNN序列长度是原代码36倍；如果直接用100 Hz则为120倍。二者的可行性未测量。
4. **推理存在batch依赖。** 空洞编码器BatchNorm设置 `track_running_stats=False`，eval仍用当前batch统计，预测可能随batch组成变化。改变为GroupNorm或启用running stats属于结构/训练变化；忠实实现应保留并明确固定batch及顺序。
5. **随机性与资源。** 原脚本未固定训练/采样seed，推理也未包 `no_grad`；迁移应记录训练seed、采样seed、N、steps、batch和顺序，使用inference_mode。5000次网络调用不等于已测得5000倍时延。
6. **原评价不能直接挪用。** RR搜索全正频率，和本项目[0.05,0.70] Hz口径不同；60 s窗步长20 s的循环数量含 `+1`，但第273/277行分母遗漏该项。min-max也未防常量分母，DDIM第318行使用整个batch全局min/max且没有常量保护。这些均需显式记录修正和验证。
7. **设备和接口需要整理。** DDPM混用NumPy调度数组、CPU索引/输出容器和模型设备；应以synthetic fixture验证CPU/device迁移与state_dict round-trip，不能凭源码断言当前GPU环境已跑通。

## 6. 建议的后续实现边界

为回答当前BCG→THO任务上的模型比较，建议建立独立的“RespDiff-THO适配”实验：保留条件扩散与多尺度双向RNN核心，复用项目已冻结的主体划分、输入/target来源身份与评价窗口。PPG→BCG是输入模态变化，必须从头训练；现有W0 checkpoint无法直接转为RespDiff参数。

下一步先确定采样率/上下文长度/归一化、FFT版本及loss缩放、训练预算、validation selector、推理N/steps和最终报告指标，再实现：

- 独立模型、训练loss入口与采样入口；推理适配输出 `{"waveform": [B,1,18000]}`。
- 专属train/validation runner与配置，复用数据准入/主体隔离/指标函数，显式关闭独立test访问；不复用带旧来源锁的完整W0专项runner。
- 来源版本、变更说明、配置与命令记录，以及新且不可覆盖的输出identity。
- 最小synthetic CPU验证：shape、有限性、一次训练梯度、调度与采样可重复性、小尺寸采样、state_dict，以及batch/loss缩放约定。原始规模GPU验收与真实训练需另获执行授权。

如果目标改为BIDMC原结果复现，应单列原数据/53折实验；它回答的科学问题与THO同数据比较不同。现有W0结果保持其原合同，新增RespDiff结果使用相同已选定的报告口径后才能并表。最终指标若选择20260927版本，应引用相应W0来源，不把旧Whole/Local RR表直接换列名。

本次只创建分支与本文、只读查看代码和少量运行回执；未导入模型、安装依赖、执行训练/推理、读取真实波形或独立测试集。
