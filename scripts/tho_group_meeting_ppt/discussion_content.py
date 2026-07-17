"""面向首次接触项目听众的 THO research v2 研究讨论语义单元。"""

from __future__ import annotations

from dataclasses import dataclass


REPORT = "docs/stage_reports/2026-07-08-tho-research-v2-group-meeting-stage-report.md"
METRICS = "docs/experiments/metric_schema.md"
LEDGER = "docs/experiments/current_evidence_ledger.md"
FUSION = "docs/experiments/time_frequency_input_fusion_plan.md"
CONFIG = "configs/tho_research_v2.yaml"


@dataclass(frozen=True)
class DiscussionUnit:
    """一个可被后续自由合并的研究内容单元，而非版面或页码定义。"""

    key: str
    section: str
    title: str
    kind: str
    question: str
    method_steps: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    rationale: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    limits: tuple[str, ...] = ()
    discussion_prompt: str | None = None
    visual_keys: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


def _u(
    key: str,
    section: str,
    title: str,
    question: str,
    *,
    method_steps: tuple[str, ...],
    parameters: tuple[str, ...] = (),
    rationale: tuple[str, ...] = (),
    evidence: tuple[str, ...] = (),
    limits: tuple[str, ...] = (),
    discussion_prompt: str | None = None,
    visual_keys: tuple[str, ...] = (),
    sources: tuple[str, ...],
) -> DiscussionUnit:
    return DiscussionUnit(
        key=key,
        section=section,
        title=title,
        kind="technical",
        question=question,
        method_steps=method_steps,
        parameters=parameters,
        rationale=rationale,
        evidence=evidence,
        limits=limits,
        discussion_prompt=discussion_prompt,
        visual_keys=visual_keys,
        sources=sources,
    )


DISCUSSION_UNITS = (
    # 1. 任务与信号直觉
    _u(
        "task_mapping", "任务与信号直觉", "从床垫 BCG 到 THO-like 呼吸表示", "模型到底恢复什么？",
        method_steps=("输入固定时长 BCG 窗口。", "输出同长度 THO soft-z 波形。", "用呼吸节律、形态和强弱指标评价。"),
        rationale=("逐点误差不能覆盖临床上关心的呼吸节律与局部变化。",),
        limits=("当前输出是 THO-like 标准化波形，不是胸带原始物理幅值。",),
        discussion_prompt="最终任务应继续回归完整波形，还是改为波形加呼吸状态的多任务表示？",
        visual_keys=("task_signal_pair",), sources=(f"{REPORT}#任务动机与问题定义",),
    ),
    _u(
        "bcg_signal_mixture", "任务与信号直觉", "BCG 是呼吸、心冲击与接触变化的混合观测", "为什么不能直接低通得到目标？",
        method_steps=("区分呼吸带低频成分。", "识别心冲击、体动、姿态和接触条件成分。", "比较这些成分与 THO 的同步关系。"),
        rationale=("同一频带内仍可能同时存在真实呼吸、倍频和运动干扰。",),
        limits=("当前证据能说明混合机制，但不能把每个误差唯一归因到某种物理来源。",),
        discussion_prompt="哪些 BCG-only 质量特征最可能区分可恢复呼吸与运动伪迹？",
        visual_keys=("bcg_component_intuition",), sources=(f"{REPORT}#任务动机与问题定义",),
    ),
    _u(
        "tho_reference_limits", "任务与信号直觉", "THO 参考也有时延、方向和局部质量问题", "监督目标是否等同于真值？",
        method_steps=("检查 THO 可观测与有效标记。", "允许有限时延后比较低频形态。", "对局部质量下降窗口回到完整波形复核。"),
        rationale=("避免把传感器差异和参考质量问题误判为模型错误。",),
        limits=("THO 仍是本研究参考，不代表无噪声生理真值。",),
        discussion_prompt="需要引入第三种参考信号来判定 BCG 与 THO 分歧的责任来源吗？",
        visual_keys=("reference_uncertainty",), sources=(f"{REPORT}#任务动机与问题定义", f"{METRICS}#低频形态与时延诊断"),
    ),
    _u(
        "window_180s_tradeoff", "任务与信号直觉", "180 秒窗口是稳定估计与局部变化之间的折中", "为什么一个样本长达三分钟？",
        method_steps=("以 100 Hz 采样切取 18,000 点。", "整窗估计主呼吸节律。", "再用滑动子窗评估局部 RR。"),
        parameters=("窗口 180 秒。", "采样率 100 Hz。", "相邻索引通常间隔 30 秒。"),
        rationale=("长窗给频谱与稳健峰值估计足够周期，同时保留局部诊断空间。",),
        limits=("窗口重叠使窗口级样本并不独立，且长窗会混合状态变化。",),
        discussion_prompt="主模型是否应缩短输入窗，并用跨窗状态模型补充长期上下文？",
        visual_keys=("window_timescale",), sources=(f"{REPORT}#数据集与预处理", CONFIG),
    ),

    # 2. 数据来源与样本形成
    _u(
        "data_provenance", "数据来源与样本形成", "以导出包 provenance 冻结数据口径", "训练数据如何追溯？",
        method_steps=("读取导出 manifest 与配置快照。", "将当前训练配置映射到导出字段。", "记录制作代码版本和未提交状态。"),
        rationale=("数据制作工作区曾有未提交改动，不能只引用当前仓库状态。",),
        limits=("本仓库只消费导出数据，原始制作细节需回到对应 provenance 核验。",),
        discussion_prompt="正式结论是否需要重新生成一个工作区完全干净的数据导出包？",
        sources=(f"{REPORT}#数据集与预处理",),
    ),
    _u(
        "index_to_window", "数据来源与样本形成", "索引行把整晚记录变成固定窗口样本", "一个 dataset row 如何形成？",
        method_steps=("从 index 读取 subject、split 与起止时间。", "定位受试者 NPZ。", "按字段键和时间范围切片。", "返回完整输入/目标波形、RR 评价 mask 与 metadata。"),
        parameters=("window_start_s/window_end_s 决定切片。", "每窗期望 18,000 点。"),
        rationale=("索引把样本选择与信号存储解耦，便于审计。",),
        limits=("同一受试者的相邻窗口高度重叠。",),
        discussion_prompt="后续统计是否应把连续重叠窗聚合成独立时间块？",
        visual_keys=("index_to_tensor",), sources=("resp_train/data/index.py", "resp_train/data/research_v2.py"),
    ),
    _u(
        "waveform_window_filter", "数据来源与样本形成", "可用窗口还要通过 waveform 任务筛选", "哪些索引行真正进入模型？",
        method_steps=("要求 allowed_losses 包含 waveform。", "排除 reason 非空窗口。", "检查 hard-valid 与 state-alignment-valid 比例。"),
        parameters=("hard_valid_ratio ≥ 0.80。", "state_alignment_valid_ratio ≥ 0.80。"),
        rationale=("不让明显无效或未对齐窗口主导训练与评价。",),
        evidence=("报告记录筛选后 train/val/test 可用窗口池。",),
        limits=("阈值是数据可用性护栏，不等同于信号质量连续评分。",),
        discussion_prompt="二元筛选是否应升级为连续置信权重，并验证是否改变困难窗口表现？",
        sources=(CONFIG, f"{REPORT}#窗口筛选和受试者分配", "resp_train/data/research_v2.py"),
    ),
    _u(
        "subject_split_boundary", "数据来源与样本形成", "split 的泄漏边界是受试者而非窗口", "如何防止重叠窗跨集合泄漏？",
        method_steps=("按 subject 分配 train/val/test。", "核对三组 subject 集合交集为空。", "在集合内部保留原窗口重叠。"),
        parameters=("当前可用池为 32/7/8 名受试者。",),
        rationale=("同受试者形态和重叠时间片进入不同 split 会严重高估泛化。",),
        evidence=("阶段报告给出可用窗口与受试者清单。",),
        limits=("受试者隔离不消除设备、场景或采集批次的共同偏差。",),
        discussion_prompt="现有 split 是否足以支撑跨场景泛化，还是需要按设备或夜晚再分组？",
        visual_keys=("subject_split",), sources=(f"{REPORT}#窗口筛选和受试者分配", "resp_train/data/independence.py"),
    ),

    # 3. 输入与目标预处理
    _u(
        "bcg_wideband_input", "输入与目标预处理", "主输入保留 0.03–20 Hz 宽频 BCG", "为何不用现成呼吸带 BCG？",
        method_steps=("从 100 Hz BCG 做零相位带通。", "保留呼吸低频、心冲击和运动上下文。", "后续由模型选择与呼吸相关的成分。"),
        parameters=("4 阶 Butterworth。", "0.03–20.0 Hz。"),
        rationale=("只保留 0.05–0.7 Hz 可能丢掉帮助识别干扰和接触状态的信息。",),
        limits=("宽频输入也增加模型学习非呼吸干扰的风险。",),
        discussion_prompt="是否应显式分离呼吸带与带外上下文，而不是把全部选择交给主干？",
        sources=(f"{REPORT}#滤波处理",),
    ),
    _u(
        "bcg_segment_robust_z", "输入与目标预处理", "BCG 先按状态段估计 robust-z", "跨姿态和接触状态如何统一尺度？",
        method_steps=("按 coupling state 切分统计段。", "排除无效、体动和状态边界秒点。", "估计 robust center/scale。", "在段边界平滑过渡。"),
        parameters=("每段至少 60 秒可靠样本。", "边界平滑 5 秒。"),
        rationale=("局部稳健统计减少跨状态幅度漂移与异常值影响。",),
        limits=("状态标签和统计池本身若错误，会把偏差带入全部窗口。",),
        discussion_prompt="能否用在线、BCG-only 的尺度估计替代依赖预先状态段的处理？",
        visual_keys=("segment_robust_z",), sources=(f"{REPORT}#归一化和软压缩",),
    ),
    _u(
        "bcg_soft_z", "输入与目标预处理", "soft-z 压缩体动长尾但保留排序", "为什么 robust-z 后还要压缩？",
        method_steps=("在 z 空间识别大幅值尾部。", "从 knee 开始用 log1p_tail 软压缩。", "在过渡区平滑衔接。"),
        parameters=("scale_abs=2。", "knee_abs=6。", "limit_abs=10。", "过渡 2 秒。"),
        rationale=("避免少数极端体动把数值尺度和梯度主导。",),
        limits=("压缩会改变幅值物理意义，也可能弱化真正的大呼吸事件。",),
        discussion_prompt="是否需要对体动段显式建模，而不是仅做幅值压缩？",
        sources=(f"{REPORT}#归一化和软压缩",),
    ),
    _u(
        "target_soft_z_and_mask", "输入与目标预处理", "目标 soft-z、整窗筛选与 RR mask 是三条不同链路", "目标和质量标记实际如何进入训练与评价？",
        method_steps=("按 THO amplitude segment 估计稳健统计并做 soft-z 压缩。", "用 allowed_losses、reason、hard-valid ratio 与 state-alignment-valid ratio 做整窗筛选。", "Dataset 返回完整 target 波形，训练引擎将其原样交给 WeakSyncLoss。", "rr_peak_valid_mask 随 metadata 进入预测产物，仅供 masked raw peak RR 评价及有效比例/连续有效段统计。"),
        parameters=("soft scale/knee/limit 为 2/6/10。", "整窗 hard-valid 与 state-alignment-valid 门槛均为 0.80。"),
        rationale=("区分样本准入、完整波形监督和评价 mask，避免把评价期坏段处理误解为 sample-level masked loss。",),
        evidence=("ResearchV2WindowDataset 返回完整 target 与独立的 rr_peak_valid_mask metadata。", "train_one_epoch 将完整 target 传给 WeakSyncLoss；当前基础 loss 不接收逐点 valid mask。", "evaluate.py 仅在 masked raw peak RR 路径使用 rr_peak_valid_mask，并另外输出 valid ratio 与 segment count。"),
        limits=("完整 target 波形中的局部低质量仍会进入当前基础训练损失；若未来引入逐点 mask，属于训练语义变更，必须重新验证。",),
        discussion_prompt="是否需要另做显式逐点置信监督消融，而不是把现有 RR 评价 mask 直接复用到 loss？",
        visual_keys=("target_mask_pipeline",), sources=("resp_train/data/research_v2.py", "resp_train/engine/train.py", "resp_train/losses/weak.py", "resp_train/metrics/evaluate.py"),
    ),

    # 4. 模型输入与计算图
    _u(
        "model_tensor_contract", "模型输入与计算图", "计算图从 B×1×18000 波形开始", "模型各分支接收什么张量？",
        method_steps=("DataLoader 形成 batch、channel、time。", "时序主干直接接收波形。", "可选时频分支从同一 BCG 在线派生。"),
        parameters=("单通道输入和输出。", "时间长度 18,000。"),
        rationale=("共享原始输入保证对照只改变辅助表示。",),
        limits=("当前模型不显式接收 subject、状态或质量 metadata。",),
        discussion_prompt="质量 metadata 应作为模型输入、训练权重，还是只用于评价分层？",
        visual_keys=("tensor_contract",), sources=(CONFIG, "resp_train/models/timeseries.py"),
    ),
    _u(
        "patchmixer_token_shape", "模型输入与计算图", "PatchMixer 把长波形切成重叠 token 再折回", "PatchMixer 的 shape 如何变化？",
        method_steps=("末端补齐到可 unfold 长度。", "按 patch_len/stride 切重叠 patch。", "展平通道与 patch 点并线性映射为 token。", "mixer 后线性还原 patch 并 overlap-add。"),
        parameters=("正式对照配置以 run config 为准；实现默认 patch_len=256、stride=128、base_channels=16。",),
        rationale=("patch token 降低长序列计算量，同时保留局部波形上下文。",),
        evidence=("代码明确规定 token 为 B×C_token×N_patch。",),
        limits=("当前常驻 config 不是正式 G 系列 resolved config，参数不得仅凭默认值断言。",),
        discussion_prompt="patch 尺度是否与呼吸周期和心冲击时标匹配，是否需要多尺度 token？",
        visual_keys=("patch_token_shape",), sources=("resp_train/models/timeseries.py", f"{REPORT}#模型结构概览"),
    ),
    _u(
        "stft_resolution", "模型输入与计算图", "STFT 分辨率同时决定频率辨别与时间更新速度", "20 秒窗、2.5 秒步长改变了什么？",
        method_steps=("对同一 soft-z BCG 分帧。", "计算 log(1+|STFT|)。", "截取 0.05–8 Hz。", "把 STFT 帧对齐到 patch token 栅格。"),
        parameters=("wide: win=2000 点、hop=250 点。", "F0: 30 秒窗、5 秒步长。", "wide 分析带 0.05–8 Hz。"),
        rationale=("更短窗与步长提高辅助信息时间分辨率，宽频带保留呼吸外上下文。",),
        limits=("更短窗降低频率分辨力；插值对齐不能创造新的时间信息。",),
        discussion_prompt="下一轮应先优化 STFT 物理分辨率，还是转向可学习滤波器组？",
        visual_keys=("stft_resolution_grid",), sources=(f"{REPORT}#当前比较方案", "resp_train/models/stft_branch.py"),
    ),
    _u(
        "pre_mixer_injection", "模型输入与计算图", "pre-mixer 注入是同 shape token 的加法融合", "时频信息在哪里进入主干？",
        method_steps=("时频编码器输出 token。", "投影并对齐到时序 token shape。", "在 mixer block 前逐元素相加。", "共享后续 mixer 与 waveform decoder。"),
        parameters=("inject_position=pre_mixer。",),
        rationale=("让后续时序混合共同解释时间与频率信息，同时保持输出路径一致。",),
        limits=("无条件加法缺少样本级抑制机制，可能误伤容易窗口。",),
        discussion_prompt="选择性修正更适合 token gate、残差 head，还是两个模型之间的输出 gate？",
        visual_keys=("pre_mixer_graph",), sources=("resp_train/models/timeseries.py", "resp_train/models/stft_branch.py", f"{REPORT}#模型结构概览"),
    ),
    _u(
        "bandenergy_encoder", "模型输入与计算图", "bandenergy 把频谱压成五条重叠频带序列", "低维频带表示保留和丢掉什么？",
        method_steps=("计算同一输入的 log-magnitude STFT。", "在五个预设重叠频带内求均值。", "得到每帧五维能量。", "编码并对齐到 patch token。"),
        parameters=("频带为 0.05–0.3、0.1–0.7、0.3–1.2、0.7–3.0、3.0–8.0 Hz。",),
        rationale=("压缩频率自由度，突出可解释的呼吸、谐波和带外强弱变化。",),
        limits=("带内平均会丢掉峰位置、窄带形态和同带多峰结构。",),
        discussion_prompt="五个固定频带是否应由生理先验冻结，还是允许数据驱动边界微调？",
        visual_keys=("bandenergy_bands",), sources=(f"{REPORT}#实验范围与控制变量", "resp_train/models/stft_branch.py"),
    ),

    # 5. 训练与损失
    _u(
        "train_val_test_roles", "训练与损失", "训练、验证、测试承担不同决策角色", "数据集合如何参与模型选择？",
        method_steps=("train 更新参数。", "val 早停并检查方向 gate。", "冻结方案后在 held-out test 复评。", "同方案保留独立 seed。"),
        parameters=("每方案 3 次独立训练。",),
        rationale=("把超参数选择与最终泛化评价分开。",),
        limits=("方案研究仍多次观察验证结果，可能积累验证集适应。",),
        discussion_prompt="是否需要嵌套验证或重新冻结一个决策集来控制迭代偏差？",
        sources=(f"{REPORT}#训练流程与损失设置",),
    ),
    _u(
        "composite_loss_goal", "训练与损失", "复合损失约束任务性质而非逐点复刻", "为什么没有把 MSE 作为唯一目标？",
        method_steps=("分别计算包络、频谱、平滑、高频、相对包络和方向项。", "按配置权重相加。", "按 epoch 更新方向项权重。"),
        parameters=("基础项权重由 configs/tho_research_v2.yaml 冻结，方向项在 epoch 1–6 调度。",),
        rationale=("BCG 与 THO 允许局部相位和形态差异，但节律、强弱与方向仍应一致。",),
        limits=("多项损失可能梯度冲突；当前权重主要来自阶段性经验。",),
        discussion_prompt="是否需要记录各损失梯度夹角，判断复合目标是否互相抵消？",
        visual_keys=("loss_components",), sources=(CONFIG, "resp_train/losses/weak.py", f"{REPORT}#训练流程与损失设置"),
    ),
    _u(
        "loss_component_weights", "训练与损失", "主要损失项对应包络、频谱与输出平滑", "每一项具体约束什么？",
        method_steps=("2 秒 RMS 包络标准化后做相关损失。", "比较 0.05–0.7 Hz 归一化功率。", "惩罚相邻点差分。", "惩罚 0.7 Hz 以上相对能量。", "比较局部相对包络。"),
        parameters=("权重依次为 1.0、0.2、0.1、0.2、0.03。",),
        rationale=("覆盖整体强弱、主频分布、尖锐噪声和局部幅度变化。",),
        limits=("这些可微代理与最终峰值 RR、周期计数并非一一对应。",),
        discussion_prompt="哪一项最可能造成容易窗口被过度平滑，如何做最小消融确认？",
        sources=(CONFIG, "resp_train/losses/weak.py", f"{REPORT}#训练流程与损失设置"),
    ),
    _u(
        "polarity_schedule", "训练与损失", "方向损失在早期填浅反极性 basin", "为什么方向约束要调度？",
        method_steps=("带通并标准化预测与目标。", "计算 signed correlation 与 cosine。", "第 1–6 epoch 线性调整权重。", "之后保留相关项而关闭 cosine。"),
        parameters=("signed_corr 0.6→0.2。", "signed_cos 0.1→0。", "调度 epoch 1–6。"),
        rationale=("初始化早期防止进入反向解，后期避免方向辅助项压过其他任务目标。",),
        evidence=("配置注释记录该调度用于降低方向 gate 失败。",),
        limits=("方向一致不代表相位、周期数或局部形态正确。",),
        discussion_prompt="能否通过结构约束消除极性歧义，从而减少人工调度？",
        visual_keys=("polarity_schedule",), sources=(CONFIG, "resp_train/losses/weak.py"),
    ),
    _u(
        "checkpoint_selection", "训练与损失", "G 系列先保留 gate 后的 val-loss top-k，再按任务指标旁路复评", "本轮 G 系列的代表 checkpoint 实际怎样产生？",
        method_steps=("每个 epoch 同时记录 val loss 并检查 auto_direction 方向 gate。", "只有通过方向 gate 的 epoch 才进入候选，按 val loss 排序保存 checkpoint_top1/2/3.pt 与 checkpoint_topk.csv。", "训练结束后对 top-k 做当前任务指标的旁路复评，再选择进入 held-out canonical test 的代表 checkpoint。", "canonical test manifest 逐 seed 记录实际采用的 top1、top2 或 top3 路径。"),
        parameters=("本轮方向 gate 为 auto_direction、max=0.5。", "每次训练保留 val-loss top-k=3。"),
        rationale=("低 val loss 不保证峰值 RR、count 或 lag-aware 形态更好。",),
        evidence=("正式运行证据示例：runs/g_series_stft_input/g3_c_wide_8p0/dual/20260630_211450_861579/config.yaml 与同目录 checkpoint_topk.csv。", "最终选择证据：runs/test_eval_g_series_20260709_local_rr_canonical/g_series_test_eval_manifest.csv；其中不同 seed 实际采用不同 top-k rank。"),
        limits=("checkpoint_best_task.pt 是新版可选机制，不是本轮 G 系列已经保存或用于 canonical test 的既成事实。", "当前自动选择已固化 robust RR 与 count 主护栏；lag-aware 形态和 local RR 仍需人工复核。"),
        discussion_prompt="应把哪组任务指标固化为唯一 checkpoint 选择规则，如何处理多目标冲突？",
        visual_keys=("checkpoint_semantics",), sources=("scripts/README.md", METRICS, "docs/experiments/g_series_stft_input_resolution_band_plan.md", "resp_train/experiments/base.py", "resp_train/experiments/tho.py", f"{REPORT}#训练流程与损失设置"),
    ),

    # 6. 指标计算与失效场景
    _u(
        "metric_robust_rr", "指标计算与失效场景", "稳健峰值 RR 先带通、谱引导，再数峰间距", "整窗主要呼吸节律如何估计？",
        method_steps=("对预测和目标做 0.05–0.7 Hz 带通。", "Welch 主峰引导最小峰距。", "稳健找峰并取相邻峰间距中位数。", "计算两侧 bpm 绝对差。"),
        parameters=("最小峰间距至少 2 秒。", "4 阶零相位 Butterworth。"),
        rationale=("降低局部尖峰、弱伪峰和坏段对整窗 RR 的影响。",),
        limits=("双峰、谐波抢峰或节律快速变化仍可能给出误导性单值。",),
        discussion_prompt="整窗 RR 是否应输出多峰不确定性，而不是强制给出单一值？",
        visual_keys=("robust_rr_steps",), sources=(f"{REPORT}#整窗呼吸率", METRICS, "resp_train/metrics/signal.py"),
    ),
    _u(
        "metric_breath_count", "指标计算与失效场景", "周期计数用带通信号的双向过零估计", "三分钟内恢复了多少次呼吸？",
        method_steps=("呼吸带滤波。", "分别统计上升与下降过零。", "两者平均并取整为周期数。", "预测与目标作差并除以窗口分钟数。"),
        parameters=("180 秒时 bpm error=周期数误差/3。",),
        rationale=("不依赖每个峰的精确形态，补充 RR 主峰指标。",),
        limits=("基线漂移、低幅噪声和一个周期内多次过零会放大误差。",),
        discussion_prompt="是否应改用事件匹配或滞回过零，以减少低幅抖动影响？",
        visual_keys=("zero_cross_count",), sources=(f"{REPORT}#周期数量", METRICS, "resp_train/metrics/evaluate.py"),
    ),
    _u(
        "metric_lag_corr", "指标计算与失效场景", "4 秒 lag-aware correlation 分离形态与持续时延", "相位不齐时如何比较波形？",
        method_steps=("对两侧做呼吸带滤波。", "在 ±4 秒逐采样搜索重叠相关。", "选最高相关并在近似并列时偏向零 lag。", "同时报告相关和最佳 lag。"),
        parameters=("搜索范围 [-4,4] 秒。",),
        rationale=("避免持续时延把相似低频形态误判为失败。",),
        limits=("周期信号可能在错误周期偏移处也获得高相关；大 lag 仍可能不可接受。",),
        discussion_prompt="最佳 lag 应作为容忍项、模型输出，还是需要显式校正后再评价？",
        visual_keys=("lag_search",), sources=(f"{REPORT}#低频形态与时延", METRICS, "resp_train/metrics/signal.py"),
    ),
    _u(
        "metric_relative_envelope", "指标计算与失效场景", "相对包络比较呼吸增强和减弱而非绝对幅值", "模型是否跟上强弱变化？",
        method_steps=("按最佳 4 秒 lag 对齐。", "计算 2 秒 RMS 包络。", "用约 40 秒平滑趋势归一化。", "对数化后计算相关与 MAE。"),
        parameters=("RMS 2 秒。", "趋势约 40 秒。"),
        rationale=("去除整体尺度，突出窗口内部的相对起伏。",),
        limits=("趋势窗会抹平更慢变化，低包络处的对数比例更敏感。",),
        discussion_prompt="趋势时标是否应随呼吸率或状态自适应？",
        visual_keys=("relative_envelope",), sources=(f"{REPORT}#相对呼吸强弱", METRICS, "resp_train/metrics/signal.py"),
    ),
    _u(
        "metric_local_rr", "指标计算与失效场景", "局部 RR 用 40 秒窗追踪三分钟内的节律变化", "整窗 RR 正确时局部是否也正确？",
        method_steps=("两侧先做呼吸带滤波。", "40 秒窗每 10 秒移动。", "每个子窗做严格谱引导稳健找峰。", "仅在双方有效位置算 MAE、corr 与 valid fraction。"),
        parameters=("局部窗 40 秒。", "步长 10 秒。"),
        rationale=("揭示整窗平均指标掩盖的局部快慢变化。",),
        limits=("不是逐呼吸事件匹配；target 谱峰硬切换、谐波和低 valid fraction 会失真。",),
        discussion_prompt="下一步应优先改进局部 RR 估计器，还是直接评价逐呼吸事件？",
        visual_keys=("local_rr_curve",), sources=(f"{REPORT}#局部呼吸率", METRICS, "resp_train/metrics/signal.py"),
    ),

    # 7. 对照实验设计
    _u(
        "controlled_variables", "对照实验设计", "四方案共享数据、主干、目标、训练与评价", "比较能否归因到时频表示？",
        method_steps=("固定 split 与预处理。", "固定 PatchMixer 主干和输出。", "固定损失、训练设置和评价。", "只切换辅助时频表示。"),
        rationale=("减少多因素同时变化造成的归因歧义。",),
        limits=("不同分支参数量与优化难度仍不完全等价。",),
        discussion_prompt="是否需要参数量或 FLOPs 匹配的额外对照，排除容量差异？",
        visual_keys=("control_matrix",), sources=(f"{REPORT}#实验范围与控制变量",),
    ),
    _u(
        "time_only_baseline", "对照实验设计", "time-only 定义不使用显式时频辅助的 substrate", "STFT 的净增益以谁为零点？",
        method_steps=("仅将 BCG 波形送入 PatchMixer。", "使用相同 decoder 输出 THO soft-z。", "按相同 seed 与指标链评价。"),
        rationale=("回答时频表示是否提供超出时序主干的额外信息。",),
        limits=("时序网络本身也能隐式学习频率特征，因此不是无频率信息模型。",),
        discussion_prompt="还需要一个参数量匹配但输入无信息的假分支对照吗？",
        visual_keys=("time_only_graph",), sources=(f"{REPORT}#当前比较方案", "resp_train/models/stft_branch.py"),
    ),
    _u(
        "f0_vs_wide", "对照实验设计", "F0 与 wide 同为完整 STFT，但时间和频带分辨率不同", "更高时间分辨率是否更适合作为辅助？",
        method_steps=("F0 使用 30 秒/5 秒旧配置。", "wide 使用 20 秒/2.5 秒且覆盖至 8 Hz。", "固定融合位置为 pre-mixer。", "做 seed 配对差值。"),
        parameters=("F0: 30s/5s。", "wide: 20s/2.5s、0.05–8Hz。"),
        rationale=("直接检验辅助谱图时间更新速度与宽频上下文的共同作用。",),
        limits=("窗长、步长和高频上限一起变化，不能进一步拆出单因素因果。",),
        discussion_prompt="若要解释 wide 收益，应优先补窗长、步长还是频带上限的单因素消融？",
        visual_keys=("f0_wide_comparison",), sources=(f"{REPORT}#当前比较方案", LEDGER),
    ),
    _u(
        "wide_vs_bandenergy", "对照实验设计", "wide 与 bandenergy 比较完整谱图和低维频带摘要", "固定频带摘要能否替代完整 STFT？",
        method_steps=("共享 20 秒/2.5 秒 STFT。", "wide 用 conv2d 编码完整频率网格。", "bandenergy 对五频带求均。", "比较总体、困难层与完整案例。"),
        parameters=("两者分析上限均为 8 Hz。",),
        rationale=("区分细粒度谱形与可解释频带能量哪种更有任务价值。",),
        limits=("两种编码器容量与归纳偏置同时变化。",),
        discussion_prompt="bandenergy 的局部优势来自频带先验还是更强正则化？",
        visual_keys=("wide_bandenergy",), sources=(f"{REPORT}#当前比较方案", "resp_train/models/stft_branch.py", LEDGER),
    ),

    # 8. 整体结果与稳定性
    _u(
        "overall_metric_values", "整体结果与稳定性", "整体测试值显示 wide 综合更均衡、bandenergy 局部指标略优", "四方案在统一口径下表现如何？",
        method_steps=("逐窗口计算五类核心指标。", "先汇总到单次训练。", "再对每方案三个 seed 取均值。", "按主护栏与诊断指标共同解释。"),
        parameters=("wide robust RR MAE 0.712186 bpm；lag4 corr 0.870774，无量纲。", "bandenergy 周期数绝对误差 mean 2.054401，对应 180 秒窗口的周期计数 bpm 误差 0.684800；local RR MAE 0.773299 bpm。"),
        evidence=("当前证据账本给出四方案三 seed 的 canonical test 汇总。",),
        rationale=("同时保留节律、计数、形态与局部 RR，避免单指标胜出被误称为全面更好。",),
        limits=("数值是重叠窗口级汇总，不能直接视作受试者级独立重复。",),
        discussion_prompt="默认 anchor 应按综合护栏选择，还是为不同任务指标保留多个专家模型？",
        visual_keys=("overall_metric_table",), sources=(LEDGER, f"{REPORT}#关键证据 2：宽频、更高时间分辨率的 STFT 更适合作为当前基准配置"),
    ),
    _u(
        "paired_delta", "整体结果与稳定性", "paired delta 在相同 seed 上隔离方案差异", "均值改善是否来自可配对的训练变化？",
        method_steps=("按训练 seed 配对候选与 time-only。", "逐 seed 计算候选减 baseline。", "检查方向是否一致。", "再报告 delta 均值而非只比较两组均值。"),
        parameters=("当前每个方案使用 3 个可配对训练 seed。",),
        rationale=("配对可减少初始化难度差异对模型对照的噪声。",),
        evidence=("账本记录 wide 相对 time-only 的 robust RR、count 与 local RR mean delta。",),
        limits=("只有三个 seed，无法可靠估计尾部分布或显著性。",),
        discussion_prompt="下一步预算应投向增加 seed，还是增加受试者级重采样？",
        visual_keys=("paired_delta",), sources=(LEDGER, f"{METRICS}#当前排序建议"),
    ),
    _u(
        "seed_stability", "整体结果与稳定性", "三次独立训练只能检查方向稳定性，不能证明总体稳定", "结论对初始化敏感吗？",
        method_steps=("保留每个 seed 的单次汇总。", "检查核心指标方向与异常 seed。", "将平均收益与 seed 间离散同时呈现。"),
        parameters=("当前每方案 3 seed。",),
        rationale=("避免单次幸运初始化主导 anchor 决策。",),
        limits=("样本量小，不适合把均值差解释成精确的总体效应。",),
        discussion_prompt="什么稳定性门槛足以支持结构决策：同向 seed 数、效应量还是置信区间？",
        visual_keys=("seed_dots",), sources=(LEDGER, f"{REPORT}#训练流程与损失设置"),
    ),
    _u(
        "result_decision_rule", "整体结果与稳定性", "anchor 选择优先主护栏，再看形态与局部诊断", "为什么 wide 胜出而非每项最优的拼盘？",
        method_steps=("先核对数据、split、seed 和 checkpoint 口径。", "检查 robust RR 与 count 主护栏。", "再解释 lag-aware 形态、相对包络和 local RR。", "标记冲突指标而不做单一总分。"),
        rationale=("不同指标回答不同问题，任意加权总分会隐藏临床含义。",),
        limits=("自动选择只固化 robust RR 与 count 主护栏，不把 lag-aware 形态和 local RR 压成单一总分。",),
        discussion_prompt="是否需要预注册一个 Pareto 决策规则，避免后续结果导向地换权重？",
        visual_keys=("decision_ladder",), sources=(METRICS, LEDGER),
    ),

    # 9. 分层分析与完整案例
    _u(
        "subject_stratification", "分层分析与完整案例", "受试者分层检验总体均值是否由少数人主导", "收益能否跨受试者复现？",
        method_steps=("按 subject 汇总同一批测试窗。", "比较 paired model delta。", "同时报告窗口数和受试者差异。"),
        rationale=("各受试者窗口数极不均衡，窗口均值会给长记录更高权重。",),
        limits=("只有 8 名测试受试者，个体层结论仍是初步证据。",),
        discussion_prompt="正式统计应采用受试者等权、层级 bootstrap 还是混合效应模型？",
        visual_keys=("subject_forest",), sources=(f"{REPORT}#窗口筛选和受试者分配", f"{REPORT}#8 名测试受试者的直接结果"),
    ),
    _u(
        "quality_difficulty_strata", "分层分析与完整案例", "质量与难度分层用于解释收益发生在哪里", "平均收益来自容易窗还是困难窗？",
        method_steps=("用冻结 baseline 定义 hard/easy。", "按谱相似、count error 或质量标记分层。", "在同一 strata 内做模型 paired 比较。", "报告 strata 覆盖量。"),
        rationale=("选择性修正需要知道候选在哪类窗口有净收益。",),
        limits=("现有 hard、count 与 low-spectrum 分层含目标或模型误差信息，不能作为推理 gate。",),
        discussion_prompt="哪些 BCG-only proxy 最可能复现这些离线 strata，且不会泄漏目标？",
        visual_keys=("strata_matrix",), sources=(f"{REPORT}#关键证据 4：分层结果揭示选择性修正的潜在收益", FUSION),
    ),
    _u(
        "rr_bin_strata", "分层分析与完整案例", "按目标 RR 区间检查慢、正常与快呼吸角色差异", "频率区间会改变模型排名吗？",
        method_steps=("用目标参考将窗口放入冻结 RR bin。", "在每个 bin 内比较同窗 paired delta。", "检查 wide 与 bandenergy 的角色是否交换。"),
        rationale=("固定频带与 STFT 分辨率可能对不同呼吸速度产生不同偏置。",),
        limits=("目标 RR 分层仅用于回顾性解释，在线不可用。",),
        discussion_prompt="是否要训练 RR 条件化模型，还是只在输出后做频率区间校正？",
        visual_keys=("rr_bins",), sources=(f"{REPORT}#RR 分层",),
    ),
    _u(
        "complete_case_review", "分层分析与完整案例", "完整案例必须同时看输入、目标、预测、频谱与指标", "如何避免只展示好看的片段？",
        method_steps=("按预定义类别选择成功、失败、分歧和边界案例。", "固定 row 后展示四模型同一 seed。", "叠加 BCG/THO 频谱与核心指标。", "解释与总体统计是否一致。"),
        rationale=("完整信号链能发现单个指标未暴露的相位、倍频和局部崩溃。",),
        limits=("少量案例不能估计总体效应，选择规则必须先于看图冻结。",),
        discussion_prompt="案例抽样还应加入哪些反例类别，才能暴露选择性 gate 的风险？",
        visual_keys=("complete_case_grid",), sources=(f"{REPORT}#代表案例与证据边界",),
    ),

    # 10. BCG 二次谐波问题
    _u(
        "harmonic_definition", "BCG 二次谐波问题", "二次谐波问题是 BCG 在 2×THO 基频附近占优", "什么叫倍频困难窗口？",
        method_steps=("对 BCG 与 THO 呼吸带分量计算频谱。", "以 THO 参考频率 f 为基频。", "检查 BCG 主峰是否靠近 2f。", "再检查 2f 相对 f 与全带能量。"),
        parameters=("分析带 0.05–0.7 Hz。",),
        rationale=("把模型 hard window 连接到可解释的输入双峰或倍频机制。",),
        limits=("定义依赖 THO 参考，只能用于离线分析。",),
        discussion_prompt="能否从 BCG 自身的基频/谐波竞争和时间一致性建立无目标定义？",
        visual_keys=("harmonic_spectrum",), sources=(f"{REPORT}#冻结规则与覆盖率", "resp_train/analysis/second_harmonic.py"),
    ),
    _u(
        "harmonic_thresholds", "BCG 二次谐波问题", "阈值在完整验证集人工复核后冻结", "阳性标签如何确定？",
        method_steps=("先要求 THO 稳健 RR 与谱 RR 相差不超过 1 bpm。", "要求 2f 仍在分析带内。", "验证集发现候选并人工复核。", "冻结后一次应用测试集。"),
        parameters=("主峰距 2f 相对误差 ≤10%。", "E2/E1 ≥0.5393。", "E2/Eband ≥0.1647。"),
        rationale=("先冻结阈值再看测试结果，降低测试集调阈值风险。",),
        limits=("阈值来自 7 名验证受试者，跨数据集稳定性未知。",),
        discussion_prompt="阈值应保持规则式，还是用验证集训练可校准概率模型？",
        visual_keys=("harmonic_threshold_plane",), sources=(f"{REPORT}#冻结规则与覆盖率", "resp_train/analysis/second_harmonic.py"),
    ),
    _u(
        "harmonic_subtypes", "BCG 二次谐波问题", "strong、peak-doubling 与 harmonic-prominent 分开解释", "阳性窗口内部是否同质？",
        method_steps=("两类证据都成立标 strong。", "仅主峰证据标 peak-doubling。", "仅能量证据标 harmonic-prominent。", "并集用于覆盖率，子层用于机制解释。"),
        evidence=("测试集中 peak-doubling 子层样本极少，不能独立形成稳定结论。",),
        rationale=("主峰位置错误和能量显著不是同一种输入机制。",),
        limits=("子层仍可能受 THO 谱稳定性和窗口重叠影响。",),
        discussion_prompt="后续应合并稀少子层，还是扩大样本后保留机制区分？",
        visual_keys=("harmonic_venn",), sources=(f"{REPORT}#冻结规则与覆盖率",),
    ),
    _u(
        "harmonic_correction", "BCG 二次谐波问题", "纠偏要求输出回到基频且压低相对二倍频能量", "怎样判定模型真的纠正倍频？",
        method_steps=("估计模型输出主峰。", "检查主峰是否回到 THO 基频 10% 容差内。", "比较输出与输入的 E2/E1。", "要求相对比值至少下降 20%。"),
        parameters=("基频容差 10%。", "谐波比下降阈值 20%。"),
        evidence=("四类模型纠正率均约 95% 以上，但任务指标仍有差异。",),
        rationale=("只看主峰可能把仍保留强双峰的输出误判为已纠正。",),
        limits=("高纠正率不等于周期计数、局部 RR 与形态全部正确。",),
        discussion_prompt="纠偏定义是否还应加入局部时间一致性，而不只看整窗频谱？",
        visual_keys=("correction_before_after",), sources=(f"{REPORT}#四模型在固定阳性窗口上的任务表现", "resp_train/analysis/second_harmonic.py"),
    ),
    _u(
        "harmonic_boundary_cases", "BCG 二次谐波问题", "边界与失败案例约束谐波结论的外推", "哪些案例最能暴露规则局限？",
        method_steps=("固定四模型均纠正案例。", "固定四模型均失败案例。", "加入模型分歧案例。", "加入阈值附近案例并检查标签敏感性。"),
        parameters=("已冻结案例 row 为 640、873、1353、3584。",),
        rationale=("同时展示共有能力、共同失败、模型差异与阈值脆弱性。",),
        limits=("阳性窗口集中于少数受试者，且这些 row 只是机制示例。",),
        discussion_prompt="阈值附近样本应作为软标签、不确定类，还是继续二元分类？",
        visual_keys=("harmonic_cases",), sources=(f"{REPORT}#代表案例与证据边界",),
    ),

    # 11. 研究议题与下一步
    _u(
        "decision_output_space", "研究议题与下一步", "先决定继续波形回归还是转向任务表示", "哪种输出最接近研究目标？",
        method_steps=("比较完整波形、局部 RR、呼吸事件与状态输出。", "列出每种输出可监督性和评价口径。", "用现有失败案例检查信息损失。"),
        rationale=("当前复合损失和多指标冲突可能来自输出空间与最终任务不一致。",),
        limits=("现有证据主要来自波形回归 checkpoint，不能直接证明替代输出更优。",),
        discussion_prompt="下一轮最值得验证的输出空间是哪一个，什么结果会让我们放弃完整波形？",
        sources=(LEDGER, f"{REPORT}#风险与不确定性"),
    ),
    _u(
        "decision_bcg_only_gate", "研究议题与下一步", "选择性修正的前提是无泄漏 BCG-only gate", "何时应调用 bandenergy 修正？",
        method_steps=("从 BCG 构造谱多峰、质量和不确定性特征。", "只在 train/val 冻结 gate。", "在完整 test 比较默认 wide 与条件修正。", "单独检查容易窗保护率。"),
        rationale=("bandenergy 的收益集中在部分困难层，全窗口替换会牺牲综合稳定性。",),
        limits=("现有困难层依赖目标或 baseline error，尚无已验证在线代理。",),
        discussion_prompt="gate 应优化困难窗召回，还是优先保证容易窗几乎不被误触发？",
        visual_keys=("selective_gate",), sources=(LEDGER, f"{REPORT}#下一阶段建议", FUSION),
    ),
    _u(
        "decision_statistical_unit", "研究议题与下一步", "把统计单位从重叠窗口提升到受试者和时间块", "怎样证明收益不是伪重复？",
        method_steps=("按 subject 等权重算核心指标。", "按非重叠时间块做 block bootstrap。", "同时报告窗口级与受试者级效应。", "检查谐波阳性受试者集中度。"),
        rationale=("180 秒窗每 30 秒移动，普通窗口 bootstrap 会高估有效样本量。",),
        limits=("测试受试者只有 8 名，区间仍可能很宽。",),
        discussion_prompt="现阶段什么统计证据足以推进方法，什么证据必须等新增受试者？",
        visual_keys=("statistical_units",), sources=(f"{REPORT}#风险与不确定性", f"{REPORT}#代表案例与证据边界"),
    ),
    _u(
        "decision_minimal_next_experiment", "研究议题与下一步", "下一实验应回答一个可停止的选择性修正问题", "最小可决策实验是什么？",
        method_steps=("固定 wide anchor 与 bandenergy candidate。", "冻结一个 BCG-only gate 和阈值。", "用配对 seed 比较 always-wide、always-bandenergy 与 conditional。", "预设容易窗保护、困难窗收益和受试者稳定性停止条件。"),
        rationale=("避免再次扩大 STFT、CWT、SST 或融合矩阵而没有明确决策。",),
        limits=("gate 尚未定义，当前只能给出实验判定结构，不能声称预期收益。",),
        discussion_prompt="哪一个 BCG-only gate 候选最值得先做，以及接受它的定量门槛是什么？",
        visual_keys=("next_experiment_decision",), sources=(LEDGER, f"{REPORT}#下一阶段建议", FUSION),
    ),
)
