呼吸节律估计
BCG → 呼吸频率轨迹 RR(t)
BCG → 呼吸周期长度
BCG → 呼吸峰/谷的大致位置
要求和 PSG 峰谷严格对齐，只要求在 30 s、60 s 或更长窗口内，呼吸次数、瞬时呼吸率、频率变化趋势一致。


呼吸努力幅值趋势估计  趋势、相对幅值、事件前后变化
BCG → 呼吸努力强弱变化
BCG → 低通气前后幅值下降
BCG → 呼吸暂停后恢复呼吸增强
BCG → 周期性呼吸/努力波动


预测努力包络 与 PSG胸/腹带包络趋势一致
预测幅值下降位置 与 PSG事件附近一致
预测恢复增强 与 PSG趋势一致


呼吸频率轨迹
RR_ref[k]：每 10 s / 30 s / 60 s 的呼吸率
RR_pred[k]
L_RR = MAE(RR_pred, RR_ref)
L_RR = min over τ MAE(RR_pred(t), RR_ref(t + τ))


呼吸努力包络
A_ref(t) = 胸/腹带或 RIPsum 的局部峰-谷幅值
A_pred(t) = 预测呼吸代理信号的局部幅值
包络趋势相关性
幅值下降比例
事件前后相对变化
Drop_ref(t) = A_ref(t) / baseline_A_ref
Drop_pred(t) = A_pred(t) / baseline_A_pred


呼吸频谱 ridge
S_ref(t, f)：胸/腹带在呼吸频带内的归一化时频图
S_pred(t, f)：模型预测呼吸代理信号的时频图
主呼吸频率是否一致
频率变化趋势是否一致
呼吸频带能量是否连续
低通气/暂停附近能量是否下降


呼吸频带归一化谱 KL loss
ridge frequency MAE
多分辨率 STFT loss
窗口级频谱相关性


心冲击相关特征仍然要保留
心冲击幅值被呼吸调制
心冲击基线随呼吸运动变化
心搏形态随胸腔压力和姿势变化
心率/心搏间期受呼吸和觉醒影响


BCG高频/心冲击候选带
↓
心搏包络
心搏能量
心率轨迹
心冲击形态特征
↓
辅助呼吸估计


BCG
↓
respiratory latent branch
cardiac latent branch
contact / low-frequency context branch
residual branch
↓
融合后输出呼吸努力代理信号


r_hat(t)：BCG时间轴上的呼吸代理波形
A_hat(t)：呼吸努力包络
RR_hat(t)：呼吸率轨迹
P_breath(t)：呼吸峰/谷概率
Q_hat(t)：置信度


输入：
  raw BCG clean segment
  BCG低频呼吸候选带
  BCG心冲击候选带
  心冲击包络
  rolling energy / STD / derivative energy
  CWT/STFT时频图
  clean mask / motion mask

编码器：
  1D temporal encoder
  2D time-frequency encoder
  cardiac modulation encoder
  quality/context encoder

融合：
  gated fusion 或 cross-attention

输出：
  BCG-derived respiratory surrogate waveform
  respiratory rate trajectory
  respiratory effort envelope
  breath peak/valley probability
  respiratory event probability
  confidence / uncertainty

loss
  L_env = 1 - Corr(A_hat, A_ref)
  L_RR = SoftDTW(RR_hat, RR_ref)
  L_drop = Huber(Drop_hat, Drop_ref)
  L_spec = KL(P_hat(f | window), P_ref(f | window))
  L_ridge = MAE(f_resp_hat, f_resp_ref)
  Q_hat(t)：该段呼吸估计是否可信
  L = exp(-s) * L_task + s 这里 s 是模型预测的不确定性。损失越难解释，模型可以提高不确定性，但会受到 +s 惩罚，避免全部输出低置信度。


BCG混叠信号
→ 多视图输入：时域 + 时频域 + 心冲击包络
→ 模型估计呼吸频率 ridge、努力包络、呼吸代理波形
→ 以窗口级 PSG 标签约束