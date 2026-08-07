# 当前研究口径

当前工作树只承载 THO 呼吸重建重启协议。完整定义与阶段结果见 `docs/experiments/loss_metrics_restart_plan_20260729.md`。

- 数据与 split 沿用 research v2 soft-z 口径，不改变 target、subject/session 隔离或窗口长度。
- B0 是纯时域 `patch_mixer1d` 协议 baseline；当前时频 active candidates 为 T2 fullband native 与 T4 bandenergy native。
- 正式输出统一为完整 180 秒上的 $\Pi=S\circ B$，频带为 `0.05–0.70 Hz`。
- Loss 只保留同步与努力趋势：`L_sync + 0.25 L_effort`。
- Checkpoint 只按完整 validation 的 Local RR MAE 严格最小值选择，关闭 early stopping。
- Validation/research-test 使用逐 sample direct mean；IBI 同时报告 MedAE 与 coverage；coherence 和 nDTW 只在 research-test 运行；不增加 event F1 或 CCC。
- 现有 `test` 是可重复观察、可形成后续研究问题的 research-test，不是无偏 held-out 证据；访问必须显式确认，且不得回头重选既有 run 的 epoch/checkpoint。
- 2026-08-07 已完成 B0/T2/T4 三 seed及 F0/IEWT 的完整 research-test。T4 在学习模型的多数主轴与 nDTW 上最强；T2 保持包络分层排序优势；B0 保持最低 IBI-MedAE。F0 的最高 coherence 同时伴随较差 RR/PCC/coverage，确认 coherence 只应作补充指标。

旧代码、旧配置和旧说明不放入仓库内 archive，也不维持兼容入口；需要追溯时使用 Git 历史。历史 `runs/`、checkpoint、CSV、图表和原始数据不改写。
