# 当前研究口径

当前工作树只承载 2026-08-02 冻结的 THO 呼吸重建重启协议。完整定义见 `docs/experiments/loss_metrics_restart_plan_20260729.md`。

- 数据与 split 沿用 research v2 soft-z 口径，不改变 target、subject/session 隔离或窗口长度。
- 第一版 baseline 是纯时域 `patch_mixer1d`，从头训练；旧实验 checkpoint 和结果不参与比较。
- 正式输出统一为完整 180 秒上的 $\Pi=S\circ B$，频带为 `0.05–0.70 Hz`。
- Loss 只保留同步、节律、努力趋势和训练早期极性四项。
- Checkpoint 只按完整 validation 的 Local RR MAE 严格最小值选择，关闭 early stopping。
- Validation/test 使用逐 sample direct mean；IBI 同时报告 MedAE 与 coverage；coherence 和 nDTW 只在 designated test 运行；不增加 event F1 或 CCC。
- 第一阶段只做 smoke、单 seed pilot 和三 seed baseline，不打开 designated test。

旧代码、旧配置和旧说明不放入仓库内 archive，也不维持兼容入口；需要追溯时使用 Git 历史。历史 `runs/`、checkpoint、CSV、图表和原始数据不改写。
