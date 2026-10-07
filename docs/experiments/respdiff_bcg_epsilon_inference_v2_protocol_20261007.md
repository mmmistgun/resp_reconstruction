# ε-RespDiff 推理协议 v2

协议 ID：`respdiff-bcg-epsilon-inference-v2-20261007`。

当前状态（2026-10-07）：CPU 合成验证、66-parent diagnostic subset、固定 N 选择及一次完整 DDIM6 N16 validation 均完成，本轮关闭。固定规则判定 `ensemble not converged by N=16`，按预算上限 N16 执行全量，未声称收敛。145 项产物哈希核验通过，checkpoint 字节和参数未改变。完整结果、图与证据入口见[ε 推理 v2 结果](respdiff_bcg_epsilon_inference_v2_results_20261007.md)。以下执行命令保留作历史记录，已有运行不重跑，独立测试集未开放。

## 论文语义与来源

以本地 RespDiff 论文 arXiv:2410.04366v1 第 2–4 页为先。论文采用 ε prediction、DDPM/DDIM，表 II 比较 50 NFE 与 6 NFE；正文没有说明 N=100 是必要的独立 trajectory 数量。公开源码中的默认 N 只作为实现线索，本轮通过固定预算曲线确定 N。

论文文件与参考代码定位见[参数核对记录](respdiff_parameter_alignment_20261003.md)。该记录的历史 DDPM50×100 默认值由本协议在此任务作用域内覆盖，不作为本轮执行要求。主模型为[双低通三臂结果](respdiff_bcg_baseband_v1_results_20261007.md)的 source-equivalent，seed=20260811，final update=6400；checkpoint SHA256 固定为 `08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6`。

source-equivalent 只描述既有 `mean(epsilon residual²)+(0.01/128)*full-spectrum FFTmagMSE` 的相对 loss balance，不宣称等同作者未公开实现。本次只改变推理协议；不改变 checkpoint 参数化、训练 loss、数据 split 或选模。

## 数据与保幅重建

保留 BCG→THO、soft-z、parent-level BCG/THO 双 1 Hz LPF、20 Hz、180 s parent、30 s chunk、15 s hop、13 chunks、reflect padding、periodic Hann OLA 和 Fourier interpolation 至 100 Hz。输入及 target 的预处理与 source run 相同，validation 准入和顺序须与 source run 的 `val_parents.csv` 一致。

每个 chunk 的 trajectory predictions 先取在线均值，再进行 Hann OLA、完整 180 s 插值至 100 Hz，最后对 prediction 做 parent 1 Hz LPF。后滤波复用 preprocessing 的 8 阶 Butterworth SOS、`sosfiltfilt(padtype='odd',padlen=27)`，float64 计算后转 FP32。仅在完整 parent 上执行，不在 chunk 或每个 reverse step 中执行。

主评价用 postfiltered prediction，同时完整保存 raw/postfiltered 波形、五项公共指标和幅值/能量诊断。所有操作保留模型生成幅值，不对 prediction 使用 clamp、输出激活范围限制、min-max、target-dependent rescale 或 target std 校正。后滤波是本任务定义的推理步骤；论文没有明确报告该额外 inference postfilter，不把它写成论文明确声明的实验设置。

## sampler 与 trajectory

- DDIM：固定 timesteps=[49,39,29,19,9,0]、eta=0，每条 trajectory 6 次 denoiser。末步干净状态 alpha_bar=1。
- DDPM：timesteps=49…0，每条 trajectory 50 次 denoiser；标准 posterior variance `beta_t*(1-alpha_bar_(t-1))/(1-alpha_bar_t)`，t=0 不再加噪，复用已有 reverse-step 公式。
- NFE 指每条 trajectory 的调用数，N 指独立 trajectory 数。DDIM6 N8 为 48 calls/chunk，不把 batch forward 与逐 chunk 调用预算混淆。
- initial noise identity 包含 split、parent row、trajectory_id 和 noise_seed；基础 seed=20261003。trajectory 0 映射至旧 `parent_noise` 身份，数值完全兼容。其余 trajectory 使用独立 v2 hash namespace。
- 每个 parent/trajectory 使用独立 4200 点标准 Gaussian 场，其 13 chunks 对应重叠切片，保持 overlap initial noise 一致。DDPM 每个 reverse step 的独立 Gaussian 场还将 step 写入身份；t=0 不抽噪。
- N=k 的前 k 条与更大 N 的前 k 条相同，噪声不依赖 batch 划分。推理 batch 固定 64；模型 BN 使用当前 batch 统计，不能声称 prediction 对 batch 划分不变。
- 在线 FP64 sum 后转 FP32 mean；不保存完整 trajectory 维 tensor。子集仅保留 N 网格上的累计均值以重建各设置。

## 固定 diagnostic subset 和预算

按 source validation rows 的顺序，用 `linspace(0,n_parents-1,64,dtype=int)` 等距取 64 parent，再与 row 10382/12226 的身份取并集，按原顺序排序。两个 row 若已在等距集合中不重复，否则总数可为 65/66。不使用指标挑选样本。选择清单、source parent indices 和 chunk 清单写入新 run。

执行 DDIM6 N={1,2,4,8,16} 以及 DDPM50 N={1,2,4}；共享同一模型、condition、基础 seed 和 initial trajectory identities。DDPM 不扩大到 N>4，不做完整 validation ensemble。

每个 sampler 只采样至最大 N，一次生成同时记录各 nested prefix mean；不重复计算已经生成的前缀。记录每个 N 的累计前缀采样墙钟、NFE×N calls/chunk、batched forward 总调用次数。墙钟包括噪声生成、GPU 同步、在线均值及前缀拷贝，reconstruction/metrics 时间另记；不能把共享执行总时间当作某个 N 的独立单次总耗时。

## 固定 N 选择

使用主结果 postfiltered 曲线；P99 为该 subset 逐 parent 原始幅值 max 的 P99，top1% 为逐 parent 时间点能量集中度的均值，PCC/Whole RR 为公共逐 parent 均值。

从 N=2 起检查 N→2N，依次 2→4、4→8、8→16。要求四项同时满足：P99 相对变化 <5%，top1% 相对变化 <5%，PCC 绝对变化 <0.01，Whole RR MAE 相对变化 <5%。相对变化为 `abs(next-current)/abs(current)`；零分母且两者同为零时记 0，零分母但 next 非零时记无定义并判不满足。严格使用 `<`。

选择满足条件的最小较小 N。若三对都不满足，记录 `ensemble not converged by N=16`，选 N16 作为预算上限完成一次 full validation，不宣称收敛。N1 只作曲线起点，不能因其计算便宜被选为正式 N。

完整 validation 仅执行该选定 N 的 DDIM6、一次；读取已完成 subset receipt 的 selection 与曲线，核验哈希和规则，不重新挑 checkpoint 或阈值。其输出与旧 N1/raw 的历史结果按推理合同区分，不覆盖历史文件。

## 指标、产物与图

五项公共指标：Whole RR MAE、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。保存 eligibility/计数，非有限关键指标显式失败。

raw/postfiltered 各记录 parent max P95/P99/global max、prediction/target std mean/median、top1% 时间点能量均值、>1 Hz 与呼吸带能量比。top1% 与频谱能量定义复用 baseband 输出诊断，不引入幅值修正。零能量的比例保存 null 与有效数；正式 reference std=0 则显式失败。

根目录输出 `ensemble_curve.csv/json`、`runtime_profile.csv`、`selection.json`、固定 row NPZ、五类图（metrics_vs_N、max_amplitude_vs_N、runtime_vs_N、两条 row 的 progression）。每个 `ddim_Nk`/`ddpm_Nk` 子目录保存 raw/postfiltered waveform NPZ、diagnostics CSV、主 validation_summary、raw summary、逐 parent metrics、两个 row NPZ。full-validation 根目录另提供主结果波形、诊断和 summary 的入口。

源码快照、sampler 配置、环境与命令、数据 source stats/索引 hash、checkpoint hash/元数据和 inference 前后参数+buffer hash 均保存；运行前后 checkpoint 字节不变。每次必须使用新输出目录，失败 lifecycle 保留，receipt 列出所有嵌套文件哈希。

## 入口与执行

```bash
cd /home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
"$PY" scripts/run_respdiff_bcg_epsilon_inference_v2.py synthetic-smoke \
  --output /tmp/respdiff_epsilon_inference_v2_smoke_v1
"$PY" scripts/run_respdiff_bcg_epsilon_inference_v2.py subset --device cuda:0 \
  --source-run runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1 \
  --output runs/respdiff_bcg_epsilon_inference_v2/subset_seed20261003_v1
"$PY" scripts/run_respdiff_bcg_epsilon_inference_v2.py full-validation --device cuda:0 \
  --source-run runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1 \
  --subset-run runs/respdiff_bcg_epsilon_inference_v2/subset_seed20261003_v1 \
  --output runs/respdiff_bcg_epsilon_inference_v2/full_validation_seed20261003_v1
```

synthetic-smoke 使用小网络、两个人工 parent、B8、DDIM N={1,2,4}/DDPM N={1,2}；不属于 source checkpoint 科研证据，不访问真实数据。正式模式配置在 `configs/respdiff_bcg_epsilon_inference_v2/{ddim,ddpm}.yaml`，固定预算不接受 CLI 扩大。

本阶段的数值比较与实验完成状态以新 run receipt 和后续结果记录为准。

2026-10-07 CPU 验证：`PYTHONPATH=. "$PY" -m pytest tests/test_respdiff_bcg_sampling.py -q`，10 passed in 34.90s。覆盖 N1 与历史 DDIM 精确一致、trajectory0 噪声兼容、trajectory 独立与 batch 划分一致、nested N4/N8、online/stack mean、两种 step 序列及 DDPM posterior 公式、完整 parent 后滤波、无范围限制的有限输出、checkpoint 字节与参数不变、固定 subset、严格 plateau 阈值和 synthetic 全链路。首轮参考断言中 FP32 除法/乘法求值顺序产生差异；统一为公式的逆系数乘法后通过，sampler 更新公式未因此修改。合成产物位于 `/tmp/respdiff_inference_v2_tests_20261007_v2`。
