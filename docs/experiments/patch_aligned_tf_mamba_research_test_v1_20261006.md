# Patch-aligned TF-Mamba 固定 checkpoint research-test 协议

协议 ID：`patch-aligned-tf-mamba-research-test-v1-20261006`。当前授权为实现与来源准备；实际 research-test 由用户执行带 `--confirm-research-test` 的入口。训练和 validation 已关闭，当前结果入口为 `patch_aligned_tf_mamba_validation_results_20261006.md`。

## 固定范围

评价全部 1/2/4 秒 patch × seeds 20260811/20260812/20260813，共 9 个 checkpoint。每组只取完整 validation Local RR MAE 严格最小、并列最早的 epoch。固定 epoch：1 秒为 5/12/9，2 秒为 5/6/8，4 秒为 33/4/3。

训练 session 为 `runs/patch_aligned_tf_mamba/formal_v1_20261005`，SHA-256 为 `ad34f3a03680cadd88000ee7bcf3c2a7053fd48d501cdb9153b2e4062f07ae3c`。validation summary receipt SHA-256 为 `5394a9187db44588dbc1649f0c592982f742318b428c3d5a438607473996fd7f`。准备 allowlist 时重新核对全部完成回执、完整 validation history、选点、模型 state、配置和 development 行身份。

本测试集曾用于既有研究开发，证据属性为 **reused research/development evidence**。本轮用于固定矩阵跨 split 复核，不作为未经开发使用的新确认集。test 不参与结构、checkpoint、阈值、指标或剩余矩阵调整。

## 样本、条件与推理

固定 test=2310 窗口、8 名受试者、180 秒/100 Hz，sample seed=20260612。有序 row ID 字节 SHA-256 为 `184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。dataset index 沿用训练来源记录的路径与 SHA-256。验证 test 与 train/validation 受试者集合零交集。

复用原生冻结 test W cache，identity=`40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`，manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。原 reader 校验文件哈希、shape、dtype 和 row identity；独立数据 adapter 按实际频率选择 `(0.8,8] Hz` 的 41 行，要求与 checkpoint 中频率坐标逐值相等。

逐 batch 校验 row 顺序及 `(B,1,18000)` 波形/参考、`(B,41,360)` 条件。非有限 input、target、prediction、checkpoint 和关键指标显式失败。prediction degeneracy 单独记录，保持样本分母。

官方模型 strict 加载，eval 模式，batch=64、BF16，尾批完整保留。冻结训练配置只在评价副本中调整设备及显示选项；数据工厂负责原始波形/参考，独立包装器提供 H-CWT。训练侧 reader 继续只接受 train/val。

## 指标与汇总

五项主指标沿用 validation：Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。开启原生 test 辅助指标：IBI/coverage、coherence、constrained nDTW、分层 envelope Spearman。保留所有资格、分母与退化标记。

完整 9-cell 后生成逐 seed、三 seed mean/SD、逐受试者与 subject-macro、Local RR tail、资格分母和原生辅助汇总。预设对比为 1 秒、4 秒分别相对 2 秒，同 seed 配对差；误差下降和 PCC 上升均记为正改善。不同 cell 的 row identity 与 target 资格须完全一致。

## 生命周期与命令

实现位于独立脚本，保持训练源码与协议字节身份。`prepare` 只读取 development 产物和代码，不读取 test manifest、波形、目标或 cache 数组。allowlist 固定全部 checkpoint、配置、环境、实际频率和 test 合同，并保存源码快照。同一路径准备成功后复用；评价前校验 allowlist、代码及来源身份。

评价成功产物校验后复用；未完成 attempt 经检查后以 `--retry-failed` 创建独立重试。每个 cell 使用互斥文件锁。两张 GPU 按训练矩阵顺序分成 5/4 两组，每卡串行评价；每组失败即停止该分片，全部成功才允许完整汇总。

在 worktree 根目录准备一次：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_patch_aligned_tf_mamba.py prepare --training-session runs/patch_aligned_tf_mamba/formal_v1_20261005 --output runs/patch_aligned_tf_mamba/research_test_v1_20261006
```

两个终端均进入上述目录，各自执行：

```bash
# GPU 0
bash scripts/run_patch_tf_test_shard.sh runs/patch_aligned_tf_mamba/research_test_v1_20261006 0 0
```

```bash
# GPU 1
bash scripts/run_patch_tf_test_shard.sh runs/patch_aligned_tf_mamba/research_test_v1_20261006 1 1
```

脚本内部包含 `env -u LD_LIBRARY_PATH -u LD_PRELOAD` 和 `--confirm-research-test`，执行该脚本即授权其固定分片的 test 访问。手动命令为：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -u scripts/eval_patch_aligned_tf_mamba.py evaluate --allowlist runs/patch_aligned_tf_mamba/research_test_v1_20261006 --shard-index 0 --shard-count 2 --device cuda:0 --confirm-research-test
```

状态与最终汇总：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_patch_aligned_tf_mamba.py status --allowlist runs/patch_aligned_tf_mamba/research_test_v1_20261006
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/eval_patch_aligned_tf_mamba.py summarize --allowlist runs/patch_aligned_tf_mamba/research_test_v1_20261006 --confirm-research-test
```

## 实现验证

2026-10-06：12 项合成 CPU 定向测试通过，覆盖确认门禁、完整 validation 最早并列选点、H-CWT 频率选择及漂移拒绝、batch 身份/shape/finite/完整性、原生 test 辅助指标、完整矩阵和 target 资格核对、成功复用、重试及产物篡改检测。shell 脚本通过 `bash -n`。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q tests/test_patch_aligned_tf_research_test.py
```

本轮实际 test 评价尚未启动。代码定向验证仅使用 synthetic/disposable fixtures。
