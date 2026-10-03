# CWT-APOR v2运行顺序

日期：2026-10-01。状态：已修订；用户明确要求“修订完成后启动”，已授权串联新的GPU验收与60-cell正式train/validation。实际阶段以后台日志与新会话manifest为准。研究矩阵、80/30/15/0停止合同及复用边界见[协议](cwt_apor_v2_protocol_20261001.md)。

```bash
cd /home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction
CWT_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
export PYTHONPATH=.
```

CPU定向测试和配置检查：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$CWT_PY" -m pytest tests/test_cwt_apor_v2.py -q
"$CWT_PY" scripts/run_cwt_apor_v2.py plan
```

本次授权的串联入口，复用已完成的模型无关产物，APOR验收通过后自动交接正式阶段：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$CWT_PY" -u scripts/run_cwt_apor_v2_engineering.py --devices cuda:0 cuda:1 --confirm-synthetic --then-formal --confirm-training \
  --preprocessing-session /home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction/runs/cwt_time_frequency_v1/session_20260930T184031Z_812b424a1c92
```

也可用`run_cwt_apor_v2.py prepare-reuse --source-session ...`单独建立仅合成会话，再显式运行`gpu-acceptance`。这两个动作均不复用W0的GPU回执，不启动正式训练。

若采用分步入口，APOR验收通过后可用以下等价正式命令（串联入口会自动执行，无需重复启动）：

```bash
APOR_ENGINEERING_SESSION='/新的APOR验收session绝对路径'
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$CWT_PY" -u scripts/run_cwt_apor_v2_formal.py --engineering-session "$APOR_ENGINEERING_SESSION" \
  --devices cuda:0 cuda:1 --confirm-training
```

正式交接会新建train/validation会话，保留缓存/信号分析的只读引用。准备阶段验证既有产物后跳过已完成计算，随后按交错分片执行60-cell APOR矩阵并汇总。任何身份不一致或数值失败都会停止后续步骤，不自动重训、减小batch或缩减矩阵。

Research-test继续使用固定完整矩阵checkpoint allowlist与当次`--confirm-research-test`，本次未开放。
