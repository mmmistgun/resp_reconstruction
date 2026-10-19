#!/usr/bin/env bash
# 两个 GPU 分片完成三个 seed，两个 worker 均成功后生成完整配对汇总。
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "用法: bash $0 SESSION GPU_FIRST GPU_SECOND" >&2
  exit 2
fi
for M4_LINEAR_GPU in "$2" "$3"; do
  case "$M4_LINEAR_GPU" in
    0|1) ;;
    *) echo "GPU 编号应为 0 或 1" >&2; exit 2 ;;
  esac
done
if [[ "$2" == "$3" ]]; then
  echo "两个 worker 应使用不同 GPU" >&2
  exit 2
fi
M4_LINEAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
M4_LINEAR_PYTHON="$M4_LINEAR_ROOT/.venv/bin/python"
cd "$M4_LINEAR_ROOT"
M4_LINEAR_SESSION="$1"
if [[ ! -f "$M4_LINEAR_SESSION/session.json" ]]; then
  echo "请先 prepare session: $M4_LINEAR_SESSION" >&2
  exit 2
fi
M4_LINEAR_LOG_FIRST="$M4_LINEAR_SESSION/gpu_$2.log"
M4_LINEAR_LOG_SECOND="$M4_LINEAR_SESSION/gpu_$3.log"
if [[ -e "$M4_LINEAR_LOG_FIRST" || -L "$M4_LINEAR_LOG_FIRST" || -e "$M4_LINEAR_LOG_SECOND" || -L "$M4_LINEAR_LOG_SECOND" ]]; then
  echo "GPU 日志已存在；保留现有日志并停止启动" >&2
  exit 2
fi
# noclobber 原子创建两个日志，防止并发启动覆盖日志。
set -C
exec 3>"$M4_LINEAR_LOG_FIRST"
exec 4>"$M4_LINEAR_LOG_SECOND"
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_LINEAR_PYTHON" -u scripts/run_m4_linear_fusion.py run --session "$M4_LINEAR_SESSION" \
  --shard-index 0 --shard-count 2 --device "cuda:$2" >&3 2>&1 &
M4_LINEAR_PID_FIRST=$!
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_LINEAR_PYTHON" -u scripts/run_m4_linear_fusion.py run --session "$M4_LINEAR_SESSION" \
  --shard-index 1 --shard-count 2 --device "cuda:$3" >&4 2>&1 &
M4_LINEAR_PID_SECOND=$!
exec 3>&- 4>&-

# 独立收集两个退出状态；一个分片失败时，仍等待另一个分片落盘。
M4_LINEAR_STATUS_FIRST=0
M4_LINEAR_STATUS_SECOND=0
wait "$M4_LINEAR_PID_FIRST" || M4_LINEAR_STATUS_FIRST=$?
wait "$M4_LINEAR_PID_SECOND" || M4_LINEAR_STATUS_SECOND=$?
if (( M4_LINEAR_STATUS_FIRST != 0 || M4_LINEAR_STATUS_SECOND != 0 )); then
  echo "worker 退出状态: GPU $2=$M4_LINEAR_STATUS_FIRST, GPU $3=$M4_LINEAR_STATUS_SECOND；日志和失败产物已保留" >&2
  if (( M4_LINEAR_STATUS_FIRST != 0 )); then
    exit "$M4_LINEAR_STATUS_FIRST"
  fi
  exit "$M4_LINEAR_STATUS_SECOND"
fi
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_LINEAR_PYTHON" -u scripts/run_m4_linear_fusion.py summarize --session "$M4_LINEAR_SESSION"
