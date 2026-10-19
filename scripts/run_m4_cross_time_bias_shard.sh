#!/usr/bin/env bash
# 一张空闲 GPU 串行完成三 seed 的 Δt 消融，成功后自动比较原始 M4。
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "用法: bash $0 SESSION GPU_INDEX" >&2
  exit 2
fi
case "$2" in 0|1) ;; *) echo "GPU_INDEX 应为 0 或 1" >&2; exit 2;; esac
M4_TIME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
M4_TIME_PYTHON="$M4_TIME_ROOT/.venv/bin/python"
cd "$M4_TIME_ROOT"
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_TIME_PYTHON" -u scripts/run_m4_cross_time_bias.py run --session "$1" --device "cuda:$2"
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_TIME_PYTHON" -u scripts/run_m4_cross_time_bias.py summarize --session "$1"
