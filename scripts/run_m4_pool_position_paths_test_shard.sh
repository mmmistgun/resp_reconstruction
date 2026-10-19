#!/usr/bin/env bash
# 单张空闲 GPU 串行评价全部十二个固定 checkpoint，成功后自动汇总。
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "用法: bash $0 ALLOWLIST GPU_INDEX" >&2
  exit 2
fi
case "$2" in 0|1) ;; *) echo "GPU_INDEX 应为 0 或 1" >&2; exit 2;; esac
M4_TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
M4_TEST_PYTHON="$M4_TEST_ROOT/.venv/bin/python"
cd "$M4_TEST_ROOT"
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_TEST_PYTHON" -u scripts/eval_m4_pool_position_paths.py evaluate \
  --allowlist "$1" --device "cuda:$2" --confirm-research-test
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_TEST_PYTHON" -u scripts/eval_m4_pool_position_paths.py summarize \
  --allowlist "$1" --confirm-research-test
