#!/usr/bin/env bash
# 两张 GPU 各串行运行 9 个 cell。
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "用法: bash $0 SESSION SHARD_INDEX GPU_INDEX [--resume]" >&2
  exit 2
fi
case "$2" in 0|1) ;; *) echo "SHARD_INDEX 应为 0 或 1" >&2; exit 2;; esac
case "$3" in 0|1) ;; *) echo "GPU_INDEX 应为 0 或 1" >&2; exit 2;; esac
if [[ $# -eq 4 && "$4" != --resume ]]; then
  echo "第四个参数只能为 --resume" >&2
  exit 2
fi
M4_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
M4_PYTHON="${M4_PYTHON:-/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python}"
cd "$M4_ROOT"
exec env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_PYTHON" -u scripts/run_m4_residual.py run --session "$1" \
  --shard-index "$2" --shard-count 2 --device "cuda:$3" "${@:4}"
