#!/usr/bin/env bash
# 两卡评价三个固定 checkpoint，两个分片均成功后生成完整 test/validation 对照。
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "用法: bash $0 ALLOWLIST GPU_FIRST GPU_SECOND" >&2
  exit 2
fi
for M4_LINEAR_TEST_GPU in "$2" "$3"; do
  case "$M4_LINEAR_TEST_GPU" in 0|1) ;; *) echo "GPU 编号应为 0 或 1" >&2; exit 2;; esac
done
if [[ "$2" == "$3" ]]; then
  echo "两个分片应使用不同 GPU" >&2
  exit 2
fi
M4_LINEAR_TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$M4_LINEAR_TEST_ROOT"
if [[ ! -f "$1/allowlist_receipt.json" ]]; then
  echo "请先 prepare allowlist" >&2
  exit 2
fi
if [[ -e "$1/gpu_$2.log" || -L "$1/gpu_$2.log" || -e "$1/gpu_$3.log" || -L "$1/gpu_$3.log" ]]; then
  echo "GPU 日志已存在；保留日志并停止启动" >&2
  exit 2
fi
set -C
exec 3>"$1/gpu_$2.log"
exec 4>"$1/gpu_$3.log"
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -u scripts/eval_m4_linear_fusion.py evaluate --allowlist "$1" --device "cuda:$2" \
  --shard-index 0 --shard-count 2 --confirm-research-test >&3 2>&1 &
M4_LINEAR_TEST_PID_FIRST=$!
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -u scripts/eval_m4_linear_fusion.py evaluate --allowlist "$1" --device "cuda:$3" \
  --shard-index 1 --shard-count 2 --confirm-research-test >&4 2>&1 &
M4_LINEAR_TEST_PID_SECOND=$!
exec 3>&- 4>&-
M4_LINEAR_TEST_STATUS_FIRST=0
M4_LINEAR_TEST_STATUS_SECOND=0
wait "$M4_LINEAR_TEST_PID_FIRST" || M4_LINEAR_TEST_STATUS_FIRST=$?
wait "$M4_LINEAR_TEST_PID_SECOND" || M4_LINEAR_TEST_STATUS_SECOND=$?
if (( M4_LINEAR_TEST_STATUS_FIRST != 0 || M4_LINEAR_TEST_STATUS_SECOND != 0 )); then
  echo "worker 退出状态: $2=$M4_LINEAR_TEST_STATUS_FIRST, $3=$M4_LINEAR_TEST_STATUS_SECOND；失败产物保留" >&2
  if (( M4_LINEAR_TEST_STATUS_FIRST != 0 )); then exit "$M4_LINEAR_TEST_STATUS_FIRST"; fi
  exit "$M4_LINEAR_TEST_STATUS_SECOND"
fi
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -u scripts/eval_m4_linear_fusion.py summarize --allowlist "$1" --confirm-research-test
