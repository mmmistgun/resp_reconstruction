from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.harmonized_rhythm_test_summary import run_summary_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结独立测试集节律指标统一补充汇总 v2")
    parser.parse_args()
    print(run_summary_v2(repo_root=REPO_ROOT, command=shlex.join([sys.executable, *sys.argv])))


if __name__ == "__main__":
    main()
