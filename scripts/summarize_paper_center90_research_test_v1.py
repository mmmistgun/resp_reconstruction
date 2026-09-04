from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.center90_research_test_summary import run_research_test_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 center90 三 seed 独立测试集上下文尺度汇总")
    parser.parse_args()
    command = shlex.join([sys.executable, *sys.argv])
    print(run_research_test_summary(repo_root=REPO_ROOT, command=command))


if __name__ == "__main__":
    main()
