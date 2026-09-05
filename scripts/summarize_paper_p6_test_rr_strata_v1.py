from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.p6_rr_strata import summarize_test_rr_strata


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 P6 独立测试集 RR 区间指标")
    parser.parse_args()
    command = " ".join(shlex.quote(value) for value in sys.argv)
    output = summarize_test_rr_strata(repo_root=REPO_ROOT, command=command)
    print(output)


if __name__ == "__main__":
    main()
