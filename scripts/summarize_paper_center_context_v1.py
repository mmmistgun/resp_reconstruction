from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.center_context_summary import run_p3_validation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 P3 单 seed validation 方向性汇总")
    parser.parse_args()
    command = shlex.join([sys.executable, *sys.argv])
    receipt = run_p3_validation_summary(repo_root=REPO_ROOT, command=command)
    print(receipt)


if __name__ == "__main__":
    main()
