from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.center30_p4s3_summary import run_p4s3_validation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 center30 P4-S3 三 seed validation 相对变化汇总")
    parser.parse_args()
    command = shlex.join([sys.executable, *sys.argv])
    print(run_p4s3_validation_summary(repo_root=REPO_ROOT, command=command))


if __name__ == "__main__":
    main()
