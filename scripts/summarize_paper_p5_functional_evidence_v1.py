from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.p5_functional_evidence import run_p5_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="只读整理冻结 W0 P−1 时频功能证据")
    parser.parse_args()
    print(run_p5_summary(repo_root=REPO_ROOT, command=shlex.join([sys.executable, *sys.argv])))


if __name__ == "__main__":
    main()
