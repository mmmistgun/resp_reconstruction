#!/usr/bin/env python3
"""冻结 W0 逐窗口指标的 E3 描述性再分析入口。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e3_metric_association import prepare_lock, run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare-lock', 'analyze'))
    args = parser.parse_args()
    print(prepare_lock() if args.phase == 'prepare-lock' else run_analysis())


if __name__ == '__main__':
    main()
