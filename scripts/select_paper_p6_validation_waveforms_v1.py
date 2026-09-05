from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.p6_multi_attribute import select_p6_validation_waveforms


def main() -> None:
    parser = argparse.ArgumentParser(description="按冻结规则选择五个 P6 validation 波形 rows")
    parser.parse_args()
    output = select_p6_validation_waveforms(
        repo_root=REPO_ROOT,
        command=shlex.join([sys.executable, *sys.argv]),
    )
    print(output)


if __name__ == "__main__":
    main()
