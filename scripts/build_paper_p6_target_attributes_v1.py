from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.p6_multi_attribute import build_p6_target_attributes


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 P6 train/validation target-only 属性与 train-frozen RR cutpoints")
    parser.add_argument("--confirm-target-attribute-build", action="store_true")
    args = parser.parse_args()
    if not args.confirm_target_attribute_build:
        raise SystemExit("必须显式传入 --confirm-target-attribute-build")
    output = build_p6_target_attributes(
        repo_root=REPO_ROOT,
        command=shlex.join([sys.executable, *sys.argv]),
    )
    print(output)


if __name__ == "__main__":
    main()
