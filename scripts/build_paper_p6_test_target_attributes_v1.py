from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.p6_rr_strata import build_test_target_attributes


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 P6 独立测试集 target-only RR 属性")
    parser.add_argument("--confirm-test-target-attribute-build", action="store_true")
    args = parser.parse_args()
    if not args.confirm_test_target_attribute_build:
        parser.error("必须显式传入 --confirm-test-target-attribute-build")
    command = " ".join(shlex.quote(value) for value in sys.argv)
    output = build_test_target_attributes(repo_root=REPO_ROOT, command=command)
    print(output)


if __name__ == "__main__":
    main()
