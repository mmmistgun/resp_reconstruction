from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies
from resp_train.paper_evidence.context_length_research_test_cache import (
    build_context_length_research_test_w_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 center30+center60 联合 research-test input-only W cache")
    parser.add_argument("--confirm-research-test-cache-build", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_test_cache_build:
        raise SystemExit("上下文 research-test W cache 需要显式 --confirm-research-test-cache-build")
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("上下文 research-test cache 环境不满足冻结要求: " + "; ".join(problems))
    print(build_context_length_research_test_w_cache(show_progress=True))


if __name__ == "__main__":
    main()
