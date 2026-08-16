from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.tf_v1_research_test_summary import generate_research_test_summary


if __name__ == "__main__":
    print(generate_research_test_summary().resolve())

