from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.tf_v1_p4 import current_state_path, state_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="查看当前干净 commit 的 CRD-TF P4 matrix 状态")
    parser.add_argument("--state", default=None, help="可显式指定历史 matrix_state.json")
    args = parser.parse_args()
    path = Path(args.state).resolve() if args.state else current_state_path()
    if not path.is_file():
        raise SystemExit(f"matrix state 尚不存在: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    payload = state_summary(state)
    payload["state_path"] = str(path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
