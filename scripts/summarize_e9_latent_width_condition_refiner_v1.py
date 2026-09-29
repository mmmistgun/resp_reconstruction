from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_summary import (
    run_summary,
    verify_summary_attempt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E9完整validation一次性汇总")
    parser.add_argument("--confirm-validation-summary", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_summary_attempt(args.verify)
    else:
        if not args.confirm_validation_summary:
            raise SystemExit("E9 summary要求显式传入--confirm-validation-summary")
        result = str(run_summary())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

