from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.temporal.gpu_engineering import DEFAULT_CONFIG_PATH, run_gpu_engineering


def main() -> None:
    parser = argparse.ArgumentParser(description="RTM-v1 frozen synthetic GPU engineering benchmark")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)))
    args = parser.parse_args()
    command = shlex.join([sys.executable, *sys.argv])
    receipt = run_gpu_engineering(config_path=args.config, command=command)
    print(f"RTM-v1 GPU engineering receipt: {receipt}")


if __name__ == "__main__":
    main()
