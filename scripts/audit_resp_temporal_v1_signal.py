from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.temporal_signal_audit import DEFAULT_CONFIG_PATH, run_train_signal_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RTM-v1 完整 train-only 信号审计；不接受 split、频带、输出目录或抽样覆盖",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    command = shlex.join([sys.executable, *sys.argv])
    receipt = run_train_signal_audit(config_path=args.config, command=command)
    print(f"RTM-v1 train-only signal audit receipt: {receipt}")


if __name__ == "__main__":
    main()
