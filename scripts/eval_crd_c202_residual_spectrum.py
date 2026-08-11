from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.crd.decoder_diagnostics import DEFAULT_OUTPUT_ROOT, evaluate_c202_residual_spectrum


def main() -> None:
    parser = argparse.ArgumentParser(description="C202 validation selected-checkpoint residual 带外能量描述")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    print(
        evaluate_c202_residual_spectrum(
            checkpoint_path=args.checkpoint,
            device=args.device,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()

