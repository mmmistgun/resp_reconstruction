from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.crd.prototype_diagnostics import (
    DEFAULT_PROTOTYPE_OUTPUT_ROOT,
    evaluate_morphology_prototypes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD_204 validation prototype usage/entropy 描述性审计")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=str(DEFAULT_PROTOTYPE_OUTPUT_ROOT))
    args = parser.parse_args()
    print(
        evaluate_morphology_prototypes(
            checkpoint_path=args.checkpoint,
            device=args.device,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
