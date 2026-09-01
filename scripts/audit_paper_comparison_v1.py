from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.paper_evidence.comparison_audit import run_p0_comparison_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="执行论文代表性方法只读兼容性审计")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument(
        "--output-dir",
        default="runs/paper_evidence_v1/p0_comparison_audit",
    )
    args = parser.parse_args()
    manifest = run_p0_comparison_audit(
        source_manifest_path=args.source_manifest,
        output_dir=args.output_dir,
        repo_root=REPO_ROOT,
        require_clean_git=True,
    )
    print(manifest)


if __name__ == "__main__":
    main()
