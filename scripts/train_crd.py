from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages
from resp_train.crd.config import check_crd_dependencies, load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.model import CRD_TF_P6_VARIANTS, CRD_TF_VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description="训练冻结版 CRD-v1.1 S0/S1 模型")
    parser.add_argument("--config", required=True, help="configs/crd_v1 下的配置文件")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()
    problems = check_required_packages() + check_crd_dependencies()
    if problems:
        raise SystemExit("CRD 环境不满足冻结要求: " + "; ".join(problems))
    cfg = load_crd_config(args.config, overrides=args.overrides)
    if str(cfg.model.variant) in CRD_TF_VARIANTS and str(cfg.protocol.run_role) == "formal":
        from resp_train.crd.tf_v1_p4 import validate_p4_formal_preflight

        validate_p4_formal_preflight()
    if str(cfg.model.variant) in CRD_TF_P6_VARIANTS and str(cfg.protocol.run_role) == "formal":
        raise SystemExit("CRD-TF P6a formal 尚未开放：必须先完成并冻结 P6a CUDA synthetic 与 batch-128 acceptance")
    print(CRDExperiment(cfg).train())


if __name__ == "__main__":
    main()
