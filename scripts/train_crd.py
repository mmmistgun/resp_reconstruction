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
from resp_train.crd.model import CRD_TF_P6_VARIANTS, CRD_TF_VARIANTS, CRD_TF_W_V2_VARIANTS


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
        from resp_train.crd.tf_v1_p6a_preflight import validate_p6a_formal_preflight

        validate_p6a_formal_preflight()
    if str(cfg.model.variant) in CRD_TF_W_V2_VARIANTS:
        from resp_train.crd.tf_w_v2 import P1_VARIANTS, P3_VARIANTS

        if str(cfg.model.variant) in P1_VARIANTS:
            from resp_train.crd.tf_w_v2_p1 import validate_p1_training_preflight

            validate_p1_training_preflight(cfg)
        elif str(cfg.model.variant) in P3_VARIANTS:
            from resp_train.crd.tf_w_v2_p3 import validate_p3_training_preflight

            validate_p3_training_preflight(cfg)
    print(CRDExperiment(cfg).train())


if __name__ == "__main__":
    main()
