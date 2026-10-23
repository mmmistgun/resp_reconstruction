"""M4-S1冻结入口的JSON绑定兼容层；统一元组/列表表示并保留全部身份校验。"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import m4_s1_cwt_parameters_artifacts as artifacts


@contextmanager
def canonical_stage_bindings():
    """JSON会将元组保存为列表，期待绑定也按相同规则规范化后再完整比较。"""
    original = artifacts.verify_stage

    def verify_stage(path, expected_binding=None):
        canonical = None if expected_binding is None else json.loads(
            json.dumps(expected_binding, ensure_ascii=False, allow_nan=False))
        return original(path, canonical)

    artifacts.verify_stage = verify_stage
    try:
        yield
    finally:
        artifacts.verify_stage = original


if __name__ == "__main__":
    from scripts.run_m4_s1_cwt_parameters import main

    with canonical_stage_bindings():
        main()
