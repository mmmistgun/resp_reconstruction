from __future__ import annotations

import json

from resp_train.paper_evidence import e7_scale_encoding_aggregation as e7


def test_closed_e7_lock_accepts_frozen_mainline_compatibility() -> None:
    lock, digest = e7.load_implementation_lock()
    compatibility_path = e7.ROOT / e7.MAINLINE_COMPATIBILITY_PATH
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))

    assert digest == compatibility["base_implementation_lock"]["sha256"]
    assert compatibility["scientific_contract_changed"] is False
    assert compatibility["experiment_rerun_authorized"] is False
    assert set(compatibility["amended_code_files"]) == {
        str(e7.CONTROL_PATH),
        "resp_train/crd/experiment.py",
    }
    assert lock["status"] == "p1_implemented_not_run"
