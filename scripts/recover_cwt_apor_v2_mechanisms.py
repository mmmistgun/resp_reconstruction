"""通过可追溯的预测字段适配恢复已冻结CWT-APOR机制评价。"""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import argparse
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NOTE = ROOT / "docs/experiments/cwt_apor_v2_mechanisms_recovery_20261002.md"
TEST = ROOT / "tests/test_cwt_apor_v2_mechanisms_recovery.py"


@contextmanager
def prediction_keys():
    """只为机制模块绑定标准预测/参考字段名，退出后恢复原函数引用。"""
    from resp_train.engine.train import _prediction_dict_from_arrays
    from resp_train.paper_evidence.cwt_apor_v2 import interventions
    original = interventions._prediction_dict_from_arrays
    if original is not _prediction_dict_from_arrays:
        raise RuntimeError("机制预测构造器已被其他适配修改")
    interventions._prediction_dict_from_arrays = partial(
        original, pred_key="r_tho_hat", target_key="tho_ref")
    try:
        yield
    finally:
        interventions._prediction_dict_from_arrays = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error("恢复机制分析需要当次 --confirm-research-test")
    from resp_train.paper_evidence.cwt_apor_v2 import artifacts as io, interventions, research_test as test
    from resp_train.paper_evidence.cwt_apor_v2.spec import SEEDS
    session = args.session.resolve()
    io.load_session(session)
    summary = io.completed(session / "research_test/summary", test.test_key(session, "test_summary"))
    if summary is None:
        raise RuntimeError("必须已有完整60-cell test汇总")

    def stop(signum, _):
        raise KeyboardInterrupt(f"收到停止信号{signum}")
    signal.signal(signal.SIGTERM, stop)
    identities = {str(p.relative_to(ROOT)): io.identity(p) for p in (Path(__file__).resolve(), NOTE, TEST)}
    original_loader = interventions.evaluate_loader

    def recorded_loader(model, loader, cfg, rep, shifts, case_ids, output, seed):
        # 原源码及其冻结快照保持不变；每个新attempt独立登记实际生效的增量适配。
        for relative, identity in identities.items():
            source = ROOT / relative
            io.verify(source, identity)
            (output / source.name).write_bytes(source.read_bytes())
        io.write_json(output / "recovery_source.json", {
            "files": identities, "change": "bind_prediction_keys",
            "pred_key": "r_tho_hat", "target_key": "tho_ref",
            "base_session": io.identity(session / "session.json"),
        })
        with prediction_keys():
            return original_loader(model, loader, cfg, rep, shifts, case_ids, output, seed)

    key = test.test_key(session, "mechanisms_recovery", adapter=identities)
    with io.attempt(session / "research_test/mechanisms_recovery", key) as output:
        print(f"RECOVERY={output}", flush=True)
        io.write_json(output / "authorization.json", {"user_instruction": "test跑完了，看看机制分析报错了",
                      "prior_test_authorization": "开启test", "scope": "three_seed_mechanisms_and_summary",
                      "existing_test_summary": str(summary), "files": identities, "command": sys.argv})
        interventions.evaluate_loader = recorded_loader
        try:
            for index, seed in enumerate(SEEDS):
                parent = session / "research_test/mechanisms" / f"seed_{seed}"
                failures = sorted(parent.glob("attempt_*/failed.json"))
                for failure in failures:
                    trace = io.read_json(failure)["traceback"]
                    if "missing 2 required keyword-only arguments: 'pred_key' and 'target_key'" not in trace:
                        raise RuntimeError(f"发现未纳入本次修复的失败，先核查: {failure}")
                print(f"PHASE_START=mechanisms_{seed}", flush=True)
                result = test.mechanisms(session, seed, f"cuda:{index % 2}", confirmed=True, retry=bool(failures))
                io.write_json(output / f"seed_{seed}.json", {"path": str(result), "manifest": io.identity(result / "manifest.json"),
                              "prior_failures": {str(p): io.identity(p) for p in failures}})
                print(f"PHASE_COMPLETE=mechanisms_{seed} OUTPUT={result}", flush=True)
            result = test.summarize_mechanisms(session)
            io.write_json(output / "completion.json", {"completed": True, "mechanisms_summary": str(result)})
        finally:
            interventions.evaluate_loader = original_loader
    print("COMPLETED=mechanisms_recovery", flush=True)


if __name__ == "__main__":
    main()
