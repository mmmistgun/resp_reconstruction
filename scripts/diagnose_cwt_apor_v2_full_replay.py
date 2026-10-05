"""只诊断已失败FULL重放的固定窗口；不修改验收阈值或正式评价结果。"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error("需要 --confirm-research-test")
    from resp_train.paper_evidence.cwt_apor_v2 import artifacts as io, research_test as test
    from resp_train.paper_evidence.cwt_apor_v2.model import build_model
    from resp_train.paper_evidence.cwt_apor_v2.interventions import captured_forward
    from resp_train.metrics.task import TaskMetricConfig, _whole_rr, _periodogram
    session = args.session.resolve()
    frozen = io.load_session(session)
    _, allowed = test.allowlist(session)
    seed, row_id, device = 20260812, 5807, "cuda:0"
    entry = next(e for e in allowed["entries"] if (e["arm"], e["seed"]) == ("A0", seed))
    ref = io.completed(session / f"research_test/evaluation/A0/seed_{seed}", test.test_key(session, "test_evaluation", arm="A0", seed=seed))
    rows = pd.read_csv(ref / "metrics.csv")
    position = int(np.flatnonzero(rows.dataset_row_id.eq(row_id))[0])
    batch_start = position//128*128
    chunk_start = position//8*8
    cfg = OmegaConf.load(entry["sources"]["config"]["path"])
    cfg.training.device = device
    test.require_gpu(session, device)
    key = test.test_key(session, "full_replay_diagnostic", seed=seed, row_id=row_id,
                        script=io.identity(Path(__file__)))
    with io.attempt(session / "research_test/full_replay_diagnostic", key) as output:
        print(f"DIAGNOSTIC={output}", flush=True)
        (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        io.write_json(output / "access.json", {"confirmed": True, "purpose": "fixed_failed_row_batch_and_hook_diagnostic",
                      "row_id": row_id, "position": position, "batch_start": batch_start,
                      "checkpoint": entry["sources"]["checkpoint"], "normal_reference": str(ref)})
        bundle = test.test_bundle(session, "A0", cfg, confirmed=True)
        batch = default_collate([bundle.dataset[i] for i in range(batch_start, batch_start+128)])
        assert int(batch['meta']['dataset_row_id'][position-batch_start]) == row_id
        x, w = batch['x'].to(device), batch['tf']['w'].to(device)
        offset = chunk_start-batch_start
        x8, w8 = x[offset:offset+8], w[offset:offset+8]
        model = build_model(seed, frozen['representations']['A0']).to(device).eval()
        io.verify(entry['sources']['checkpoint']['path'], entry['sources']['checkpoint'])
        checkpoint = torch.load(entry['sources']['checkpoint']['path'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        target = batch['target'][position-batch_start].numpy().ravel()
        saved = np.asarray(np.load(ref/'prediction.npy', mmap_mode='r')[position]).ravel()
        predictions = {'saved_batch128_bf16': saved}
        # 除batch形状、autocast和hook外，输入、权重及设备保持相同。
        for name, xx, ww, amp, capture, inference in [
            ('plain128_bf16', x, w, True, False, False),
            ('plain8_bf16', x8, w8, True, False, False),
            ('plain8_inference_bf16', x8, w8, True, False, True),
            ('captured8_bf16', x8, w8, True, True, True),
            ('plain128_fp32', x, w, False, False, False),
            ('plain8_fp32', x8, w8, False, False, False),
        ]:
            with (torch.inference_mode() if inference else torch.no_grad()), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                prediction = captured_forward(model, xx, ww)[0] if capture else model(xx, tf={'w':ww})['waveform']
            local = position-(batch_start if len(xx)==128 else chunk_start)
            predictions[name] = prediction[local].float().cpu().numpy().ravel()
            del prediction
        protocol = TaskMetricConfig.from_config(cfg)
        target_rr = _whole_rr(target, protocol)
        records = []
        for name, prediction in predictions.items():
            rr = _whole_rr(prediction, protocol)
            power = np.median(np.stack([_periodogram(prediction[i:i+6000], protocol) for i in range(0,12001,3000)]),axis=0)
            frequencies = np.fft.rfftfreq(6000,d=1/protocol.fs)
            bins = np.flatnonzero((frequencies>=protocol.band_low_hz)&(frequencies<=protocol.band_high_hz))
            order = bins[np.argsort(power[bins])[-3:][::-1]]
            diff = prediction.astype(float)-saved
            records.append({'path': name, 'rr_bpm':rr, 'target_rr_bpm':target_rr, 'whole_rr_abs_error_bpm':abs(rr-target_rr),
                            'vs_saved_max_abs':float(abs(diff).max()), 'vs_saved_relative_rms':float(np.linalg.norm(diff)/np.linalg.norm(saved)),
                            'vs_saved_corr':float(np.corrcoef(prediction,saved)[0,1]),
                            'top_bins_bpm':(frequencies[order]*60).tolist(), 'top_bin_power':power[order].tolist()})
        comparisons = {}
        for a,b in [('plain128_bf16','saved_batch128_bf16'), ('plain8_bf16','plain8_inference_bf16'),
                    ('plain8_inference_bf16','captured8_bf16'), ('plain128_fp32','plain8_fp32')]:
            diff=predictions[a].astype(float)-predictions[b]
            comparisons[f'{a}__{b}']={'max_abs':float(abs(diff).max()),'relative_rms':float(np.linalg.norm(diff)/np.linalg.norm(predictions[b]))}
        np.savez_compressed(output/'fixed_row_waveforms.npz', reference=target, **predictions)
        io.write_json(output/'report.json', {'row_id':row_id,'seed':seed,'records':records,'comparisons':comparisons})
        print(io.read_json(output/'report.json'), flush=True)
    print('COMPLETED=full_replay_diagnostic',flush=True)


if __name__ == '__main__':
    main()
