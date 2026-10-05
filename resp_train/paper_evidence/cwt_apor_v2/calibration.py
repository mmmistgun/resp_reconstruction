"""TF-S1 合成校准与表示几何导出；只有用户显式运行入口才执行计算。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import artifacts as io
from .features import representation, transform
from .spec import ARMS, OUTPUT_ROOT, PROTOCOL, load_spec, plan


def synthetic_signals():
    t = np.arange(18000, dtype=np.float64) / 100
    phase = 2 * np.pi * (.10 * t + .5 * (.60 - .10) / 180 * t**2)
    respiratory = np.sin(phase)
    amplitude = 1 + .5 * np.sin(2 * np.pi * .23 * t)
    independent = 1 + .4 * np.sin(2 * np.pi * .37 * t + .8)
    rng = np.random.default_rng(20260930)
    transient = 4 * np.exp(-.5 * ((t - 93) / .15)**2) + .02 * rng.normal(size=len(t))
    return {"drift": respiratory.astype(np.float32),
            "slow_cycle": np.sin(2*np.pi*.05*t).astype(np.float32),
            "am_common": (amplitude * (np.cos(2*np.pi*2*t) + np.cos(2*np.pi*5*t))).astype(np.float32),
            "am_independent": (amplitude * np.cos(2*np.pi*2*t) + independent * np.cos(2*np.pi*5*t)).astype(np.float32),
            "am_extended": (amplitude * np.cos(2*np.pi*12*t) + independent * np.cos(2*np.pi*18*t)).astype(np.float32),
            "transient": (respiratory + transient).astype(np.float32)}, {
                "time_seconds": t, "respiratory_frequency_hz": .10 + .50*t/180,
                "amplitude_023": amplitude, "amplitude_037": independent, "transient_component": transient}


def response_table(rep):
    from ssqueezepy import Wavelet, cwt
    impulse = np.zeros(18000, np.float32)
    impulse[9000] = 1
    wavelet = Wavelet(("morlet", {"mu": rep["arm"]["mu"]}), N=18000, dtype="float32")
    kernels, _ = cwt(impulse, wavelet=wavelet, scales=np.array(rep["scales_cwt_order"], np.float64),
                     fs=100., padtype="reflect", rpadded=False, vectorized=True, astensor=False, nan_checks=False)
    kernels = np.asarray(kernels)[rep["frequency_order"]]
    frequencies = np.asarray(rep["frequencies_hz"])
    rows = []
    fft_frequencies = np.fft.fftfreq(18000, .01)
    positive = fft_frequencies >= 0
    times = (np.arange(18000) - 9000) / 100
    for i, kernel in enumerate(kernels):
        power = np.abs(kernel)**2
        spectrum = np.abs(np.fft.fft(kernel))**2
        if not np.isfinite(power).all() or not np.isfinite(spectrum).all() or power.sum() <= 0:
            raise FloatingPointError("小波响应非有限/退化")
        support = np.flatnonzero(power >= power.max() / 2)
        fp = fft_frequencies[positive]
        sp = spectrum[positive]
        bandwidth = np.flatnonzero(sp >= sp.max() / 2)
        cumulative = np.cumsum(power, dtype=np.float64) / power.sum(dtype=np.float64)
        lo, hi = np.searchsorted(cumulative, [.005, .995])
        radius = float(max(abs(times[lo]), abs(times[min(hi, 17999)])))
        rows.append({"scale_slot": i, "frequency_hz": frequencies[i],
                     "time_power_fwhm_seconds": (support[-1] - support[0] + 1) / 100,
                     "frequency_power_fwhm_hz": (fp[bandwidth[-1]] - fp[bandwidth[0]]) if len(bandwidth) else 0.,
                     "energy_99_radius_seconds": radius,
                     "finite_window_edge_energy_fraction": float((power[:100].sum() + power[-100:].sum()) / power.sum()),
                     "conv5_low_hz": frequencies[max(0, i-2)], "conv5_high_hz": frequencies[min(len(frequencies)-1, i+2)],
                     "conv7_low_hz": frequencies[max(0, i-3)], "conv7_high_hz": frequencies[min(len(frequencies)-1, i+3)]})
    return pd.DataFrame(rows)


def calibrate(retry=False):
    from resp_train.crd.tf_v1_features import cwt_magnitude_features
    spec = load_spec()
    source = {str(p): io.identity(p) for p in io.source_files()}
    key = {"phase": "synthetic_calibration", "spec": io.json_hash(spec), "source": io.json_hash(source)}
    parent = OUTPUT_ROOT / "calibration" / key["source"][:16]
    prior = io.completed(parent, key)
    if prior:
        return prior
    with io.attempt(parent, key, retry) as output:
        provenance = io.provenance(output)
        signals, truth = synthetic_signals()
        np.savez_compressed(output / "synthetic_truth.npz", **signals, **truth)
        reps, checks, calibration_errors = {}, [], []
        for name, arm in ARMS.items():
            rep = reps[name] = representation(arm)
            response_table(rep).to_csv(output / f"{name}_response.csv", index=False)
            arrays = {case: transform(signal, rep) for case, signal in signals.items()}
            np.savez_compressed(output / f"{name}_calibration.npz", **arrays)
            frequency = np.array(rep["frequencies_hz"])
            time = np.array(rep["time_seconds"])
            central = (time >= 30) & (time < 150)
            target_frequency = truth["respiratory_frequency_hz"].reshape(arm.frames, arm.pool_samples).mean(-1)
            ridge = frequency[np.argmax(arrays["drift"], axis=0)]
            calibration_errors.append({"arm": name, "case": "drift", "measurement": "dominant_scale_absolute_error_hz",
                                       "value": float(np.mean(np.abs(ridge[central]-target_frequency[central]))),
                                       "support": "30_to_150_seconds", "target_in_coverage": bool(np.all((target_frequency[central] >= frequency.min()) & (target_frequency[central] <= frequency.max())))})
            from scipy.stats import pearsonr
            for case, carrier, amplitude_key in (("am_common", 2., "amplitude_023"), ("am_common", 5., "amplitude_023"),
                                                  ("am_extended", 12., "amplitude_023"), ("am_extended", 18., "amplitude_037")):
                if not frequency.min() <= carrier <= frequency.max():
                    continue
                slot = int(np.argmin(np.abs(np.log(frequency/carrier))))
                observed = arrays[case][slot, central]
                target = np.log1p(truth[amplitude_key]).reshape(arm.frames, arm.pool_samples).mean(-1)[central]
                defined = np.std(observed) > 1e-8 and np.std(target) > 1e-8
                calibration_errors.append({"arm": name, "case": case, "measurement": "log_amplitude_pearson_r",
                                           "carrier_hz": carrier, "actual_scale_hz": frequency[slot],
                                           "value": float(pearsonr(observed, target).statistic) if defined else np.nan,
                                           "support": "30_to_150_seconds", "target_in_coverage": True, "defined": defined})
            if name == "A0":
                for case, signal in signals.items():
                    old, frequencies = cwt_magnitude_features(signal)
                    np.testing.assert_array_equal(frequencies, rep["frequencies_hz"])
                    np.testing.assert_allclose(arrays[case], old, rtol=1e-6, atol=1e-7)
                    checks.append({"case": case, "baseline_max_abs": float(np.max(np.abs(arrays[case]-old)))})
            pool = arm.pool_samples
            modulation_hz = np.linspace(0, 2., 1001)
            response = np.abs(np.exp(-2j*np.pi*modulation_hz[:, None]*np.arange(pool)[None, :]/100).mean(1))
            pd.DataFrame({"modulation_hz": modulation_hz, "boxcar_amplitude": response}).to_csv(output / f"{name}_pool_response.csv", index=False)
        b = reps["A0"]
        for name in ("C_0p8", "C_2", "C_4", "H"):
            a = ARMS[name]
            f = np.array(b["frequencies_hz"])
            keep = (f > .8) & (f <= 8) if a.high_only else f <= a.high_hz
            np.testing.assert_array_equal(reps[name]["frequencies_hz"], f[keep])
            # 不把中心频率一致当作数值等价；合成样本验证实际变换子集。
            for signal in signals.values():
                np.testing.assert_allclose(transform(signal, reps[name]), transform(signal, b)[keep], rtol=1e-6, atol=1e-7)
        pd.DataFrame(calibration_errors).to_csv(output / "calibration_errors.csv", index=False)
        io.write_json(output / "calibration.json", {"protocol": PROTOCOL, "spec": spec, "representations": reps,
                      "baseline_checks": checks, "provenance": provenance, "synthetic_only": True,
                      "numerical_checks_passed": True, "scientific_parameter_review_required": True,
                      "response_definition": "finite_180s_centered_impulse_power_FWHM_and_99_percent_energy"})
        io.verify_provenance(provenance, output)
    return output


def prepare(calibration, review=None, *, engineering_only=False):
    """review 是用户在校准/滤波核查后的明确结论，随 session 冻结。"""
    io.verify_stage(calibration)
    payload = io.read_json(calibration / "calibration.json")
    io.verify_provenance(payload["provenance"], calibration)
    if (not payload["numerical_checks_passed"] or payload["spec"] != load_spec()
            or (not engineering_only and (not review or not review.strip()))):
        raise ValueError("缺少有效校准及参数核查说明")
    output = OUTPUT_ROOT / ("session_" + io.stamp())
    output.mkdir(parents=True, exist_ok=False)
    try:
        io.write_json(output / "session.json", {"protocol": PROTOCOL, "spec": load_spec(), "plan": plan(),
                      "representations": payload["representations"], "provenance": io.provenance(output),
                      "calibration": {"path": str(calibration.resolve()), **io.identity(calibration / "manifest.json")},
                      "execution_scope": "synthetic_only" if engineering_only else "train_validation",
                      "parameter_review": None if engineering_only else review, "research_test_open": False})
        io.write_json(output / "session_receipt.json", io.identity(output / "session.json"))
    except BaseException:
        import traceback
        io.write_json(output / "failed.json", {"traceback": traceback.format_exc()})
        raise
    return output
