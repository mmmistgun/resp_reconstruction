from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from resp_train.crd.tf_v1_features import (
    CONTEXT_LENGTH,
    MORLET_MU,
    PROTOCOL,
    SAMPLE_RATE,
    S_SCALE_COUNT,
    S_VOICES_PER_OCTAVE,
    W_SCALE_COUNT,
    W_VOICES_PER_OCTAVE,
    WINDOW_SAMPLES,
    LearnableCarrierModulation,
    RidgeParameters,
    cwt_magnitude_features,
    fixed_transform_spec,
    morlet_scales_and_frequencies,
    multires_stft_features,
    one_sided_input_spectrum,
    wsst_energy_map,
    wsst_ridge_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_v1/calibration"
CALIBRATION_FILENAME = "calibration.json"
PENALTY_CANDIDATES = (0.5, 1.0, 2.0, 4.0)
SUPPRESSION_CANDIDATES = (2, 4, 6)


def run_tf_v1_calibration(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    require_clean: bool = True,
) -> Path:
    """运行完整 synthetic calibration；该入口可能较慢，默认由用户执行。"""

    if require_clean:
        _assert_clean_repository()
    spec = fixed_transform_spec()
    spec_sha256 = _sha256_json(spec)
    implementation = calibration_implementation_identity()
    calibration_identity_sha256 = _sha256_json(
        {"spec_sha256": spec_sha256, "implementation": implementation}
    )
    output_dir = Path(output_root) / calibration_identity_sha256
    if output_dir.exists():
        raise FileExistsError(f"CRD-TF calibration 禁止覆盖: {output_dir}")

    signals = _synthetic_signals()
    results = {
        "m": calibrate_m(signals),
        "w": calibrate_w(signals),
        "l": calibrate_l(signals),
        "s": calibrate_s(signals),
    }
    passed = bool(all(bool(result["passed"]) for result in results.values()))
    payload = {
        "protocol": PROTOCOL,
        "spec_sha256": spec_sha256,
        "spec": spec,
        "calibration_identity_sha256": calibration_identity_sha256,
        "implementation": implementation,
        "complete": True,
        "passed": passed,
        "research_test_used": False,
        "validation_target_used": False,
        "real_waveform_used": False,
        "synthetic_only": True,
        "dependencies": _dependency_versions(),
        "git": _git_identity(),
        "results": results,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        (temporary / CALIBRATION_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / CALIBRATION_FILENAME


def calibrate_m(signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for frequency in (0.05, 0.10, 0.20, 0.40, 0.70):
        features = multires_stft_features(signals[f"sin_{frequency:.2f}"]["waveform"])
        observed = _mean_spectrum_peak(features["m_slow"], np.arange(1, 37) * SAMPLE_RATE / 3000)
        error = abs(observed - frequency)
        cases.append(
            {
                "branch": "slow",
                "target_hz": frequency,
                "observed_hz": observed,
                "absolute_error_hz": error,
                "threshold_hz": 0.5 * SAMPLE_RATE / 3000,
                "passed": bool(error <= 0.5 * SAMPLE_RATE / 3000 + 1e-12),
            }
        )
    for frequency in (1.0, 2.0, 4.0, 6.0):
        features = multires_stft_features(signals[f"sin_{frequency:.2f}"]["waveform"])
        observed = _mean_spectrum_peak(features["m_fast"], np.arange(5, 49) * SAMPLE_RATE / 600)
        error = abs(observed - frequency)
        cases.append(
            {
                "branch": "fast",
                "target_hz": frequency,
                "observed_hz": observed,
                "absolute_error_hz": error,
                "threshold_hz": 0.5 * SAMPLE_RATE / 600,
                "passed": bool(error <= 0.5 * SAMPLE_RATE / 600 + 1e-12),
            }
        )
    return {"passed": bool(all(case["passed"] for case in cases)), "cases": cases}


def calibrate_w(signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases = []
    frequencies_used: np.ndarray | None = None
    for frequency in (0.05, 0.10, 0.20, 0.40, 0.70, 1.0, 2.0, 4.0, 6.0):
        features, frequencies = cwt_magnitude_features(signals[f"sin_{frequency:.2f}"]["waveform"])
        frequencies_used = frequencies
        peak_path = frequencies[np.argmax(features, axis=0)]
        regions = {
            "start_15s": peak_path[:30],
            "middle": peak_path[30:-30],
            "end_15s": peak_path[-30:],
        }
        region_errors = {
            key: float(np.median(np.abs(np.log2(np.maximum(values, 1e-12) / frequency))))
            for key, values in regions.items()
        }
        threshold = 1.0 / W_VOICES_PER_OCTAVE
        cases.append(
            {
                "target_hz": frequency,
                "median_log2_error": region_errors,
                "middle_threshold_octave": threshold,
                "passed": bool(region_errors["middle"] <= threshold + 1e-12),
            }
        )
    scales, mapped = morlet_scales_and_frequencies(W_VOICES_PER_OCTAVE, W_SCALE_COUNT)
    if frequencies_used is None:
        raise RuntimeError("W calibration 没有 case")
    return {
        "passed": bool(all(case["passed"] for case in cases)),
        "target_frequencies_hz": _float_list(
            8.0 * np.power(2.0, -np.arange(W_SCALE_COUNT) / W_VOICES_PER_OCTAVE)[::-1]
        ),
        "scales": _float_list(scales),
        "mapped_frequencies_hz": _float_list(mapped),
        "output_frequencies_hz": _float_list(frequencies_used),
        "cases": cases,
    }


def calibrate_l(signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    torch.manual_seed(20260812)
    model = LearnableCarrierModulation().cpu()
    cases = []
    for name, modulation_frequency in (("am_2.00_0.20", 0.20), ("am_4.00_0.40", 0.40)):
        spectrum = one_sided_input_spectrum(signals[name]["waveform"])
        model.zero_grad(set_to_none=True)
        output = model(one_sided_spectrum=torch.from_numpy(spectrum)[None, :])
        channel_energy = output.detach().square().mean(dim=-1)[0]
        channel = int(torch.argmax(channel_energy).item())
        observed = _dominant_frequency(output.detach().numpy()[0, channel], sample_rate=2.0, low_hz=0.03, high_hz=0.80)
        loss = output.square().mean()
        loss.backward()
        gradients_finite = bool(
            model.center_logits.grad is not None
            and model.bandwidth_logits.grad is not None
            and torch.isfinite(model.center_logits.grad).all()
            and torch.isfinite(model.bandwidth_logits.grad).all()
        )
        gradients_nonzero = bool(
            model.center_logits.grad is not None
            and model.bandwidth_logits.grad is not None
            and (model.center_logits.grad.abs().max() > 0)
            and (model.bandwidth_logits.grad.abs().max() > 0)
        )
        error = abs(observed - modulation_frequency)
        cases.append(
            {
                "signal": name,
                "target_modulation_hz": modulation_frequency,
                "observed_modulation_hz": observed,
                "absolute_error_hz": error,
                "selected_filter": channel,
                "gradients_finite": gradients_finite,
                "gradients_nonzero": gradients_nonzero,
                "passed": bool(error <= 0.01 and gradients_finite and gradients_nonzero),
            }
        )
    centers = model.centers().detach().numpy()
    rhos = model.rhos().detach().numpy()
    bounded = bool(
        np.all(np.diff(centers) > 0.0)
        and np.all(centers >= 0.70 - 1e-6)
        and np.all(centers <= 8.00 + 1e-6)
        and np.all(rhos >= 0.08 - 1e-6)
        and np.all(rhos <= 0.30 + 1e-6)
    )
    return {
        "passed": bool(bounded and all(case["passed"] for case in cases)),
        "centers_bounded_and_ordered": bounded,
        "initial_centers_hz": _float_list(centers),
        "initial_rhos": _float_list(rhos),
        "cases": cases,
    }


def calibrate_s(signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prepared = {}
    for name in ("sin_0.20", "resp_chirp", "resp_crossing", "carrier_harmonics"):
        energy, frequencies = wsst_energy_map(signals[name]["waveform"])
        prepared[name] = (energy, frequencies)

    candidates = []
    for penalty in PENALTY_CANDIDATES:
        for radius in SUPPRESSION_CANDIDATES:
            parameters = RidgeParameters(penalty, radius)
            cases = _score_s_candidate(prepared, signals, parameters)
            score = float(np.mean([case["score_octave"] for case in cases]))
            candidates.append(
                {
                    "smoothness_penalty": penalty,
                    "suppression_radius_bins": radius,
                    "score_octave": score,
                    "cases": cases,
                }
            )
    selected = min(
        candidates,
        key=lambda item: (
            float(item["score_octave"]),
            float(item["smoothness_penalty"]),
            int(item["suppression_radius_bins"]),
        ),
    )
    case_map = {case["signal"]: case for case in selected["cases"]}
    passed = bool(
        case_map["sin_0.20"]["score_octave"] <= 1.0 / S_VOICES_PER_OCTAVE
        and case_map["resp_chirp"]["score_octave"] <= 1.0 / S_VOICES_PER_OCTAVE
        and case_map["resp_crossing"]["score_octave"] <= 2.0 / S_VOICES_PER_OCTAVE
        and case_map["carrier_harmonics"]["score_octave"] <= 2.0 / S_VOICES_PER_OCTAVE
    )
    scales, mapped = morlet_scales_and_frequencies(S_VOICES_PER_OCTAVE, S_SCALE_COUNT)
    return {
        "passed": passed,
        "selected": {
            "smoothness_penalty": selected["smoothness_penalty"],
            "suppression_radius_bins": selected["suppression_radius_bins"],
            "score_octave": selected["score_octave"],
            "cases": selected["cases"],
        },
        "candidate_grid": candidates,
        "target_frequencies_hz": _float_list(
            8.0 * np.power(2.0, -np.arange(S_SCALE_COUNT) / S_VOICES_PER_OCTAVE)[::-1]
        ),
        "scales": _float_list(scales),
        "mapped_frequencies_hz": _float_list(mapped),
        "confidence_formula": "ridge_energy / (band_energy_sum + 1e-8)",
        "confidence_min": 1e-4,
        "missing_rule": "band_total<=1e-8 or confidence<1e-4 => frequency/amplitude/confidence=0",
        "tie_break": "lower frequency index",
    }


def _score_s_candidate(
    prepared: dict[str, tuple[np.ndarray, np.ndarray]],
    signals: dict[str, dict[str, Any]],
    parameters: RidgeParameters,
) -> list[dict[str, Any]]:
    cases = []
    for name, (energy, frequencies) in prepared.items():
        features = wsst_ridge_features(energy, frequencies, parameters)
        truth = np.asarray(signals[name]["truth_hz"], dtype=np.float64)
        if name == "carrier_harmonics":
            predicted = features[[6, 9]].astype(np.float64)
        else:
            predicted = features[[0, 3]].astype(np.float64)
        if truth.ndim == 1:
            valid = predicted[0] > 0.0
            error = np.abs(np.log2(np.maximum(predicted[0, valid], 1e-12) / truth[valid]))
        else:
            valid = (predicted > 0.0).all(axis=0)
            direct = np.abs(np.log2(predicted[:, valid] / truth[:, valid])).sum(axis=0)
            swapped = np.abs(np.log2(predicted[::-1, valid] / truth[:, valid])).sum(axis=0)
            error = np.minimum(direct, swapped) / 2.0
        score = float(np.median(error)) if error.size else math.inf
        cases.append(
            {
                "signal": name,
                "score_octave": score,
                "valid_frame_fraction": float(np.mean(valid)),
            }
        )
    return cases


def _synthetic_signals() -> dict[str, dict[str, Any]]:
    time = np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    pooled_time = (np.arange(CONTEXT_LENGTH, dtype=np.float64) * 0.5) + 0.245
    signals: dict[str, dict[str, Any]] = {}
    for frequency in (0.05, 0.10, 0.20, 0.40, 0.70, 1.0, 2.0, 4.0, 6.0):
        signals[f"sin_{frequency:.2f}"] = {
            "waveform": np.sin(2.0 * np.pi * frequency * time).astype(np.float32),
            "truth_hz": np.full(CONTEXT_LENGTH, frequency, dtype=np.float64),
        }
    for carrier, modulation in ((2.0, 0.20), (4.0, 0.40)):
        waveform = (1.0 + 0.5 * np.sin(2.0 * np.pi * modulation * time)) * np.sin(
            2.0 * np.pi * carrier * time
        )
        signals[f"am_{carrier:.2f}_{modulation:.2f}"] = {"waveform": waveform.astype(np.float32)}

    chirp_frequency = 0.08 + (0.45 - 0.08) * time / time[-1]
    chirp_phase = 2.0 * np.pi * np.cumsum(chirp_frequency) / SAMPLE_RATE
    signals["resp_chirp"] = {
        "waveform": np.sin(chirp_phase).astype(np.float32),
        "truth_hz": 0.08 + (0.45 - 0.08) * pooled_time / time[-1],
    }

    first = 0.10 + 0.30 * time / time[-1]
    second = 0.40 - 0.30 * time / time[-1]
    phase_first = 2.0 * np.pi * np.cumsum(first) / SAMPLE_RATE
    phase_second = 2.0 * np.pi * np.cumsum(second) / SAMPLE_RATE
    signals["resp_crossing"] = {
        "waveform": (np.sin(phase_first) + 0.9 * np.sin(phase_second)).astype(np.float32),
        "truth_hz": np.stack(
            (
                0.10 + 0.30 * pooled_time / time[-1],
                0.40 - 0.30 * pooled_time / time[-1],
            ),
            axis=0,
        ),
    }

    rng = np.random.default_rng(20260812)
    carrier_harmonics = (
        np.sin(2.0 * np.pi * 2.0 * time)
        + 0.7 * np.sin(2.0 * np.pi * 4.0 * time)
        + 0.05 * rng.standard_normal(WINDOW_SAMPLES)
    )
    signals["carrier_harmonics"] = {
        "waveform": carrier_harmonics.astype(np.float32),
        "truth_hz": np.stack(
            (
                np.full(CONTEXT_LENGTH, 2.0),
                np.full(CONTEXT_LENGTH, 4.0),
            ),
            axis=0,
        ),
    }
    return signals


def _mean_spectrum_peak(feature: np.ndarray, frequencies: np.ndarray) -> float:
    index = int(np.argmax(np.asarray(feature, dtype=np.float64).mean(axis=1)))
    return float(np.asarray(frequencies, dtype=np.float64)[index])


def _dominant_frequency(values: np.ndarray, *, sample_rate: float, low_hz: float, high_hz: float) -> float:
    signal = np.asarray(values, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / float(sample_rate))
    selected = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        raise RuntimeError("dominant-frequency band 没有 bins")
    return float(frequencies[indices[int(np.argmax(spectrum[selected]))]])


def _float_list(values: np.ndarray | torch.Tensor) -> list[float]:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return [float(value) for value in array.reshape(-1)]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibration_implementation_identity() -> dict[str, str]:
    files = {
        "tf_v1_features.py": Path(__file__).with_name("tf_v1_features.py"),
        "tf_v1_calibration.py": Path(__file__),
    }
    return {name: _sha256_file(path) for name, path in files.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    distributions = {
        "numpy": "numpy",
        "torch": "torch",
        "scipy": "scipy",
        "PyWavelets": "PyWavelets",
        "ssqueezepy": "ssqueezepy",
    }
    return {name: importlib.metadata.version(distribution) for name, distribution in distributions.items()}


def _git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "error": None if commit.returncode == 0 and status.returncode == 0 else "git identity failed",
    }


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"无法检查 Git 工作树: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("完整 CRD-TF calibration 要求干净 Git 工作树")
