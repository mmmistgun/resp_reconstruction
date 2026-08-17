from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "crd-tf-w-v2-research-informed-20260817"
CANDIDATE_LOCK = REPO_ROOT / "docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json"
CANDIDATE_LOCK_SHA256 = "6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6"
SOURCE_CACHE_ROOT = REPO_ROOT / (
    "runs/crd_tf_v1/cache/bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0"
)
SOURCE_CACHE_MANIFEST_SHA256 = "6fb44aad2689d9426ad78dc1f054db5aaac698792af5818bc01a54563cb9f0b8"
W_FREQUENCY_FILE_SHA256 = "15cc722c38e5572b3284c92137cfdeec7cbc4153b4588c2c1e086f26ae3238b3"
W_FREQUENCY_CONTENT_SHA256 = "9fb164e7b09d42b31f7ee57a7d3e966af7adabc6d8c6a0e1eff3090b00b73c0c"
FULL_6V_MAPPED_FREQUENCY_SHA256 = "2eda45200cc329c4989a516fa3611f0862c3c546bca26bb124025a2bb8146b29"

P1_VARIANT_W_VIEW: dict[str, str] = {
    "crd_tfw_v2_w1_resp_12v_film_d6": "resp",
    "crd_tfw_v2_w2_carrier_12v_film_d6": "carrier",
    "crd_tfw_v2_w3_full_6v_film_d6": "full_6v",
}
P1_VARIANT_REPRESENTATIONS = {variant: ("w",) for variant in P1_VARIANT_W_VIEW}
P1_VARIANTS = tuple(P1_VARIANT_W_VIEW)

RESP_INDICES = np.arange(0, 56, dtype="<i8")
CARRIER_INDICES = np.arange(56, 97, dtype="<i8")
FULL_6V_INDICES = np.arange(0, 97, 2, dtype="<i8")
VIEW_INDICES = {
    "resp": RESP_INDICES,
    "carrier": CARRIER_INDICES,
    "full_6v": FULL_6V_INDICES,
}
VIEW_INDEX_SHA256 = {
    "resp": "460c40aca1f47cb149e783f690fc0f2aec0cda4717a2da644ec7927a6a744f36",
    "carrier": "b9e31adaf7120cbdb7f503cfd806bb59831f0b09549ac705845f12d1e19e95c4",
    "full_6v": "3734c4183c17744eb4779ac65ac8521de1897f6a12f364e913aeb02528e250d6",
}


def apply_p1_w_view(value: torch.Tensor, variant: str) -> torch.Tensor:
    """在已复制的 12V batch tensor 上应用 P1 固定 mask/view，不修改 source。"""

    normalized = str(variant).strip().lower()
    if normalized not in P1_VARIANT_W_VIEW:
        raise ValueError(f"未知 CRD-TF-W v2 P1 variant={variant!r}")
    if value.ndim != 3 or tuple(value.shape[1:]) != (97, 360):
        raise ValueError(f"P1 source W 期望 (B,97,360)，实际 {tuple(value.shape)}")
    view = P1_VARIANT_W_VIEW[normalized]
    if view == "resp":
        output = torch.zeros_like(value)
        output[:, :56] = value[:, :56]
        return output
    if view == "carrier":
        output = torch.zeros_like(value)
        output[:, 56:] = value[:, 56:]
        return output
    if view == "full_6v":
        # 切片与冻结的 indices 0,2,...,96 完全等价；保持 input-only 只读 view。
        return value[:, ::2, :]
    raise AssertionError(view)


def p1_w_scale_count(variant: str) -> int:
    normalized = str(variant).strip().lower()
    if normalized not in P1_VARIANT_W_VIEW:
        raise ValueError(f"未知 CRD-TF-W v2 P1 variant={variant!r}")
    return 49 if P1_VARIANT_W_VIEW[normalized] == "full_6v" else 97


def p1_variant_contract(variant: str) -> dict[str, Any]:
    normalized = str(variant).strip().lower()
    if normalized not in P1_VARIANT_W_VIEW:
        raise ValueError(f"未知 CRD-TF-W v2 P1 variant={variant!r}")
    view = P1_VARIANT_W_VIEW[normalized]
    active_scale_count = int(VIEW_INDICES[view].size)
    input_scale_count = p1_w_scale_count(normalized)
    return {
        **verify_p1_source_identity(),
        "variant": normalized,
        "w_view": view,
        "source_shape_per_sample": [97, 360],
        "model_input_shape_per_sample": [input_scale_count, 360],
        "source_tensor_elements_per_sample": 97 * 360,
        "model_input_tensor_elements_per_sample": input_scale_count * 360,
        "active_scale_count": active_scale_count,
        "view_index_sha256": VIEW_INDEX_SHA256[view],
        "application_point": "after cache copy and immediately before CwtBranch.conv_in",
        "materialization_strategy": (
            "zeros_like input-only mask" if view in {"resp", "carrier"} else "strided input-only slice [:,::2,:]"
        ),
    }


@lru_cache(maxsize=1)
def verify_p1_source_identity(cache_root: str | Path = SOURCE_CACHE_ROOT) -> dict[str, Any]:
    """复核 P0 锁定的 source cache、frequency 和三个 P1 index contract。"""

    resolved_root = Path(cache_root).resolve()
    if resolved_root != SOURCE_CACHE_ROOT.resolve():
        raise ValueError(f"P1 只接受冻结 source cache: {SOURCE_CACHE_ROOT}")
    if _sha256_file(CANDIDATE_LOCK) != CANDIDATE_LOCK_SHA256:
        raise RuntimeError("P1 candidate lock SHA-256 漂移")
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    cache_lock = lock.get("cache_lock", {})
    if (
        lock.get("protocol") != PROTOCOL
        or lock.get("p0_outcome", {}).get("candidate_lock_complete") is not True
        or (REPO_ROOT / str(cache_lock.get("root", ""))).resolve() != resolved_root
    ):
        raise RuntimeError("P1 candidate lock completion/cache contract 不合格")

    manifest_path = resolved_root / "cache_manifest.json"
    if _sha256_file(manifest_path) != SOURCE_CACHE_MANIFEST_SHA256:
        raise RuntimeError("P1 source cache manifest SHA-256 漂移")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frequency_path = resolved_root / "w_frequencies_hz.npy"
    if _sha256_file(frequency_path) != W_FREQUENCY_FILE_SHA256:
        raise RuntimeError("P1 W frequency file SHA-256 漂移")
    frequencies = np.load(frequency_path, allow_pickle=False)
    if frequencies.shape != (97,) or frequencies.dtype != np.float64:
        raise RuntimeError("P1 W frequencies shape/dtype 漂移")
    frequency_content_sha256 = hashlib.sha256(np.asarray(frequencies, dtype="<f8").tobytes()).hexdigest()
    if frequency_content_sha256 != W_FREQUENCY_CONTENT_SHA256:
        raise RuntimeError("P1 W frequency content SHA-256 漂移")

    for view, indices in VIEW_INDICES.items():
        observed = hashlib.sha256(indices.tobytes()).hexdigest()
        if observed != VIEW_INDEX_SHA256[view]:
            raise RuntimeError(f"P1 {view} local index SHA-256 漂移")
        locked_view = cache_lock["views"][view]
        if int(locked_view["count"]) != int(indices.size) or locked_view["indices_sha256"] != observed:
            raise RuntimeError(f"P1 {view} candidate-lock index contract 漂移")

    expected_resp = np.flatnonzero(frequencies <= 0.80).astype("<i8", copy=False)
    expected_carrier = np.flatnonzero(frequencies > 0.80).astype("<i8", copy=False)
    if not np.array_equal(expected_resp, RESP_INDICES) or not np.array_equal(expected_carrier, CARRIER_INDICES):
        raise RuntimeError("P1 RESP/CARRIER frequency rule 与冻结 indices 不一致")
    full_6v_frequencies = np.asarray(frequencies[FULL_6V_INDICES], dtype="<f8")
    full_6v_sha256 = hashlib.sha256(full_6v_frequencies.tobytes()).hexdigest()
    if full_6v_sha256 != FULL_6V_MAPPED_FREQUENCY_SHA256:
        raise RuntimeError("P1 full-6V mapped-frequency identity 漂移")
    target_grid = 0.03125 * np.power(2.0, np.arange(49, dtype=np.float64) / 6.0)
    target_even_grid = 0.03125 * np.power(2.0, FULL_6V_INDICES.astype(np.float64) / 12.0)
    grid_max_abs_delta = float(np.max(np.abs(target_grid - target_even_grid)))
    if grid_max_abs_delta > 2e-15:
        raise RuntimeError("P1 full-6V nominal target grid identity 漂移")

    for split in ("train", "val"):
        name = f"{split}_w.npy"
        metadata = manifest["files"][name]
        locked = cache_lock[f"{split}_w"]
        path = resolved_root / name
        if (
            path.stat().st_size != int(locked["size_bytes"])
            or metadata["shape"] != locked["shape"]
            or metadata["dtype"] != locked["dtype"]
            or metadata.get("finite") is not True
        ):
            raise RuntimeError(f"P1 source {name} identity 漂移")

    return {
        "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
        "source_cache_root": str(resolved_root),
        "source_cache_manifest_sha256": SOURCE_CACHE_MANIFEST_SHA256,
        "frequency_file_sha256": W_FREQUENCY_FILE_SHA256,
        "frequency_content_sha256": W_FREQUENCY_CONTENT_SHA256,
        "view_index_sha256": dict(VIEW_INDEX_SHA256),
        "full_6v_mapped_frequency_sha256": FULL_6V_MAPPED_FREQUENCY_SHA256,
        "full_6v_mapped_frequency_hz": [float(full_6v_frequencies[0]), float(full_6v_frequencies[-1])],
        "full_6v_nominal_grid_max_abs_delta_hz": grid_max_abs_delta,
        "source_train_shape": list(cache_lock["train_w"]["shape"]),
        "source_val_shape": list(cache_lock["val_w"]["shape"]),
        "source_train_w_sha256": str(cache_lock["train_w"]["sha256"]),
        "source_val_w_sha256": str(cache_lock["val_w"]["sha256"]),
        "source_cache_modified": False,
        "target_read": False,
        "research_test_used": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
