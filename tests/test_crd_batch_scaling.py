from __future__ import annotations

import pytest

from scripts.benchmark_crd_batch_scaling import parse_batch_scheme, summarize_benchmark


def test_parse_batch_scheme_preserves_effective_batch_128() -> None:
    assert parse_batch_scheme("32x4").label == "32x4"
    assert parse_batch_scheme("64×2").effective_batch_size == 128
    assert parse_batch_scheme("128x1").accumulation_steps == 1
    with pytest.raises(ValueError, match="effective batch=128"):
        parse_batch_scheme("64x4")


def _records(*, throughput_32: float, throughput_64: float, throughput_128: float, memory_128: float = 4000.0):
    records = []
    for label, throughput, memory in (
        ("32x4", throughput_32, 2000.0),
        ("64x2", throughput_64, 3000.0),
        ("128x1", throughput_128, memory_128),
    ):
        for repeat in (1, 2, 3):
            records.append(
                {
                    "scheme": label,
                    "repeat_index": repeat,
                    "status": "passed",
                    "samples_per_second": throughput,
                    "elapsed_seconds": 128.0 / throughput,
                    "peak_allocated_mib": memory - 100.0,
                    "peak_reserved_mib": memory,
                }
            )
    return records


def test_summary_selects_fastest_safe_scheme_above_practical_gain() -> None:
    summary = summarize_benchmark(
        _records(throughput_32=10.0, throughput_64=11.5, throughput_128=14.0),
        total_memory_mib=16000.0,
        expected_repeats=3,
    )
    assert summary["recommended_scheme"] == "128x1"


def test_summary_keeps_baseline_when_gain_small_or_memory_unsafe() -> None:
    small_gain = summarize_benchmark(
        _records(throughput_32=10.0, throughput_64=10.5, throughput_128=10.9),
        total_memory_mib=16000.0,
        expected_repeats=3,
    )
    assert small_gain["recommended_scheme"] == "32x4"

    unsafe = summarize_benchmark(
        _records(throughput_32=10.0, throughput_64=10.5, throughput_128=14.0, memory_128=14000.0),
        total_memory_mib=16000.0,
        expected_repeats=3,
    )
    assert unsafe["recommended_scheme"] == "32x4"
