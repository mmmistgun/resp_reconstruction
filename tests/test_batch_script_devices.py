import importlib

import pytest


BATCH_MODULES = [
    "scripts.run_b2_native_decode",
    "scripts.run_e2b_band_group",
    "scripts.run_e2c_band_energy",
    "scripts.run_e3_a0_probe",
    "scripts.run_e3_b_probe",
    "scripts.run_e3_c1_injection_probe",
    "scripts.run_e4_sst_probe",
]


@pytest.mark.parametrize("module_name", BATCH_MODULES)
def test_batch_script_device_resolution_does_not_mix_explicit_with_default(module_name):
    module = importlib.import_module(module_name)

    assert module._resolve_devices(None) == ["cuda:0"]
    assert module._resolve_devices(["cuda:1"]) == ["cuda:1"]
    assert module._resolve_devices(["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]


@pytest.mark.parametrize("module_name", BATCH_MODULES)
def test_batch_script_device_assignment_is_round_robin(module_name):
    module = importlib.import_module(module_name)
    specs = module.build_run_specs()[:5]
    assignments = module._assign_devices(specs, ["cuda:0", "cuda:1"])

    assert [device for _, device in assignments] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]


@pytest.mark.parametrize("module_name", BATCH_MODULES)
def test_batch_script_commands_default_to_preloaded_windows(module_name):
    module = importlib.import_module(module_name)
    spec = module.build_run_specs()[0]
    command = " ".join(module._command_for_spec(spec, "cuda:0"))

    assert "data.preload_windows=true" in command
    assert "training.num_workers=0" in command
    assert "training.persistent_workers=true" not in command
    assert "training.prefetch_factor=2" not in command


@pytest.mark.parametrize("module_name", BATCH_MODULES)
def test_batch_script_launch_plan_staggers_parallel_slots(module_name):
    module = importlib.import_module(module_name)
    specs = module.build_run_specs()[:5]
    plan = module._build_launch_plan(
        specs,
        devices=["cuda:0", "cuda:1"],
        max_parallel=2,
        start_stagger_sec=30.0,
    )

    assert [(device, delay) for _, device, delay in plan] == [
        ("cuda:0", 0.0),
        ("cuda:1", 30.0),
        ("cuda:0", 0.0),
        ("cuda:1", 30.0),
        ("cuda:0", 0.0),
    ]
