import os
import subprocess


def _visible_device_ids():
    """Physical GPU ids this run is allowed to touch.

    Honours an existing CUDA_VISIBLE_DEVICES so callers can fence off busy GPUs,
    e.g. CUDA_VISIBLE_DEVICES=1,2,3 pytest -n 3.
    """
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if preset is not None:
        return [d.strip() for d in preset.split(",") if d.strip()]
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=30)
    except Exception:
        return []
    return [str(i) for i, ln in enumerate(out.splitlines()) if ln.startswith("GPU ")]


def _pin_worker_to_gpu():
    """Give each xdist worker its own GPU, round-robin.

    Must run before torch initialises CUDA -- hence at conftest import time,
    above `import torch`. Each worker is a separate process and pytest-xdist
    sets PYTEST_XDIST_WORKER ("gw0", "gw1", ...) before conftest is imported.
    With more workers than GPUs the assignment wraps, so -n 16 on 8 GPUs puts
    two workers per GPU rather than all 16 on cuda:0.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:  # not running under -n; leave the environment alone
        return
    devices = _visible_device_ids()
    if not devices:
        return
    try:
        idx = int(worker.removeprefix("gw"))
    except ValueError:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[idx % len(devices)]


_pin_worker_to_gpu()

import pytest
import torch

requires_gpu = pytest.mark.gpu
requires_hopper = pytest.mark.hopper
requires_blackwell = pytest.mark.blackwell
requires_sm120 = pytest.mark.sm120

GPU_AVAILABLE = torch.cuda.is_available()

ARCH = None
if GPU_AVAILABLE:
    try:
        import tilelang.contrib.nvcc
        _cv = tilelang.contrib.nvcc.get_target_compute_version()
        if _cv == "9.0":
            ARCH = "SM90"
        elif _cv == "10.0":
            ARCH = "SM100"
        elif _cv == "10.3":
            ARCH = "SM103"
        elif _cv == "12.0":
            ARCH = "SM120"
    except Exception:
        pass


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not GPU_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))
        if "hopper" in item.keywords and ARCH != "SM90":
            item.add_marker(pytest.mark.skip(reason="Hopper (SM90) GPU required"))
        if "blackwell" in item.keywords and ARCH not in ["SM100", "SM103"]:
            item.add_marker(pytest.mark.skip(reason="Blackwell (SM100 or SM103) GPU required"))
        if "sm120" in item.keywords and ARCH != "SM120":
            item.add_marker(pytest.mark.skip(reason="Blackwell SM120 GPU required"))
