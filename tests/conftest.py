import atexit
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


def _under_torchrun():
    """Whether this process was launched by torchrun as part of a group.

    Multi-rank tests are driven by torchrun rather than pytest-xdist, because
    the ranks must rendezvous: xdist workers are independent processes with no
    collective between them. The two launchers are mutually exclusive -- run
    SP=1 under `pytest -n 16` to compile the kernels quickly, and SP>1 under
    `torchrun --nproc_per_node=$SP -m pytest`.
    """
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and "MASTER_ADDR" in os.environ


def _pin_torchrun_rank_to_gpu():
    """Give each torchrun rank its own GPU.

    Like the xdist pinning below this must run before torch initialises CUDA,
    so it narrows CUDA_VISIBLE_DEVICES rather than calling set_device later:
    importing the package reads device properties, which would otherwise bind
    every rank to the same physical GPU before any test runs. Each rank then
    sees its GPU as device 0, which is what NCCL expects.
    """
    if not _under_torchrun():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    devices = _visible_device_ids()
    if not devices:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[int(local_rank) % len(devices)]


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


_pin_torchrun_rank_to_gpu()
_pin_worker_to_gpu()

import pytest
import torch
import torch.distributed as dist

requires_gpu = pytest.mark.gpu
requires_multigpu = pytest.mark.multigpu
requires_hopper = pytest.mark.hopper
requires_blackwell = pytest.mark.blackwell
requires_sm120 = pytest.mark.sm120

GPU_AVAILABLE = torch.cuda.is_available()

# The whole process group is the sequence-parallel dimension: no device mesh,
# no expert or pipeline parallelism to carve out of it.
UNDER_TORCHRUN = _under_torchrun()
SP_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1)) if UNDER_TORCHRUN else 1
SP_RANK = int(os.environ.get("RANK", 0)) if UNDER_TORCHRUN else 0

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
        elif _cv in ["12.0", "12.1"]:
            ARCH = "SM120"
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def init_dist():
    """Stand up the process group once per session, when torchrun launched us."""
    if not UNDER_TORCHRUN or dist.is_initialized():
        return
    dist.init_process_group(backend="nccl")
    atexit.register(dist.destroy_process_group)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "multigpu" in item.keywords:
            if os.environ.get("PYTEST_XDIST_WORKER"):
                item.add_marker(pytest.mark.skip(
                    reason="multi-rank tests need torchrun, not xdist workers"
                ))
            elif SP_WORLD_SIZE < 2:
                item.add_marker(pytest.mark.skip(
                    reason="run under torchrun --nproc_per_node=<SP> with SP > 1"
                ))
        if "gpu" in item.keywords and not GPU_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))
        if "hopper" in item.keywords and ARCH != "SM90":
            item.add_marker(pytest.mark.skip(reason="Hopper (SM90) GPU required"))
        if "blackwell" in item.keywords and ARCH not in ["SM100", "SM103"]:
            item.add_marker(pytest.mark.skip(reason="Blackwell (SM100 or SM103) GPU required"))
        if "sm120" in item.keywords and ARCH != "SM120":
            item.add_marker(pytest.mark.skip(reason="Blackwell SM120 GPU required"))
