"""Timing / memory measurement helpers shared by every benchmark script.

CPU memory is measured via `tracemalloc` (pure-Python/NumPy heap) combined
with the process RSS delta from `resource.getrusage`, since neither alone
captures everything a tensor library allocates (NumPy arrays go through the
C allocator and are visible to RSS but not always to tracemalloc; small
Python-object overhead is the opposite).

GPU memory is measured via the backend's own peak-allocator counter
(`torch.cuda.max_memory_allocated`, `cupy`'s memory pool, or JAX's device
memory stats) since host-side RSS does not reflect device allocations.
These GPU helpers are written for completeness but are not exercised in
this environment, which has no GPU.
"""

import csv
import gc
import os
import resource
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, asdict


@dataclass
class StepMeasurement:
    library: str
    algorithm: str
    symmetry: str  # "dense" or "u1"
    device: str  # "cpu" or "gpu"
    backend: str  # e.g. "numpy", "jax", "torch", autodiff/manual-grad tag
    L: int
    chi: int
    step_time_sec: float
    peak_mem_mb: float


class CSVResultWriter:
    """Append StepMeasurement rows to a CSV file, one per benchmark run."""

    FIELDS = list(StepMeasurement.__dataclass_fields__.keys())

    def __init__(self, path):
        self.path = path
        self._wrote_header = os.path.exists(path) and os.path.getsize(path) > 0

    def write(self, measurement: StepMeasurement):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(asdict(measurement))


@contextmanager
def cpu_timed_block():
    """Measure wall-clock time and peak (tracemalloc + RSS-delta) memory of
    a CPU code block. Usage::

        with cpu_timed_block() as result:
            do_work()
        result["time_sec"], result["peak_mem_mb"]
    """
    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    tracemalloc.start()
    t0 = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        t1 = time.perf_counter()
        _, peak_tracemalloc = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_delta_mb = max(0.0, (rss_after - rss_before) / 1024.0)
        tracemalloc_mb = peak_tracemalloc / (1024.0 * 1024.0)
        result["time_sec"] = t1 - t0
        # ru_maxrss is a high-water mark for the whole process, so it only
        # grows; tracemalloc gives a tighter per-block estimate for the
        # Python/NumPy heap. Report whichever is larger as the conservative
        # peak-memory estimate for this block.
        result["peak_mem_mb"] = max(rss_delta_mb, tracemalloc_mb)


@contextmanager
def torch_gpu_timed_block(device="cuda"):
    """GPU analogue of cpu_timed_block using torch.cuda's peak allocator
    counter. Not exercised in this environment (no GPU available)."""
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        result["time_sec"] = t1 - t0
        result["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0**2)


@contextmanager
def jax_gpu_timed_block():
    """GPU analogue of cpu_timed_block for the JAX backend. JAX has no
    built-in peak-allocator counter as direct as torch's, so this reads the
    device memory stats exposed by the backend (XLA/CUDA) before/after the
    block. Not exercised in this environment (no GPU available)."""
    import jax

    device = jax.devices("gpu")[0]
    t0 = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        t1 = time.perf_counter()
        result["time_sec"] = t1 - t0
        stats = device.memory_stats() or {}
        result["peak_mem_mb"] = stats.get("peak_bytes_in_use", 0) / (1024.0**2)


@contextmanager
def cytnx_gpu_timed_block():
    """GPU analogue of cpu_timed_block for Cytnx's CUDA backend. Cytnx
    allocates GPU memory through its own cached allocator (see
    src/Device.cpp); peak usage is read back via cytnx.cytnx_memory_usage()
    style device queries where available. Not exercised in this
    environment (no GPU available)."""
    import cytnx

    cytnx.cudaDeviceSynchronize()
    t0 = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        cytnx.cudaDeviceSynchronize()
        t1 = time.perf_counter()
        result["time_sec"] = t1 - t0
        # cytnx exposes per-device memory accounting through cytnx.Device;
        # fall back to 0 if the installed version does not expose it.
        result["peak_mem_mb"] = getattr(cytnx, "cudaMemGetInfo", lambda: (0, 0))()[0] / (1024.0**2)
