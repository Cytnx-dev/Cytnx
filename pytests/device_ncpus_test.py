"""Tests for cytnx.Device.Ncpus, the thread limit of Cytnx's own CPU kernels."""

import os
import subprocess
import sys
import textwrap

import pytest

import cytnx


def test_ncpus_default_is_at_least_one():
    assert cytnx.Device.Ncpus >= 1


def test_setting_ncpus_reads_back():
    saved = cytnx.Device.Ncpus
    try:
        cytnx.Device.Ncpus = 3
        assert cytnx.Device.Ncpus == 3
    finally:
        cytnx.Device.Ncpus = saved


def test_ncpus_rejects_values_below_one():
    saved = cytnx.Device.Ncpus
    with pytest.raises(cytnx.CytnxError, match="at least 1"):
        cytnx.Device.Ncpus = 0
    assert cytnx.Device.Ncpus == saved


def test_device_constants_unchanged():
    assert cytnx.Device.cpu == -1
    assert cytnx.Device.cuda == 0
    assert cytnx.Device.Ngpus >= 0


_TRANSPOSE_THREADS_SCRIPT = textwrap.dedent("""
    import cytnx

    def thread_count():
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("Threads:"):
                    return int(line.split()[1])

    cytnx.Device.Ncpus = 1
    tensor = cytnx.random.uniform([64, 64, 64], -1.0, 1.0)
    before = thread_count()
    tensor.permute(2, 1, 0).contiguous()
    print(thread_count() - before)
""")


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="reads the thread count from /proc/self/status")
def test_setting_ncpus_limits_transpose_threads():
    """Setting Device.Ncpus from Python reaches the C++ library.

    A 64^3 Double transpose (2 MiB) has work for several threads and OpenMP
    allows four, but with Device.Ncpus = 1 the process must start none. A fresh
    interpreter ensures no OpenMP thread pool exists beforehand.
    """
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    completed = subprocess.run([sys.executable, "-c", _TRANSPOSE_THREADS_SCRIPT],
                               env=env, capture_output=True, text=True,
                               check=True)
    assert completed.stdout.split()[-1] == "0"
