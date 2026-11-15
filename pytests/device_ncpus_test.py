"""Tests for cytnx.Device.Ncpus, the thread limit of Cytnx's own CPU kernels."""

import os
import subprocess
import sys
import textwrap

import pytest

import cytnx


def _ncpus_in_fresh_interpreter(cytnx_num_threads):
    """Returns (Device.Ncpus, stderr) for a new interpreter.

    The default is read from CYTNX_NUM_THREADS when cytnx is imported, so each
    setting needs its own process. None leaves the variable unset.
    """
    env = dict(os.environ)
    env.pop("CYTNX_NUM_THREADS", None)
    if cytnx_num_threads is not None:
        env["CYTNX_NUM_THREADS"] = cytnx_num_threads
    completed = subprocess.run(
        [sys.executable, "-c", "import cytnx; print(cytnx.Device.Ncpus)"],
        env=env, capture_output=True, text=True, check=True)
    return int(completed.stdout.split()[-1]), completed.stderr


def test_ncpus_default_is_one():
    assert _ncpus_in_fresh_interpreter(None)[0] == 1


def test_cytnx_num_threads_sets_the_default():
    assert _ncpus_in_fresh_interpreter("3")[0] == 3


@pytest.mark.parametrize("setting", ["0", "-2", "four", "2x"])
def test_malformed_cytnx_num_threads_falls_back_to_one(setting):
    ncpus, stderr = _ncpus_in_fresh_interpreter(setting)
    assert ncpus == 1
    assert "CYTNX_NUM_THREADS" in stderr


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
    import sys

    import cytnx

    def thread_count():
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("Threads:"):
                    return int(line.split()[1])

    cytnx.Device.Ncpus = int(sys.argv[1])
    tensor = cytnx.random.uniform([64, 64, 64], -1.0, 1.0)
    before = thread_count()
    tensor.permute(2, 1, 0).contiguous()
    print(thread_count() - before)
""")


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="reads the thread count from /proc/self/status")
@pytest.mark.parametrize("initial, assigned, expect_threads", [
    (None, 4, True),
    ("4", 1, False),
])
def test_assigning_ncpus_reaches_transposes(initial, assigned, expect_threads):
    """Assigning Device.Ncpus from Python changes the threads a transpose uses.

    A 64^3 Double transpose (2 MiB) has work for many threads and OpenMP allows
    four. Raising Device.Ncpus to 4 must start threads; lowering it to 1 after
    CYTNX_NUM_THREADS=4 must start none. A fresh interpreter ensures no OpenMP
    thread pool exists beforehand.
    """
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    env.pop("CYTNX_NUM_THREADS", None)
    if initial is not None:
        env["CYTNX_NUM_THREADS"] = initial
    completed = subprocess.run(
        [sys.executable, "-c", _TRANSPOSE_THREADS_SCRIPT, str(assigned)],
        env=env, capture_output=True, text=True, check=True)
    started = int(completed.stdout.split()[-1])
    if expect_threads:
        assert started > 0
    else:
        assert started == 0
