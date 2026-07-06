"""GIL-release tests for expensive pybind11 bindings (Phase 1 Task 5).

These bindings (Svd, Lanczos/Arnoldi w/ LinOp, Contract, Network.Launch, ...)
now carry py::call_guard<py::gil_scoped_release>() so a long-running C++
computation does not block other Python threads. The threading test below
starts a "ticker" thread that timestamps itself every ~1ms while the main
thread repeatedly calls cytnx.linalg.Svd on a modestly large matrix; if the
GIL were held throughout the Svd call, the ticker would starve and the max
gap between consecutive ticks would balloon to roughly the Svd call
duration. We compare the observed max gap against the measured duration of
a single serial Svd call (scaled with margin) rather than a fixed wall-clock
constant, since the fixed-threshold approach is sensitive to machine load;
this keeps the assertion meaningful across faster/slower CI machines.
"""

import threading
import time

import cytnx


def _time_one_svd(a):
    t0 = time.monotonic()
    cytnx.linalg.Svd(a)
    return time.monotonic() - t0


def test_svd_releases_gil():
    # 400x400 is too fast on Accelerate/LAPACK-backed builds (~0.08s per
    # call) to reliably exceed the ticker-starvation threshold even when the
    # GIL is held for the whole call; 900x900 gives a single-call duration
    # of several hundred ms, which is comfortably observable.
    a = cytnx.random.normal([900, 900], mean=0.0, std=1.0)

    # Measure a single serial Svd call to calibrate the "GIL held too long"
    # threshold to this machine's speed instead of a brittle fixed constant.
    serial_duration = _time_one_svd(a)
    # If a single call is not clearly above the 0.2s jitter floor, the test
    # would pass vacuously even with the GIL held; force a resize instead.
    assert serial_duration > 0.25, (
        "matrix too small to discriminate GIL hold from floor"
    )

    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.monotonic())
            time.sleep(0.001)

    # daemon=True as a backstop: even if the finally below is somehow
    # skipped, a leaked ticker must not hang interpreter shutdown.
    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    try:
        # Give the ticker a moment to start collecting before the heavy calls.
        time.sleep(0.01)
        for _ in range(3):
            cytnx.linalg.Svd(a)
    finally:
        # Always stop the ticker, even if an Svd call raises - a leaked
        # ticker would otherwise outlive the test and mask the real error.
        stop.set()
        th.join()

    gaps = [t1 - t0 for t0, t1 in zip(ticks, ticks[1:])]
    assert gaps, "ticker thread did not record any ticks"
    max_gap = max(gaps)
    # If the GIL were held for the whole call, max_gap would be roughly
    # serial_duration. Require it to be well under that (half, with a floor
    # of 0.2s to stay robust on loaded machines where scheduling jitter alone
    # can cost tens of ms).
    threshold = max(0.2, serial_duration * 0.5)
    assert max_gap < threshold, (
        f"GIL held too long: max_gap={max_gap:.3f}s, "
        f"serial_svd_duration={serial_duration:.3f}s, threshold={threshold:.3f}s"
    )


def test_lanczos_with_python_linop_still_works():
    # Precondition smoke test: subclassing LinOp from Python and overriding
    # matvec() must still work at all (direct call, no solver involved).
    # This does NOT exercise the GIL-released path - see
    # test_lanczos_gnd_with_python_linop below for the trampoline-under-
    # guarded-solver case.
    class MyOp(cytnx.LinOp):
        def __init__(self):
            cytnx.LinOp.__init__(self, "mv", 4)

        def matvec(self, v):
            return v * 2.0

    op = MyOp()
    v = cytnx.ones([4])
    out = op.matvec(v)
    assert out.shape() == [4]


def test_lanczos_gnd_with_python_linop():
    # Cheap real Lanczos call (Gnd = ground state) through a python-defined
    # LinOp, mirroring example/DMRG/dmrg_two_sites_dense.py's Hxx pattern.
    # This exercises the full guarded Lanczos binding with the trampoline
    # matvec callback while the GIL is released around the C++ solver body.
    class Diag(cytnx.LinOp):
        def __init__(self, nx):
            cytnx.LinOp.__init__(self, "mv", nx, cytnx.Type.Double, cytnx.Device.cpu)

        def matvec(self, v):
            out = v.clone()
            for i in range(out.shape()[0]):
                out[i] = out[i].item() * (i + 1)
            return out

    nx = 4
    op = Diag(nx)
    v0 = cytnx.ones([nx])
    energy, psivec = cytnx.linalg.Lanczos(
        Hop=op, method="Gnd", Maxiter=200, CvgCrit=1e-10, Tin=v0
    )
    # smallest eigenvalue of diag(1,2,3,4) is 1
    assert abs(energy[0].item() - 1.0) < 1e-6
