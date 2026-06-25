"""pytest-benchmark/pytest-memray regression tests for tebd.py.

Run timing with `pytest --benchmark-only test_tebd.py`, memory with
`pytest --memray test_tebd.py`.

`test_tebd_sweep` scans the full `common.model.param_grid()` (chi, L)
grid instead of the single regression point above; run a specific point
with e.g. `pytest "test_tebd.py::test_tebd_sweep[16-20]" --benchmark-only`
so a slow/timed-out point doesn't block the rest.
"""
import math

import pytest

from . import tebd
from common.model import STEP_TIMEOUT_SEC, param_grid

CHI = 16
L = 20
REFERENCE_ENERGY = 4.750010689839629


@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_tebd_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(tebd.run_one, args=(chi, length), rounds=1, iterations=1)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)


@pytest.mark.limit_memory("60 MB")
@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_tebd_memory(chi, length):
    energy = tebd.run_one(chi, length)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("chi,length", list(param_grid()))
def test_tebd_sweep(benchmark, chi, length):
    energy = benchmark.pedantic(tebd.run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert math.isfinite(energy)
