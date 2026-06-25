"""pytest-benchmark/pytest-memray regression tests for tdvp.py.

Run timing with `pytest --benchmark-only test_tdvp.py`, memory with
`pytest --memray test_tdvp.py`.

`test_tdvp_sweep` scans the full `common.model.param_grid()` (chi, L)
grid instead of the single regression point above; run a specific point
with e.g. `pytest "test_tdvp.py::test_tdvp_sweep[16-20]" --benchmark-only`
so a slow/timed-out point doesn't block the rest.
"""
import math

import pytest

from . import tdvp
from common.model import STEP_TIMEOUT_SEC, param_grid

CHI = 16
L = 20
REFERENCE_ENERGY = -9.99970394


@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_tdvp_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(tdvp.run_one, args=(chi, length), rounds=1, iterations=1)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)


@pytest.mark.limit_memory("40 MB")
@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_tdvp_memory(chi, length):
    energy = tdvp.run_one(chi, length)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("chi,length", list(param_grid()))
def test_tdvp_sweep(benchmark, chi, length):
    energy = benchmark.pedantic(tdvp.run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert math.isfinite(energy)
