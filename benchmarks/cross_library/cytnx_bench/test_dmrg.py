"""pytest-benchmark/pytest-memray regression tests for dmrg_dense.py and
dmrg_symmetric.py.

Run timing with `pytest --benchmark-only test_dmrg.py`, memory with
`pytest --memray test_dmrg.py`. Both the dense and U(1)-symmetric MPO/MPS
code paths are exercised, matching the dmrg_dense.py/dmrg_symmetric.py
split; the two should converge to the same ground energy since the
symmetric MPO was verified against the dense one to machine precision.

`test_dmrg_sweep` scans the full `common.model.param_grid()` (chi, L)
grid instead of the single regression point above; run a specific point
with e.g. `pytest "test_dmrg.py::test_dmrg_sweep[dense-16-20]"
--benchmark-only` so a slow/timed-out point doesn't block the rest.
"""
import math

import pytest

from . import dmrg_dense, dmrg_symmetric
from common.model import STEP_TIMEOUT_SEC, param_grid

CHI = 16
L = 20

SYMMETRY_CASES = [
    pytest.param(dmrg_dense.run_one, -8.682468366682716, id="dense"),
    pytest.param(dmrg_symmetric.run_one, -8.682468353957058, id="u1"),
]
SYMMETRY_MEMORY_CASES = [
    pytest.param(dmrg_dense.run_one, -8.682468366682716, marks=pytest.mark.limit_memory("20 MB"), id="dense"),
    pytest.param(dmrg_symmetric.run_one, -8.682468353957058, marks=pytest.mark.limit_memory("20 MB"), id="u1"),
]
SWEEP_CASES = [
    pytest.param(dmrg_dense.run_one, id="dense"),
    pytest.param(dmrg_symmetric.run_one, id="u1"),
]


@pytest.mark.parametrize("chi,length", [(CHI, L)])
@pytest.mark.parametrize("run_one,reference_energy", SYMMETRY_CASES)
def test_dmrg_benchmark(benchmark, run_one, reference_energy, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    assert energy == pytest.approx(reference_energy, rel=1e-4)


@pytest.mark.parametrize("chi,length", [(CHI, L)])
@pytest.mark.parametrize("run_one,reference_energy", SYMMETRY_MEMORY_CASES)
def test_dmrg_memory(run_one, reference_energy, chi, length):
    energy = run_one(chi, length)
    assert energy == pytest.approx(reference_energy, rel=1e-4)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("chi,length", list(param_grid()))
@pytest.mark.parametrize("run_one", SWEEP_CASES)
def test_dmrg_sweep(benchmark, run_one, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert math.isfinite(energy)
