"""pytest-benchmark/pytest-memray regression tests for
variational_manual_grad.py.

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. As with the
Cytnx counterpart, the energy tolerance is wider than the DMRG benchmarks'
because the MPS here starts from an unseeded random unitary evolution and
only takes a fixed, small number of gradient-descent steps.

`test_variational_manual_grad_sweep` scans the full
`common.model.param_grid()` (chi, L) grid instead of the single
regression point above; run a specific point with e.g. `pytest
"test_variational_manual_grad.py::test_variational_manual_grad_sweep[16-20]"
--benchmark-only` so a slow/timed-out point doesn't block the rest.
"""
import math

import pytest

from . import variational_manual_grad
from common.model import STEP_TIMEOUT_SEC, param_grid

CHI = 16
L = 20
REFERENCE_ENERGY = -8.05198427725437


@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_variational_manual_grad_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(variational_manual_grad.run_one, args=(chi, length), rounds=1, iterations=1)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-2)


@pytest.mark.limit_memory("40 MB")
@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_variational_manual_grad_memory(chi, length):
    energy = variational_manual_grad.run_one(chi, length)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-2)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("chi,length", list(param_grid()))
def test_variational_manual_grad_sweep(benchmark, chi, length):
    energy = benchmark.pedantic(variational_manual_grad.run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert math.isfinite(energy)
