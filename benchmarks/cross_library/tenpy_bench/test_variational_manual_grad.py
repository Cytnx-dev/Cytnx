"""pytest-benchmark/pytest-memray regression tests for
variational_manual_grad.py.

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. As with the
Cytnx counterpart, the energy tolerance is wider than the DMRG benchmarks'
because the MPS here starts from an unseeded random unitary evolution and
only takes a fixed, small number of gradient-descent steps.
"""
import pytest

from . import variational_manual_grad

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
