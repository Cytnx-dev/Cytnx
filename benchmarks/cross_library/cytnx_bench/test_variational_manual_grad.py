"""pytest-benchmark/pytest-memray regression tests for
variational_manual_grad.py.

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. The
tolerance on the energy assertion is wider than the DMRG benchmarks'
because the MPS tensors here start from an unseeded random initialization
and only take a fixed, small number of gradient-descent steps rather than
running to convergence, so the reached energy is more sensitive to the
random starting point.
"""
import pytest

from . import variational_manual_grad

CHI = 16
L = 20
REFERENCE_ENERGY = -8.682468442899435


@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_variational_manual_grad_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(variational_manual_grad.run_one, args=(chi, length), rounds=1, iterations=1)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-2)


@pytest.mark.limit_memory("20 MB")
@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_variational_manual_grad_memory(chi, length):
    energy = variational_manual_grad.run_one(chi, length)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-2)
