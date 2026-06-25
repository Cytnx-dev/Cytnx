"""pytest-benchmark/pytest-memray regression tests for tdvp.py.

Run timing with `pytest --benchmark-only test_tdvp.py`, memory with
`pytest --memray test_tdvp.py`.
"""
import pytest

from . import tdvp

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
