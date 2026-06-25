"""pytest-benchmark/pytest-memray regression tests for tebd.py.

Run timing with `pytest --benchmark-only test_tebd.py`, memory with
`pytest --memray test_tebd.py`.
"""
import pytest

from . import tebd

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
