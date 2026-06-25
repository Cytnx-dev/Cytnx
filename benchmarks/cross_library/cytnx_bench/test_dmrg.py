"""pytest-benchmark/pytest-memray regression tests for dmrg_dense.py and
dmrg_symmetric.py.

Run timing with `pytest --benchmark-only test_dmrg.py`, memory with
`pytest --memray test_dmrg.py`. Both the dense and U(1)-symmetric MPO/MPS
code paths are exercised, matching the dmrg_dense.py/dmrg_symmetric.py
split; the two should converge to the same ground energy since the
symmetric MPO was verified against the dense one to machine precision.
"""
import pytest

from . import dmrg_dense, dmrg_symmetric

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
