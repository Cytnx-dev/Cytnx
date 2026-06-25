"""pytest-benchmark/pytest-memray regression tests for variational_ad.py.

Run timing with `pytest --benchmark-only test_variational_ad.py`, memory
with `pytest --memray test_variational_ad.py`. Both the jax and torch
backends are exercised, matching the module's `run_one_jax`/`run_one_torch`
split. The MPS here is seeded (`MPS_rand_state(..., seed=0)`), so a
tight tolerance is appropriate.
"""
import pytest

from . import variational_ad

CHI = 16
L = 20

BACKEND_CASES = [
    pytest.param(variational_ad.run_one_jax, -8.344500541687012, id="jax"),
    pytest.param(variational_ad.run_one_torch, -8.34450185868216, id="torch"),
]
BACKEND_MEMORY_CASES = [
    pytest.param(variational_ad.run_one_jax, -8.344500541687012, marks=pytest.mark.limit_memory("800 MB"), id="jax"),
    pytest.param(variational_ad.run_one_torch, -8.34450185868216, marks=pytest.mark.limit_memory("100 MB"), id="torch"),
]


@pytest.mark.parametrize("chi,length", [(CHI, L)])
@pytest.mark.parametrize("run_one,reference_energy", BACKEND_CASES)
def test_variational_ad_benchmark(benchmark, run_one, reference_energy, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    assert energy == pytest.approx(reference_energy, rel=1e-4)


@pytest.mark.parametrize("chi,length", [(CHI, L)])
@pytest.mark.parametrize("run_one,reference_energy", BACKEND_MEMORY_CASES)
def test_variational_ad_memory(run_one, reference_energy, chi, length):
    energy = run_one(chi, length)
    assert energy == pytest.approx(reference_energy, rel=1e-4)
