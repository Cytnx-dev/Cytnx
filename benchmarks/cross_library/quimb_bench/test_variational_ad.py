"""pytest-benchmark/pytest-memray regression tests for variational_ad.py.

Run timing with `pytest --benchmark-only test_variational_ad.py`, memory
with `pytest --memray test_variational_ad.py`. Both the jax and torch
backends are exercised, matching the module's `--backend jax`/`--backend
torch` split. The MPS here is seeded (`MPS_rand_state(..., seed=0)`), so a
tight tolerance is appropriate.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.import_util import load_sibling_module

_variational_ad = load_sibling_module(__file__, "variational_ad")
run_one_jax = _variational_ad.run_one_jax
run_one_torch = _variational_ad.run_one_torch

CHI = 16
L = 20

BACKEND_CASES = [
    pytest.param(run_one_jax, -8.344500541687012, id="jax"),
    pytest.param(run_one_torch, -8.34450185868216, id="torch"),
]
BACKEND_MEMORY_CASES = [
    pytest.param(run_one_jax, -8.344500541687012, marks=pytest.mark.limit_memory("800 MB"), id="jax"),
    pytest.param(run_one_torch, -8.34450185868216, marks=pytest.mark.limit_memory("100 MB"), id="torch"),
]


@pytest.mark.parametrize("chi,length", [(CHI, L)])
@pytest.mark.parametrize("run_one,reference_energy", BACKEND_CASES)
def test_variational_ad_benchmark(benchmark, run_one, reference_energy, chi, length):
    *_, energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    assert energy == pytest.approx(reference_energy, rel=1e-4)


@pytest.mark.parametrize("chi,length", [(CHI, L)])
@pytest.mark.parametrize("run_one,reference_energy", BACKEND_MEMORY_CASES)
def test_variational_ad_memory(run_one, reference_energy, chi, length):
    *_, energy = run_one(chi, length)
    assert energy == pytest.approx(reference_energy, rel=1e-4)
