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
REFERENCE_ENERGY_JAX = -8.344500541687012
REFERENCE_ENERGY_TORCH = -8.34450185868216


def test_variational_ad_jax_benchmark(benchmark):
    *_, energy = benchmark.pedantic(run_one_jax, args=(CHI, L), rounds=1, iterations=1)
    assert energy == pytest.approx(REFERENCE_ENERGY_JAX, rel=1e-4)


@pytest.mark.limit_memory("800 MB")
def test_variational_ad_jax_memory():
    *_, energy = run_one_jax(CHI, L)
    assert energy == pytest.approx(REFERENCE_ENERGY_JAX, rel=1e-4)


def test_variational_ad_torch_benchmark(benchmark):
    *_, energy = benchmark.pedantic(run_one_torch, args=(CHI, L), rounds=1, iterations=1)
    assert energy == pytest.approx(REFERENCE_ENERGY_TORCH, rel=1e-4)


@pytest.mark.limit_memory("100 MB")
def test_variational_ad_torch_memory():
    *_, energy = run_one_torch(CHI, L)
    assert energy == pytest.approx(REFERENCE_ENERGY_TORCH, rel=1e-4)
