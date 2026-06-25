"""pytest-benchmark/pytest-memray regression tests for dmrg_symmetric.py.

Run timing with `pytest --benchmark-only test_dmrg_symmetric.py`, memory
with `pytest --memray test_dmrg_symmetric.py`. Per the module docstring,
this script runs imaginary-time evolution of a random (seeded) state
rather than a converged ground-state search, so the reference value below
is a reproducibility check against a previously observed run, not a
ground-energy correctness check.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.import_util import load_sibling_module

run_one = load_sibling_module(__file__, "dmrg_symmetric").run_one

CHI = 16
L = 20
REFERENCE_ENERGY = -0.564136128480123


def test_dmrg_symmetric_benchmark(benchmark):
    _step_time, _peak_mem_mb, energy = benchmark.pedantic(run_one, args=(CHI, L), rounds=1, iterations=1)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)


@pytest.mark.limit_memory("700 MB")
def test_dmrg_symmetric_memory():
    _step_time, _peak_mem_mb, energy = run_one(CHI, L)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)
