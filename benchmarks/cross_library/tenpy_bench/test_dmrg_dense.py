"""pytest-benchmark/pytest-memray regression tests for dmrg_dense.py.

Run timing with `pytest --benchmark-only test_dmrg_dense.py`, memory with
`pytest --memray test_dmrg_dense.py`.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.import_util import load_sibling_module

run_one = load_sibling_module(__file__, "dmrg_dense").run_one

CHI = 16
L = 20
REFERENCE_ENERGY = -8.682468456352291


def test_dmrg_dense_benchmark(benchmark):
    _step_time, _peak_mem_mb, energy = benchmark.pedantic(run_one, args=(CHI, L), rounds=1, iterations=1)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-4)


@pytest.mark.limit_memory("50 MB")
def test_dmrg_dense_memory():
    _step_time, _peak_mem_mb, energy = run_one(CHI, L)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-4)
