"""pytest-benchmark/pytest-memray regression tests for tdvp.py.

Run timing with `pytest --benchmark-only test_tdvp.py`, memory with
`pytest --memray test_tdvp.py`.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.import_util import load_sibling_module

run_one = load_sibling_module(__file__, "tdvp").run_one

CHI = 16
L = 20
REFERENCE_ENERGY = -9.99970394


def test_tdvp_benchmark(benchmark):
    *_, energy = benchmark.pedantic(run_one, args=(CHI, L), rounds=1, iterations=1)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)


@pytest.mark.limit_memory("40 MB")
def test_tdvp_memory():
    *_, energy = run_one(CHI, L)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)
