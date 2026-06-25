"""pytest-benchmark/pytest-memray regression tests for tebd.py.

Run timing with `pytest --benchmark-only test_tebd.py`, memory with
`pytest --memray test_tebd.py`.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.import_util import load_sibling_module

run_one = load_sibling_module(__file__, "tebd").run_one

CHI = 16
L = 20
REFERENCE_ENERGY = 4.750010689839629


def test_tebd_benchmark(benchmark):
    _step_time, _peak_mem_mb, energy = benchmark.pedantic(run_one, args=(CHI, L), rounds=1, iterations=1)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)


@pytest.mark.limit_memory("60 MB")
def test_tebd_memory():
    _step_time, _peak_mem_mb, energy = run_one(CHI, L)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-6)
