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
REFERENCE_ENERGY = -19.000359069981286


@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_tebd_benchmark(benchmark, chi, length):
    *_, energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-6)


@pytest.mark.limit_memory("20 MB")
@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_tebd_memory(chi, length):
    *_, energy = run_one(chi, length)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-6)
