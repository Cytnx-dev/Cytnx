"""pytest-benchmark/pytest-memray regression tests for dmrg_dense.py.

Run timing with `pytest --benchmark-only test_dmrg_dense.py`, memory with
`pytest --memray test_dmrg_dense.py`. The energy assertion in both runs
guards against the benchmark silently drifting onto a wrong physical
answer; the reference value below was captured from a verified run of
this script at the same (chi, L) point.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.import_util import load_sibling_module

run_one = load_sibling_module(__file__, "dmrg_dense").run_one

CHI = 16
L = 20
REFERENCE_ENERGY = -8.682468366682716


@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_dmrg_dense_benchmark(benchmark, chi, length):
    *_, energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-4)


@pytest.mark.limit_memory("20 MB")
@pytest.mark.parametrize("chi,length", [(CHI, L)])
def test_dmrg_dense_memory(chi, length):
    *_, energy = run_one(chi, length)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-4)
