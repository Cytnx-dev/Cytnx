"""pytest-benchmark/pytest-memray regression tests for
variational_manual_grad.py.

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. The
tolerance on the energy assertion is wider than the DMRG benchmarks'
because the MPS tensors here start from an unseeded random initialization
and only take a fixed, small number of gradient-descent steps rather than
running to convergence, so the reached energy is more sensitive to the
random starting point.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.import_util import load_sibling_module

run_one = load_sibling_module(__file__, "variational_manual_grad").run_one

CHI = 16
L = 20
REFERENCE_ENERGY = -8.682468442899435


def test_variational_manual_grad_benchmark(benchmark):
    _step_time, _peak_mem_mb, energy = benchmark.pedantic(run_one, args=(CHI, L), rounds=1, iterations=1)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-2)


@pytest.mark.limit_memory("20 MB")
def test_variational_manual_grad_memory():
    _step_time, _peak_mem_mb, energy = run_one(CHI, L)
    assert energy == pytest.approx(REFERENCE_ENERGY, rel=1e-2)
