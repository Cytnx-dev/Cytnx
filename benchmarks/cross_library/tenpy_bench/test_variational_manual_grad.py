"""pytest-benchmark/pytest-memray regression tests for
variational_manual_grad.py.

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. As with the
Cytnx counterpart, the energy tolerance is wider than the DMRG benchmarks'
because the MPS here starts from an unseeded random unitary evolution and
only takes a fixed, small number of gradient-descent steps.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.import_util import load_sibling_module

run_one = load_sibling_module(__file__, "variational_manual_grad").run_one

CHI = 16
L = 20
REFERENCE_ENERGY = -8.05198427725437


def test_variational_manual_grad_benchmark(benchmark):
    *_, energy = benchmark.pedantic(run_one, args=(CHI, L), rounds=1, iterations=1)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-2)


@pytest.mark.limit_memory("40 MB")
def test_variational_manual_grad_memory():
    *_, energy = run_one(CHI, L)
    assert float(energy) == pytest.approx(REFERENCE_ENERGY, rel=1e-2)
