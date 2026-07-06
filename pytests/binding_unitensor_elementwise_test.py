"""Tests for Phase-2 Task 3 of the API-semantics plan (issue #934, #753, #675).

Maintainer ruling (yingjerkao, 2026-07-06): `UniTensor` is a tensor-network
object, not a raw array. Element-wise `UniTensor (+) UniTensor` arithmetic
(`+`, `-`, `*`, `/`, and their reflected/in-place forms) has no general
tensor-network meaning -- for BlockUniTensor/BlockFermionicUniTensor it either
destroys the block structure or is basis-dependent nonsense (see #934's
enumeration; #753/#675 document the same operations losing labels). The
python-facing dunders for the UniTensor-vs-UniTensor overloads are removed
and now raise TypeError with guidance pointing at Contract()/Kron()/scalar
arithmetic. This is INTENTIONALLY narrower than the C++ layer: the internal
C++ `operator+`/`operator-` on UniTensor (declared in include/linalg.hpp) is
kept, because src/linalg/Lanczos_Gnd_Ut.cpp and src/linalg/Lanczos_Exp.cpp
use it as genuine Krylov-subspace vector-space arithmetic (axpy-style linear
combinations) inside the Lanczos/BiCGSTAB solvers that back
`cytnx.linalg.Lanczos(..., method="Gnd")` and `cytnx.linalg.Lanczos_Exp`
(used by the DMRG and TDVP examples). Python users never call `+`/`-` on two
UniTensors directly for that purpose; they call the solver entry points.

Scalar (Python/numpy number, cytnx.Scalar) <-> UniTensor arithmetic is
unaffected and stays public in both directions (`2.0 * ut`, `ut * 2.0`,
`ut + 1.5`, `ut += 2.0`, etc.).
"""

import pytest
import cytnx


def _pair():
    ut1 = cytnx.UniTensor(cytnx.ones([2, 2]))
    ut2 = cytnx.UniTensor(cytnx.ones([2, 2]))
    return ut1, ut2


# ---------------------------------------------------------------------------
# Removed: UniTensor (+) UniTensor elementwise arithmetic (out-of-place)
# ---------------------------------------------------------------------------


def test_add_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*Contract"):
        ut1 + ut2


def test_sub_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*Contract"):
        ut1 - ut2


def test_mul_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*Kron"):
        ut1 * ut2


def test_truediv_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*get_block"):
        ut1 / ut2


# ---------------------------------------------------------------------------
# Removed: UniTensor (+)= UniTensor elementwise arithmetic (in-place)
# ---------------------------------------------------------------------------


def test_iadd_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*Contract"):
        ut1 += ut2


def test_isub_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*Contract"):
        ut1 -= ut2


def test_imul_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*Kron"):
        ut1 *= ut2


def test_itruediv_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*get_block"):
        ut1 /= ut2


# ---------------------------------------------------------------------------
# Kept: scalar (+) UniTensor / UniTensor (+) scalar in both directions,
# out-of-place and in-place.
# ---------------------------------------------------------------------------


def test_scalar_mul_unitensor_both_directions_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r1 = 2.0 * ut
    r2 = ut * 2.0
    assert r1.get_block()[0, 0].item() == 2.0
    assert r2.get_block()[0, 0].item() == 2.0


def test_scalar_add_unitensor_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r1 = ut + 1.5
    r2 = 1.5 + ut
    assert r1.get_block()[0, 0].item() == 2.5
    assert r2.get_block()[0, 0].item() == 2.5


def test_scalar_sub_unitensor_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r1 = ut - 0.5
    r2 = 3.0 - ut
    assert r1.get_block()[0, 0].item() == 0.5
    assert r2.get_block()[0, 0].item() == 2.0


def test_scalar_truediv_unitensor_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r1 = ut / 2.0
    r2 = 4.0 / ut
    assert r1.get_block()[0, 0].item() == 0.5
    assert r2.get_block()[0, 0].item() == 4.0


def test_iadd_scalar_still_works():
    # NOTE: identity preservation for in-place dunders is Task 2's concern
    # (binding_inplace_test.py); this test only pins down that the scalar
    # in-place arithmetic still succeeds and produces the right value.
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    ut += 2.0
    assert ut.get_block()[0, 0].item() == 3.0


def test_isub_scalar_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    ut -= 0.5
    assert ut.get_block()[0, 0].item() == 0.5


def test_imul_scalar_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    ut *= 3.0
    assert ut.get_block()[0, 0].item() == 3.0


def test_itruediv_scalar_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    ut /= 2.0
    assert ut.get_block()[0, 0].item() == 0.5
