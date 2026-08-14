"""Tests for Phase-2 Task 3 of the API-semantics plan (issue #934, #753, #675).

Maintainer ruling (yingjerkao, 2026-07-06, amended): `UniTensor` is a
tensor-network object, not a raw array. The four elementwise
`UniTensor (+) UniTensor` operator families split by whether the operation is
well defined:

  * ``+`` ``-`` ``+=`` ``-=`` are KEPT, but guarded: they require the two
    operands to describe the same tensor slot (matching type, rank, labels,
    rowrank, is_diag, bonds). With matching metadata the sum/difference is
    unambiguous and label-preserving; with mismatched metadata the labels would
    be silently discarded (the #934/#753/#675 complaint), so they raise
    TypeError instead.
  * ``*`` ``/`` ``*=`` ``/=`` are REMOVED: the elementwise (Hadamard)
    product/quotient of two UniTensors is basis-dependent and has no
    tensor-network meaning. They raise TypeError with guidance. Python's
    floor-division ``//`` / ``//=`` map to the distinct ``__floordiv__`` /
    ``__ifloordiv__`` dunders but route to the same removed elementwise
    division, so they must raise too (otherwise ``a // b`` back-doors the
    quotient that ``a / b`` rejects).

The C++ ``operator+``/``operator-`` on UniTensor (declared in
include/linalg.hpp) is unchanged: src/linalg/Lanczos_Gnd_Ut.cpp and
src/linalg/Lanczos_Exp.cpp use it as genuine Krylov-subspace vector-space
arithmetic backing ``cytnx.linalg.Lanczos(..., method="Gnd")`` /
``cytnx.linalg.Lanczos_Exp`` (used by the DMRG and TDVP examples).

Scalar (native Python/numpy number) <-> UniTensor arithmetic is unaffected
and stays public in both directions (`2.0 * ut`, `ut * 2.0`, `ut + 1.5`,
`ut += 2.0`, etc.).
"""

import pytest
import cytnx


def _pair():
    # two structurally-identical Dense UniTensors: same default labels, bonds,
    # rowrank, is_diag -> matching metadata.
    ut1 = cytnx.UniTensor(cytnx.ones([2, 2]))
    ut2 = cytnx.UniTensor(cytnx.ones([2, 2]))
    return ut1, ut2


# ---------------------------------------------------------------------------
# Kept (guarded): UniTensor +/- UniTensor with matching metadata works and
# preserves the shared labels (the #934 label-discarding is what we guard away).
# ---------------------------------------------------------------------------


def test_add_unitensor_matching_metadata_works():
    ut1, ut2 = _pair()
    labels = ut1.labels()
    out = ut1 + ut2
    assert out.get_block()[0, 0].item() == 2.0
    assert out.labels() == labels  # labels preserved, not reset to a plain range


def test_sub_unitensor_matching_metadata_works():
    ut1, ut2 = _pair()
    labels = ut1.labels()
    out = ut1 - ut2
    assert out.get_block()[0, 0].item() == 0.0
    assert out.labels() == labels


def test_iadd_unitensor_matching_metadata_works():
    ut1, ut2 = _pair()
    labels = ut1.labels()
    ut1 += ut2
    assert ut1.get_block()[0, 0].item() == 2.0
    assert ut1.labels() == labels


def test_isub_unitensor_matching_metadata_works():
    ut1, ut2 = _pair()
    labels = ut1.labels()
    ut1 -= ut2
    assert ut1.get_block()[0, 0].item() == 0.0
    assert ut1.labels() == labels


# ---------------------------------------------------------------------------
# Guarded: UniTensor +/- UniTensor with mismatched metadata raises TypeError.
# ---------------------------------------------------------------------------


def test_add_unitensor_mismatched_labels_raises():
    ut1 = cytnx.UniTensor(cytnx.ones([2, 2])).relabel_(["a", "b"])
    ut2 = cytnx.UniTensor(cytnx.ones([2, 2])).relabel_(["c", "d"])
    with pytest.raises(TypeError, match=r"(?s)labels.*934"):
        ut1 + ut2


def test_sub_unitensor_mismatched_labels_raises():
    ut1 = cytnx.UniTensor(cytnx.ones([2, 2])).relabel_(["a", "b"])
    ut2 = cytnx.UniTensor(cytnx.ones([2, 2])).relabel_(["c", "d"])
    with pytest.raises(TypeError, match=r"(?s)labels.*934"):
        ut1 - ut2


def test_add_unitensor_mismatched_bonds_raises():
    # same labels, different bond dimension on leg 'b' -> bond mismatch.
    ut1 = cytnx.UniTensor(cytnx.ones([2, 2])).relabel_(["a", "b"])
    ut2 = cytnx.UniTensor(cytnx.ones([2, 3])).relabel_(["a", "b"])
    with pytest.raises(TypeError, match=r"(?s)934"):
        ut1 + ut2


def test_iadd_unitensor_mismatched_labels_raises():
    ut1 = cytnx.UniTensor(cytnx.ones([2, 2])).relabel_(["a", "b"])
    ut2 = cytnx.UniTensor(cytnx.ones([2, 2])).relabel_(["c", "d"])
    with pytest.raises(TypeError, match=r"(?s)labels.*934"):
        ut1 += ut2


# ---------------------------------------------------------------------------
# Removed: UniTensor * / UniTensor elementwise arithmetic (out-of-place & in-place)
# ---------------------------------------------------------------------------


def test_mul_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*Kron"):
        ut1 * ut2


def test_truediv_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*get_block"):
        ut1 / ut2


def test_imul_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*Kron"):
        ut1 *= ut2


def test_itruediv_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*get_block"):
        ut1 /= ut2


def test_floordiv_unitensor_unitensor_raises():
    # '//' (__floordiv__) is a distinct dunder from '/' but routes to the same
    # removed elementwise division -- it must not back-door the quotient.
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*get_block"):
        ut1 // ut2


def test_ifloordiv_unitensor_unitensor_raises():
    ut1, ut2 = _pair()
    with pytest.raises(TypeError, match=r"(?s)934.*get_block"):
        ut1 //= ut2


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


def test_scalar_floordiv_unitensor_still_works():
    # scope boundary: only the two-UniTensor '//' overload is removed; scalar
    # floor division stays, mirroring scalar '/'. (It routes to linalg::Div, so
    # 'ut // 2.0' matches 'ut / 2.0' rather than performing an integer floor.)
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r1 = ut // 2.0
    assert r1.get_block()[0, 0].item() == 0.5
    ut //= 2.0
    assert ut.get_block()[0, 0].item() == 0.5


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
