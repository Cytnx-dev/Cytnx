"""Regression tests for the binding-level bugs fixed while consolidating
per-dtype duplicate overloads into "keep-sets" (see pybind/pyint_dispatch.hpp's
"KEEP-SET ORDERING"): Storage.from_pylist mis-selecting a dtype for a Python
bool/int list, and Scalar's plain-int constructor's uint64/int64 tie-break.
Also covers a few reachability checks (UniTensor.get_block's qnum dispatch,
LinOp.set_elem's numpy-scalar dispatch, Tensor's numpy-scalar-on-the-left
dead reverse operators) added alongside the same consolidation.
"""

import numpy as np
import pytest

import cytnx
from cytnx import Type


# ---------------------------------------------------------------------------
# Storage.from_pylist: Python bool is an int subclass, so an int-list overload
# checked ahead of the bool overload silently produced a Uint64/Int64-dtype
# Storage from a list of bools instead of Bool.
# ---------------------------------------------------------------------------


def test_storage_from_pylist_bool_dtype():
    s = cytnx.Storage.from_pylist([True, False])
    assert s.dtype() == Type.Bool
    assert bool(s[0]) is True
    assert bool(s[1]) is False


def test_storage_from_pylist_int_prefers_int64():
    s = cytnx.Storage.from_pylist([1, 2, 3])
    assert s.dtype() == Type.Int64


def test_storage_from_pylist_negative_int_stays_int64():
    s = cytnx.Storage.from_pylist([-1, 2, -3])
    assert s.dtype() == Type.Int64
    assert int(s[0]) == -1


def test_storage_from_pylist_int_overflow_uses_uint64():
    big = 2**63 + 5  # does not fit in int64, needs uint64
    s = cytnx.Storage.from_pylist([big])
    assert s.dtype() == Type.Uint64
    assert int(s[0]) == big


def test_storage_from_pylist_negative_overflow_raises():
    # A magnitude too negative for int64 does not fit uint64 either (uint64
    # cannot represent any negative value); must raise, not silently cast.
    # Match the specific out-of-range message so this doesn't pass merely
    # because *some* unrelated CytnxError was raised.
    too_negative = -(2**64)
    with pytest.raises(cytnx.CytnxError, match="out of the supported int64/uint64 range"):
        cytnx.Storage.from_pylist([too_negative])


def test_storage_from_pylist_float_and_complex_dtype():
    assert cytnx.Storage.from_pylist([1.5]).dtype() == Type.Double
    assert cytnx.Storage.from_pylist([1 + 2j]).dtype() == Type.ComplexDouble


# ---------------------------------------------------------------------------
# Scalar's plain-int constructor: adopts dispatch_pyint's int64-preferred
# convention (matching every other keep-set in the codebase) instead of
# uint64 winning purely from being registered first.
# ---------------------------------------------------------------------------


def test_scalar_plain_int_prefers_int64():
    assert cytnx.Scalar(5).dtype() == Type.Int64
    assert cytnx.Scalar(-5).dtype() == Type.Int64
    assert int(cytnx.Scalar(-5)) == -5


def test_scalar_plain_int_overflow_uses_uint64():
    big = 2**63 + 5
    s = cytnx.Scalar(big)
    assert s.dtype() == Type.Uint64
    # float(), not int(): Scalar.__int__ has a separate, pre-existing signed-
    # reinterpretation bug for Uint64 values above int64 max, out of scope
    # here -- this only checks the constructor picked the right dtype.
    assert float(s) == pytest.approx(float(big))


# ---------------------------------------------------------------------------
# UniTensor.get_block/get_block_: qnum vector<int64>/vector<uint64> collapsed
# into a single py::sequence overload dispatching on magnitude.
# ---------------------------------------------------------------------------


def _u1_pair():
    bi = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1], [cytnx.Symmetry.U1()])
    return cytnx.UniTensor([bi, bi.redirect()])


def test_unitensor_get_block_by_qnum_int_list():
    B = _u1_pair()
    t = cytnx.zeros([1, 1])
    t[0, 0] = 42.0
    B.put_block(t, [0, 0])
    assert B.get_block([0, 0], False)[0, 0].item() == pytest.approx(42.0)
    assert B.get_block_([0, 0], False)[0, 0].item() == pytest.approx(42.0)


def test_unitensor_get_block_by_qnum_negative_overflow_raises():
    # Same magnitude-dispatch helper as Storage.from_pylist: a value too
    # negative for int64 does not fit uint64 either and must raise. This is
    # not about U1 physically supporting negative charges (it does) -- match
    # the specific out-of-range message so the test fails if some unrelated
    # qnum-validation error fires instead of dispatch_pyint_vector's check.
    B = _u1_pair()
    too_negative = -(2**64)
    with pytest.raises(cytnx.CytnxError, match="out of the supported int64/uint64 range"):
        B.get_block([too_negative, 0], False)


# ---------------------------------------------------------------------------
# LinOp.set_elem: numpy_scalar overloads registered ahead of the plain
# double/complex128 ones, matching every other keep-set's convention. This is
# a reachability check only, not a discriminating regression test: set_elem's
# target assignment (Tensor::Tproxy::operator=) tolerates a value routed
# through a wider overload (e.g. a real value cast via the raw complex128
# overload instead of numpy_scalar<float>) without raising or changing the
# resulting numeric value, so no test can observe *which* overload matched
# from the result alone. See 09cdee2's commit message.
# ---------------------------------------------------------------------------


def test_linop_set_elem_numpy_scalar_dtypes_are_accepted():
    lop = cytnx.LinOp("mv_elem", nx=2, dtype=Type.ComplexDouble)
    lop.set_elem(0, 0, np.float32(1.5))
    lop.set_elem(0, 1, np.complex64(2 + 1j))
    Tin = cytnx.ones([2], dtype=Type.ComplexDouble)
    out = lop.matvec(Tin)
    assert complex(out[0].item()) == pytest.approx(1.5 + (2 + 1j))


# ---------------------------------------------------------------------------
# Tensor's __r*__ operators: the numpy_scalar-typed overloads were dead code
# (issue #692, a genuine unfixed bug, not by design) because Tensor's
# __iter__ makes numpy's ufunc dispatch try to iterate a numpy-scalar-on-the-
# left operand before ever falling back to __radd__/etc.; removed to fix the
# corresponding mypy.stubtest [misc] errors. The still-broken numpy-scalar-
# on-the-left case is intentionally not pinned by a test here -- #692 tracks
# it, and a test asserting it still fails would need updating (and could be
# forgotten) the moment #692 is fixed. The regression coverage that matters
# for this PR is that the *kept* overloads (int/Scalar/double/complex128)
# still work, below.
# ---------------------------------------------------------------------------


def test_tensor_plain_python_and_scalar_reverse_ops_still_work():
    t = cytnx.ones([2], dtype=Type.Double)
    assert (5 + t)[0].item() == pytest.approx(6.0)
    assert (3.5 + t)[0].item() == pytest.approx(4.5)
    assert (cytnx.Scalar(2.0) + t)[0].item() == pytest.approx(3.0)
