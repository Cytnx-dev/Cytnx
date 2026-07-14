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


# ---------------------------------------------------------------------------
# LinOp.set_elem: numpy_scalar overloads registered ahead of the plain
# double/complex128 ones so every numpy dtype in the keep-set dispatches
# through its own overload rather than an unrelated wider one.
# ---------------------------------------------------------------------------


def test_linop_set_elem_numpy_scalar_dtypes():
    lop = cytnx.LinOp("mv_elem", nx=2, dtype=Type.ComplexDouble)
    lop.set_elem(0, 0, np.float32(1.5))
    lop.set_elem(0, 1, np.complex64(2 + 1j))
    Tin = cytnx.ones([2], dtype=Type.ComplexDouble)
    out = lop.matvec(Tin)
    assert complex(out[0].item()) == pytest.approx(1.5 + (2 + 1j))


# ---------------------------------------------------------------------------
# Tensor's __r*__ operators: the numpy_scalar-typed overloads are dead code
# (issue #692) because Tensor's __iter__ makes numpy's ufunc dispatch try to
# iterate a numpy-scalar-on-the-left operand before ever falling back to
# __radd__/etc.; removed to fix the corresponding mypy.stubtest [misc]
# errors. The plain py::int_/Scalar/double/complex128 overloads are
# unaffected (a plain Python operand never goes through numpy's ufunc
# dispatch) and the pre-existing failure mode for a numpy scalar on the left
# is unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "np_value",
    [np.int64(3), np.float32(1.5), np.complex64(1 + 2j), np.uint16(4), np.bool_(True)],
)
def test_tensor_numpy_scalar_on_left_still_raises_typeerror(np_value):
    t = cytnx.zeros([2], dtype=Type.Double)
    with pytest.raises(TypeError):
        _ = np_value + t


def test_tensor_plain_python_and_scalar_reverse_ops_still_work():
    t = cytnx.ones([2], dtype=Type.Double)
    assert (5 + t)[0].item() == pytest.approx(6.0)
    assert (3.5 + t)[0].item() == pytest.approx(4.5)
    assert (cytnx.Scalar(2.0) + t)[0].item() == pytest.approx(3.0)
