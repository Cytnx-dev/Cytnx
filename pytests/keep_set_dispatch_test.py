"""Regression tests for the binding-level bugs fixed while consolidating
per-dtype duplicate overloads into "keep-sets" (see pybind/pyint_dispatch.hpp's
"KEEP-SET ORDERING"): Storage.from_pylist mis-selecting a dtype for a Python
bool/int list, and Scalar's plain-int constructor's uint64/int64 tie-break.
Also covers a few reachability checks (UniTensor.get_block's qindex dispatch,
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
# Storage.from_pylist: numpy integer/bool scalars are not subclasses of
# Python int/bool (unlike np.float64/np.complex128, which subclass Python
# float/complex and already worked), so a list of them matched nothing in
# pybind11's no-convert pass and fell through to the plain Bool overload's
# convert-pass truthiness fallback -- e.g. from_pylist([np.int32(2)])
# silently produced a Bool storage holding True instead of an Int32 value of
# 2. Fixed by adding a numpy_scalar keep-set ahead of a Bool overload that
# now requires an exact Python bool (no truthiness fallback).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "np_dtype,dtype",
    [
        (np.int64, Type.Int64),
        (np.uint64, Type.Uint64),
        (np.int32, Type.Int32),
        (np.uint32, Type.Uint32),
        (np.int16, Type.Int16),
        (np.uint16, Type.Uint16),
    ],
)
def test_storage_from_pylist_numpy_int_scalar_dtype_and_value(np_dtype, dtype):
    s = cytnx.Storage.from_pylist([np_dtype(2), np_dtype(3)])
    assert s.dtype() == dtype
    assert int(s[0]) == 2
    assert int(s[1]) == 3


def test_storage_from_pylist_numpy_float32_scalar_dtype_and_value():
    s = cytnx.Storage.from_pylist([np.float32(1.5)])
    assert s.dtype() == Type.Float
    assert float(s[0]) == pytest.approx(1.5)


def test_storage_from_pylist_numpy_complex64_scalar_dtype():
    s = cytnx.Storage.from_pylist([np.complex64(1 + 2j)])
    assert s.dtype() == Type.ComplexFloat


def test_storage_from_pylist_numpy_bool_scalar_dtype():
    s = cytnx.Storage.from_pylist([np.bool_(True), np.bool_(False)])
    assert s.dtype() == Type.Bool
    assert bool(s[0]) is True
    assert bool(s[1]) is False


@pytest.mark.parametrize(
    "np_dtype,dtype",
    [(np.int8, Type.Int16), (np.uint8, Type.Uint16)],
)
def test_storage_from_pylist_numpy_8bit_int_widens_to_16bit_not_double(np_dtype, dtype):
    # Same gap as Scalar's constructor (cytnx has no Int8/Uint8 dtype): an
    # 8-bit numpy int has no exact numpy_scalar overload here, and without
    # one it fell through to the double overload -- from_pylist([np.int8(5)])
    # silently produced a Double storage instead of an integer one.
    s = cytnx.Storage.from_pylist([np_dtype(5)])
    assert s.dtype() == dtype
    assert int(s[0]) == 5


def test_storage_from_pylist_empty_list_defaults_to_complex_float():
    # An empty list has no elements for any vector<T> caster to check, so
    # pybind11's no-convert pass accepts it for every from_pylist overload
    # equally and the first-registered one wins regardless of T --
    # from_pylist([]) is at the mercy of registration order alone. This
    # changed from ComplexDouble (before the keep-set consolidation) to
    # ComplexFloat, since the numpy_scalar block (which must precede the
    # plain Bool/py::int_ overloads for correctness -- see
    # test_storage_from_pylist_numpy_int_scalar_dtype_and_value above) opens
    # with numpy_scalar<complex64>. Pinned here as a regression test for the
    # empty-list default itself, not an endorsement of ComplexFloat as
    # "correct" -- see storage_py.cpp's from_pylist comment for why no
    # registration order gives every non-empty list its correct dtype AND
    # preserves the old ComplexDouble empty-list default.
    s = cytnx.Storage.from_pylist([])
    assert s.dtype() == Type.ComplexFloat


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
# Scalar's constructor: cytnx has no Int8/Uint8 dtype, so np.int8/np.uint8
# have no numpy_scalar overload of their own -- without one ahead of the
# raw cytnx_double constructor, they fell through to it and silently
# produced a floating-point Scalar from an integer input (e.g.
# Scalar(np.int8(5)).dtype() became Double). Widened to Int16/Uint16, the
# narrowest integer dtype cytnx does support, instead.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "np_dtype,dtype",
    [(np.int8, Type.Int16), (np.uint8, Type.Uint16)],
)
def test_scalar_numpy_8bit_int_widens_to_16bit_not_double(np_dtype, dtype):
    s = cytnx.Scalar(np_dtype(5))
    assert s.dtype() == dtype
    assert int(s) == 5


def test_scalar_numpy_int8_preserves_sign():
    s = cytnx.Scalar(np.int8(-5))
    assert s.dtype() == Type.Int16
    assert int(s) == -5


# ---------------------------------------------------------------------------
# UniTensor.get_block/get_block_: despite the pybind-level py::arg("qnum")
# name (pre-existing, unrelated to this PR, not renamed here), the vector
# argument is the per-leg quantum-number SECTOR INDEX -- BlockUniTensor's
# get_block matches it against _inner_to_outer_idx after casting to unsigned
# (include/UniTensor.hpp, "this one for Block will return the indicies!!"),
# not the physical charge value. See the guide's worked example:
# https://cytnx-dev.github.io/Cytnx/dev/guide/uniten/blocks.html#getting-a-block-by-its-quantum-number-indices
# (charges [Qs(1), Qs(-1), Qs(0)] map to sector indices [0, 1, 1]). The bond
# below deliberately uses charges 0/1 that happen to equal their own sector
# index, purely so a single assertion can stand in for "the right block was
# selected" -- it is not evidence that get_block accepts charge values in
# general. Originally the vector<int64>/vector<uint64> overloads were
# collapsed into a single py::sequence overload dispatching on magnitude.
# ---------------------------------------------------------------------------


def _u1_pair():
    bi = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1], [cytnx.Symmetry.U1()])
    return cytnx.UniTensor([bi, bi.redirect()])


def test_unitensor_get_block_by_qindex_int_list():
    B = _u1_pair()
    t = cytnx.zeros([1, 1])
    t[0, 0] = 42.0
    B.put_block(t, [0, 0])
    assert B.get_block([0, 0], False)[0, 0].item() == pytest.approx(42.0)
    assert B.get_block_([0, 0], False)[0, 0].item() == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# UniTensor.get_block/get_block_: the qindex consolidation above replaced raw
# vector<cytnx_int64>/vector<cytnx_uint64> overloads (whose arithmetic-type
# casters accept anything satisfying __index__, numpy integer scalars
# included) with a single std::vector<py::int_> dispatcher, whose caster does
# a strict isinstance(int) check with no such fallback. Without a dedicated
# numpy_scalar keep-set ahead of it, get_block([np.int64(0), np.int64(0)],
# False) regressed from working to raising TypeError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "np_dtype",
    [np.int64, np.uint64, np.int32, np.uint32, np.int16, np.uint16],
)
def test_unitensor_get_block_by_qindex_numpy_scalar_list(np_dtype):
    B = _u1_pair()
    t = cytnx.zeros([1, 1])
    t[0, 0] = 42.0
    B.put_block(t, [0, 0])
    qindex = [np_dtype(0), np_dtype(0)]
    assert B.get_block(qindex, False)[0, 0].item() == pytest.approx(42.0)
    assert B.get_block_(qindex, False)[0, 0].item() == pytest.approx(42.0)


def test_unitensor_get_block_by_qindex_numpy_scalar_list_labeled():
    B = _u1_pair()
    t = cytnx.zeros([1, 1])
    t[0, 0] = 42.0
    B.put_block(t, [0, 0])
    qindex = [np.int32(0), np.int32(0)]
    assert B.get_block(B.labels(), qindex, False)[0, 0].item() == pytest.approx(42.0)
    assert B.get_block_(B.labels(), qindex, False)[0, 0].item() == pytest.approx(42.0)


def test_unitensor_get_block_by_qindex_negative_overflow_raises():
    # Same magnitude-dispatch helper as Storage.from_pylist: a value too
    # negative for int64 does not fit uint64 either and must raise. This is
    # a mechanical check of dispatch_pyint_vector's overflow guard -- not a
    # claim that a negative sector index is otherwise meaningful (it isn't;
    # see the section comment above). Match the specific out-of-range
    # message so the test fails if some other error fires instead.
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
