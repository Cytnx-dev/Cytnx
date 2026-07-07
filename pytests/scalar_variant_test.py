"""Tests for Phase-3 T2 (#847/#935/#937): Scalar's PIMPL+virtual-dispatch
hierarchy replaced by a std::variant-backed value type.

Covers the Python-visible surface: arithmetic across dtypes, the new
mixed-dtype in-place throw behavior (Ruling 1 -- surfaces as RuntimeError on
this base, since #989's cytnx.CytnxError translation has not merged here),
Sproxy round-trip via Storage indexing, and numpy-scalar constructor paths.
"""

import numpy as np
import pytest

import cytnx
from cytnx import Type


def _s(value, np_dtype):
    """Construct a cytnx.Scalar with an explicit dtype via a numpy scalar,
    since the Python binding only exposes the 11 single-arg typed
    constructors (+ numpy-scalar overloads), not the C++-only (value, dtype)
    two-argument constructor."""
    return cytnx.Scalar(np_dtype(value))


# ---------------------------------------------------------------------------
# Basic arithmetic across types
# ---------------------------------------------------------------------------


def test_float_arithmetic_correct_values():
    a = cytnx.Scalar(np.float64(3.0))
    b = cytnx.Scalar(np.float64(2.0))
    assert (a + b).dtype() == Type.Double
    assert float(a + b) == pytest.approx(5.0)
    assert float(a - b) == pytest.approx(1.0)
    assert float(a * b) == pytest.approx(6.0)
    assert float(a / b) == pytest.approx(1.5)


def test_complex_arithmetic_correct_values():
    a = cytnx.Scalar(np.complex128(3 + 4j))
    b = cytnx.Scalar(np.complex128(1 - 2j))
    r = a + b
    assert r.dtype() == Type.ComplexDouble
    assert complex(r) == pytest.approx(4 + 2j)


@pytest.mark.parametrize(
    "np_l,np_r,expected_dtype",
    [
        (np.int64, np.int32, Type.Int64),
        (np.float64, np.float32, Type.Double),
        (np.complex128, np.float64, Type.ComplexDouble),
        (np.complex64, np.float64, Type.ComplexDouble),  # #935 fix: not ComplexFloat
    ],
)
def test_binary_arithmetic_promotes_via_type_promote(np_l, np_r, expected_dtype):
    a = _s(3, np_l)
    b = _s(2, np_r)
    r = a + b
    assert r.dtype() == expected_dtype


def test_mixed_signed_unsigned_binary_arithmetic_does_not_raise():
    # Out-of-place arithmetic between signed/unsigned integers always
    # promotes via Type.type_promote and never raises.
    a = cytnx.Scalar(np.int64(3))
    b = cytnx.Scalar(np.uint64(2))
    r = a + b
    assert float(r) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Ruling 1: in-place mixed-dtype ops raise (RuntimeError on this base, ahead
# of #989's cytnx.CytnxError translation).
# ---------------------------------------------------------------------------


def test_inplace_int_plus_float_raises_runtime_error():
    s = cytnx.Scalar(np.int64(3))
    with pytest.raises(RuntimeError):
        s += cytnx.Scalar(np.float64(2.0))


def test_inplace_signed_unsigned_mix_raises_runtime_error():
    # Unsigned-dtype Scalars are constructed via astype() here: the pybind
    # numpy-scalar constructor overloads currently resolve every integer
    # numpy type except int64 to the Int64 constructor (a pre-existing
    # binding quirk, see test_numpy_scalar_constructor_paths_narrower_ints_
    # preexisting_quirk below), so astype() is the reliable way to pin an
    # unsigned dtype from Python.
    s = cytnx.Scalar(np.int64(3)).astype(Type.Uint64)
    assert s.dtype() == Type.Uint64
    with pytest.raises(RuntimeError):
        s -= cytnx.Scalar(np.int64(2))


def test_inplace_real_complex_mix_raises_runtime_error():
    s = cytnx.Scalar(np.float64(3.0))
    with pytest.raises(RuntimeError):
        s *= cytnx.Scalar(np.complex128(2 + 1j))


def test_inplace_bool_with_nonbool_raises_runtime_error():
    s = cytnx.Scalar(np.bool_(True))
    with pytest.raises(RuntimeError):
        s += cytnx.Scalar(np.int64(1))


def test_inplace_lossless_widening_does_not_raise():
    # Ruling 1 is permissive whenever Type.type_promote(lhs, rhs) ==
    # lhs.dtype(): Int64 += Int32, Double += Float, ComplexDouble += Double
    # all remain valid in-place, matching #937's own migration guidance
    # ("cast to a floating type first" is unnecessary when the destination
    # already losslessly contains the source).
    s = cytnx.Scalar(np.int64(3))
    s += cytnx.Scalar(np.int32(2))
    assert s.dtype() == Type.Int64
    assert int(s) == 5

    d = cytnx.Scalar(np.float64(3.0))
    d += cytnx.Scalar(np.float32(2.0))
    assert d.dtype() == Type.Double
    assert float(d) == pytest.approx(5.0)

    cd = cytnx.Scalar(np.complex128(3 + 0j))
    cd += cytnx.Scalar(np.float64(2.0))
    assert cd.dtype() == Type.ComplexDouble
    assert complex(cd) == pytest.approx(5 + 0j)


def test_inplace_same_dtype_never_raises():
    s = cytnx.Scalar(np.int32(3))
    s += cytnx.Scalar(np.int32(2))
    s -= cytnx.Scalar(np.int32(1))
    s *= cytnx.Scalar(np.int32(2))
    assert int(s) == 8


def test_numpy_scalar_inplace_operators_raise_on_lossy_mix():
    # The pybind FOR_EACH_NUMPY_ITYPE overloads route through Scalar's C++
    # operator+=/-=/*=//=, so a numpy-scalar RHS is subject to the same
    # Ruling 1 guard as a Scalar RHS.
    s = cytnx.Scalar(np.int64(3))
    with pytest.raises(RuntimeError):
        s += np.float64(2.0)


# ---------------------------------------------------------------------------
# #935 fixes visible from Python: abs()/iabs() dtype, minval, self-assignment
# ---------------------------------------------------------------------------


def test_abs_returns_real_dtype_for_complex():
    z = cytnx.Scalar(np.complex128(3 + 4j))
    a = z.abs()
    assert a.dtype() == Type.Double
    assert float(a) == pytest.approx(5.0)


def test_minval_double_is_most_negative_not_smallest_positive():
    m = cytnx.Scalar.minval(Type.Double)
    assert float(m) < 0.0
    assert float(m) == pytest.approx(np.finfo(np.float64).min)


def test_self_assignment_and_self_inplace_are_safe():
    s = cytnx.Scalar(np.float64(3.5))
    s += s
    assert float(s) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Storage indexed assignment from a Scalar (the Python-visible entry point
# into Scalar::Sproxy / Storage::set_item(idx, const Scalar&) -- see
# pybind/storage_py.cpp's __setitem__/__getitem__, which cast the Scalar via
# its __int__/__float__/__complex__ operators into self.at<T>(idx) rather
# than returning a Scalar from __getitem__. Bool storage is excluded: casting
# a Scalar to C++ bool has no pybind path today (no __bool__ registered in
# scalar_py.cpp) -- a pre-existing gap, unchanged by this refactor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,np_dtype",
    [
        (Type.ComplexDouble, np.complex128),
        (Type.ComplexFloat, np.complex64),
        (Type.Double, np.float64),
        (Type.Float, np.float32),
        (Type.Int64, np.int64),
        (Type.Uint64, np.uint64),
        (Type.Int32, np.int32),
        (Type.Uint32, np.uint32),
        (Type.Int16, np.int16),
        (Type.Uint16, np.uint16),
    ],
)
def test_storage_setitem_getitem_scalar_roundtrip(dtype, np_dtype):
    storage = cytnx.Storage(4, dtype)
    value = cytnx.Scalar(np_dtype(3))
    storage[1] = value
    got = storage[1]
    if isinstance(got, (float, complex)):
        assert got == pytest.approx(3)
    else:
        assert got == 3


def test_storage_setitem_cross_dtype_scalar_converts():
    storage = cytnx.Storage(2, Type.Double)
    storage[0] = cytnx.Scalar(np.int64(7))
    got = storage[0]
    assert got == pytest.approx(7.0)


def test_tensor_setitem_with_scalar_still_works():
    t = cytnx.zeros([3], dtype=Type.Double)
    t[1] = cytnx.Scalar(np.float64(9.0))
    assert t[1].item() == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# numpy scalar constructor paths (all 11 dtypes + complex)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "np_dtype,dtype",
    [
        (np.complex128, Type.ComplexDouble),
        (np.complex64, Type.ComplexFloat),
        (np.float64, Type.Double),
        (np.float32, Type.Float),
        (np.int64, Type.Int64),
        (np.bool_, Type.Bool),
    ],
)
def test_numpy_scalar_constructor_paths(np_dtype, dtype):
    val = np_dtype(True) if dtype == Type.Bool else np_dtype(5)
    s = cytnx.Scalar(val)
    assert s.dtype() == dtype


@pytest.mark.parametrize(
    "np_dtype",
    [np.uint64, np.int32, np.uint32, np.int16, np.uint16],
)
def test_numpy_scalar_constructor_paths_narrower_ints_preexisting_quirk(np_dtype):
    # Pre-existing pybind11 overload-resolution behavior (confirmed present
    # on master d6dcd160, before this refactor): numpy scalar types narrower
    # than int64/uint64 (uint64, int32, uint32, int16, uint16) all resolve to
    # the Int64 constructor overload instead of their own. This is a
    # pre-existing pybind/numpy_scalar dispatch quirk in scalar_py.cpp's
    # __init__ overload set, not something introduced or fixed by the
    # Scalar->std::variant refactor (T2's scope is Scalar's internals; the
    # pybind constructor overload set is unchanged). Documented here so a
    # future fix has a regression test to flip green; not treated as a T2
    # gate failure.
    s = cytnx.Scalar(np_dtype(5))
    assert s.dtype() == Type.Int64  # pinned pre-existing behavior, not the "correct" dtype


def test_comparison_operators_across_dtypes():
    a = cytnx.Scalar(np.int64(3))
    b = cytnx.Scalar(np.int32(3))
    assert a == b
    assert not (a != b)
    assert a <= b
    assert a >= b


def test_complex_equality_does_not_raise():
    # Equality is well-defined for complex Scalars (unlike ordering), and
    # must not raise.
    a = cytnx.Scalar(np.complex128(1 + 2j))
    b = cytnx.Scalar(np.complex128(1 + 2j))
    assert a == b


def test_complex_ordering_raises():
    a = cytnx.Scalar(np.complex128(1 + 2j))
    b = cytnx.Scalar(np.complex128(3 + 4j))
    with pytest.raises(RuntimeError):
        _ = a < b
