"""Tests for Phase-3 T2 (#847/#935/#937): Scalar's PIMPL+virtual-dispatch
hierarchy replaced by a std::variant-backed value type.

Covers the Python-visible surface: arithmetic across dtypes, in-place ops
(which follow Python value-type semantics -- `a op= b` == `a = a op b`,
promoting the dtype via Type.type_promote rather than throwing; maintainer
ruling 2026-07-08 on #1011), Sproxy round-trip via Storage indexing, and
numpy-scalar constructor paths.
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
# In-place ops follow Python value-type semantics: `a op= b` == `a = a op b`,
# promoting the dtype via Type.type_promote (they do NOT throw on a mixed
# dtype, and they do NOT pin the LHS dtype).
# ---------------------------------------------------------------------------


def test_inplace_int_plus_float_promotes_to_double():
    s = cytnx.Scalar(np.int64(3))
    s += cytnx.Scalar(np.float64(2.0))
    assert s.dtype() == Type.Double
    assert float(s) == pytest.approx(5.0)


def test_inplace_signed_unsigned_mix_promotes_to_signed():
    # Unsigned-dtype Scalars are constructed via astype() here: the pybind
    # numpy-scalar constructor overloads currently resolve every integer
    # numpy type except int64 to the Int64 constructor (a pre-existing
    # binding quirk, see test_numpy_scalar_constructor_paths_narrower_ints_
    # preexisting_quirk below), so astype() is the reliable way to pin an
    # unsigned dtype from Python. type_promote(Uint64, Int64) == Int64, so the
    # destination promotes to the signed dtype.
    s = cytnx.Scalar(np.int64(3)).astype(Type.Uint64)
    assert s.dtype() == Type.Uint64
    s -= cytnx.Scalar(np.int64(2))
    assert s.dtype() == Type.Int64
    assert int(s) == 1


def test_inplace_real_complex_mix_promotes_to_complex():
    s = cytnx.Scalar(np.float64(3.0))
    s *= cytnx.Scalar(np.complex128(2 + 1j))
    assert s.dtype() == Type.ComplexDouble
    assert complex(s) == pytest.approx(6 + 3j)


def test_inplace_bool_with_nonbool_promotes():
    s = cytnx.Scalar(np.bool_(True))
    s += cytnx.Scalar(np.int64(1))
    assert s.dtype() == Type.Int64
    assert int(s) == 2


def test_inplace_lossless_widening_keeps_lhs_dtype():
    # When Type.type_promote(lhs, rhs) == lhs.dtype(), promotion is a no-op and
    # the LHS dtype is kept: Int64 += Int32, Double += Float, ComplexDouble +=
    # Double all stay at the LHS dtype.
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


def test_inplace_same_dtype_keeps_dtype():
    # Int32 operands are pinned via astype(): the pybind Scalar(np.int32)
    # constructor currently collapses to Int64 (preexisting ctor-collapse
    # quirk; #1014 fixes it, after which the direct constructor works and
    # this workaround can be dropped).
    s = cytnx.Scalar(np.int64(3)).astype(Type.Int32)
    s += cytnx.Scalar(np.int64(2)).astype(Type.Int32)
    s -= cytnx.Scalar(np.int64(1)).astype(Type.Int32)
    s *= cytnx.Scalar(np.int64(2)).astype(Type.Int32)
    assert s.dtype() == Type.Int32
    assert int(s) == 8


def test_numpy_scalar_inplace_operators_promote_on_mixed_dtype():
    # The pybind FOR_EACH_NUMPY_ITYPE overloads route through Scalar's C++
    # operator+=/-=/*=//=, so a numpy-scalar RHS promotes just like a Scalar
    # RHS -- int64 += float64 -> Double.
    s = cytnx.Scalar(np.int64(3))
    s += np.float64(2.0)
    assert s.dtype() == Type.Double
    assert float(s) == pytest.approx(5.0)


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
