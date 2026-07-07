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
    """Construct a cytnx.Scalar with an explicit dtype via a numpy scalar:
    each numpy scalar type maps to its own constructor overload, while the
    C++-only (value, dtype) two-argument constructor is not bound."""
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
    s = cytnx.Scalar(np.uint64(3))
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
# its __bool__/__int__/__float__/__complex__ operators into self.at<T>(idx)
# rather than returning a Scalar from __getitem__. The Bool row needs the
# __bool__ binding: pybind's bool caster only accepts objects whose type
# fills the nb_bool slot.
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
        (Type.Bool, np.bool_),
    ],
)
def test_storage_setitem_getitem_scalar_roundtrip(dtype, np_dtype):
    storage = cytnx.Storage(4, dtype)
    value = np_dtype(True) if dtype == Type.Bool else np_dtype(3)
    storage[1] = cytnx.Scalar(value)
    got = storage[1]
    if dtype == Type.Bool:
        assert got is True
    elif isinstance(got, (float, complex)):
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
# numpy scalar constructor paths (all 11 dtypes + complex): every numpy
# scalar type resolves to its own constructor overload. Before the keep-set
# reorder of scalar_py.cpp's __init__ overloads (see "KEEP-SET ORDERING" in
# pybind/pyint_dispatch.hpp), the plain integral constructors were registered
# ahead of the numpy_scalar ones, so every in-range numpy integer scalar was
# consumed via __index__ by the first plain integral overload and came out
# dtype Int64.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "np_dtype,dtype",
    [
        (np.complex128, Type.ComplexDouble),
        (np.complex64, Type.ComplexFloat),
        (np.float64, Type.Double),
        (np.float32, Type.Float),
        (np.int64, Type.Int64),
        (np.uint64, Type.Uint64),
        (np.int32, Type.Int32),
        (np.uint32, Type.Uint32),
        (np.int16, Type.Int16),
        (np.uint16, Type.Uint16),
        (np.bool_, Type.Bool),
    ],
)
def test_numpy_scalar_constructor_paths(np_dtype, dtype):
    val = np_dtype(True) if dtype == Type.Bool else np_dtype(5)
    s = cytnx.Scalar(val)
    assert s.dtype() == dtype


def test_numpy_uint64_above_int64_max_preserves_value():
    big = np.uint64(2**64 - 1)
    s = cytnx.Scalar(big)
    assert s.dtype() == Type.Uint64
    assert float(s) == pytest.approx(float(big))


# ---------------------------------------------------------------------------
# plain Python scalar constructor keep-set: int dispatches on magnitude
# (int64 when it fits, covering all negatives; uint64 otherwise; clean error
# beyond uint64 range instead of a silent complex upcast), and bool/float/
# complex map to Bool/Double/ComplexDouble.
# ---------------------------------------------------------------------------


def test_python_int_constructor_dispatches_on_magnitude():
    assert cytnx.Scalar(5).dtype() == Type.Int64
    assert cytnx.Scalar(-5).dtype() == Type.Int64
    assert cytnx.Scalar(2**63).dtype() == Type.Uint64
    assert cytnx.Scalar(2**64 - 1).dtype() == Type.Uint64


def test_python_int_constructor_out_of_range_raises():
    with pytest.raises(RuntimeError):
        cytnx.Scalar(2**64)
    with pytest.raises(RuntimeError):
        cytnx.Scalar(-(2**63) - 1)


def test_python_bool_float_complex_constructor_dtypes():
    assert cytnx.Scalar(True).dtype() == Type.Bool
    assert cytnx.Scalar(0.5).dtype() == Type.Double
    assert cytnx.Scalar(1 + 2j).dtype() == Type.ComplexDouble


# ---------------------------------------------------------------------------
# __bool__: numpy-consistent truthiness (nonzero value -> True; for complex,
# nonzero real OR imaginary part).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "np_dtype",
    [
        np.complex128,
        np.complex64,
        np.float64,
        np.float32,
        np.int64,
        np.uint64,
        np.int32,
        np.uint32,
        np.int16,
        np.uint16,
        np.bool_,
    ],
)
def test_bool_matches_numpy_truthiness(np_dtype):
    assert bool(cytnx.Scalar(np_dtype(0))) is False
    assert bool(cytnx.Scalar(np_dtype(1))) is True


def test_bool_complex_pure_imaginary_is_true():
    # numpy: bool(np.complex128(1j)) is True -- truthiness must consider the
    # imaginary part, not just the real part.
    assert bool(cytnx.Scalar(np.complex128(1j))) is True


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
