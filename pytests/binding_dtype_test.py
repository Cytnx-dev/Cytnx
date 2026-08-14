"""Tests for Task 3 of the binding-hygiene plan: collapse per-dtype operator
overloads, fix `__ne__`/`__bool__`, close numpy gaps (issues #928/#916/#692).

Background: every Tensor arithmetic dunder (`__add__`, `__iadd__`, ...) used
to be bound once per C++ scalar type (12 native cytnx_* types) *and* once per
numpy scalar type (11 more), even though many of those overloads collapse to
the same Python-visible signature. This task collapses each operator down to
the "keep-set" (Tensor/UniTensor, cytnx_double, cytnx_complex128, py::int_,
numpy_scalar<float>/<complex64>, numpy_scalar<int64/uint64/int32/uint32/
int16/uint16/bool>) and adds the missing `__ne__`/`__bool__` semantics.

Root cause found empirically for the numpy integer-dtype-preservation bug
(recorded here because it explains several "unexpectedly red" baselines
below): pybind11's plain arithmetic type_caster (used for cytnx_int64,
cytnx_uint64, ... and, transitively, for anything that looks like a Python
int) accepts ANY object satisfying the `__index__` protocol even in the
*no-convert* resolution pass (pybind11 3.x cast.h, arithmetic type_caster's
no-convert `__index__` acceptance via `PyIndex_Check`). Every numpy integer
scalar
(`np.int32`, `np.uint64`, `np.int16`, ...) implements `__index__`, so a
`cytnx_int64`-typed overload registered *before* the corresponding
`py::numpy_scalar<int32_t>` overload greedily wins the no-convert pass and
the numpy_scalar overload is never reached -- silently downcasting/upcasting
every numpy integer scalar to Int64 (or whatever fixed-width integral
overload happens to be registered first), regardless of its actual numpy
dtype. `numpy_scalar<float>`/`numpy_scalar<complex64>` do NOT suffer this
because the floating-point branch of the same caster only accepts
`PyFloat_Check` in the no-convert pass, and `np.float32`/`np.complex64`
are not `PyFloat_Check`. This is exactly why the plan mandates registering
numpy_scalar overloads BEFORE py::int_/cytnx_double/cytnx_complex128 in each
operator group: it is not just a stub-quality nicety, it is required for the
integer numpy_scalar overloads to be reachable at all.

Red/green notes (recorded honestly against the branch-tip build, i.e. after
T2's direct-binding in-place ops but before this task's overload-collapse
changes):
  - test_numpy_float32_preserves_dtype: `t + np.float32` and `t += np.float32`
    PASS at baseline (numpy_scalar<float> already reachable because the plain
    double/float casters reject non-PyFloat_Check objects in the no-convert
    pass). `np.float32(1.0) + t` (numpy scalar on the LEFT) is a separate,
    pre-existing, out-of-scope bug: Tensor defines `__iter__`, so numpy's
    `__radd__` machinery tries to treat the Tensor as an array-like and
    iterate it instead of calling `Tensor.__radd__`, raising
    `TypeError: 'TensorIterator' object is not iterable`. This reproduces
    identically before and after this task's changes (verified) and is
    already called out in the existing source comments ("does not work
    because an iterator for the Tensor is defined") and in issue #692's
    thread ("an operation between a numpy object and a UniTensor object does
    not work if the numpy object is on the left"). Fixing Tensor/numpy
    __iter__ interop is out of scope for an operator-overload-collapse task
    and is NOT attempted here; only the already-working forward direction is
    asserted.
  - test_numpy_complex64_preserves_dtype: same shape/reasoning as float32;
    PASSES at baseline for the forward direction, reverse direction not
    asserted for the same __iter__ reason.
  - test_numpy_int32_preserves_dtype / *_uint32_* / *_int16_* / *_uint16_* /
    *_uint64_preserves_dtype: FAIL at baseline -- see the __index__ root
    cause above. `t + np.int32(1)` currently returns Int64, not Int32
    (verified: dtype() == 5 (Int64) instead of 7 (Int32)); same for
    uint32/int16/uint16, and np.uint64(1) also collapses to Int64 (5)
    instead of staying Uint64 (6). This is genuine, in-scope red: fixed by
    re-registering the numpy_scalar<T> overloads ahead of the plain integral
    overloads per operator group.
  - test_python_int_works_and_large_int: PASSES at baseline in full -- `t + 1`
    dispatches through the existing cytnx_int64 overload, and the
    `> int64 max` uint64 case also already works because pybind11's
    convert-pass for the plain `cytnx_uint64` overload (registered directly,
    not through a dispatch_pyint-style selector) correctly widens through
    `PyLong_As*`/`as_unsigned` for a value beyond int64 range but within
    uint64 range. Kept as a regression guard for the collapse (a naive
    dispatch_pyint reimplementation could plausibly narrow this by mistake),
    not because it was red.
  - test_pow_accepts_int_and_numpy: PASSES at baseline -- __pow__ only accepts
    a plain `cytnx_double` exponent, but pybind11's convert-pass already
    accepts a Python `int` or `np.float32` there via `PyFloat_AsDouble`
    (`convert=true` short-circuits the exact-`PyFloat_Check` gate quoted
    above), and `linalg::Pow`'s output dtype follows the base tensor's dtype,
    not the exponent's, so this was never actually broken for these two
    inputs. Kept as a regression guard for the keep-set rewrite (the plan
    still asks for explicit py::int_/numpy_scalar<float> overloads on
    __pow__ so a numpy exponent that does NOT satisfy PyFloat_Check-or-
    convert, e.g. a value pybind rejects some other way, is not silently
    mishandled, and so the stub will show a precise signature once #915
    lands rather than relying on the implicit-conversion fallback).
  - test_setitem_numpy_scalar_preserves_dtype: FAILS at baseline, and worse
    than expected -- Tensor's __setitem__ has zero numpy_scalar overloads
    (only the 11 native cytnx_* ones, ordered complex128, complex64, double,
    float, int64, ...), and `np.float32(2.5)` does not even reach the float
    or double overload: it raises RuntimeError
    ("cannot assign complex element to real container") because it is
    accepted by the FIRST-registered `cytnx_complex128` overload. Python's
    `complex()` builtin falls back to `__float__` when `__complex__` is
    absent, so pybind11's complex type_caster's convert pass happily accepts
    any float-like object, and registration order (complex128 first) lets it
    win before double/float ever get a chance. This is a real, in-scope bug
    (not just a missing-coverage gap) fixed by applying the same
    keep-set-with-numpy-scalars-first ordering used for the arithmetic
    operators to __setitem__.
  - test_ne_is_not_silently_wrong / test_ne_elementwise_values: FAIL at
    baseline -- __ne__ is unbound, so Python's default falls back to
    `not (a == b)`; cytnx's __eq__ returns an elementwise Bool Tensor, and
    `not Tensor` calls __bool__/__len__, collapsing to a single bare `False`
    for any non-empty tensor -- a silently wrong scalar instead of an
    elementwise comparison. (This is exactly the failure mode #928/#916
    background describes; verified: `a != b` under Python's default returns
    the Python constant `False`, not a Tensor, for two non-empty operands.)
  - test_bool_multielement_raises / test_bool_uninitialized_raises: FAIL at
    baseline -- __bool__ is unbound, so truthiness falls through to __len__
    (`if tensor:` just checks shape()[0] != 0), never raising ValueError for
    a multi-element tensor, and raises RuntimeError (not ValueError) for an
    uninitialized (Void-dtype) tensor because __len__ itself cytnx_error_msg's
    out.
  - test_imatmul_preserves_identity: PASSES at baseline in this worktree
    (T2's real, non-shadow `__imatmul__` binding already runs and does
    `self.cast<Tensor&>() = Dot(...)` on the SAME py::object, so identity is
    preserved here). Recorded per the plan's requirement to pin this down
    explicitly: on unmodified master (before T2), `@=` only appeared to work
    because the old python wrapper method was misspelled `__imatmul`
    (missing trailing underscore) instead of `__imatmul__`, so the C++
    `__imatmul__` binding was unreachable dead code and Python's `@=` silently
    fell back to `t = t.__matmul__(x)` (rebinding the *name* in the caller's
    scope, not mutating the object). This test guards against a future
    regression (e.g. an accidental by-value self-cast) silently reintroducing
    that non-in-place fallback behavior.

Scope note: the plan's Task 3 inventory only lists Tensor's __setitem__ for
the numpy-scalar-keep-set treatment ("mirror UniTensor's existing coverage").
Recon here found UniTensor's own __setitem__/set_elem has the IDENTICAL
registration-order bug (np.float32 hits the first-registered
cytnx_complex128 overload) -- and worse than Tensor's, because setting a
real scalar through the complex128 overload does not merely mis-preserve
dtype, it raises a hard RuntimeError ("Cannot set Complex Double to
Float") from the underlying storage. Since "UniTensor's existing coverage"
was itself broken, mirroring it verbatim would have propagated the bug
instead of fixing it, so UniTensor's __setitem__/set_elem (both the
vector-locator and single-int-locator, i.e. diagonal-UniTensor, overload
families) were given the same keep-set-with-numpy-scalars-first fix as
Tensor's. This is a modest, consistency-motivated scope extension beyond
the plan's literal inventory, called out explicitly in the commit body.
"""

import numpy as np
import pytest

import cytnx
from cytnx import Type


def test_numpy_float32_preserves_dtype():
    t = cytnx.zeros([2], dtype=Type.Float)
    assert (t + np.float32(1.0)).dtype() == Type.Float
    t += np.float32(1.0)
    assert t.dtype() == Type.Float


def test_numpy_complex64_preserves_dtype():
    t = cytnx.zeros([2], dtype=Type.ComplexFloat)
    assert (t + np.complex64(1.0)).dtype() == Type.ComplexFloat
    t += np.complex64(1.0)
    assert t.dtype() == Type.ComplexFloat


def test_numpy_int32_preserves_dtype():
    t = cytnx.zeros([2], dtype=Type.Int32)
    assert (t + np.int32(1)).dtype() == Type.Int32
    t += np.int32(1)
    assert t.dtype() == Type.Int32


def test_numpy_uint32_preserves_dtype():
    t = cytnx.zeros([2], dtype=Type.Uint32)
    assert (t + np.uint32(1)).dtype() == Type.Uint32
    t += np.uint32(1)
    assert t.dtype() == Type.Uint32


def test_numpy_int16_preserves_dtype():
    t = cytnx.zeros([2], dtype=Type.Int16)
    assert (t + np.int16(1)).dtype() == Type.Int16


def test_numpy_uint16_preserves_dtype():
    t = cytnx.zeros([2], dtype=Type.Uint16)
    assert (t + np.uint16(1)).dtype() == Type.Uint16


def test_numpy_uint64_preserves_dtype():
    t = cytnx.zeros([2], dtype=Type.Uint64)
    assert (t + np.uint64(1)).dtype() == Type.Uint64


def test_python_int_works_and_large_int():
    t = cytnx.zeros([2], dtype=Type.Int64)
    assert (t + 1).dtype() == Type.Int64
    t2 = cytnx.zeros([2], dtype=Type.Uint64)
    _ = t2 + (2**63 + 1)  # > int64 max: must dispatch to uint64, not raise


def test_pow_accepts_int_and_numpy():
    t = cytnx.ones([2])
    assert (t**2)[0].item() == 1.0
    assert (t ** np.float32(2.0))[0].item() == 1.0


def test_setitem_numpy_scalar_preserves_dtype():
    t = cytnx.zeros([3], dtype=Type.Float)
    t[0] = np.float32(2.5)
    assert t.dtype() == Type.Float


def test_ne_is_not_silently_wrong():
    a = cytnx.ones([3])
    b = cytnx.zeros([3])
    r = a != b
    # either elementwise Tensor of ones, or TypeError -- never a bare False
    assert not (r is False)


def test_ne_elementwise_values():
    a = cytnx.ones([3])
    b = cytnx.zeros([3])
    r = a != b
    assert isinstance(r, cytnx.Tensor)
    assert r.dtype() == Type.Bool
    assert bool(r[0].item()) is True

    c = cytnx.ones([3])
    r2 = a != c
    assert bool(r2[0].item()) is False


def test_ne_scalar_operand():
    # The __ne__ keep-set mirrors __eq__: native Python int/float/complex and
    # numpy scalars all reach it. cytnx.Scalar is no longer exposed on the
    # Python surface (issue #1045); a plain Python int flows through the
    # py::int_ overload (dispatch_pyint), so an integer operand beyond 2**53
    # keeps full precision instead of degrading through a float cast.
    t = cytnx.ones([3])
    r = t != 2.0
    assert isinstance(r, cytnx.Tensor)
    assert r.dtype() == Type.Bool
    assert bool(r[0].item()) is True

    r2 = t != 1.0
    assert bool(r2[0].item()) is False

    # Large integer operand: routed as an exact int64 (not rounded to a double).
    r3 = t != (2**60 + 1)
    assert bool(r3[0].item()) is True


def test_bool_multielement_raises():
    t = cytnx.ones([3])
    with pytest.raises(ValueError):
        bool(t)
    assert bool(cytnx.ones([1]))
    assert not bool(cytnx.zeros([1]))


def test_bool_uninitialized_raises():
    t = cytnx.Tensor()
    with pytest.raises(ValueError):
        bool(t)


def test_imatmul_preserves_identity():
    a = cytnx.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]]))
    ref = a
    b = cytnx.from_numpy(np.array([[2.0, 0.0], [0.0, 2.0]]))
    a @= b
    assert a is ref
    assert a[0, 0].item() == 2.0
    assert a[1, 1].item() == 2.0
    assert a[0, 1].item() == 0.0


# UniTensor coverage for the same keep-set collapse (scope extension: recon
# found UniTensor's __setitem__/set_elem shared Tensor's setitem bug, see the
# module docstring's "Scope note").


def test_unitensor_numpy_int32_preserves_dtype():
    ut = cytnx.UniTensor(cytnx.zeros([2, 2], dtype=Type.Int32))
    assert (ut + np.int32(1)).dtype() == Type.Int32
    ut += np.int32(1)
    assert ut.dtype() == Type.Int32


def test_unitensor_pow_accepts_int_and_numpy():
    ut = cytnx.UniTensor(cytnx.ones([1]))
    assert (ut**2).item() == 1.0
    assert (ut ** np.float32(2.0)).item() == 1.0


def test_unitensor_set_elem_numpy_scalar_preserves_dtype():
    ut = cytnx.UniTensor(cytnx.zeros([3], dtype=Type.Float))
    ut.set_elem([0], np.float32(2.5))
    assert ut.dtype() == Type.Float


def test_unitensor_setitem_numpy_scalar_preserves_dtype():
    ut = cytnx.UniTensor(cytnx.zeros([3], dtype=Type.Float))
    ut[[0]] = np.float32(2.5)
    assert ut.dtype() == Type.Float
