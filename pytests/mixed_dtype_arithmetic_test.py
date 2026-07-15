"""Regression tests for the #941 typed-storage CPU arithmetic conversion.

Two python-visible behavior fixes are pinned here (branch
refactor/typed-storage-foundation, commit B):

1. In-place mixed-dtype arithmetic (`t -= other`, `t += other`, ...) now
   promotes the lhs tensor's dtype exactly like the equivalent out-of-place
   operation, instead of silently truncating results into the original
   (narrower) storage type. Previously `Int16 -= Double` wrote truncated
   values into Int16 storage on CPU.

2. Division follows Python true-division semantics: integer/integer division
   produces a floating (Double) result instead of truncating (#941's "True
   division" section), for both `/` and `/=`.
"""

import numpy as np
import pytest

import cytnx


def test_inplace_isub_promotes_int_to_double():
    lt = cytnx.zeros([3], dtype=cytnx.Type.Int16)
    lt += 10  # fill with ints: [10, 10, 10]
    rt = cytnx.zeros([3], dtype=cytnx.Type.Double)
    rt += 0.5
    lt -= rt
    assert lt.dtype() == cytnx.Type.Double
    np.testing.assert_allclose(lt.numpy(), [9.5, 9.5, 9.5])


def test_inplace_iadd_promotes_int_to_double():
    lt = cytnx.zeros([2], dtype=cytnx.Type.Int32)
    lt += 1
    rt = cytnx.zeros([2], dtype=cytnx.Type.Double)
    rt += 1.25
    lt += rt
    assert lt.dtype() == cytnx.Type.Double
    np.testing.assert_allclose(lt.numpy(), [2.25, 2.25])


def test_true_division_int_over_int_produces_double():
    lt = cytnx.zeros([1], dtype=cytnx.Type.Int64)
    lt += 3
    rt = cytnx.zeros([1], dtype=cytnx.Type.Int64)
    rt += 2
    out = lt / rt
    assert out.dtype() == cytnx.Type.Double
    assert out.numpy()[0] == 1.5


def test_inplace_true_division_promotes_and_divides():
    # The exact family from #941's "Tests Required" section.
    for src_dtype in (cytnx.Type.Int64, cytnx.Type.Int32, cytnx.Type.Int16,
                      cytnx.Type.Uint16):
        lt = cytnx.zeros([1], dtype=src_dtype)
        lt += 3
        rt = cytnx.zeros([1], dtype=cytnx.Type.Double)
        rt += 2.0
        lt /= rt
        assert lt.dtype() == cytnx.Type.Double, cytnx.Type.getname(src_dtype)
        assert lt.numpy()[0] == 1.5, cytnx.Type.getname(src_dtype)


def test_inplace_rank0_tensor_rhs_promotes_not_weak_scalar():
    # A genuine rank-0 tensor RHS is a first-class tensor value, NOT a python
    # weak scalar: `int_tensor += rank0_double_tensor` must promote the lhs to
    # Double (like a shape-[1] tensor RHS and the out-of-place op), rather than
    # truncating into the narrower lhs storage. Only a python-scalar RHS
    # (scalar_as_rank0_tensor) preserves the lhs dtype. Regression for the
    # weak-scalar detection keying on rank instead of an explicit caller flag.
    rt0 = cytnx.zeros([], dtype=cytnx.Type.Double)
    rt0 += 2.5
    assert rt0.rank() == 0

    # += rank-0 Double tensor -> promote to Double (contrast a python scalar).
    lt = cytnx.zeros([3], dtype=cytnx.Type.Int32)
    lt += 2  # [2, 2, 2]
    lt += rt0
    assert lt.dtype() == cytnx.Type.Double
    np.testing.assert_allclose(lt.numpy(), [4.5, 4.5, 4.5])

    # A python scalar RHS still preserves the lhs (weak scalar).
    lt_scalar = cytnx.zeros([3], dtype=cytnx.Type.Int32)
    lt_scalar += 2
    lt_scalar += 2.5  # weak scalar -> stays Int32, truncates to 4
    assert lt_scalar.dtype() == cytnx.Type.Int32
    np.testing.assert_allclose(lt_scalar.numpy(), [4, 4, 4])

    # -= rank-0 tensor with a negative value promotes and subtracts.
    lt_sub = cytnx.zeros([2], dtype=cytnx.Type.Int16)
    lt_sub += 1  # [1, 1]
    rt_neg = cytnx.zeros([], dtype=cytnx.Type.Double)
    rt_neg += -0.5
    lt_sub -= rt_neg
    assert lt_sub.dtype() == cytnx.Type.Double
    np.testing.assert_allclose(lt_sub.numpy(), [1.5, 1.5])

    # /= rank-0 integer tensor still yields true division (Double), matching the
    # out-of-place Int / Int -> Double rule for a genuine tensor RHS.
    lt_div = cytnx.zeros([1], dtype=cytnx.Type.Int64)
    lt_div += 3
    rt_div = cytnx.zeros([], dtype=cytnx.Type.Int64)
    rt_div += 2
    lt_div /= rt_div
    assert lt_div.dtype() == cytnx.Type.Double
    assert lt_div.numpy()[0] == 1.5


def test_out_of_place_add_keeps_promotion():
    # Sanity: the already-correct out-of-place promotion path is unchanged.
    lt = cytnx.zeros([2], dtype=cytnx.Type.Int32)
    lt += 1
    rt = cytnx.zeros([2], dtype=cytnx.Type.Double)
    rt += 2.5
    out = lt + rt
    assert out.dtype() == cytnx.Type.Double
    np.testing.assert_allclose(out.numpy(), [3.5, 3.5])


def test_tensor_floordiv_unbound():
    # #941: "// makes no sense at all for any tensor object that Cytnx cares
    # about -- leave it unbound." Previously __floordiv__ aliased Div, which
    # after the true-division change would have made `t // x` silently
    # perform TRUE division -- strictly worse than the old truncation.
    t = cytnx.zeros([2], dtype=cytnx.Type.Int64)
    t += 3
    with pytest.raises(TypeError):
        t // 2
    with pytest.raises(TypeError):
        2 // t
    with pytest.raises(TypeError):
        t //= 2


def test_unitensor_floordiv_scalar_kept_tensor_raises():
    # #1049 (supersedes #1015's UniTensor unbind): scalar floordiv is KEPT on
    # UniTensor -- `ut // s` routes to Div, i.e. true division, mirroring scalar
    # `/` -- while UniTensor // UniTensor raises (the removed Hadamard quotient).
    # (Plain-Tensor // stays fully unbound per #941; see
    # test_tensor_floordiv_unbound.)
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    ut2 = cytnx.UniTensor(cytnx.ones([2, 2]))
    # scalar floordiv is kept (mirrors scalar `/`):
    _ = ut // 2
    _ = 2 // ut
    ut //= 2
    # UniTensor // UniTensor raises:
    with pytest.raises(TypeError):
        ut2 // ut2
