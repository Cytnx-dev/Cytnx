"""Tests for linalg.ExpH / linalg.ExpM scalar-argument handling.

ExpH and ExpM expose one binding per scalar dtype (the numpy_scalar variants, a
Python-int overload dispatching to the int64/uint64 kernels, and the
double/complex128 kernels). These tests pin that those bindings accept Python
and numpy scalars of every dtype and stay numerically correct, matching a NumPy
eigendecomposition reference.

ComplexFloat (complex64) scalars are intentionally routed through the
ComplexDouble kernel because linalg.Exp is broken for complex64 tensors
(cytnx-dev/cytnx#914); the complex64 cases below therefore assert the correct
(complex128-precision) result rather than single precision.
"""

import numpy as np
import pytest

import cytnx

# A real symmetric matrix, so ExpH (and ExpM, for a symmetric input) equal the
# eigendecomposition reference exp(a*H) = V @ diag(exp(a*w)) @ V^H.
_H = np.array([[0.0, 3.0], [3.0, 6.0]])
_W, _V = np.linalg.eigh(_H)


def _reference(a, b=0):
    return _V @ np.diag(np.exp(a * _W + b)) @ _V.conj().T


def _unitensor():
    t = cytnx.from_numpy(_H.copy())
    return cytnx.UniTensor(t, rowrank=1)


# (scalar value, complex-valued?) for both Python and numpy scalar types.
_REAL_SCALARS = [2, 2.0, np.float64(2.0), np.float32(2.0), np.int64(2)]
_COMPLEX_SCALARS = [2 + 1j, np.complex128(2 + 1j), np.complex64(2 + 1j)]


@pytest.mark.parametrize("func", ["ExpH", "ExpM"])
@pytest.mark.parametrize("a", _REAL_SCALARS + _COMPLEX_SCALARS)
def test_exp_accepts_python_and_numpy_scalars(func, a):
    """The per-dtype overloads accept every Python/numpy scalar dtype."""
    out = getattr(cytnx.linalg, func)(_unitensor(), a)
    result = np.array(out.get_block().numpy()).reshape(2, 2)
    np.testing.assert_allclose(result, _reference(complex(a)), rtol=1e-4)


@pytest.mark.parametrize("func", ["ExpH", "ExpM"])
@pytest.mark.parametrize("a, b", [(1, 0.5), (2.0, 1 + 0j), (2, 1), (1.0, np.float64(0.5))])
def test_exp_mixed_scalar_dtypes_promote(func, a, b):
    """A mixed (a, b) call resolves to a common-dtype overload accepting both."""
    out = getattr(cytnx.linalg, func)(_unitensor(), a, b)
    result = np.array(out.get_block().numpy()).reshape(2, 2)
    np.testing.assert_allclose(result, _reference(complex(a), complex(b)), rtol=1e-4)


@pytest.mark.parametrize("func", ["ExpH", "ExpM"])
def test_exp_complex64_matches_complex128(func):
    """complex64 scalars are routed through the complex128 kernel (issue #914)."""
    fn = getattr(cytnx.linalg, func)
    r64 = np.array(fn(_unitensor(), np.complex64(2 + 1j)).get_block().numpy()).ravel()
    r128 = np.array(fn(_unitensor(), np.complex128(2 + 1j)).get_block().numpy()).ravel()
    np.testing.assert_allclose(r64, r128, rtol=1e-6)
