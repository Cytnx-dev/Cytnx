import math

import pytest

import cytnx
from cytnx import Type


def test_tensor_norm_returns_float_and_matches_deprecated_norm():
    t = cytnx.arange(4).astype(Type.Double)
    n = t.norm()
    assert isinstance(n, float)
    expected = math.sqrt(sum(i * i for i in range(4)))
    assert n == pytest.approx(expected)

    with pytest.warns(DeprecationWarning):
        legacy = t.Norm()
    assert n == pytest.approx(legacy.item())


def test_tensor_norm_complex_input_returns_real_double():
    t = cytnx.ones([3]).astype(Type.ComplexDouble)
    n = t.norm()
    assert isinstance(n, float)
    assert n == pytest.approx(math.sqrt(3.0))


def test_unitensor_norm_returns_float_and_matches_deprecated_norm():
    ut = cytnx.UniTensor(cytnx.arange(4).astype(Type.Double).reshape(2, 2))
    n = ut.norm()
    assert isinstance(n, float)

    with pytest.warns(DeprecationWarning):
        legacy = ut.Norm()
    assert n == pytest.approx(legacy.item())


def test_tensor_Norm_emits_deprecation_warning():
    t = cytnx.ones([3])
    with pytest.warns(DeprecationWarning, match="norm"):
        t.Norm()


def test_unitensor_Norm_emits_deprecation_warning():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    with pytest.warns(DeprecationWarning, match="norm"):
        ut.Norm()


def test_linalg_norm_free_function_returns_float():
    t = cytnx.arange(4).astype(Type.Double)
    n = cytnx.linalg.norm(t)
    assert isinstance(n, float)
    assert n == pytest.approx(t.norm())

    ut = cytnx.UniTensor(t.reshape(2, 2))
    nu = cytnx.linalg.norm(ut)
    assert isinstance(nu, float)
    assert nu == pytest.approx(ut.norm())


def test_linalg_Norm_free_function_emits_deprecation_warning():
    t = cytnx.ones([3])
    with pytest.warns(DeprecationWarning, match="norm"):
        result = cytnx.linalg.Norm(t)
    assert result.item() == pytest.approx(t.norm())

    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    with pytest.warns(DeprecationWarning, match="norm"):
        result_ut = cytnx.linalg.Norm(ut)
    assert result_ut.item() == pytest.approx(ut.norm())
