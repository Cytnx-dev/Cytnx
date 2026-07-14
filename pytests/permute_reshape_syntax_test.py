"""Tests for permute()/reshape() call-form unification (#293, ruling 4):
both the variadic form (`t.permute(1, 2, 0)`) and the list form
(`t.permute([1, 2, 0])`) are accepted on both Tensor and UniTensor;
UniTensor additionally accepts both int and string labels in each form.
"""

import cytnx
from cytnx import Type


def _tensor_1_2_0():
    # shape [2, 3, 4] -> permute(1, 2, 0) -> shape [3, 4, 2]
    return cytnx.arange(24).reshape(2, 3, 4).astype(Type.Double)


def _labeled_unitensor():
    ut = cytnx.UniTensor(cytnx.arange(24).reshape(2, 3, 4).astype(Type.Double))
    assert ut.labels() == ["0", "1", "2"]
    return ut


# ---------------------------------------------------------------------------
# Tensor.permute / Tensor.permute_
# ---------------------------------------------------------------------------


def test_tensor_permute_variadic():
    t = _tensor_1_2_0()
    assert t.permute(1, 2, 0).shape() == [3, 4, 2]


def test_tensor_permute_list():
    t = _tensor_1_2_0()
    assert t.permute([1, 2, 0]).shape() == [3, 4, 2]


def test_tensor_permute_variadic_and_list_agree():
    t = _tensor_1_2_0()
    a = t.permute(1, 2, 0)
    b = t.permute([1, 2, 0])
    assert a.shape() == b.shape()
    assert (a - b).norm() == 0.0


def test_tensor_permute__variadic_inplace_returns_self():
    t = _tensor_1_2_0()
    r = t.permute_(1, 2, 0)
    assert r is t
    assert t.shape() == [3, 4, 2]


def test_tensor_permute__list_inplace_returns_self():
    t = _tensor_1_2_0()
    r = t.permute_([1, 2, 0])
    assert r is t
    assert t.shape() == [3, 4, 2]


# ---------------------------------------------------------------------------
# Tensor.reshape / Tensor.reshape_
# ---------------------------------------------------------------------------


def test_tensor_reshape_variadic():
    t = cytnx.arange(24)
    assert t.reshape(2, 3, 4).shape() == [2, 3, 4]


def test_tensor_reshape_list():
    t = cytnx.arange(24)
    assert t.reshape([2, 3, 4]).shape() == [2, 3, 4]


def test_tensor_reshape__variadic_inplace_returns_self():
    t = cytnx.arange(24)
    r = t.reshape_(2, 3, 4)
    assert r is t
    assert t.shape() == [2, 3, 4]


def test_tensor_reshape__list_inplace_returns_self():
    t = cytnx.arange(24)
    r = t.reshape_([2, 3, 4])
    assert r is t
    assert t.shape() == [2, 3, 4]


# ---------------------------------------------------------------------------
# UniTensor.permute / UniTensor.permute_  (int list / str list / int variadic /
# str variadic -- all four call forms)
# ---------------------------------------------------------------------------


def test_unitensor_permute_list_int():
    ut = _labeled_unitensor()
    assert ut.permute([1, 2, 0]).shape() == [3, 4, 2]


def test_unitensor_permute_list_str():
    ut = _labeled_unitensor()
    assert ut.permute(["1", "2", "0"]).shape() == [3, 4, 2]


def test_unitensor_permute_variadic_int():
    ut = _labeled_unitensor()
    assert ut.permute(1, 2, 0).shape() == [3, 4, 2]


def test_unitensor_permute_variadic_str():
    ut = _labeled_unitensor()
    assert ut.permute("1", "2", "0").shape() == [3, 4, 2]


def test_unitensor_permute_all_four_forms_agree():
    labels = ["0", "1", "2"]
    shapes = []
    for form in (
        lambda ut: ut.permute([1, 2, 0]),
        lambda ut: ut.permute(["1", "2", "0"]),
        lambda ut: ut.permute(1, 2, 0),
        lambda ut: ut.permute("1", "2", "0"),
    ):
        ut = _labeled_unitensor()
        shapes.append(form(ut).shape())
    assert all(s == [3, 4, 2] for s in shapes)


def test_unitensor_permute_rowrank_kwarg_works_in_all_forms():
    ut1 = _labeled_unitensor()
    assert ut1.permute([1, 2, 0], rowrank=1).rowrank() == 1

    ut2 = _labeled_unitensor()
    assert ut2.permute(1, 2, 0, rowrank=1).rowrank() == 1

    ut3 = _labeled_unitensor()
    assert ut3.permute("1", "2", "0", rowrank=1).rowrank() == 1


def test_unitensor_permute__variadic_int_inplace_returns_self():
    ut = _labeled_unitensor()
    r = ut.permute_(1, 2, 0)
    assert r is ut
    assert ut.shape() == [3, 4, 2]


def test_unitensor_permute__variadic_str_inplace_returns_self():
    ut = _labeled_unitensor()
    r = ut.permute_("1", "2", "0")
    assert r is ut
    assert ut.shape() == [3, 4, 2]


def test_unitensor_permute__list_str_inplace_returns_self():
    ut = _labeled_unitensor()
    r = ut.permute_(["1", "2", "0"])
    assert r is ut
    assert ut.shape() == [3, 4, 2]


# ---------------------------------------------------------------------------
# UniTensor.reshape / UniTensor.reshape_
# ---------------------------------------------------------------------------


def test_unitensor_reshape_variadic():
    ut = cytnx.UniTensor(cytnx.arange(24).astype(Type.Double))
    assert ut.reshape(2, 3, 4).shape() == [2, 3, 4]


def test_unitensor_reshape_list():
    ut = cytnx.UniTensor(cytnx.arange(24).astype(Type.Double))
    assert ut.reshape([2, 3, 4]).shape() == [2, 3, 4]


def test_unitensor_reshape_rowrank_kwarg_works_in_both_forms():
    ut1 = cytnx.UniTensor(cytnx.arange(24).astype(Type.Double))
    assert ut1.reshape(2, 3, 4, rowrank=1).rowrank() == 1

    ut2 = cytnx.UniTensor(cytnx.arange(24).astype(Type.Double))
    assert ut2.reshape([2, 3, 4], rowrank=1).rowrank() == 1


def test_unitensor_reshape__variadic_inplace_returns_self():
    ut = cytnx.UniTensor(cytnx.arange(24).astype(Type.Double))
    r = ut.reshape_(2, 3, 4)
    assert r is ut
    assert ut.shape() == [2, 3, 4]


def test_unitensor_reshape__list_inplace_returns_self():
    ut = cytnx.UniTensor(cytnx.arange(24).astype(Type.Double))
    r = ut.reshape_([2, 3, 4])
    assert r is ut
    assert ut.shape() == [2, 3, 4]
