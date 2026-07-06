"""Tests for Task 2 of the binding-hygiene plan: killing the `c__*`/`c_*`
shadow API (issues #779, #336).

In-place dunders and `_`-suffixed methods used to be bound under shadow names
(`c__iadd__`, `cConj_`, `c_set_name`, ...) with the real Python name supplied by
a monkey-patch wrapper in `cytnx/*_conti.py` that called the shadow binding and
then `return self`. Because the *Python* wrapper always returns the same
`self` object it was called with, object identity (`r is t`) was already
guaranteed at baseline for every one of these methods -- the wrapper papered
over the fact that the underlying C++ binding does NOT return the caller's
PyObject (pybind wraps the returned C++ reference in a *new* Python object).
That baseline behavior is recorded honestly below; nothing here is "red" in
the sense of raising or asserting false at baseline. The real defect being
fixed is invisible at the identity level and only shows up as: (1) a wasted
Python object allocation/copy on every in-place op, (2) the shadow names
cluttering `dir()`/`help()`/stubs, and (3) two names doing the same thing.
This test file's job is to pin down the *public* contract (identity +
chaining + correctness) so the conversion can be verified not to regress it,
and to assert that the shadow names are gone post-conversion.
"""

import cytnx
from cytnx import Type


def test_iadd_preserves_identity():
    t = cytnx.zeros([4])
    tid = id(t)
    t += 1.0
    assert id(t) == tid
    assert t[0].item() == 1.0


def test_inplace_named_methods_chain():
    t = cytnx.ones([2, 2])
    r = t.Conj_().Abs_()
    assert r is t


def test_unitensor_inplace_chain():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r = ut.set_name_("x").relabel_(["a", "b"])
    assert r is ut
    assert ut.name() == "x"
    assert ut.labels() == ["a", "b"]


def test_storage_pylist_roundtrip():
    s = cytnx.Storage(3, Type.Double)
    s.fill(1.5)
    assert s.pylist() == [1.5, 1.5, 1.5]


def test_bond_group_duplicates_tuple():
    b = cytnx.Bond(cytnx.BD_KET, [[0], [0], [1]], [1, 1, 2], [cytnx.Symmetry.U1()])
    nb, mapper = b.group_duplicates()
    assert isinstance(mapper, list)
    assert mapper == [0, 0, 1]


def test_bond_redirect__chains():
    b = cytnx.Bond(2, cytnx.BD_KET)
    bid = id(b)
    r = b.redirect_()
    assert r is b
    assert id(r) == bid
    assert b.type() == cytnx.BD_BRA


def test_shadow_bindings_are_gone():
    """The c__*/c_* shadow names must no longer be reachable on any instance;
    only the real dunder/method names remain."""
    t = cytnx.zeros([2])
    for name in (
        "c__iadd__", "c__isub__", "c__imul__", "c__itruediv__", "c__ifloordiv__",
        "c__ipow__", "c__imatmul__", "cConj_", "cExp_", "cInvM_", "cInv_", "cAbs_", "cPow_",
    ):
        assert not hasattr(t, name), f"Tensor.{name} shadow binding still present"

    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    for name in (
        "cConj_", "cTrace_", "cTranspose_", "cnormalize_", "cDagger_", "ctag",
        "c__ipow__", "cPow_", "cInv_", "ctruncate_", "c_set_name", "c_set_label",
        "c_set_labels", "c_relabel_", "c_relabels_", "c_set_rowrank_", "cfrom",
    ):
        assert not hasattr(ut, name), f"UniTensor.{name} shadow binding still present"

    s = cytnx.Storage(3, Type.Double)
    for name in (
        "c_pylist_double", "c_pylist_complex128", "c_pylist_float", "c_pylist_complex64",
        "c_pylist_uint64", "c_pylist_int64", "c_pylist_uint32", "c_pylist_int32",
        "c_pylist_uint16", "c_pylist_int16", "c_pylist_bool",
    ):
        assert not hasattr(s, name), f"Storage.{name} shadow binding still present"

    b = cytnx.Bond(2, cytnx.BD_KET)
    for name in ("c_redirect_", "c_getDegeneracy_refarg", "c_group_duplicates_refarg"):
        assert not hasattr(b, name), f"Bond.{name} shadow binding still present"
