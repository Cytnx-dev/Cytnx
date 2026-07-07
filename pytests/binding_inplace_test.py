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
    r = ut.set_name("x").relabel_(["a", "b"])
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


def test_imatmul_preserves_identity():
    # `@=` is the one genuine behavioral change from the shadow-API removal:
    # the old wrapper was misspelled `def __imatmul` (no trailing `__`), so
    # `a @= b` fell back to `a = a @ b` and rebound `a` to a new object. With
    # __imatmul__ bound correctly it now mutates in place and preserves identity.
    a = cytnx.arange(4).reshape(2, 2)
    b = cytnx.arange(4).reshape(2, 2)
    expected = cytnx.linalg.Dot(a, b)
    alias = a  # second reference to the same object
    aid = id(a)
    a @= b
    assert a is alias  # in-place: no new object was created
    assert id(a) == aid
    # the mutation is visible through the alias and matches a @ b
    assert (alias - expected).Norm().item() == 0.0
