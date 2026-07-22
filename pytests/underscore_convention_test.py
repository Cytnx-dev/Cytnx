"""Tests for the underscore convention (#335/#336/#381/#421/#422, ruling 3):

`_`-suffixed methods are canonical in-place mutators returning self; the
un-suffixed spellings that used to mutate in place are kept as deprecated
shims (DeprecationWarning) delegating to the new ones, for one release.
"""

import pytest

import cytnx


def _labeled_tensor():
    return cytnx.UniTensor(cytnx.arange(24).reshape(2, 3, 4).astype(cytnx.Type.Double))


# ---------------------------------------------------------------------------
# set_name_ / set_name
# ---------------------------------------------------------------------------


def test_set_name__is_inplace_and_returns_self():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r = ut.set_name_("foo")
    assert r is ut
    assert ut.name() == "foo"


def test_set_name_deprecated_warns_and_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    with pytest.warns(DeprecationWarning, match="set_name_"):
        r = ut.set_name("bar")
    assert r is ut
    assert ut.name() == "bar"


# ---------------------------------------------------------------------------
# set_label_ / set_label
# ---------------------------------------------------------------------------


def test_set_label__by_index_is_inplace_and_returns_self():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r = ut.set_label_(0, "x")
    assert r is ut
    assert ut.labels() == ["x", "1"]


def test_set_label__by_name_is_inplace_and_returns_self():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    r = ut.set_label_("0", "x")
    assert r is ut
    assert ut.labels() == ["x", "1"]


def test_set_label_deprecated_warns_and_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    with pytest.warns(DeprecationWarning, match="set_label_"):
        r = ut.set_label(0, "y")
    assert r is ut
    assert ut.labels() == ["y", "1"]


# ---------------------------------------------------------------------------
# tag_ / tag
# ---------------------------------------------------------------------------


def test_tag__is_inplace_and_returns_self():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    assert not ut.is_tag()
    r = ut.tag_()
    assert r is ut
    assert ut.is_tag()


def test_tag_deprecated_warns_and_still_works():
    ut = cytnx.UniTensor(cytnx.ones([2, 2]))
    with pytest.warns(DeprecationWarning, match="tag_"):
        r = ut.tag()
    assert r is ut
    assert ut.is_tag()


# ---------------------------------------------------------------------------
# combineBond_ (in-place) / combineBond (out-of-place, new) / combineBonds (deprecated)
# ---------------------------------------------------------------------------


def test_combineBond__is_inplace_and_returns_self():
    ut = _labeled_tensor()
    r = ut.combineBond_(["1", "2"])
    assert r is ut
    assert ut.shape() == [2, 12]


def test_combineBond_is_out_of_place_and_returns_new_object():
    ut = _labeled_tensor()
    out = ut.combineBond(["1", "2"])
    assert ut.shape() == [2, 3, 4]  # original unchanged
    assert out.shape() == [2, 12]  # new UniTensor with bonds combined
    assert out is not ut


def test_combineBonds_deprecated_warns_and_still_works():
    ut = _labeled_tensor()
    with pytest.warns(DeprecationWarning, match="combineBond"):
        ut.combineBonds(["1", "2"])
    assert ut.shape() == [2, 12]


def test_combineBonds_int64_form_deprecated_warns_and_still_works():
    ut = _labeled_tensor()
    with pytest.warns(DeprecationWarning, match="combineBond"):
        ut.combineBonds([1, 2])
    assert ut.shape() == [2, 12]


# ---------------------------------------------------------------------------
# convert_from_ / convert_from
# ---------------------------------------------------------------------------


def test_convert_from__is_inplace_and_returns_self():
    bi = cytnx.Bond(cytnx.BD_KET, [[0], [1]], [1, 1], [cytnx.Symmetry.U1()])
    B = cytnx.UniTensor([bi, bi.redirect()])
    D = cytnx.UniTensor(cytnx.zeros(B.shape()))
    r = D.convert_from_(B)
    assert r is D


def test_convert_from_deprecated_warns_and_still_works():
    bi = cytnx.Bond(cytnx.BD_KET, [[0], [1]], [1, 1], [cytnx.Symmetry.U1()])
    B = cytnx.UniTensor([bi, bi.redirect()])
    D = cytnx.UniTensor(cytnx.zeros(B.shape()))
    with pytest.warns(DeprecationWarning, match="convert_from_"):
        r = D.convert_from(B)
    assert r is D


# ---------------------------------------------------------------------------
# Already-handled pairs (no code change in this commit): confirm the
# existing deprecation shims for set_labels/relabels/relabels_/set_rowrank
# still behave per the underscore convention (in-place _ returns self;
# deprecated forms warn).
# ---------------------------------------------------------------------------


def test_set_labels_deprecated_still_warns_and_delegates_to_relabel_():
    ut = _labeled_tensor()
    with pytest.warns(DeprecationWarning):
        r = ut.set_labels(["a", "b", "c"])
    assert r is ut
    assert ut.labels() == ["a", "b", "c"]


def test_set_rowrank__is_inplace_and_set_rowrank_is_out_of_place():
    ut = _labeled_tensor()
    r = ut.set_rowrank_(2)
    assert r is ut
    assert ut.rowrank() == 2

    out = ut.set_rowrank(1)
    assert ut.rowrank() == 2  # unchanged
    assert out.rowrank() == 1
    assert out is not ut
