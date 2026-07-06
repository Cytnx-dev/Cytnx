import warnings

import pytest

import cytnx


def test_get_blocks_deprecated_slient_warning():
    """Test that using 'slient' parameter triggers FutureWarning"""
    # Create a BlockTensor for testing
    bond = cytnx.Bond(cytnx.BD_IN,
                      [cytnx.Qs(1) >> 1, cytnx.Qs(-1) >> 1],
                      [cytnx.Symmetry.U1()])
    unitensor = cytnx.UniTensor([bond])

    # Test that using deprecated 'slient' parameter raises FutureWarning
    with pytest.warns(
            FutureWarning,
            match=
            "Keyword 'slient' is deprecated and will be removed in v2.0.0; use 'silent' instead."
    ):
        unitensor.get_blocks_(slient=True)

    # Test that the method still works with deprecated parameter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result_deprecated = unitensor.get_blocks_(slient=True)
        result_new = unitensor.get_blocks_(silent=True)
        # Both should return the same result
        assert len(result_deprecated) == len(result_new)


def test_get_blocks_new_silent_parameter():
    """Test that new 'silent' parameter works without warnings"""
    bond = cytnx.Bond(cytnx.BD_IN,
                      [cytnx.Qs(1) >> 1, cytnx.Qs(-1) >> 1],
                      [cytnx.Symmetry.U1()])
    unitensor = cytnx.UniTensor([bond])

    # Test that new 'silent' parameter doesn't trigger warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into exceptions
        result_silent_true = unitensor.get_blocks_(silent=True)
        result_silent_false = unitensor.get_blocks_(silent=False)
        result_default = unitensor.get_blocks_()

        # All should work without warnings
        assert isinstance(result_silent_true, list)
        assert isinstance(result_silent_false, list)
        assert isinstance(result_default, list)


def test_get_blocks_positional_argument():
    """Test that positional argument still works"""
    bond = cytnx.Bond(cytnx.BD_IN,
                      [cytnx.Qs(1) >> 1, cytnx.Qs(-1) >> 1],
                      [cytnx.Symmetry.U1()])
    unitensor = cytnx.UniTensor([bond])

    # Test positional argument
    result_pos_true = unitensor.get_blocks_(True)
    result_pos_false = unitensor.get_blocks_(False)

    assert isinstance(result_pos_true, list)
    assert isinstance(result_pos_false, list)


def test_get_blocks_argument_validation():
    """Test argument validation for get_blocks_"""
    bond = cytnx.Bond(cytnx.BD_IN,
                      [cytnx.Qs(1) >> 1, cytnx.Qs(-1) >> 1],
                      [cytnx.Symmetry.U1()])
    unitensor = cytnx.UniTensor([bond])

    # Test too many arguments
    with pytest.raises(
            TypeError, match="get_blocks_\\(\\) takes at most 1 argument"):
        unitensor.get_blocks_(True, False)

    with pytest.raises(
            TypeError, match="get_blocks_\\(\\) takes at most 1 argument"):
        unitensor.get_blocks_(silent=True, slient=False)

    with pytest.raises(
            TypeError, match="get_blocks_\\(\\) takes at most 1 argument"):
        unitensor.get_blocks_(True, silent=False)

    # Test invalid keyword argument
    with pytest.raises(
            TypeError,
            match=
            "'invalid_arg' is an invalid keyword argument for get_blocks_\\(\\)"
    ):
        unitensor.get_blocks_(invalid_arg=True)


def _u1_block_pair():
    """A 2-leg U(1) BlockUniTensor (blocks at the (0,0) and (1,1) sectors) and its bond."""
    bi = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1],
                    [cytnx.Symmetry.U1()])
    B = cytnx.UniTensor([bi, bi.redirect()])
    return bi, B


def test_convert_from_returns_self():
    """convert_from returns the (in-place converted) UniTensor for fluent chaining."""
    bi, B = _u1_block_pair()
    B.at([0, 0]).value = 2.0
    B.at([1, 1]).value = 3.0

    # Dense of the same shape, filled from the block tensor.
    D = cytnx.UniTensor(cytnx.zeros(B.shape()))
    r = D.convert_from(B)
    assert r is D  # returns self, not None
    assert abs(r.at([0, 0]).value - 2.0) < 1e-12
    assert abs(r.at([1, 1]).value - 3.0) < 1e-12
    assert abs(r.at([0, 1]).value) < 1e-12  # symmetry-forbidden entry stays zero

    # Round-trip: dense back into a fresh block recovers the original blocks.
    B2 = cytnx.UniTensor([bi, bi.redirect()])
    B2.convert_from(D)
    assert (B.get_block_(0) - B2.get_block_(0)).Norm().item() < 1e-12
    assert (B.get_block_(1) - B2.get_block_(1)).Norm().item() < 1e-12


def test_truncate_inplace_string_label_shrinks_bond():
    """truncate_ with a string label mutates in place, like the bond-index overload.

    Regression test: this overload used to call the non-mutating truncate()
    and discard the result, making it a silent no-op.
    """
    T = cytnx.UniTensor(cytnx.arange(12).reshape(3, 4), labels=["a", "b"])
    r = T.truncate_("b", 2)
    assert r is T  # in-place method returns self for chaining
    assert list(T.shape()) == [3, 2]  # the bond actually shrank
    # kept data is the leading slice: element [2][1] of arange(12).reshape(3,4)
    assert abs(T.at([2, 1]).value - 9.0) < 1e-12


def test_truncate_inplace_string_and_index_overloads_agree():
    """The string-label and bond-index truncate_ overloads produce the same result."""
    A = cytnx.UniTensor(cytnx.arange(12).reshape(3, 4), labels=["a", "b"])
    B = A.clone()
    A.truncate_("b", 2)
    B.truncate_(1, 2)
    assert list(A.shape()) == list(B.shape())
    assert (A.get_block_() - B.get_block_()).Norm().item() < 1e-12


def test_convert_from_tol_default_rejects_forbidden_nonzero():
    """Default tol=0 rejects a dense->block conversion with a nonzero forbidden entry."""
    bi, B = _u1_block_pair()
    D = cytnx.UniTensor(cytnx.zeros(B.shape()))
    D.at([0, 1]).value = 1.0  # nonzero in a symmetry-forbidden position

    with pytest.raises(Exception):
        B.convert_from(D)  # tol defaults to 0 -> must raise

    # force=True ignores forbidden entries and succeeds, still returning self.
    B2 = cytnx.UniTensor([bi, bi.redirect()])
    r = B2.convert_from(D, force=True)
    assert r is B2
