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
