import warnings

import pytest

import cytnx


def test_zn_combine_rule_canonical_inputs_no_warning():
    """Canonical qnums in [0, n) must not trigger any deprecation warning."""
    z3 = cytnx.Symmetry.Zn(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert z3.combine_rule(1, 2) == 0
        assert z3.combine_rule(2, 2) == 1
        assert z3.combine_rule(0, 0) == 0
        assert z3.combine_rule(1, 1, is_reverse=True) == 1
        assert z3.reverse_rule(0) == 0
        assert z3.reverse_rule(1) == 2
        assert z3.reverse_rule(2) == 1


def test_zn_combine_rule_negative_input_warns_and_normalizes():
    """Out-of-range Zn inputs to combine_rule emit FutureWarning and are normalized."""
    z2 = cytnx.Symmetry.Zn(2)
    with pytest.warns(
            FutureWarning,
            match=
            "Passing out-of-range Z2 qnum -1 to 'combine_rule' \\(argument 'qnL'\\) is deprecated"):
        assert z2.combine_rule(-1, 0) == 1

    with pytest.warns(
            FutureWarning,
            match=
            "Passing out-of-range Z2 qnum -1 to 'combine_rule' \\(argument 'qnR'\\) is deprecated"):
        assert z2.combine_rule(0, -1) == 1


def test_zn_combine_rule_above_range_input_warns_and_normalizes():
    """qnum >= n is normalized via modulo."""
    z3 = cytnx.Symmetry.Zn(3)
    with pytest.warns(FutureWarning,
                      match="Passing out-of-range Z3 qnum 5"):
        assert z3.combine_rule(5, 0) == 2

    with pytest.warns(FutureWarning,
                      match="Passing out-of-range Z3 qnum 4"):
        assert z3.combine_rule(1, 4) == 2


def test_zn_combine_rule_with_is_reverse_warns_and_normalizes():
    """is_reverse path also warns + normalizes.

    Uses Z3 (N > 2) so the reverse step is non-trivial: in Z2 every qnum
    is its own inverse, which would let a bug in the reverse path pass
    silently.
    """
    z3 = cytnx.Symmetry.Zn(3)
    with pytest.warns(FutureWarning, match="out-of-range Z3 qnum -1"):
        # combine(-1, 0) normalizes to combine(2, 0) = 2; reverse(2) in Z3 = 1.
        assert z3.combine_rule(-1, 0, is_reverse=True) == 1


def test_zn_reverse_rule_out_of_range_warns_and_normalizes():
    """reverse_rule warns + normalizes for out-of-range inputs."""
    z3 = cytnx.Symmetry.Zn(3)
    with pytest.warns(
            FutureWarning,
            match=
            "Passing out-of-range Z3 qnum -1 to 'reverse_rule' \\(argument 'qin'\\) is deprecated"):
        assert z3.reverse_rule(-1) == 1

    with pytest.warns(FutureWarning,
                      match="Passing out-of-range Z3 qnum 4"):
        assert z3.reverse_rule(4) == 2


def test_u1_does_not_normalize_or_warn():
    """U1 accepts arbitrary integers; no warning, no modulo."""
    u1 = cytnx.Symmetry.U1()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert u1.combine_rule(-3, 5) == 2
        assert u1.combine_rule(7, -10) == -3
        assert u1.reverse_rule(-4) == 4
        assert u1.reverse_rule(9) == -9
