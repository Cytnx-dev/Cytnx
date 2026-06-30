"""Tests for Bond Python-side helpers added in Bond_conti.py."""

import cytnx


def _u1_bond():
    """A symmetric Bond with sectors at quantum numbers 0 and 1."""
    return cytnx.Bond(
        cytnx.BD_IN,
        [cytnx.Qs(0) >> 2, cytnx.Qs(1) >> 3],
        [cytnx.Symmetry.U1()],
    )


def test_get_degeneracy_default_returns_scalar():
    """Without return_indices the degeneracy is returned on its own."""
    bd = _u1_bond()
    deg = bd.getDegeneracy([0])
    assert isinstance(deg, int)
    # An explicit False must behave like the default.
    assert bd.getDegeneracy([0], False) == deg


def test_get_degeneracy_return_indices_returns_tuple():
    """return_indices=True opts in to the (degeneracy, indices) tuple."""
    bd = _u1_bond()
    deg = bd.getDegeneracy([0])
    out = bd.getDegeneracy([0], return_indices=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0] == deg
    assert list(out[1]) == [0]


def test_get_degeneracy_accepts_qs():
    """A cytnx.Qs is accepted as qnum, matching the native overload, for both
    return shapes."""
    bd = _u1_bond()
    deg = bd.getDegeneracy([0])
    assert bd.getDegeneracy(cytnx.Qs(0)) == deg
    out = bd.getDegeneracy(cytnx.Qs(0), return_indices=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0] == deg
    assert list(out[1]) == [0]
