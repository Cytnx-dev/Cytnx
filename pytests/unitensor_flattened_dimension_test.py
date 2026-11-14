"""Tests for UniTensor.flattened_dimension() and cytnx.flattened_dimension(bonds)."""

import cytnx


def _u1_bond(direction, sectors):
    """A U1 Bond with the given {quantum number: degeneracy} sectors, in order."""
    return cytnx.Bond(direction, [cytnx.Qs(q) >> d for q, d in sectors],
                      [cytnx.Symmetry.U1()])


def test_dense_is_product_of_bond_dimensions():
    ut = cytnx.UniTensor(cytnx.zeros([2, 3, 4]))
    assert ut.flattened_dimension() == 24


def test_u1_counts_allowed_sectors_and_depends_on_direction():
    # ket {q=1: 2, q=0: 1} with bra {q=1: 1, q=0: 3}: sectors (1,1) and (0,0), 2 + 3 = 5.
    ket = _u1_bond(cytnx.BD_KET, [(1, 2), (0, 1)])
    bra = _u1_bond(cytnx.BD_BRA, [(1, 1), (0, 3)])
    ut = cytnx.UniTensor([ket, bra], labels=["a", "b"])
    assert ut.flattened_dimension() == 5
    assert cytnx.flattened_dimension([ket, bra]) == 5
    # The product of the shape (3 * 4) is not the dimension of a symmetric tensor.
    assert ut.shape() == [3, 4]
    # With the second bond a ket, only (0,0) is allowed: 1 * 3 = 3.
    ket2 = _u1_bond(cytnx.BD_KET, [(1, 1), (0, 3)])
    assert cytnx.flattened_dimension([ket, ket2]) == 3


def test_diagonal_uses_first_bond():
    ket = _u1_bond(cytnx.BD_KET, [(1, 2), (0, 3)])
    assert cytnx.flattened_dimension([ket, ket.redirect()], is_diag=True) == 5
    assert cytnx.flattened_dimension([ket, ket.redirect()]) == 2 * 2 + 3 * 3
