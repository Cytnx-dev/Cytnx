"""Regression tests for Cytnx-dev/Cytnx#1206.

``relabel`` returns a view that shares block storage with its parent. Contracting
two such views of one block-sparse UniTensor used to raise inside
``BlockUniTensor::contract`` (a reshape of a permuted block) and left the parent's
first block reshaped. The expected value is computed independently with numpy on
the dense array assembled from all blocks (``to_dense()`` does not expand a
symmetric UniTensor; it only expands diagonal storage).
"""

import numpy as np

import cytnx


def _z2_bond(direction, dim):
    return cytnx.Bond(direction, [cytnx.Qs(0) >> dim // 2, cytnx.Qs(1) >> dim // 2],
                      [cytnx.Symmetry.Zn(2)])


def _random_z2_rank4(seed=0, dim=4):
    rng = np.random.default_rng(seed)
    T = cytnx.UniTensor([_z2_bond(cytnx.BD_IN, dim), _z2_bond(cytnx.BD_IN, dim),
                         _z2_bond(cytnx.BD_OUT, dim), _z2_bond(cytnx.BD_OUT, dim)],
                        labels=["a", "b", "c", "d"], rowrank=2)
    for i, blk in enumerate(T.get_blocks()):
        T.put_block_(cytnx.from_numpy(rng.standard_normal(blk.shape())), i)
    return T


def _dense(T):
    """The full dense array of a block-sparse UniTensor, every block in its sector slot."""
    bonds = T.bonds()
    offsets = [np.concatenate([[0], np.cumsum(b.getDegeneracies())]) for b in bonds]
    out = np.zeros([b.dim() for b in bonds])
    for i in range(T.Nblocks()):
        blk = T.get_block_(i).numpy()
        where = tuple(slice(offsets[leg][q], offsets[leg][q] + blk.shape[leg])
                      for leg, q in enumerate(T.get_qindices(i)))
        out[where] = blk
    return out


def test_dense_assembly_covers_every_block():
    T = _random_z2_rank4()
    dense = _dense(T)
    assert dense.shape == (4, 4, 4, 4)
    assert np.count_nonzero(dense) == sum(blk.numpy().size for blk in T.get_blocks())


def test_contract_of_two_relabel_views_matches_numpy_and_leaves_parent_intact():
    T = _random_z2_rank4()
    shapes_before = [blk.shape() for blk in T.get_blocks()]
    dense_before = _dense(T)

    C = cytnx.Contract(T.relabel(["a", "b", "c", "s"]), T.relabel(["s", "q", "r", "d"]))

    assert C.labels() == ["a", "b", "c", "q", "r", "d"]
    expected = np.einsum("abcs,sqrd->abcqrd", dense_before, dense_before)
    assert C.Nblocks() > 1
    np.testing.assert_allclose(_dense(C), expected, rtol=0, atol=1e-12)
    # the operands' shared storage must be untouched by the contraction
    assert [blk.shape() for blk in T.get_blocks()] == shapes_before
    np.testing.assert_array_equal(_dense(T), dense_before)


def test_contract_of_a_view_with_its_own_parent():
    T = _random_z2_rank4(seed=1)
    dense_before = _dense(T)
    T.relabel_(["a", "b", "c", "s"])
    C = cytnx.Contract(T, T.relabel(["s", "q", "r", "d"]))
    expected = np.einsum("abcs,sqrd->abcqrd", dense_before, dense_before)
    np.testing.assert_allclose(_dense(C), expected, rtol=0, atol=1e-12)
    assert T.labels() == ["a", "b", "c", "s"]
    np.testing.assert_array_equal(_dense(T), dense_before)
