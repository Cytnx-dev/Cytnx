import numpy as np

import cytnx


def _dense_block_values(ut):
    return ut.get_block().numpy()


def _block_values(ut):
    return [block.numpy() for block in ut.get_blocks()]


def _make_dense_unitensor():
    ut = cytnx.UniTensor.zeros([2, 3], labels=["left", "right"], dtype=cytnx.Type.Double)
    ut.set_name_("template")
    return ut


def _make_block_unitensor():
    sym = cytnx.Symmetry.U1()
    left = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 2, cytnx.Qs(1) >> 2], [sym])
    right = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0) >> 2, cytnx.Qs(1) >> 2], [sym])
    ut = cytnx.UniTensor([left, right], labels=["left", "right"])
    ut.set_name_("block template")
    return ut


def _assert_blocks_equal(lhs, rhs):
    assert len(lhs) == len(rhs)
    for lhs_block, rhs_block in zip(lhs, rhs):
        np.testing.assert_array_equal(lhs_block, rhs_block)


def test_dense_unitensor_uniform_same_seed_is_reproducible():
    lhs = _make_dense_unitensor()
    rhs = _make_dense_unitensor()

    lhs.uniform_(low=-1.0, high=1.0, seed=7)
    rhs.uniform_(low=-1.0, high=1.0, seed=7)

    np.testing.assert_array_equal(_dense_block_values(lhs), _dense_block_values(rhs))
    assert lhs.shape() == [2, 3]
    assert lhs.labels() == ["left", "right"]
    assert lhs.name() == "template"
    assert lhs.dtype() == cytnx.Type.Double


def test_dense_unitensor_normal_same_seed_is_reproducible():
    lhs = _make_dense_unitensor()
    rhs = _make_dense_unitensor()

    lhs.normal_(mean=2.0, std=3.0, seed=11)
    rhs.normal_(mean=2.0, std=3.0, seed=11)

    np.testing.assert_array_equal(_dense_block_values(lhs), _dense_block_values(rhs))
    assert lhs.shape() == [2, 3]
    assert lhs.labels() == ["left", "right"]
    assert lhs.name() == "template"
    assert lhs.dtype() == cytnx.Type.Double


def test_block_unitensor_uniform_same_seed_is_reproducible_with_distinct_blocks():
    lhs = _make_block_unitensor()
    rhs = _make_block_unitensor()

    lhs.uniform_(low=-1.0, high=1.0, seed=7)
    rhs.uniform_(low=-1.0, high=1.0, seed=7)

    lhs_blocks = _block_values(lhs)
    rhs_blocks = _block_values(rhs)
    _assert_blocks_equal(lhs_blocks, rhs_blocks)
    assert len(lhs_blocks) == 2
    assert not np.array_equal(lhs_blocks[0], lhs_blocks[1])
    assert lhs.labels() == ["left", "right"]
    assert lhs.name() == "block template"
    assert lhs.dtype() == cytnx.Type.Double


def test_block_unitensor_normal_same_seed_is_reproducible():
    lhs = _make_block_unitensor()
    rhs = _make_block_unitensor()

    lhs.normal_(mean=2.0, std=3.0, seed=11)
    rhs.normal_(mean=2.0, std=3.0, seed=11)

    _assert_blocks_equal(_block_values(lhs), _block_values(rhs))
    assert lhs.labels() == ["left", "right"]
    assert lhs.name() == "block template"
    assert lhs.dtype() == cytnx.Type.Double
