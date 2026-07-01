"""Tests for UniTensor.put_block / put_block_ argument handling.

Covers the typed overloads (by block index, by qidx, by labels+qidx), the
deprecated ``force`` argument, and the deprecated ``in`` keyword argument that
is accepted through ``**kwargs`` and forwarded to ``Tin``.
"""

import warnings

import pytest

import cytnx


def _u1_block_pair():
    """A 2-leg U(1) BlockUniTensor with 1x1 blocks at the (0,0) and (1,1) sectors."""
    bi = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1],
                    [cytnx.Symmetry.U1()])
    return cytnx.UniTensor([bi, bi.redirect()])


def _block(value):
    """A 1x1 block tensor holding a single value (matches the sectors above)."""
    t = cytnx.zeros([1, 1])
    t[0, 0] = value
    return t


# --------------------------------------------------------------------------
# put_block: typed overloads
# --------------------------------------------------------------------------

def test_put_block_by_index_positional_and_keyword():
    B = _u1_block_pair()
    B.put_block(_block(5.0), 0)
    assert B.get_block_(0)[0, 0].item() == 5.0
    B.put_block(Tin=_block(6.0), idx=0)
    assert B.get_block_(0)[0, 0].item() == 6.0


def test_put_block_by_qidx():
    B = _u1_block_pair()
    B.put_block(_block(7.0), [0, 0])  # sector of block 0
    assert B.get_block_(0)[0, 0].item() == 7.0
    B.put_block(Tin=_block(8.0), qidx=[1, 1])  # sector of block 1
    assert B.get_block_(1)[0, 0].item() == 8.0


def test_put_block_by_labels_and_qidx():
    B = _u1_block_pair()
    B.put_block(_block(9.0), ['0', '1'], [0, 0])
    assert B.get_block_(0)[0, 0].item() == 9.0
    B.put_block(Tin=_block(10.0), labels=['0', '1'], qidx=[1, 1])
    assert B.get_block_(1)[0, 0].item() == 10.0


# --------------------------------------------------------------------------
# put_block: deprecated 'force' argument
# --------------------------------------------------------------------------

def test_put_block_force_argument_is_deprecated():
    B = _u1_block_pair()
    with pytest.warns(FutureWarning, match="'force' is deprecated"):
        B.put_block(_block(1.0), [0, 0], True)
    with pytest.warns(FutureWarning, match="'force' is deprecated"):
        B.put_block(_block(2.0), ['0', '1'], [0, 0], True)


# --------------------------------------------------------------------------
# put_block: deprecated 'in' keyword argument
# --------------------------------------------------------------------------

def test_put_block_in_keyword_is_deprecated_and_forwards_to_Tin():
    B = _u1_block_pair()
    with pytest.warns(FutureWarning, match="'in' keyword argument"):
        B.put_block(**{"in": _block(3.0), "idx": 0})
    assert B.get_block_(0)[0, 0].item() == 3.0


def test_put_block_rejects_both_in_and_Tin():
    B = _u1_block_pair()
    with pytest.raises(TypeError, match="both the deprecated 'in' and 'Tin'"):
        B.put_block(**{"in": _block(1.0), "Tin": _block(1.0)})


def test_put_block_rejects_kwargs_without_in():
    B = _u1_block_pair()
    with pytest.raises(TypeError, match="incompatible arguments"):
        B.put_block(idx=0)


# --------------------------------------------------------------------------
# put_block_ (in-place): same surface
# --------------------------------------------------------------------------

def test_put_block_inplace_typed_overloads():
    B = _u1_block_pair()
    B.put_block_(_block(5.0), 0)
    assert B.get_block_(0)[0, 0].item() == 5.0
    B.put_block_(_block(6.0), [1, 1])
    assert B.get_block_(1)[0, 0].item() == 6.0
    B.put_block_(_block(7.0), ['0', '1'], [0, 0])
    assert B.get_block_(0)[0, 0].item() == 7.0


def test_put_block_inplace_force_argument_is_deprecated():
    B = _u1_block_pair()
    with pytest.warns(FutureWarning, match="'force' is deprecated"):
        B.put_block_(_block(1.0), [0, 0], True)
    with pytest.warns(FutureWarning, match="'force' is deprecated"):
        B.put_block_(_block(2.0), ['0', '1'], [0, 0], True)


def test_put_block_inplace_in_keyword_is_deprecated_and_forwards_to_Tin():
    B = _u1_block_pair()
    with pytest.warns(FutureWarning, match="'in' keyword argument"):
        B.put_block_(**{"in": _block(3.0), "idx": 0})
    assert B.get_block_(0)[0, 0].item() == 3.0


def test_put_block_inplace_rejects_both_in_and_Tin():
    B = _u1_block_pair()
    with pytest.raises(TypeError, match="both the deprecated 'in' and 'Tin'"):
        B.put_block_(**{"in": _block(1.0), "Tin": _block(1.0)})


def test_put_block_inplace_rejects_kwargs_without_in():
    B = _u1_block_pair()
    with pytest.raises(TypeError, match="incompatible arguments"):
        B.put_block_(idx=0)
