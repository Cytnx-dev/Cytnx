import pytest

import cytnx
from cytnx import Type


def test_rank_zero_tensor_empty_tuple_get_set():
    tensor = cytnx.zeros([], dtype=Type.Double)
    assert tensor.rank() == 0
    assert list(tensor.shape()) == []
    assert tensor.is_scalar()

    tensor[()] = 3.25
    selected = tensor[()]
    assert selected.rank() == 0
    assert selected.is_scalar()
    assert selected.item() == 3.25

    replacement = cytnx.zeros([], dtype=Type.Double)
    replacement[()] = -2.0
    tensor[()] = replacement
    assert tensor[()].item() == -2.0

    with pytest.raises(cytnx.CytnxError, match="rank-0 Tensor"):
        _ = tensor[0]
    with pytest.raises(cytnx.CytnxError, match="rank-0 Tensor"):
        tensor[0] = 1.0


def test_rank_zero_tensor_is_scalar_operand_not_just_size_one():
    scalar = cytnx.zeros([], dtype=Type.Double)
    scalar[()] = 2.0
    vec = cytnx.arange(3).astype(Type.Double)

    out = scalar * vec
    assert list(out.shape()) == [3]
    assert out[0].item() == 0.0
    assert out[1].item() == 2.0
    assert out[2].item() == 4.0

    legacy_shape_one = cytnx.zeros([1], dtype=Type.Double)
    legacy_shape_one[0] = 2.0
    assert not legacy_shape_one.is_scalar()
    assert legacy_shape_one.item() == 2.0
    with pytest.raises(cytnx.CytnxError):
        _ = legacy_shape_one + vec


def test_vectordot_returns_rank_zero_tensor():
    vec = cytnx.arange(3).astype(Type.Double)
    dot = cytnx.linalg.Vectordot(vec, vec, False)

    assert dot.is_scalar()
    assert list(dot.shape()) == []
    assert dot.item() == 5.0
    assert (dot * vec)[2].item() == 10.0


def test_tensordot_full_contraction_returns_rank_zero_tensor():
    vec = cytnx.arange(3).astype(Type.Double)
    dot = cytnx.linalg.Tensordot(vec, vec, [0], [0])

    assert dot.is_scalar()
    assert list(dot.shape()) == []
    assert dot.item() == 5.0
    assert (dot + vec)[2].item() == 7.0


def test_rank_zero_unitensor_empty_tuple_get_set():
    tensor = cytnx.zeros([], dtype=Type.Double)
    tensor[()] = 3.25
    unitensor = cytnx.UniTensor(tensor, False, 0)

    selected = unitensor[()]
    assert selected.rank() == 0
    assert selected.get_block_().is_scalar()
    assert selected.item() == 3.25

    unitensor[()] = 4.0
    assert unitensor.item() == 4.0

    replacement = cytnx.zeros([], dtype=Type.Double)
    replacement[()] = -2.0
    unitensor[()] = replacement
    assert unitensor.item() == -2.0

    replacement_ut = cytnx.UniTensor(replacement, False, 0)
    replacement_ut[()] = 7.0
    unitensor[()] = replacement_ut
    assert unitensor.item() == 7.0

    with pytest.raises(cytnx.CytnxError, match="rank-0 UniTensor"):
        _ = unitensor[0]
    with pytest.raises(cytnx.CytnxError, match="rank-0 UniTensor"):
        unitensor[0] = replacement
