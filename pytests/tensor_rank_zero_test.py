import pytest
import numpy as np

import cytnx
from cytnx import Type


def test_python_generators_distinguish_integer_size_and_empty_shape():
    vector = cytnx.zeros(5, dtype=Type.Double)
    assert vector.rank() == 1
    assert list(vector.shape()) == [5]
    assert not vector.is_scalar()

    scalar = cytnx.zeros([], dtype=Type.Double)
    assert scalar.rank() == 0
    assert list(scalar.shape()) == []
    assert scalar.is_scalar()
    assert scalar.item() == 0.0

    ones_vector = cytnx.ones(5, dtype=Type.Float)
    assert ones_vector.rank() == 1
    assert list(ones_vector.shape()) == [5]
    assert not ones_vector.is_scalar()

    ones_scalar = cytnx.ones([], dtype=Type.Float)
    assert ones_scalar.rank() == 0
    assert list(ones_scalar.shape()) == []
    assert ones_scalar.is_scalar()
    assert ones_scalar.item() == 1.0


def test_from_numpy_accepts_zero_dimensional_buffer():
    scalar = cytnx.zeros([], dtype=Type.Double)
    scalar[()] = 3.25

    roundtrip = cytnx.from_numpy(scalar.numpy())
    assert roundtrip.dtype() == Type.Double
    assert roundtrip.rank() == 0
    assert list(roundtrip.shape()) == []
    assert roundtrip.is_scalar()
    assert roundtrip.item() == 3.25

    from_numpy_scalar = cytnx.from_numpy(np.array(4.5, dtype=np.float64))
    assert from_numpy_scalar.dtype() == Type.Double
    assert from_numpy_scalar.rank() == 0
    assert list(from_numpy_scalar.shape()) == []
    assert from_numpy_scalar.is_scalar()
    assert from_numpy_scalar.item() == 4.5

    int16_scalar = cytnx.from_numpy(np.array(-7, dtype=np.int16))
    assert int16_scalar.dtype() == Type.Int16
    assert int16_scalar.rank() == 0
    assert int16_scalar.item() == -7

    uint16_scalar = cytnx.from_numpy(np.array(9, dtype=np.uint16))
    assert uint16_scalar.dtype() == Type.Uint16
    assert uint16_scalar.rank() == 0
    assert uint16_scalar.item() == 9


@pytest.mark.parametrize(("dtype", "np_dtype", "value"),
                         [(Type.Int16, np.int16, -7), (Type.Uint16, np.uint16, 9)])
def test_tensor_numpy_roundtrips_int16_zero_dimensional_buffer(dtype, np_dtype, value):
    scalar = cytnx.zeros([], dtype=dtype)
    scalar[()] = value

    ndarray = scalar.numpy()
    assert ndarray.shape == ()
    assert ndarray.dtype == np.dtype(np_dtype)
    assert ndarray.item() == value

    roundtrip = cytnx.from_numpy(ndarray)
    assert roundtrip.dtype() == dtype
    assert roundtrip.rank() == 0
    assert roundtrip.item() == value


def test_python_unitensor_generators_distinguish_integer_size_and_empty_shape():
    vector = cytnx.UniTensor.zeros(5, dtype=Type.Double)
    assert vector.rank() == 1
    assert list(vector.shape()) == [5]

    scalar = cytnx.UniTensor.zeros([], dtype=Type.Double)
    assert scalar.rank() == 0
    assert list(scalar.shape()) == []
    assert scalar.get_block_().is_scalar()
    assert scalar.item() == 0.0

    ones_vector = cytnx.UniTensor.ones(5, dtype=Type.Float)
    assert ones_vector.rank() == 1
    assert list(ones_vector.shape()) == [5]

    ones_scalar = cytnx.UniTensor.ones([], dtype=Type.Float)
    assert ones_scalar.rank() == 0
    assert list(ones_scalar.shape()) == []
    assert ones_scalar.get_block_().is_scalar()
    assert ones_scalar.item() == 1.0


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


def test_sort_rejects_void_tensor():
    with pytest.raises(cytnx.CytnxError, match="void Tensor"):
        cytnx.algo.Sort(cytnx.Tensor())


def test_tensor_single_slice_does_not_print(capsys):
    tensor = cytnx.arange(4)

    sliced = tensor[1:3]
    captured = capsys.readouterr()

    assert captured.out == ""
    assert list(sliced.shape()) == [2]


def test_tensor_tuple_slice_rejects_too_many_indices_before_shape_lookup():
    tensor = cytnx.arange(3)

    with pytest.raises(cytnx.CytnxError, match="too many indices"):
        _ = tensor[:, :]
    with pytest.raises(cytnx.CytnxError, match="too many indices"):
        tensor[:, :] = tensor
    with pytest.raises(cytnx.CytnxError, match="too many indices"):
        tensor[:, :] = 1.0


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


def test_tensordot_rank_zero_axis_out_of_bounds():
    scalar = cytnx.zeros([], dtype=Type.Double)

    with pytest.raises(cytnx.CytnxError, match="axis .*out of bounds"):
        cytnx.linalg.Tensordot(scalar, scalar, [0], [0])


def test_tensordot_rejects_duplicate_axes_before_dispatch():
    lhs = cytnx.zeros([2], dtype=Type.Double)
    rhs = cytnx.zeros([2, 2], dtype=Type.Double)

    with pytest.raises(cytnx.CytnxError, match="duplicate contracted axis"):
        cytnx.linalg.Tensordot(lhs, rhs, [0, 0], [0, 1])

    with pytest.raises(cytnx.CytnxError, match="duplicate contracted axis"):
        cytnx.linalg.Tensordot(lhs, rhs, [0, 0], [0, 0])


def test_tensordot_dg_rejects_duplicate_axes_before_dispatch():
    diag = cytnx.ones([2], dtype=Type.Double)
    rhs = cytnx.ones([2, 2], dtype=Type.Double)

    with pytest.raises(cytnx.CytnxError, match="duplicate contracted axis"):
        cytnx.linalg.Tensordot_dg(diag, rhs, [0, 0], [0, 1], True)


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


def test_unitensor_tuple_slice_rejects_too_many_indices_before_shape_lookup():
    tensor = cytnx.arange(3)
    unitensor = cytnx.UniTensor(tensor)

    with pytest.raises(cytnx.CytnxError, match="too many indices"):
        _ = unitensor[:, :]
    with pytest.raises(cytnx.CytnxError, match="too many indices"):
        unitensor[:, :] = tensor
    with pytest.raises(cytnx.CytnxError, match="too many indices"):
        unitensor[:, :] = cytnx.UniTensor(tensor)


def test_rank_zero_block_unitensor_item():
    bond = cytnx.Bond(cytnx.BD_IN,
                      [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1],
                      [cytnx.Symmetry.U1()])
    unitensor = cytnx.UniTensor([bond, bond.redirect()])
    unitensor.at([0, 0]).value = 2.0
    unitensor.at([1, 1]).value = 3.0

    with pytest.raises(cytnx.CytnxError, match="non-scalar UniTensor"):
        unitensor.item()

    traced = unitensor.Trace(0, 1)
    assert traced.uten_type_str() == "Block"
    assert traced.rank() == 0
    assert list(traced.shape()) == []
    assert traced.item() == 5.0

    dense = cytnx.UniTensor(cytnx.zeros([], dtype=Type.Double))
    dense.convert_from(traced)
    assert dense.uten_type_str() == "Dense"
    assert dense.rank() == 0
    assert dense.item() == 5.0

    block = traced.clone()
    block.convert_from(dense)
    assert block.uten_type_str() == "Block"
    assert block.rank() == 0
    assert block.item() == 5.0
