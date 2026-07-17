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
    left = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 2, cytnx.Qs(1) >> 1], [sym])
    right = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0) >> 2, cytnx.Qs(1) >> 1], [sym])
    ut = cytnx.UniTensor([left, right], labels=["left", "right"])
    ut.set_name_("block template")
    return ut


def test_dense_unitensor_uniform_randomizes_exact_seeded_values():
    ut = _make_dense_unitensor()

    ut.uniform_(low=-1.0, high=1.0, seed=7)

    expected = np.array(
        [
            [-0.5453218500705863, -0.3620555443782737, 0.9564457924284084],
            [-0.08883018432023693, -0.38397446555179104, -0.47225831843051325],
        ]
    )
    np.testing.assert_array_equal(_dense_block_values(ut), expected)
    assert ut.shape() == [2, 3]
    assert ut.labels() == ["left", "right"]
    assert ut.name() == "template"
    assert ut.dtype() == cytnx.Type.Double


def test_dense_unitensor_normal_randomizes_exact_seeded_values():
    ut = _make_dense_unitensor()

    ut.normal_(mean=2.0, std=3.0, seed=11)

    expected = np.array(
        [
            [2.600942056996955, 0.425072427659269, 1.3217000451409238],
            [1.02361612487187, 2.460543861218456, 7.500269922860935],
        ]
    )
    np.testing.assert_array_equal(_dense_block_values(ut), expected)
    assert ut.shape() == [2, 3]
    assert ut.labels() == ["left", "right"]
    assert ut.name() == "template"
    assert ut.dtype() == cytnx.Type.Double


def test_block_unitensor_uniform_randomizes_exact_seeded_values():
    ut = _make_block_unitensor()

    ut.uniform_(low=-1.0, high=1.0, seed=7)

    expected = [
        np.array(
            [
                [-0.5453218500705863, -0.3620555443782737],
                [0.9564457924284084, -0.08883018432023693],
            ]
        ),
        np.array([[-0.5453218500705863]]),
    ]
    actual = _block_values(ut)
    assert len(actual) == len(expected)
    for actual_block, expected_block in zip(actual, expected):
        np.testing.assert_array_equal(actual_block, expected_block)
    assert ut.labels() == ["left", "right"]
    assert ut.name() == "block template"
    assert ut.dtype() == cytnx.Type.Double


def test_block_unitensor_normal_randomizes_exact_seeded_values():
    ut = _make_block_unitensor()

    ut.normal_(mean=2.0, std=3.0, seed=11)

    expected = [
        np.array(
            [
                [2.600942056996955, 0.425072427659269],
                [1.3217000451409238, 1.02361612487187],
            ]
        ),
        np.array([[2.600942056996955]]),
    ]
    actual = _block_values(ut)
    assert len(actual) == len(expected)
    for actual_block, expected_block in zip(actual, expected):
        np.testing.assert_array_equal(actual_block, expected_block)
    assert ut.labels() == ["left", "right"]
    assert ut.name() == "block template"
    assert ut.dtype() == cytnx.Type.Double
