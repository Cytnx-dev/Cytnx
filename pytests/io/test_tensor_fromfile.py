import numpy as np

import cytnx as cy


def test_tensor_fromfile_reads_raw_tofile_output(tmp_path):
    path = tmp_path / "tensor_raw.bin"
    tensor = cy.arange(0, 5, 1, dtype=cy.Type.Double)

    tensor.Tofile(str(path))
    loaded = cy.Tensor.Fromfile(str(path), cy.Type.Double)

    assert loaded.dtype() == cy.Type.Double
    assert list(loaded.shape()) == [5]
    np.testing.assert_allclose(loaded.numpy(), np.arange(5, dtype=np.float64))


def test_tensor_fromfile_count_zero_returns_empty_tensor(tmp_path):
    path = tmp_path / "tensor_raw.bin"
    tensor = cy.arange(0, 5, 1, dtype=cy.Type.Double)

    tensor.Tofile(str(path))
    loaded = cy.Tensor.Fromfile(str(path), cy.Type.Double, 0)

    assert loaded.dtype() == cy.Type.Double
    assert list(loaded.shape()) == [0]
    assert loaded.numpy().size == 0
