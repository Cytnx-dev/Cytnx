"""Tests for ``cytnx.linalg.Eigh_truncate`` on block-sparse UniTensors (Cytnx-dev/Cytnx#1209).

Expected values come from numpy's ``eigvalsh`` on the dense blocks; the truncation
keeps the eigenpairs of largest magnitude across all charge sectors, honours the
per-block floors ``min_blockdim`` before ``keepdim``, and returns unitary eigenvectors.
"""

import numpy as np
import pytest

import cytnx


def _hermitian_z3(seed=0, dims=(3, 2, 2)):
    rng = np.random.default_rng(seed)
    qs = [cytnx.Qs(q) >> d for q, d in enumerate(dims)]
    row = cytnx.Bond(cytnx.BD_IN, qs, [cytnx.Symmetry.Zn(3)])
    M = cytnx.UniTensor([row, row.redirect()], labels=["r", "c"], rowrank=1)
    blocks = []
    for b, d in enumerate(dims):
        A = rng.standard_normal((d, d))
        H = A + A.T
        M.put_block_(cytnx.from_numpy(H), b)
        blocks.append(H)
    return M, blocks


def _kept_by_magnitude(blocks, keepdim):
    """(sector, eigenvalue) pairs of the keepdim largest |eigenvalue| over all sectors."""
    pairs = [(b, lam) for b, H in enumerate(blocks) for lam in np.linalg.eigvalsh(H)]
    pairs.sort(key=lambda p: -abs(p[1]))
    return pairs[:keepdim]


def _as_rows(e):
    return {b: e.get_block_(b).numpy().tolist() for b in range(e.Nblocks())}


def test_keeps_largest_magnitudes_across_sectors():
    M, blocks = _hermitian_z3()
    keepdim = 4
    e, V = cytnx.linalg.Eigh_truncate(M, keepdim)
    expected = {}
    for b, lam in _kept_by_magnitude(blocks, keepdim):
        expected.setdefault(b, []).append(lam)
    expected = {b: sorted(v) for b, v in expected.items()}
    qnums = [q[0] for q in e.bonds()[0].qnums()]
    got = {qnums[i]: vals for i, vals in _as_rows(e).items()}
    assert sorted(got) == sorted(expected)
    for b in expected:
        np.testing.assert_allclose(got[b], expected[b], rtol=0, atol=1e-10)
    assert e.bonds()[0].getDegeneracies() == [len(expected[q]) for q in sorted(expected)]


def test_eigenvectors_are_orthonormal_and_satisfy_the_eigen_equation():
    M, _ = _hermitian_z3(seed=1)
    e, V = cytnx.linalg.Eigh_truncate(M, 3)
    VdV = cytnx.Contract(V.Dagger().relabel("_aux_L", "_aux_R"), V)  # (_aux_R, _aux_L)
    for blk in VdV.get_blocks():
        np.testing.assert_allclose(blk.numpy(), np.eye(blk.shape()[0]), rtol=0, atol=1e-12)
    MV = cytnx.Contract(M, V.relabel("r", "c"))
    Ve = cytnx.Contract(V, e).relabel("_aux_R", "_aux_L")
    diff = MV - Ve.permute(MV.labels())
    assert diff.norm() < 1e-10


def test_min_blockdim_floor_beats_keepdim_and_zero_drops_the_block():
    M, blocks = _hermitian_z3(seed=2)
    e, V = cytnx.linalg.Eigh_truncate(M, 1, [1, 1, 1])
    assert e.bonds()[0].getDegeneracies() == [1, 1, 1]
    for b, H in enumerate(blocks):
        lam = np.linalg.eigvalsh(H)
        np.testing.assert_allclose(e.get_block_(b).numpy(), [lam[np.argmax(abs(lam))]],
                                   rtol=0, atol=1e-10)
    e0, _ = cytnx.linalg.Eigh_truncate(M, 2, [0, 2, 0])
    assert e0.Nblocks() == 1
    assert e0.bonds()[0].qnums() == [[1]]
    assert e0.bonds()[0].getDegeneracies() == [2]
    with pytest.raises(cytnx.CytnxError, match="same number of elements"):
        cytnx.linalg.Eigh_truncate(M, 1, [1, 1])


def test_err_and_return_err_report_discarded_magnitudes():
    M, blocks = _hermitian_z3(seed=3)
    all_lam = sorted((abs(l) for H in blocks for l in np.linalg.eigvalsh(H)), reverse=True)
    threshold = (all_lam[2] + all_lam[3]) / 2
    e, V, dropped = cytnx.linalg.Eigh_truncate(M, 7, threshold, True, 2)
    assert e.bonds()[0].dim() == 3
    np.testing.assert_allclose(dropped.get_block_().numpy(), all_lam[3:], rtol=0, atol=1e-10)
    e1, V1, largest = cytnx.linalg.Eigh_truncate(M, 7, threshold, True, 1)
    assert abs(largest.get_block_().item() - all_lam[3]) < 1e-10


def _dense_array(U):
    """Full dense array of a block-sparse UniTensor, every block in its sector slot."""
    bonds = U.bonds()
    offsets = [np.concatenate([[0], np.cumsum(b.getDegeneracies())]) for b in bonds]
    out = np.zeros([b.dim() for b in bonds])
    for i in range(U.Nblocks()):
        blk = U.get_block_(i).numpy()
        where = tuple(slice(offsets[leg][q], offsets[leg][q] + blk.shape[leg])
                      for leg, q in enumerate(U.get_qindices(i)))
        out[where] = blk
    return out


@pytest.mark.parametrize("keepdim", [1, 3, 4, 6])
def test_block_sparse_path_equals_dense_path(keepdim):
    """dense rep -> Eigh_truncate  ==  block-sparse rep -> Eigh_truncate -> dense, for one matrix.

    Compared through the gauge-invariant objects: the sorted kept eigenvalues, the projector
    V V^dagger onto the kept subspace and the reconstruction V e V^dagger; numpy's eigh on the
    dense matrix is the third, independent reference (the keepdim largest |eigenvalue|).
    """
    M, _ = _hermitian_z3(seed=2)
    dense = _dense_array(M)
    assert np.count_nonzero(dense) > 0 and np.allclose(dense, dense.T)

    e_blk, v_blk = cytnx.linalg.Eigh_truncate(M, keepdim)
    e_dn, v_dn = cytnx.linalg.Eigh_truncate(
        cytnx.UniTensor(cytnx.from_numpy(dense), rowrank=1, labels=["r", "c"]), keepdim)

    def invariants(e_flat, v_matrix):
        proj = v_matrix @ v_matrix.T
        recon = v_matrix @ np.diag(e_flat) @ v_matrix.T
        return np.sort(e_flat), proj, recon

    e_b = np.concatenate([e_blk.get_block_(i).numpy() for i in range(e_blk.Nblocks())])
    inv_blk = invariants(e_b, _dense_array(v_blk))
    inv_dn = invariants(e_dn.get_block().numpy(), v_dn.get_block().numpy())
    lam, vec = np.linalg.eigh(dense)
    kept = np.argsort(-np.abs(lam), kind="stable")[:keepdim]
    inv_np = invariants(lam[kept], vec[:, kept])

    for a, b in zip(inv_blk, inv_dn):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    for a, b in zip(inv_blk, inv_np):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
    assert len(e_b) == keepdim


def _diagonal_input(values, blocked):
    """Build one known spectrum in dense or one-sector block form."""
    matrix = np.diag(np.asarray(values, dtype=np.float64))
    if not blocked:
        return cytnx.UniTensor(cytnx.from_numpy(matrix), rowrank=1, labels=["r", "c"])
    row = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> len(values)], [cytnx.Symmetry.Zn(2)])
    tensor = cytnx.UniTensor([row, row.redirect()], labels=["r", "c"], rowrank=1)
    tensor.put_block_(cytnx.from_numpy(matrix), 0)
    return tensor


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("boundary", [2**63 - 1, 2**63, 2**64 - 1])
@pytest.mark.parametrize("return_err", [1, 2])
@pytest.mark.parametrize("large_minimum", [False, True])
def test_unsigned_dimension_boundaries(blocked, boundary, return_err, large_minimum):
    tensor = _diagonal_input([1, 2, 3], blocked)
    kwargs = {"keepdim": boundary}
    if large_minimum:
        kwargs = {"keepdim": 3, "mindim": boundary, "err": 100.0}
    eigenvalues, error = cytnx.linalg.Eigh_truncate(
        tensor, is_V=False, return_err=return_err, **kwargs)
    np.testing.assert_array_equal(eigenvalues.get_block_(0).numpy(), [1, 2, 3])
    if return_err == 1:
        assert error.get_block_().item() == 0
    else:
        assert error.get_block_().shape() == [0]


@pytest.mark.parametrize("return_err", [1, 2])
@pytest.mark.parametrize("floors, expected, discarded", [
    ([0, 2], [1, 4], [5, 3, 0.5]),
    ([9, 9], [-3, 0.5, 1, 4, 5], []),
    ([1, 1], [4, 5], [3, 1, 0.5]),
])
def test_floors_with_cutoff_and_error_reports(return_err, floors, expected, discarded):
    row = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 3, cytnx.Qs(1) >> 2],
                     [cytnx.Symmetry.Zn(2)])
    tensor = cytnx.UniTensor([row, row.redirect()], labels=["r", "c"], rowrank=1)
    tensor.put_block_(cytnx.from_numpy(np.diag([-3.0, 0.5, 5.0])), 0)
    tensor.put_block_(cytnx.from_numpy(np.diag([1.0, 4.0])), 1)
    eigenvalues, error = cytnx.linalg.Eigh_truncate(
        tensor, 2, floors, err=100.0, is_V=False, return_err=return_err)
    actual = np.concatenate([block.numpy() for block in eigenvalues.get_blocks()])
    np.testing.assert_array_equal(np.sort(actual), expected)
    if return_err == 1:
        assert error.get_block_().item() == (discarded[0] if discarded else 0)
    else:
        np.testing.assert_array_equal(error.get_block_().numpy(), discarded)


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("values, keepdim, cutoff, mindim, expected, discarded", [
    ([0, 0, 0], 1, 0.0, 1, [0, 0, 0], []),
    ([1, 2, 3], 3, 2.0, 1, [2, 3], [1]),
    ([1, 2, 3], 3, 100.0, 2, [2, 3], [1]),
    ([1, 2, 3], 1, 100.0, 3, [3], [2, 1]),
    ([1, 1 + 1e-10, 3], 2, 0.0, 1, [1 + 1e-10, 3], [1]),
])
def test_selection_boundary_interactions(
        blocked, values, keepdim, cutoff, mindim, expected, discarded):
    tensor = _diagonal_input(values, blocked)
    eigenvalues, error = cytnx.linalg.Eigh_truncate(
        tensor, keepdim, err=cutoff, mindim=mindim, is_V=False, return_err=2)
    np.testing.assert_allclose(eigenvalues.get_block_(0).numpy(), expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(error.get_block_().numpy(), discarded)


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_dtypes_and_input_preservation(blocked, dtype):
    matrix = np.array([[2, 1], [1, -2]], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        matrix[0, 1] += 0.5j
        matrix[1, 0] -= 0.5j
    tensor = _diagonal_input([1, 2], blocked)
    tensor = tensor.astype(cytnx.from_numpy(matrix).dtype())
    tensor.put_block_(cytnx.from_numpy(matrix), 0)
    before = tensor.get_block_(0).numpy().copy()
    eigenvalues, vectors = cytnx.linalg.Eigh_truncate(tensor, 2)
    values = eigenvalues.get_block_(0).numpy()
    basis = vectors.get_block_(0).numpy()
    expected_real_dtype = np.float32 if dtype in (np.float32, np.complex64) else np.float64
    assert values.dtype == expected_real_dtype
    assert basis.dtype == dtype
    tolerance = 1e-5 if expected_real_dtype == np.float32 else 1e-12
    np.testing.assert_allclose(values, np.linalg.eigvalsh(matrix), atol=tolerance, rtol=0)
    np.testing.assert_allclose(matrix @ basis, basis * values, atol=tolerance, rtol=0)
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(2), atol=tolerance, rtol=0)
    np.testing.assert_array_equal(tensor.get_block_(0).numpy(), before)
    assert tensor.labels() == ["r", "c"]
    assert tensor.rowrank() == 1
    with pytest.raises(cytnx.CytnxError, match="keepdim must be >=1"):
        cytnx.linalg.Eigh_truncate(tensor, 0)
    np.testing.assert_array_equal(tensor.get_block_(0).numpy(), before)


@pytest.mark.parametrize("blocked, expected", [(False, [3]), (True, [1, 2, 3])])
def test_dense_floor_is_a_cutoff_minimum(blocked, expected):
    eigenvalues, _ = cytnx.linalg.Eigh_truncate(_diagonal_input([1, 2, 3], blocked), 1, [3])
    np.testing.assert_array_equal(eigenvalues.get_block_(0).numpy(), expected)


def test_dense_noncontiguous_higher_rank_preserves_input_and_flattens_vectors():
    array = np.diag([1.0, 2.0, 3.0]).reshape(1, 3, 1, 3)
    tensor = cytnx.UniTensor(cytnx.from_numpy(array), rowrank=2, labels=["a", "b", "c", "d"])
    tensor = tensor.permute([2, 3, 0, 1], rowrank=2)
    assert not tensor.is_contiguous()
    eigenvalues, vectors = cytnx.linalg.Eigh_truncate(tensor, 2)
    np.testing.assert_array_equal(eigenvalues.get_block_().numpy(), [2, 3])
    assert vectors.shape() == [3, 2]
    assert vectors.rowrank() == 1
    assert tensor.shape() == [1, 3, 1, 3]
    assert tensor.rowrank() == 2
    assert tensor.labels() == ["c", "d", "a", "b"]
    np.testing.assert_array_equal(tensor.get_block_().numpy(), array)


@pytest.mark.parametrize("shape, rowrank, diagonal, message", [
    ([3], 0, False, "rank>1"),
    ([2, 2], 0, False, "rowrank>0"),
    ([2, 2], 2, False, "rowrank<rank"),
    ([2, 3], 1, False, "square"),
    ([3], 1, True, "diagonal"),
])
def test_invalid_inputs(shape, rowrank, diagonal, message):
    tensor = cytnx.UniTensor(cytnx.zeros(shape), is_diag=diagonal, rowrank=rowrank)
    before = tensor.get_block_().numpy().copy()
    before_shape, before_labels = tensor.shape(), tensor.labels()
    with pytest.raises(cytnx.CytnxError, match=message):
        cytnx.linalg.Eigh_truncate(tensor, 1)
    np.testing.assert_array_equal(tensor.get_block_().numpy(), before)
    assert tensor.shape() == before_shape
    assert tensor.labels() == before_labels
    assert tensor.rowrank() == rowrank


@pytest.mark.parametrize("dimension", [2**63 - 1, 2**63, 2**64 - 1])
@pytest.mark.parametrize("large_minimum", [False, True])
def test_unsigned_dimensions_with_reserved_floors(dimension, large_minimum):
    tensor, blocks = _hermitian_z3(seed=7)
    kwargs = {"keepdim": dimension}
    if large_minimum:
        kwargs = {"keepdim": 7, "mindim": dimension, "err": 100.0}
    eigenvalues, error = cytnx.linalg.Eigh_truncate(
        tensor, min_blockdim=[1, 1, 1], is_V=False, return_err=2, **kwargs)
    assert eigenvalues.bonds()[0].getDegeneracies() == [3, 2, 2]
    for index, matrix in enumerate(blocks):
        np.testing.assert_allclose(eigenvalues.get_block_(index).numpy(),
                                   np.linalg.eigvalsh(matrix), atol=1e-12, rtol=0)
    assert error.get_block_().shape() == [0]
