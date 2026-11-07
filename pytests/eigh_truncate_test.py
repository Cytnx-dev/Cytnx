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
    assert diff.Norm().item() < 1e-10


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
    with pytest.raises(Exception):
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
