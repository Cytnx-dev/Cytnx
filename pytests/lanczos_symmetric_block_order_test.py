"""Regression test for #738: Lanczos with a LinOp acting on symmetric UniTensors.

The operator is the reproducer attached to the issue: three copies of a Z2-symmetric rank-4
tensor contracted into a 16 x 16 transfer matrix. The contraction that applies it to a vector
returns the blocks of the result in another order than the input stores them, and the order
depends on the contraction order. The reference spectrum comes from numpy alone.

Which order a contraction emits its blocks in is an implementation detail, so this test only
checks the spectrum; tests/linalg_test/Arpack_block_order_test.cpp reorders blocks explicitly.
"""

import numpy as np
import pytest

import cytnx


def _dense_a() -> np.ndarray:
    """Returns the Z2-symmetric rank-4 tensor A[i, j, k, l] of the issue as a numpy array."""
    a = np.zeros([2, 2, 2, 2])
    for i, j, k, l in np.ndindex(2, 2, 2, 2):
        if (i + j - k - l) % 2 == 0:
            a[i, j, k, l] = i + j + k + l - 1
    return a


def _reference_spectrum(parity: int) -> np.ndarray:
    """Returns the eigenvalues of the transfer matrix in one parity sector, largest |.| first.

    The matrix is M[(i, j, k, d), (x, y, z, w)] = A[d, i, x, a] A[a, j, y, b] A[b, k, z, w].
    """
    a = _dense_a()
    matrix = np.einsum("dixa,ajyb,bkzw->ijkdxyzw", a, a, a).reshape(16, 16)
    in_sector = np.array([bin(n).count("1") % 2 == parity for n in range(16)])
    eigenvalues = np.linalg.eigvals(matrix[np.ix_(in_sector, in_sector)])
    return eigenvalues[np.argsort(-np.abs(eigenvalues))]


def _symmetric_a() -> cytnx.UniTensor:
    """Returns A as a Z2-symmetric complex UniTensor."""
    sym = [cytnx.Symmetry.Zn(2)]
    bond_in = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1], sym)
    bond_out = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1], sym)
    a = cytnx.UniTensor([bond_in, bond_in, bond_out, bond_out], rowrank=2)
    a.convert_from_(cytnx.UniTensor(cytnx.from_numpy(_dense_a()), rowrank=2))
    return a.astype(cytnx.Type.ComplexDouble)


class _TransferOp(cytnx.LinOp):
    """Applies the transfer matrix to a vector, contracting in a chosen order."""

    def __init__(self, nx: int, a: cytnx.UniTensor, order: str):
        cytnx.LinOp.__init__(self, "mv", nx, cytnx.Type.ComplexDouble)
        self._a = a
        self._order = order

    def matvec(self, v: cytnx.UniTensor) -> cytnx.UniTensor:
        net = cytnx.Network()
        net.FromString([
            "v0  : 101,102,103,104,qnum",
            "T1  :   4, 1, 101,  -1",
            "T2  :  -1, 2, 102,  -2",
            "T3  :  -2, 3, 103, 104",
            "TOUT: 1,2,3,4; qnum",
            "ORDER: " + self._order,
        ])
        net.PutUniTensor("v0", v)
        for name in ("T1", "T2", "T3"):
            net.PutUniTensor(name, self._a)
        return net.Launch()


@pytest.mark.parametrize("order", ["(((v0,T1),T2),T3)", "(((v0,T3),T2),T1)"])
@pytest.mark.parametrize("parity", [0, 1])
def test_lanczos_symmetric_linop_matches_dense_spectrum(order, parity):
    a = _symmetric_a()
    bond = a.bonds()[1]
    bond_total = cytnx.Bond(cytnx.BD_OUT, [[parity]], [1], [cytnx.Symmetry.Zn(2)])
    v0 = cytnx.UniTensor([bond] * 4 + [bond_total], rowrank=4, dtype=cytnx.Type.ComplexDouble)
    v0.uniform_(-1.0, 1.0, 1)

    op = _TransferOp(8, a, order)
    eigenvalues = cytnx.linalg.Lanczos(op, Tin=v0, which="LM", Maxiter=100000, k=6, is_V=False)[0]
    eigenvalues = eigenvalues.get_block().numpy()
    eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]

    reference = _reference_spectrum(parity)
    # The leading eigenvalue is real and non-degenerate: 29.27795999 (parity 0), 17.46092005
    # (parity 1). The next ones are a complex pair or a degenerate triplet, so compare |.|.
    assert eigenvalues[0] == pytest.approx(reference[0], abs=1e-9)
    np.testing.assert_allclose(np.abs(eigenvalues[:3]), np.abs(reference[:3]), atol=1e-8)
