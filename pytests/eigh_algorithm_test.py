"""Tests for the ``algorithm`` option of ``cytnx.linalg.Eigh`` (Cytnx-dev/Cytnx#1214).

The expected eigenvalues are closed-form spectra. The eigenvectors are checked
through the residual, the orthonormality and the reconstruction, evaluated with
numpy arithmetic; no test pins an eigenvector sign or a basis of a degenerate
eigenspace.
"""

import numpy as np
import pytest

import cytnx

ALGORITHMS = ["default", "qr", "dc"]
DTYPES = [np.float64, np.float32, np.complex128, np.complex64]


def _tolerance(dtype):
    single = np.dtype(dtype) in (np.float32, np.complex64)
    return 5e-5 if single else 1e-11


def _tridiagonal(n, diagonal, hopping):
    """Returns the matrix and its spectrum, diagonal + 2|hopping|cos(k pi/(n+1))."""
    matrix = np.diag(np.full(n, diagonal, dtype=np.complex128))
    matrix += np.diag(np.full(n - 1, hopping), 1)
    matrix += np.diag(np.full(n - 1, np.conj(hopping)), -1)
    k = np.arange(1, n + 1)
    return matrix, diagonal + 2 * abs(hopping) * np.cos(k * np.pi / (n + 1))


def _rank_one_update(n, shift, weight, phase):
    """Returns shift * 1 + weight * u u^dagger and its (degenerate) spectrum."""
    j = np.arange(n)
    u = (j + 1) * np.exp(1j * phase * j)
    matrix = shift * np.eye(n) + weight * np.outer(u, u.conj())
    spectrum = np.full(n, shift)
    spectrum[-1] = shift + weight * n * (n + 1) * (2 * n + 1) / 6
    return matrix, spectrum


def _cases(dtype):
    is_complex = np.issubdtype(dtype, np.complexfloating)
    hopping = 1.3 * np.exp(0.7j) if is_complex else -1.3
    cases = [
        _tridiagonal(7, 0.5, hopping),
        _tridiagonal(40, -0.25, hopping),
        _rank_one_update(6, -0.75, 0.02, 0.9 if is_complex else 0.0),
        (np.array([[-2.5]]), np.array([-2.5])),
    ]
    return [(m if is_complex else m.real).astype(dtype) for m, _ in cases], [
        np.sort(s) for _, s in cases
    ]


def _check(matrix, e, v, spectrum, tol):
    np.testing.assert_allclose(e, spectrum, rtol=0, atol=tol)
    np.testing.assert_allclose(matrix @ v, v * e, rtol=0, atol=tol)
    np.testing.assert_allclose(v.conj().T @ v, np.eye(len(e)), rtol=0, atol=tol)
    np.testing.assert_allclose((v * e) @ v.conj().T, matrix, rtol=0, atol=tol)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tensor_matches_the_closed_form_spectrum(algorithm, dtype):
    matrices, spectra = _cases(dtype)
    for matrix, spectrum in zip(matrices, spectra):
        e, v = cytnx.linalg.Eigh(cytnx.from_numpy(matrix), algorithm=algorithm)
        assert e.numpy().dtype == np.zeros(1, dtype).real.dtype
        assert v.numpy().dtype == np.dtype(dtype)
        _check(matrix.astype(np.complex128), e.numpy().astype(np.float64),
               v.numpy().astype(np.complex128), spectrum, _tolerance(dtype))
        (e_only,) = cytnx.linalg.Eigh(cytnx.from_numpy(matrix), is_V=False,
                                      algorithm=algorithm)
        np.testing.assert_allclose(e_only.numpy(), spectrum, rtol=0,
                                   atol=_tolerance(dtype))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_tensor_member_function_takes_the_option(algorithm):
    (matrix, *_), (spectrum, *_) = _cases(np.float64)
    e, v = cytnx.from_numpy(matrix).Eigh(algorithm=algorithm)
    _check(matrix, e.numpy(), v.numpy(), spectrum, 1e-11)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_block_unitensor_sector_by_sector(algorithm, dtype):
    is_complex = np.issubdtype(dtype, np.complexfloating)
    sectors = [
        _rank_one_update(4, -0.5, 0.1, 0.6 if is_complex else 0.0),
        _tridiagonal(3, 0.25, 0.9 * np.exp(-0.4j) if is_complex else 0.9),
        (np.array([[-1.75]]), np.array([-1.75])),
    ]
    qs = [cytnx.Qs(q) >> len(s) for q, (_, s) in enumerate(sectors)]
    row = cytnx.Bond(cytnx.BD_IN, qs, [cytnx.Symmetry.Zn(3)])
    tensor_dtype = (cytnx.Type.ComplexDouble
                    if is_complex else cytnx.Type.Double)
    ut = cytnx.UniTensor([row, row.redirect()], labels=["r", "c"], rowrank=1,
                         dtype=tensor_dtype)
    blocks = []
    for b, (matrix, _) in enumerate(sectors):
        blocks.append((matrix if is_complex else matrix.real).astype(dtype))
        ut.put_block_(cytnx.from_numpy(blocks[-1]), b)

    e, v = cytnx.linalg.Eigh(ut, algorithm=algorithm)
    assert e.Nblocks() == v.Nblocks() == len(sectors)
    for b, (_, spectrum) in enumerate(sectors):
        _check(blocks[b], e.get_block_(b).numpy(), v.get_block_(b).numpy(),
               np.sort(spectrum), 1e-11)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_eigh_truncate_takes_the_option(algorithm):
    row = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 3, cytnx.Qs(1) >> 2],
                     [cytnx.Symmetry.Zn(2)])
    ut = cytnx.UniTensor([row, row.redirect()], labels=["r", "c"], rowrank=1)
    ut.put_block_(cytnx.from_numpy(np.diag([5.0, -3.0, 0.5])), 0)
    ut.put_block_(cytnx.from_numpy(np.diag([4.0, 1.0])), 1)
    e, _ = cytnx.linalg.Eigh_truncate(ut, 3, algorithm=algorithm)
    np.testing.assert_allclose(e.get_block_(0).numpy(), [-3.0, 5.0], atol=1e-12)
    np.testing.assert_allclose(e.get_block_(1).numpy(), [4.0], atol=1e-12)


@pytest.mark.parametrize("algorithm", ["no_such_algorithm", "", "DC", "mrrr"])
def test_unknown_algorithm_raises(algorithm):
    tensor = cytnx.from_numpy(np.eye(3))
    with pytest.raises(cytnx.CytnxError, match="unknown algorithm"):
        cytnx.linalg.Eigh(tensor, algorithm=algorithm)
    with pytest.raises(cytnx.CytnxError, match="unknown algorithm"):
        tensor.Eigh(algorithm=algorithm)
    with pytest.raises(cytnx.CytnxError, match="unknown algorithm"):
        cytnx.linalg.Eigh(cytnx.UniTensor(tensor), algorithm=algorithm)
    with pytest.raises(cytnx.CytnxError, match="unknown algorithm"):
        cytnx.linalg.Eigh_truncate(cytnx.UniTensor(tensor), 2,
                                   algorithm=algorithm)
