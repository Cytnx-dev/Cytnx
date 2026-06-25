"""TeNPy benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, with a
hand-derived (manual) gradient instead of automatic differentiation.

TeNPy has no autodiff backend, so the gradient of the Rayleigh quotient
    E(psi) = <psi|H|psi> / <psi|psi>
with respect to a single MPS tensor A_i (holding all other tensors fixed)
is computed analytically rather than via backprop. For an MPS tensor that
sits at the orthogonality center of a mixed-canonical gauge, the standard
result is

    dE/dA_i* = 2 * (H_eff,i(A_i) - E * A_i)

where H_eff,i is the effective one-site Hamiltonian obtained by
contracting the MPO with the left/right boundary environments around site
i. We build the environment-update and effective-Hamiltonian contractions
by hand with `tenpy.linalg.np_conserved.tensordot` calls on plain lists of
`Array` tensors -- mirroring the Cytnx benchmark's `Network`-based
`_update_L`/`_update_R`/`_h_eff` (`cytnx_bench/test_variational_manual_grad.py`)
rather than going through `tenpy.networks.mps.MPS`/`MPOEnvironment`. Two of
TeNPy's `MPS`-class conveniences are unsound for this manual sweep: (1)
`MPS.get_theta(i, n=1)` ignores any `formL`/`formR` argument and always
requests TeNPy's fully-symmetric form, silently double-applying a gauge
factor whenever the caller has already folded a leftover QR/SVD factor
into the read-out tensor; (2) every `MPOEnvironment` lazy LP/RP
recomputation calls `ket.get_B(i, form='A'/'B')` unconditionally, which
rescales the stored tensor through `psi.S` -- consistent only if `psi.S`
exactly tracks a strict Vidal-form relationship, an invariant a hand-rolled
per-site gradient step does not maintain. Working with raw `Array` tensors
sidesteps both: `update_L`/`update_R`/`h_eff` below are plain tensor
contractions with no implicit form-dependent rescaling.

The initial MPS is also built the same way as the Cytnx benchmark: i.i.d.
normal-random site tensors (same per-site bond-dimension formula and
per-site `seed`), right-canonicalized via a chain of SVDs
(`canonicalize_right`) -- not TeNPy's `MPS.from_random_unitary_evolution`
from a Neel-like product state, which stays much closer to a product
state after only a few two-site random gates and converges far more
slowly under gradient descent at the shared `LEARNING_RATE`/`N_GRAD_STEPS`
budget. `MPOEnvironment.init_LP(0)`/`init_RP(L-1)` (which depend only on
the trivial-boundary structure of `H_MPO`, not on any particular state)
still supply the boundary vectors.

Each updated site is immediately left-canonicalized (SVD into U, with S*Vh
folded into the next not-yet-visited site), exactly as in the Cytnx
benchmark; `canonicalize_right` restores an all-right-canonical gauge
(except A[0]) at the end of every sweep, mirroring Cytnx's
`_canonicalize_right`. CPU only.

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. The initial
MPS is seeded (per-site `np.random.RandomState(seed)`), so a tight
tolerance is appropriate.
"""
import numpy as np
import pytest
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import ChargeInfo, LegCharge
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOEnvironment

from common.model import CHI_VALUES, HEISENBERG_J, L_VALUES, N_GRAD_STEPS, STEP_TIMEOUT_SEC

LEARNING_RATE = 0.1

REFERENCE_ENERGIES = {
    (16, 20): -8.68246845559462,
    (16, 30): -13.111313297814329,
    (16, 50): -21.971572169443398,
    (32, 20): -8.682473317775269,
    (32, 30): -13.11135548954557,
    (32, 50): -21.972106252821092,
    (64, 20): -8.682473333622692,
    (64, 30): -13.111355749012512,
    (64, 50): -21.972110271827823,
}


def update_L(LP, A_i, W_i):
    LP = npc.tensordot(LP, A_i, axes=('vR', 'vL'))
    LP = npc.tensordot(W_i, LP, axes=(['p0*', 'wL'], ['p0', 'wR']))
    LP = npc.tensordot(A_i.conj(), LP, axes=(['p0*', 'vL*'], ['p0', 'vR*']))
    return LP


def update_R(RP, A_i, W_i):
    RP = npc.tensordot(A_i, RP, axes=('vR', 'vL'))
    RP = npc.tensordot(RP, W_i, axes=(['p0', 'wL'], ['p0*', 'wR']))
    RP = npc.tensordot(RP, A_i.conj(), axes=(['p0', 'vL*'], ['p0*', 'vR*']))
    return RP


def h_eff(theta, LP, RP, W_i):
    t = npc.tensordot(LP, theta, axes=['vR', 'vL'])
    t = npc.tensordot(W_i, t, axes=[['wL', 'p0*'], ['wR', 'p0']])
    t = npc.tensordot(t, RP, axes=[['wR', 'vR'], ['wL', 'vL']])
    t.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
    t.itranspose(['vL', 'p0', 'vR'])
    return t


def _random_trivial(shape, seed, labels, qconjs=(1, 1, -1)):
    rng = np.random.RandomState(seed)
    data = rng.normal(size=shape)
    chinfo = ChargeInfo()
    legs = [LegCharge.from_trivial(s, chinfo, qconj=q) for s, q in zip(shape, qconjs)]
    return npc.Array.from_ndarray(data, legs, labels=labels)


def _build_mps(L, chi, d=2):
    A = [None] * L
    A[0] = _random_trivial([1, d, min(chi, d)], 0, ['vL', 'p0', 'vR'])
    for k in range(1, L):
        dim1 = A[k - 1].get_leg('vR').ind_len
        dim3 = min(min(chi, dim1 * d), d ** (L - k - 1))
        A[k] = _random_trivial([dim1, d, dim3], k, ['vL', 'p0', 'vR'])
    canonicalize_right(A, L)
    return A


def canonicalize_right(A, L):
    for p in range(L - 1, 0, -1):
        mat = A[p].combine_legs(['p0', 'vR'], qconj=-1)
        u, s, vh = npc.svd(mat, inner_labels=['vR', 'vL'])
        vh = vh.split_legs(1)
        s_arr = npc.diag(s, u.get_leg('vR'), labels=['vL', 'vR'])
        A[p] = vh
        A[p - 1] = npc.tensordot(A[p - 1], npc.tensordot(u, s_arr, axes=['vR', 'vL']), axes=['vR', 'vL'])
        A[p - 1].itranspose(['vL', 'p0', 'vR'])
    A[0] /= npc.norm(A[0])


def run_one(chi, L):
    M = SpinChain(dict(
        L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
        bc_MPS="finite", conserve=None,
    ))
    Ws = [M.H_MPO.get_W(i).replace_labels(['p', 'p*'], ['p0', 'p0*']) for i in range(L)]
    sites = M.lat.mps_sites()
    psi0 = MPS.from_product_state(sites, ["up"] * L, bc="finite")
    env0 = MPOEnvironment(psi0, M.H_MPO, psi0)
    L0 = env0.init_LP(0)
    R0 = env0.init_RP(L - 1)

    A = _build_mps(L, chi)

    def grad_step():
        Renv = [None] * (L + 1)
        Renv[L] = R0
        for p in range(L - 1, 0, -1):
            Renv[p] = update_R(Renv[p + 1], A[p], Ws[p])
        Lenv = L0
        energy = None
        for p in range(L):
            theta = A[p]
            ht = h_eff(theta, Lenv, Renv[p + 1], Ws[p])
            norm_sq = npc.inner(theta, theta, axes='labels', do_conj=True)
            energy = npc.inner(theta, ht, axes='labels', do_conj=True) / norm_sq
            grad = 2 * (ht - energy * theta)
            new_theta = theta - LEARNING_RATE * grad
            new_theta /= npc.norm(new_theta)
            if p < L - 1:
                mat = new_theta.combine_legs(['vL', 'p0'], qconj=+1)
                u, s, vh = npc.svd(mat, inner_labels=['vR', 'vL'])
                u = u.split_legs(0)
                s_arr = npc.diag(s, vh.get_leg('vL'), labels=['vL', 'vR'])
                A[p] = u
                A[p + 1] = npc.tensordot(npc.tensordot(s_arr, vh, axes=['vR', 'vL']), A[p + 1], axes=['vR', 'vL'])
                A[p + 1].itranspose(['vL', 'p0', 'vR'])
            else:
                A[p] = new_theta
            Lenv = update_L(Lenv, A[p], Ws[p])
        canonicalize_right(A, L)
        return energy.real

    energy = None
    for _ in range(N_GRAD_STEPS):
        energy = grad_step()
    return energy


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_variational_manual_grad_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(chi, length)], rel=1e-6)


@pytest.mark.limit_memory("40 MB")
def test_variational_manual_grad_memory():
    energy = run_one(16, 20)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
