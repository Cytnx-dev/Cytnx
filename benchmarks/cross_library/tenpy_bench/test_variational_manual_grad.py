"""TeNPy benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, with a
hand-derived (manual) gradient instead of automatic differentiation.

TeNPy has no autodiff backend, so the gradient of the Rayleigh quotient

    E(psi) = <psi|H|psi> / <psi|psi>

with respect to every MPS tensor simultaneously is computed analytically
rather than via backprop. This is TeNPy's counterpart to quimb's
`run_one_jax`/`run_one_torch` (`quimb_bench/test_variational_ad.py`), which
differentiate straight through the same whole-network contraction with the
backend's own autodiff. Like quimb's benchmark, every MPS tensor is updated
from a single gradient step taken simultaneously, with no orthogonality
center and no per-step canonicalization -- only a single global rescale
derived from the new `<psi|psi>` is applied after each step, distributed
evenly across all L tensors. Because the surrounding tensors are not kept
isometric, the gradient needs both an H-environment term (the effective
one-site Hamiltonian, as in a one-site sweep) and a norm-environment term:

    dE/dA_i* = (2 / den) * (H_eff,i(A_i) - E * N_eff,i(A_i))

where `den = <psi|psi>`, `H_eff,i` contracts the MPO with the left/right
H-environments around site i, and `N_eff,i` is the analogous contraction
with trivial (no-MPO) norm-environments. `N_eff,i` only collapses to `A_i`
when the rest of the chain is isometric (the one-site-sweep case); here it
does not, so it is computed explicitly via `update_L_N`/`update_R_N`/
`n_eff`. All four environment sets (`LH`, `RH`, `LN`, `RN`) are rebuilt
from scratch every gradient step, since every tensor changes at once and
no sweep-order incremental reuse is possible.

We build every contraction by hand with `tenpy.linalg.np_conserved.tensordot`
calls on plain lists of `Array` tensors -- mirroring the Cytnx benchmark's
`Network`-based `_update_L`/`_update_R`/`_h_eff`/`_update_L_N`/`_update_R_N`/
`_n_eff` (`cytnx_bench/test_variational_manual_grad.py`) rather than going
through `tenpy.networks.mps.MPS`/`MPOEnvironment`, since neither of those
classes assumes the non-canonical, simultaneously-updated state this
algorithm produces.

The initial MPS is also built the same way as the Cytnx benchmark: i.i.d.
normal-random site tensors (same per-site bond-dimension formula and
per-site `seed`), right-canonicalized via a chain of SVDs
(`canonicalize_right`) purely to fix a well-defined starting point -- not
TeNPy's `MPS.from_random_unitary_evolution` from a Neel-like product state,
which stays much closer to a product state after only a few two-site
random gates and converges far more slowly under gradient descent.
`MPOEnvironment.init_LP(0)`/`init_RP(L-1)` (which depend only on the
trivial-boundary structure of `H_MPO`, not on any particular state) still
supply the H-environment boundary vectors; the norm-environment boundaries
are the corresponding bond-dimension-1 scalar identities. CPU only.

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. The initial
MPS is seeded (per-site `np.random.RandomState(seed)`). Unlike a one-site
sweep -- a strong local optimizer whose converged energy is largely
insensitive to small initial-state differences -- this whole-network
update is a weaker optimizer whose converged energy is sensitive to the
initial state, so (as with quimb's AD benchmark) a tight per-library
self-consistency tolerance is still used, but no cross-library energy
comparison is expected to land as close as the one-site-sweep designs do.
"""
import numpy as np
import pytest
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.charges import ChargeInfo, LegCharge
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOEnvironment

from common.model import BOND_DIM_VALUES, HEISENBERG_J, NUM_SITES_VALUES, GRID_POINT_TIMEOUT_SEC

LEARNING_RATE = 0.5


def _n_grad_steps(L):
    return 8 * L


REFERENCE_ENERGIES = {
    (16, 20): -8.67160693635544,
    (16, 30): -13.094234734927348,
    (16, 50): -21.94394809820539,
    (32, 20): -8.674622771089668,
    (32, 30): -13.09947631043835,
    (32, 50): -21.954795152613357,
    (64, 20): -8.67665261172479,
    (64, 30): -13.103427625375156,
    (64, 50): -21.960829029634567,
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


def update_L_N(LP, A_i):
    LP = npc.tensordot(LP, A_i, axes=('vR', 'vL'))
    LP = npc.tensordot(A_i.conj(), LP, axes=(['p0*', 'vL*'], ['p0', 'vR*']))
    return LP


def update_R_N(RP, A_i):
    RP = npc.tensordot(A_i, RP, axes=('vR', 'vL'))
    RP = npc.tensordot(RP, A_i.conj(), axes=(['p0', 'vL*'], ['p0*', 'vR*']))
    return RP


def n_eff(theta, LP, RP):
    t = npc.tensordot(LP, theta, axes=['vR', 'vL'])
    t = npc.tensordot(t, RP, axes=['vR', 'vL'])
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
    LN0 = npc.Array.from_ndarray_trivial(np.array([[1.0]]), labels=['vR*', 'vR'])
    RN0 = npc.Array.from_ndarray_trivial(np.array([[1.0]]), labels=['vL*', 'vL'])

    A = _build_mps(L, chi)

    def grad_step(A):
        LH = [None] * (L + 1)
        LH[0] = L0
        for p in range(L):
            LH[p + 1] = update_L(LH[p], A[p], Ws[p])
        RH = [None] * (L + 1)
        RH[L] = R0
        for p in range(L - 1, -1, -1):
            RH[p] = update_R(RH[p + 1], A[p], Ws[p])
        LN = [None] * (L + 1)
        LN[0] = LN0
        for p in range(L):
            LN[p + 1] = update_L_N(LN[p], A[p])
        RN = [None] * (L + 1)
        RN[L] = RN0
        for p in range(L - 1, -1, -1):
            RN[p] = update_R_N(RN[p + 1], A[p])

        num = LH[L].to_ndarray().reshape(-1)[-1]
        den = LN[L].to_ndarray().item()
        energy = num / den

        new_A = [None] * L
        for p in range(L):
            ht = h_eff(A[p], LH[p], RH[p + 1], Ws[p])
            nt = n_eff(A[p], LN[p], RN[p + 1])
            grad = (2.0 / den) * (ht - energy * nt)
            new_A[p] = A[p] - LEARNING_RATE * grad

        LNn = LN0
        for p in range(L):
            LNn = update_L_N(LNn, new_A[p])
        den_new = LNn.to_ndarray().item()
        scale = den_new ** (-1.0 / (2 * L))
        new_A = [a * scale for a in new_A]
        return new_A, energy.real

    energy = None
    for _ in range(_n_grad_steps(L)):
        A, energy = grad_step(A)
    return energy


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_variational_manual_grad_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("40 MB")
def test_variational_manual_grad_memory():
    energy = run_one(16, 20)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
