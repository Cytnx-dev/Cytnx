"""Cytnx benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, with a
hand-derived (manual) gradient instead of automatic differentiation.

Cytnx has no autodiff backend, so the gradient of the Rayleigh quotient

    E(psi) = <psi|H|psi> / <psi|psi>

with respect to every MPS tensor simultaneously is computed analytically
rather than via backprop. This is Cytnx's counterpart to quimb's
`run_one_jax`/`run_one_torch` (`quimb_bench/test_variational_ad.py`), which
differentiate straight through the same whole-network contraction with the
backend's own autodiff. Like quimb's benchmark, every MPS tensor is updated
from a single gradient step taken simultaneously, with no orthogonality
center and no per-step canonicalization -- only a single global rescale
derived from the new `<psi|psi>` is applied after each step, distributed
evenly across all num_sites tensors. Because the surrounding tensors are not kept
isometric, the gradient needs both an H-environment term (the effective
one-site Hamiltonian, as in a one-site sweep) and a norm-environment term:

    dE/dA_i* = (2 / den) * (H_eff,i(A_i) - E * N_eff,i(A_i))

where `den = <psi|psi>`, `H_eff,i` contracts the MPO with the left/right
H-environments around site i, and `N_eff,i` is the analogous contraction
with trivial (no-MPO) norm-environments. `N_eff,i` only collapses to `A_i`
when the rest of the chain is isometric (the one-site-sweep case); here it
does not, so it is computed explicitly via the `_Networks.LN_UPDATE_NET`/
`RN_UPDATE_NET`/`N_EFF_NET` Cytnx `Network` topologies below, which are
the same `L_UPDATE_NET`/`R_UPDATE_NET`/`HEFF_NET` contractions with the
MPO leg dropped. All four environment sets (H-left, H-right, norm-left,
norm-right) are rebuilt from scratch every gradient step, since every
tensor changes at once and no sweep-order incremental reuse is possible.

CPU and GPU code paths are both written; the GPU path moves every MPS/MPO
UniTensor to `cytnx.Device.cuda` before the gradient step. It cannot be
exercised in this environment (no GPU).

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. The initial
MPS is seeded (per-site `cytnx.UniTensor.normal(..., seed=k)`). Unlike a
one-site sweep -- a strong local optimizer whose converged energy is
largely insensitive to small initial-state differences -- this
whole-network update is a weaker optimizer whose converged energy is
sensitive to the initial state, so (as with quimb's AD benchmark) a tight
per-library self-consistency tolerance is still used, but no cross-library
energy comparison is expected to land as close as the one-site-sweep
designs do.
"""
import pytest

import cytnx

from common.model import BOND_DIM_VALUES, HEISENBERG_J, NUM_SITES_VALUES, GRID_POINT_TIMEOUT_SEC

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below
LEARNING_RATE = 0.5


def _n_grad_steps(num_sites):
    return 8 * num_sites


REFERENCE_ENERGIES = {
    (16, 20): -8.669390762907128,
    (16, 30): -13.093516691905897,
    (16, 50): -21.945283123631583,
    (32, 20): -8.674290703078219,
    (32, 30): -13.100972670479699,
    (32, 50): -21.953298655520914,
    (64, 20): -8.67676882878297,
    (64, 30): -13.103318241859627,
    (64, 50): -21.96029731572234,
}

def _build_mpo(J, device):
    d = 2
    D = 5
    Sp = cytnx.zeros([d, d], device=device)
    Sp[0, 1] = 1.0  # S+: |down> -> |up>
    Sm = cytnx.zeros([d, d], device=device)
    Sm[1, 0] = 1.0  # S-: |up> -> |down>
    Sz = cytnx.zeros([d, d], device=device)
    Sz[0, 0] = 0.5
    Sz[1, 1] = -0.5
    eye = cytnx.eye(d, device=device)

    M = cytnx.zeros([D, D, d, d], device=device)
    M[0, 0] = eye
    M[D - 1, D - 1] = eye
    M[0, 1] = Sp
    M[0, 2] = Sm
    M[0, 3] = Sz
    M[1, D - 1] = (J / 2.0) * Sm
    M[2, D - 1] = (J / 2.0) * Sp
    M[3, D - 1] = J * Sz
    M = cytnx.UniTensor(M, 0).set_name("MPO")

    L0 = cytnx.UniTensor.zeros([D, 1, 1], device=device).set_rowrank_(0).set_name("L0")
    R0 = cytnx.UniTensor.zeros([D, 1, 1], device=device).set_rowrank_(0).set_name("R0")
    L0[0, 0, 0] = 1.0
    R0[D - 1, 0, 0] = 1.0
    return M, L0, R0


def _build_mps(num_sites, bond_dim, device):
    d = 2
    A = [None] * num_sites
    lbls = [[str(2 * k), str(2 * k + 1), str(2 * k + 2)] for k in range(num_sites)]
    A[0] = cytnx.UniTensor.normal([1, d, min(bond_dim, d)], 0., 1., seed=0, device=device).set_rowrank_(2)
    A[0].relabel_(lbls[0]).set_name("A0")
    for k in range(1, num_sites):
        dim1 = A[k - 1].shape()[2]
        dim3 = min(min(bond_dim, dim1 * d), d ** (num_sites - k - 1))
        A[k] = cytnx.UniTensor.normal([dim1, d, dim3], 0., 1., seed=k, device=device).set_rowrank_(2)
        A[k].relabel_(lbls[k]).set_name(f"A{k}")
    _canonicalize_right(A, lbls, num_sites)
    return A, lbls


def _canonicalize_right(A, lbls, num_sites):
    for p in range(num_sites - 1, 0, -1):
        A[p].set_rowrank_(1)
        s, u, vt = cytnx.linalg.Gesvd(A[p])
        A[p] = vt
        A[p - 1] = cytnx.Contract(A[p - 1], cytnx.Contract(u, s))
        A[p].relabel_(lbls[p]).set_name(f"A{p}")
        A[p - 1].relabel_(lbls[p - 1]).set_name(f"A{p-1}")
    A[0] = A[0] / A[0].Norm().item()
    A[0].relabel_(lbls[0]).set_name("A0")


class _Networks:
    """Caches one parsed `cytnx.Network` per static contraction topology used
    in `run_one`'s gradient step, since `FromString` re-parses the topology
    string on every call -- with `8 * num_sites` whole-network gradient steps
    and O(num_sites) such contractions per step, re-parsing on every call
    dominates the runtime at large num_sites. Each `Network` is built once and
    refilled via
    `PutUniTensors`/`Launch` on every reuse."""

    L_UPDATE_NET = ["L: -2,-1,-3", "A: -1,-4,1", "M: -2,0,-4,-5", "A_Conj: -3,-5,2", "TOUT: 0,1,2"]
    R_UPDATE_NET = ["R: -2,-1,-3", "B: 1,-4,-1", "M: 0,-2,-4,-5", "B_Conj: 2,-5,-3", "TOUT: 0,1,2"]
    HEFF_NET = ["psi: -1,-2,-3", "L: -4,-1,0", "R: -5,-3,2", "M: -4,-5,-2,1", "TOUT: 0,1;2"]

    LN_UPDATE_NET = ["L: -1,-2", "A: -1,-3,0", "A_Conj: -2,-3,1", "TOUT: 0,1"]
    RN_UPDATE_NET = ["R: -1,-2", "B: 0,-3,-1", "B_Conj: 1,-3,-2", "TOUT: 0,1"]
    N_EFF_NET = ["L: -1,0", "psi: -1,1,-2", "R: -2,2", "TOUT: 0,1,2"]

    def __init__(self):
        self.l_update = cytnx.Network()
        self.l_update.FromString(self.L_UPDATE_NET)
        self.r_update = cytnx.Network()
        self.r_update.FromString(self.R_UPDATE_NET)
        self.heff = cytnx.Network()
        self.heff.FromString(self.HEFF_NET)
        self.ln_update = cytnx.Network()
        self.ln_update.FromString(self.LN_UPDATE_NET)
        self.rn_update = cytnx.Network()
        self.rn_update.FromString(self.RN_UPDATE_NET)
        self.n_eff_net = cytnx.Network()
        self.n_eff_net.FromString(self.N_EFF_NET)

    def update_L(self, L_env, A_new, A_conj, M):
        self.l_update.PutUniTensors(["L", "A", "A_Conj", "M"], [L_env, A_new, A_conj, M])
        return self.l_update.Launch()

    def update_R(self, R_env, B_new, B_conj, M):
        self.r_update.PutUniTensors(["R", "B", "M", "B_Conj"], [R_env, B_new, M, B_conj])
        return self.r_update.Launch()

    def h_eff(self, theta, L_env, R_env, M):
        self.heff.PutUniTensors(["psi", "L", "R", "M"], [theta, L_env, R_env, M])
        out = self.heff.Launch()
        out.relabel_(theta.labels())
        return out

    def update_L_N(self, LN_env, A_new, A_conj):
        self.ln_update.PutUniTensors(["L", "A", "A_Conj"], [LN_env, A_new, A_conj])
        return self.ln_update.Launch()

    def update_R_N(self, RN_env, B_new, B_conj):
        self.rn_update.PutUniTensors(["R", "B", "B_Conj"], [RN_env, B_new, B_conj])
        return self.rn_update.Launch()

    def n_eff(self, theta, LN_env, RN_env):
        self.n_eff_net.PutUniTensors(["L", "psi", "R"], [LN_env, theta, RN_env])
        out = self.n_eff_net.Launch()
        out.relabel_(theta.labels())
        return out


def run_one(bond_dim, num_sites):
    device = cytnx.Device.cuda if DEVICE == "gpu" else cytnx.Device.cpu
    M, L0, R0 = _build_mpo(HEISENBERG_J, device)
    A, lbls = _build_mps(num_sites, bond_dim, device)
    LN0 = cytnx.UniTensor.zeros([1, 1], device=device).set_rowrank_(0)
    LN0[0, 0] = 1.0
    RN0 = cytnx.UniTensor.zeros([1, 1], device=device).set_rowrank_(0)
    RN0[0, 0] = 1.0
    nets = _Networks()

    def grad_step(A):
        A_conj = [a.Dagger().permute_(a.labels()) for a in A]

        LH = [None] * (num_sites + 1)
        LH[0] = L0
        for p in range(num_sites):
            LH[p + 1] = nets.update_L(LH[p], A[p], A_conj[p], M)
        RH = [None] * (num_sites + 1)
        RH[num_sites] = R0
        for p in range(num_sites - 1, -1, -1):
            RH[p] = nets.update_R(RH[p + 1], A[p], A_conj[p], M)
        LN = [None] * (num_sites + 1)
        LN[0] = LN0
        for p in range(num_sites):
            LN[p + 1] = nets.update_L_N(LN[p], A[p], A_conj[p])
        RN = [None] * (num_sites + 1)
        RN[num_sites] = RN0
        for p in range(num_sites - 1, -1, -1):
            RN[p] = nets.update_R_N(RN[p + 1], A[p], A_conj[p])

        D = LH[num_sites].shape()[0]
        num = LH[num_sites][D - 1, 0, 0].item()
        den = LN[num_sites][0, 0].item()
        energy = num / den

        new_A = [None] * num_sites
        for p in range(num_sites):
            ht = nets.h_eff(A[p], LH[p], RH[p + 1], M)
            nt = nets.n_eff(A[p], LN[p], RN[p + 1])
            grad = (2.0 / den) * (ht - energy * nt)
            new_A[p] = A[p] - LEARNING_RATE * grad
            new_A[p].relabel_(lbls[p]).set_name(f"A{p}")

        LNn = LN0
        for p in range(num_sites):
            LNn = nets.update_L_N(LNn, new_A[p], new_A[p].Dagger().permute_(new_A[p].labels()))
        den_new = LNn[0, 0].item()
        scale = den_new ** (-1.0 / (2 * num_sites))
        new_A = [a * scale for a in new_A]
        for p in range(num_sites):
            new_A[p].relabel_(lbls[p]).set_name(f"A{p}")
        return new_A, energy

    energy = None
    for _ in range(_n_grad_steps(num_sites)):
        A, energy = grad_step(A)
    return energy


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_variational_manual_grad_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("40 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_variational_manual_grad_memory(bond_dim, num_sites):
    energy = run_one(bond_dim, num_sites)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)
